import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================
# 键盘常量
# ============================================================
KEY_EXIT = 27  # ESC键
KEY_R = 114  # 重置
KEY_F = 102  # 聚焦

# 平移控制
KEY_X_POS = 100  # D
KEY_X_NEG = 97  # A
KEY_Y_POS = 119  # W
KEY_Y_NEG = 115  # S
KEY_Z_POS = 101  # E
KEY_Z_NEG = 113  # Q

# 旋转控制
KEY_ROLL_POS = 106  # J
KEY_ROLL_NEG = 108  # L
KEY_PITCH_POS = 105  # I
KEY_PITCH_NEG = 107  # K
KEY_YAW_POS = 117  # U
KEY_YAW_NEG = 111  # O

print("=" * 70)
print("飞机机身 + 机翼 六自由度控制 (形心基准)")
print("=" * 70)

# ============================================================
# 配置
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

WING_STL = os.path.join(script_dir, "model-left-wing.stl")
FUSELAGE_STL = os.path.join(script_dir, "model-body.stl")

if not os.path.exists(WING_STL):
    print(f"找不到机翼文件: {WING_STL}")
    exit(1)
if not os.path.exists(FUSELAGE_STL):
    print(f"找不到机身文件: {FUSELAGE_STL}")
    exit(1)

print(f"找到机翼: {WING_STL}")
print(f"找到机身: {FUSELAGE_STL}")


# ============================================================
# 创建URDF文件
# ============================================================

def create_urdf(stl_path, name, scale=0.01, mass=1000.0, color=[0.7, 0.7, 0.8]):
    abs_path = os.path.abspath(stl_path).replace('\\', '/')
    urdf_content = f'''<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="{abs_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
      <material name="material">
        <color rgba="{color[0]} {color[1]} {color[2]} 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{abs_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1000.0" ixy="0" ixz="0" iyy="1000.0" iyz="0" izz="1000.0"/>
    </inertial>
  </link>
</robot>'''
    return urdf_content


# 创建临时URDF文件
wing_urdf = os.path.join(script_dir, "wing_temp.urdf")
fuselage_urdf = os.path.join(script_dir, "fuselage_temp.urdf")

with open(wing_urdf, 'w', encoding='utf-8') as f:
    f.write(create_urdf(WING_STL, "wing", scale=0.01, mass=5000.0, color=[0.2, 0.4, 0.8]))

with open(fuselage_urdf, 'w', encoding='utf-8') as f:
    f.write(create_urdf(FUSELAGE_STL, "fuselage", scale=0.01, mass=2000.0, color=[0.7, 0.7, 0.8]))

print("URDF文件创建完成")

# ============================================================
# 启动PyBullet
# ============================================================

print("启动PyBullet...")
physicsClient = p.connect(p.GUI)
if physicsClient < 0:
    print("物理服务器连接失败，程序退出")
    exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # 无重力

# ============================================================
# 加载地面
# ============================================================

print("加载地面...")

ground_visual = p.createVisualShape(
    p.GEOM_PLANE,
    halfExtents=[50, 50],
    rgbaColor=[0.6, 0.65, 0.7, 1.0]
)
ground_collision = p.createCollisionShape(p.GEOM_PLANE, halfExtents=[50, 50])
ground_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=ground_visual,
    baseCollisionShapeIndex=ground_collision,
    basePosition=[0, 0, 0]
)

# 网格线
for i in range(-50, 51, 5):
    p.addUserDebugLine([i, -50, 0.01], [i, 50, 0.01], [0.5, 0.5, 0.5, 1], lineWidth=1)
    p.addUserDebugLine([-50, i, 0.01], [50, i, 0.01], [0.5, 0.5, 0.5, 1], lineWidth=1)

print("地面加载完成 (50x50米)")

# ============================================================
# 加载机身（固定）
# ============================================================

print("加载机身...")
try:
    fuselage_id = p.loadURDF(
        fuselage_urdf,
        basePosition=[0, 0, 0.5],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
    )
    print("机身加载成功 (固定)")
except Exception as e:
    print(f"机身加载失败: {e}")
    if p.isConnected():
        p.disconnect()
    exit(1)

# ============================================================
# 获取机身信息
# ============================================================

fuselage_aabb = p.getAABB(fuselage_id)
fuselage_center = [(fuselage_aabb[0][i] + fuselage_aabb[1][i]) / 2 for i in range(3)]
fuselage_size = [fuselage_aabb[1][i] - fuselage_aabb[0][i] for i in range(3)]
print(
    f"机身: 尺寸 {fuselage_size[0]:.2f}x{fuselage_size[1]:.2f}x{fuselage_size[2]:.2f}m, 中心 ({fuselage_center[0]:.2f}, {fuselage_center[1]:.2f}, {fuselage_center[2]:.2f})")

# ============================================================
# 加载机翼（可移动，六自由度）
# ============================================================

print("加载机翼...")
try:
    initial_position = [3.0, 0, 1.0]
    initial_orientation = [0, 0, 0, 1]

    wing_id = p.loadURDF(
        wing_urdf,
        basePosition=initial_position,
        baseOrientation=initial_orientation,
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION
    )
    print("机翼加载成功 (六自由度可动)")

    p.changeDynamics(wing_id, -1,
                     lateralFriction=0.8,
                     restitution=0.05,
                     rollingFriction=50.0,
                     spinningFriction=50.0)

except Exception as e:
    print(f"机翼加载失败: {e}")
    if p.isConnected():
        p.disconnect()
    exit(1)

# ============================================================
# 计算机翼固定参数
# ============================================================

# 计算机翼相对于基点的局部形心偏移（模型坐标系下固定不变）
wing_init_pos, wing_init_quat = p.getBasePositionAndOrientation(wing_id)
wing_init_aabb = p.getAABB(wing_id)
wing_init_centroid = [(wing_init_aabb[0][i] + wing_init_aabb[1][i]) / 2 for i in range(3)]
centroid_local = np.array(wing_init_centroid) - np.array(wing_init_pos)

wing_size = [wing_init_aabb[1][i] - wing_init_aabb[0][i] for i in range(3)]
print(
    f"机翼: 尺寸 {wing_size[0]:.2f}x{wing_size[1]:.2f}x{wing_size[2]:.2f}m, 形心 ({wing_init_centroid[0]:.2f}, {wing_init_centroid[1]:.2f}, {wing_init_centroid[2]:.2f})")

# ============================================================
# 辅助标记
# ============================================================

# 世界坐标轴
p.addUserDebugLine([0, 0, 0], [5, 0, 0], [1, 0, 0], lineWidth=3)
p.addUserDebugLine([0, 0, 0], [0, 5, 0], [0, 1, 0], lineWidth=3)
p.addUserDebugLine([0, 0, 0], [0, 0, 5], [0, 0, 1], lineWidth=3)

p.addUserDebugText("X", [5.5, 0, 0], textColorRGB=[1, 0, 0], textSize=1.5)
p.addUserDebugText("Y", [0, 5.5, 0], textColorRGB=[0, 1, 0], textSize=1.5)
p.addUserDebugText("Z", [0, 0, 5.5], textColorRGB=[0, 0, 1], textSize=1.5)

# 机身标记
p.addUserDebugText("机身 (固定)", [fuselage_center[0], fuselage_center[1], fuselage_center[2] + 2.5],
                   textColorRGB=[0.7, 0.7, 0.8], textSize=1.5)

# 机翼标记
p.addUserDebugText("机翼 (六自由度)", [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 2.0],
                   textColorRGB=[0.2, 0.4, 0.8], textSize=1.5)

# 机翼初始形心标记
p.addUserDebugLine([wing_init_centroid[0] - 0.3, wing_init_centroid[1], wing_init_centroid[2]],
                   [wing_init_centroid[0] + 0.3, wing_init_centroid[1], wing_init_centroid[2]], [1, 1, 0], lineWidth=3)
p.addUserDebugLine([wing_init_centroid[0], wing_init_centroid[1] - 0.3, wing_init_centroid[2]],
                   [wing_init_centroid[0], wing_init_centroid[1] + 0.3, wing_init_centroid[2]], [1, 1, 0], lineWidth=3)
p.addUserDebugLine([wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] - 0.3],
                   [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 0.3], [1, 1, 0], lineWidth=3)

p.addUserDebugText("形心", [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 0.6],
                   textColorRGB=[1, 1, 0], textSize=1.2)

# ============================================================
# 设置视角
# ============================================================

p.resetDebugVisualizerCamera(
    cameraDistance=20,
    cameraYaw=-30,
    cameraPitch=-35,
    cameraTargetPosition=[1.5, 0, 1.5]
)

# ============================================================
# 打印控制说明
# ============================================================

print("\n" + "=" * 70)
print("六自由度控制说明 (以机翼形心为基准)")
print("=" * 70)
print("")
print("  平移控制 (速度 2m/s):")
print("     A/D  -> X轴负/正方向移动")
print("     S/W  -> Y轴负/正方向移动")
print("     Q/E  -> Z轴负/正方向移动 (下降/上升)")
print("")
print("  旋转控制 (速度 30°/s，绕机体局部轴):")
print("     J/L  -> 绕X轴旋转 (滚转)")
print("     K/I  -> 绕Y轴旋转 (俯仰)")
print("     U/O  -> 绕Z轴旋转 (偏航)")
print("")
print("  其他功能:")
print("     R    -> 重置机翼位置和姿态")
print("     F    -> 聚焦到机翼")
print("     ESC  -> 退出程序")
print("=" * 70 + "\n")

# ============================================================
# 六自由度控制参数与函数
# ============================================================

current_pos = list(initial_position)
current_quat = list(initial_orientation)

TRANSLATION_SPEED = 2.0   # 平移速度：米/秒
ROTATION_SPEED = 30.0     # 旋转速度：度/秒


def get_wing_centroid():
    """获取机翼当前的形心世界坐标"""
    pos, quat = p.getBasePositionAndOrientation(wing_id)
    r = R.from_quat(quat)
    centroid_world = np.array(pos) + r.apply(centroid_local)
    return centroid_world.tolist()


def reset_wing():
    """重置机翼位置和姿态"""
    global current_pos, current_quat
    current_pos = [3.0, 0, 1.0]
    current_quat = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)
    p.resetBaseVelocity(wing_id, [0, 0, 0], [0, 0, 0])
    print("机翼已重置")


def move_wing_translation(dx, dy, dz):
    """平移机翼（世界坐标系下平移）"""
    global current_pos
    current_pos[0] += dx
    current_pos[1] += dy
    current_pos[2] += dz
    # 位置边界限制
    current_pos = np.clip(current_pos, [-20, -20, 0.1], [20, 20, 20]).tolist()
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)


def rotate_wing(axis, angle_deg):
    """绕机翼局部坐标轴旋转，旋转中心为机翼形心"""
    global current_pos, current_quat

    # 获取当前位姿
    pos, quat = p.getBasePositionAndOrientation(wing_id)
    r = R.from_quat(quat)

    # 计算当前形心世界坐标
    centroid_world = np.array(pos) + r.apply(centroid_local)

    # 生成局部旋转增量，四元数右乘实现绕机体轴旋转
    delta_r = R.from_euler(axis, angle_deg, degrees=True)
    new_r = r * delta_r
    new_quat = new_r.as_quat()

    # 计算新的基点位置：形心保持不动，基点随旋转偏移
    base_offset_local = -centroid_local
    base_offset_rotated = new_r.apply(base_offset_local)
    new_pos = centroid_world + base_offset_rotated

    # 更新全局状态并应用到物理引擎
    current_pos = new_pos.tolist()
    current_quat = new_quat.tolist()
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)
    p.resetBaseVelocity(wing_id, [0, 0, 0], [0, 0, 0])


def focus_on_wing():
    """将相机聚焦到机翼形心"""
    centroid = get_wing_centroid()
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=centroid
    )
    print("已聚焦到机翼")


def print_status():
    """实时打印机翼状态信息"""
    pos, quat = p.getBasePositionAndOrientation(wing_id)
    centroid = get_wing_centroid()
    r = R.from_quat(quat)
    euler = r.as_euler('xyz', degrees=True)
    print(f"\r机翼位置: X={pos[0]:6.2f} Y={pos[1]:6.2f} Z={pos[2]:6.2f} | "
          f"形心: X={centroid[0]:6.2f} Y={centroid[1]:6.2f} Z={centroid[2]:6.2f} | "
          f"姿态: Roll={euler[0]:6.1f}° Pitch={euler[1]:6.1f}° Yaw={euler[2]:6.1f}°", end="")


# ============================================================
# 主循环
# ============================================================

frame_count = 0
last_time = time.time()

try:
    while True:
        # 计算帧间隔，保证运动速度与帧率无关
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        p.stepSimulation()
        time.sleep(1. / 240.)

        frame_count += 1
        if frame_count % 60 == 0:
            print_status()

        keys = p.getKeyboardEvents()

        # 退出程序
        if KEY_EXIT in keys and keys[KEY_EXIT] & p.KEY_WAS_TRIGGERED:
            print("\n退出程序")
            break

        # 重置机翼
        if KEY_R in keys and keys[KEY_R] & p.KEY_WAS_TRIGGERED:
            reset_wing()

        # 聚焦机翼
        if KEY_F in keys and keys[KEY_F] & p.KEY_WAS_TRIGGERED:
            focus_on_wing()

        # 平移控制
        step_t = TRANSLATION_SPEED * dt
        if KEY_X_NEG in keys and keys[KEY_X_NEG] & p.KEY_IS_DOWN:
            move_wing_translation(-step_t, 0, 0)
        if KEY_X_POS in keys and keys[KEY_X_POS] & p.KEY_IS_DOWN:
            move_wing_translation(step_t, 0, 0)
        if KEY_Y_NEG in keys and keys[KEY_Y_NEG] & p.KEY_IS_DOWN:
            move_wing_translation(0, -step_t, 0)
        if KEY_Y_POS in keys and keys[KEY_Y_POS] & p.KEY_IS_DOWN:
            move_wing_translation(0, step_t, 0)
        if KEY_Z_NEG in keys and keys[KEY_Z_NEG] & p.KEY_IS_DOWN:
            move_wing_translation(0, 0, -step_t)
        if KEY_Z_POS in keys and keys[KEY_Z_POS] & p.KEY_IS_DOWN:
            move_wing_translation(0, 0, step_t)

        # 旋转控制
        step_r = ROTATION_SPEED * dt
        if KEY_ROLL_NEG in keys and keys[KEY_ROLL_NEG] & p.KEY_IS_DOWN:
            rotate_wing('x', -step_r)
        if KEY_ROLL_POS in keys and keys[KEY_ROLL_POS] & p.KEY_IS_DOWN:
            rotate_wing('x', step_r)
        if KEY_PITCH_NEG in keys and keys[KEY_PITCH_NEG] & p.KEY_IS_DOWN:
            rotate_wing('y', -step_r)
        if KEY_PITCH_POS in keys and keys[KEY_PITCH_POS] & p.KEY_IS_DOWN:
            rotate_wing('y', step_r)
        if KEY_YAW_POS in keys and keys[KEY_YAW_POS] & p.KEY_IS_DOWN:
            rotate_wing('z', step_r)
        if KEY_YAW_NEG in keys and keys[KEY_YAW_NEG] & p.KEY_IS_DOWN:
            rotate_wing('z', -step_r)

except KeyboardInterrupt:
    print("\n用户中断")

finally:
    # 安全断开连接，避免未连接状态下报错
    if p.isConnected():
        p.disconnect()
    # 清理临时URDF文件
    try:
        if os.path.exists(wing_urdf):
            os.remove(wing_urdf)
        if os.path.exists(fuselage_urdf):
            os.remove(fuselage_urdf)
    except Exception:
        pass
    print("仿真已关闭")