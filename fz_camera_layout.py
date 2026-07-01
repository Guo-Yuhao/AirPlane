import pybullet as p
import pybullet_data
import time
import os
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ============================================================
# 键盘常量
# ============================================================
KEY_EXIT = 27  # ESC键
KEY_R = 114    # 重置机翼
KEY_F = 102    # 聚焦到机翼
KEY_T = 116    # 将相机目标点更新到当前机翼形心
KEY_C = 99     # 保存三台相机图像
KEY_G = 103    # 聚焦到测量目标点
KEY_1 = 49     # GUI视角切到相机1
KEY_2 = 50     # GUI视角切到相机2
KEY_3 = 51     # GUI视角切到相机3
KEY_0 = 48     # 返回总览视角，并解除相机视角锁定
KEY_V = 118    # 在相机1/2/3/总览之间循环切换
KEY_P = 112    # 启动/停止定时自动拍摄
KEY_H = 104    # 按当前机翼/机身位置重新计算推荐相机布局

# 平移控制
KEY_X_POS = 100  # D
KEY_X_NEG = 97   # A
KEY_Y_POS = 119  # W
KEY_Y_NEG = 115  # S
KEY_Z_POS = 101  # E
KEY_Z_NEG = 113  # Q

# 旋转控制
KEY_ROLL_POS = 106   # J
KEY_ROLL_NEG = 108   # L
KEY_PITCH_POS = 105  # I
KEY_PITCH_NEG = 107  # K
KEY_YAW_POS = 117    # U
KEY_YAW_NEG = 111    # O

print("=" * 78)
print("✈️ 飞机机身 + 机翼六自由度控制 + 自动三相机布置")
print("=" * 78)

# ============================================================
# 配置
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "camera_output")
os.makedirs(output_dir, exist_ok=True)


def find_existing_file(folder, candidates):
    """在 folder 中寻找候选文件；兼容 .stl / .STL 大小写。"""
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    lower_map = {name.lower(): name for name in os.listdir(folder)}
    for name in candidates:
        matched = lower_map.get(name.lower())
        if matched is not None:
            return os.path.join(folder, matched)

    return None


WING_STL = find_existing_file(script_dir, ["model-left-wing.stl", "model-left-wing.STL"])
FUSELAGE_STL = find_existing_file(script_dir, ["model-body.stl", "model-body.STL"])

if WING_STL is None:
    print("❌ 找不到机翼文件：model-left-wing.stl / model-left-wing.STL")
    exit(1)
if FUSELAGE_STL is None:
    print("❌ 找不到机身文件：model-body.stl / model-body.STL")
    exit(1)

print(f"✅ 找到机翼: {WING_STL}")
print(f"✅ 找到机身: {FUSELAGE_STL}")
print(f"📁 相机图像输出目录: {output_dir}")
if not PIL_AVAILABLE:
    print("⚠️ 未检测到 Pillow，按 C 保存 PNG 会失败。请先运行: pip install pillow")

# ============================================================
# 创建 URDF 文件
# ============================================================


def create_urdf(stl_path, name, scale=0.01, mass=1000.0, color=None, use_collision=True):
    if color is None:
        color = [0.7, 0.7, 0.8]
    abs_path = os.path.abspath(stl_path).replace('\\', '/')

    collision_block = ""
    if use_collision:
        # 当前程序已有六自由度控制，保留 collision 可以继续使用 getAABB 等功能。
        # 如果加载很慢，可把 use_collision 改为 False，只做视觉显示。
        collision_block = f'''
    <collision>
      <geometry>
        <mesh filename="{abs_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>'''

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
    </visual>{collision_block}
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1000.0" ixy="0" ixz="0" iyy="1000.0" iyz="0" izz="1000.0"/>
    </inertial>
  </link>
</robot>'''
    return urdf_content


# 注意：这里沿用你原代码的 scale=0.01，保证模型尺度不突然变化。
# 如果以后要按 mm -> m 的物理尺度，请统一改成 scale=0.001，并同步调整相机距离。
MESH_SCALE = 0.01
USE_COLLISION_MESH = True

wing_urdf = os.path.join(script_dir, "wing_temp.urdf")
fuselage_urdf = os.path.join(script_dir, "fuselage_temp.urdf")

with open(wing_urdf, 'w', encoding='utf-8') as f:
    f.write(create_urdf(WING_STL, "wing", scale=MESH_SCALE, mass=5000.0,
                        color=[0.2, 0.4, 0.8], use_collision=USE_COLLISION_MESH))

with open(fuselage_urdf, 'w', encoding='utf-8') as f:
    f.write(create_urdf(FUSELAGE_STL, "fuselage", scale=MESH_SCALE, mass=2000.0,
                        color=[0.7, 0.7, 0.8], use_collision=USE_COLLISION_MESH))

print("✅ URDF 文件创建完成")

# ============================================================
# 启动 PyBullet
# ============================================================

print("⏳ 启动 PyBullet...")
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # 无重力
p.setTimeStep(1.0 / 240.0)

# ============================================================
# 加载地面
# ============================================================

print("⏳ 加载地面...")

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

for i in range(-50, 51, 5):
    p.addUserDebugLine([i, -50, 0.01], [i, 50, 0.01], [0.5, 0.5, 0.5], lineWidth=1)
    p.addUserDebugLine([-50, i, 0.01], [50, i, 0.01], [0.5, 0.5, 0.5], lineWidth=1)

print("✅ 地面加载完成 (50x50米)")

# ============================================================
# 加载机身（固定）
# ============================================================

print("⏳ 加载机身...")
try:
    fuselage_id = p.loadURDF(
        fuselage_urdf,
        basePosition=[0, 0, 0.5],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
    )
    print("✅ 机身加载成功 (固定)")
except Exception as e:
    print(f"❌ 机身加载失败: {e}")
    p.disconnect()
    exit(1)

fuselage_aabb = p.getAABB(fuselage_id)
fuselage_center = [(fuselage_aabb[0][i] + fuselage_aabb[1][i]) / 2 for i in range(3)]
fuselage_size = [fuselage_aabb[1][i] - fuselage_aabb[0][i] for i in range(3)]
print(
    f"📐 机身: 尺寸 {fuselage_size[0]:.2f}x{fuselage_size[1]:.2f}x{fuselage_size[2]:.2f}m, "
    f"中心 ({fuselage_center[0]:.2f}, {fuselage_center[1]:.2f}, {fuselage_center[2]:.2f})"
)

# ============================================================
# 加载机翼（可移动，六自由度）
# ============================================================

print("⏳ 加载机翼...")
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
    print("✅ 机翼加载成功 (六自由度可动)")

    p.changeDynamics(
        wing_id, -1,
        lateralFriction=0.8,
        restitution=0.05,
        rollingFriction=50.0,
        spinningFriction=50.0
    )

except Exception as e:
    print(f"❌ 机翼加载失败: {e}")
    p.disconnect()
    exit(1)

wing_aabb = p.getAABB(wing_id)
wing_centroid = [(wing_aabb[0][i] + wing_aabb[1][i]) / 2 for i in range(3)]
wing_size = [wing_aabb[1][i] - wing_aabb[0][i] for i in range(3)]
print(
    f"📐 机翼: 尺寸 {wing_size[0]:.2f}x{wing_size[1]:.2f}x{wing_size[2]:.2f}m, "
    f"形心 ({wing_centroid[0]:.2f}, {wing_centroid[1]:.2f}, {wing_centroid[2]:.2f})"
)

# ============================================================
# 辅助标记
# ============================================================

p.addUserDebugLine([0, 0, 0], [5, 0, 0], [1, 0, 0], lineWidth=3)
p.addUserDebugLine([0, 0, 0], [0, 5, 0], [0, 1, 0], lineWidth=3)
p.addUserDebugLine([0, 0, 0], [0, 0, 5], [0, 0, 1], lineWidth=3)
p.addUserDebugText("X", [5.5, 0, 0], textColorRGB=[1, 0, 0], textSize=1.5)
p.addUserDebugText("Y", [0, 5.5, 0], textColorRGB=[0, 1, 0], textSize=1.5)
p.addUserDebugText("Z", [0, 0, 5.5], textColorRGB=[0, 0, 1], textSize=1.5)

p.addUserDebugText("Fuselage fixed", [fuselage_center[0], fuselage_center[1], fuselage_center[2] + 2.5],
                   textColorRGB=[0.7, 0.7, 0.8], textSize=1.3)
p.addUserDebugText("Wing 6-DOF", [wing_centroid[0], wing_centroid[1], wing_centroid[2] + 2.0],
                   textColorRGB=[0.2, 0.4, 0.8], textSize=1.3)

# ============================================================
# 六自由度控制变量与函数
# ============================================================

current_pos = [3.0, 0, 1.0]
current_quat = [0, 0, 0, 1]

TRANSLATION_STEP = 0.1
ROTATION_STEP = 2.0


def get_wing_centroid():
    """获取机翼当前的 AABB 形心位置。"""
    aabb = p.getAABB(wing_id)
    return [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]


def reset_wing():
    """重置机翼。"""
    global current_pos, current_quat
    current_pos = [3.0, 0, 1.0]
    current_quat = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)
    p.resetBaseVelocity(wing_id, [0, 0, 0], [0, 0, 0])
    print("\n🔄 机翼已重置")


def move_wing_translation(dx, dy, dz):
    """平移机翼。"""
    global current_pos
    current_pos[0] += dx
    current_pos[1] += dy
    current_pos[2] += dz
    current_pos = np.clip(current_pos, [-20, -20, 0.1], [20, 20, 20]).tolist()
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)


def rotate_wing(axis, angle_deg):
    """绕当前机翼 AABB 形心旋转。"""
    global current_pos, current_quat

    centroid = get_wing_centroid()
    r = R.from_quat(current_quat)

    if axis == 'x':
        delta_r = R.from_euler('x', angle_deg, degrees=True)
    elif axis == 'y':
        delta_r = R.from_euler('y', angle_deg, degrees=True)
    elif axis == 'z':
        delta_r = R.from_euler('z', angle_deg, degrees=True)
    else:
        return

    new_r = delta_r * r
    new_quat = new_r.as_quat()

    pos, _ = p.getBasePositionAndOrientation(wing_id)
    centroid_local = np.array(centroid) - np.array(pos)
    centroid_local_rotated = delta_r.apply(centroid_local)
    new_pos = np.array(centroid) - centroid_local_rotated

    current_pos = new_pos.tolist()
    current_quat = new_quat.tolist()

    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)
    p.resetBaseVelocity(wing_id, [0, 0, 0], [0, 0, 0])


def focus_on_wing():
    """GUI 视角聚焦到机翼。"""
    pos, _ = p.getBasePositionAndOrientation(wing_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=pos
    )
    print("\n🎯 GUI 已聚焦到机翼")


def print_status():
    """打印机翼状态。"""
    pos, quat = p.getBasePositionAndOrientation(wing_id)
    centroid = get_wing_centroid()
    euler = R.from_quat(quat).as_euler('xyz', degrees=True)
    print(f"\r📍 机翼位置: X={pos[0]:6.2f} Y={pos[1]:6.2f} Z={pos[2]:6.2f} | "
          f"形心: X={centroid[0]:6.2f} Y={centroid[1]:6.2f} Z={centroid[2]:6.2f} | "
          f"姿态: Roll={euler[0]:6.1f}° Pitch={euler[1]:6.1f}° Yaw={euler[2]:6.1f}°", end="")

# ============================================================
# 三相机配置与调试模块
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 55.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 100.0

def estimate_interface_y(wing_aabb, fuselage_aabb):
    """
    根据机翼和机身 AABB 自动估计翼身接口附近的 y 坐标。

    你的模型中，机身和左机翼主要沿 y 方向连接；因此这里优先用 y 轴
    判断翼身交界位置：
    - 若两个 AABB 在 y 方向有重叠，取重叠区中点；
    - 若没有重叠，取二者最近边界的中点。
    """
    w_min = np.array(wing_aabb[0], dtype=float)
    w_max = np.array(wing_aabb[1], dtype=float)
    f_min = np.array(fuselage_aabb[0], dtype=float)
    f_max = np.array(fuselage_aabb[1], dtype=float)

    if w_min[1] <= f_max[1] and f_min[1] <= w_max[1]:
        overlap_min = max(w_min[1], f_min[1])
        overlap_max = min(w_max[1], f_max[1])
        return 0.5 * (overlap_min + overlap_max)

    if w_max[1] < f_min[1]:
        return 0.5 * (w_max[1] + f_min[1])

    return 0.5 * (f_max[1] + w_min[1])


def compute_user_requested_camera_layout():
    """
    面向“机翼-机身对接”的三相机布局：
    - Cam1：侧向相机，沿 X 方向观察翼身接口，主要约束横向/高度误差；
    - Cam2：接口正向相机，沿 Y 方向观察翼身接口，主要约束对接间隙/推进方向误差；
    - Cam3：上方斜俯视相机，观察 X-Y 平面误差和 yaw 偏差；

    布局全部基于当前机身 AABB 与机翼 AABB 自动计算。
    相机共同 target 不再使用机翼形心，而是使用两者 AABB 的最近/重叠区域中心，
    更接近翼身接口区域。
    """
    wing_aabb_now = p.getAABB(wing_id)
    fuselage_aabb_now = p.getAABB(fuselage_id)

    w_min = np.array(wing_aabb_now[0], dtype=float)
    w_max = np.array(wing_aabb_now[1], dtype=float)
    f_min = np.array(fuselage_aabb_now[0], dtype=float)
    f_max = np.array(fuselage_aabb_now[1], dtype=float)

    wing_center = 0.5 * (w_min + w_max)
    fuselage_center = 0.5 * (f_min + f_max)
    wing_size_now = w_max - w_min

    # ------------------------------------------------------------
    # 1. 自动估计翼身接口中心点 interface_point
    #    对每个坐标轴：
    #    - 若机翼/机身 AABB 在该轴有重叠，则取重叠区中点；
    #    - 若没有重叠，则取最近两个边界的中点。
    # ------------------------------------------------------------
    interface_point = []
    for axis in range(3):
        if w_min[axis] <= f_max[axis] and f_min[axis] <= w_max[axis]:
            overlap_min = max(w_min[axis], f_min[axis])
            overlap_max = min(w_max[axis], f_max[axis])
            interface_point.append(0.5 * (overlap_min + overlap_max))
        elif w_max[axis] < f_min[axis]:
            interface_point.append(0.5 * (w_max[axis] + f_min[axis]))
        else:
            interface_point.append(0.5 * (f_max[axis] + w_min[axis]))

    target = np.array(interface_point, dtype=float)

    # 为了让相机高度更接近机翼根部，把 target.z 稍微向机翼中心高度靠拢。
    # 这样不会因为 AABB 的上下边界噪声导致相机盯到过高/过低的位置。
    target[2] = 0.65 * wing_center[2] + 0.35 * target[2]

    # ------------------------------------------------------------
    # 2. 判断机翼相对机身位于哪一侧，用来决定相机放在外侧。
    # ------------------------------------------------------------
    wing_side_x = np.sign(wing_center[0] - fuselage_center[0])
    if abs(wing_side_x) < 1e-6:
        wing_side_x = -1.0

    wing_side_y = np.sign(wing_center[1] - fuselage_center[1])
    if abs(wing_side_y) < 1e-6:
        wing_side_y = -1.0

    # ------------------------------------------------------------
    # 3. 设置相机距离。
    #    数值随模型尺寸自适应，但限制在一个不至于过近/过远的范围。
    # ------------------------------------------------------------
    dominant_size = float(max(wing_size_now[0], wing_size_now[1], 1.0))

    x_side_distance = float(np.clip(0.45 * dominant_size, 3.0, 9.0))
    y_front_distance = float(np.clip(0.45 * dominant_size, 3.0, 9.0))

    # 侧向/正向相机与机翼平齐，但略微抬高，避免完全水平视角被模型边缘遮挡。
    side_height_offset = float(np.clip(0.06 * dominant_size, 0.25, 1.20))

    # 顶部相机改为“斜俯视”，不是完全正上方；高度也控制得较低。
    top_height = float(np.clip(0.35 * dominant_size, 2.0, 5.0))
    top_y_offset = float(np.clip(0.22 * y_front_distance, 0.8, 2.5))

    # ------------------------------------------------------------
    # 4. 三台相机位置。
    # ------------------------------------------------------------
    # Cam1：在机翼外侧沿 X 方向看接口。
    cam1_eye = target + np.array([
        wing_side_x * x_side_distance,
        0.0,
        side_height_offset,
    ], dtype=float)

    # Cam2：沿 Y 方向看接口，专门观察对接推进方向/接口间隙。
    cam2_eye = target + np.array([
        0.0,
        wing_side_y * y_front_distance,
        side_height_offset,
    ], dtype=float)

    # Cam3：上方斜俯视，略偏向 Cam2 所在侧，兼顾 X-Y 平面误差与 yaw 偏差。
    cam3_eye = target + np.array([
        0.0,
        wing_side_y * top_y_offset,
        top_height,
    ], dtype=float)

    offsets = [cam1_eye - target, cam2_eye - target, cam3_eye - target]

    layout_info = {
        "wing_aabb": [w_min.tolist(), w_max.tolist()],
        "fuselage_aabb": [f_min.tolist(), f_max.tolist()],
        "wing_center": wing_center.tolist(),
        "fuselage_center": fuselage_center.tolist(),
        "interface_point": target.tolist(),
        "interface_y": float(target[1]),
        "wing_side_x": float(wing_side_x),
        "wing_side_y": float(wing_side_y),
        "x_side_distance": x_side_distance,
        "y_front_distance": y_front_distance,
        "side_height_offset": side_height_offset,
        "top_y_offset": top_y_offset,
        "top_height": top_height,
    }

    return target, offsets, layout_info

# 初始测量目标点与三相机偏移：直接由机翼/机身 AABB 自动计算。
measurement_target, camera_base_offsets, auto_layout_info = compute_user_requested_camera_layout()

print(
    f"🎯 自动相机目标点: X={measurement_target[0]:.2f}, "
    f"Y={measurement_target[1]:.2f}, Z={measurement_target[2]:.2f}"
)
print(
    f"📌 自动估计翼身接口 y={auto_layout_info['interface_y']:.2f}, "
    f"Cam1 X侧距离={auto_layout_info['x_side_distance']:.2f}, "
    f"Cam2 Y向距离={auto_layout_info['y_front_distance']:.2f}, "
    f"Cam3 斜俯视高度={auto_layout_info['top_height']:.2f}"
)

camera_names = ["cam_1_x_side_interface", "cam_2_y_gap_interface", "cam_3_oblique_top"]
camera_colors = [[1, 0, 0], [0, 1, 0], [0, 0.35, 1]]
# Cam3 为上方斜俯视，相机 up 使用 Y 方向，避免接近俯视时与视线方向共线。
camera_up_vectors = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]

# GUI 滑块：在自动计算出的相机位置基础上做微调。
# 注意：滑块读数是 fine adjustment，不是世界坐标，也不是完整 offset。
camera_sliders = []
for i, offset in enumerate(camera_base_offsets):
    prefix = f"Cam{i + 1}"
    sx = p.addUserDebugParameter(f"{prefix} fine X", -10, 10, 0.0)
    sy = p.addUserDebugParameter(f"{prefix} fine Y", -10, 10, 0.0)
    sz = p.addUserDebugParameter(f"{prefix} fine Z", -10, 10, 0.0)
    camera_sliders.append((sx, sy, sz))

# 目标点微调滑块。一般先不用动，特殊情况下可以微调相机共同看向的位置。
target_slider_x = p.addUserDebugParameter("Target fine X", -5, 5, 0)
target_slider_y = p.addUserDebugParameter("Target fine Y", -5, 5, 0)
target_slider_z = p.addUserDebugParameter("Target fine Z", -5, 5, 0)

# 自动定时拍摄间隔滑块。按 P 启动/停止；启动后按该间隔自动保存三台相机图像。
auto_capture_interval_slider = p.addUserDebugParameter("Auto capture interval (s)", 1.0, 30.0, 3.0)

camera_marker_ids = []
for i in range(3):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.15,
        rgbaColor=[camera_colors[i][0], camera_colors[i][1], camera_colors[i][2], 1]
    )
    marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=-1,
        basePosition=[0, 0, 0]
    )
    camera_marker_ids.append(marker_id)

target_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.12,
    rgbaColor=[1, 1, 0, 1]
)
target_marker_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=target_visual,
    baseCollisionShapeIndex=-1,
    basePosition=measurement_target.tolist()
)

camera_debug_line_ids = []
camera_debug_text_ids = []

def get_camera_target():
    """读取目标点微调量，返回当前相机共同目标点。"""
    fine = np.array([
        p.readUserDebugParameter(target_slider_x),
        p.readUserDebugParameter(target_slider_y),
        p.readUserDebugParameter(target_slider_z),
    ], dtype=float)
    return measurement_target + fine


def get_camera_configs():
    """从 GUI 滑块读取三台相机的实时配置。"""
    target = get_camera_target()
    configs = []
    for i, sliders in enumerate(camera_sliders):
        fx = p.readUserDebugParameter(sliders[0])
        fy = p.readUserDebugParameter(sliders[1])
        fz = p.readUserDebugParameter(sliders[2])
        fine = np.array([fx, fy, fz], dtype=float)
        eye = target + camera_base_offsets[i] + fine
        configs.append({
            "name": camera_names[i],
            "eye": eye,
            "target": target,
            "up": camera_up_vectors[i],
            "color": camera_colors[i],
            "fov": CAMERA_FOV,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
        })
    return configs


def update_camera_debug_visuals():
    """更新相机位置小球、视线和文字。"""
    global camera_debug_line_ids, camera_debug_text_ids

    for item_id in camera_debug_line_ids + camera_debug_text_ids:
        try:
            p.removeUserDebugItem(item_id)
        except Exception:
            pass
    camera_debug_line_ids = []
    camera_debug_text_ids = []

    configs = get_camera_configs()
    target = get_camera_target()
    p.resetBasePositionAndOrientation(target_marker_id, target.tolist(), [0, 0, 0, 1])

    for i, cam in enumerate(configs):
        eye = cam["eye"]
        color = cam["color"]
        p.resetBasePositionAndOrientation(camera_marker_ids[i], eye.tolist(), [0, 0, 0, 1])

        line_id = p.addUserDebugLine(
            eye.tolist(),
            cam["target"].tolist(),
            color,
            lineWidth=3
        )
        text_id = p.addUserDebugText(
            cam["name"],
            (eye + np.array([0, 0, 0.35])).tolist(),
            textColorRGB=color,
            textSize=1.1
        )
        camera_debug_line_ids.append(line_id)
        camera_debug_text_ids.append(text_id)


def get_view_projection(cam):
    """根据相机配置计算 view/projection matrix。"""
    eye = cam["eye"].tolist()
    target = cam["target"].tolist()

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=cam.get("up", [0, 0, 1])
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR
    )
    return view_matrix, projection_matrix


def render_camera(cam):
    """渲染单台虚拟相机 RGB 图像；当前调试版不再保存 depth/seg .npy。"""
    view_matrix, projection_matrix = get_view_projection(cam)
    img = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb = np.reshape(img[2], (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
    return rgb


def save_all_camera_images(frame_id, capture_mode="manual"):
    """保存三台相机的 RGB 图像和相机参数 JSON；不再保存 depth/seg .npy。"""
    if not PIL_AVAILABLE:
        print("\n❌ 无法保存 PNG：未安装 Pillow。请运行 pip install pillow")
        return

    configs = get_camera_configs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"{capture_mode}_{stamp}_frame_{frame_id}")
    os.makedirs(batch_dir, exist_ok=True)

    params = {
        "frame_id": frame_id,
        "time": stamp,
        "capture_mode": capture_mode,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fov_deg": CAMERA_FOV,
        "near": CAMERA_NEAR,
        "far": CAMERA_FAR,
        "target": get_camera_target().tolist(),
        "cameras": []
    }

    for cam in configs:
        rgb = render_camera(cam)
        rgb_filename = f"{cam['name']}_rgb.png"
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(batch_dir, rgb_filename))
        params["cameras"].append({
            "name": cam["name"],
            "eye": cam["eye"].tolist(),
            "target": cam["target"].tolist(),
            "up": cam.get("up", [0, 0, 1]),
            "fov_deg": CAMERA_FOV,
            "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
            "rgb_file": rgb_filename
        })

    with open(os.path.join(batch_dir, "camera_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"\n📷 已保存三台相机 RGB 图像和参数: {batch_dir}")
    return batch_dir


# ============================================================
# 自动定时拍摄控制
# ============================================================

auto_capture_enabled = False
last_auto_capture_time = 0.0


def get_auto_capture_interval():
    """读取自动拍摄间隔，单位为秒。"""
    try:
        interval = float(p.readUserDebugParameter(auto_capture_interval_slider))
    except Exception:
        interval = 3.0
    return max(0.5, interval)


def toggle_auto_capture(frame_id):
    """按 P 启动/停止自动定时拍摄。启动时立即拍一组。"""
    global auto_capture_enabled, last_auto_capture_time

    auto_capture_enabled = not auto_capture_enabled

    if auto_capture_enabled:
        interval = get_auto_capture_interval()
        print(f"\n▶️ 自动定时拍摄已启动：每 {interval:.1f} s 保存一组三相机图像。按 P 再次停止。")
        save_all_camera_images(frame_id, capture_mode="auto_start")
        last_auto_capture_time = time.monotonic()
    else:
        print("\n⏹️ 自动定时拍摄已停止。")


def maybe_auto_capture(frame_id):
    """若自动拍摄开启，并且到达时间间隔，则保存一组三相机图像。"""
    global last_auto_capture_time

    if not auto_capture_enabled:
        return

    now = time.monotonic()
    interval = get_auto_capture_interval()

    if now - last_auto_capture_time >= interval:
        last_auto_capture_time = now
        save_all_camera_images(frame_id, capture_mode="auto")


def focus_gui_to_target():
    """GUI 视角聚焦到当前测量目标点。"""
    target = get_camera_target()
    p.resetDebugVisualizerCamera(
        cameraDistance=12,
        cameraYaw=-30,
        cameraPitch=-35,
        cameraTargetPosition=target.tolist()
    )
    print("\n🎯 GUI 已聚焦到测量目标点")


# 当前 GUI 是否锁定到某台虚拟相机。
# None 表示自由总览视角；0/1/2 表示锁定到对应相机。
active_camera_view_index = None


def eye_target_to_debug_camera(eye, target):
    """
    将虚拟相机的 eye-target 表示转换为 PyBullet GUI debug camera 参数。

    注意：resetDebugVisualizerCamera 是轨道相机，不是严格的投影相机。
    这里让 GUI 观察方向尽量与 computeViewMatrix 的 eye->target 方向一致，
    用于人工检查相机视野；真正的数据采集仍以 getCameraImage 为准。
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    direction = target - eye
    distance = float(np.linalg.norm(direction))

    if distance < 1e-6:
        return 5.0, 0.0, -30.0

    # PyBullet debug camera 常用约定：
    # yaw=0 大致从 -Y 方向看向目标，pitch<0 表示相机高于目标并向下看。
    yaw = math.degrees(math.atan2(direction[0], direction[1]))
    pitch = math.degrees(math.asin(np.clip(direction[2] / distance, -1.0, 1.0)))
    pitch = float(np.clip(pitch, -89.0, 89.0))

    return distance, yaw, pitch


def set_gui_view_to_virtual_camera(camera_index, lock=True, verbose=True):
    """
    把 PyBullet GUI 观察视角切到某台虚拟相机的视角。

    lock=True 时，后续拖动相机滑块后，GUI 会继续跟随这台相机。
    按 0 可以解除锁定并回到总览视角。
    """
    global active_camera_view_index

    configs = get_camera_configs()
    if camera_index < 0 or camera_index >= len(configs):
        return

    cam = configs[camera_index]
    distance, yaw, pitch = eye_target_to_debug_camera(cam["eye"], cam["target"])

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=cam["target"].tolist()
    )

    if lock:
        active_camera_view_index = camera_index

    if verbose:
        eye = cam["eye"]
        target = cam["target"]
        print(
            f"\n👁️ GUI 已切换到 {cam['name']} 视角 "
            f"| eye=({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f}) "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        )


def set_gui_overview():
    """返回总览视角，并解除相机视角锁定。"""
    global active_camera_view_index
    active_camera_view_index = None
    p.resetDebugVisualizerCamera(
        cameraDistance=20,
        cameraYaw=-30,
        cameraPitch=-35,
        cameraTargetPosition=[1.5, 0, 1.5]
    )
    print("\n🌐 已返回总览视角，相机视角锁定已解除")


def cycle_gui_camera_view():
    """V 键循环：Cam1 → Cam2 → Cam3 → 总览。"""
    global active_camera_view_index
    if active_camera_view_index is None:
        set_gui_view_to_virtual_camera(0, lock=True, verbose=True)
    elif active_camera_view_index == 0:
        set_gui_view_to_virtual_camera(1, lock=True, verbose=True)
    elif active_camera_view_index == 1:
        set_gui_view_to_virtual_camera(2, lock=True, verbose=True)
    else:
        set_gui_overview()


def sync_locked_camera_view():
    """如果 GUI 锁定到某台相机，则让 GUI 跟随相机滑块变化。"""
    if active_camera_view_index is not None:
        set_gui_view_to_virtual_camera(active_camera_view_index, lock=False, verbose=False)


def update_measurement_target_to_wing():
    """将三台相机的共同目标点更新为当前机翼形心；保留当前 offset。"""
    global measurement_target
    measurement_target = np.array(get_wing_centroid(), dtype=float)
    print(
        f"\n🎯 相机目标点已更新为当前机翼形心: "
        f"X={measurement_target[0]:.2f}, Y={measurement_target[1]:.2f}, Z={measurement_target[2]:.2f}"
    )


def reset_camera_layout_to_current_geometry():
    """
    按当前机翼/机身 AABB 重新计算三相机自动布局。

    说明：PyBullet 的 debug slider 不方便在运行时重置数值，因此这里会更新
    measurement_target 和 camera_base_offsets；Cam fine X/Y/Z 的微调量会继续叠加。
    如果想完全回到自动推荐位置，把 Cam fine 和 Target fine 都拖回 0。
    """
    global measurement_target, camera_base_offsets, auto_layout_info

    measurement_target, camera_base_offsets, auto_layout_info = compute_user_requested_camera_layout()
    print(
        f"\n📷 已按当前机翼/机身位置重新计算对接三相机布局："
        f"target=({measurement_target[0]:.2f}, {measurement_target[1]:.2f}, {measurement_target[2]:.2f}), "
        f"interface_y={auto_layout_info['interface_y']:.2f}, "
        f"x_side_distance={auto_layout_info['x_side_distance']:.2f}, "
        f"y_front_distance={auto_layout_info['y_front_distance']:.2f}, "
        f"top_height={auto_layout_info['top_height']:.2f}"
    )


def print_auto_layout_info(prefix="📌 当前自动布局参数"):
    try:
        _, _, info = compute_user_requested_camera_layout()
        print(
            f"\n{prefix}: interface_y={info['interface_y']:.2f}, "
            f"x_side_distance={info['x_side_distance']:.2f}, "
            f"y_front_distance={info['y_front_distance']:.2f}, "
            f"top_height={info['top_height']:.2f}"
        )
    except Exception as e:
        print(f"\n⚠️ 自动布局参数读取失败: {e}")


def print_camera_configs():
    configs = get_camera_configs()
    print("\n📷 当前三相机参数:")
    for cam in configs:
        eye = cam["eye"]
        target = cam["target"]
        print(f"  {cam['name']}: eye=({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f}) "
              f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

# 初始显示相机
update_camera_debug_visuals()

# ============================================================
# 设置 GUI 初始观察视角
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

print("\n" + "=" * 78)
print("🎮 控制说明")
print("=" * 78)
print("")
print("  📍 机翼平移控制 (步长 0.1m):")
print("     A/D  → X轴负/正方向移动")
print("     S/W  → Y轴负/正方向移动")
print("     Q/E  → Z轴负/正方向移动 (下降/上升)")
print("")
print("  🔄 机翼旋转控制 (步长 2°):")
print("     J/L  → 绕X轴旋转 (滚转)")
print("     K/I  → 绕Y轴旋转 (俯仰)")
print("     U/O  → 绕Z轴旋转 (偏航)")
print("")
print("  📷 三相机调试:")
print("     右侧 Debug Sliders → 调整 Cam1/Cam2/Cam3 的 fine X/Y/Z")
print("     T    → 将三台相机共同目标点更新为当前机翼形心")
print("     C    → 保存三台相机 RGB 图像和 camera_params.json（不保存 .npy）")
print("     P    → 启动/停止定时自动拍摄")
print("     H    → 按当前机翼/机身AABB重新计算对接推荐相机布局")
print("     1/2/3→ GUI 切换到相机1/2/3的视角，并锁定跟随")
print("     0    → 返回总览视角，并解除相机视角锁定")
print("     V    → 在相机1/2/3/总览之间循环切换")
print("     G    → GUI 聚焦到测量目标点")
print("")
print("  🎯 其他:")
print("     R    → 重置机翼位置和姿态")
print("     F    → GUI 聚焦到机翼")
print("     ESC  → 退出程序")
print("=" * 78 + "\n")

# ============================================================
# 主循环
# ============================================================

frame_count = 0
last_camera_debug_update = 0

try:
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        frame_count += 1

        # 不必每帧刷新 debug line；10 Hz 足够，避免 GUI 卡顿。
        if frame_count - last_camera_debug_update >= 24:
            update_camera_debug_visuals()
            sync_locked_camera_view()
            last_camera_debug_update = frame_count

        if frame_count % 60 == 0:
            print_status()

        maybe_auto_capture(frame_count)

        keys = p.getKeyboardEvents()

        if KEY_EXIT in keys and keys[KEY_EXIT] & p.KEY_WAS_TRIGGERED:
            print("\n👋 退出程序")
            break

        if KEY_R in keys and keys[KEY_R] & p.KEY_WAS_TRIGGERED:
            reset_wing()

        if KEY_F in keys and keys[KEY_F] & p.KEY_WAS_TRIGGERED:
            focus_on_wing()

        if KEY_G in keys and keys[KEY_G] & p.KEY_WAS_TRIGGERED:
            focus_gui_to_target()

        if KEY_T in keys and keys[KEY_T] & p.KEY_WAS_TRIGGERED:
            update_measurement_target_to_wing()
            print_camera_configs()

        if KEY_C in keys and keys[KEY_C] & p.KEY_WAS_TRIGGERED:
            save_all_camera_images(frame_count, capture_mode="manual")
            print_camera_configs()

        if KEY_P in keys and keys[KEY_P] & p.KEY_WAS_TRIGGERED:
            toggle_auto_capture(frame_count)

        if KEY_H in keys and keys[KEY_H] & p.KEY_WAS_TRIGGERED:
            reset_camera_layout_to_current_geometry()
            update_camera_debug_visuals()
            print_camera_configs()

        if KEY_1 in keys and keys[KEY_1] & p.KEY_WAS_TRIGGERED:
            set_gui_view_to_virtual_camera(0, lock=True, verbose=True)
        if KEY_2 in keys and keys[KEY_2] & p.KEY_WAS_TRIGGERED:
            set_gui_view_to_virtual_camera(1, lock=True, verbose=True)
        if KEY_3 in keys and keys[KEY_3] & p.KEY_WAS_TRIGGERED:
            set_gui_view_to_virtual_camera(2, lock=True, verbose=True)
        if KEY_0 in keys and keys[KEY_0] & p.KEY_WAS_TRIGGERED:
            set_gui_overview()
        if KEY_V in keys and keys[KEY_V] & p.KEY_WAS_TRIGGERED:
            cycle_gui_camera_view()

        step_t = TRANSLATION_STEP
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

        step_r = ROTATION_STEP
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
    print("\n👋 用户中断")

finally:
    p.disconnect()
    try:
        os.remove(wing_urdf)
        os.remove(fuselage_urdf)
    except Exception:
        pass
    print("🔚 仿真已关闭")
