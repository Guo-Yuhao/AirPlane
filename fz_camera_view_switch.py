import pybullet as p
import pybullet_data
import time
import os
import json
import math
import struct
import tempfile
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from typing import Any

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image: Any = None
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
KEY_H = 104    # 重新计算当前机翼/机身相机布局

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
print("飞机机身 + 机翼六自由度控制 + 三相机布置调试")
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
    print("找不到机翼文件：model-left-wing.stl / model-left-wing.STL")
    exit(1)
if FUSELAGE_STL is None:
    print("找不到机身文件：model-body.stl / model-body.STL")
    exit(1)

print(f"找到机翼: {WING_STL}")
print(f"找到机身: {FUSELAGE_STL}")
print(f"相机图像输出目录: {output_dir}")
if not PIL_AVAILABLE:
    print("警告：未检测到 Pillow，按 C 保存 PNG 会失败。请先运行: pip install pillow")

# ============================================================
# 创建 URDF 文件
# ============================================================


def create_urdf(stl_path, name, scale=0.01, mass=1000.0, color=None, use_collision=True):
    if color is None:
        color = [0.7, 0.7, 0.8]
    abs_path = os.path.abspath(stl_path).replace('\\', '/')

    collision_block = ""
    if use_collision:
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

print("URDF 文件创建完成")


def _load_stl_vertices(stl_path):
    """从二进制 STL 文件中读取顶点坐标。"""
    vertices = []
    with open(stl_path, 'rb') as f:
        header = f.read(80)
        tri_count_data = f.read(4)
        if len(tri_count_data) < 4:
            raise ValueError("不是有效的二进制 STL 文件")
        tri_count = int.from_bytes(tri_count_data, byteorder='little')
        for _ in range(tri_count):
            f.read(12)  # normal
            coords = f.read(36)
            if len(coords) < 36:
                break
            verts = struct.unpack('<9f', coords)
            vertices.extend([(verts[0], verts[1], verts[2]), (verts[3], verts[4], verts[5]), (verts[6], verts[7], verts[8])])
            f.read(2)  # attribute byte count
    return np.array(vertices, dtype=float)


def _create_convex_collision_shape_from_stl(stl_path):
    """从 STL 顶点创建一个近似的凸包碰撞形状。"""
    vertices = _load_stl_vertices(stl_path)
    if vertices.shape[0] == 0:
        raise ValueError("STL 文件未包含顶点")

    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    index_map = {orig_idx: new_idx + 1 for new_idx, orig_idx in enumerate(hull.vertices)}

    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False, mode='w', encoding='utf-8') as f:
        for v in hull_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for simplex in hull.simplices:
            f.write(
                f"f {index_map[simplex[0]]} {index_map[simplex[1]]} {index_map[simplex[2]]}\n"
            )
        obj_path = f.name

    return obj_path


def create_mesh_body(stl_path, basePosition, baseOrientation, fixed, mass, color):
    """使用真实网格创建 PyBullet 可视体，并根据刚体类型创建合适碰撞体。"""
    abs_path = os.path.abspath(stl_path).replace('\\', '/')
    visual_shape = p.createVisualShape(
        p.GEOM_MESH,
        fileName=abs_path,
        meshScale=[MESH_SCALE, MESH_SCALE, MESH_SCALE],
        rgbaColor=[color[0], color[1], color[2], 1.0]
    )

    if USE_COLLISION_MESH:
        if fixed:
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=abs_path,
                meshScale=[MESH_SCALE, MESH_SCALE, MESH_SCALE],
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH
            )
        else:
            obj_path = _create_convex_collision_shape_from_stl(stl_path)
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=obj_path,
                meshScale=[MESH_SCALE, MESH_SCALE, MESH_SCALE]
            )
            try:
                os.remove(obj_path)
            except Exception:
                pass
    else:
        collision_shape = -1

    body_mass = 0 if fixed else mass
    return p.createMultiBody(
        baseMass=body_mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=basePosition,
        baseOrientation=baseOrientation
    )

# ============================================================
# 启动 PyBullet
# ============================================================

print("正在启动 PyBullet...")
physicsClient = p.connect(p.GUI)
if physicsClient < 0:
    print("物理服务器连接失败，程序退出")
    exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # 无重力
p.setTimeStep(1.0 / 240.0)

# ============================================================
# 加载地面
# ============================================================

print("正在加载地面...")

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

print("地面加载完成 (50x50米)")

# ============================================================
# 加载机身（固定）
# ============================================================

print("正在加载机身...")
try:
    fuselage_id = create_mesh_body(
        FUSELAGE_STL,
        basePosition=[0, 0, 0.5],
        baseOrientation=[0, 0, 0, 1],
        fixed=True,
        mass=2000.0,
        color=[0.7, 0.7, 0.8]
    )
    print("机身加载成功 (固定)")
except Exception as e:
    print(f"机身加载失败: {e}")
    if p.isConnected():
        p.disconnect()
    exit(1)

fuselage_aabb = p.getAABB(fuselage_id)
fuselage_center = [(fuselage_aabb[0][i] + fuselage_aabb[1][i]) / 2 for i in range(3)]
fuselage_size = [fuselage_aabb[1][i] - fuselage_aabb[0][i] for i in range(3)]
print(
    f"机身: 尺寸 {fuselage_size[0]:.2f}x{fuselage_size[1]:.2f}x{fuselage_size[2]:.2f}m, "
    f"中心 ({fuselage_center[0]:.2f}, {fuselage_center[1]:.2f}, {fuselage_center[2]:.2f})"
)

# ============================================================
# 加载机翼（可移动，六自由度）
# ============================================================

print("正在加载机翼...")
try:
    initial_position = [3.0, 0, 1.0]
    initial_orientation = [0, 0, 0, 1]

    wing_id = create_mesh_body(
        WING_STL,
        basePosition=initial_position,
        baseOrientation=initial_orientation,
        fixed=False,
        mass=5000.0,
        color=[0.2, 0.4, 0.8]
    )
    print("机翼加载成功 (六自由度可动)")

    p.changeDynamics(
        wing_id, -1,
        lateralFriction=0.8,
        restitution=0.05,
        rollingFriction=50.0,
        spinningFriction=50.0
    )

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
    f"机翼: 尺寸 {wing_size[0]:.2f}x{wing_size[1]:.2f}x{wing_size[2]:.2f}m, "
    f"形心 ({wing_init_centroid[0]:.2f}, {wing_init_centroid[1]:.2f}, {wing_init_centroid[2]:.2f})"
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
p.addUserDebugText("Wing 6-DOF", [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 2.0],
                   textColorRGB=[0.2, 0.4, 0.8], textSize=1.3)

# 机翼初始形心标记
p.addUserDebugLine([wing_init_centroid[0] - 0.3, wing_init_centroid[1], wing_init_centroid[2]],
                   [wing_init_centroid[0] + 0.3, wing_init_centroid[1], wing_init_centroid[2]], [1, 1, 0], lineWidth=3)
p.addUserDebugLine([wing_init_centroid[0], wing_init_centroid[1] - 0.3, wing_init_centroid[2]],
                   [wing_init_centroid[0], wing_init_centroid[1] + 0.3, wing_init_centroid[2]], [1, 1, 0], lineWidth=3)
p.addUserDebugLine([wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] - 0.3],
                   [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 0.3], [1, 1, 0], lineWidth=3)

p.addUserDebugText("centroid", [wing_init_centroid[0], wing_init_centroid[1], wing_init_centroid[2] + 0.6],
                   textColorRGB=[1, 1, 0], textSize=1.2)

# ============================================================
# 六自由度控制参数与函数
# ============================================================

current_pos = list(initial_position)
current_quat = list(initial_orientation)

TRANSLATION_SPEED = 2.0   # 平移速度：米/秒
ROTATION_SPEED = 30.0     # 旋转速度：度/秒


def get_wing_centroid():
    """获取机翼当前的形心世界坐标（基于固定局部形心偏移计算）"""
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
    print("\n机翼已重置")


def _is_wing_position_valid(candidate_pos):
    """检测机翼在给定位姿是否与机身碰撞。"""
    old_pos, old_quat = p.getBasePositionAndOrientation(wing_id)
    p.resetBasePositionAndOrientation(wing_id, candidate_pos, current_quat)
    contacts = p.getClosestPoints(bodyA=fuselage_id, bodyB=wing_id, distance=0.0)
    p.resetBasePositionAndOrientation(wing_id, old_pos, old_quat)
    return len(contacts) == 0


def move_wing_translation(dx, dy, dz):
    """平移机翼（世界坐标系下平移），碰撞时按轴阻断对应方向移动。"""
    global current_pos

    max_bounds = np.array([20, 20, 20], dtype=float)
    min_bounds = np.array([-20, -20, 0.1], dtype=float)

    target_pos = np.array(current_pos, dtype=float)
    for axis, delta in enumerate((dx, dy, dz)):
        if abs(delta) < 1e-9:
            continue
        trial_pos = target_pos.copy()
        trial_pos[axis] += delta
        trial_pos = np.clip(trial_pos, min_bounds, max_bounds)
        if _is_wing_position_valid(trial_pos.tolist()):
            target_pos = trial_pos
        else:
            # 遇到与机身碰撞时，只阻断该方向，不影响其他轴移动。
            pass

    current_pos = target_pos.tolist()
    p.resetBasePositionAndOrientation(wing_id, current_pos, current_quat)
    p.resetBaseVelocity(wing_id, [0, 0, 0], [0, 0, 0])


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
    """GUI 视角聚焦到机翼形心"""
    centroid = get_wing_centroid()
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=centroid
    )
    print("\nGUI 已聚焦到机翼")


def print_status():
    """实时打印机翼状态信息"""
    pos, quat = p.getBasePositionAndOrientation(wing_id)
    centroid = get_wing_centroid()
    euler = R.from_quat(quat).as_euler('xyz', degrees=True)
    print(f"\r机翼位置: X={pos[0]:6.2f} Y={pos[1]:6.2f} Z={pos[2]:6.2f} | "
          f"形心: X={centroid[0]:6.2f} Y={centroid[1]:6.2f} Z={centroid[2]:.2f} | "
          f"姿态: Roll={euler[0]:6.1f}° Pitch={euler[1]:6.1f}° Yaw={euler[2]:6.1f}°", end="")


def estimate_interface_y(wing_aabb, fuselage_aabb):
    """估计翼身接口附近的 y 坐标。"""
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
    """根据当前机翼和机身 AABB 计算推荐的三相机布局。"""
    wing_aabb_now = p.getAABB(wing_id)
    fuselage_aabb_now = p.getAABB(fuselage_id)

    w_min = np.array(wing_aabb_now[0], dtype=float)
    w_max = np.array(wing_aabb_now[1], dtype=float)
    f_min = np.array(fuselage_aabb_now[0], dtype=float)
    f_max = np.array(fuselage_aabb_now[1], dtype=float)

    wing_center = 0.5 * (w_min + w_max)
    fuselage_center = 0.5 * (f_min + f_max)
    wing_size_now = w_max - w_min

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
    target[2] = 0.65 * wing_center[2] + 0.35 * target[2]

    wing_side_x = np.sign(wing_center[0] - fuselage_center[0])
    if abs(wing_side_x) < 1e-6:
        wing_side_x = -1.0

    wing_side_y = np.sign(wing_center[1] - fuselage_center[1])
    if abs(wing_side_y) < 1e-6:
        wing_side_y = -1.0

    dominant_size = float(max(wing_size_now[0], wing_size_now[1], 1.0))
    x_side_distance = float(np.clip(0.45 * dominant_size, 3.0, 9.0))
    y_front_distance = float(np.clip(0.45 * dominant_size, 3.0, 9.0))
    side_height_offset = float(np.clip(0.06 * dominant_size, 0.25, 1.20))
    top_height = float(np.clip(0.35 * dominant_size, 2.0, 5.0))
    top_y_offset = float(np.clip(0.22 * y_front_distance, 0.8, 2.5))

    cam1_eye = target + np.array([wing_side_x * x_side_distance, 0.0, side_height_offset], dtype=float)
    cam2_eye = target + np.array([0.0, wing_side_y * y_front_distance, side_height_offset], dtype=float)
    cam3_eye = target + np.array([0.0, wing_side_y * top_y_offset, top_height], dtype=float)

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

# ============================================================
# 三相机配置与调试模块
# ============================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 55.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 100.0

# 初始测量目标点与三相机偏移：直接由机翼/机身 AABB 自动计算。
measurement_target, camera_base_offsets, auto_layout_info = compute_user_requested_camera_layout()
print(
    f"初始相机测量目标点: X={measurement_target[0]:.2f}, Y={measurement_target[1]:.2f}, Z={measurement_target[2]:.2f}"
)
print(
    f"自动估计翼身接口 y={auto_layout_info['interface_y']:.2f}, "
    f"Cam1 X侧距离={auto_layout_info['x_side_distance']:.2f}, "
    f"Cam2 Y向距离={auto_layout_info['y_front_distance']:.2f}, "
    f"Cam3 斜俯视高度={auto_layout_info['top_height']:.2f}"
)

# 三台相机相对 measurement_target 的初始偏移。
camera_names = ["cam_1_x_side_interface", "cam_2_y_gap_interface", "cam_3_oblique_top"]
camera_colors = [[1, 0, 0], [0, 1, 0], [0, 0.35, 1]]
camera_up_vectors = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]

# GUI 滑块：在自动计算出的相机位置基础上做 fine 调整。
# slider 表示相对于自动偏移的微调量。
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
        fine_offset = np.array([fx, fy, fz], dtype=float)
        eye = target + camera_base_offsets[i] + fine_offset
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
    # 更新相机位置小球、视线和文字。
    if not p.isConnected():
        return
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
    """渲染单台虚拟相机图像。"""
    view_matrix, projection_matrix = get_view_projection(cam)
    img = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb = np.reshape(img[2], (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
    depth = np.reshape(img[3], (CAMERA_HEIGHT, CAMERA_WIDTH))
    seg = np.reshape(img[4], (CAMERA_HEIGHT, CAMERA_WIDTH))
    return rgb, depth, seg


def save_all_camera_images(frame_id, capture_mode="manual"):
    """保存三台相机的 RGB 图像，并保存相机参数 JSON。"""
    if not PIL_AVAILABLE:
        print("\n无法保存 PNG：未安装 Pillow。请运行 pip install pillow")
        return

    configs = get_camera_configs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"{capture_mode}_{stamp}_frame_{frame_id}")
    os.makedirs(batch_dir, exist_ok=True)

    params = {
        "frame_id": frame_id,
        "capture_mode": capture_mode,
        "time": stamp,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fov_deg": CAMERA_FOV,
        "near": CAMERA_NEAR,
        "far": CAMERA_FAR,
        "target": get_camera_target().tolist(),
        "cameras": []
    }

    for cam in configs:
        rgb, depth, seg = render_camera(cam)
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(batch_dir, f"{cam['name']}_rgb.png"))
        np.save(os.path.join(batch_dir, f"{cam['name']}_depth.npy"), depth)
        np.save(os.path.join(batch_dir, f"{cam['name']}_seg.npy"), seg)
        params["cameras"].append({
            "name": cam["name"],
            "eye": cam["eye"].tolist(),
            "target": cam["target"].tolist(),
            "fov_deg": CAMERA_FOV,
            "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT]
        })

    with open(os.path.join(batch_dir, "camera_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"\n已保存三台相机图像和参数: {batch_dir}")
    return batch_dir


def focus_gui_to_target():
    """GUI 视角聚焦到当前测量目标点。"""
    target = get_camera_target()
    p.resetDebugVisualizerCamera(
        cameraDistance=12,
        cameraYaw=-30,
        cameraPitch=-35,
        cameraTargetPosition=target.tolist()
    )
    print("\nGUI 已聚焦到测量目标点")


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
        print(f"\n自动定时拍摄已启动：每 {interval:.1f} s 保存一组三相机图像。按 P 再次停止。")
        save_all_camera_images(frame_id, capture_mode="auto_start")
        last_auto_capture_time = time.monotonic()
    else:
        print("\n自动定时拍摄已停止。")


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


def reset_camera_layout_to_current_geometry():
    """按当前机翼/机身 AABB 重新计算三相机自动布局。"""
    global measurement_target, camera_base_offsets, auto_layout_info

    measurement_target, camera_base_offsets, auto_layout_info = compute_user_requested_camera_layout()
    print(
        f"\n已按当前机翼/机身位置重新计算对接三相机布局："
        f"target=({measurement_target[0]:.2f}, {measurement_target[1]:.2f}, {measurement_target[2]:.2f}), "
        f"interface_y={auto_layout_info['interface_y']:.2f}, "
        f"x_side_distance={auto_layout_info['x_side_distance']:.2f}, "
        f"y_front_distance={auto_layout_info['y_front_distance']:.2f}, "
        f"top_height={auto_layout_info['top_height']:.2f}"
    )


def print_auto_layout_info(prefix="当前自动布局参数"):
    try:
        _, _, info = compute_user_requested_camera_layout()
        print(
            f"\n{prefix}: interface_y={info['interface_y']:.2f}, "
            f"x_side_distance={info['x_side_distance']:.2f}, "
            f"y_front_distance={info['y_front_distance']:.2f}, "
            f"top_height={info['top_height']:.2f}"
        )
    except Exception as e:
        print(f"\n自动布局参数读取失败: {e}")


# 当前 GUI 是否锁定到某台虚拟相机。
# None 表示自由总览视角；0/1/2 表示锁定到对应相机。
active_camera_view_index = None


def eye_target_to_debug_camera(eye, target):
    """
    将虚拟相机的 eye-target 表示转换为 PyBullet GUI debug camera 参数。
    用于人工检查相机视野；真正的数据采集仍以 getCameraImage 为准。
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    direction = target - eye
    distance = float(np.linalg.norm(direction))

    if distance < 1e-6:
        return 5.0, 0.0, -30.0

    yaw = math.degrees(math.atan2(direction[0], direction[1]))
    pitch = math.degrees(math.asin(np.clip(direction[2] / distance, -1.0, 1.0)))
    pitch = float(np.clip(pitch, -89.0, 89.0))

    return distance, yaw, pitch


def set_gui_view_to_virtual_camera(camera_index, lock=True, verbose=True):
    """
    把 PyBullet GUI 观察视角切到某台虚拟相机的视角。
    lock=True 时，后续拖动相机滑块后，GUI 会继续跟随这台相机。
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
            f"\nGUI 已切换到 {cam['name']} 视角 "
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
    print("\n已返回总览视角，相机视角锁定已解除")


def cycle_gui_camera_view():
    """V 键循环：Cam1 -> Cam2 -> Cam3 -> 总览。"""
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
    """将三台相机的共同目标点更新为当前机翼形心。"""
    global measurement_target
    measurement_target = np.array(get_wing_centroid(), dtype=float)
    print(
        f"\n相机目标点已更新为当前机翼形心: "
        f"X={measurement_target[0]:.2f}, Y={measurement_target[1]:.2f}, Z={measurement_target[2]:.2f}"
    )


def print_camera_configs():
    configs = get_camera_configs()
    print("\n当前三相机参数:")
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
print("控制说明")
print("=" * 78)
print("")
print("  机翼平移控制 (速度 2m/s):")
print("     A/D  -> X轴负/正方向移动")
print("     S/W  -> Y轴负/正方向移动")
print("     Q/E  -> Z轴负/正方向移动 (下降/上升)")
print("")
print("  机翼旋转控制 (速度 30°/s，绕机体局部轴):")
print("     J/L  -> 绕X轴旋转 (滚转)")
print("     K/I  -> 绕Y轴旋转 (俯仰)")
print("     U/O  -> 绕Z轴旋转 (偏航)")
print("")
print("  三相机调试:")
print("     右侧 Debug Sliders -> 调整 Cam1/Cam2/Cam3 的 offset X/Y/Z")
print("     T    -> 将三台相机共同目标点更新为当前机翼形心")
print("     C    -> 保存三台相机 RGB/Depth/Seg 图像和 camera_params.json")
print("     1/2/3-> GUI 切换到相机1/2/3的视角，并锁定跟随")
print("     0    -> 返回总览视角，并解除相机视角锁定")
print("     V    -> 在相机1/2/3/总览之间循环切换")
print("     G    -> GUI 聚焦到测量目标点")
print("")
print("  其他功能:")
print("     R    -> 重置机翼位置和姿态")
print("     F    -> GUI 聚焦到机翼")
print("     ESC  -> 退出程序")
print("=" * 78 + "\n")

# ============================================================
# 主循环
# ============================================================

frame_count = 0
last_camera_debug_update = 0
last_time = time.time()

try:
    while True:
        # 计算帧间隔，保证运动速度与帧率无关
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

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
            print("\n退出程序")
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
except p.error:
    print("\n物理连接已断开，程序退出")

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