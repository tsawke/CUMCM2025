# Q1_fy1_m1_fixed.py
# 问题1：FY1 以 120 m/s 朝假目标方向飞行；受领 1.5 s 后投弹，3.6 s 后起爆。
# 计算烟幕对 M1 的有效遮蔽时长（严格判据：圆柱整体落入以导弹为锥顶、指向云团中心的无限延申圆锥内）。

import math
import numpy as np

# ---------------------- 常量与题目数据 ----------------------
g = 9.8
MISSILE_SPEED = 300.0  # m/s
UAV_SPEED = 120.0      # m/s (Q1固定)
CLOUD_R = 10.0         # 云团半径
CLOUD_SINK = 3.0       # m/s（下沉）
CLOUD_EFFECT = 20.0    # s（有效遮蔽持续时间）

# 真目标（圆柱）参数（题意：底面圆心在 (0,200,0)，半径7m，高10m，轴向+z）
CYL_BASE_CENTER = np.array([0.0, 200.0, 0.0])
CYL_R = 7.0
CYL_H = 10.0

# 初始位置（题面）
M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0])

# 事件时刻（题面）
DROP_T = 1.5
FUSE_DT = 3.6
EXPL_T = DROP_T + FUSE_DT

# ---------------------- 工具函数 ----------------------
def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def missile_state(t, M_init):
    # 直线匀速，指向原点
    dir_to_origin = unit(FAKE_TARGET_ORIGIN - M_init)
    v = MISSILE_SPEED * dir_to_origin
    return M_init + v * t, v

def uav_state_horizontal(t, uav_init, uav_speed, heading_rad):
    # 等高度、水平直线，heading 只作用于 x-y 平面，z 保持初始
    vx = uav_speed * math.cos(heading_rad)
    vy = uav_speed * math.sin(heading_rad)
    x = uav_init[0] + vx * t
    y = uav_init[1] + vy * t
    z = uav_init[2]
    return np.array([x, y, z]), np.array([vx, vy, 0.0])

def cylinder_inside_infinite_cone(M, C, r_cloud,
                                  cyl_base_center=CYL_BASE_CENTER, R=CYL_R, H=CYL_H,
                                  n_phi=180, n_z_side=5):
    """判断圆柱是否完全在以 M 为锥顶、轴向 (C-M)、半顶角 asin(r_cloud/|C-M|) 的无限延申圆锥内。"""
    v = C - M
    L = np.linalg.norm(v)
    if L <= 1e-9 or r_cloud >= L:
        return True  # 导弹与云团中心极近/在云内，视为完全遮蔽

    # sin(alpha)=r/L => cos(alpha)=sqrt(1-(r/L)^2)
    cos_alpha = math.sqrt(max(0.0, 1.0 - (r_cloud/L)**2))
    vL = v / L

    # 采样：底/顶圆 + 侧面若干圈
    B = np.array(cyl_base_center, dtype=float)
    T = B + np.array([0.0, 0.0, H], dtype=float)
    phis = np.linspace(0.0, 2.0*math.pi, n_phi, endpoint=False)

    def in_cone(X):
        w = X - M
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            return True
        cos_theta = np.dot(w, v) / (wn * L)
        return cos_theta + 1e-12 >= cos_alpha

    # 底/顶圆周
    for phi in phis:
        Xb = B + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])
        Xt = T + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])
        if not in_cone(Xb) or not in_cone(Xt):
            return False

    # 侧面圈
    if n_z_side >= 2:
        for k in range(n_z_side):
            z = H * (k/(n_z_side-1.0))
            center = B + np.array([0.0, 0.0, z])
            for phi in phis:
                X = center + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])
                if not in_cone(X):
                    return False

    return True

# ---------------------- Q1 计算流程 ----------------------
def compute_q1_shield_time(dt=0.001, n_phi=180, n_z_side=5):
    # FY1 航向：朝向假目标方向（水平）
    heading = math.atan2(0.0 - FY1_INIT[1], 0.0 - FY1_INIT[0])  # (0,0)方向
    # 投弹瞬间位置&速度
    drop_pos, uav_v = uav_state_horizontal(DROP_T, FY1_INIT, UAV_SPEED, heading)
    # 起爆位置（水平匀速、竖直自由落体）
    expl_xy = drop_pos[:2] + uav_v[:2] * FUSE_DT
    expl_z = drop_pos[2] - 0.5 * g * (FUSE_DT**2)
    expl_pos = np.array([expl_xy[0], expl_xy[1], expl_z])

    # 导弹击中假目标的时间（到原点）
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED

    t0 = EXPL_T
    t1 = min(EXPL_T + CLOUD_EFFECT, hit_time)

    shield = 0.0
    t = t0
    while t <= t1 + 1e-12:
        M, _ = missile_state(t, M1_INIT)
        C = np.array([expl_pos[0], expl_pos[1], expl_pos[2] - CLOUD_SINK * max(0.0, t - EXPL_T)])
        if cylinder_inside_infinite_cone(M, C, CLOUD_R, n_phi=n_phi, n_z_side=n_z_side):
            shield += dt
        t += dt

    return shield, expl_pos, hit_time 

if __name__ == "__main__":
    shield_time, expl_pos, hit_time = compute_q1_shield_time()
    print(f"[Q1] 遮蔽时长（严格圆锥判据）= {shield_time:.3f} s")
    print(f"起爆位置 = {expl_pos}, 导弹命中时间 ~ {hit_time:.3f} s")
