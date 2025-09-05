# Q2_fy1_m1_optimize_single_with_logs.py
# 任务：FY1 仅投 1 枚烟幕弹；搜索航向、速度(70~140)、投弹时刻、引信延时，最大化遮蔽。
# 判据：圆柱整体在无限延申圆锥中（严格）。
# 日志：粗搜/细化阶段打印进度与当前最好解。

import math
import numpy as np

# ---------------------- 常量与数据 ----------------------
g = 9.8
MISSILE_SPEED = 300.0
CLOUD_R = 10.0
CLOUD_SINK = 3.0
CLOUD_EFFECT = 20.0

CYL_BASE_CENTER = np.array([0.0, 200.0, 0.0])
CYL_R = 7.0
CYL_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0])

# 采样设置（兼顾时间与精度）
DT = 0.05
NPHI = 60
NZ_SIDE = 3

# 搜索空间（刻意压缩到可信 + <30min 的范围）
# 航向围绕“朝原点”±15°，步长 3°
BASE_HEADING = math.atan2(0.0 - FY1_INIT[1], 0.0 - FY1_INIT[0])
HEADING_LIST = [BASE_HEADING + math.radians(d) for d in range(-15, 16, 3)]
SPEED_LIST = [70.0, 100.0, 140.0]
DROP_LIST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
FUSE_LIST = [1.0, 2.0, 3.0, 4.0, 5.0]

# ---------------------- 工具 ----------------------
def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12: return v
    return v / n

def missile_state(t, M_init):
    vdir = unit(FAKE_TARGET_ORIGIN - M_init)
    v = MISSILE_SPEED * vdir
    return M_init + v * t, v

def uav_pos_vel_at(t, uav_init, speed, heading):
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([uav_init[0] + vx*t, uav_init[1] + vy*t, uav_init[2]]), np.array([vx, vy, 0.0])

def cylinder_inside_infinite_cone(M, C, r_cloud,
                                  cyl_base_center=CYL_BASE_CENTER, R=CYL_R, H=CYL_H,
                                  n_phi=NPHI, n_z_side=NZ_SIDE):
    v = C - M
    L = np.linalg.norm(v)
    if L <= 1e-9 or r_cloud >= L: return True
    cos_alpha = math.sqrt(max(0.0, 1.0 - (r_cloud/L)**2))
    B = np.array(cyl_base_center, dtype=float)
    T = B + np.array([0.0, 0.0, H], dtype=float)
    phis = np.linspace(0.0, 2.0*math.pi, n_phi, endpoint=False)
    def ok(X):
        w = X - M
        wn = np.linalg.norm(w)
        if wn < 1e-12: return True
        cos_theta = np.dot(w, v) / (wn * L)
        return cos_theta + 1e-12 >= cos_alpha
    for phi in phis:
        if not ok(B + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])): return False
        if not ok(T + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])): return False
    if n_z_side >= 2:
        for k in range(n_z_side):
            z = H * (k/(n_z_side - 1.0))
            center = B + np.array([0.0, 0.0, z])
            for phi in phis:
                if not ok(center + np.array([R*math.cos(phi), R*math.sin(phi), 0.0])): return False
    return True

def explosion_from(speed, heading, drop_t, fuse_dt):
    drop_pos, uav_v = uav_pos_vel_at(drop_t, FY1_INIT, speed, heading)
    expl_xy = drop_pos[:2] + uav_v[:2] * fuse_dt
    expl_z = drop_pos[2] - 0.5 * g * (fuse_dt**2)
    return drop_pos, np.array([expl_xy[0], expl_xy[1], expl_z]), (drop_t + fuse_dt)

def simulate_shield_time(expl_t, expl_pos):
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED
    t0 = expl_t
    t1 = min(expl_t + CLOUD_EFFECT, hit_time)
    steps = int((t1 - t0) / DT) + 1
    shield = 0.0
    for i in range(steps):
        t = t0 + i * DT
        M, _ = missile_state(t, M1_INIT)
        C = np.array([expl_pos[0], expl_pos[1], expl_pos[2] - CLOUD_SINK * max(0.0, t - expl_t)])
        if cylinder_inside_infinite_cone(M, C, CLOUD_R):
            shield += DT
    return shield

# ---------------------- 搜索（含日志） ----------------------
def search_q2():
    print("[Q2] 开始粗搜（限定航向±15°，速度{70,100,140}，drop/fuse离散）...")
    total = len(SPEED_LIST) * len(HEADING_LIST) * len(DROP_LIST) * len(FUSE_LIST)
    cnt = 0
    best = ( -1.0, None )  # (shield_time, params)

    for s in SPEED_LIST:
        for h in HEADING_LIST:
            for d in DROP_LIST:
                for f in FUSE_LIST:
                    cnt += 1
                    if cnt % max(1, total // 20) == 0:
                        print(f"    [Q2] 粗搜进度 {int(100*cnt/total)}%  (s={s}, head={math.degrees(h):.1f}°, drop={d}, fuse={f})")
                    dp, ep, et = explosion_from(s, h, d, f)
                    if ep[2] <= 0:  # 起爆前已落地
                        continue
                    val = simulate_shield_time(et, ep)
                    if val > best[0]:
                        best = (val, (s, h, d, f, ep, et))
                        print(f"      [Q2] 新的粗搜最优：{val:.3f}s  @ speed={s}, heading={math.degrees(h):.1f}°, drop={d}, fuse={f}")

    # 局部细化（在最优附近小范围加密）
    best_val, (s0, h0, d0, f0, ep0, et0) = best
    print(f"[Q2] 粗搜完成：当前最优 {best_val:.3f}s，进入局部细化...")

    s_list = np.clip(np.linspace(s0-20, s0+20, 5), 70, 140)
    h_list = np.linspace(h0-math.radians(10), h0+math.radians(10), 9)
    d_list = np.clip(np.linspace(d0-2, d0+2, 9), 0, 12)
    f_list = np.clip(np.linspace(f0-1.5, f0+1.5, 7), 0.5, 8.0)

    total2 = len(s_list) * len(h_list) * len(d_list) * len(f_list)
    cnt2 = 0
    for s in s_list:
        for h in h_list:
            for d in d_list:
                for f in f_list:
                    cnt2 += 1
                    if cnt2 % max(1, total2 // 10) == 0:
                        print(f"    [Q2] 细化进度 {int(100*cnt2/total2)}%")
                    dp, ep, et = explosion_from(s, h, d, f)
                    if ep[2] <= 0:
                        continue
                    val = simulate_shield_time(et, ep)
                    if val > best_val:
                        best_val = val
                        s0, h0, d0, f0, ep0, et0 = s, h, d, f, ep, et
                        print(f"      [Q2] 新的细化最优：{best_val:.3f}s  @ speed={s0:.1f}, heading={math.degrees(h0):.1f}°, drop={d0:.2f}, fuse={f0:.2f}")

    print(f"[Q2] 搜索完成：最优遮蔽（严格判据）= {best_val:.3f}s")
    print(f"     速度={s0:.1f} m/s，航向={math.degrees(h0):.2f}°，投弹={d0:.2f}s，引信={f0:.2f}s")
    print(f"     起爆@{ep0}，时刻={et0:.2f}s")
    return best_val, (s0, h0, d0, f0, ep0, et0)

if __name__ == "__main__":
    search_q2()
