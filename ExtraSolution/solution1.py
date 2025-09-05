import numpy as np

# 常量
g = 9.8
missile_speed = 300.0
uav_speed = 120.0
t_release = 1.5
fuse_delay = 3.6
t_burst = t_release + fuse_delay
cloud_sink = 3.0
R = 10.0
cloud_effect = 20.0
dt = 0.01

# 位置
M0 = np.array([20000.0, 0.0, 2000.0])
D0 = np.array([17800.0, 0.0, 1800.0])
T  = np.array([0.0, 200.0, 0.0])

# 导弹直线参数
u_m = -M0 / np.linalg.norm(M0)
v_m = missile_speed * u_m
t_hit = np.linalg.norm(M0) / missile_speed

# 起爆点（水平随无人机，竖直自由落体）
PB = np.array([
    D0[0] - uav_speed * t_burst,
    0.0,
    D0[2] - 0.5 * g * (fuse_delay ** 2)
], dtype=float)

def cloud_center(t):
    # t >= t_burst
    return np.array([PB[0], PB[1], PB[2] - cloud_sink * (t - t_burst)], dtype=float)

def missile_pos(t):
    return M0 + v_m * t

def dist_point_to_segment(P, A, B):
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0:
        return np.linalg.norm(P - A)
    u = np.dot(AP, AB) / ab2
    u = max(0.0, min(1.0, u))
    Q = A + u * AB
    return np.linalg.norm(P - Q)

def d_to_sightline(t):
    C = cloud_center(t)
    M = missile_pos(t)
    return dist_point_to_segment(C, M, T)

def refine_crossing(t_lo, t_hi, target=R, tol=1e-7, iters=80):
    f = lambda x: d_to_sightline(x) - target
    flo, fhi = f(t_lo), f(t_hi)
    if flo == 0.0: return t_lo
    if fhi == 0.0: return t_hi
    if flo * fhi > 0:
        return None
    for _ in range(iters):
        tm = 0.5 * (t_lo + t_hi)
        fm = f(tm)
        if abs(fm) < tol or (t_hi - t_lo) < tol:
            return tm
        if flo * fm <= 0:
            t_hi, fhi = tm, fm
        else:
            t_lo, flo = tm, fm
    return 0.5 * (t_lo + t_hi)

def compute_intervals(dt=0.01):
    t0 = t_burst
    t1 = min(t_burst + cloud_effect, t_hit)
    ts = np.arange(t0, t1 + 1e-12, dt)

    # 向量化距离
    Cx = PB[0] * np.ones_like(ts)
    Cy = PB[1] * np.ones_like(ts)
    Cz = PB[2] - cloud_sink * (ts - t_burst)
    C = np.stack([Cx, Cy, Cz], axis=1)  # (T,3)

    M = M0[None, :] + ts[:, None] * v_m[None, :]  # (T,3)
    AB = (T[None, :] - M)                         # (T,3)
    AB2 = np.sum(AB * AB, axis=1) + 1e-12
    AP = C - M
    u = np.sum(AP * AB, axis=1) / AB2
    u = np.clip(u, 0.0, 1.0)
    Q = M + u[:, None] * AB
    dist = np.linalg.norm(C - Q, axis=1)

    inside = dist <= R
    edges = np.diff(inside.astype(np.int8))
    enters = np.where(edges == 1)[0] + 1  # 进入索引
    exits  = np.where(edges == -1)[0] + 1 # 退出索引

    intervals = []

    # 开头就在内
    if inside[0]:
        # 精化从 ts[0]-dt 到 ts[0]（近似），若无法精化，取 ts[0]
        t_enter = ts[0]
        if len(exits) > 0:
            i = exits[0]
            t_exit = refine_crossing(ts[i-1], ts[i]) or ts[i]
            intervals.append((t_enter, t_exit))
            exits = exits[1:]
        else:
            intervals.append((ts[0], ts[-1]))

    # 正常配对
    for i_ent in enters:
        # 进入边界精化
        t_enter = refine_crossing(ts[i_ent-1], ts[i_ent]) or ts[i_ent]
        # 找对应退出
        next_exits = exits[exits > i_ent]
        if len(next_exits) == 0:
            # 未退出，直到末尾
            t_exit = ts[-1]
        else:
            i_ex = next_exits[0]
            t_exit = refine_crossing(ts[i_ex-1], ts[i_ex]) or ts[i_ex]
            exits = exits[exits != i_ex]
        intervals.append((t_enter, t_exit))

    # 如果还残留退出（理论上不会），忽略

    total = sum(max(0.0, b - a) for a, b in intervals)
    return intervals, total

if __name__ == "__main__":
    print(f"起爆时刻 t_burst = {t_burst:.3f} s")
    print(f"起爆点 PB = ({PB[0]:.3f}, {PB[1]:.3f}, {PB[2]:.3f})")

    intervals, total = compute_intervals(dt=dt)
    print("遮蔽时间区间：")
    for a, b in intervals:
        print(f"  [{a:.3f}, {b:.3f}]  时长 = {b - a:.3f} s")
    print(f"总有效遮蔽时长 = {total:.3f} s")