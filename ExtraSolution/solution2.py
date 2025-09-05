import numpy as np

# 常量
g = 9.8
missile_speed = 300.0
R = 10.0
cloud_sink = 3.0
cloud_effect = 20.0
z_uav = 1800.0
v_min, v_max = 70.0, 140.0
M0 = np.array([20000.0, 0.0, 2000.0])
D0 = np.array([17800.0, 0.0, 1800.0])
T  = np.array([0.0, 200.0, 0.0])
dt = 0.02

# 导弹直线参数
u_m = -M0 / np.linalg.norm(M0)
v_m = missile_speed * u_m
t_hit = np.linalg.norm(M0) / missile_speed

# 生成一批候选（示例：随机1e5个）
def sample_candidates(n):
    # 航向范围：以 D0->(0,0) 为中心 ±90°
    base_theta = np.arctan2(-D0[1], -D0[0])
    theta = base_theta + (np.random.rand(n) - 0.5) * np.deg2rad(180.0)
    v = v_min + (v_max - v_min) * np.random.rand(n)
    t_burst = 2.0 + 58.0 * np.random.rand(n)        # 2..60
    tau_max = np.minimum(t_burst, np.sqrt(2*z_uav/g))
    tau = tau_max * np.random.rand(n)               # 0..tau_max
    return theta, v, t_burst, tau

import numpy as np

g = 9.8
missile_speed = 300.0
R = 10.0
cloud_sink = 3.0
cloud_effect = 20.0
z_uav = 1800.0
dt = 0.02

M0 = np.array([20000.0, 0.0, 2000.0])
D0 = np.array([17800.0, 0.0, 1800.0])
T  = np.array([0.0, 200.0, 0.0])

u_m = -M0 / np.linalg.norm(M0)
v_m = missile_speed * u_m
t_hit = np.linalg.norm(M0) / missile_speed

def eval_batch(theta, v, t_burst, tau):
    n = theta.shape[0]

    # 约束
    t_release = t_burst - tau
    z_burst = z_uav - 0.5 * g * tau * tau
    feasible = (t_release >= 0.0) & (z_burst >= 0.0)

    # 起爆点
    vx, vy = v*np.cos(theta), v*np.sin(theta)
    PB = np.stack([D0[0] + vx*t_burst,
                   D0[1] + vy*t_burst,
                   z_burst], axis=1)  # (N,3)

    # 各候选自身的有效时间窗
    t_min = np.maximum(t_burst, 0.0)
    t_max = np.minimum(t_burst + cloud_effect, t_hit)

    # 全局统一时间网格（避免 Python 循环）
    global_t0 = np.min(t_min).item()
    global_t1 = np.max(t_max).item()
    if not np.isfinite(global_t0) or global_t1 <= global_t0:
        return np.full(n, -np.inf)

    ts = np.arange(global_t0, global_t1 + 1e-9, dt)  # (T,)

    # 导弹轨迹
    M_t = M0[None, :] + ts[:, None] * v_m[None, :]   # (T,3)
    AB = (T[None, :] - M_t)                          # (T,3)
    AB2 = np.sum(AB*AB, axis=1) + 1e-12             # (T,)

    # 云团中心 C (N,T,3)：xy 固定、z 随时间下沉
    Cxy = PB[:, None, :2]                            # (N,1,2)
    Cxy = np.broadcast_to(Cxy, (n, ts.shape[0], 2))  # (N,T,2)
    dz = cloud_sink * np.clip(ts[None, :] - t_burst[:, None], 0.0, None)  # (N,T)
    Cz = (z_burst[:, None] - dz)[..., None]          # (N,T,1)
    C = np.concatenate([Cxy, Cz], axis=2)            # (N,T,3)

    # 点到线段距离
    A = M_t[None, :, :]                               # (1,T,3)
    AB_ = AB[None, :, :]                              # (1,T,3)
    AB2_ = AB2[None, :]                               # (1,T)
    AP = C - A                                        # (N,T,3)
    u = np.sum(AP*AB_, axis=2) / AB2_                 # (N,T)
    u = np.clip(u, 0.0, 1.0)
    Q = A + AB_ * u[:, :, None]                       # (N,T,3)
    dist = np.linalg.norm(C - Q, axis=2)              # (N,T)

    # 时间窗掩码 + 累计时长
    time_mask = (ts[None, :] >= t_min[:, None]) & (ts[None, :] <= t_max[:, None])
    valid = (dist <= R) & time_mask
    total_time = valid.sum(axis=1) * dt

    # 不可行置 -inf
    total_time = np.where(feasible, total_time, -np.inf)
    return total_time

if __name__ == "__main__":
    np.random.seed(0)
    theta, v, t_burst, tau = sample_candidates(100_000)
    total_time = eval_batch(theta, v, t_burst, tau)
    best = np.argmax(total_time)
    print(f"best occlusion = {total_time[best]:.3f}s")
    print(f"theta={np.degrees(theta[best]):.2f}°, v={v[best]:.2f}, t_burst={t_burst[best]:.2f}, tau={tau[best]:.2f}")