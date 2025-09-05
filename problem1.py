# -*- coding: utf-8 -*-
"""
问题一：给定 FY1 的速度/航向/时序，对 M1 的“有效遮蔽时长”计算脚本（完整、可运行、逐步注释）

使用方法：
1) 直接运行本脚本：python problem1.py
2) 在末尾的“参数区”可修改时间步长/采样数做更精细评估。

建模要点（与题面一致）：
- 导弹恒速直线朝向假目标（原点）
- 无人机等高直线匀速飞行，投放后干扰弹仅受重力，起爆位置的“水平坐标”等于无人机若继续直飞到起爆时刻的坐标，起爆高度 z = z0 - 1/2 g (t_e - t_d)^2
- 云团为半径 10 m 的球，起爆后存在 20 s，并以 3 m/s 向下沉降
- 有效遮蔽判定：导弹位置与真目标圆柱之间的任一视线段，与球形云团相交（采用“线段到球心最近距离 ≤ 10 m”的保守几何判定）
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List

# =====================
# 一、常量与场景参数
# =====================

g = 9.8                               # 重力加速度 (m/s^2)
CLOUD_RADIUS = 10.0                   # 云团有效半径 (m)
CLOUD_ACTIVE = 20.0                   # 云团存续时间 (s)
CLOUD_SINK = 3.0                      # 云团向下沉降速度 (m/s)
MISSILE_SPEED = 300.0                 # 导弹速度 (m/s)

# 真实目标为圆柱：半径 7 m，高 10 m，下底圆心在 (0, 200, 0)
TARGET_CENTER_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0

# 三枚导弹初始状态（题面给定），本题只用到 M1
M_INIT = {
    "M1": (20000.0, 0.0, 2000.0),
    "M2": (19000.0, 600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}

# 五架无人机初始状态（题面给定），本题只用到 FY1
FY_INIT = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}

# =====================
# 二、几何与物理辅助函数
# =====================

def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """向量单位化。若零向量则返回 (0,0,0)。"""
    x, y, z = v
    n = math.hypot(x, math.hypot(y, z))
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def missile_pos(missile_id: str, t: float) -> Tuple[float, float, float]:
    """
    导弹在时刻 t 的位置：从初始点沿“指向原点”的单位方向向量，以恒速 300 m/s 直线运动。
    """
    x0, y0, z0 = M_INIT[missile_id]
    dx, dy, dz = normalize((-x0, -y0, -z0))
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_xy_pos(uav_id: str, v: float, heading_rad: float, t: float) -> Tuple[float, float]:
    """
    无人机等高直飞的水平位置（x,y）。
    v：水平速度；heading_rad：在 xy 平面的航向角（相对 x 轴正向，逆时针为正）。
    """
    x0, y0, _ = FY_INIT[uav_id]
    return (x0 + v * t * math.cos(heading_rad),
            y0 + v * t * math.sin(heading_rad))

def explosion_point(uav_id: str, v: float, heading_rad: float, t_drop: float, t_explode: float) -> Tuple[float, float, float]:
    """
    计算干扰弹“起爆点”坐标。
    - 水平 (x,y) 取无人机若继续直飞到 t_explode 的位置；
    - 高度 z = z0 - 1/2 g (t_explode - t_drop)^2（投放时垂直初速视为 0）。
    """
    xe, ye = uav_xy_pos(uav_id, v, heading_rad, t_explode)
    z0 = FY_INIT[uav_id][2]
    tau = max(0.0, t_explode - t_drop)
    ze = z0 - 0.5 * g * tau * tau
    return (xe, ye, ze)

def cloud_center_at(cE: Tuple[float, float, float], t_explode: float, t: float) -> Tuple[float, float, float]:
    """
    起爆后的云团中心位置：仅沿 z 轴以 3 m/s 下沉，水平不变。
    """
    return (cE[0], cE[1], cE[2] - CLOUD_SINK * (t - t_explode))

def point_seg_dist(p: Tuple[float, float, float],
                   a: Tuple[float, float, float],
                   b: Tuple[float, float, float]) -> float:
    """
    点 p 到线段 ab 的最短距离（3D）。用于判断“球心到视线段”的距离是否小于等于云团半径。
    """
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    ab = (bx - ax, by - ay, bz - az)
    ap = (px - ax, py - ay, pz - az)
    ab2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    if ab2 == 0.0:  # 退化为点
        dx, dy, dz = (px - ax, py - ay, pz - az)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab2
    t = max(0.0, min(1.0, t))
    q = (ax + ab[0]*t, ay + ab[1]*t, az + ab[2]*t)  # 最近点
    dx, dy, dz = (px - q[0], py - q[1], pz - q[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def covered_at_time(c_center: Tuple[float, float, float],
                    missile_id: str,
                    t: float,
                    z_samples: int = 5) -> bool:
    """
    在时刻 t 判断是否“有效遮蔽”：
    - 将目标圆柱的轴线 (0,200,z), z∈[0,10] 做 z_samples 等分采样；
    - 对每个采样点形成线段 [导弹位置, 目标轴线点]；
    - 若“球心到线段”的最近距离 ≤ CLOUD_RADIUS，则判为有效。
    """
    m = missile_pos(missile_id, t)
    for k in range(z_samples):
        z = TARGET_Z0 + (TARGET_Z1 - TARGET_Z0) * (k / (z_samples - 1) if z_samples > 1 else 0.5)
        tgt = (TARGET_CENTER_XY[0], TARGET_CENTER_XY[1], z)
        if point_seg_dist(c_center, m, tgt) <= CLOUD_RADIUS:
            return True
    return False

def coverage_time_for_plan(uav_id: str,
                           missile_id: str,
                           v: float,
                           heading_rad: float,
                           t_drop: float,
                           t_explode: float,
                           t_start: float = 0.0,
                           t_end: float = 100.0,
                           dt: float = 0.05,
                           z_samples: int = 5) -> float:
    """
    累计“有效遮蔽时长”（秒）：
    - 从 max(t_start, t_explode) 到 min(t_end, t_explode+20) 逐步累计；
    - 每一步构造云团中心并调用 covered_at_time 判定；
    - 返回 ∑有效步数×dt。
    """
    cE = explosion_point(uav_id, v, heading_rad, t_drop, t_explode)
    t0 = max(t_start, t_explode)
    t1 = min(t_end, t_explode + CLOUD_ACTIVE)
    if t0 >= t1:
        return 0.0
    t = t0
    covered = 0.0
    while t <= t1 + 1e-12:
        c = cloud_center_at(cE, t_explode, t)
        if covered_at_time(c, missile_id, t, z_samples=z_samples):
            covered += dt
        t += dt
    return covered

# =====================
# 三、问题一：参数设置与计算
# =====================

def solve_problem1() -> float:
    # ---- 题面给定：FY1 以 120 m/s 朝向“假目标（原点）”飞行 ----
    uav_id = "FY1"
    v = 120.0
    x0, y0, _ = FY_INIT[uav_id]
    heading = math.atan2(-y0, -x0)  # 指向原点的航向角

    # ---- 投放/起爆时序（题面给定）：t_drop=1.5 s；3.6 s 后起爆 -> t_explode=5.1 s ----
    t_drop = 1.5
    t_explode = t_drop + 3.6

    # ---- 计算有效遮蔽时长（对 M1）----
    seconds = coverage_time_for_plan(
        uav_id=uav_id,
        missile_id="M1",
        v=v,
        heading_rad=heading,
        t_drop=t_drop,
        t_explode=t_explode,
        t_start=0.0,
        t_end=100.0,
        dt=0.05,
        z_samples=5
    )
    return seconds

if __name__ == "__main__":
    secs = solve_problem1()
    print(f"[问题一] FY1 单次投放对 M1 的有效遮蔽时长 ≈ {secs:.2f} s")
    # 需要更高精度：可把 dt 降到 0.02、z_samples 提升到 11；注意运行时间会略增。
