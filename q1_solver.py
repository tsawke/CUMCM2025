# q1_fixed_params.py
# 计算：FY1 以 120 m/s 朝“假目标”方向飞行，t=1.5s 投放，3.6s 后起爆，
# 对 M1 的“有效遮蔽时长”（单位：秒）。

import math, os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ========= 1) 题面/物理参数（如与 PDF 不同请在此修改） =========
g = 9.8
CLOUD_RADIUS = 10.0      # 烟幕球半径 (m)
CLOUD_ACTIVE = 20.0      # 起爆后持续 (s)
CLOUD_SINK = 3.0         # 起爆后竖直匀速下沉速度 (m/s)

MISSILE_SPEED = 300.0    # 导弹速度 (m/s) —— 若 PDF 有明确数值请替换
# 目标（真目标）—— 竖直条，XY 为中心点，高度 z ∈ [TARGET_Z0, TARGET_Z1]
TRUE_TARGET_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0

# “假目标”坐标（**请按 PDF 改**；若不清楚，先用占位值测试）
FAKE_TARGET_XY = (0.0, -200.0)

# 初始位置（若 PDF 有明确数值请替换）
M_INIT = {
    "M1": (20000.0, 0.0, 2000.0),
}
FY_INIT = {
    "FY1": (17800.0, 0.0, 1800.0),
}

# 第一问锁定的动作参数
UAV_ID = "FY1"
UAV_SPEED = 120.0             # 朝“假目标”方向的水平速度 (m/s)
T_DROP = 1.5                  # 受领 1.5 s 后投放
TAU = 3.6                     # 投放后 3.6 s 起爆
TMAX = 60.0                   # 仿真总时长（充分覆盖云有效期即可）
DT = 0.02                     # 时间步（越小越精细；0.02s 足够稳）

# ========= 2) 几何/运动学 =========
def _normalize(v3):
    x,y,z = v3
    n = math.hypot(x, math.hypot(y,z))
    return (x/n, y/n, z/n) if n>0 else (0.0,0.0,0.0)

def missile_pos_M1(t: float):
    """M1 在 t 的位置。假设 M1 直线等速飞向 ‘真目标’ 区域（中心高度取条的中值）。"""
    x0,y0,z0 = M_INIT["M1"]
    xT,yT = TRUE_TARGET_XY
    zTmid = 0.5*(TARGET_Z0 + TARGET_Z1)
    dx,dy,dz = _normalize((xT - x0, yT - y0, zTmid - z0))
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_heading_toward_fake():
    """FY1 朝‘假目标’方向的航向（弧度）。"""
    x0,y0,_ = FY_INIT[UAV_ID]
    xf,yf = FAKE_TARGET_XY
    return math.atan2(yf - y0, xf - x0)

def uav_xy_at(t: float, heading: float):
    """FY1 在 t 时刻的水平位置（按固定航向、固定速度 120 m/s）。"""
    x0,y0,_ = FY_INIT[UAV_ID]
    return (x0 + UAV_SPEED*t*math.cos(heading),
            y0 + UAV_SPEED*t*math.sin(heading))

def cloud_center_at(t: float, heading: float):
    """
    烟幕球心 C(t)：
      t < T_DROP：未投放（返回 None）
      T_DROP ≤ t < T_E：仍未起爆（返回 None，因为球还没形成）
      t ≥ T_E：云心 = 起爆点 - v_sink*(t - T_E) 在 z 方向
    起爆点（t = T_E = T_DROP + TAU）：
      xE,yE：随飞机水平匀速到起爆时刻的位置
      zE   ：投放高度 z0 自由落体 TAU 秒
    """
    if t < T_DROP:
        return None
    T_E = T_DROP + TAU
    # 起爆点
    xE, yE = uav_xy_at(T_E, heading)
    z0 = FY_INIT[UAV_ID][2]
    zE = z0 - 0.5*g*(TAU**2)
    if t < T_E:
        return None  # 尚未起爆，不形成球
    return (xE, yE, zE - CLOUD_SINK*(t - T_E))

def point_to_segment_dist(P, A, B) -> float:
    """三维点 P 到线段 AB 的最小距离。"""
    ax,ay,az = A; bx,by,bz = B; px,py,pz = P
    AB = (bx-ax, by-ay, bz-az)
    AP = (px-ax, py-ay, pz-az)
    ab2 = AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]
    if ab2 == 0.0:
        dx,dy,dz = (px-ax, py-ay, pz-az)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    s = (AP[0]*AB[0] + AP[1]*AB[1] + AP[2]*AB[2]) / ab2
    if s < 0.0: s = 0.0
    elif s > 1.0: s = 1.0
    qx = ax + AB[0]*s; qy = ay + AB[1]*s; qz = az + AB[2]*s
    dx,dy,dz = (px-qx, py-qy, pz-qz)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def covered_at_time(t: float, eps=1e-8) -> bool:
    """
    “完全遮蔽”判定：对目标条 z∈[0,10] 采样（含上下端），
    检查每条线段 M(t)->T(z) 与球是否有交（等价：球心到线段距离 ≤ 半径）。
    """
    C = cloud_center_at(t, heading=_UAV_HEADING)
    if C is None:
        return False
    # 云有效期外直接 False
    T_E = T_DROP + TAU
    if t > T_E + CLOUD_ACTIVE + 1e-9:
        return False

    M = missile_pos_M1(t)
    xT,yT = TRUE_TARGET_XY
    # 取 21 点（含端点），对高度更保险；可加密为 41
    z_samples = np.linspace(TARGET_Z0, TARGET_Z1, num=21)
    for z in z_samples:
        if point_to_segment_dist(C, M, (xT,yT,z)) > CLOUD_RADIUS + eps:
            return False
    return True

# ========= 3) 计算遮蔽时长并导出 =========
def main():
    global _UAV_HEADING
    _UAV_HEADING = uav_heading_toward_fake()

    ts = np.arange(0.0, TMAX + 1e-12, DT)
    cover_mask = np.zeros_like(ts, dtype=bool)
    for i,t in enumerate(ts):
        cover_mask[i] = covered_at_time(t)

    seconds = float(cover_mask.sum()) * DT

    # 打印结果
    print("="*60)
    print("第一问固定参数：FY1@120 m/s，t_drop=1.5 s，tau=3.6 s，仅干扰 M1")
    print(f"真目标中心XY = {TRUE_TARGET_XY}, 假目标中心XY = {FAKE_TARGET_XY}")
    print(f"烟幕球：半径={CLOUD_RADIUS} m，有效期={CLOUD_ACTIVE} s，下沉速度={CLOUD_SINK} m/s")
    print(f"有效遮蔽时长（对 M1）≈ {seconds:.3f} s")
    print("="*60)

    # 导出 result1.xlsx（便于校验）
    # 补充：投放点/起爆点等信息
    xD,yD = uav_xy_at(T_DROP, _UAV_HEADING)
    z0 = FY_INIT[UAV_ID][2]
    T_E = T_DROP + TAU
    xE,yE = uav_xy_at(T_E, _UAV_HEADING)
    zE = z0 - 0.5*g*(TAU**2)

    rows = [{
        "无人机编号": UAV_ID,
        "无人机运动方向(度)": (math.degrees(_UAV_HEADING) % 360.0),
        "无人机运动速度(m/s)": UAV_SPEED,
        "投放时刻t_drop(s)": T_DROP,
        "起爆时刻tE(s)": T_E,
        "投放点x(m)": xD, "投放点y(m)": yD, "投放点z(m)": z0,
        "起爆点x(m)": xE, "起爆点y(m)": yE, "起爆点z(m)": zE,
        "对应导弹": "M1",
        "有效遮蔽时长(s)": seconds
    }]
    df = pd.DataFrame(rows)
    out = "result1.xlsx"
    try:
        df.to_excel(out, index=False)
        written = out
    except Exception:
        csv = "result1.csv"
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        written = csv

    # 同时给一个 JSON 摘要
    summary = {
        "cover_seconds": seconds,
        "params": {
            "UAV": UAV_ID, "UAV_speed": UAV_SPEED,
            "T_drop": T_DROP, "tau": TAU,
            "true_target_xy": TRUE_TARGET_XY,
            "fake_target_xy": FAKE_TARGET_XY,
            "cloud": {"radius": CLOUD_RADIUS, "active": CLOUD_ACTIVE, "sink": CLOUD_SINK},
            "missile_speed": MISSILE_SPEED, "dt": DT
        }
    }
    Path("q1_answer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"导出：{written} 与 q1_answer_summary.json")

if __name__ == "__main__":
    main()
