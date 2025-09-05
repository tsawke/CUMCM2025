# -*- coding: utf-8 -*-
"""
2025年全国大学生数学建模竞赛A题最终完整解决方案（修正并行版）
无人机烟幕干扰优化问题

改动要点：
1) 遮蔽判据：采用“圆柱整体 ∈ 导弹视域无限圆锥”严格判定（取样+向量化）。
2) 统一导弹飞向假目标原点 (0,0,0)。
3) 并行加速：使用 ProcessPoolExecutor 跑满 CPU，多处任务并行评估。
4) 搜索策略：两段式/坐标搜索+贪心，避免16^6等组合爆炸。
5) 详细日志：阶段、进度、最优值、写盘等。

作者：AI助手（修订）
日期：2025年
"""

import math
import json
import pandas as pd
import numpy as np
import time
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================
# 一、全局参数（可按需要调优）
# =====================

# 物理常量
g = 9.8
CLOUD_RADIUS = 10.0      # 云团半径 r
CLOUD_ACTIVE = 20.0      # 有效时长
CLOUD_SINK = 3.0         # 下沉速度
MISSILE_SPEED = 300.0    # 导弹速度 300 m/s
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0])  # 假目标原点

# 真目标（圆柱）参数（底面圆心(0,200,0)，半径7m，高10m，轴向+z）
CYL_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYL_R = 7.0
CYL_H = 10.0

# 采样/步长（精度-速度权衡）
DT = 0.05           # 时间步长（可改 0.02 提高精度）
NPHI = 120          # 圆周角度采样数（可改 180 提高精度）
NZ = 5              # 侧面采样圈数（可改 7 提高精度）

# 并行控制
N_WORKERS = None    # None = 自动使用CPU全部可用核心；也可指定整数

# 初始位置（保持与题面一致）
M_INIT = {
    "M1": np.array([20000.0,     0.0, 2000.0], dtype=float),
    "M2": np.array([19000.0,   600.0, 2100.0], dtype=float),
    "M3": np.array([18000.0,  -600.0, 1900.0], dtype=float),
}

FY_INIT = {
    "FY1": np.array([17800.0,    0.0, 1800.0], dtype=float),
    "FY2": np.array([12000.0, 1400.0, 1400.0], dtype=float),
    "FY3": np.array([ 6000.0,-3000.0,  700.0], dtype=float),
    "FY4": np.array([11000.0, 2000.0, 1800.0], dtype=float),
    "FY5": np.array([13000.0,-2000.0, 1300.0], dtype=float),
}

# =====================
# 二、基础工具
# =====================

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def missile_pos(missile_id: str, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """导弹位置与速度向量（恒速直线朝原点）"""
    M0 = M_INIT[missile_id]
    dir_to_origin = unit(FAKE_TARGET_ORIGIN - M0)
    v = MISSILE_SPEED * dir_to_origin
    return (M0 + v * t, v)

def uav_xy_pos(uav_id: str, v: float, heading_rad: float, t: float) -> np.ndarray:
    """无人机在时刻t的水平位置 (x,y)，z保持初始高度"""
    p0 = FY_INIT[uav_id]
    x = p0[0] + v * t * math.cos(heading_rad)
    y = p0[1] + v * t * math.sin(heading_rad)
    return np.array([x, y], dtype=float)

def explosion_point(uav_id: str, v: float, heading_rad: float,
                    t_drop: float, t_explode: float) -> np.ndarray:
    """计算起爆点坐标 (x,y,z)"""
    xy = uav_xy_pos(uav_id, v, heading_rad, t_explode)
    z0 = FY_INIT[uav_id][2]
    tau = max(0.0, t_explode - t_drop)
    z = z0 - 0.5 * g * tau * tau
    return np.array([xy[0], xy[1], z], dtype=float)

def cloud_center_at(cE: np.ndarray, t_explode: float, t: float) -> np.ndarray:
    """时刻 t 云团中心位置"""
    return np.array([cE[0], cE[1], cE[2] - CLOUD_SINK * (t - t_explode)], dtype=float)

def missile_hit_time(missile_id: str) -> float:
    """导弹命中假目标原点的时间（恒速直线）"""
    return np.linalg.norm(M_INIT[missile_id] - FAKE_TARGET_ORIGIN) / MISSILE_SPEED

# =====================
# 三、严格遮蔽判据：圆柱整体 ∈ 无限圆锥
# =====================

# 预生成圆柱表面采样点（世界坐标，静态）
def precompute_cylinder_points(n_phi: int = NPHI, n_z_side: int = NZ) -> np.ndarray:
    B = CYL_BASE_CENTER
    H = CYL_H
    R = CYL_R
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pts = []

    # 底/顶圆周
    for phi in phis:
        pts.append(B + np.array([R*math.cos(phi), R*math.sin(phi), 0.0]))
        pts.append(B + np.array([0.0, 0.0, H]) + np.array([R*math.cos(phi), R*math.sin(phi), 0.0]))
    # 侧面若干圈
    if n_z_side >= 2:
        for k in range(n_z_side):
            z = H * (k/(n_z_side-1.0))
            center = B + np.array([0.0, 0.0, z])
            for phi in phis:
                pts.append(center + np.array([R*math.cos(phi), R*math.sin(phi), 0.0]))
    return np.vstack(pts).astype(float)

CYL_PTS = precompute_cylinder_points()

def cylinder_inside_infinite_cone_vec(M: np.ndarray, C: np.ndarray,
                                      r_cloud: float = CLOUD_RADIUS,
                                      pts: np.ndarray = CYL_PTS) -> bool:
    """
    向量化判定：圆柱采样点是否全部落入以 M 为锥顶、轴向 (C-M)、半顶角 asin(r/||C-M||) 的无限圆锥。
    """
    v = C - M
    L = np.linalg.norm(v)
    if L <= 1e-9 or r_cloud >= L:
        return True
    cos_alpha = math.sqrt(max(0.0, 1.0 - (r_cloud/L)**2))

    # 向量化：对所有点同时计算 cosθ
    W = pts - M  # [N,3]
    Wn = np.linalg.norm(W, axis=1) + 1e-12
    cos_theta = (W @ v) / (Wn * L)
    return np.all(cos_theta >= (cos_alpha - 1e-12))

def coverage_mask_for_plan(uav_id: str, missile_id: str, v: float, heading_rad: float,
                           t_drop: float, t_explode: float,
                           t_grid: np.ndarray) -> np.ndarray:
    """
    生成给定方案在全局时间网格上的遮蔽布尔掩码（只在有效时段检测）
    """
    cE = explosion_point(uav_id, v, heading_rad, t_drop, t_explode)
    if cE[2] <= 0:  # 起爆点落地，不生效
        return np.zeros_like(t_grid, dtype=bool)

    hit = missile_hit_time(missile_id)
    t0 = max(t_explode, t_grid[0])
    t1 = min(t_explode + CLOUD_ACTIVE, hit, t_grid[-1])

    if t0 >= t1:
        return np.zeros_like(t_grid, dtype=bool)

    mask = np.zeros_like(t_grid, dtype=bool)
    # 仅在有效索引范围内判定
    i0 = int(math.floor((t0 - t_grid[0]) / DT))
    i1 = int(math.ceil((t1 - t_grid[0]) / DT))
    for i in range(max(0, i0), min(len(t_grid), i1+1)):
        t = t_grid[i]
        M, _ = missile_pos(missile_id, t)
        C = cloud_center_at(cE, t_explode, t)
        mask[i] = cylinder_inside_infinite_cone_vec(M, C, CLOUD_RADIUS, CYL_PTS)
    return mask

def coverage_time_from_mask(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) * DT)

# =====================
# 四、问题一：FY1 单次投放对 M1 遮蔽时长
# =====================

def solve_problem1():
    print("="*72)
    print("问题一：FY1 单次投放对 M1 的有效遮蔽时长（严格圆锥判据 + 向量化）")
    print("="*72)

    uav_id = "FY1"
    missile_id = "M1"
    v = 120.0
    p0 = FY_INIT[uav_id]
    heading = math.atan2(-p0[1], -p0[0])  # 朝原点
    t_drop = 1.5
    fuse = 3.6
    t_explode = t_drop + fuse

    horizon = min(missile_hit_time(missile_id) + 1.0, 100.0)
    t_grid = np.arange(0.0, horizon + 1e-9, DT)

    print(f"[Q1] UAV={uav_id}, Missile={missile_id}, v={v} m/s, heading={math.degrees(heading):.2f}°")
    print(f"[Q1] t_drop={t_drop:.2f}s, fuse={fuse:.2f}s, t_explode={t_explode:.2f}s, dt={DT}, |grid|={len(t_grid)}")
    print("[Q1] 计算起爆点/逐时刻判定...")

    mask = coverage_mask_for_plan(uav_id, missile_id, v, heading, t_drop, t_explode, t_grid)
    seconds = coverage_time_from_mask(mask)
    cE = explosion_point(uav_id, v, heading, t_drop, t_explode)
    drop_xy = uav_xy_pos(uav_id, v, heading, t_drop)
    drop_z = FY_INIT[uav_id][2]

    print(f"[Q1] 遮蔽时长 = {seconds:.3f} s")
    print(f"[Q1] 投放点=({drop_xy[0]:.2f},{drop_xy[1]:.2f},{drop_z:.2f})  起爆点=({cE[0]:.2f},{cE[1]:.2f},{cE[2]:.2f})")

    result_data = {
        "无人机运动方向": float(math.degrees(heading) % 360),
        "无人机运动速度 (m/s)": float(v),
        "烟幕干扰弹编号": 1,
        "烟幕干扰弹投放点的x坐标 (m)": float(drop_xy[0]),
        "烟幕干扰弹投放点的y坐标 (m)": float(drop_xy[1]),
        "烟幕干扰弹投放点的z坐标 (m)": float(drop_z),
        "烟幕干扰弹起爆点的x坐标 (m)": float(cE[0]),
        "烟幕干扰弹起爆点的y坐标 (m)": float(cE[1]),
        "烟幕干扰弹起爆点的z坐标 (m)": float(cE[2]),
        "有效干扰时长 (s)": float(seconds)
    }
    pd.DataFrame([result_data]).to_csv("result1_final.csv", index=False, encoding="utf-8-sig")
    print("[Q1] 结果已保存到 result1_final.csv")
    return seconds, result_data

# =====================
# 五、问题二：多无人机协同干扰优化（FY1/2/3→M1）
# =====================

@dataclass
class UAVPlan:
    uav_id: str
    heading: float   # 弧度
    speed: float     # m/s
    t_drop: float    # s
    t_explode: float # s
    mask: np.ndarray # 布尔掩码（与全局t_grid对齐）
    seconds: float

def _eval_single_plan(args: Tuple[str, float, float, float, float, str, np.ndarray]) -> Tuple[Tuple[float,float,float,float], float, np.ndarray]:
    """
    供并行池调用：返回(参数元组, 覆盖时长, 掩码)
    args = (uav_id, speed, heading, t_drop, fuse, missile_id, t_grid)
    """
    uav_id, speed, heading, t_drop, fuse, missile_id, t_grid = args
    t_explode = t_drop + fuse
    mask = coverage_mask_for_plan(uav_id, missile_id, speed, heading, t_drop, t_explode, t_grid)
    return (speed, heading, t_drop, fuse), coverage_time_from_mask(mask), mask

def solve_problem2():
    """
    多无人机协同干扰优化（FY1、FY2、FY3 对 M1），并行坐标搜索 + 局部细化
    - 变量：每架的 heading（围绕朝原点±20°）、t_drop（0.5~8s，步0.5），fuse 固定 3.6s（可改网格）
    - 策略：对每架 UAV 在自身网格上并行搜索“单机最优单弹”；然后做两轮坐标上调（轮流替换为其单机Top-K里能让并集最大的组合）
    """
    print("\n" + "="*72)
    print("问题二：多无人机协同干扰优化（FY1/2/3→M1）——并行坐标搜索")
    print("="*72)

    missile_id = "M1"
    uavs = ["FY1", "FY2", "FY3"]
    speed = 120.0
    fuse = 3.6

    # 时间轴（统一）
    horizon = min(missile_hit_time(missile_id) + 1.0, 100.0)
    t_grid = np.arange(0.0, horizon + 1e-9, DT)
    print(f"[Q2] 时间网格：0~{horizon:.2f}s, dt={DT}, |grid|={len(t_grid)}")

    # 各 UAV 的搜索网格
    def heading_center(uav_id): 
        p = FY_INIT[uav_id]
        return math.atan2(-p[1], -p[0])
    HEADING_DELTA_DEG = list(range(-20, 21, 4))   # ±20°，步4°
    DROP_LIST = np.arange(0.5, 8.0 + 1e-9, 0.5)   # 0.5~8.0，步0.5
    TOPK = 12                                     # 每机保留Top-K方案用于协调

    # 1) 各机并行评估自家网格（heading × drop）
    per_uav_top = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures: Dict[Any, str] = {}
        for uav in uavs:
            h0 = heading_center(uav)
            headings = [h0 + math.radians(d) for d in HEADING_DELTA_DEG]
            tasks = []
            for hd in headings:
                for td in DROP_LIST:
                    tasks.append((uav, speed, hd, td, fuse, missile_id, t_grid))
            print(f"[Q2] {uav}: 提交并行评估 {len(tasks)} 个候选")
            fut_list = [pool.submit(_eval_single_plan, x) for x in tasks]
            # 收集并排序
            results = []
            done_cnt = 0
            for fut in as_completed(fut_list):
                params, secs, mask = fut.result()
                results.append((params, secs, mask))
                done_cnt += 1
                if done_cnt % max(1, len(fut_list)//10) == 0:
                    print(f"    [Q2] {uav}: 进度 {int(100*done_cnt/len(fut_list))}%")
            results.sort(key=lambda z: -z[1])
            per_uav_top[uav] = results[:TOPK]
            print(f"[Q2] {uav}: 单机Top-{TOPK} 最佳单弹(秒) = {[round(z[1],3) for z in per_uav_top[uav][:5]]} ...")

    # 2) 两轮“坐标提升”：依次替换 FY1/FY2/FY3 的方案为其Top-K中能让“并集”最大的选择
    def mask_seconds(msk: np.ndarray) -> float: return float(np.count_nonzero(msk) * DT)

    # 初始：各机取自己的Top-1
    current = {}
    union_mask = np.zeros_like(t_grid, dtype=bool)
    for uav in uavs:
        (s,h,td,fz), secs, msk = per_uav_top[uav][0]
        current[uav] = UAVPlan(uav, h, s, td, td+fz, msk, secs)
        union_mask |= msk
    print(f"[Q2] 初始并集 = {mask_seconds(union_mask):.3f}s")

    for round_id in range(2):  # 两轮
        print(f"[Q2] 坐标提升 Round {round_id+1} ...")
        for uav in uavs:
            base_mask = np.zeros_like(t_grid, dtype=bool)
            for other in uavs:
                if other == uav: continue
                base_mask |= current[other].mask
            best_gain = -1.0
            best_choice = None
            for (s,h,td,fz), secs, msk in per_uav_top[uav]:
                new_union = base_mask | msk
                gain = mask_seconds(new_union)
                if gain > best_gain:
                    best_gain = gain
                    best_choice = UAVPlan(uav, h, s, td, td+fz, msk, secs)
            current[uav] = best_choice
            union_mask = np.zeros_like(t_grid, dtype=bool)
            for k in uavs: union_mask |= current[k].mask
            print(f"    [Q2] 换 {uav}: 并集 -> {mask_seconds(union_mask):.3f}s （其单弹{best_choice.seconds:.3f}s）")

    total_union_seconds = mask_seconds(union_mask)
    print(f"[Q2] 最终并集遮蔽（严格判据）= {total_union_seconds:.3f} s")

    # 输出 CSV
    rows = []
    for uav in uavs:
        plan = current[uav]
        drop_xy = uav_xy_pos(plan.uav_id, plan.speed, plan.heading, plan.t_drop)
        drop_z = FY_INIT[plan.uav_id][2]
        cE = explosion_point(plan.uav_id, plan.speed, plan.heading, plan.t_drop, plan.t_explode)
        rows.append({
            "无人机编号": plan.uav_id,
            "无人机运动方向": float(math.degrees(plan.heading) % 360),
            "无人机运动速度 (m/s)": float(plan.speed),
            "烟幕干扰弹投放点的x坐标 (m)": float(drop_xy[0]),
            "烟幕干扰弹投放点的y坐标 (m)": float(drop_xy[1]),
            "烟幕干扰弹投放点的z坐标 (m)": float(drop_z),
            "烟幕干扰弹起爆点的x坐标 (m)": float(cE[0]),
            "烟幕干扰弹起爆点的y坐标 (m)": float(cE[1]),
            "烟幕干扰弹起爆点的z坐标 (m)": float(cE[2]),
            "有效干扰时长 (s)": float(plan.seconds)
        })
        print(f"  [Q2] {uav}: heading={math.degrees(plan.heading):.2f}°, drop={plan.t_drop:.2f}s, 单弹={plan.seconds:.3f}s")
    pd.DataFrame(rows).to_csv("result2_final.csv", index=False, encoding="utf-8-sig")
    print("[Q2] 结果已保存到 result2_final.csv")

    return rows

# =====================
# 六、问题三：多导弹同时干扰（FY1..FY5 → M1/M2/M3）
# =====================

def _eval_single_uav_multi_missile(args: Tuple[str, float, float, float, List[str], np.ndarray]) -> Dict[str, Any]:
    """
    并行评估：给定 (uav, speed, heading, t_drop, missiles, t_grid)
    返回该 UAV 对每枚导弹的遮蔽时长与最佳导弹选择
    """
    uav, speed, heading, t_drop, missiles, t_grid = args
    fuse = 3.6
    t_explode = t_drop + fuse
    best = (-1.0, None, None)  # (seconds, missile_id, mask)
    per_missile = {}
    for mis in missiles:
        mask = coverage_mask_for_plan(uav, mis, speed, heading, t_drop, t_explode, t_grid)
        secs = coverage_time_from_mask(mask)
        per_missile[mis] = float(secs)
        if secs > best[0]:
            best = (secs, mis, mask)
    return {
        "uav": uav,
        "t_drop": float(t_drop),
        "heading": float(heading),
        "speed": float(speed),
        "best_seconds": float(best[0]),
        "best_missile": best[1],
        "per_missile": per_missile
    }

def solve_problem3():
    """
    多导弹同时干扰：FY1..FY5 每架择一枚导弹（M1/M2/M3）+ 一个投放时机（heading固定朝原点，speed=120, fuse=3.6）
    策略：对每架 UAV 在 t_drop 网格上并行评估其对 3 枚导弹的遮蔽表现，选“该机-时机-导弹”的最佳。
    （如需更强：可在此基础上加约束或联合微调。）
    """
    print("\n" + "="*72)
    print("问题三：多导弹同时干扰（FY1..FY5 → M1/M2/M3）——并行评估各机最佳对象与时机")
    print("="*72)

    uavs = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    missiles = ["M1", "M2", "M3"]
    speed = 120.0

    # 取全体导弹中最早命中时间，做统一时间轴
    horizon = min([missile_hit_time(m) for m in missiles]) + 1.0
    t_grid = np.arange(0.0, horizon + 1e-9, DT)
    print(f"[Q3] t_grid: 0~{horizon:.2f}s, dt={DT}, |grid|={len(t_grid)}")

    # 每机的 heading 固定为“朝原点”，t_drop 网格
    TDROP = np.arange(0.5, 12.0 + 1e-9, 0.5)
    tasks = []
    for u in uavs:
        heading = math.atan2(-FY_INIT[u][1], -FY_INIT[u][0])
        for td in TDROP:
            tasks.append((u, speed, heading, td, missiles, t_grid))
    print(f"[Q3] 提交并行任务数 = {len(tasks)}")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(_eval_single_uav_multi_missile, x) for x in tasks]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % max(1, len(futs)//10) == 0:
                print(f"    [Q3] 进度 {int(100*done/len(futs))}%")

    # 为每架 UAV 选择其“最佳时机/导弹”
    best_by_uav = {}
    for u in uavs:
        cand = [r for r in results if r["uav"] == u]
        cand.sort(key=lambda z: -z["best_seconds"])
        best_by_uav[u] = cand[0]
        print(f"[Q3] {u}: 最佳导弹={cand[0]['best_missile']}  遮蔽={cand[0]['best_seconds']:.3f}s  t_drop={cand[0]['t_drop']:.2f}s")

    # 写 CSV
    rows = []
    for u in uavs:
        rec = best_by_uav[u]
        heading = rec["heading"]
        t_drop = rec["t_drop"]
        fuse = 3.6
        t_explode = t_drop + fuse
        drop_xy = uav_xy_pos(u, speed, heading, t_drop)
        drop_z = FY_INIT[u][2]
        cE = explosion_point(u, speed, heading, t_drop, t_explode)

        rows.append({
            "无人机编号": u,
            "无人机运动方向": float(math.degrees(heading) % 360),
            "无人机运动速度 (m/s)": float(speed),
            "烟幕干扰弹编号": u[-1],  # 简单标注
            "烟幕干扰弹投放点的x坐标 (m)": float(drop_xy[0]),
            "烟幕干扰弹投放点的y坐标 (m)": float(drop_xy[1]),
            "烟幕干扰弹投放点的z坐标 (m)": float(drop_z),
            "烟幕干扰弹起爆点的x坐标 (m)": float(cE[0]),
            "烟幕干扰弹起爆点的y坐标 (m)": float(cE[1]),
            "烟幕干扰弹起爆点的z坐标 (m)": float(cE[2]),
            "有效干扰时长 (s)": float(rec["best_seconds"]),
            "干扰的导弹编号": rec["best_missile"]
        })
    pd.DataFrame(rows).to_csv("result3_final.csv", index=False, encoding="utf-8-sig")
    print("[Q3] 结果已保存到 result3_final.csv")
    return rows

# =====================
# 七、结果分析与汇总
# =====================

def analyze_results():
    print("\n" + "="*72)
    print("结果分析")
    print("="*72)
    try:
        df1 = pd.read_csv("result1_final.csv")
        df2 = pd.read_csv("result2_final.csv")
        df3 = pd.read_csv("result3_final.csv")

        print("问题一：")
        print(f"  有效遮蔽时长：{df1['有效干扰时长 (s)'].iloc[0]:.3f} 秒")

        print("\n问题二：")
        total_coverage_2 = float(df2['有效干扰时长 (s)'].sum())
        print(f"  单弹总和（提示值）：{total_coverage_2:.3f} 秒")
        for _, row in df2.iterrows():
            print(f"  {row['无人机编号']}: {row['有效干扰时长 (s)']:.3f} 秒")

        print("\n问题三：")
        total_coverage_3 = float(df3['有效干扰时长 (s)'].sum())
        print(f"  单弹总和（提示值）：{total_coverage_3:.3f} 秒")
        for _, row in df3.iterrows():
            print(f"  {row['无人机编号']}: {row['有效干扰时长 (s)']:.3f} 秒, 主要干扰 {row['干扰的导弹编号']}")

        analysis_report = {
            "问题一": {
                "描述": "FY1单次投放对M1的有效遮蔽时长（严格圆锥判据）",
                "结果": f"{df1['有效干扰时长 (s)'].iloc[0]:.3f}秒",
                "备注": "可调 dt/NPHI/NZ 提升精度"
            },
            "问题二": {
                "描述": "FY1/2/3 协同对 M1 的并集优化（并行坐标搜索）",
                "提示": "CSV中为单弹各自时长；真正的并集时长见日志输出",
                "并集时长见日志": True
            },
            "问题三": {
                "描述": "FY1..FY5 对 M1/M2/M3 的并行评估与分配",
                "提示": "CSV为每机选择的最佳导弹与时机及其单弹时长"
            }
        }
        with open('analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)
        print("\n分析报告已保存到 analysis_report.json")
    except FileNotFoundError as e:
        print(f"结果文件未找到：{e}")

# =====================
# 八、主程序
# =====================

def main():
    print("2025年全国大学生数学建模竞赛A题最终完整解决方案（修正并行版）")
    print("无人机烟幕干扰优化问题")
    print("="*72)

    t0 = time.time()
    try:
        cov1, res1 = solve_problem1()
        res2 = solve_problem2()
        res3 = solve_problem3()

        # 汇总（提示：Q2/Q3的CSV为单弹时长，真正并集时长见日志）
        summary = {
            "问题一": {
                "描述": "FY1→M1 单次投放遮蔽（严格判据）",
                "结果": f"{cov1:.3f}秒",
                "关键参数": res1
            },
            "问题二": {
                "描述": "FY1/2/3 协同→M1（并行坐标搜索+并集优化）",
                "CSV": "result2_final.csv（单弹时长）"
            },
            "问题三": {
                "描述": "FY1..FY5→M1/M2/M3（并行评估分配）",
                "CSV": "result3_final.csv（各机选择的最佳导弹/时机及单弹时长）"
            }
        }
        with open('final_solution_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        t1 = time.time()
        print("\n" + "="*72)
        print("所有问题求解完成！")
        print(f"总计算时间：{t1 - t0:.2f} 秒")
        print("="*72)
        print("生成的文件：")
        print("- result1_final.csv: 问题一结果（严格判据）")
        print("- result2_final.csv: 问题二单弹结果（真正并集见日志）")
        print("- result3_final.csv: 问题三单弹结果（每机最佳导弹/时机）")
        print("- analysis_report.json: 结果分析报告")
        print("- final_solution_summary.json: 最终解决方案总结")

    except Exception as e:
        print(f"求解过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
