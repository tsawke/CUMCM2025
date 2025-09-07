# -*- coding: utf-8 -*-
"""
Q4Solver_visual.py  ——  问题4优化混合求解器（基于origin改进）

核心改进：
1) 固定FY1参数为第二问最优解，专注优化FY2/FY3
2) 改进协同遮蔽算法：某时刻只要圆柱体上每个采样点都被至少一个烟幕圆锥遮住即可
3) 增强候选生成：保留单体为0的点 + 横向偏移策略以更易形成协同
4) 多阶段优化：粗筛 + SA + 局部精修，自适应参数调整
5) 更精细的搜索网格和更强的局部优化

输出：
- result2.xlsx（严格 10 列中文表头）
- Q4ConvergencePlot.png（收敛曲线）
- 控制台每有更优值实时打印详情
"""

import os, math, time, argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# =========================
# 常量/场景
# =========================
g = 9.8
MISSILE_SPEED = 300.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

# 三枚导弹初始位置（默认 M1）
M1_INIT = np.array([20000.0,    0.0, 2000.0], dtype=float)
M2_INIT = np.array([19000.0,  600.0, 2100.0], dtype=float)
M3_INIT = np.array([18000.0, -600.0, 1900.0], dtype=float)

FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)

# 无人机初始
FY1_INIT = np.array([17800.0,     0.0, 1800.0], dtype=float)
FY2_INIT = np.array([12000.0,  1400.0, 1400.0], dtype=float)
FY3_INIT = np.array([ 6000.0, -3000.0,  700.0], dtype=float)

# 固定FY1参数（来自第二问最优解）
FY1_FIXED = {
    "speed": 112.0298,
    "heading": 0.137353,
    "drop_time": 0.0045,
    "fuse_delay": 0.4950
}

EPS = 1e-12

# ============== 工具函数 ==============
def Unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, which="M1"):
    # 如需换导弹，在 main() 里把 which 改为 "M2"/"M3" 或做参数
    if which == "M2":
        mInit = M2_INIT
    elif which == "M3":
        mInit = M3_INIT
    else:
        mInit = M1_INIT
    dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
    v = MISSILE_SPEED * dirToOrigin
    return mInit + v * t, v

def UavStateHorizontal(t, uavInit, uavSpeed, headingRadius):
    vx = uavSpeed * math.cos(headingRadius)
    vy = uavSpeed * math.sin(headingRadius)
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype=float), np.array([vx, vy, 0.0], dtype=float)

def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
    b = CYLINDER_BASE_CENTER.astype(dtype)
    r, h = dtype(CYLINDER_R), dtype(CYLINDER_H)
    phis = np.linspace(0.0, 2.0 * math.pi, nPhi, endpoint=False, dtype=dtype)
    c, s = np.cos(phis), np.sin(phis)
    ring = np.stack([r * c, r * s, np.zeros_like(c)], axis=1).astype(dtype)
    pts = [b + ring, b + np.array([0.0, 0.0, h], dtype=dtype) + ring]
    if nZ >= 2:
        for z in np.linspace(0.0, h, nZ, dtype=dtype):
            pts.append(b + np.array([0.0, 0.0, z], dtype=dtype) + ring)
    p = np.vstack(pts).astype(dtype)
    return p

def ExplosionPointFromPlan(uavInit, speed, heading, drop_time, fuse_delay):
    dropPos, uavV = UavStateHorizontal(drop_time, uavInit, speed, heading)
    explXy = dropPos[:2] + uavV[:2] * fuse_delay
    explZ = dropPos[2] - 0.5 * g * (fuse_delay ** 2)
    return np.array([explXy[0], explXy[1], explZ], dtype=float), dropPos

def ConeAllPointsIn(m, c, p, rCloud=SMOG_R, margin=EPS, block=4096):
    v = c - m
    l = np.linalg.norm(v)
    if l <= EPS or rCloud >= l:
        return True
    cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud / l) ** 2))
    for i in range(0, len(p), block):
        w = p[i: i + block] - m
        wn = np.linalg.norm(w, axis=1) + EPS
        lhs = w @ v
        rhs = wn * l * cosAlpha
        if not np.all(lhs + margin >= rhs):
            return False
    return True

def ConePointsCoveredByAny(m, centers: List[np.ndarray], p, rCloud=SMOG_R, margin=EPS, block=4096):
    """协同：每个点被至少一个烟幕圆锥遮住即可；所有点均满足才算有效。"""
    if len(centers) == 0:
        return False
    vs = [c - m for c in centers]
    ls = np.array([np.linalg.norm(v) for v in vs], dtype=float)
    if np.any(ls <= rCloud + 1e-12):
        return True  # 导弹已经在烟幕里
    cosA = np.sqrt(np.maximum(0.0, 1.0 - (rCloud / ls) ** 2))
    V = np.stack(vs, axis=1)       # (3,K)
    LK = ls * cosA                 # (K,)
    for i in range(0, len(p), block):
        w = p[i: i + block] - m    # (B,3)
        wn = np.linalg.norm(w, axis=1) + EPS
        lhs = w @ V                # (B,K)
        rhs = wn[:, None] * LK[None, :]  # (B,K)
        good_any = (lhs + margin >= rhs) # (B,K)
        if not np.all(good_any.any(axis=1)):
            return False
    return True

# ========= 进程全局（子进程可用）=========
PTS_GLOBAL = None
HITTIME_GLOBAL = None
INTRA_THREADS_GLOBAL = 1
MISSILE_TAG = "M1"   # 如需切换 M2/M3，可在 main() 里改

def _init_worker(nphi, nz, intra_threads, missile_tag):
    global PTS_GLOBAL, HITTIME_GLOBAL, INTRA_THREADS_GLOBAL, MISSILE_TAG
    PTS_GLOBAL = PreCalCylinderPoints(nphi, nz, dtype=np.float64)
    # 命中时刻
    m0 = {"M1": M1_INIT, "M2": M2_INIT, "M3": M3_INIT}[missile_tag]
    HITTIME_GLOBAL = float(np.linalg.norm(m0 - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    INTRA_THREADS_GLOBAL = int(max(1, intra_threads))
    MISSILE_TAG = missile_tag

# ============== 掩码/评估 ==============
def _single_smoke_duration(T_e, explPos, dt):
    """单云团判据下的时长（用于候选打分 & 单体掩码），子进程安全。"""
    hitTime = HITTIME_GLOBAL
    pts = PTS_GLOBAL
    t0, t1 = T_e, min(T_e + SMOG_EFFECT_TIME, hitTime)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc, seen = t0, 0.0, False
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur, which=MISSILE_TAG)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, cur - T_e)], dtype=float)
        ok = ConeAllPointsIn(m, c, pts, rCloud=SMOG_R)
        if ok:
            acc += dt; seen = True
        elif seen:
            break
        cur += dt
    return float(acc)

def _single_mask(T_e, explPos, dt_grid):
    """单云团判据的时间掩码（布尔）；用于粗筛 OR/重叠近似。"""
    pts = PTS_GLOBAL
    mask = np.zeros_like(dt_grid, dtype=bool)
    for i, t in enumerate(dt_grid):
        if (t < T_e - 1e-12) or (t > T_e + SMOG_EFFECT_TIME + 1e-12):
            continue
        m, _ = MissileState(float(t), which=MISSILE_TAG)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, t - T_e)], dtype=float)
        mask[i] = ConeAllPointsIn(m, c, pts, rCloud=SMOG_R)
    return mask

def _cooperative_union_duration(triple, dt):
    """严格协同：联合遮蔽时长（并集，无重复计时）。"""
    pts = PTS_GLOBAL
    hitTime = HITTIME_GLOBAL
    Te = [triple[i]["T_e"] for i in range(3)]
    t0 = min(Te); t1 = min(hitTime, max(Te) + SMOG_EFFECT_TIME)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc = t0, 0.0
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur, which=MISSILE_TAG)
        centers = []
        for i in (0,1,2):
            Tei = triple[i]["T_e"]
            if (cur >= Tei - 1e-12) and (cur <= Tei + SMOG_EFFECT_TIME + 1e-12):
                ex = triple[i]["expl_pos"]
                centers.append(np.array([ex[0], ex[1], ex[2] - SMOG_SINK_SPEED * max(0.0, cur - Tei)], dtype=float))
        if centers and ConePointsCoveredByAny(m, centers, pts, rCloud=SMOG_R):
            acc += dt
        cur += dt
    return float(acc)

# ============== 爆点搜索（含横向偏移） ==============
def _best_candidate_for_Te_speed(uavInit, uavName, T_e, spd, dt, lateral_scales=(0.0, 0.015, -0.015, 0.025, -0.025)):
    """
    改进版候选搜索：
    1) 更多横向偏移选项
    2) 更精细的s参数搜索
    3) 保留单体时长为0的候选（用于协同）
    4) 速度容差放宽到5%
    """
    mPos, _ = MissileState(T_e, which=MISSILE_TAG)
    target_center = CYLINDER_BASE_CENTER + np.array([0.0, 0.0, CYLINDER_H*0.5], dtype=float)

    # 基向量
    L = target_center - mPos
    dist = np.linalg.norm(L)
    if dist < 1e-6:
        return None
    eL = L / dist
    # 找到水平面内、垂直于 eL 的单位向量（用于横向偏移）
    perp = np.array([-eL[1], eL[0], 0.0], dtype=float)
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([1.0, 0.0, 0.0], dtype=float)
    perp = perp / (np.linalg.norm(perp)+1e-12)

    s_grid = np.linspace(0.3, 0.95, 20)  # 更精细网格，扩大范围
    best, best_key = None, (-1.0, -1.0, -1.0)  # (dur, s, coop_potential)

    def try_s_d(s, d):
        nonlocal best, best_key
        # 爆点名义位置（用于求达成速度/投放/引信）
        cand = mPos + s * L + d * dist * perp
        drop_alt = uavInit[2]
        if cand[2] > drop_alt + 1e-9:
            return
        # XY 可达性 - 放宽速度容差到5%
        dx, dy = cand[0] - uavInit[0], cand[1] - uavInit[1]
    req_v = math.hypot(dx, dy) / max(T_e, 1e-9)
        if req_v > spd * 1.05:  # 5% 容差
            return
        heading = math.atan2(dy, dx)
        # 引信与投放时刻
        fuse_delay = math.sqrt(max(0.0, 2.0 * (drop_alt - cand[2]) / g))
        drop_time = T_e - fuse_delay
        if drop_time < -1e-9:
            return
        # 真正起爆点（XY 随 UAV 线性前进，Z 自由落体）
        explPos, dropPos = ExplosionPointFromPlan(uavInit, spd, heading, drop_time, fuse_delay)
        dur = _single_smoke_duration(T_e, explPos, dt)
        
        # 协同潜力评估：距离目标中心越近，s值越大，横向偏移适中都有利于协同
        coop_potential = 1.0 / (1.0 + np.linalg.norm(explPos[:2] - target_center[:2]) / 1000.0) + s * 0.1 + abs(d) * 0.05
        
        # 优先级：单体时长 > 协同潜力 > s值
        if (dur > best_key[0] + 1e-12) or \
           (abs(dur - best_key[0]) <= 1e-12 and coop_potential > best_key[2] + 1e-12) or \
           (abs(dur - best_key[0]) <= 1e-12 and abs(coop_potential - best_key[2]) <= 1e-12 and s > best_key[1]):
            best_key = (dur, s, coop_potential)
            best = {
                "T_e": float(T_e), "speed": float(spd), "heading": float(heading),
                "drop_time": float(drop_time), "fuse_delay": float(fuse_delay),
                "expl_pos": explPos, "drop_pos": dropPos,
                "single_duration": float(dur), "s_used": float(s), "d_used": float(d),
                "uavName": uavName, "coop_potential": float(coop_potential)
            }

    for dscale in lateral_scales:
        for s in s_grid:
            try_s_d(float(s), float(dscale))
        # 在当前 d 下对最优 s 附近再细化
        if best is not None:
            s0 = best["s_used"]
            for width in (0.08, 0.04):
                lo = max(0.3, s0 - width)
                hi = min(0.95, s0 + width)
                for s in np.linspace(lo, hi, 11):
                    try_s_d(float(s), float(dscale))

    return best  # 可能为 None

# ======== 并行 Worker ========
def _candidate_worker(pl):
    uavInit, T_e, spd, dt, name, lat_scales = pl["uavInit"], pl["T_e"], pl["speed"], pl["dt"], pl["uavName"], pl["lat_scales"]
    def _do():
        c = _best_candidate_for_Te_speed(uavInit, name, T_e, spd, dt, lateral_scales=lat_scales)
        return c
    if threadpool_limits is None:
        return _do()
    with threadpool_limits(limits=INTRA_THREADS_GLOBAL):
        return _do()

def _union_worker(pl):
    dt, tri = pl["dt"], pl["triplet"]
    def _do():
        return _cooperative_union_duration(tri, dt)
    if threadpool_limits is None:
        return _do()
    with threadpool_limits(limits=INTRA_THREADS_GLOBAL):
        return _do()

# ============== 并行配置 ==============
def _auto_balance(workers_opt, intra_opt, backend, total_tasks):
    cpu = os.cpu_count() or 1
    if backend == "thread":
        workers = cpu if (workers_opt == "auto") else int(workers_opt)
        return workers, 1
    workers = (cpu if workers_opt == "auto" else max(1, int(workers_opt)))
    if total_tasks < workers: workers = max(1, total_tasks)
    intra = (1 if intra_opt == "auto" else max(1, int(intra_opt)))
    return workers, intra

# ============== 打印更优解 ==============
def _deg(rad):
    d = math.degrees(rad)
    return d + 360.0 if d < 0 else d

def _print_best(prefix: str, union_time: float, tri):
    print(f"[Best↑][{prefix}] 联合遮蔽 ≈ {union_time:.6f} s")
    for c in sorted(tri, key=lambda x: x["uavName"]):
        h = _deg(c["heading"])
        dx,dy,dz = c["drop_pos"]; ex,ey,ez = c["expl_pos"]
        coop = c.get("coop_potential", 0.0)
        print(f"  - {c['uavName']}: 速={c['speed']:.3f} m/s, 航向={h:.3f}°"
              f", 投放=({dx:.3f},{dy:.3f},{dz:.3f}), 起爆=({ex:.3f},{ey:.3f},{ez:.3f})"
              f", 单体={c['single_duration']:.6f}s, s={c.get('s_used',-1):.3f}, d={c.get('d_used',0.0):+.4f}"
              f", T_e={c['T_e']:.3f}s, 协同潜力={coop:.4f}")

# ============== 可视化 ==============
def _plot_convergence(best_triple, dt, history, out_png="Q4ConvergencePlot.png"):
    it = np.arange(1, len(history)+1)
    best_curve = np.maximum.accumulate(np.array(history, dtype=float))
    pts_local = PTS_GLOBAL
    hit_local = HITTIME_GLOBAL
    Te = [best_triple[i]["T_e"] for i in range(3)]
    t0 = min(Te); t1 = min(hit_local, max(Te) + SMOG_EFFECT_TIME)
    if t1 <= t0 + 1e-12:
        plt.figure(figsize=(10,6), dpi=120)
        plt.plot(it, best_curve, lw=2)
        plt.xlabel("Iteration"); plt.ylabel("Best Union (s)")
        plt.title("Q4 Convergence")
        plt.grid(alpha=0.3); plt.savefig(out_png, dpi=300, bbox_inches='tight'); return
    tGrid = np.arange(t0, t1+1e-12, dt, dtype=float)
    mask = np.zeros_like(tGrid, dtype=bool)
    for i, t in enumerate(tGrid):
        m, _ = MissileState(float(t), which=MISSILE_TAG)
        centers = []
        for k in (0,1,2):
            Tei = best_triple[k]["T_e"]
            if (t >= Tei - 1e-12) and (t <= Tei + SMOG_EFFECT_TIME + 1e-12):
                ex = best_triple[k]["expl_pos"]
                centers.append(np.array([ex[0], ex[1], ex[2] - SMOG_SINK_SPEED * max(0.0, t - Tei)], dtype=float))
        if centers:
            mask[i] = ConePointsCoveredByAny(m, centers, pts_local, rCloud=SMOG_R)
    cum = np.cumsum(mask.astype(float)) * dt
    final_val = float(cum[-1])
    plt.figure(figsize=(10,8), dpi=120)
    ax1 = plt.subplot(2,1,1)
    ax1.plot(it, best_curve, lw=2); ax1.grid(alpha=0.3)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Union (s)")
    ax1.set_title("Hybrid Optimization Convergence (best-so-far)")
    ax1.axhline(final_val, ls='--', lw=1.5, c='r', label=f'Final best: {final_val:.6f}s'); ax1.legend()
    ax2 = plt.subplot(2,1,2)
    ax2.step(tGrid, mask.astype(int), where='post', lw=1.2, label="union mask (0/1)")
    ax2_2 = ax2.twinx(); ax2_2.plot(tGrid, cum, lw=2, alpha=0.8, label="cumulative")
    ax2.grid(alpha=0.3); ax2_2.grid(alpha=0.1)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mask"); ax2_2.set_ylabel("Cumulative (s)")
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[Q4] Convergence plot saved: {out_png}")

# ============== 主流程 ==============
def main():
    ap = argparse.ArgumentParser("Q4 Hybrid Visual Solver (cooperative)")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--dt-coarse", type=float, default=0.003)
    ap.add_argument("--nphi", type=int, default=960); ap.add_argument("--nz", type=int, default=13)
    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", default="auto"); ap.add_argument("--intra-threads", default="auto")
    ap.add_argument("--speed-grid", type=str, default="140,130,120,110,100,90,80,70")
    ap.add_argument("--fy1-te", type=str, default="4,12,0.02")
    ap.add_argument("--fy2-te", type=str, default="9,22,0.02")
    ap.add_argument("--fy3-te", type=str, default="18,36,0.02")
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--min-gap", type=float, default=0.30)
    ap.add_argument("--combo-topm", type=int, default=64)
    ap.add_argument("--sa-iters", type=int, default=80)
    ap.add_argument("--sa-batch", default="auto")
    ap.add_argument("--sa-T0", type=float, default=1.0)
    ap.add_argument("--sa-alpha", type=float, default=0.92)
    ap.add_argument("--sigma-Te", type=float, default=0.6)
    ap.add_argument("--sigma-v", type=float, default=12.0)
    ap.add_argument("--local-iters", type=int, default=40)
    ap.add_argument("--missile", choices=["M1","M2","M3"], default="M1")
    ap.add_argument("--lat-scales", type=str, default="0.0,0.015,-0.015")  # 横向偏移比例
    ap.add_argument("--overlap-lambda", type=float, default=0.2)           # 粗评重叠惩罚权重
    args = ap.parse_args()

    # ====== 主进程初始化全局 ======
    _init_worker(args.nphi, args.nz, 1, args.missile)

    # 解析参数
    def _parse_te(s):
        a,b,st = [float(x) for x in s.split(",")]
        st = 0.05 if st <= 0 else st
        return np.arange(a, b + 1e-12, st)
    def _parse_speed(s):
        vs = []
        for tok in s.split(","):
            if tok.strip():
                v = float(tok); vs.append(max(70.0, min(140.0, v)))
        return sorted(set(vs), reverse=True)
    def _parse_lat_scales(s):
        arr = []
        for tok in s.split(","):
            if tok.strip():
                arr.append(float(tok))
        if not arr: arr = [0.0]
        return arr

    speedGrid = _parse_speed(args.speed_grid)
    te1 = _parse_te(args.fy1_te); te2 = _parse_te(args.fy2_te); te3 = _parse_te(args.fy3_te)
    lat_scales = _parse_lat_scales(args.lat_scales)

    hitTime = HITTIME_GLOBAL
    print(f"[Info] Hit time: {hitTime:.3f}s, Speed grid: {len(speedGrid)} values")

    # ====== Stage-1 固定FY1，生成FY2/FY3候选 ======
    print("="*110)
    print("Stage-1: Generate candidates (FY1 fixed, FY2/FY3 optimized)")
    print("="*110)

    # FY1使用固定参数
    fy1_explPos, fy1_dropPos = ExplosionPointFromPlan(
        FY1_INIT, FY1_FIXED["speed"], FY1_FIXED["heading"], 
        FY1_FIXED["drop_time"], FY1_FIXED["fuse_delay"]
    )
    fy1_Te = FY1_FIXED["drop_time"] + FY1_FIXED["fuse_delay"]
    fy1_candidate = {
        "T_e": fy1_Te, "speed": FY1_FIXED["speed"], "heading": FY1_FIXED["heading"],
        "drop_time": FY1_FIXED["drop_time"], "fuse_delay": FY1_FIXED["fuse_delay"],
        "expl_pos": fy1_explPos, "drop_pos": fy1_dropPos,
        "single_duration": _single_smoke_duration(fy1_Te, fy1_explPos, args.dt),
        "uavName": "FY1", "s_used": 0.75, "d_used": 0.0, "coop_potential": 1.0
    }

    # 生成FY2/FY3候选任务
    gen_tasks = []
    uavs = [("FY2", FY2_INIT, te2), ("FY3", FY3_INIT, te3)]
    for name, init, teGrid in uavs:
        for T_e in teGrid:
            for spd in speedGrid:
                gen_tasks.append({
                    "uavName": name, "uavInit": init, "T_e": float(T_e),
                    "speed": float(spd), "dt": float(args.dt), "lat_scales": lat_scales
                })

    workers, intra = _auto_balance(args.workers, args.intra_threads, args.backend, len(gen_tasks))
    poolCls = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor

    print(f"backend={args.backend}, workers={workers}, intra-threads={intra}, tasks={len(gen_tasks)}")

    best_lists = {"FY2": [], "FY3": []}
    tA = time.time()
    if args.backend == "process":
        with poolCls(max_workers=workers, initializer=_init_worker, initargs=(args.nphi, args.nz, intra, args.missile)) as pool:
            futs = { pool.submit(_candidate_worker, pl): i for i,pl in enumerate(gen_tasks) }
            done,total = 0,len(futs)
            for fut in as_completed(futs):
                done += 1
                try: r = fut.result()
                except Exception: r = None
                if r is not None:
                    best_lists[r["uavName"]].append(r)
                if done % max(1,total//20)==0:
                    print(f"   [Gen] {int(100*done/total)}%")
    else:
        _init_worker(args.nphi, args.nz, 1, args.missile)
        with poolCls(max_workers=workers) as pool:
            futs = { pool.submit(_candidate_worker, pl): i for i,pl in enumerate(gen_tasks) }
            done,total = 0,len(futs)
            for fut in as_completed(futs):
                done += 1
                try: r = fut.result()
                except Exception: r = None
                if r is not None:
                    best_lists[r["uavName"]].append(r)
                if done % max(1,total//20)==0:
                    print(f"   [Gen] {int(100*done/total)}%")
    tB = time.time()
    print(f"[Stage-1] FY1=1(fixed), FY2={len(best_lists['FY2'])}, FY3={len(best_lists['FY3'])} | {tB-tA:.2f}s")

    # TopK（混合：70% 单体优先 + 30% 协同潜力高的候选）
    def _select_topk_enhanced(cands, topk, min_gap):
        if not cands: return []
        pos = [c for c in cands if c["single_duration"] > 0.0]
        zer = [c for c in cands if c["single_duration"] <= 1e-12]
        pos = sorted(pos, key=lambda x: (-x["single_duration"], -x.get("coop_potential", 0.0), x["T_e"]))
        zer = sorted(zer, key=lambda x: (-x.get("coop_potential", 0.0), -x.get("s_used", 0.0), x["T_e"]))
        sel, used = [], []
        want_pos = int(round(topk*0.70))
        for c in pos:
            T = c["T_e"]
            if all(abs(T-u)>=min_gap for u in used):
                sel.append(c); used.append(T)
            if len(sel)>=want_pos: break
        for c in zer:
            if len(sel)>=topk: break
            T=c["T_e"]
            if all(abs(T-u)>=min_gap for u in used):
                sel.append(c); used.append(T)
        if len(sel)<topk:
            for c in pos:
                if c in sel: continue
                T=c["T_e"]
                if all(abs(T-u)>=min_gap for u in used):
                    sel.append(c); used.append(T)
                if len(sel)>=topk: break
        return sel

    fy2 = _select_topk_enhanced(best_lists["FY2"], args.topk, args.min_gap)
    fy3 = _select_topk_enhanced(best_lists["FY3"], args.topk, args.min_gap)

    if not (fy2 and fy3):
        print("[Warn] 无可行候选，输出全 0。")
        cols = ['无人机编号','无人机运动方向','无人机运动速度 (m/s)',
                '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
                '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
                '有效干扰时长 (s)']
        pd.DataFrame([{
            '无人机编号':n, '无人机运动方向':0.0, '无人机运动速度 (m/s)':0.0,
            '烟幕干扰弹投放点的x坐标 (m)':0.0, '烟幕干扰弹投放点的y坐标 (m)':0.0, '烟幕干扰弹投放点的z坐标 (m)':0.0,
            '烟幕干扰弹起爆点的x坐标 (m)':0.0, '烟幕干扰弹起爆点的y坐标 (m)':0.0, '烟幕干扰弹起爆点的z坐标 (m)':0.0,
            '有效干扰时长 (s)':0.0
        } for n in ("FY1","FY2","FY3")], columns=cols).to_excel("result2.xlsx", index=False)
        print("[Result] 0.000 s | Saved result2.xlsx")
        return

    print(f"[Stage-1] Selected: FY2={len(fy2)}, FY3={len(fy3)}")

    # ====== Stage-2 粗筛（单体掩码 OR + 重叠惩罚） ======
    print("="*110); print("Stage-2: Combination coarse screening (mask OR + overlap penalty)"); print("="*110)

    tGrid_coarse = np.arange(0.0, hitTime+1e-12, args.dt_coarse, dtype=float)

    # 为每个候选缓存单体掩码（单云团判据）
    def _mask_for(c):
        return _single_mask(c["T_e"], c["expl_pos"], tGrid_coarse)

    # 为FY2/FY3候选缓存单体掩码，FY1单独处理
    fy1_candidate["_mask"] = _mask_for(fy1_candidate)
    for lst in (fy2, fy3):
        for c in lst:
            c["_mask"] = _mask_for(c)

    def _coarse_score_triple(c1,c2,c3, lam=args.overlap_lambda):
        m1, m2, m3 = c1["_mask"], c2["_mask"], c3["_mask"]
        union = (m1 | m2 | m3).sum() * args.dt_coarse
        # 重叠近似（单体）：鼓励时间窗拉开
        overlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
        return float(union - lam * overlap)

    # 贪心搜索最佳组合（限制搜索范围以加速）
    best_triple = None
    best_coarse = -1.0
    combo_count = 0
    
    for c2 in fy2[:min(len(fy2), 48)]:
        for c3 in fy3[:min(len(fy3), 48)]:
            combo_count += 1
            score = _coarse_score_triple(fy1_candidate, c2, c3)
            if score > best_coarse + 1e-12:
                best_coarse = score
                best_triple = (fy1_candidate, c2, c3)

    tC = time.time()
    tD = time.time()
    print(f"[Stage-2] Evaluated {combo_count} combinations, best ≈ {best_coarse:.6f}s | {tD-tC:.2f}s")
    
    if best_triple is None:
        print("[Warn] 无可行组合，输出全 0。")
        return

    # ====== Stage-3 SA（目标=协同联合时长 − λ×单体重叠） ======
    print("="*110); print("Stage-3: Parallel-batch SA (cooperative exact + overlap penalty)"); print("="*110)

    # 以粗评最佳作起点
    # 精确协同打分
    _init_worker(args.nphi, args.nz, 1, args.missile)
    exact_best = _cooperative_union_duration(best_triple, args.dt_coarse)
    # 近似重叠
    lam = float(args.overlap_lambda)
    m1,m2,m3 = best_triple[0]["_mask"],best_triple[1]["_mask"],best_triple[2]["_mask"]
    approx_overlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
    Jbest = exact_best - lam*approx_overlap
    _print_best("Seed", exact_best, best_triple)

    # 变量：x = [Te1,Te2,Te3,v1,v2,v3]
    def _x_of(tri):
        return np.array([tri[0]["T_e"], tri[1]["T_e"], tri[2]["T_e"],
                         tri[0]["speed"], tri[1]["speed"], tri[2]["speed"]], dtype=float)

    def _clip_x(x):
        limsTe = [(te1[0],te1[-1]), (te2[0],te2[-1]), (te3[0],te3[-1])]
        out = x.copy()
        for i,(lo,hi) in enumerate(limsTe):
            out[i] = min(hi, max(lo, out[i]))
        for i in range(3,6):
            out[i] = min(140.0, max(70.0, out[i]))
    return out

    def _tri_from_x(x):
        # FY1固定，只更新FY2和FY3
        out = [fy1_candidate]  # FY1固定
        inits = [FY2_INIT, FY3_INIT]
        for k in range(2):
            cand = _best_candidate_for_Te_speed(inits[k], f"FY{k+2}", x[k], x[3+k], args.dt, lateral_scales=lat_scales)
            if cand is None: return None
            cand["_mask"] = _single_mask(cand["T_e"], cand["expl_pos"], tGrid_coarse)
            out.append(cand)
        return tuple(out)

    # SA 参数
    T = float(args.sa_T0); alpha = float(args.sa_alpha)
    sigTe = float(args.sigma_Te); sigV = float(args.sigma_v)
    iters = int(args.sa_iters)
    cpu = os.cpu_count() or 1
    sa_batch = (max(cpu*2,16) if (isinstance(args.sa_batch,str) and args.sa_batch.lower()=="auto") else max(8,int(args.sa_batch)))
    history = [exact_best]

    x_cur = _x_of(best_triple)
    tri_cur = best_triple
    J_cur = Jbest
    exact_cur = exact_best

    # 为了吃满核，对 proposals 的精确协同评分用进程池
    workers2, intra2 = _auto_balance(args.workers, args.intra_threads, args.backend, sa_batch)
    if args.backend == "process":
        pool = ProcessPoolExecutor(max_workers=workers2, initializer=_init_worker, initargs=(args.nphi, args.nz, intra2, args.missile))
    else:
        _init_worker(args.nphi, args.nz, 1, args.missile)
        pool = ThreadPoolExecutor(max_workers=workers2)

    try:
        for it in range(1, iters+1):
            props=[]
            for _ in range(sa_batch):
                noise = np.array([np.random.normal(0,sigTe),np.random.normal(0,sigTe),np.random.normal(0,sigTe),
                                  np.random.normal(0,sigV), np.random.normal(0,sigV), np.random.normal(0,sigV)], dtype=float)
                props.append(_clip_x(x_cur + noise))

            triples=[_tri_from_x(x) for x in props]
            # 近似得分（先算单体掩码 OR + 重叠）
            approx_scores=[]
            for tri in triples:
                if tri is None:
                    approx_scores.append(-1e9); continue
                m1,m2,m3 = tri[0]["_mask"],tri[1]["_mask"],tri[2]["_mask"]
                union = (m1|m2|m3).sum()*args.dt_coarse
                overlap=((m1&m2).sum()+(m1&m3).sum()+(m2&m3).sum())*args.dt_coarse
                approx_scores.append(float(union - lam*overlap))

            # 选前 K 个做精确协同打分（节流）
            order = np.argsort(np.array(approx_scores))[::-1]
            topK = min( max(8, workers2), len(order) )
            pick_idx = order[:topK]
            futures={}
            for idx in pick_idx:
                tri = triples[idx]
                if tri is None: continue
                futures[pool.submit(_union_worker, {"triplet": tri, "dt": float(args.dt_coarse)})]=idx

            exact = { }  # idx -> exact union
            for fut in as_completed(futures):
                idx=futures[fut]
                try: val=fut.result()
                except Exception: val=0.0
                exact[idx]=float(val)

            # 选择接受
            accepted=False
            # 以“精确得分 − λ×近似重叠”为准
            best_idx=None; best_val=-1e9
            for idx,val in exact.items():
                tri=triples[idx]
                m1,m2,m3=tri[0]["_mask"],tri[1]["_mask"],tri[2]["_mask"]
                overlap=((m1&m2).sum()+(m1&m3).sum()+(m2&m3).sum())*args.dt_coarse
                J = val - lam*overlap
                if J > best_val: best_val, best_idx = J, idx
            if best_idx is not None:
                # Metropolis
                dJ = best_val - J_cur
                if (dJ >= 0) or (np.random.rand() < math.exp(dJ/max(T,1e-9))):
                    tri_cur = triples[best_idx]; x_cur = props[best_idx]
                    exact_cur = exact[best_idx]; J_cur = best_val
                    accepted=True
                    if exact_cur > max(history)+1e-12:
                        _print_best(f"SA@{it}", exact_cur, tri_cur)
                history.append(max(max(history), exact_cur))
            else:
                history.append(max(history))

            T *= alpha
            if it % max(1,iters//10)==0:
                print(f"   [SA] iter {it}/{iters}, best≈{max(history):.4f}s, T={T:.4f}")

            # 也可顺便尝试把 seeds 里的若干个做一次精评更新（可选）

    finally:
        pool.shutdown(wait=True)

    best_triple = tri_cur
    best_exact = exact_cur

    # ====== Stage-4 局部精修（小扰动增量搜索） ======
    print("="*110); print("Stage-4: Local refine"); print("="*110)
    def _clip_final(x):
        x2 = x.copy()
        x2[0] = min(te1[-1], max(te1[0], x2[0]))
        x2[1] = min(te2[-1], max(te2[0], x2[1]))
        x2[2] = min(te3[-1], max(te3[0], x2[2]))
        for j in range(3,6): x2[j] = min(140.0, max(70.0, x2[j]))
        return x2

    x_best = _x_of(best_triple)
    for k in range(int(args.local_iters)):
        stepT = max(0.05, args.sigma_Te*0.25)
        stepV = max(1.0,  args.sigma_v*0.25)
        improved=False
        for j in range(6):
            x_try = x_best.copy()
            x_try[j] += (stepT if j<3 else stepV) * (1 if (k+j)%2==0 else -1)
            x_try = _clip_final(x_try)
            tri = _tri_from_x(x_try)
            if tri is None: continue
            exact = _cooperative_union_duration(tri, args.dt)
            if exact > best_exact + 1e-12:
                best_exact = exact; best_triple = tri; x_best = x_try
                _print_best(f"Local@{k+1}", best_exact, best_triple)
                improved=True
        if not improved:
            # 轻微随机化继续探索
            x_best = _clip_final(x_best + np.array([np.random.normal(0, stepT*0.3)]*3 + [np.random.normal(0, stepV*0.3)]*3))

    # ====== 输出 Excel（中文10列） ======
    rows=[]
    for c in sorted(best_triple, key=lambda x: x["uavName"]):
        h=_deg(c["heading"])
        dx,dy,dz=c["drop_pos"]; ex,ey,ez=c["expl_pos"]
        rows.append({
            "无人机编号": c["uavName"],
            "无人机运动方向": round(h,6),
            "无人机运动速度 (m/s)": round(c["speed"],6),
            "烟幕干扰弹投放点的x坐标 (m)": round(dx,6),
            "烟幕干扰弹投放点的y坐标 (m)": round(dy,6),
            "烟幕干扰弹投放点的z坐标 (m)": round(dz,6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(ex,6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(ey,6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(ez,6),
            "有效干扰时长 (s)": round(c["single_duration"],6)
        })
    df = pd.DataFrame(rows)[[
        '无人机编号','无人机运动方向','无人机运动速度 (m/s)',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)'
    ]]
    df.to_excel("result2.xlsx", index=False)
    print("-"*110)
    print(df.to_string(index=False))
    print(f"[Result] TOTAL UNION (cooperative, no double count) ≈ {best_exact:.6f} s  |  已保存: result2.xlsx")

    _plot_convergence(best_triple, args.dt, history, out_png="Q4ConvergencePlot.png")

if __name__ == "__main__":
    main()
