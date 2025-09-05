# -*- coding: utf-8 -*-
"""
Q2_Optimizer_All.py
问题二：优化 FY1 的航向角、速度、投放时刻、引信延时，使对 M1 的遮蔽时长最大化。
一次性依次执行五种算法：Hybrid / DE / PSO / SA / Pattern
- 并行评估 + 两阶段精度（粗筛/终评）
- 兼容 Q1Solver 的几何判据（导入失败有回退实现）
- 每个算法各自生成收敛图；Hybrid 生成多子图（LHS/Pattern/DE）

示例：
    python Q2_Optimizer_All.py --algo all --pop 64 --iter 60 --workers auto --topk 12 \
        --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 \
        --dt-final 0.0005 --nphi-final 960 --nz-final 13
"""

import os
import sys
import math
import time
import json
import random
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# 可视化
import matplotlib
matplotlib.use("Agg")  # 兼容无图形界面的环境
import matplotlib.pyplot as plt

# =========================
# 0) 尝试导入 Q1Solver（首选）；失败则回退到内置实现
# =========================
Q1_IMPORTED = False
try:
    import Q1Solver as Q1
    # 关键常量与函数
    g = getattr(Q1, "g", 9.8)
    MISSILE_SPEED = getattr(Q1, "MISSILE_SPEED", 300.0)
    SMOG_R = getattr(Q1, "SMOG_R", 10.0)
    SMOG_SINK_SPEED = getattr(Q1, "SMOG_SINK_SPEED", 3.0)
    SMOG_EFFECT_TIME = getattr(Q1, "SMOG_EFFECT_TIME", 20.0)
    FY1_INIT = getattr(Q1, "FY1_INIT", np.array([17800.0, 0.0, 1800.0], dtype=float))
    M1_INIT = getattr(Q1, "M1_INIT", np.array([20000.0, 0.0, 2000.0], dtype=float))
    FAKE_TARGET_ORIGIN = getattr(Q1, "FAKE_TARGET_ORIGIN", np.array([0.0, 0.0, 0.0], dtype=float))
    EPS = getattr(Q1, "EPS", 1e-12)

    Unit = Q1.Unit
    MissileState = Q1.MissileState
    UavStateHorizontal = Q1.UavStateHorizontal
    PreCalCylinderPoints = Q1.PreCalCylinderPoints
    ConeAllPointsIn = Q1.ConeAllPointsIn

    Q1_IMPORTED = True
    print("[Q2] 已成功导入 Q1Solver 中的常量与函数。")
except Exception as e:
    print(f"[Q2] 未能导入 Q1Solver，使用内置回退实现。原因：{e}")
    # ------- 回退实现（与 Q1 严格几何判据一致） -------
    g = 9.8
    MISSILE_SPEED = 300.0
    SMOG_R = 10.0
    SMOG_SINK_SPEED = 3.0
    SMOG_EFFECT_TIME = 20.0
    FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype=float)
    M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)
    FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)
    EPS = 1e-12

    def Unit(v):
        n = np.linalg.norm(v)
        return v if n < EPS else (v / n)

    def MissileState(t, mInit):
        dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
        v = MISSILE_SPEED * dirToOrigin
        return mInit + v * t, v

    def UavStateHorizontal(t, uavInit, uavSpeed, headingRad):
        vx = uavSpeed * math.cos(headingRad)
        vy = uavSpeed * math.sin(headingRad)
        return np.array([uavInit[0] + vx * t,
                         uavInit[1] + vy * t,
                         uavInit[2]], dtype=float), \
               np.array([vx, vy, 0.0], dtype=float)

    def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
        b = np.array([0.0, 200.0, 0.0], dtype=dtype)
        r, h = dtype(7.0), dtype(10.0)
        phis = np.linspace(0.0, 2.0*math.pi, nPhi, endpoint=False, dtype=dtype)
        c, s = np.cos(phis), np.sin(phis)
        ring = np.stack([r*c, r*s, np.zeros_like(c)], axis=1).astype(dtype)
        pts = [b + ring, b + np.array([0.0,0.0,h], dtype=dtype) + ring]
        if nZ >= 2:
            for z in np.linspace(0.0, h, nZ, dtype=dtype):
                pts.append(b + np.array([0.0,0.0,z], dtype=dtype) + ring)
        p = np.vstack(pts).astype(dtype)
        return p

    def ConeAllPointsIn(m, c, p, rCloud=SMOG_R, margin=EPS, block=8192):
        v = c - m
        l = np.linalg.norm(v)
        if l <= EPS or rCloud >= l:
            return True
        cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud/l)**2))
        for i in range(0, len(p), block):
            w = p[i:i+block] - m
            wn = np.linalg.norm(w, axis=1) + EPS
            lhs = w @ v
            rhs = wn * l * cosAlpha
            if not np.all(lhs + margin >= rhs):
                return False
        return True
    # -----------------------------------------------

# =========================
# 1) 变量范围 & 公共工具
# =========================

HEADING_MIN, HEADING_MAX = 0.0, 2.0 * math.pi
SPEED_MIN, SPEED_MAX = 70.0, 140.0     # ★ 题面约束：受领后以 70~140 m/s 等高度匀速直线飞行
DROP_MIN, DROP_MAX = 0.0, 60.0         # 导弹命中约 67s，60 足以
FUSE_MIN, FUSE_MAX = 0.0, 18.0         # 过长易落地失效

def clamp_params(x):
    """边界处理 + 航向角归一到 [0, 2π)"""
    h, v, td, tf = x
    h = h % (2.0 * math.pi)
    v = min(max(v, SPEED_MIN), SPEED_MAX)
    td = min(max(td, DROP_MIN), DROP_MAX)
    tf = min(max(tf, FUSE_MIN), FUSE_MAX)
    return np.array([h, v, td, tf], dtype=float)

def explosion_point_from_params(heading, speed, drop, fuse):
    """由参数算起爆点（水平匀速 + 自由落体）"""
    dropPos, uavV = UavStateHorizontal(drop, FY1_INIT, speed, heading)
    expl_xy = dropPos[:2] + uavV[:2] * fuse
    expl_z = dropPos[2] - 0.5 * g * (fuse ** 2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=float)

def quick_hard_filters(heading, speed, drop, fuse):
    """
    便宜的硬过滤：爆点落地/时间窗为空 直接淘汰。
    返回 (是否通过, t0, t1, hit_time, expl_pos)
    """
    expl_pos = explosion_point_from_params(heading, speed, drop, fuse)
    hit_time = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    t0 = drop + fuse
    t1 = min(t0 + SMOG_EFFECT_TIME, hit_time)
    if expl_pos[2] <= 0.0:  # 爆在地面/地下
        return False, t0, t1, hit_time, expl_pos
    if t1 <= t0:            # 没有有效时间窗
        return False, t0, t1, hit_time, expl_pos
    return True, t0, t1, hit_time, expl_pos

# =========================
# 2) 遮蔽评估（两阶段精度）
# =========================

def evaluate_occlusion(heading, speed, drop, fuse,
                       dt=0.002, nphi=480, nz=9,
                       margin=EPS, block=8192) -> Tuple[float, np.ndarray, float]:
    """
    评估遮蔽总时长（严格圆锥判据）。用于粗筛/终评，参数可变。
    返回： (时长 seconds, 起爆点 expl_pos, 导弹命中时间 hit_time)
    """
    ok, t0, t1, hit_time, expl_pos = quick_hard_filters(heading, speed, drop, fuse)
    if not ok:
        return 0.0, expl_pos, hit_time

    # 时间网格
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)
    # 目标采样点
    pts = PreCalCylinderPoints(nphi, nz, dtype=float)

    mask = np.zeros(len(t_grid), dtype=bool)
    for j, t in enumerate(t_grid):
        m_pos, _ = MissileState(float(t), M1_INIT)
        c_pos = np.array([expl_pos[0], expl_pos[1],
                          expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype=float)
        mask[j] = ConeAllPointsIn(m_pos, c_pos, pts, rCloud=SMOG_R, margin=margin, block=block)
    seconds = float(np.count_nonzero(mask) * dt)
    return seconds, expl_pos, hit_time

# 并行池包装（函数需顶层可 picklable）
def _eval_tuple(args):
    (heading, speed, drop, fuse, dt, nphi, nz, margin, block) = args
    seconds, expl, hit = evaluate_occlusion(heading, speed, drop, fuse, dt, nphi, nz, margin, block)
    return seconds, (heading, speed, drop, fuse), expl, hit

# =========================
# 3) 采样器（LHS）与局部搜索（Pattern）
# =========================

def latin_hypercube(n_samples: int) -> np.ndarray:
    """在 4 维参数空间做简易 LHS（独立分层 + 随机打乱）"""
    rng = np.random.default_rng()
    u = (rng.random((n_samples,4)) + np.arange(n_samples)[:,None]) / n_samples
    rng.shuffle(u, axis=0)
    # 映射到实际范围
    headings = HEADING_MIN + u[:,0]*(HEADING_MAX - HEADING_MIN)
    speeds   = SPEED_MIN   + u[:,1]*(SPEED_MAX   - SPEED_MIN)
    drops    = DROP_MIN    + u[:,2]*(DROP_MAX    - DROP_MIN)
    fuses    = FUSE_MIN    + u[:,3]*(FUSE_MAX    - FUSE_MIN)
    # 轻微避免 drop=0 的陷阱：将极小值抬升一点点（不改变可行域）
    drops = np.maximum(drops, 0.05 * (DROP_MAX - DROP_MIN) / n_samples)
    return np.column_stack([headings, speeds, drops, fuses])

def pattern_search(x0: np.ndarray,
                   steps: np.ndarray,
                   eval_kwargs: dict,
                   max_iter: int = 60,
                   shrink: float = 0.5,
                   workers: int = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    模式/坐标搜索（无导数）：沿每个坐标±方向试探，若改进则接受；否则步长缩小。
    并行评估所有 8 个方向（4 维 × ±）。
    返回：best_val, best_x, expl_pos, hit_time, convergence_history
    """
    x = clamp_params(x0.copy())
    best_val, best_expl, best_hit = evaluate_occlusion(*x, **eval_kwargs)
    dirs = np.eye(4, dtype=float)
    convergence_history = [best_val]

    for it in range(max_iter):
        cands = []
        for d in dirs:
            cands.append(x + steps * d)
            cands.append(x - steps * d)
        # 并行评估
        tasks = [(clamp_params(c)[0], clamp_params(c)[1], clamp_params(c)[2], clamp_params(c)[3],
                  eval_kwargs.get("dt"), eval_kwargs.get("nphi"), eval_kwargs.get("nz"),
                  eval_kwargs.get("margin"), eval_kwargs.get("block")) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xcand, expl, hit = fut.result()
                vals.append((sec, np.array(xcand, dtype=float), expl, hit))
        # 选择更优
        improved = False
        for sec, xc, expl, hit in vals:
            if sec > best_val + 1e-12:
                best_val, best_expl, best_hit = sec, expl, hit
                x = xc
                improved = True

        convergence_history.append(best_val)

        if improved:
            continue
        # 否则缩步
        steps *= shrink
        if np.all(steps < np.array([1e-4, 0.05, 1e-3, 1e-3])):
            break
    return best_val, x, best_expl, best_hit, convergence_history

# =========================
# 4) 元启发式：DE / PSO / SA
# =========================

def de_opt(pop: int, iters: int, eval_kwargs: dict, workers: int = None,
           F: float = 0.7, CR: float = 0.9, strategy: str = "best1bin",
           init_seeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    差分进化（自实现），支持 'best1bin' / 'rand1bin'
    返回：best_score, best_x, best_expl, hit_time, convergence_history
    """
    rng = np.random.default_rng()
    # 初始化种群
    X = []
    if init_seeds:
        for s in init_seeds:
            X.append(clamp_params(s))
    while len(X) < pop:
        X.append(clamp_params(np.array([
            rng.uniform(HEADING_MIN, HEADING_MAX),
            rng.uniform(SPEED_MIN, SPEED_MAX),
            rng.uniform(DROP_MIN, DROP_MAX),
            rng.uniform(FUSE_MIN, FUSE_MAX),
        ], dtype=float)))
    X = np.array(X, dtype=float)

    # 初始评估（并行）
    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    scores, expls = np.zeros(pop, dtype=float), [None]*pop
    hit_time = 0.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            scores[i] = sec
            expls[i] = expl
            hit_time = hit
    best_idx = int(np.argmax(scores))
    gbest, gbest_score, gbest_expl = X[best_idx].copy(), float(scores[best_idx]), expls[best_idx]

    convergence_history = [gbest_score]

    for it in range(iters):
        newX = np.zeros_like(X)
        newScores = np.zeros_like(scores)
        newExpls = [None]*pop

        # 生成子代
        for i in range(pop):
            idxs = list(range(pop))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            if strategy == "best1bin":
                base = gbest
                mutant = base + F*(X[r1] - X[r2])
            else:  # rand1bin
                base = X[r1]
                mutant = base + F*(X[r2] - X[r3])
            # 交叉
            trial = np.empty(4, dtype=float)
            jrand = rng.integers(0, 4)
            for j in range(4):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = X[i][j]
            trial = clamp_params(trial)
            newX[i] = trial

        # 并行评估子代
        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in newX]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                newScores[i] = sec
                newExpls[i] = expl
                hit_time = hit

        # 贪婪选择
        for i in range(pop):
            if newScores[i] >= scores[i]:
                X[i] = newX[i]
                scores[i] = newScores[i]
                expls[i] = newExpls[i]

        # 更新全局最优
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > gbest_score:
            gbest_score = float(scores[best_idx])
            gbest = X[best_idx].copy()
            gbest_expl = expls[best_idx]

        convergence_history.append(gbest_score)

        if it % max(1, iters//10) == 0 or it == iters-1:
            print(f"[DE] iter {it+1}/{iters}  gbest={gbest_score:.6f}")

    return gbest_score, gbest, gbest_expl, hit_time, convergence_history

def pso_opt(pop: int, iters: int, eval_kwargs: dict, workers: int = None,
            w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
            init_seeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    粒子群优化（角度取模、越界反弹），外层并行评估
    返回：best_score, best_x, best_expl, hit_time, convergence_history
    """
    rng = np.random.default_rng()
    # 初始化
    X, V = [], []
    if init_seeds:
        for s in init_seeds:
            x = clamp_params(s)
            X.append(x)
            V.append(np.zeros(4, dtype=float))
    while len(X) < pop:
        x = np.array([rng.uniform(HEADING_MIN, HEADING_MAX),
                      rng.uniform(SPEED_MIN, SPEED_MAX),
                      rng.uniform(DROP_MIN, DROP_MAX),
                      rng.uniform(FUSE_MIN, FUSE_MAX)], dtype=float)
        X.append(clamp_params(x))
        V.append(rng.uniform(-1,1,size=4))
    X, V = np.array(X, dtype=float), np.array(V, dtype=float)

    # 初评
    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    Pbest = X.copy()
    Pscore = np.zeros(pop, dtype=float)
    expls = [None]*pop
    hit_time = 0.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            Pscore[i] = sec
            expls[i] = expl
            hit_time = hit
    g_idx = int(np.argmax(Pscore))
    Gbest, Gscore, Gexpl = X[g_idx].copy(), float(Pscore[g_idx]), expls[g_idx]

    convergence_history = [Gscore]

    for it in range(iters):
        # 速度/位置更新
        for i in range(pop):
            r1, r2 = rng.random(4), rng.random(4)
            V[i] = w*V[i] + c1*r1*(Pbest[i]-X[i]) + c2*r2*(Gbest-X[i])
            X[i] = clamp_params(X[i] + V[i])
            # 非角度维超界反弹
            for j,(mn,mx) in enumerate([(HEADING_MIN,HEADING_MAX),
                                        (SPEED_MIN,SPEED_MAX),
                                        (DROP_MIN,DROP_MAX),
                                        (FUSE_MIN,FUSE_MAX)]):
                if j==0:  # 角度不用反弹
                    continue
                if X[i][j] <= mn or X[i][j] >= mx:
                    V[i][j] *= -0.5

        # 并行评估
        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                if sec > Pscore[i]:
                    Pscore[i] = sec
                    Pbest[i] = X[i].copy()
                if sec > Gscore:
                    Gscore = sec
                    Gbest = X[i].copy()
                    Gexpl = expl
                hit_time = hit

        convergence_history.append(Gscore)

        if it % max(1, iters//10) == 0 or it == iters-1:
            print(f"[PSO] iter {it+1}/{iters}  gbest={Gscore:.6f}")
    return Gscore, Gbest, Gexpl, hit_time, convergence_history

def sa_opt(x0: np.ndarray, eval_kwargs: dict,
           T0: float = 1.0, Tend: float = 1e-3, iters: int = 20000,
           step_scale: np.ndarray = None, workers: int = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    模拟退火（单轨迹）
    返回：best_score, best_x, best_expl, hit_time, convergence_history
    """
    rng = np.random.default_rng()
    x = clamp_params(x0.copy())
    f, expl, hit = evaluate_occlusion(*x, **eval_kwargs)
    best_f, best_x, best_expl = f, x.copy(), expl
    convergence_history = [best_f]

    for k in range(1, iters+1):
        T = T0 * (Tend/T0) ** (k/iters)  # 指数降温
        if step_scale is None:
            step = np.array([0.02, 2.0, 0.1, 0.1]) * max(T, 1e-3)
        else:
            step = step_scale * max(T, 1e-3)
        # 高斯扰动
        cand = x + rng.normal(0.0, 1.0, size=4) * step
        cand = clamp_params(cand)
        fc, explc, hitc = evaluate_occlusion(*cand, **eval_kwargs)
        if fc >= f or rng.random() < math.exp((fc - f)/max(T,1e-9)):
            x, f = cand, fc
            expl, hit = explc, hitc
            if f > best_f:
                best_f, best_x, best_expl = f, x.copy(), expl

        if k % 10 == 0:
            convergence_history.append(best_f)

        if k % max(1, iters//10) == 0 or k == iters:
            print(f"[SA] step {k}/{iters}  current={f:.6f}  best={best_f:.6f}")
    return best_f, best_x, best_expl, hit, convergence_history

# =========================
# 5) HYBRID：LHS -> Pattern -> DE -> 终评（并记录三段历史）
# =========================

def hybrid_opt(pop: int, iters: int, workers: int,
               topk: int,
               dt_coarse: float, nphi_coarse: int, nz_coarse: int,
               dt_final: float, nphi_final: int, nz_final: int):
    """
    混合策略，返回：
      best_val, best_x, best_expl, hit_time, top_list, hybrid_history
    其中 hybrid_history = {
        "lhs": [ ... ],                         # LHS 粗筛阶段的“当前最好值”历史
        "pattern_each": [hist1, hist2, ...],    # 每个种子的 pattern_search 历史
        "pattern_global": [ ... ],              # 随着种子逐个完成后的全局最好值历史
        "de": [ ... ]                           # DE 的历史
    }
    """
    # --- 阶段1：LHS 粗筛 ---
    print("[HYBRID] 阶段1：LHS 粗筛")
    seeds = latin_hypercube(pop)
    tasks = [(row[0], row[1], row[2], row[3], dt_coarse, nphi_coarse, nz_coarse, EPS, 8192) for row in seeds]
    vals = []
    lhs_history = []
    best_so_far = 0.0

    stride = max(1, pop // 20)  # 每 5% 记录一次
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            vals.append((sec, np.array(x), expl, hit))
            if sec > best_so_far: best_so_far = sec
            done += 1
            if done % stride == 0 or done == pop:
                lhs_history.append(best_so_far)
                print(f"    [HYBRID] LHS 进度 {int(100*done/pop)}%  best_so_far={best_so_far:.6f}")

    vals.sort(key=lambda x: x[0], reverse=True)
    top = vals[:topk]
    print(f"[HYBRID] 粗筛 Top-{topk} 最佳 = {top[0][0]:.6f} s")

    # --- 阶段2：Top-K Pattern 局部精修 ---
    print("[HYBRID] 阶段2：Top-K 局部 Pattern Search 精修")
    refined = []
    pattern_each_histories = []
    pattern_global_history = []
    global_best_after_patterns = 0.0

    for rank,(sec0, x0, expl0, hit0) in enumerate(top, start=1):
        print(f"  -> Pattern 起点#{rank}  初值={sec0:.6f}")
        steps = np.array([0.1, 5.0, 0.5, 0.5], dtype=float)
        best_sec, best_x, best_expl, best_hit, hist = pattern_search(
            x0, steps,
            eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
            max_iter=60, shrink=0.6, workers=workers
        )
        refined.append((best_sec, best_x, best_expl, best_hit))
        pattern_each_histories.append(hist)
        global_best_after_patterns = max(global_best_after_patterns, best_sec)
        pattern_global_history.append(global_best_after_patterns)
        print(f"    Pattern 完成：{best_sec:.6f} s")

    refined.sort(key=lambda x: x[0], reverse=True)
    seeds2 = [r[1] for r in refined[:min(len(refined), max(4, topk//2))]]

    # --- 阶段3：以局部精修种子启动 DE（粗精度） ---
    print("[HYBRID] 阶段3：DE 全局收敛（粗评口径）")
    best_de, x_de, expl_de, hit_de, de_history = de_opt(
        pop=pop, iters=iters,
        eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
        workers=workers, F=0.7, CR=0.9, strategy="best1bin",
        init_seeds=seeds2
    )
    print(f"[HYBRID] DE 粗精度结果：{best_de:.6f} s")

    # --- 阶段4：终评（高精度） ---
    print("[HYBRID] 阶段4：终评（高精度复评）")
    candidates = []
    candidates.append((best_de, x_de, expl_de, hit_de))
    for r in refined[:min(10, len(refined))]:
        candidates.append(r)
    # 去重
    uniq, seen = [], set()
    def keyz(x):
        arr = x[1]
        return (round(arr[0],3), round(arr[1],2), round(arr[2],3), round(arr[3],3))
    for c in candidates:
        k = keyz(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)

    tasks = [(clamp_params(x[1])[0], clamp_params(x[1])[1], clamp_params(x[1])[2], clamp_params(x[1])[3],
              dt_final, nphi_final, nz_final, EPS, 8192) for x in uniq]
    finals = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            finals.append((sec, np.array(x), expl, hit))
    finals.sort(key=lambda x: x[0], reverse=True)
    best = finals[0]

    hybrid_history = dict(
        lhs=lhs_history,
        pattern_each=pattern_each_histories,
        pattern_global=pattern_global_history,
        de=de_history
    )
    return best[0], best[1], best[2], best[3], finals[:min(10,len(finals))], hybrid_history

# =========================
# 6) 报告与可视化
# =========================

def save_report(filename: str,
                algo: str,
                best_val: float, best_x: np.ndarray, best_expl: np.ndarray, hit_time: float,
                top_list: List[Tuple[float,np.ndarray,np.ndarray]] = None,
                params: Dict = None):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"Q2 Optimization Results Report - {algo.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write("Calculation Parameters:\n")
        f.write("-"*40 + "\n")
        if params:
            for k,v in params.items():
                f.write(f"{k}: {v}\n")
        # 明确题面运动学与速度约束
        f.write(f"Speed bounds enforced: [{SPEED_MIN:.1f}, {SPEED_MAX:.1f}] m/s\n")
        f.write("Kinematics: heading & speed are fixed after instant reorientation; level straight flight (z constant).\n")
        f.write(f"Algorithm: {algo}\n\n")

        f.write("Best Solution:\n")
        f.write("-"*40 + "\n")
        f.write(f"Occlusion duration = {best_val:.6f} seconds\n")
        f.write(f"Heading (rad) = {best_x[0]:.6f}\n")
        f.write(f"Speed (m/s)  = {best_x[1]:.6f}\n")
        f.write(f"Drop time (s) = {best_x[2]:.6f}\n")
        f.write(f"Fuse delay (s)= {best_x[3]:.6f}\n")
        f.write(f"Explosion point = ({best_expl[0]:.6f}, {best_expl[1]:.6f}, {best_expl[2]:.6f})\n")
        f.write(f"Missile hit time ≈ {hit_time:.6f} seconds\n\n")

        if top_list:
            f.write("Top Candidates (final evaluation):\n")
            f.write("-"*40 + "\n")
            for i,(sec,x,expl,_) in enumerate(top_list, start=1):
                f.write(f"#{i}: {sec:.6f}s | "
                        f"h={x[0]:.6f}, v={x[1]:.3f}, drop={x[2]:.3f}, fuse={x[3]:.3f} | "
                        f"expl=({expl[0]:.2f},{expl[1]:.2f},{expl[2]:.2f})\n")
        f.write("\n" + "="*80 + "\n")

def generate_convergence_plot(history: List[float], algorithm_name: str, best_value: float):
    """单曲线收敛图（DE/PSO/SA/Pattern）"""
    try:
        if not history or len(history) == 0:
            print(f"[Plot] {algorithm_name}: 无历史可绘图。")
            return
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        iters = list(range(len(history)))
        ax.plot(iters, history, linewidth=2, alpha=0.9, label='Best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Occlusion Duration (s)')
        ax.set_title(f'{algorithm_name.upper()} Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = f'Q2_{algorithm_name}_Convergence.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] Saved {out}")
    except Exception as e:
        print(f"[Warning] Failed to generate {algorithm_name} plot: {e}")

def generate_hybrid_convergence_plot(hybrid_history: Dict, best_value: float,
                                     coarse_cfg: Dict, final_cfg: Dict):
    """多子图：LHS / Pattern / DE 三段历史"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        # 1) LHS
        lhs = hybrid_history.get("lhs", [])
        ax = axes[0]
        if lhs:
            ax.plot(range(1, len(lhs)+1), lhs, linewidth=2, label='LHS best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_title(f"LHS Screening (coarse: dt={coarse_cfg['dt']}, nphi={coarse_cfg['nphi']}, nz={coarse_cfg['nz']})")
        ax.set_xlabel('Checkpoint (~5%)')
        ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2) Pattern
        ax = axes[1]
        pattern_each = hybrid_history.get("pattern_each", [])
        pattern_global = hybrid_history.get("pattern_global", [])
        if pattern_each:
            for i,hist in enumerate(pattern_each, start=1):
                ax.plot(range(len(hist)), hist, linewidth=1.2, alpha=0.6, label=f'seed#{i}')
        if pattern_global:
            ax.plot(range(1, len(pattern_global)+1), pattern_global, linewidth=2.5, label='Global best (after each seed)')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_title(f"Pattern Refinement (coarse: dt={coarse_cfg['dt']}, nphi={coarse_cfg['nphi']}, nz={coarse_cfg['nz']})")
        ax.set_xlabel('Pattern Iteration / Seed Index')
        ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        # 3) DE
        ax = axes[2]
        de_hist = hybrid_history.get("de", [])
        if de_hist:
            ax.plot(range(len(de_hist)), de_hist, linewidth=2, label='DE best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_title(f"DE Global Search (coarse: dt={coarse_cfg['dt']}, nphi={coarse_cfg['nphi']}, nz={coarse_cfg['nz']}; final: dt={final_cfg['dt']}, nphi={final_cfg['nphi']}, nz={final_cfg['nz']})")
        ax.set_xlabel('Generation')
        ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out = 'Q2_HYBRID_Convergence.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] Saved {out}")
    except Exception as e:
        print(f"[Warning] Failed to generate hybrid plot: {e}")

# =========================
# 7) 主流程：可选执行单算法，或 all 依次执行五种
# =========================

def main():
    ap = argparse.ArgumentParser("Q2 Optimizer (All)")
    ap.add_argument("--algo", choices=["all","hybrid","de","pso","sa","pattern"], default="all",
                    help="选择算法（all 表示依次执行五种）")
    ap.add_argument("--pop", type=int, default=64, help="population / swarm size (for hybrid/de/pso)")
    ap.add_argument("--iter", type=int, default=60, help="iterations / generations")
    ap.add_argument("--topk", type=int, default=12, help="top-k seeds for local refinement (hybrid)")
    ap.add_argument("--workers", default="auto", help="process workers, int or 'auto'")
    # 粗精度
    ap.add_argument("--dt-coarse", type=float, default=0.002)
    ap.add_argument("--nphi-coarse", type=int, default=480)
    ap.add_argument("--nz-coarse", type=int, default=9)
    # 终评精度
    ap.add_argument("--dt-final", type=float, default=0.0005)
    ap.add_argument("--nphi-final", type=int, default=960)
    ap.add_argument("--nz-final", type=int, default=13)
    args = ap.parse_args()

    workers = None if args.workers == "auto" else int(args.workers)

    # 多进程避免 BLAS 线程内卷
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # 要跑哪些算法
    algos = [args.algo] if args.algo != "all" else ["hybrid","de","pso","sa","pattern"]

    # 汇总最优
    overall_best = -1.0
    overall_rec = None  # (algo, best_val, best_x, best_expl, hit_time)

    for algo in algos:
        print("="*90)
        print(f"[Q2] Start algo={algo} | pop={args.pop} iters={args.iter} "
              f"| workers={workers or 'auto'}")
        print(f"[Q2] coarse(dt={args.dt_coarse}, nphi={args.nphi_coarse}, nz={args.nz_coarse}) "
              f"-> final(dt={args.dt_final}, nphi={args.nphi_final}, nz={args.nz_final})")
        print("="*90)

        tA = time.time()
        best_val = 0.0
        best_x = np.array([0,0,0,0], dtype=float)
        best_expl = np.array([0,0,0], dtype=float)
        hit_time = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
        top_list = None
        history = None
        hybrid_history = None

        if algo == "hybrid":
            best_val, best_x, best_expl, hit_time, top_list, hybrid_history = hybrid_opt(
                pop=args.pop, iters=args.iter, workers=workers or (os.cpu_count() or 1),
                topk=args.topk,
                dt_coarse=args.dt_coarse, nphi_coarse=args.nphi_coarse, nz_coarse=args.nz_coarse,
                dt_final=args.dt_final, nphi_final=args.nphi_final, nz_final=args.nz_final
            )
        elif algo == "de":
            best_val, best_x, best_expl, hit_time, history = de_opt(
                pop=args.pop, iters=args.iter,
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                F=0.7, CR=0.9, strategy="best1bin", init_seeds=None
            )
            # 终评
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        elif algo == "pso":
            best_val, best_x, best_expl, hit_time, history = pso_opt(
                pop=args.pop, iters=args.iter,
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                w=0.5, c1=1.5, c2=1.5, init_seeds=None
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        elif algo == "sa":
            x0 = np.array([math.atan2(-FY1_INIT[1], -FY1_INIT[0]), 120.0, 1.5, 3.6], dtype=float)
            best_val, best_x, best_expl, hit_time, history = sa_opt(
                x0=x0,
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                T0=1.0, Tend=1e-3, iters=max(20000, args.iter*500),
                step_scale=np.array([0.2, 10.0, 1.0, 1.0])
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        else:  # pattern
            x0 = np.array([math.atan2(-FY1_INIT[1], -FY1_INIT[0]), 120.0, 1.5, 3.6], dtype=float)
            best_val, best_x, best_expl, hit_time, history = pattern_search(
                x0=x0, steps=np.array([0.3, 10.0, 1.0, 1.0]),
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                max_iter=max(60, args.iter), shrink=0.6,
                workers=workers or (os.cpu_count() or 1)
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp

        tB = time.time()
        print("="*90)
        print(f"[Q2] DONE ({algo}). best={best_val:.6f} s | time={tB - tA:.2f}s")
        print("="*90)

        # 保存报告（每个算法独立文件）
        params = {
            "pop": args.pop, "iter": args.iter, "topk": args.topk,
            "coarse": dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse),
            "final": dict(dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final),
            "workers": workers or os.cpu_count() or 1
        }
        rpt = f"Q2Results_{algo}.txt"
        save_report(rpt, algo, best_val, best_x, best_expl, hit_time,
                    top_list=(None if algo!='hybrid' else None), params=params)
        print(f"[Info] Results saved to {rpt}")
        print(f"[Best-{algo}] h={best_x[0]:.6f} rad, v={best_x[1]:.3f} m/s, drop={best_x[2]:.3f} s, fuse={best_x[3]:.3f} s")
        print(f"[Best-{algo}] expl=({best_expl[0]:.2f},{best_expl[1]:.2f},{best_expl[2]:.2f}), occlusion={best_val:.6f}s")

        # 绘图
        if algo == "hybrid":
            if hybrid_history:
                generate_hybrid_convergence_plot(
                    hybrid_history, best_val,
                    coarse_cfg=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse),
                    final_cfg=dict(dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
                )
        else:
            if history is not None:
                generate_convergence_plot(history, algo, best_val)

        # 更新全局最优
        if best_val > overall_best:
            overall_best = best_val
            overall_rec = (algo, best_val, best_x, best_expl, hit_time)

    # 汇总所有算法的全局最优
    if overall_rec is not None:
        algo, best_val, best_x, best_expl, hit_time = overall_rec
        print("\n" + "="*90)
        print(f"[Q2] 全部算法完成。全局最优来自 {algo.upper()}: {best_val:.6f} s")
        print(f"     h={best_x[0]:.6f} rad, v={best_x[1]:.3f} m/s, drop={best_x[2]:.3f} s, fuse={best_x[3]:.3f} s")
        print(f"     expl=({best_expl[0]:.2f},{best_expl[1]:.2f},{best_expl[2]:.2f}), hit≈{hit_time:.3f}s")
        print("="*90)
        with open("Q2_All_Summary.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("Q2 Overall Best Across All Algorithms\n")
            f.write("="*80 + "\n")
            f.write(f"Algorithm: {algo}\n")
            f.write(f"Occlusion duration = {best_val:.6f} s\n")
            f.write(f"Heading (rad) = {best_x[0]:.6f}\n")
            f.write(f"Speed (m/s)  = {best_x[1]:.6f}\n")
            f.write(f"Drop time (s) = {best_x[2]:.6f}\n")
            f.write(f"Fuse delay (s)= {best_x[3]:.6f}\n")
            f.write(f"Explosion point = ({best_expl[0]:.6f}, {best_expl[1]:.6f}, {best_expl[2]:.6f})\n")
            f.write(f"Missile hit time ≈ {hit_time:.6f} s\n")
            f.write("="*80 + "\n")
        print("[Info] Overall summary saved to Q2_All_Summary.txt")

if __name__ == "__main__":
    main()
