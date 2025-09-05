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
        --dt-final 0.0005 --nphi-final 960 --nz-final 13 \
        --sa-iters 8000 --sa-batch 32 --sa-chains 8 --probe 256
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# 可视化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 0) 尝试导入 Q1Solver（首选）；失败则回退到内置实现
# =========================
# 先打印一个 * 不换行，成功或失败后覆盖这一行
sys.stdout.write("*")
sys.stdout.flush()

Q1_IMPORTED = False
try:
    import Q1Solver as Q1
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
    sys.stdout.write("\r[Init] Geometry backend: external(Q1Solver)\n")
    sys.stdout.flush()
except Exception:
    sys.stdout.write("\r[Init] Geometry backend: internal(fallback)\n")
    sys.stdout.flush()
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
        return (np.array([uavInit[0] + vx * t,
                          uavInit[1] + vy * t,
                          uavInit[2]], dtype=float),
                np.array([vx, vy, 0.0], dtype=float))

    def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
        """真目标圆柱：底心(0,200,0), 半径7, 高10。"""
        b = np.array([0.0, 200.0, 0.0], dtype=dtype)
        r, h = dtype(7.0), dtype(10.0)
        phis = np.linspace(0.0, 2.0*math.pi, nPhi, endpoint=False, dtype=dtype)
        ring = np.stack([r*np.cos(phis), r*np.sin(phis), np.zeros_like(phis)], axis=1).astype(dtype)
        pts = [b + ring, b + np.array([0.0,0.0,h], dtype=dtype) + ring]
        if nZ >= 2:
            for z in np.linspace(0.0, h, nZ, dtype=dtype):
                pts.append(b + np.array([0.0,0.0,z], dtype=dtype) + ring)
        return np.vstack(pts).astype(dtype)

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
# 1) 变量范围 & 公共工具（含题面约束）
# =========================

HEADING_MIN, HEADING_MAX = 0.0, 2.0 * math.pi
# 题面速度约束：70~140 m/s
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DROP_MIN, DROP_MAX_HARD = 0.0, 60.0
FUSE_MIN, FUSE_MAX = 0.0, 18.0

HIT_TIME = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
# drop 不得大到使 t0 超过导弹命中
DROP_MAX = min(DROP_MAX_HARD, max(0.0, HIT_TIME - FUSE_MIN - 1e-3))

def clamp_params(x):
    """边界处理 + 航向角归一到 [0, 2π) + 时间可行域修正(t0<hit)"""
    h, v, td, tf = x
    h = h % (2.0 * math.pi)
    v = min(max(v, SPEED_MIN), SPEED_MAX)
    td = min(max(td, DROP_MIN), DROP_MAX)
    tf = min(max(tf, FUSE_MIN), FUSE_MAX)
    # 关键：强制 t0 = td + tf < HIT_TIME
    if td + tf >= HIT_TIME - 1e-6:
        td = max(DROP_MIN, HIT_TIME - 1e-6 - tf)
    return np.array([h, v, td, tf], dtype=float)

def explosion_point_from_params(heading, speed, drop, fuse):
    """由参数算起爆点（水平匀速 + 自由落体）"""
    dropPos, uavV = UavStateHorizontal(drop, FY1_INIT, speed, heading)
    expl_xy = dropPos[:2] + uavV[:2] * fuse
    expl_z = dropPos[2] - 0.5 * g * (fuse ** 2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=float)

# 目标点云缓存，避免重复构建
_TARGET_PTS_CACHE: Dict[Tuple[int,int], np.ndarray] = {}
def get_target_points(nphi: int, nz: int) -> np.ndarray:
    key = (int(nphi), int(nz))
    pts = _TARGET_PTS_CACHE.get(key, None)
    if pts is None:
        pts = PreCalCylinderPoints(nphi, nz, dtype=float)
        _TARGET_PTS_CACHE[key] = pts
    return pts

def quick_hard_filters(heading, speed, drop, fuse):
    """
    便宜的硬过滤：爆点落地/时间窗为空 直接淘汰。
    返回 (是否通过, t0, t1, hit_time, expl_pos)
    """
    expl_pos = explosion_point_from_params(heading, speed, drop, fuse)
    t0 = drop + fuse
    t1 = min(t0 + SMOG_EFFECT_TIME, HIT_TIME)
    if expl_pos[2] <= 0.0:
        return False, t0, t1, HIT_TIME, expl_pos
    if t1 <= t0:
        return False, t0, t1, HIT_TIME, expl_pos
    return True, t0, t1, HIT_TIME, expl_pos

# =========================
# 2) 遮蔽评估（两阶段精度）
# =========================

def evaluate_occlusion(heading, speed, drop, fuse,
                       dt=0.002, nphi=480, nz=9,
                       margin=EPS, block=8192) -> Tuple[float, np.ndarray, float]:
    """
    评估遮蔽总时长（严格圆锥判据）。
    返回： (时长 seconds, 起爆点 expl_pos, 导弹命中时间 hit_time)
    """
    ok, t0, t1, hit_time, expl_pos = quick_hard_filters(heading, speed, drop, fuse)
    if not ok:
        return 0.0, expl_pos, hit_time

    t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)
    if len(t_grid) == 0:
        return 0.0, expl_pos, hit_time
    pts = get_target_points(nphi, nz)  # 使用缓存

    mask = np.zeros(len(t_grid), dtype=bool)
    for j, t in enumerate(t_grid):
        m_pos, _ = MissileState(float(t), M1_INIT)
        c_pos = np.array([expl_pos[0], expl_pos[1],
                          expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype=float)
        mask[j] = ConeAllPointsIn(m_pos, c_pos, pts, rCloud=SMOG_R, margin=margin, block=block)
    seconds = float(np.count_nonzero(mask) * dt)
    return seconds, expl_pos, hit_time

# 并行池包装
def _eval_tuple(args):
    (heading, speed, drop, fuse, dt, nphi, nz, margin, block) = args
    seconds, expl, hit = evaluate_occlusion(heading, speed, drop, fuse, dt, nphi, nz, margin, block)
    return seconds, (heading, speed, drop, fuse), expl, hit

# =========================
# 3) 种子：LHS / 物理热启动
# =========================

def latin_hypercube(n_samples: int) -> np.ndarray:
    """4维参数空间 LHS（遵守新边界），并轻微抬升 drop 下界避免 0 陷阱"""
    rng = np.random.default_rng()
    u = (rng.random((n_samples,4)) + np.arange(n_samples)[:,None]) / n_samples
    rng.shuffle(u, axis=0)
    headings = HEADING_MIN + u[:,0]*(HEADING_MAX - HEADING_MIN)
    speeds   = SPEED_MIN   + u[:,1]*(SPEED_MAX   - SPEED_MIN)
    drops    = DROP_MIN    + u[:,2]*(DROP_MAX    - DROP_MIN)
    fuses    = FUSE_MIN    + u[:,3]*(FUSE_MAX    - FUSE_MIN)
    drops = np.maximum(drops, 0.02 * (DROP_MAX - DROP_MIN))
    X = np.column_stack([headings, speeds, drops, fuses])
    X = np.array([clamp_params(x) for x in X], dtype=float)
    return X

def heuristic_seeds(k: int = 8) -> List[np.ndarray]:
    """物理可行的热启动"""
    rng = np.random.default_rng()
    seeds = []
    heading_to_origin = math.atan2(-FY1_INIT[1], -FY1_INIT[0])
    base_list = [
        np.array([heading_to_origin, 120.0, 1.5, 3.6], dtype=float),
        np.array([heading_to_origin+0.1, 110.0, 1.2, 4.0], dtype=float),
        np.array([heading_to_origin-0.1, 130.0, 1.8, 3.0], dtype=float),
        np.array([heading_to_origin, 140.0, 2.5, 2.0], dtype=float),
        np.array([heading_to_origin, 100.0, 3.0, 2.5], dtype=float),
    ]
    for b in base_list:
        seeds.append(clamp_params(b))
    while len(seeds) < k:
        d = rng.uniform(-0.15, 0.15, size=4) * np.array([1.0, 10.0, 0.8, 0.8])
        x = seeds[rng.integers(0, len(seeds))] + d
        seeds.append(clamp_params(x))
    return seeds[:k]

# =========================
# 4) 模式搜索 / DE / PSO / SA（并行退火）
# =========================

def pattern_search(x0: np.ndarray, steps: np.ndarray, eval_kwargs: dict,
                   max_iter: int = 60, shrink: float = 0.5, workers: int = None
                   ) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    x = clamp_params(x0.copy())
    best_val, best_expl, best_hit = evaluate_occlusion(*x, **eval_kwargs)
    dirs = np.eye(4, dtype=float)
    hist = [best_val]

    for _ in range(max_iter):
        cands = [x + steps*d for d in dirs] + [x - steps*d for d in dirs]
        tasks = [(clamp_params(c)[0], clamp_params(c)[1], clamp_params(c)[2], clamp_params(c)[3],
                  eval_kwargs.get("dt"), eval_kwargs.get("nphi"), eval_kwargs.get("nz"),
                  eval_kwargs.get("margin"), eval_kwargs.get("block")) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xcand, expl, hit = fut.result()
                vals.append((sec, np.array(xcand, dtype=float), expl, hit))
        improved = False
        for sec, xc, expl, hit in vals:
            if sec > best_val + 1e-12:
                best_val, best_expl, best_hit = sec, expl, hit
                x = xc
                improved = True
        hist.append(best_val)
        if improved:
            continue
        steps *= shrink
        if np.all(steps < np.array([1e-4, 0.05, 1e-3, 1e-3])):
            break
    return best_val, x, best_expl, best_hit, hist

def de_opt(pop: int, iters: int, eval_kwargs: dict, workers: int = None,
           F: float = 0.7, CR: float = 0.9, strategy: str = "best1bin",
           init_seeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    rng = np.random.default_rng()
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

    # 初评
    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    scores, expls = np.zeros(len(X), dtype=float), [None]*len(X)
    hit_time = HIT_TIME
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            scores[i] = sec
            expls[i] = expl
            hit_time = hit
    gbest_idx = int(np.argmax(scores))
    gbest, gbest_score, gbest_expl = X[gbest_idx].copy(), float(scores[gbest_idx]), expls[gbest_idx]
    hist = [gbest_score]

    percent_step = max(1, iters // 20)  # 每 ~5% 打印一次
    for it in range(iters):
        newX = np.zeros_like(X)
        for i in range(len(X)):
            idxs = list(range(len(X))); idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            if strategy == "best1bin":
                base = gbest
                mutant = base + F*(X[r1] - X[r2])
            else:
                base = X[r1]
                mutant = base + F*(X[r2] - X[r3])
            trial = np.empty(4, dtype=float)
            jrand = rng.integers(0, 4)
            for j in range(4):
                trial[j] = mutant[j] if (rng.random() < CR or j == jrand) else X[i][j]
            newX[i] = clamp_params(trial)

        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in newX]
        newScores, newExpls = np.zeros(len(X), dtype=float), [None]*len(X)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                newScores[i] = sec
                newExpls[i] = expl
                hit_time = hit
        for i in range(len(X)):
            if newScores[i] >= scores[i]:
                X[i], scores[i], expls[i] = newX[i], newScores[i], newExpls[i]
        gbest_idx = int(np.argmax(scores))
        if scores[gbest_idx] > gbest_score:
            gbest_score = float(scores[gbest_idx])
            gbest = X[gbest_idx].copy()
            gbest_expl = expls[gbest_idx]
        hist.append(gbest_score)
        if (it+1) % percent_step == 0 or it == iters-1:
            print(f"[DE] {int((it+1)/iters*100)}%  gbest={gbest_score:.6f}")
    return gbest_score, gbest, gbest_expl, hit_time, hist

def pso_opt(pop: int, iters: int, eval_kwargs: dict, workers: int = None,
            w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
            init_seeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    rng = np.random.default_rng()
    X, V = [], []
    if init_seeds:
        for s in init_seeds:
            x = clamp_params(s); X.append(x); V.append(np.zeros(4, dtype=float))
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
    Pbest, Pscore = X.copy(), np.zeros(len(X), dtype=float)
    expls = [None]*len(X)
    hit_time = HIT_TIME
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
    hist = [Gscore]

    percent_step = max(1, iters // 20)  # 每 ~5% 打印一次
    for it in range(iters):
        for i in range(len(X)):
            r1, r2 = rng.random(4), rng.random(4)
            V[i] = w*V[i] + c1*r1*(Pbest[i]-X[i]) + c2*r2*(Gbest-X[i])
            X[i] = clamp_params(X[i] + V[i])
            for j,(mn,mx) in enumerate([(HEADING_MIN,HEADING_MAX),
                                        (SPEED_MIN,SPEED_MAX),
                                        (DROP_MIN,DROP_MAX),
                                        (FUSE_MIN,FUSE_MAX)]):
                if j==0:  # 角度不反弹
                    continue
                if X[i][j] <= mn or X[i][j] >= mx:
                    V[i][j] *= -0.5
        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                if sec > Pscore[i]:
                    Pscore[i], Pbest[i] = sec, X[i].copy()
                if sec > Gscore:
                    Gscore, Gbest, Gexpl = sec, X[i].copy(), expl
                hit_time = hit
        hist.append(Gscore)
        if (it+1) % percent_step == 0 or it == iters-1:
            print(f"[PSO] {int((it+1)/iters*100)}%  gbest={Gscore:.6f}")
    return Gscore, Gbest, Gexpl, hit_time, hist

def sa_opt(eval_kwargs: dict,
           iters: int = 8000,
           n_chains: int = 8,
           batch_size: int = 32,
           T0: float = 1.0,
           Tend: float = 1e-3,
           workers: int = None,
           no_improve_patience: int = 2000
           ) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    并行退火（多链 + 批量候选 + 并行评估）
    返回：best_score, best_x, best_expl, hit_time, convergence_history
    """
    rng = np.random.default_rng()

    # 初始化多条链
    inits = heuristic_seeds(k=n_chains)
    X = np.array([clamp_params(x) for x in inits], dtype=float)
    F = np.zeros(n_chains, dtype=float)
    EXPLS = [None]*n_chains
    hit_time = HIT_TIME

    # 初评（并行）
    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            F[i] = sec; EXPLS[i] = expl; hit_time = hit

    # 全局最优
    g_idx = int(np.argmax(F))
    best_f = float(F[g_idx]); best_x = X[g_idx].copy(); best_expl = EXPLS[g_idx]
    hist = [best_f]
    last_improve = 0

    percent_step = max(1, iters // 20)

    for k in range(1, iters+1):
        T = T0 * (Tend/T0) ** (k/iters)
        # 为每条链生成 batch_size 个候选，并行评估
        cand_tasks = []
        cand_index_map = []  # (chain_id, local_j)
        for i in range(n_chains):
            # 自适应步长：随温度缩放，速度维额外收缩，时间维适中
            step = np.array([0.25, 8.0, 0.8, 0.8]) * max(T, 1e-3)
            for j in range(batch_size):
                cand = clamp_params(X[i] + rng.normal(0.0, 1.0, size=4) * step)
                cand_tasks.append((cand[0], cand[1], cand[2], cand[3],
                                   eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                                   eval_kwargs["margin"], eval_kwargs["block"]))
                cand_index_map.append((i, j))

        cand_scores = np.zeros(len(cand_tasks), dtype=float)
        cand_expls = [None]*len(cand_tasks)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): idx for idx,t in enumerate(cand_tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                idx = futs[fut]
                cand_scores[idx] = sec
                cand_expls[idx] = expl
                hit_time = hit

        # 每条链根据自己的 batch 做一次 Metropolis 选择
        for i in range(n_chains):
            start = i*batch_size
            end = (i+1)*batch_size
            scores_i = cand_scores[start:end]
            expls_i = cand_expls[start:end]

            # 若存在改进，优先选择改进中分数最高的
            better_mask = scores_i >= F[i] + 1e-12
            if np.any(better_mask):
                j = start + int(np.argmax(scores_i))  # 直接拿该批中最好
                X[i] = np.array([cand_tasks[j][0], cand_tasks[j][1], cand_tasks[j][2], cand_tasks[j][3]], dtype=float)
                EXPLS[i] = expls_i[j - start]
                F[i] = scores_i.max()
            else:
                # 玻尔兹曼按概率接受（越接近当前 f 越可能被接受）
                # 注意这里是“收益”最大化问题，构造能量 E = -score
                delta = scores_i - F[i]
                probs = np.exp(delta / max(T, 1e-9))
                probs = probs / (probs.sum() + 1e-12)
                pick = np.random.choice(batch_size, p=probs)
                j = start + pick
                X[i] = np.array([cand_tasks[j][0], cand_tasks[j][1], cand_tasks[j][2], cand_tasks[j][3]], dtype=float)
                EXPLS[i] = expls_i[pick]
                F[i] = scores_i[pick]

        # 更新全局最优
        g_idx = int(np.argmax(F))
        if F[g_idx] > best_f + 1e-12:
            best_f = float(F[g_idx]); best_x = X[g_idx].copy(); best_expl = EXPLS[g_idx]
            last_improve = k

        if (k % percent_step == 0) or (k == iters):
            hist.append(best_f)
            print(f"[SA] {int(k/iters*100)}%  best={best_f:.6f}")

        # 早停或“踢一脚”避免卡 0
        if k - last_improve >= no_improve_patience:
            # 若全体分数都很低（例如≈0），重置最差链
            worst_i = int(np.argmin(F))
            X[worst_i] = heuristic_seeds(k=1)[0]
            F[worst_i], EXPLS[worst_i], _ = evaluate_occlusion(*X[worst_i], **eval_kwargs)
            last_improve = k  # 记录一次重置
            # 如果仍长期无提升就跳出
            if F[worst_i] <= 0 and best_f <= 0:
                print(f"[SA] early stop at {k}, stagnation. best={best_f:.6f}")
                break

    return best_f, best_x, best_expl, hit_time, hist

# =========================
# 5) HYBRID：LHS -> Pattern -> DE -> 终评
# =========================

def hybrid_opt(pop: int, iters: int, workers: int, topk: int,
               dt_coarse: float, nphi_coarse: int, nz_coarse: int,
               dt_final: float, nphi_final: int, nz_final: int,
               probe: int = 0):
    """返回 best_val, best_x, best_expl, hit_time, top_list, hybrid_history"""
    print("[HYBRID] 阶段1：可行性探针 + LHS 粗筛")
    seeds = latin_hypercube(pop)
    seeds[:min(len(seeds), 6)] = heuristic_seeds(k=min(6, pop))  # 注入物理热启动
    if probe > 0:
        probers = heuristic_seeds(k=min(8, probe)) + list(latin_hypercube(max(0, probe-8)))
        probers = [clamp_params(p) for p in probers]
        tasks = [(p[0], p[1], p[2], p[3], dt_coarse, nphi_coarse, nz_coarse, EPS, 8192) for p in probers]
        feasible = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed({pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}):
                sec, *_ = fut.result()
                feasible += int(sec > 0)
        print(f"[HYBRID] Probe feasible count = {feasible}/{len(probers)}")

    tasks = [(row[0], row[1], row[2], row[3], dt_coarse, nphi_coarse, nz_coarse, EPS, 8192) for row in seeds]
    vals = []
    lhs_history = []
    best_so_far = 0.0
    stride = max(1, len(seeds) // 20)  # 每 5% 记录一次

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            vals.append((sec, np.array(x), expl, hit))
            if sec > best_so_far: best_so_far = sec
            done += 1
            if done % stride == 0 or done == len(seeds):
                lhs_history.append(best_so_far)
                print(f"[HYBRID] {int(100*done/len(seeds))}%  LHS best_so_far={best_so_far:.6f}")

    vals.sort(key=lambda x: x[0], reverse=True)
    top = vals[:topk]
    print(f"[HYBRID] 粗筛 Top-{topk} 最佳 = {top[0][0]:.6f} s")

    print("[HYBRID] 阶段2：Top-K Pattern 精修（跳过 0 分种子，仅保留至多 1 个兜底）")
    refined, pattern_each_histories, pattern_global_history = [], [], []
    global_best_after_patterns = 0.0
    zero_used = 0
    ZERO_LIMIT = 1
    TOL = 1e-12

    for rank,(sec0, x0, expl0, hit0) in enumerate(top, start=1):
        if sec0 <= TOL:
            if zero_used >= ZERO_LIMIT:
                continue
            zero_used += 1
            print(f"  -> Pattern 起点#{rank} 为 0，保留 1 个兜底样本")
        else:
            print(f"  -> Pattern 起点#{rank}  初值={sec0:.6f}")
        steps = np.array([0.15, 5.0, 0.6, 0.6], dtype=float)
        best_sec, best_x, best_expl, best_hit, hist = pattern_search(
            x0, steps,
            eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
            max_iter=60, shrink=0.6, workers=workers
        )
        refined.append((best_sec, best_x, best_expl, best_hit))
        pattern_each_histories.append(hist)
        global_best_after_patterns = max(global_best_after_patterns, best_sec)
        pattern_global_history.append(global_best_after_patterns)
        if best_sec > TOL:
            print(f"    Pattern 完成：{best_sec:.6f} s")

    refined.sort(key=lambda x: x[0], reverse=True)
    seeds2 = [r[1] for r in refined[:min(len(refined), max(4, topk//2))]]
    if not seeds2:
        seeds2 = heuristic_seeds(k=4)  # 兜底

    print("[HYBRID] 阶段3：DE 全局收敛（粗评口径）")
    best_de, x_de, expl_de, hit_de, de_history = de_opt(
        pop=pop, iters=iters,
        eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
        workers=workers, F=0.7, CR=0.9, strategy="best1bin",
        init_seeds=seeds2
    )
    print(f"[HYBRID] DE 粗精度结果：{best_de:.6f} s")

    print("[HYBRID] 阶段4：终评（高精度复评）")
    candidates = [(best_de, x_de, expl_de, hit_de)] + refined[:min(10, len(refined))]
    # 去重
    uniq, seen = [], set()
    for c in candidates:
        arr = c[1]
        k = (round(arr[0],3), round(arr[1],2), round(arr[2],3), round(arr[3],3))
        if k not in seen:
            seen.add(k); uniq.append(c)

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

    hybrid_history = dict(lhs=lhs_history,
                          pattern_each=pattern_each_histories,
                          pattern_global=pattern_global_history,
                          de=de_history)
    return best[0], best[1], best[2], best[3], finals[:min(10,len(finals))], hybrid_history

# =========================
# 6) 报告与可视化
# =========================

def save_report(filename: str, algo: str,
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
    try:
        if not history:
            print(f"[Plot] {algorithm_name}: 无历史可绘图。"); return
        fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
        iters = list(range(len(history)))
        ax.plot(iters, history, linewidth=2, alpha=0.9, label='Best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Occlusion Duration (s)')
        ax.set_title(f'{algorithm_name.upper()} Convergence')
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        out = f'Q2_{algorithm_name}_Convergence.png'
        fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"[Plot] Saved {out}")
    except Exception as e:
        print(f"[Warning] Failed to generate {algorithm_name} plot: {e}")

def generate_hybrid_convergence_plot(hybrid_history: Dict, best_value: float,
                                     coarse_cfg: Dict, final_cfg: Dict):
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        lhs = hybrid_history.get("lhs", [])
        ax = axes[0]
        if lhs:
            ax.plot(range(1, len(lhs)+1), lhs, linewidth=2, label='LHS best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_title(f"LHS Screening (coarse: dt={coarse_cfg['dt']}, nphi={coarse_cfg['nphi']}, nz={coarse_cfg['nz']})")
        ax.set_xlabel('Checkpoint (~5%)'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3); ax.legend()

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
        ax.set_xlabel('Pattern Iteration / Seed Index'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)

        ax = axes[2]
        de_hist = hybrid_history.get("de", [])
        if de_hist:
            ax.plot(range(len(de_hist)), de_hist, linewidth=2, label='DE best-so-far')
        ax.axhline(y=best_value, linestyle='--', linewidth=2, label=f'Final Best: {best_value:.6f}s')
        ax.set_title(f"DE Global Search (coarse: dt={coarse_cfg['dt']}, nphi={coarse_cfg['nphi']}, nz={coarse_cfg['nz']}; "
                     f"final: dt={final_cfg['dt']}, nphi={final_cfg['nphi']}, nz={final_cfg['nz']})")
        ax.set_xlabel('Generation'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha=0.3); ax.legend()

        fig.tight_layout()
        out = 'Q2_HYBRID_Convergence.png'
        fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
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

    # SA 特有参数
    ap.add_argument("--sa-iters", type=int, default=8000, help="SA 最大步数（含早停）")
    ap.add_argument("--sa-batch", type=int, default=32, help="SA 每条链每步的候选数量")
    ap.add_argument("--sa-chains", type=int, default=8, help="SA 并行链条数")

    ap.add_argument("--probe", type=int, default=0, help="可行性探针样本数(0关闭)")

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
    sa_chains = max(1, args.sa_chains)
    if workers is None:
        # 若用户未指定，尽量让 SA 的任务数（chains*batch）足以吃满 CPU
        try:
            cpu = os.cpu_count() or 4
        except Exception:
            cpu = 4
        workers = cpu

    # 多进程避免 BLAS 线程内卷
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    algos = [args.algo] if args.algo != "all" else ["hybrid","de","pso","sa","pattern"]

    overall_best = -1.0
    overall_rec = None  # (algo, best_val, best_x, best_expl, hit_time)

    for algo in algos:
        print("="*90)
        print(f"[Q2] Start algo={algo} | pop={args.pop} iters={args.iter} | workers={workers or 'auto'}")
        print(f"[Q2] coarse(dt={args.dt_coarse}, nphi={args.nphi_coarse}, nz={args.nz_coarse}) "
              f"-> final(dt={args.dt_final}, nphi={args.nphi_final}, nz={args.nz_final})")
        print("="*90)

        tA = time.time()
        best_val = 0.0
        best_x = np.array([0,0,0,0], dtype=float)
        best_expl = np.array([0,0,0], dtype=float)
        hit_time = HIT_TIME
        history = None
        hybrid_history = None

        if algo == "hybrid":
            best_val, best_x, best_expl, hit_time, _, hybrid_history = hybrid_opt(
                pop=args.pop, iters=args.iter, workers=workers or (os.cpu_count() or 1),
                topk=args.topk,
                dt_coarse=args.dt_coarse, nphi_coarse=args.nphi_coarse, nz_coarse=args.nz_coarse,
                dt_final=args.dt_final, nphi_final=args.nphi_final, nz_final=args.nz_final,
                probe=args.probe
            )
        elif algo == "de":
            seeds0 = heuristic_seeds(k=min(8, args.pop))
            best_val, best_x, best_expl, hit_time, history = de_opt(
                pop=args.pop, iters=args.iter,
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                F=0.7, CR=0.9, strategy="best1bin", init_seeds=seeds0
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        elif algo == "pso":
            seeds0 = heuristic_seeds(k=min(8, args.pop))
            best_val, best_x, best_expl, hit_time, history = pso_opt(
                pop=args.pop, iters=args.iter,
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                w=0.5, c1=1.5, c2=1.5, init_seeds=seeds0
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        elif algo == "sa":
            best_val, best_x, best_expl, hit_time, history = sa_opt(
                eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
                iters=args.sa_iters,
                n_chains=sa_chains,
                batch_size=args.sa_batch,
                T0=1.0, Tend=1e-3,
                workers=workers or (os.cpu_count() or 1),
                no_improve_patience=max(1000, args.sa_iters//3)
            )
            sec_tmp, expl_tmp, hit_tmp = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
            best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
        else:  # pattern
            x0 = heuristic_seeds(k=1)[0]
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
            "workers": workers or os.cpu_count() or 1,
            "speed_range": [SPEED_MIN, SPEED_MAX],
            "search_space": {"heading":"[0,2pi)","speed":"[70,140]","drop":f"[0,{DROP_MAX:.3f}]","fuse":"[0,18]"}
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

        if best_val > overall_best:
            overall_best = best_val
            overall_rec = (algo, best_val, best_x, best_expl, hit_time)

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
