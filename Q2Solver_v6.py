# -*- coding: utf-8 -*-
"""
Q2_Optimizer.py  (reparam v3)
问题二：优化 FY1 的航向角、速度、起爆时刻 t0、引信延时 τ，使对 M1 的遮蔽时长最大化。
重参化：变量由 (θ, v, drop, fuse) -> (θ, v, t0, tau)，其中 t0 = drop + tau，drop = t0 - tau ≥ 0。
并列打破：遮蔽时长近并列时，优先 tau 更小（云更高），再优先 drop 更大（更符合“先飞再投”）。

多算法：Hybrid / DE / PSO / SA / Pattern
- 并行评估（ProcessPoolExecutor）
- 两阶段精度（粗筛 / 终评）
- 兼容 Q1Solver 的几何判据（导入失败自动回退）

示例（推荐 Hybrid）：
python Q2_Optimizer.py --algo hybrid --pop 64 --iter 60 --workers auto --topk 12 \
    --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 \
    --dt-final 0.0005 --nphi-final 960 --nz-final 13
"""

import os
import math
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- 确保在导入 pyplot 前设置无界面后端 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# 0) 尝试导入 Q1Solver（首选）；失败则回退到内置实现
# =========================
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
    print("[Q2] 已成功导入 Q1Solver 中的常量与函数。")
except Exception as e:
    print(f"[Q2] 未能导入 Q1Solver，使用内置回退实现。原因：{e}")
    # ------- 回退实现（与你之前 Q1 版本等价） -------
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
# 1) 变量范围 & 公共工具（重参化）
# =========================

HEADING_MIN, HEADING_MAX = 0.0, 2.0 * math.pi
SPEED_MIN, SPEED_MAX = 0.0, 300.0
HIT_TIME = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

TAU_MIN, TAU_MAX = 0.0, 18.0
T0_MIN,  T0_MAX  = 0.0, max(0.0, HIT_TIME - 1e-3)

def clamp_params(x):
    """
    边界处理（θ, v, t0, tau），并保证 drop = t0 - tau >= 0（若违反则把 tau 截到 t0）。
    """
    h, v, t0, tau = x
    h   = h % (2.0 * math.pi)
    v   = float(np.clip(v,   SPEED_MIN, SPEED_MAX))
    t0  = float(np.clip(t0,  T0_MIN,    T0_MAX))
    tau = float(np.clip(tau, TAU_MIN,   TAU_MAX))
    if t0 < tau:
        tau = t0
    return np.array([h, v, t0, tau], dtype=float)

def explosion_point_from_params(h, v, t0, tau):
    """
    由 (θ, v, t0, τ) 计算起爆点：
      drop = t0 - tau；
      drop 时刻 FY1 位姿 dropPos, 速度 uavV；
      起爆点 XY = dropPos[:2] + uavV[:2] * tau； Z = z0 - 0.5*g*tau^2。
    """
    drop = max(0.0, t0 - tau)
    dropPos, uavV = UavStateHorizontal(drop, FY1_INIT, v, h)
    expl_xy = dropPos[:2] + uavV[:2] * tau
    expl_z  = dropPos[2]  - 0.5 * g * (tau ** 2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=float), drop

def quick_hard_filters(h, v, t0, tau):
    """
    便宜的硬过滤：起爆点落地/时间窗为空 -> 直接淘汰。
    返回 (通过?, t0, t1, hit_time, expl_pos, drop)
    """
    expl_pos, drop = explosion_point_from_params(h, v, t0, tau)
    t1 = min(t0 + SMOG_EFFECT_TIME, HIT_TIME)
    if expl_pos[2] <= 0.0:
        return False, t0, t1, HIT_TIME, expl_pos, drop
    if t1 <= t0:
        return False, t0, t1, HIT_TIME, expl_pos, drop
    return True, t0, t1, HIT_TIME, expl_pos, drop

def better(a_score, a_tau, a_drop, b_score, b_tau, b_drop, tol=1e-6):
    """并列打破：先看得分，再看 tau (小优先)，再看 drop (大优先)。"""
    if a_score > b_score + tol: return True
    if b_score > a_score + tol: return False
    if a_tau < b_tau - 1e-6: return True
    if b_tau < a_tau - 1e-6: return False
    return (a_drop > b_drop + 1e-6)

# =========================
# 2) 遮蔽评估（两阶段精度）
# =========================

def evaluate_occlusion(h, v, t0, tau,
                       dt=0.002, nphi=480, nz=9,
                       margin=EPS, block=8192) -> Tuple[float, np.ndarray, float, float]:
    """
    评估遮蔽总时长（严格圆锥判据）。用于粗筛/终评。
    返回： (时长 seconds, 起爆点 expl_pos, 导弹命中时间 hit_time, drop)
    """
    ok, t0, t1, hit_time, expl_pos, drop = quick_hard_filters(h, v, t0, tau)
    if not ok:
        return 0.0, expl_pos, hit_time, drop

    t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)
    pts = PreCalCylinderPoints(nphi, nz, dtype=float)

    mask = np.zeros(len(t_grid), dtype=bool)
    for j, t in enumerate(t_grid):
        m_pos, _ = MissileState(float(t), M1_INIT)
        c_pos = np.array([expl_pos[0], expl_pos[1],
                          expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype=float)
        mask[j] = ConeAllPointsIn(m_pos, c_pos, pts, rCloud=SMOG_R, margin=margin, block=block)
    seconds = float(np.count_nonzero(mask) * dt)
    return seconds, expl_pos, hit_time, drop

def _eval_tuple(args):
    (h, v, t0, tau, dt, nphi, nz, margin, block) = args
    sec, expl, hit, drop = evaluate_occlusion(h, v, t0, tau, dt, nphi, nz, margin, block)
    return sec, (h, v, t0, tau), expl, hit, drop

# =========================
# 3) 采样器（LHS）与局部搜索（Pattern）
# =========================

def latin_hypercube(n_samples: int) -> np.ndarray:
    """在 4 维参数空间 (θ,v,t0,tau) 做简易 LHS。保证 t0 >= tau。"""
    rng = np.random.default_rng()
    u = (rng.random((n_samples,4)) + np.arange(n_samples)[:,None]) / n_samples
    rng.shuffle(u, axis=0)
    headings = HEADING_MIN + u[:,0]*(HEADING_MAX - HEADING_MIN)
    speeds   = SPEED_MIN   + u[:,1]*(SPEED_MAX   - SPEED_MIN)
    t0s      = T0_MIN      + u[:,2]*(T0_MAX      - T0_MIN)
    taus     = TAU_MIN     + u[:,3]*(TAU_MAX     - TAU_MIN)
    mask = t0s < taus
    t0s[mask] = taus[mask]
    return np.column_stack([headings, speeds, t0s, taus])

def pattern_search(x0: np.ndarray,
                   steps: np.ndarray,
                   eval_kwargs: dict,
                   max_iter: int = 60,
                   shrink: float = 0.5,
                   workers: int = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    x = clamp_params(x0.copy())
    best_val, best_expl, best_hit, best_drop = evaluate_occlusion(*x, **eval_kwargs)
    dirs = np.eye(4, dtype=float)
    history = [best_val]

    for it in range(max_iter):
        cands = []
        for d in dirs:
            cands.append(x + steps * d)
            cands.append(x - steps * d)
        tasks = [(clamp_params(c)[0], clamp_params(c)[1], clamp_params(c)[2], clamp_params(c)[3],
                  eval_kwargs.get("dt"), eval_kwargs.get("nphi"), eval_kwargs.get("nz"),
                  eval_kwargs.get("margin"), eval_kwargs.get("block")) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xcand, expl, hit, drop = fut.result()
                vals.append((sec, np.array(xcand, dtype=float), expl, hit, drop))
        improved = False
        for sec, xc, expl, hit, drop in vals:
            tau_c = xc[3]
            drop_c = max(0.0, xc[2] - xc[3])
            if better(sec, tau_c, drop_c, best_val, x[3], best_drop):
                best_val, best_expl, best_hit = sec, expl, hit
                best_drop = drop_c
                x = xc
                improved = True

        history.append(best_val)
        if improved:
            continue
        steps *= shrink
        if np.all(steps < np.array([1e-4, 0.05, 1e-3, 1e-3])):
            break
    return best_val, x, best_expl, best_hit, history

# =========================
# 4) 元启发式：DE / PSO / SA（含并列打破）
# =========================

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
            rng.uniform(SPEED_MIN,  SPEED_MAX),
            rng.uniform(T0_MIN,     T0_MAX),
            rng.uniform(TAU_MIN,    TAU_MAX),
        ], dtype=float)))
    X = np.array(X, dtype=float)

    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    scores, expls, drops = np.zeros(pop, dtype=float), [None]*pop, np.zeros(pop, dtype=float)
    hit_time = HIT_TIME
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit, drop = fut.result()
            i = futs[fut]
            scores[i] = sec
            expls[i] = expl
            drops[i] = drop
            hit_time = hit
    best_idx = int(np.argmax(scores))
    gbest, gbest_score, gbest_expl, gbest_drop = X[best_idx].copy(), float(scores[best_idx]), expls[best_idx], drops[best_idx]
    history = [gbest_score]

    for it in range(iters):
        newX = np.zeros_like(X)
        newScores = np.zeros_like(scores)
        newExpls = [None]*pop
        newDrops = np.zeros_like(drops)

        for i in range(pop):
            idxs = list(range(pop))
            idxs.remove(i)
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
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = X[i][j]
            trial = clamp_params(trial)
            newX[i] = trial

        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in newX]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit, drop = fut.result()
                i = futs[fut]
                newScores[i] = sec
                newExpls[i]  = expl
                newDrops[i]  = drop
                hit_time = hit

        for i in range(pop):
            old = X[i]
            old_score, old_tau, old_drop = scores[i], old[3], drops[i]
            cand = newX[i]
            cand_score, cand_tau, cand_drop = newScores[i], cand[3], newDrops[i]
            if better(cand_score, cand_tau, cand_drop, old_score, old_tau, old_drop):
                X[i] = cand
                scores[i] = cand_score
                expls[i] = newExpls[i]
                drops[i] = cand_drop

        best_idx = int(np.argmax(scores))
        if better(scores[best_idx], X[best_idx][3], drops[best_idx],
                  gbest_score, gbest[3], gbest_drop):
            gbest_score = float(scores[best_idx])
            gbest = X[best_idx].copy()
            gbest_expl = expls[best_idx]
            gbest_drop = drops[best_idx]

        history.append(gbest_score)
        if it % max(1, iters//10) == 0 or it == iters-1:
            print(f"[DE] iter {it+1}/{iters}  gbest={gbest_score:.6f}")

    return gbest_score, gbest, gbest_expl, hit_time, history

def pso_opt(pop: int, iters: int, eval_kwargs: dict, workers: int = None,
            w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
            init_seeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    rng = np.random.default_rng()
    X, V = [], []
    if init_seeds:
        for s in init_seeds:
            x = clamp_params(s)
            X.append(x)
            V.append(np.zeros(4, dtype=float))
    while len(X) < pop:
        x = np.array([rng.uniform(HEADING_MIN, HEADING_MAX),
                      rng.uniform(SPEED_MIN,  SPEED_MAX),
                      rng.uniform(T0_MIN,     T0_MAX),
                      rng.uniform(TAU_MIN,    TAU_MAX)], dtype=float)
        X.append(clamp_params(x))
        V.append(rng.uniform(-1,1,size=4))
    X, V = np.array(X, dtype=float), np.array(V, dtype=float)

    tasks = [(x[0], x[1], x[2], x[3],
              eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
              eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
    Pbest = X.copy()
    Pscore = np.zeros(pop, dtype=float)
    expls = [None]*pop
    drops = np.zeros(pop, dtype=float)
    hit_time = HIT_TIME
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit, drop = fut.result()
            i = futs[fut]
            Pscore[i] = sec
            expls[i] = expl
            drops[i] = drop
            hit_time = hit
    g_idx = int(np.argmax(Pscore))
    Gbest, Gscore, Gexpl, Gdrop = X[g_idx].copy(), float(Pscore[g_idx]), expls[g_idx], drops[g_idx]
    history = [Gscore]

    for it in range(iters):
        for i in range(pop):
            r1, r2 = rng.random(4), rng.random(4)
            V[i] = w*V[i] + c1*r1*(Pbest[i]-X[i]) + c2*r2*(Gbest-X[i])
            X[i] = clamp_params(X[i] + V[i])
            for j,(mn,mx) in enumerate([(HEADING_MIN,HEADING_MAX),
                                        (SPEED_MIN,SPEED_MAX),
                                        (T0_MIN,T0_MAX),
                                        (TAU_MIN,TAU_MAX)]):
                if j==0:
                    continue
                if X[i][j] <= mn or X[i][j] >= mx:
                    V[i][j] *= -0.5

        tasks = [(x[0], x[1], x[2], x[3],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for x in X]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit, drop = fut.result()
                i = futs[fut]
                if better(sec, xc[3], drop, Pscore[i], Pbest[i][3], max(0.0, Pbest[i][2]-Pbest[i][3])):
                    Pscore[i] = sec
                    Pbest[i] = X[i].copy()
                if better(sec, xc[3], drop, Gscore, Gbest[3], Gdrop):
                    Gscore = sec
                    Gbest = X[i].copy()
                    Gexpl = expl
                    Gdrop = drop
                hit_time = hit

        history.append(Gscore)
        if it % max(1, iters//10) == 0 or it == iters-1:
            print(f"[PSO] iter {it+1}/{iters}  gbest={Gscore:.6f}")
    return Gscore, Gbest, Gexpl, hit_time, history

def sa_opt(x0: np.ndarray, eval_kwargs: dict,
           T0: float = 1.0, Tend: float = 1e-3, iters: int = 20000,
           step_scale: np.ndarray = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    rng = np.random.default_rng()
    x = clamp_params(x0.copy())
    f, expl, hit, drop = evaluate_occlusion(*x, **eval_kwargs)
    best_f, best_x, best_expl, best_drop = f, x.copy(), expl, drop
    history = [best_f]

    for k in range(1, iters+1):
        T = T0 * (Tend/T0) ** (k/iters)
        if step_scale is None:
            step = np.array([0.02, 2.0, 0.5, 0.5]) * max(T, 1e-3)
        else:
            step = step_scale * max(T, 1e-3)
        cand = clamp_params(x + rng.normal(0.0, 1.0, size=4) * step)
        fc, explc, hitc, dropc = evaluate_occlusion(*cand, **eval_kwargs)
        if better(fc, cand[3], dropc, f, x[3], drop) or (rng.random() < math.exp((fc - f)/max(T,1e-9))):
            x, f, drop = cand, fc, dropc
            expl, hit = explc, hitc
            if better(f, x[3], drop, best_f, best_x[3], best_drop):
                best_f, best_x, best_expl, best_drop = f, x.copy(), expl, drop
        if k % 10 == 0:
            history.append(best_f)
        if k % max(1, iters//10) == 0 or k == iters:
            print(f"[SA] step {k}/{iters}  current={f:.6f}  best={best_f:.6f}")
    return best_f, best_x, best_expl, hit, history

# =========================
# 5) HYBRID：LHS 粗筛 -> Pattern -> DE -> 终评
# =========================

def hybrid_opt(pop: int, iters: int, workers: int,
               topk: int,
               dt_coarse: float, nphi_coarse: int, nz_coarse: int,
               dt_final: float, nphi_final: int, nz_final: int) -> Tuple[float, np.ndarray, np.ndarray, float, List[Tuple[float,np.ndarray,np.ndarray]]]:
    print("[HYBRID] 阶段1：LHS 粗筛")
    seeds = latin_hypercube(pop)
    tasks = [(row[0], row[1], row[2], row[3], dt_coarse, nphi_coarse, nz_coarse, EPS, 8192) for row in seeds]
    vals = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            sec, x, expl, hit, drop = fut.result()
            vals.append((sec, np.array(x), expl, hit, drop))
            done += 1
            if done % max(1, pop//10) == 0:
                print(f"    [HYBRID] LHS 进度 {int(100*done/pop)}%")
    vals.sort(key=lambda x: x[0], reverse=True)
    top = vals[:topk]
    print(f"[HYBRID] 粗筛 Top-{topk} 最佳 = {top[0][0]:.6f} s")

    print("[HYBRID] 阶段2：Top-K Pattern Search 精修")
    refined = []
    for rank,(sec0, x0, expl0, hit0, drop0) in enumerate(top, start=1):
        print(f"  -> Pattern Search 起点#{rank}  初值={sec0:.6f}")
        steps = np.array([0.1, 5.0, 1.0, 1.0], dtype=float)  # θ / v / t0 / τ
        best_sec, best_x, best_expl, best_hit, _hist = pattern_search(
            x0, steps,
            eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
            max_iter=60, shrink=0.6, workers=workers
        )
        refined.append((best_sec, best_x, best_expl, best_hit))
        print(f"    Pattern Search 完成：{best_sec:.6f} s")

    refined.sort(key=lambda x: x[0], reverse=True)
    seeds2 = [r[1] for r in refined[:min(len(refined), max(4, topk//2))]]

    print("[HYBRID] 阶段3：以局部精修种子启动 DE（粗精度）")
    best_de, x_de, expl_de, hit_de, _hist_de = de_opt(
        pop=pop, iters=iters,
        eval_kwargs=dict(dt=dt_coarse, nphi=nphi_coarse, nz=nz_coarse, margin=EPS, block=8192),
        workers=workers, F=0.7, CR=0.9, strategy="best1bin",
        init_seeds=seeds2
    )
    print(f"[HYBRID] DE 粗精度结果：{best_de:.6f} s")

    print("[HYBRID] 阶段4：终评（高精度） Top-N 复评")
    candidates = []
    candidates.append((best_de, x_de, expl_de, hit_de))
    for r in refined[:min(10, len(refined))]:
        candidates.append(r)
    # 去重
    uniq = []
    def keyz(x):
        arr = x[1]
        return (round(arr[0],3), round(arr[1],2), round(arr[2],3), round(arr[3],3))
    seen = set()
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
            sec, x, expl, hit, drop = fut.result()
            finals.append((sec, np.array(x), expl, hit, drop))
    finals.sort(key=lambda x: x[0], reverse=True)
    best = finals[0]
    top_list = [(sec, x, expl, hit) for (sec,x,expl,hit,_) in finals[:min(10,len(finals))]]
    return best[0], best[1], best[2], best[3], top_list

# =========================
# 6) 报告与可视化
# =========================

def save_report(filename: str,
                algo: str,
                best_val: float, best_x: np.ndarray, best_expl: np.ndarray, hit_time: float,
                top_list: List[Tuple[float,np.ndarray,np.ndarray]] = None,
                params: Dict = None):
    h, v, t0, tau = best_x
    drop = max(0.0, t0 - tau)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Q2 Optimization Results Report (Reparam: (θ,v,t0,τ))\n")
        f.write("="*80 + "\n\n")
        f.write("Calculation Parameters:\n")
        f.write("-"*40 + "\n")
        if params:
            for k,vp in params.items():
                f.write(f"{k}: {vp}\n")
        f.write(f"Algorithm: {algo}\n\n")

        f.write("Best Solution:\n")
        f.write("-"*40 + "\n")
        f.write(f"Occlusion duration = {best_val:.6f} seconds\n")
        f.write(f"Heading (rad) = {h:.9f}\n")
        f.write(f"Speed (m/s)  = {v:.9f}\n")
        f.write(f"t0 (explode time, s) = {t0:.9f}\n")
        f.write(f"tau (fuse delay, s)  = {tau:.9f}\n")
        f.write(f"drop = t0 - tau (s)  = {drop:.9f}\n")
        f.write(f"Explosion point = ({best_expl[0]:.6f}, {best_expl[1]:.6f}, {best_expl[2]:.6f})\n")
        f.write(f"Missile hit time ≈ {hit_time:.6f} seconds\n\n")

        if top_list:
            f.write("Top Candidates (final evaluation):\n")
            f.write("-"*40 + "\n")
            for i,(sec,x,expl,_) in enumerate(top_list, start=1):
                hh, vv, tt0, ttau = x
                ddrop = max(0.0, tt0 - ttau)
                f.write(f"#{i}: {sec:.6f}s | "
                        f"h={hh:.6f}, v={vv:.3f}, t0={tt0:.3f}, tau={ttau:.3f}, drop={ddrop:.3f} | "
                        f"expl=({expl[0]:.2f},{expl[1]:.2f},{expl[2]:.2f})\n")
        f.write("\n" + "="*80 + "\n")

def generate_convergence_plot(history, algorithm_name, best_value):
    """
    生成优化算法收敛曲线，可在无显示环境下保存。
    - history: 可迭代的数值序列；允许只有 1 个点
    - algorithm_name: 算法名
    - best_value: 最终最优值
    """
    try:
        if history is None:
            print("[Plot] history is None, skip plotting.")
            return
        # 尝试将 history 转为 1D numpy 数组
        hist = np.array(list(history), dtype=float).flatten()
        if hist.size == 0:
            print("[Plot] history is empty, skip plotting.")
            return

        # 构建图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(hist.size)
        ax1.plot(x, hist, linewidth=2, alpha=0.9, label='Best-so-far')
        ax1.axhline(y=float(best_value), color='r', linestyle='--', linewidth=1.8,
                    label=f'Final Best: {best_value:.6f}s')
        ax1.set_xlabel('Iteration / Generation')
        ax1.set_ylabel('Occlusion Duration (s)')
        ax1.set_title(f'Q2 Convergence - {algorithm_name.upper()}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # 改进曲线（若至少两点）
        if hist.size >= 2:
            init = hist[0]
            improvement = hist - init
            ax2 = ax1.twinx()
            ax2.plot(x, improvement, linewidth=1.5, alpha=0.6, color='g', label='Improvement vs. init')
            ax2.set_ylabel('Improvement (s)')
            ax2.legend(loc='lower right')

        fig.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"Q2_{algorithm_name}_Convergence_{ts}.png"
        fig.savefig(fname, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"[Q2] Convergence plot saved: {fname}")
    except Exception as e:
        # 打印完整异常信息，便于定位
        import traceback
        print("[Warning] Failed to generate convergence plot due to an exception:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))

# =========================
# 7) 主流程
# =========================

def main():
    ap = argparse.ArgumentParser("Q2 Optimizer (Reparam (θ,v,t0,τ))")
    ap.add_argument("--algo", choices=["hybrid","de","pso","sa","pattern"], default="hybrid")
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
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    print("="*80)
    print(f"[Q2] Start algo={args.algo} | pop={args.pop} iters={args.iter} "
          f"| workers={workers or 'auto'}")
    print(f"[Q2] coarse(dt={args.dt_coarse}, nphi={args.nphi_coarse}, nz={args.nz_coarse}) "
          f"-> final(dt={args.dt_final}, nphi={args.nphi_final}, nz={args.nz_final})")
    print(f"[Q2] Reparam search space: θ∈[0,2π), v∈[{SPEED_MIN},{SPEED_MAX}], "
          f"t0∈[{T0_MIN:.3f},{T0_MAX:.3f}], τ∈[{TAU_MIN},{TAU_MAX}] (with t0≥τ).")
    print("="*80)

    tA = time.time()
    best_val = 0.0
    best_x = np.array([0,0,0,0], dtype=float)
    best_expl = np.array([0,0,0], dtype=float)
    hit_time = HIT_TIME
    top_list = None
    history = None

    if args.algo == "hybrid":
        best_val, best_x, best_expl, hit_time, top_list = hybrid_opt(
            pop=args.pop, iters=args.iter, workers=workers or (os.cpu_count() or 1),
            topk=args.topk,
            dt_coarse=args.dt_coarse, nphi_coarse=args.nphi_coarse, nz_coarse=args.nz_coarse,
            dt_final=args.dt_final, nphi_final=args.nphi_final, nz_final=args.nz_final
        )
    elif args.algo == "de":
        best_val, best_x, best_expl, hit_time, history = de_opt(
            pop=args.pop, iters=args.iter,
            eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            workers=workers or (os.cpu_count() or 1),
            F=0.7, CR=0.9, strategy="best1bin", init_seeds=None
        )
        sec_tmp, expl_tmp, hit_tmp, _drop = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
        best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
    elif args.algo == "pso":
        best_val, best_x, best_expl, hit_time, history = pso_opt(
            pop=args.pop, iters=args.iter,
            eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            workers=workers or (os.cpu_count() or 1),
            w=0.5, c1=1.5, c2=1.5, init_seeds=None
        )
        sec_tmp, expl_tmp, hit_tmp, _drop = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
        best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
    elif args.algo == "sa":
        x0 = np.array([math.atan2(-FY1_INIT[1], -FY1_INIT[0]), 120.0, 5.0, 3.6], dtype=float)
        best_val, best_x, best_expl, hit_time, history = sa_opt(
            x0=x0,
            eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            T0=1.0, Tend=1e-3, iters=max(20000, args.iter*500),
            step_scale=np.array([0.2, 10.0, 2.0, 1.0])
        )
        sec_tmp, expl_tmp, hit_tmp, _drop = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
        best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp
    else:  # pattern
        x0 = np.array([math.atan2(-FY1_INIT[1], -FY1_INIT[0]), 120.0, 5.0, 3.6], dtype=float)
        best_val, best_x, best_expl, hit_time, history = pattern_search(
            x0=x0, steps=np.array([0.3, 10.0, 2.0, 1.0]),
            eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            max_iter=max(60, args.iter), shrink=0.6,
            workers=workers or (os.cpu_count() or 1)
        )
        sec_tmp, expl_tmp, hit_tmp, _drop = evaluate_occlusion(*best_x, dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final)
        best_val, best_expl, hit_time = sec_tmp, expl_tmp, hit_tmp

    tB = time.time()
    print("="*80)
    print(f"[Q2] DONE. best={best_val:.6f} s | time={tB - tA:.2f}s")
    print("="*80)

    params = {
        "pop": args.pop, "iter": args.iter, "topk": args.topk,
        "coarse": dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse),
        "final": dict(dt=args.dt_final, nphi=args.nphi_final, nz=args.nz_final),
        "workers": workers or os.cpu_count() or 1,
        "search_space": dict(theta="[0,2pi)", v=f"[{SPEED_MIN},{SPEED_MAX}]",
                             t0=f"[{T0_MIN:.3f},{T0_MAX:.3f}]", tau=f"[{TAU_MIN},{TAU_MAX}]")
    }
    save_report("Q2Results.txt", args.algo, best_val, best_x, best_expl, hit_time,
                top_list=top_list, params=params)

    print(f"[Info] Results saved to Q2Results.txt")
    h, v, t0, tau = best_x
    drop = max(0.0, t0 - tau)
    print(f"[Best] h={h:.6f} rad, v={v:.3f} m/s, t0={t0:.3f} s, tau={tau:.3f} s, drop={drop:.3f} s")
    print(f"[Best] expl=({best_expl[0]:.2f},{best_expl[1]:.2f},{best_expl[2]:.2f}), occlusion={best_val:.6f}s")

    # 生成收敛曲线（只要 history 存在且至少一个点就绘制）
    if history is not None and len(history) >= 1:
        generate_convergence_plot(history, args.algo, best_val)

if __name__ == "__main__":
    main()
