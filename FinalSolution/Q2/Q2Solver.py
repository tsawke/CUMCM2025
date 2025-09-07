import os
import sys
import math
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def IsMainProcess() -> bool:
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True

def PrintInitBar(step: int, total: int):
    if not IsMainProcess():
        return
    if os.environ.get("Q2_INIT_PRINTED", "") == "1":
        return
    barLen = 30
    filled = int(barLen * step / max(total, 1))
    bar = "#" * filled + "-" * (barLen - filled)
    sys.stdout.write(f"\r[Init] [{bar}] {step}/{total}")
    sys.stdout.flush()
    if step >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()
        os.environ["Q2_INIT_PRINTED"] = "1"

# import Q1Solver or fallback
TotalInitSteps = 3
PrintInitBar(1, TotalInitSteps)

Q1_IMPORTED = False
try:
    import Q1Solver as Q1
    PrintInitBar(2, TotalInitSteps)
    g = getattr(Q1, "g", 9.8)
    MISSILE_SPEED = getattr(Q1, "MISSILE_SPEED", 300.0)
    SMOG_R = getattr(Q1, "SMOG_R", 10.0)
    SMOG_SINK_SPEED = getattr(Q1, "SMOG_SINK_SPEED", 3.0)
    SMOG_EFFECT_TIME = getattr(Q1, "SMOG_EFFECT_TIME", 20.0)
    FY1_INIT = getattr(Q1, "FY1_INIT", np.array([17800.0, 0.0, 1800.0], dtype = float))
    M1_INIT = getattr(Q1, "M1_INIT", np.array([20000.0, 0.0, 2000.0], dtype = float))
    FAKE_TARGET_ORIGIN = getattr(Q1, "FAKE_TARGET_ORIGIN", np.array([0.0, 0.0, 0.0], dtype = float))
    EPS = getattr(Q1, "EPS", 1e-12)

    Unit = Q1.Unit
    MissileState = Q1.MissileState
    UavStateHorizontal = Q1.UavStateHorizontal
    PreCalCylinderPoints = Q1.PreCalCylinderPoints
    ConeAllPointsIn = Q1.ConeAllPointsIn

    Q1_IMPORTED = True
    PrintInitBar(TotalInitSteps, TotalInitSteps)
except Exception:
    PrintInitBar(2, TotalInitSteps)
    # fallback implementation
    g = 9.8
    MISSILE_SPEED = 300.0
    SMOG_R = 10.0
    SMOG_SINK_SPEED = 3.0
    SMOG_EFFECT_TIME = 20.0
    FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype = float)
    M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype = float)
    FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype = float)
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
                          uavInit[2]], dtype = float),
                np.array([vx, vy, 0.0], dtype = float))

    def PreCalCylinderPoints(nPhi, nZ, dtype = np.float64):
        b = np.array([0.0, 200.0, 0.0], dtype = dtype)
        r, h = dtype(7.0), dtype(10.0)
        phis = np.linspace(0.0, 2.0*math.pi, nPhi, endpoint = False, dtype = dtype)
        ring = np.stack([r*np.cos(phis), r*np.sin(phis), np.zeros_like(phis)], axis = 1).astype(dtype)
        pts = [b + ring, b + np.array([0.0,0.0,h], dtype = dtype) + ring]
        if nZ >= 2:
            for z in np.linspace(0.0, h, nZ, dtype = dtype):
                pts.append(b + np.array([0.0,0.0,z], dtype = dtype) + ring)
        return np.vstack(pts).astype(dtype)

    def ConeAllPointsIn(m, c, p, rCloud = SMOG_R, margin = EPS, block = 8192):
        v = c - m
        l = np.linalg.norm(v)
        if l <= EPS or rCloud >= l:
            return True
        cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud/l)**2))
        for i in range(0, len(p), block):
            w = p[i:i+block] - m
            wn = np.linalg.norm(w, axis = 1) + EPS
            lhs = w @ v
            rhs = wn * l * cosAlpha
            if not np.all(lhs + margin >= rhs):
                return False
        return True
    PrintInitBar(TotalInitSteps, TotalInitSteps)

HEADING_MIN, HEADING_MAX = 0.0, 2.0 * math.pi
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DROP_MIN, DROP_MAX_HARD = 0.0, 60.0
FUSE_MIN, FUSE_MAX = 0.0, 18.0

HIT_TIME = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
DROP_MAX = min(DROP_MAX_HARD, max(0.0, HIT_TIME - FUSE_MIN - 1e-3))

def ClampParams(x):
    h, v, td, tf = x
    h = h % (2.0 * math.pi)
    v = min(max(v, SPEED_MIN), SPEED_MAX)
    td = min(max(td, DROP_MIN), DROP_MAX)
    tf = min(max(tf, FUSE_MIN), FUSE_MAX)
    if td + tf >= HIT_TIME - 1e-6:
        td = max(DROP_MIN, HIT_TIME - 1e-6 - tf)
    return np.array([h, v, td, tf], dtype = float)

def ExplosionPointFromParams(heading, speed, drop, fuse):
    dropPos, uavV = UavStateHorizontal(drop, FY1_INIT, speed, heading)
    explXy = dropPos[:2] + uavV[:2] * fuse
    explZ = dropPos[2] - 0.5 * g * (fuse ** 2)
    return np.array([explXy[0], explXy[1], explZ], dtype = float)

def DropPointFromParams(heading, speed, drop):
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([FY1_INIT[0] + vx * drop,
                     FY1_INIT[1] + vy * drop,
                     FY1_INIT[2]], dtype = float)

TargetPtsCache: Dict[Tuple[int,int], np.ndarray] = {}
def GetTargetPoints(nphi: int, nz: int) -> np.ndarray:
    key = (int(nphi), int(nz))
    pts = TargetPtsCache.get(key, None)
    if pts is None:
        pts = PreCalCylinderPoints(nphi, nz, dtype = float)
        TargetPtsCache[key] = pts
    return pts

def QuickHardFilters(heading, speed, drop, fuse):
    explPos = ExplosionPointFromParams(heading, speed, drop, fuse)
    t0 = drop + fuse
    t1 = min(t0 + SMOG_EFFECT_TIME, HIT_TIME)
    if explPos[2] <= 0.0:
        return False, t0, t1, HIT_TIME, explPos
    if t1 <= t0:
        return False, t0, t1, HIT_TIME, explPos
    return True, t0, t1, HIT_TIME, explPos

def EvaluateOcclusion(heading, speed, drop, fuse,
                       dt = 0.002, nphi = 480, nz = 9,
                       margin = EPS, block = 8192) -> Tuple[float, np.ndarray, float]:
    ok, t0, t1, hitTime, explPos = QuickHardFilters(heading, speed, drop, fuse)
    if not ok:
        return 0.0, explPos, hitTime

    tGrid = np.arange(t0, t1 + EPS, dt, dtype = float)
    if len(tGrid) == 0:
        return 0.0, explPos, hitTime
    pts = GetTargetPoints(nphi, nz)

    mask = np.zeros(len(tGrid), dtype = bool)
    for j, t in enumerate(tGrid):
        mPos, _ = MissileState(float(t), M1_INIT)
        cPos = np.array([explPos[0], explPos[1],
                          explPos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype = float)
        mask[j] = ConeAllPointsIn(mPos, cPos, pts, rCloud = SMOG_R, margin = margin, block = block)
    seconds = float(np.count_nonzero(mask) * dt)
    return seconds, explPos, hitTime

def EvalTuple(args):
    (heading, speed, drop, fuse, dt, nphi, nz, margin, block) = args
    seconds, expl, hit = EvaluateOcclusion(heading, speed, drop, fuse, dt, nphi, nz, margin, block)
    return seconds, (heading, speed, drop, fuse), expl, hit

def LatinHypercube(nSamples: int) -> np.ndarray:
    rng = np.random.default_rng()
    u = (rng.random((nSamples,4)) + np.arange(nSamples)[:,None]) / nSamples
    rng.shuffle(u, axis = 0)
    headings = HEADING_MIN + u[:,0]*(HEADING_MAX - HEADING_MIN)
    speeds   = SPEED_MIN   + u[:,1]*(SPEED_MAX   - SPEED_MIN)
    drops    = DROP_MIN    + u[:,2]*(DROP_MAX    - DROP_MIN)
    fuses    = FUSE_MIN    + u[:,3]*(FUSE_MAX    - FUSE_MIN)
    drops = np.maximum(drops, 0.02 * (DROP_MAX - DROP_MIN))
    X = np.column_stack([headings, speeds, drops, fuses])
    X = np.array([ClampParams(x) for x in X], dtype = float)
    return X

def HeuristicSeeds(k: int = 8) -> List[np.ndarray]:
    rng = np.random.default_rng()
    seeds = []
    headingToOrigin = math.atan2(-FY1_INIT[1], -FY1_INIT[0])
    baseList = [
        np.array([headingToOrigin, 120.0, 1.5, 3.6], dtype = float),
        np.array([headingToOrigin+0.1, 110.0, 1.2, 4.0], dtype = float),
        np.array([headingToOrigin-0.1, 130.0, 1.8, 3.0], dtype = float),
        np.array([headingToOrigin, 140.0, 2.5, 2.0], dtype = float),
        np.array([headingToOrigin, 100.0, 3.0, 2.5], dtype = float),
    ]
    for b in baseList:
        seeds.append(ClampParams(b))
    while len(seeds) < k:
        d = rng.uniform(-0.15, 0.15, size = 4) * np.array([1.0, 10.0, 0.8, 0.8])
        x = seeds[rng.integers(0, len(seeds))] + d
        seeds.append(ClampParams(x))
    return seeds[:k]

def PatternSearch(x0: np.ndarray, steps: np.ndarray, evalKwargs: dict,
                   maxIter: int = 60, shrink: float = 0.5, workers: int = None
                   ) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    x = ClampParams(x0.copy())
    bestVal, bestExpl, bestHit = EvaluateOcclusion(*x, **evalKwargs)
    dirs = np.eye(4, dtype = float)
    hist = [bestVal]

    for _ in range(maxIter):
        cands = [x + steps*d for d in dirs] + [x - steps*d for d in dirs]
        tasks = [(ClampParams(c)[0], ClampParams(c)[1], ClampParams(c)[2], ClampParams(c)[3],
                  evalKwargs.get("dt"), evalKwargs.get("nphi"), evalKwargs.get("nz"),
                  evalKwargs.get("margin"), evalKwargs.get("block")) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers = workers) as pool:
            futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xcand, expl, hit = fut.result()
                vals.append((sec, np.array(xcand, dtype = float), expl, hit))
        improved = False
        for sec, xc, expl, hit in vals:
            if sec > bestVal + 1e-12:
                bestVal, bestExpl, bestHit = sec, expl, hit
                x = xc
                improved = True
        hist.append(bestVal)
        if improved:
            continue
        steps *= shrink
        if np.all(steps < np.array([1e-4, 0.05, 1e-3, 1e-3])):
            break
    return bestVal, x, bestExpl, bestHit, hist

def DeOpt(pop: int, iters: int, evalKwargs: dict, workers: int = None,
           F: float = 0.7, CR: float = 0.9, strategy: str = "best1bin",
           initSeeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    rng = np.random.default_rng()
    X = []
    if initSeeds:
        for s in initSeeds:
            X.append(ClampParams(s))
    while len(X) < pop:
        X.append(ClampParams(np.array([
            rng.uniform(HEADING_MIN, HEADING_MAX),
            rng.uniform(SPEED_MIN, SPEED_MAX),
            rng.uniform(DROP_MIN, DROP_MAX),
            rng.uniform(FUSE_MIN, FUSE_MAX),
        ], dtype = float)))
    X = np.array(X, dtype = float)

    tasks = [(x[0], x[1], x[2], x[3],
              evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
              evalKwargs["margin"], evalKwargs["block"]) for x in X]
    scores, expls = np.zeros(len(X), dtype = float), [None]*len(X)
    hitTime = HIT_TIME
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            scores[i] = sec
            expls[i] = expl
            hitTime = hit
    gbestIdx = int(np.argmax(scores))
    gbest, gbestScore, gbestExpl = X[gbestIdx].copy(), float(scores[gbestIdx]), expls[gbestIdx]
    hist = [gbestScore]

    percentStep = max(1, iters // 20)
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
            trial = np.empty(4, dtype = float)
            jrand = rng.integers(0, 4)
            for j in range(4):
                trial[j] = mutant[j] if (rng.random() < CR or j == jrand) else X[i][j]
            newX[i] = ClampParams(trial)

        tasks = [(x[0], x[1], x[2], x[3],
                  evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
                  evalKwargs["margin"], evalKwargs["block"]) for x in newX]
        newScores, newExpls = np.zeros(len(X), dtype = float), [None]*len(X)
        with ProcessPoolExecutor(max_workers = workers) as pool:
            futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                newScores[i] = sec
                newExpls[i] = expl
                hitTime = hit
        for i in range(len(X)):
            if newScores[i] >= scores[i]:
                X[i], scores[i], expls[i] = newX[i], newScores[i], newExpls[i]
        gbest_idx = int(np.argmax(scores))
        if scores[gbest_idx] > gbest_score:
            gbest_score = float(scores[gbest_idx])
            gbest = X[gbest_idx].copy()
            gbestExpl = expls[gbest_idx]
        hist.append(gbest_score)
        if (it+1) % percentStep == 0 or it == iters-1:
            print(f"[DE] {int((it+1)/iters*100)}%  gbest={gbest_score:.6f}")
    return gbest_score, gbest, gbestExpl, hitTime, hist

def PsoOpt(pop: int, iters: int, evalKwargs: dict, workers: int = None,
            w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
            initSeeds: List[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:

    rng = np.random.default_rng()
    X, V = [], []
    if initSeeds:
        for s in initSeeds:
            x = ClampParams(s); X.append(x); V.append(np.zeros(4, dtype = float))
    while len(X) < pop:
        x = np.array([rng.uniform(HEADING_MIN, HEADING_MAX),
                      rng.uniform(SPEED_MIN, SPEED_MAX),
                      rng.uniform(DROP_MIN, DROP_MAX),
                      rng.uniform(FUSE_MIN, FUSE_MAX)], dtype = float)
        X.append(ClampParams(x))
        V.append(rng.uniform(-1,1,size=4))
    X, V = np.array(X, dtype = float), np.array(V, dtype = float)

    tasks = [(x[0], x[1], x[2], x[3],
              evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
              evalKwargs["margin"], evalKwargs["block"]) for x in X]
    Pbest, Pscore = X.copy(), np.zeros(len(X), dtype = float)
    expls = [None]*len(X)
    hitTime = HIT_TIME
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            Pscore[i] = sec
            expls[i] = expl
            hitTime = hit
    g_idx = int(np.argmax(Pscore))
    Gbest, Gscore, Gexpl = X[g_idx].copy(), float(Pscore[g_idx]), expls[g_idx]
    hist = [Gscore]

    percentStep = max(1, iters // 20)
    for it in range(iters):
        for i in range(len(X)):
            r1, r2 = rng.random(4), rng.random(4)
            V[i] = w*V[i] + c1*r1*(Pbest[i]-X[i]) + c2*r2*(Gbest-X[i])
            X[i] = ClampParams(X[i] + V[i])
            for j,(mn,mx) in enumerate([(HEADING_MIN,HEADING_MAX),
                                        (SPEED_MIN,SPEED_MAX),
                                        (DROP_MIN,DROP_MAX),
                                        (FUSE_MIN,FUSE_MAX)]):
                if j==0:  # angle no bounce
                    continue
                if X[i][j] <= mn or X[i][j] >= mx:
                    V[i][j] *= -0.5
        tasks = [(x[0], x[1], x[2], x[3],
                  evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
                  evalKwargs["margin"], evalKwargs["block"]) for x in X]
        with ProcessPoolExecutor(max_workers = workers) as pool:
            futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                i = futs[fut]
                if sec > Pscore[i]:
                    Pscore[i], Pbest[i] = sec, X[i].copy()
                if sec > Gscore:
                    Gscore, Gbest, Gexpl = sec, X[i].copy(), expl
                hitTime = hit
        hist.append(Gscore)
        if (it+1) % percentStep == 0 or it == iters-1:
            print(f"[PSO] {int((it+1)/iters*100)}%  gbest={Gscore:.6f}")
    return Gscore, Gbest, Gexpl, hitTime, hist

def SaOpt(evalKwargs: dict,
           iters: int = 8000,
           nChains: int = 8,
           batchSize: int = 32,
           T0: float = 1.0,
           Tend: float = 1e-3,
           workers: int = None,
           noImprovePatience: int = 2000
           ) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    rng = np.random.default_rng()
    inits = HeuristicSeeds(k=nChains)
    X = np.array([ClampParams(x) for x in inits], dtype = float)
    F = np.zeros(nChains, dtype = float)
    EXPLS = [None]*nChains
    hitTime = HIT_TIME

    tasks = [(x[0], x[1], x[2], x[3],
              evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
              evalKwargs["margin"], evalKwargs["block"]) for x in X]
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, xc, expl, hit = fut.result()
            i = futs[fut]
            F[i] = sec; EXPLS[i] = expl; hitTime = hit

    gIdx = int(np.argmax(F))
    bestF = float(F[gIdx]); bestX = X[gIdx].copy(); bestExpl = EXPLS[gIdx]
    hist = [bestF]
    lastImprove = 0

    percentStep = max(1, iters // 20)

    for k in range(1, iters+1):
        T = T0 * (Tend/T0) ** (k/iters)
        candTasks = []
        candIndexMap = []
        for i in range(nChains):
            step = np.array([0.25, 8.0, 0.8, 0.8]) * max(T, 1e-3)
            for j in range(batchSize):
                cand = ClampParams(X[i] + rng.normal(0.0, 1.0, size=4) * step)
                candTasks.append((cand[0], cand[1], cand[2], cand[3],
                                   evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
                                   evalKwargs["margin"], evalKwargs["block"]))
                candIndexMap.append((i, j))

        candScores = np.zeros(len(candTasks), dtype = float)
        candExpls = [None]*len(candTasks)

        with ProcessPoolExecutor(max_workers = workers) as pool:
            futs = {pool.submit(EvalTuple, t): idx for idx,t in enumerate(candTasks)}
            for fut in as_completed(futs):
                sec, xc, expl, hit = fut.result()
                idx = futs[fut]
                candScores[idx] = sec
                candExpls[idx] = expl
                hitTime = hit

        for i in range(nChains):
            start = i*batchSize
            end = (i+1)*batchSize
            scores_i = candScores[start:end]
            expls_i = candExpls[start:end]

            better_mask = scores_i >= F[i] + 1e-12
            if np.any(better_mask):
                j_rel = int(np.argmax(scores_i))
                j_abs = start + j_rel
                X[i] = np.array([candTasks[j_abs][0], candTasks[j_abs][1], candTasks[j_abs][2], candTasks[j_abs][3]], dtype = float)
                EXPLS[i] = expls_i[j_rel]
                F[i] = scores_i[j_rel]
            else:
                delta = scores_i - F[i]
                probs = np.exp(delta / max(T, 1e-9))
                probs = probs / (probs.sum() + 1e-12)
                pick = np.random.choice(len(scores_i), p=probs)
                j_abs = start + pick
                X[i] = np.array([candTasks[j_abs][0], candTasks[j_abs][1], candTasks[j_abs][2], candTasks[j_abs][3]], dtype = float)
                EXPLS[i] = expls_i[pick]
                F[i] = scores_i[pick]

        gIdx = int(np.argmax(F))
        if F[gIdx] > bestF + 1e-12:
            bestF = float(F[gIdx]); bestX = X[gIdx].copy(); bestExpl = EXPLS[gIdx]
            lastImprove = k

        if (k % percentStep == 0) or (k == iters):
            hist.append(bestF)
            print(f"[SA] {int(k/iters*100)}%  best={bestF:.6f}")

        if k - lastImprove >= noImprovePatience:
            worst_i = int(np.argmin(F))
            X[worst_i] = HeuristicSeeds(k=1)[0]
            F[worst_i], EXPLS[worst_i], _ = EvaluateOcclusion(*X[worst_i], **evalKwargs)
            lastImprove = k
            if F[worst_i] <= 0 and bestF <= 0:
                print(f"[SA] early stop at {k}, stagnation. best={bestF:.6f}")
                break

    return bestF, bestX, bestExpl, hitTime, hist

# =========================
def HybridOpt(pop: int, iters: int, workers: int, topk: int,
               dtCoarse: float, nphiCoarse: int, nzCoarse: int,
               dtFinal: float, nphiFinal: int, nzFinal: int,
               probe: int = 0):
    print("[HYBRID] Stage 1: Feasibility probe + LHS coarse screening")
    seeds = LatinHypercube(pop)
    seeds[:min(len(seeds), 6)] = HeuristicSeeds(k = min(6, pop))
    if probe > 0:
        probers = HeuristicSeeds(k=min(8, probe)) + list(LatinHypercube(max(0, probe-8)))
        probers = [ClampParams(p) for p in probers]
        tasks = [(p[0], p[1], p[2], p[3], dtCoarse, nphiCoarse, nzCoarse, EPS, 8192) for p in probers]
        feasible = 0
        with ProcessPoolExecutor(max_workers = workers) as pool:
            for fut in as_completed({pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}):
                sec, *_ = fut.result()
                feasible += int(sec > 0)
        print(f"[HYBRID] Probe feasible count = {feasible}/{len(probers)}")

    tasks = [(row[0], row[1], row[2], row[3], dtCoarse, nphiCoarse, nzCoarse, EPS, 8192) for row in seeds]
    vals = []
    lhsHistory = []
    bestSoFar = 0.0
    stride = max(1, len(seeds) // 20)

    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            vals.append((sec, np.array(x), expl, hit))
            if sec > bestSoFar: bestSoFar = sec
            done += 1
            if done % stride == 0 or done == len(seeds):
                lhsHistory.append(bestSoFar)
                print(f"[HYBRID] {int(100*done/len(seeds))}%  LHS bestSoFar={bestSoFar:.6f}")

    vals.sort(key=lambda x: x[0], reverse=True)
    top = vals[:topk]
    print(f"[HYBRID] Coarse screening Top-{topk} best = {top[0][0]:.6f} s")

    print("[HYBRID] Stage 2: Top-K Pattern refinement (skip 0-score seeds, keep at most 1 fallback)")
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
            print(f"  -> Pattern seed#{rank} is 0, keep 1 fallback sample")
        else:
            print(f"  -> Pattern seed#{rank}  initial value = {sec0:.6f}")
        steps = np.array([0.15, 5.0, 0.6, 0.6], dtype = float)
        bestSec, bestX, bestExpl, bestHit, hist = PatternSearch(
            x0, steps,
            evalKwargs=dict(dt=dtCoarse, nphi=nphiCoarse, nz=nzCoarse, margin=EPS, block=8192),
            maxIter=60, shrink=0.6, workers=workers
        )
        refined.append((bestSec, bestX, bestExpl, bestHit))
        pattern_each_histories.append(hist)
        global_best_after_patterns = max(global_best_after_patterns, bestSec)
        pattern_global_history.append(global_best_after_patterns)
        if bestSec > TOL:
            print(f"    Pattern completed: {bestSec:.6f} s")

    refined.sort(key=lambda x: x[0], reverse=True)
    seeds2 = [r[1] for r in refined[:min(len(refined), max(4, topk//2))]]
    if not seeds2:
        seeds2 = HeuristicSeeds(k = 4)  # fallback

    print("[HYBRID] Stage 3: DE global convergence (coarse evaluation)")
    bestDe, xDe, explDe, hitDe, deHistory = DeOpt(
        pop=pop, iters=iters,
        evalKwargs=dict(dt=dtCoarse, nphi=nphiCoarse, nz=nzCoarse, margin=EPS, block=8192),
        workers=workers, F=0.7, CR=0.9, strategy="best1bin",
        initSeeds=seeds2
    )
    print(f"[HYBRID] DE coarse precision result: {bestDe:.6f} s")

    print("[HYBRID] Stage 4: Final evaluation (high precision re-evaluation)")
    candidates = [(bestDe, xDe, explDe, hitDe)] + refined[:min(10, len(refined))]
    # deduplication
    uniq, seen = [], set()
    for c in candidates:
        arr = c[1]
        k = (round(arr[0],3), round(arr[1],2), round(arr[2],3), round(arr[3],3))
        if k not in seen:
            seen.add(k); uniq.append(c)

    tasks = [(ClampParams(x[1])[0], ClampParams(x[1])[1], ClampParams(x[1])[2], ClampParams(x[1])[3],
              dtFinal, nphiFinal, nzFinal, EPS, 8192) for x in uniq]
    finals = []
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            finals.append((sec, np.array(x), expl, hit))
    finals.sort(key=lambda x: x[0], reverse=True)
    best = finals[0]

    hybridHistory = dict(lhs=lhsHistory,
                          pattern_each=pattern_each_histories,
                          pattern_global=pattern_global_history,
                          de=deHistory)
    return best[0], best[1], best[2], best[3], finals[:min(10,len(finals))], hybridHistory

def SaveReport(filename: str, algo: str,
                bestVal: float, bestX: np.ndarray, bestExpl: np.ndarray, hitTime: float,
                topList: List[Tuple[float,np.ndarray,np.ndarray]] = None,
                params: Dict = None):
    heading, speed, drop, fuse = float(bestX[0]), float(bestX[1]), float(bestX[2]), float(bestX[3])
    dropPos = DropPointFromParams(heading, speed, drop)

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
        f.write(f"Occlusion duration = {bestVal:.6f} seconds\n")
        f.write(f"Heading (rad) = {heading:.6f}\n")
        f.write(f"Speed (m/s)  = {speed:.6f}\n")
        f.write(f"Drop time (s) = {drop:.6f}\n")
        f.write(f"Fuse delay (s)= {fuse:.6f}\n")
        f.write(f"Drop point = ({dropPos[0]:.6f}, {dropPos[1]:.6f}, {dropPos[2]:.6f})\n")
        f.write(f"Explosion point = ({bestExpl[0]:.6f}, {bestExpl[1]:.6f}, {bestExpl[2]:.6f})\n")
        f.write(f"Missile hit time ≈ {hitTime:.6f} seconds\n\n")

        if topList:
            f.write("Top Candidates (final evaluation):\n")
            f.write("-"*40 + "\n")
            for i,(sec,x,expl,_) in enumerate(topList, start=1):
                f.write(f"#{i}: {sec:.6f}s | "
                        f"h={x[0]:.6f}, v={x[1]:.3f}, drop={x[2]:.3f}, fuse={x[3]:.3f} | "
                        f"expl=({expl[0]:.2f},{expl[1]:.2f},{expl[2]:.2f})\n")
        f.write("\n" + "="*80 + "\n")

def GenerateConvergencePlot(history: List[float], algorithmName: str, bestValue: float):
    try:
        if not history:
            print(f"[Plot] {algorithmName}: no history to plot."); return
        fig = plt.figure(figsize = (8, 5)); ax = fig.add_subplot(111)
        iters = list(range(len(history)))
        ax.plot(iters, history, linewidth = 2, alpha = 0.9, label='Best-so-far')
        ax.axhline(y=bestValue, linestyle = '--', linewidth = 2, label=f'Final Best: {bestValue:.6f}s')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Occlusion Duration (s)')
        ax.set_title(f'{algorithmName.upper()} Convergence')
        ax.grid(True, alpha = 0.3); ax.legend()
        fig.tight_layout()
        out = f'Q2_{algorithmName}_Convergence.png'
        fig.savefig(out, dpi = 300, bbox_inches='tight'); plt.close(fig)
        print(f"[Plot] Saved {out}")
    except Exception as e:
        print(f"[Warning] Failed to generate {algorithmName} plot: {e}")

def GenerateHybridConvergencePlot(hybridHistory: Dict, bestValue: float,
                                     coarseCfg: Dict, finalCfg: Dict):
    try:
        fig, axes = plt.subplots(3, 1, figsize = (10, 12))
        lhs = hybridHistory.get("lhs", [])
        ax = axes[0]
        if lhs:
            ax.plot(range(1, len(lhs)+1), lhs, linewidth = 2, label='LHS best-so-far')
        ax.axhline(y=bestValue, linestyle = '--', linewidth = 2, label=f'Final Best: {bestValue:.6f}s')
        ax.set_title(f"LHS Screening (coarse: dt={coarseCfg['dt']}, nphi={coarseCfg['nphi']}, nz={coarseCfg['nz']})")
        ax.set_xlabel('Checkpoint (~5%)'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha = 0.3); ax.legend()

        ax = axes[1]
        pattern_each = hybridHistory.get("pattern_each", [])
        pattern_global = hybridHistory.get("pattern_global", [])
        if pattern_each:
            for i,hist in enumerate(pattern_each, start=1):
                ax.plot(range(len(hist)), hist, linewidth = 1.2, alpha = 0.6, label=f'seed#{i}')
        if pattern_global:
            ax.plot(range(1, len(pattern_global)+1), pattern_global, linewidth = 2.5, label='Global best (after each seed)')
        ax.axhline(y=bestValue, linestyle = '--', linewidth = 2, label=f'Final Best: {bestValue:.6f}s')
        ax.set_title(f"Pattern Refinement (coarse: dt={coarseCfg['dt']}, nphi={coarseCfg['nphi']}, nz={coarseCfg['nz']})")
        ax.set_xlabel('Pattern Iteration / Seed Index'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha = 0.3); ax.legend(ncol=2, fontsize=9)

        ax = axes[2]
        de_hist = hybridHistory.get("de", [])
        if de_hist:
            ax.plot(range(len(de_hist)), de_hist, linewidth = 2, label='DE best-so-far')
        ax.axhline(y=bestValue, linestyle = '--', linewidth = 2, label=f'Final Best: {bestValue:.6f}s')
        ax.set_title(f"DE Global Search (coarse: dt={coarseCfg['dt']}, nphi={coarseCfg['nphi']}, nz={coarseCfg['nz']}; "
                     f"final: dt={finalCfg['dt']}, nphi={finalCfg['nphi']}, nz={finalCfg['nz']})")
        ax.set_xlabel('Generation'); ax.set_ylabel('Occlusion (s)')
        ax.grid(True, alpha = 0.3); ax.legend()

        fig.tight_layout()
        out = 'Q2_HYBRID_Convergence.png'
        fig.savefig(out, dpi = 300, bbox_inches='tight'); plt.close(fig)
        print(f"[Plot] Saved {out}")
    except Exception as e:
        print(f"[Warning] Failed to generate hybrid plot: {e}")

def main():
    ap = argparse.ArgumentParser("Q2 Optimizer (All)")
    ap.add_argument("--algo", choices = ["all","hybrid","de","pso","sa","pattern"], default = "all",
                    help = "select algorithm (all means execute five types sequentially)")
    ap.add_argument("--pop", type = int, default = 64, help = "population / swarm size (for hybrid/de/pso)")
    ap.add_argument("--iter", type = int, default = 60, help = "iterations / generations")
    ap.add_argument("--topk", type = int, default = 12, help = "top-k seeds for local refinement (hybrid)")
    ap.add_argument("--workers", default = "auto", help = "process workers, int or 'auto'")

    ap.add_argument("--sa-iters", type = int, default = 8000, help = "SA max steps (including early stop)")
    ap.add_argument("--sa-batch", type = int, default = 32, help = "SA candidates per chain per step")
    ap.add_argument("--sa-chains", type = int, default = 8, help = "SA parallel chain count")

    ap.add_argument("--probe", type = int, default = 0, help = "feasibility probe sample count (0 to disable)")

    ap.add_argument("--dt-coarse", type = float, default = 0.002)
    ap.add_argument("--nphi-coarse", type = int, default = 480)
    ap.add_argument("--nz-coarse", type = int, default = 9)
    ap.add_argument("--dt-final", type = float, default = 0.0005)
    ap.add_argument("--nphi-final", type = int, default = 960)
    ap.add_argument("--nz-final", type = int, default = 13)
    args = ap.parse_args()

    workers = None if args.workers == "auto" else int(args.workers)
    sa_chains = max(1, args.sa_chains)
    if workers is None:
        try:
            cpu = os.cpu_count() or 4
        except Exception:
            cpu = 4
        workers = cpu

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    algos = [args.algo] if args.algo != "all" else ["hybrid","de","pso","sa","pattern"]

    overallBest = -1.0
    overallRec = None

    for algo in algos:
        print("="*90)
        print(f"[Q2] Start algo={algo} | pop={args.pop} iters={args.iter} | workers={workers or 'auto'}")
        print(f"[Q2] coarse(dt={args.dtCoarse}, nphi={args.nphiCoarse}, nz={args.nzCoarse}) "
              f"-> final(dt={args.dtFinal}, nphi={args.nphiFinal}, nz={args.nzFinal})")
        print("="*90)

        tA = time.time()
        bestVal = 0.0
        bestX = np.array([0,0,0,0], dtype = float)
        bestExpl = np.array([0,0,0], dtype = float)
        hitTime = HIT_TIME
        history = None
        hybridHistory = None

        if algo == "hybrid":
            bestVal, bestX, bestExpl, hitTime, _, hybridHistory = HybridOpt(
                pop=args.pop, iters=args.iter, workers=workers or (os.cpu_count() or 1),
                topk=args.topk,
                dtCoarse=args.dtCoarse, nphiCoarse=args.nphiCoarse, nzCoarse=args.nzCoarse,
                dtFinal=args.dtFinal, nphiFinal=args.nphiFinal, nzFinal=args.nzFinal,
                probe=args.probe
            )
        elif algo == "de":
            seeds0 = HeuristicSeeds(k=min(8, args.pop))
            bestVal, bestX, bestExpl, hitTime, history = DeOpt(
                pop=args.pop, iters=args.iter,
                evalKwargs=dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                F=0.7, CR=0.9, strategy="best1bin", initSeeds=seeds0
            )
            sec_tmp, expl_tmp, hit_tmp = EvaluateOcclusion(*bestX, dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal)
            bestVal, bestExpl, hitTime = sec_tmp, expl_tmp, hit_tmp
        elif algo == "pso":
            seeds0 = HeuristicSeeds(k=min(8, args.pop))
            bestVal, bestX, bestExpl, hitTime, history = PsoOpt(
                pop=args.pop, iters=args.iter,
                evalKwargs=dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse, margin=EPS, block=8192),
                workers=workers or (os.cpu_count() or 1),
                w=0.5, c1=1.5, c2=1.5, initSeeds=seeds0
            )
            sec_tmp, expl_tmp, hit_tmp = EvaluateOcclusion(*bestX, dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal)
            bestVal, bestExpl, hitTime = sec_tmp, expl_tmp, hit_tmp
        elif algo == "sa":
            bestVal, bestX, bestExpl, hitTime, history = SaOpt(
                evalKwargs=dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse, margin=EPS, block=8192),
                iters=args.sa_iters,
                nChains=sa_chains,
                batchSize=args.sa_batch,
                T0=1.0, Tend=1e-3,
                workers=workers or (os.cpu_count() or 1),
                noImprovePatience=max(1000, args.sa_iters//3)
            )
            sec_tmp, expl_tmp, hit_tmp = EvaluateOcclusion(*bestX, dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal)
            bestVal, bestExpl, hitTime = sec_tmp, expl_tmp, hit_tmp
        else:
            x0 = HeuristicSeeds(k=1)[0]
            bestVal, bestX, bestExpl, hitTime, history = PatternSearch(
                x0=x0, steps=np.array([0.3, 10.0, 1.0, 1.0]),
                evalKwargs=dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse, margin=EPS, block=8192),
                maxIter=max(60, args.iter), shrink=0.6,
                workers=workers or (os.cpu_count() or 1)
            )
            sec_tmp, expl_tmp, hit_tmp = EvaluateOcclusion(*bestX, dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal)
            bestVal, bestExpl, hitTime = sec_tmp, expl_tmp, hit_tmp

        tB = time.time()
        print("="*90)
        print(f"[Q2] DONE ({algo}). best={bestVal:.6f} s | time={tB - tA:.2f}s")
        print("="*90)

        params = {
            "pop": args.pop, "iter": args.iter, "topk": args.topk,
            "coarse": dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse),
            "final": dict(dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal),
            "workers": workers or os.cpu_count() or 1,
            "speed_range": [SPEED_MIN, SPEED_MAX],
            "search_space": {"heading":"[0,2pi)","speed":"[70,140]","drop":f"[0,{DROP_MAX:.3f}]","fuse":"[0,18]"}
        }
        rpt = f"Q2Results_{algo}.txt"
        SaveReport(rpt, algo, bestVal, bestX, bestExpl, hitTime,
                    topList=(None if algo!='hybrid' else None), params=params)
        print(f"[Info] Results saved to {rpt}")
        print(f"[Best-{algo}] h={bestX[0]:.6f} rad, v={bestX[1]:.3f} m/s, drop={bestX[2]:.3f} s, fuse={bestX[3]:.3f} s")
        dropPosPrint = DropPointFromParams(bestX[0], bestX[1], bestX[2])
        print(f"[Best-{algo}] drop=({dropPosPrint[0]:.2f},{dropPosPrint[1]:.2f},{dropPosPrint[2]:.2f})")
        print(f"[Best-{algo}] expl=({bestExpl[0]:.2f},{bestExpl[1]:.2f},{bestExpl[2]:.2f}), occlusion={bestVal:.6f}s")

        if algo == "hybrid":
            if hybridHistory:
                GenerateHybridConvergencePlot(
                    hybridHistory, bestVal,
                    coarseCfg=dict(dt=args.dtCoarse, nphi=args.nphiCoarse, nz=args.nzCoarse),
                    finalCfg=dict(dt=args.dtFinal, nphi=args.nphiFinal, nz=args.nzFinal)
                )
        else:
            if history is not None:
                GenerateConvergencePlot(history, algo, bestVal)

        if bestVal > overallBest:
            overallBest = bestVal
            overallRec = (algo, bestVal, bestX, bestExpl, hitTime)

    if overallRec is not None:
        algo, bestVal, bestX, bestExpl, hitTime = overallRec
        print("\n" + "="*90)
        print(f"[Q2] All algorithms completed. Global optimum from {algo.upper()}: {bestVal:.6f} s")
        print(f"     h={bestX[0]:.6f} rad, v={bestX[1]:.3f} m/s, drop={bestX[2]:.3f} s, fuse={bestX[3]:.3f} s")
        dropPosBest = DropPointFromParams(bestX[0], bestX[1], bestX[2])
        print(f"     drop=({dropPosBest[0]:.2f},{dropPosBest[1]:.2f},{dropPosBest[2]:.2f})")
        print(f"     expl=({bestExpl[0]:.2f},{bestExpl[1]:.2f},{bestExpl[2]:.2f}), hit≈{hitTime:.3f}s")
        print("="*90)
        with open("Q2_All_Summary.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("Q2 Overall Best Across All Algorithms\n")
            f.write("="*80 + "\n")
            f.write(f"Algorithm: {algo}\n")
            f.write(f"Occlusion duration = {bestVal:.6f} s\n")
            f.write(f"Heading (rad) = {bestX[0]:.6f}\n")
            f.write(f"Speed (m/s)  = {bestX[1]:.6f}\n")
            f.write(f"Drop time (s) = {bestX[2]:.6f}\n")
            f.write(f"Fuse delay (s)= {bestX[3]:.6f}\n")
            f.write(f"Drop point = ({dropPosBest[0]:.6f}, {dropPosBest[1]:.6f}, {dropPosBest[2]:.6f})\n")
            f.write(f"Explosion point = ({bestExpl[0]:.6f}, {bestExpl[1]:.6f}, {bestExpl[2]:.6f})\n")
            f.write(f"Missile hit time ≈ {hitTime:.6f} s\n")
            f.write("="*80 + "\n")
        print("[Info] Overall summary saved to Q2_All_Summary.txt")

if __name__ == "__main__":
    main()
