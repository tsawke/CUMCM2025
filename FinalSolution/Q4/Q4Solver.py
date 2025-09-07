import os
import math
import time
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# constants
g = 9.8
MISSILE_SPEED = 300.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype = float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

# missile positions
M1_INIT = np.array([20000.0,    0.0, 2000.0], dtype = float)
M2_INIT = np.array([19000.0,  600.0, 2100.0], dtype = float)
M3_INIT = np.array([18000.0, -600.0, 1900.0], dtype = float)

FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype = float)

# UAV positions
FY1_INIT = np.array([17800.0,     0.0, 1800.0], dtype = float)
FY2_INIT = np.array([12000.0,  1400.0, 1400.0], dtype = float)
FY3_INIT = np.array([ 6000.0, -3000.0,  700.0], dtype = float)

# FY1 fixed params
FY1_FIXED = {
    "speed": 112.0298,
    "heading": 0.137353,
    "drop_time": 0.0045,
    "fuse_delay": 0.4950
}

EPS = 1e-12

# utils
def Unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, which = "M1"):
    # switch missile: "M2"/"M3"
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
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype = float), np.array([vx, vy, 0.0], dtype = float)

def PreCalCylinderPoints(nPhi, nZ, dtype = np.float64):
    b = CYLINDER_BASE_CENTER.astype(dtype)
    r, h = dtype(CYLINDER_R), dtype(CYLINDER_H)
    phis = np.linspace(0.0, 2.0 * math.pi, nPhi, endpoint = False, dtype = dtype)
    c, s = np.cos(phis), np.sin(phis)
    ring = np.stack([r * c, r * s, np.zeros_like(c)], axis = 1).astype(dtype)
    pts = [b + ring, b + np.array([0.0, 0.0, h], dtype = dtype) + ring]
    if nZ >= 2:
        for z in np.linspace(0.0, h, nZ, dtype = dtype):
            pts.append(b + np.array([0.0, 0.0, z], dtype = dtype) + ring)
    p = np.vstack(pts).astype(dtype)
    return p

def ExplosionPointFromPlan(uavInit, speed, heading, dropTime, fuseDelay):
    dropPos, uavV = UavStateHorizontal(dropTime, uavInit, speed, heading)
    explXy = dropPos[:2] + uavV[:2] * fuseDelay
    explZ = dropPos[2] - 0.5 * g * (fuseDelay ** 2)
    return np.array([explXy[0], explXy[1], explZ], dtype = float), dropPos

def ConeAllPointsIn(m, c, p, rCloud = SMOG_R, margin = EPS, block = 4096):
    v = c - m
    l = np.linalg.norm(v)
    if l <= EPS or rCloud >= l:
        return True
    cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud / l) ** 2))
    for i in range(0, len(p), block):
        w = p[i : i + block] - m
        wn = np.linalg.norm(w, axis = 1) + EPS
        lhs = w @ v
        rhs = wn * l * cosAlpha
        if not np.all(lhs + margin >= rhs):
            return False
    return True

def ConePointsCoveredByAny(m, centers: List[np.ndarray], p, rCloud = SMOG_R, margin = EPS, block = 4096):
    """Cooperative occlusion check"""
    if len(centers) == 0:
        return False
    vs = [c - m for c in centers]
    ls = np.array([np.linalg.norm(v) for v in vs], dtype = float)
    if np.any(ls <= rCloud + 1e-12):
        return True
    cosA = np.sqrt(np.maximum(0.0, 1.0 - (rCloud / ls) ** 2))
    V = np.stack(vs, axis = 1)
    LK = ls * cosA
    for i in range(0, len(p), block):
        w = p[i : i + block] - m
        wn = np.linalg.norm(w, axis = 1) + EPS
        lhs = w @ V
        rhs = wn[:, None] * LK[None, :]
        goodAny = (lhs + margin >= rhs)
        if not np.all(goodAny.any(axis = 1)):
            return False
    return True

# global vars
PTS_GLOBAL = None
HITTIME_GLOBAL = None
INTRA_THREADS_GLOBAL = 1
MISSILE_TAG = "M1"

def InitWorker(nphi, nz, intraThreads, missileTag):
    global PTS_GLOBAL, HITTIME_GLOBAL, INTRA_THREADS_GLOBAL, MISSILE_TAG
    PTS_GLOBAL = PreCalCylinderPoints(nphi, nz, dtype = np.float64)
    m0 = {"M1": M1_INIT, "M2": M2_INIT, "M3": M3_INIT}[missileTag]
    HITTIME_GLOBAL = float(np.linalg.norm(m0 - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    INTRA_THREADS_GLOBAL = int(max(1, intraThreads))
    MISSILE_TAG = missileTag

# evaluation
def SingleSmokeDuration(Te, explPos, dt):
    """Single cloud duration"""
    hitTime = HITTIME_GLOBAL
    pts = PTS_GLOBAL
    t0, t1 = Te, min(Te + SMOG_EFFECT_TIME, hitTime)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc, seen = t0, 0.0, False
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur, which = MISSILE_TAG)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, cur - Te)], dtype = float)
        ok = ConeAllPointsIn(m, c, pts, rCloud = SMOG_R)
        if ok:
            acc += dt; seen = True
        elif seen:
            break
        cur += dt
    return float(acc)

def SingleMask(Te, explPos, dtGrid):
    """Single cloud time mask"""
    pts = PTS_GLOBAL
    mask = np.zeros_like(dtGrid, dtype = bool)
    for i, t in enumerate(dtGrid):
        if (t < Te - 1e-12) or (t > Te + SMOG_EFFECT_TIME + 1e-12):
            continue
        m, _ = MissileState(float(t), which = MISSILE_TAG)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, t - Te)], dtype = float)
        mask[i] = ConeAllPointsIn(m, c, pts, rCloud = SMOG_R)
    return mask

def CooperativeUnionDuration(triple, dt):
    """Cooperative union duration"""
    pts = PTS_GLOBAL
    hitTime = HITTIME_GLOBAL
    Te = [triple[i]["T_e"] for i in range(3)]
    t0 = min(Te); t1 = min(hitTime, max(Te) + SMOG_EFFECT_TIME)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc = t0, 0.0
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur, which = MISSILE_TAG)
        centers = []
        for i in (0, 1, 2):
            Tei = triple[i]["T_e"]
            if (cur >= Tei - 1e-12) and (cur <= Tei + SMOG_EFFECT_TIME + 1e-12):
                ex = triple[i]["expl_pos"]
                centers.append(np.array([ex[0], ex[1], ex[2] - SMOG_SINK_SPEED * max(0.0, cur - Tei)], dtype = float))
        if centers and ConePointsCoveredByAny(m, centers, pts, rCloud = SMOG_R):
            acc += dt
        cur += dt
    return float(acc)

# explosion search
def BestCandidateForTeSpeed(uavInit, uavName, Te, spd, dt, lateralScales = (0.0, 0.015, -0.015, 0.025, -0.025)):
    """Candidate search with lateral offset"""
    mPos, _ = MissileState(Te, which = MISSILE_TAG)
    targetCenter = CYLINDER_BASE_CENTER + np.array([0.0, 0.0, CYLINDER_H * 0.5], dtype = float)

    L = targetCenter - mPos
    dist = np.linalg.norm(L)
    if dist < 1e-6:
        return None
    eL = L / dist
    perp = np.array([-eL[1], eL[0], 0.0], dtype = float)
    if np.linalg.norm(perp) < 1e-9:
        perp = np.array([1.0, 0.0, 0.0], dtype = float)
    perp = perp / (np.linalg.norm(perp) + 1e-12)

    sGrid = np.linspace(0.3, 0.95, 20)
    best, bestKey = None, (-1.0, -1.0, -1.0)

    def TrySd(s, d):
        nonlocal best, bestKey
        cand = mPos + s * L + d * dist * perp
        dropAlt = uavInit[2]
        if cand[2] > dropAlt + 1e-9:
            return
        dx, dy = cand[0] - uavInit[0], cand[1] - uavInit[1]
        reqV = math.hypot(dx, dy) / max(Te, 1e-9)
        if reqV > spd * 1.05:
            return
        heading = math.atan2(dy, dx)
        fuseDelay = math.sqrt(max(0.0, 2.0 * (dropAlt - cand[2]) / g))
        dropTime = Te - fuseDelay
        if dropTime < -1e-9:
            return
        explPos, dropPos = ExplosionPointFromPlan(uavInit, spd, heading, dropTime, fuseDelay)
        dur = SingleSmokeDuration(Te, explPos, dt)
        
        coopPotential = 1.0 / (1.0 + np.linalg.norm(explPos[:2] - targetCenter[:2]) / 1000.0) + s * 0.1 + abs(d) * 0.05
        
        if (dur > bestKey[0] + 1e-12) or \
           (abs(dur - bestKey[0]) <= 1e-12 and coopPotential > bestKey[2] + 1e-12) or \
           (abs(dur - bestKey[0]) <= 1e-12 and abs(coopPotential - bestKey[2]) <= 1e-12 and s > bestKey[1]):
            bestKey = (dur, s, coopPotential)
            best = {
                "T_e": float(Te), "speed": float(spd), "heading": float(heading),
                "drop_time": float(dropTime), "fuse_delay": float(fuseDelay),
                "expl_pos": explPos, "drop_pos": dropPos,
                "single_duration": float(dur), "s_used": float(s), "d_used": float(d),
                "uavName": uavName, "coop_potential": float(coopPotential)
            }

    for dscale in lateralScales:
        for s in sGrid:
            TrySd(float(s), float(dscale))
        if best is not None:
            s0 = best["s_used"]
            for width in (0.08, 0.04):
                lo = max(0.3, s0 - width)
                hi = min(0.95, s0 + width)
                for s in np.linspace(lo, hi, 11):
                    TrySd(float(s), float(dscale))

    return best

# workers
def CandidateWorker(pl):
    uavInit, Te, spd, dt, name, latScales = pl["uavInit"], pl["T_e"], pl["speed"], pl["dt"], pl["uavName"], pl["lat_scales"]
    def DoWork():
        c = BestCandidateForTeSpeed(uavInit, name, Te, spd, dt, lateralScales = latScales)
        return c
    if threadpool_limits is None:
        return DoWork()
    with threadpool_limits(limits = INTRA_THREADS_GLOBAL):
        return DoWork()

def UnionWorker(pl):
    dt, tri = pl["dt"], pl["triplet"]
    def DoWork():
        return CooperativeUnionDuration(tri, dt)
    if threadpool_limits is None:
        return DoWork()
    with threadpool_limits(limits = INTRA_THREADS_GLOBAL):
        return DoWork()

# config
def AutoBalance(workersOpt, intraOpt, backend, totalTasks):
    cpu = os.cpu_count() or 1
    if backend == "thread":
        workers = cpu if (workersOpt == "auto") else int(workersOpt)
        return workers, 1
    workers = (cpu if workersOpt == "auto" else max(1, int(workersOpt)))
    if totalTasks < workers: workers = max(1, totalTasks)
    intra = (1 if intraOpt == "auto" else max(1, int(intraOpt)))
    return workers, intra

# output
def Deg(rad):
    d = math.degrees(rad)
    return d + 360.0 if d < 0 else d

def PrintBest(prefix: str, unionTime: float, tri):
    print(f"[Best↑][{prefix}] Joint occlusion ≈ {unionTime:.6f} s")
    for c in sorted(tri, key = lambda x: x["uavName"]):
        h = Deg(c["heading"])
        dx, dy, dz = c["drop_pos"]; ex, ey, ez = c["expl_pos"]
        coop = c.get("coop_potential", 0.0)
        print(f"  - {c['uavName']}: speed={c['speed']:.3f} m/s, heading={h:.3f}°"
              f", drop=({dx:.3f},{dy:.3f},{dz:.3f}), explode=({ex:.3f},{ey:.3f},{ez:.3f})"
              f", individual={c['single_duration']:.6f}s, s={c.get('s_used',-1):.3f}, d={c.get('d_used',0.0):+.4f}"
              f", T_e={c['T_e']:.3f}s, coop_potential={coop:.4f}")

# plot
def PlotConvergence(bestTriple, dt, history, outPng = "Q4ConvergencePlot.png"):
    it = np.arange(1, len(history) + 1)
    bestCurve = np.maximum.accumulate(np.array(history, dtype = float))
    ptsLocal = PTS_GLOBAL
    hitLocal = HITTIME_GLOBAL
    Te = [bestTriple[i]["T_e"] for i in range(3)]
    t0 = min(Te); t1 = min(hitLocal, max(Te) + SMOG_EFFECT_TIME)
    if t1 <= t0 + 1e-12:
        plt.figure(figsize = (10, 6), dpi = 120)
        plt.plot(it, bestCurve, lw = 2)
        plt.xlabel("Iteration"); plt.ylabel("Best Union (s)")
        plt.title("Q4 Convergence")
        plt.grid(alpha = 0.3); plt.savefig(outPng, dpi = 300, bbox_inches = 'tight'); return
    tGrid = np.arange(t0, t1 + 1e-12, dt, dtype = float)
    mask = np.zeros_like(tGrid, dtype = bool)
    for i, t in enumerate(tGrid):
        m, _ = MissileState(float(t), which = MISSILE_TAG)
        centers = []
        for k in (0, 1, 2):
            Tei = bestTriple[k]["T_e"]
            if (t >= Tei - 1e-12) and (t <= Tei + SMOG_EFFECT_TIME + 1e-12):
                ex = bestTriple[k]["expl_pos"]
                centers.append(np.array([ex[0], ex[1], ex[2] - SMOG_SINK_SPEED * max(0.0, t - Tei)], dtype = float))
        if centers:
            mask[i] = ConePointsCoveredByAny(m, centers, ptsLocal, rCloud = SMOG_R)
    cum = np.cumsum(mask.astype(float)) * dt
    finalVal = float(cum[-1])
    plt.figure(figsize = (10, 8), dpi = 120)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(it, bestCurve, lw = 2); ax1.grid(alpha = 0.3)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Union (s)")
    ax1.set_title("Hybrid Optimization Convergence (best-so-far)")
    ax1.axhline(finalVal, ls = '--', lw = 1.5, c = 'r', label = f'Final best: {finalVal:.6f}s'); ax1.legend()
    ax2 = plt.subplot(2, 1, 2)
    ax2.step(tGrid, mask.astype(int), where = 'post', lw = 1.2, label = "union mask (0/1)")
    ax2_2 = ax2.twinx(); ax2_2.plot(tGrid, cum, lw = 2, alpha = 0.8, label = "cumulative")
    ax2.grid(alpha = 0.3); ax2_2.grid(alpha = 0.1)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mask"); ax2_2.set_ylabel("Cumulative (s)")
    plt.tight_layout(); plt.savefig(outPng, dpi = 300, bbox_inches = 'tight')
    print(f"[Q4] Convergence plot saved: {outPng}")

# main
def main():
    ap = argparse.ArgumentParser("Q4 Hybrid Visual Solver (cooperative)")
    ap.add_argument("--dt", type = float, default = 0.001)
    ap.add_argument("--dt-coarse", type = float, default = 0.003)
    ap.add_argument("--nphi", type = int, default = 960); ap.add_argument("--nz", type = int, default = 13)
    ap.add_argument("--backend", choices = ["process", "thread"], default = "process")
    ap.add_argument("--workers", default = "auto"); ap.add_argument("--intra-threads", default = "auto")
    ap.add_argument("--speed-grid", type = str, default = "140,130,120,110,100,90,80,70")
    ap.add_argument("--fy1-te", type = str, default = "4,12,0.02")
    ap.add_argument("--fy2-te", type = str, default = "9,22,0.02")
    ap.add_argument("--fy3-te", type = str, default = "18,36,0.02")
    ap.add_argument("--topk", type = int, default = 24)
    ap.add_argument("--min-gap", type = float, default = 0.30)
    ap.add_argument("--combo-topm", type = int, default = 64)
    ap.add_argument("--sa-iters", type = int, default = 80)
    ap.add_argument("--sa-batch", default = "auto")
    ap.add_argument("--sa-T0", type = float, default = 1.0)
    ap.add_argument("--sa-alpha", type = float, default = 0.92)
    ap.add_argument("--sigma-Te", type = float, default = 0.6)
    ap.add_argument("--sigma-v", type = float, default = 12.0)
    ap.add_argument("--local-iters", type = int, default = 40)
    ap.add_argument("--missile", choices = ["M1", "M2", "M3"], default = "M1")
    ap.add_argument("--lat-scales", type = str, default = "0.0,0.015,-0.015")  # lateral offset scales
    ap.add_argument("--overlap-lambda", type = float, default = 0.2)           # coarse evaluation overlap penalty weight
    args = ap.parse_args()

    InitWorker(args.nphi, args.nz, 1, args.missile)

    def ParseTe(s):
        a, b, st = [float(x) for x in s.split(",")]
        st = 0.05 if st <= 0 else st
        return np.arange(a, b + 1e-12, st)
    def ParseSpeed(s):
        vs = []
        for tok in s.split(","):
            if tok.strip():
                v = float(tok); vs.append(max(70.0, min(140.0, v)))
        return sorted(set(vs), reverse = True)
    def ParseLatScales(s):
        arr = []
        for tok in s.split(","):
            if tok.strip():
                arr.append(float(tok))
        if not arr: arr = [0.0]
        return arr

    speedGrid = ParseSpeed(args.speed_grid)
    te1 = ParseTe(args.fy1_te); te2 = ParseTe(args.fy2_te); te3 = ParseTe(args.fy3_te)
    latScales = ParseLatScales(args.lat_scales)

    hitTime = HITTIME_GLOBAL
    print(f"[Info] Hit time: {hitTime:.3f}s, Speed grid: {len(speedGrid)} values")

    # Stage-1: Generate candidates
    print("=" * 110)
    print("Stage-1: Generate candidates (FY1 fixed, FY2/FY3 optimized)")
    print("=" * 110)

    fy1ExplPos, fy1DropPos = ExplosionPointFromPlan(
        FY1_INIT, FY1_FIXED["speed"], FY1_FIXED["heading"], 
        FY1_FIXED["drop_time"], FY1_FIXED["fuse_delay"]
    )
    fy1Te = FY1_FIXED["drop_time"] + FY1_FIXED["fuse_delay"]
    fy1Candidate = {
        "T_e": fy1Te, "speed": FY1_FIXED["speed"], "heading": FY1_FIXED["heading"],
        "drop_time": FY1_FIXED["drop_time"], "fuse_delay": FY1_FIXED["fuse_delay"],
        "expl_pos": fy1ExplPos, "drop_pos": fy1DropPos,
        "single_duration": SingleSmokeDuration(fy1Te, fy1ExplPos, args.dt),
        "uavName": "FY1", "s_used": 0.75, "d_used": 0.0, "coop_potential": 1.0
    }

    genTasks = []
    uavs = [("FY2", FY2_INIT, te2), ("FY3", FY3_INIT, te3)]
    for name, init, teGrid in uavs:
        for Te in teGrid:
            for spd in speedGrid:
                genTasks.append({
                    "uavName": name, "uavInit": init, "T_e": float(Te),
                    "speed": float(spd), "dt": float(args.dt), "lat_scales": latScales
                })

    workers, intra = AutoBalance(args.workers, args.intra_threads, args.backend, len(genTasks))
    poolCls = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor

    print(f"backend={args.backend}, workers={workers}, intra-threads={intra}, tasks={len(genTasks)}")

    bestLists = {"FY2": [], "FY3": []}
    tA = time.time()
    if args.backend == "process":
        with poolCls(max_workers = workers, initializer = InitWorker, initargs = (args.nphi, args.nz, intra, args.missile)) as pool:
            futs = { pool.submit(CandidateWorker, pl): i for i, pl in enumerate(genTasks) }
            done, total = 0, len(futs)
            for fut in as_completed(futs):
                done += 1
                try: r = fut.result()
                except Exception: r = None
                if r is not None:
                    bestLists[r["uavName"]].append(r)
                if done % max(1, total // 20) == 0:
                    print(f"   [Gen] {int(100 * done / total)}%")
    else:
        InitWorker(args.nphi, args.nz, 1, args.missile)
        with poolCls(max_workers = workers) as pool:
            futs = { pool.submit(CandidateWorker, pl): i for i, pl in enumerate(genTasks) }
            done, total = 0, len(futs)
            for fut in as_completed(futs):
                done += 1
                try: r = fut.result()
                except Exception: r = None
                if r is not None:
                    bestLists[r["uavName"]].append(r)
                if done % max(1, total // 20) == 0:
                    print(f"   [Gen] {int(100 * done / total)}%")
    tB = time.time()
    print(f"[Stage-1] FY1=1(fixed), FY2={len(bestLists['FY2'])}, FY3={len(bestLists['FY3'])} | {tB - tA:.2f}s")

    def SelectTopkEnhanced(cands, topk, minGap):
        if not cands: return []
        pos = [c for c in cands if c["single_duration"] > 0.0]
        zer = [c for c in cands if c["single_duration"] <= 1e-12]
        pos = sorted(pos, key = lambda x: (-x["single_duration"], -x.get("coop_potential", 0.0), x["T_e"]))
        zer = sorted(zer, key = lambda x: (-x.get("coop_potential", 0.0), -x.get("s_used", 0.0), x["T_e"]))
        sel, used = [], []
        wantPos = int(round(topk * 0.70))
        for c in pos:
            T = c["T_e"]
            if all(abs(T - u) >= minGap for u in used):
                sel.append(c); used.append(T)
            if len(sel) >= wantPos: break
        for c in zer:
            if len(sel) >= topk: break
            T = c["T_e"]
            if all(abs(T - u) >= minGap for u in used):
                sel.append(c); used.append(T)
        if len(sel) < topk:
            for c in pos:
                if c in sel: continue
                T = c["T_e"]
                if all(abs(T - u) >= minGap for u in used):
                    sel.append(c); used.append(T)
                if len(sel) >= topk: break
        return sel

    fy2 = SelectTopkEnhanced(bestLists["FY2"], args.topk, args.min_gap)
    fy3 = SelectTopkEnhanced(bestLists["FY3"], args.topk, args.min_gap)

    if not (fy2 and fy3):
        print("[Warn] No feasible candidates, output all zeros.")
        cols = ['UAV_ID', 'UAV_Direction', 'UAV_Speed_ms',
                'Drop_X_m', 'Drop_Y_m', 'Drop_Z_m',
                'Explode_X_m', 'Explode_Y_m', 'Explode_Z_m',
                'Effective_Duration_s']
        pd.DataFrame([{
            'UAV_ID': n, 'UAV_Direction': 0.0, 'UAV_Speed_ms': 0.0,
            'Drop_X_m': 0.0, 'Drop_Y_m': 0.0, 'Drop_Z_m': 0.0,
            'Explode_X_m': 0.0, 'Explode_Y_m': 0.0, 'Explode_Z_m': 0.0,
            'Effective_Duration_s': 0.0
        } for n in ("FY1", "FY2", "FY3")], columns = cols).to_excel("result2.xlsx", index = False)
        print("[Result] 0.000 s | Saved result2.xlsx")
        return

    print(f"[Stage-1] Selected: FY2={len(fy2)}, FY3={len(fy3)}")

    # Stage-2: Coarse screening
    print("=" * 110); print("Stage-2: Combination coarse screening (mask OR + overlap penalty)"); print("=" * 110)

    tGridCoarse = np.arange(0.0, hitTime + 1e-12, args.dt_coarse, dtype = float)

    def MaskFor(c):
        return SingleMask(c["T_e"], c["expl_pos"], tGridCoarse)

    fy1Candidate["_mask"] = MaskFor(fy1Candidate)
    for lst in (fy2, fy3):
        for c in lst:
            c["_mask"] = MaskFor(c)

    def CoarseScoreTriple(c1, c2, c3, lam = args.overlap_lambda):
        m1, m2, m3 = c1["_mask"], c2["_mask"], c3["_mask"]
        union = (m1 | m2 | m3).sum() * args.dt_coarse
        overlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
        return float(union - lam * overlap)

    bestTriple = None
    bestCoarse = -1.0
    comboCount = 0
    
    for c2 in fy2[:min(len(fy2), 48)]:
        for c3 in fy3[:min(len(fy3), 48)]:
            comboCount += 1
            score = CoarseScoreTriple(fy1Candidate, c2, c3)
            if score > bestCoarse + 1e-12:
                bestCoarse = score
                bestTriple = (fy1Candidate, c2, c3)

    tC = time.time()
    tD = time.time()
    print(f"[Stage-2] Evaluated {comboCount} combinations, best ≈ {bestCoarse:.6f}s | {tD - tC:.2f}s")
    
    if bestTriple is None:
        print("[Warn] No feasible combination, output all zeros.")
        return

    # Stage-3: SA
    print("=" * 110); print("Stage-3: Parallel-batch SA (cooperative exact + overlap penalty)"); print("=" * 110)

    InitWorker(args.nphi, args.nz, 1, args.missile)
    exactBest = CooperativeUnionDuration(bestTriple, args.dt_coarse)
    lam = float(args.overlap_lambda)
    m1, m2, m3 = bestTriple[0]["_mask"], bestTriple[1]["_mask"], bestTriple[2]["_mask"]
    approxOverlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
    Jbest = exactBest - lam * approxOverlap
    PrintBest("Seed", exactBest, bestTriple)

    def XOf(tri):
        return np.array([tri[0]["T_e"], tri[1]["T_e"], tri[2]["T_e"],
                         tri[0]["speed"], tri[1]["speed"], tri[2]["speed"]], dtype = float)

    def ClipX(x):
        limsTe = [(te1[0], te1[-1]), (te2[0], te2[-1]), (te3[0], te3[-1])]
        out = x.copy()
        for i, (lo, hi) in enumerate(limsTe):
            out[i] = min(hi, max(lo, out[i]))
        for i in range(3, 6):
            out[i] = min(140.0, max(70.0, out[i]))
        return out

    def TriFromX(x):
        out = [fy1Candidate]
        inits = [FY2_INIT, FY3_INIT]
        for k in range(2):
            cand = BestCandidateForTeSpeed(inits[k], f"FY{k + 2}", x[k], x[3 + k], args.dt, lateralScales = latScales)
            if cand is None: return None
            cand["_mask"] = SingleMask(cand["T_e"], cand["expl_pos"], tGridCoarse)
            out.append(cand)
        return tuple(out)

    T = float(args.sa_T0); alpha = float(args.sa_alpha)
    sigTe = float(args.sigma_Te); sigV = float(args.sigma_v)
    iters = int(args.sa_iters)
    cpu = os.cpu_count() or 1
    saBatch = (max(cpu * 2, 16) if (isinstance(args.sa_batch, str) and args.sa_batch.lower() == "auto") else max(8, int(args.sa_batch)))
    history = [exactBest]

    xCur = XOf(bestTriple)
    triCur = bestTriple
    JCur = Jbest
    exactCur = exactBest

    workers2, intra2 = AutoBalance(args.workers, args.intra_threads, args.backend, totalTasks = saBatch)

    if args.backend == "process":
        pool = ProcessPoolExecutor(max_workers = workers2, initializer = InitWorker, initargs = (args.nphi, args.nz, intra2, args.missile))
    else:
        InitWorker(args.nphi, args.nz, 1, args.missile)
        pool = ThreadPoolExecutor(max_workers = workers2)

    try:
        for it in range(1, iters + 1):
            props = []
            for _ in range(saBatch):
                noise = np.array([np.random.normal(0, sigTe), np.random.normal(0, sigTe), np.random.normal(0, sigTe),
                                  np.random.normal(0, sigV), np.random.normal(0, sigV), np.random.normal(0, sigV)], dtype = float)
                props.append(ClipX(xCur + noise))

            triples = [TriFromX(x) for x in props]
            approxScores = []
            for tri in triples:
                if tri is None:
                    approxScores.append(-1e9); continue
                m1, m2, m3 = tri[0]["_mask"], tri[1]["_mask"], tri[2]["_mask"]
                union = (m1 | m2 | m3).sum() * args.dt_coarse
                overlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
                approxScores.append(float(union - lam * overlap))

            order = np.argsort(np.array(approxScores))[::-1]
            topK = min(max(8, workers2), len(order))
            pickIdx = order[:topK]
            futures = {}
            for idx in pickIdx:
                tri = triples[idx]
                if tri is None: continue
                futures[pool.submit(UnionWorker, {"triplet": tri, "dt": float(args.dt_coarse)})] = idx

            exact = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                try: val = fut.result()
                except Exception: val = 0.0
                exact[idx] = float(val)

            accepted = False
            bestIdx = None; bestVal = -1e9
            for idx, val in exact.items():
                tri = triples[idx]
                m1, m2, m3 = tri[0]["_mask"], tri[1]["_mask"], tri[2]["_mask"]
                overlap = ((m1 & m2).sum() + (m1 & m3).sum() + (m2 & m3).sum()) * args.dt_coarse
                J = val - lam * overlap
                if J > bestVal: bestVal, bestIdx = J, idx
            if bestIdx is not None:
                dJ = bestVal - JCur
                if (dJ >= 0) or (np.random.rand() < math.exp(dJ / max(T, 1e-9))):
                    triCur = triples[bestIdx]; xCur = props[bestIdx]
                    exactCur = exact[bestIdx]; JCur = bestVal
                    accepted = True
                    if exactCur > max(history) + 1e-12:
                        PrintBest(f"SA@{it}", exactCur, triCur)
                history.append(max(max(history), exactCur))
            else:
                history.append(max(history))

            T *= alpha
            if it % max(1, iters // 10) == 0:
                print(f"   [SA] iter {it}/{iters}, best≈{max(history):.4f}s, T={T:.4f}")

    finally:
        pool.shutdown(wait = True)

    bestTriple = triCur
    bestExact = exactCur

    # Stage-4: Local refinement
    print("=" * 110); print("Stage-4: Local refine"); print("=" * 110)
    def ClipFinal(x):
        x2 = x.copy()
        x2[0] = min(te1[-1], max(te1[0], x2[0]))
        x2[1] = min(te2[-1], max(te2[0], x2[1]))
        x2[2] = min(te3[-1], max(te3[0], x2[2]))
        for j in range(3, 6): x2[j] = min(140.0, max(70.0, x2[j]))
        return x2

    def XOf(tri):
        return np.array([tri[0]["T_e"], tri[1]["T_e"], tri[2]["T_e"],
                         tri[0]["speed"], tri[1]["speed"], tri[2]["speed"]], dtype = float)

    xBest = XOf(bestTriple)
    for k in range(int(args.local_iters)):
        stepT = max(0.05, args.sigma_Te * 0.25)
        stepV = max(1.0,  args.sigma_v * 0.25)
        improved = False
        for j in range(6):
            xTry = xBest.copy()
            xTry[j] += (stepT if j < 3 else stepV) * (1 if (k + j) % 2 == 0 else -1)
            xTry = ClipFinal(xTry)
            tri = TriFromX(xTry)
            if tri is None: continue
            exact = CooperativeUnionDuration(tri, args.dt)
            if exact > bestExact + 1e-12:
                bestExact = exact; bestTriple = tri; xBest = xTry
                PrintBest(f"Local@{k + 1}", bestExact, bestTriple)
                improved = True
        if not improved:
            xBest = ClipFinal(xBest + np.array([np.random.normal(0, stepT * 0.3)] * 3 + [np.random.normal(0, stepV * 0.3)] * 3))

    rows = []
    for c in sorted(bestTriple, key = lambda x: x["uavName"]):
        h = Deg(c["heading"])
        dx, dy, dz = c["drop_pos"]; ex, ey, ez = c["expl_pos"]
        rows.append({
            "UAV_ID": c["uavName"],
            "UAV_Direction": round(h, 6),
            "UAV_Speed_ms": round(c["speed"], 6),
            "Drop_X_m": round(dx, 6),
            "Drop_Y_m": round(dy, 6),
            "Drop_Z_m": round(dz, 6),
            "Explode_X_m": round(ex, 6),
            "Explode_Y_m": round(ey, 6),
            "Explode_Z_m": round(ez, 6),
            "Effective_Duration_s": round(c["single_duration"], 6)
        })
    df = pd.DataFrame(rows)[[
        'UAV_ID', 'UAV_Direction', 'UAV_Speed_ms',
        'Drop_X_m', 'Drop_Y_m', 'Drop_Z_m',
        'Explode_X_m', 'Explode_Y_m', 'Explode_Z_m',
        'Effective_Duration_s'
    ]]
    df.to_excel("result2.xlsx", index = False)
    print("-" * 110)
    print(df.to_string(index = False))
    print(f"[Result] TOTAL UNION (cooperative, no double count) ≈ {bestExact:.6f} s  |  Saved: result2.xlsx")

    PlotConvergence(bestTriple, args.dt, history, outPng = "Q4ConvergencePlot.png")

if __name__ == "__main__":
    main()