import os
import math
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# import Q1 geometry/criteria or fallback
Q1_IMPORTED = False
try:
    import Q1Solver as Q1
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
except Exception:
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
        v = MISSILE_SPEED * Unit(FAKE_TARGET_ORIGIN - mInit)
        return mInit + v * t, v

    def UavStateHorizontal(t, uavInit, uavSpeed, headingRad):
        vx = uavSpeed * math.cos(headingRad)
        vy = uavSpeed * math.sin(headingRad)
        return np.array([uavInit[0] + vx * t,
                         uavInit[1] + vy * t,
                         uavInit[2]], dtype = float), np.array([vx, vy, 0.0], dtype = float)

    CYL_BASE = np.array([0.0, 200.0, 0.0], dtype = float)
    CYL_R, CYL_H = 7.0, 10.0
    def PreCalCylinderPoints(nPhi, nZ, dtype = np.float64):
        b = CYL_BASE.astype(dtype)
        r, h = dtype(CYL_R), dtype(CYL_H)
        phis = np.linspace(0.0, 2.0*math.pi, nPhi, endpoint = False, dtype = dtype)
        c, s = np.cos(phis), np.sin(phis)
        ring = np.stack([r*c, r*s, np.zeros_like(c)], axis=1).astype(dtype)
        pts = [b + ring, b + np.array([0.0,0.0,h], dtype=dtype) + ring]
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
            wn = np.linalg.norm(w, axis=1) + EPS
            lhs = w @ v
            rhs = wn * l * cosAlpha
            if not np.all(lhs + margin >= rhs):
                return False
        return True

SPEED_MIN, SPEED_MAX = 70.0, 140.0
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DROP_MIN, DROP_MAX = 0.0, 60.0
FUSE_MIN, FUSE_MAX = 0.0, 18.0

def ClampSpeed(v: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(v)))

def HeadingToUnit(headingRad: float) -> np.ndarray:
    return np.array([math.cos(headingRad), math.sin(headingRad), 0.0], dtype = float)

def MissileHitTime() -> float:
    return float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

def ExplosionFromTimes(headingRad: float, speed: float, tDrop: float, tau: float, dtype = np.float64):
    dropPos, uavV = UavStateHorizontal(tDrop, FY1_INIT.astype(dtype), dtype(speed), headingRad)
    explXy = dropPos[:2] + uavV[:2] * dtype(tau)
    explZ  = dropPos[2] - dtype(0.5) * dtype(g) * (dtype(tau)**2)
    return np.array([explXy[0], explXy[1], explZ], dtype = dtype)

def EvaluateOcclusionFixedHv(heading: float, speed: float,
                                tDrop: float, tau: float,
                                dt = 0.002, nphi = 480, nz = 9,
                                margin = EPS, block = 8192) -> Tuple[float, np.ndarray, float]:
    speed = ClampSpeed(speed)
    t0 = tDrop + tau
    hit = MissileHitTime()
    if t0 >= hit:
        return 0.0, np.array([0,0,-1.0], dtype = float), hit

    expl = ExplosionFromTimes(heading, speed, tDrop, tau, dtype = float)
    if expl[2] <= 0.0:
        return 0.0, expl, hit

    t1 = min(t0 + SMOG_EFFECT_TIME, hit)
    if t1 <= t0:
        return 0.0, expl, hit

    tGrid = np.arange(t0, t1 + EPS, dt, dtype = float)
    pts = PreCalCylinderPoints(nphi, nz, dtype = float)

    mask = np.zeros(len(tGrid), dtype = bool)
    for i, t in enumerate(tGrid):
        mPos, _ = MissileState(float(t), M1_INIT)
        cPos = np.array([expl[0], expl[1], expl[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype = float)
        mask[i] = ConeAllPointsIn(mPos, cPos, pts, rCloud = SMOG_R, margin = margin, block = block)

    return float(np.count_nonzero(mask) * dt), expl, hit

def EvalTuple(args):
    heading, speed, tDrop, tau, dt, nphi, nz, margin, block = args
    sec, expl, hit = EvaluateOcclusionFixedHv(heading, speed, tDrop, tau, dt, nphi, nz, margin, block)
    return sec, (tDrop, tau), expl, hit

def LhsRect(n: int, bounds: List[Tuple[float,float]]) -> np.ndarray:
    rng = np.random.default_rng()
    d = len(bounds)
    u = (rng.random((n,d)) + np.arange(n)[:,None]) / n
    rng.shuffle(u, axis=0)
    out = np.empty_like(u)
    for j,(lo,hi) in enumerate(bounds):
        out[:,j] = lo + u[:,j]*(hi-lo)
    out[:,0] = np.maximum(out[:,0], 0.02*(bounds[0][1]-bounds[0][0])/n)
    return out

def PatternSearch2d(heading: float, speed: float, x0: np.ndarray,
                      evalKwargs: dict,
                      steps = (0.6, 0.4),
                      shrink = 0.6, maxIter = 60,
                      workers = None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    bestVal, bestExpl, bestHit = EvaluateOcclusionFixedHv(
        heading, speed, x0[0], x0[1], **evalKwargs)
    hist = [bestVal]
    s = np.array(steps, dtype=float)

    for it in range(maxIter):
        cands = np.array([
            [x0[0]+s[0], x0[1]], [x0[0]-s[0], x0[1]],
            [x0[0], x0[1]+s[1]], [x0[0], x0[1]-s[1]],
            [x0[0]+s[0], x0[1]+s[1]], [x0[0]-s[0], x0[1]-s[1]],
            [x0[0]+s[0], x0[1]-s[1]], [x0[0]-s[0], x0[1]+s[1]],
        ], dtype = float)

        cands[:,0] = np.clip(cands[:,0], DROP_MIN, DROP_MAX)
        cands[:,1] = np.clip(cands[:,1], FUSE_MIN, FUSE_MAX)

        tasks = [(heading, speed, c[0], c[1],
                  evalKwargs["dt"], evalKwargs["nphi"], evalKwargs["nz"],
                  evalKwargs["margin"], evalKwargs["block"]) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers = workers) as pool:
            futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, (td,tf), expl, hit = fut.result()
                vals.append((sec, np.array([td,tf], dtype = float), expl, hit))

        improved = False
        for sec, x, expl, hit in vals:
            if sec > bestVal + 1e-12:
                bestVal, bestExpl, bestHit = sec, expl, hit
                x0 = x
                improved = True

        hist.append(bestVal)
        if improved:
            continue

        s *= shrink
        if np.all(s < np.array([1e-3, 1e-3])):
            break

    return bestVal, x0, bestExpl, bestHit, hist

def main():
    ap = argparse.ArgumentParser("Q2 Optimize (Given heading & speed)")
    ap.add_argument("--heading-deg", type = float, default = None, help = "heading angle (degrees)")
    ap.add_argument("--heading-rad", type = float, default = None, help = "heading angle (radians)")
    ap.add_argument("--speed", type = float, required = True, help = "FY1 speed (m/s), auto-clamped to [70,140]")

    ap.add_argument("--lhs", type = int, default = 4096, help = "LHS coarse screening samples")
    ap.add_argument("--topk", type = int, default = 32, help = "candidates entering Pattern")
    ap.add_argument("--pat-iter", type = int, default = 60, help = "Pattern search max iterations")
    ap.add_argument("--workers", default = "auto", help = "parallel processes, int or 'auto'")

    ap.add_argument("--dt-coarse", type = float, default = 0.002)
    ap.add_argument("--nphi-coarse", type = int, default = 480)
    ap.add_argument("--nz-coarse", type = int, default = 9)

    ap.add_argument("--dt-final", type = float, default = 0.001)
    ap.add_argument("--nphi-final", type = int, default = 960)
    ap.add_argument("--nz-final", type = int, default = 13)

    args = ap.parse_args()

    if (args.heading_deg is None) == (args.heading_rad is None):
        raise SystemExit("Please specify heading angle with --heading-deg or --heading-rad.")
    heading = (args.heading_rad if args.heading_rad is not None else math.radians(args.heading_deg))
    speed = ClampSpeed(args.speed)

    workers = None if str(args.workers).lower() == "auto" else int(args.workers)

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    tA = time.time()
    bounds = [(DROP_MIN, DROP_MAX), (FUSE_MIN, FUSE_MAX)]
    samples = LhsRect(args.lhs, bounds)

    tasks = [(heading, speed, s[0], s[1],
              args.dt_coarse, args.nphi_coarse, args.nz_coarse, EPS, 8192) for s in samples]

    vals = []
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        done, stride = 0, max(1, len(tasks)//20)
        bestSoFar = 0.0
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            vals.append((sec, x, expl, hit))
            done += 1
            if sec > bestSoFar: bestSoFar = sec
            if done % stride == 0 or done == len(tasks):
                pct = int(100*done/len(tasks))
                print(f"[LHS] Progress {pct:3d}%  best_so_far={bestSoFar:.6f}")

    vals.sort(key=lambda z: z[0], reverse=True)
    top = vals[:min(args.topk, len(vals))]
    if not top or top[0][0] <= 0.0:
        print("[Info] LHS found no valid occlusion (>0), will still try Pattern refinement.")

    refined = []
    globalBest = 0.0
    histAll = []
    for i,(sec0, x0, expl0, hit0) in enumerate(top, start=1):
        print(f"[PAT] seed#{i:02d} initial={sec0:.6f}")
        bestSec, bestX, bestExpl, bestHit, hist = PatternSearch2d(
            heading, speed, x0, evalKwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            steps=(0.8, 0.6), shrink=0.6, maxIter=args.pat_iter, workers=workers
        )
        refined.append((bestSec, bestX, bestExpl, bestHit))
        histAll.append(hist)
        globalBest = max(globalBest, bestSec)
        if i % max(1, len(top)//20) == 0 or i == len(top):
            pct = int(100*i/len(top))
            print(f"[PAT] Progress {pct:3d}%  global_best={globalBest:.6f}")

    refined.sort(key=lambda z: z[0], reverse=True)
    candidates = refined[:min(10, len(refined))] if refined else top[:min(10, len(top))]

    finals = []
    tasks = [(heading, speed, c[1][0], c[1][1],
              args.dt_final, args.nphi_final, args.nz_final, EPS, 8192) for c in candidates]
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futs = {pool.submit(EvalTuple, t): i for i,t in enumerate(tasks)}
        done, stride = 0, max(1, len(tasks)//20)
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            finals.append((sec, x, expl, hit))
            done += 1
            if done % stride == 0 or done == len(tasks):
                pct = int(100*done/len(tasks))
                print(f"[FINAL] Progress {pct:3d}%")

    if finals:
        finals.sort(key=lambda z: z[0], reverse=True)
        best = finals[0]
        bestSec, bestX, bestExpl, bestHit = best
    else:
        bestSec, bestX, bestExpl, bestHit = top[0]

    tB = time.time()
    print("="*80)
    print(f"[Best] heading={heading:.6f} rad, speed={speed:.3f} m/s")
    print(f"       t_drop={bestX[0]:.3f} s, tau={bestX[1]:.3f} s, t0={bestX[0]+bestX[1]:.3f} s")
    print(f"       explosion=({bestExpl[0]:.2f},{bestExpl[1]:.2f},{bestExpl[2]:.2f})")
    print(f"       occlusion={bestSec:.6f} s | total time={tB - tA:.2f}s")
    print("="*80)

    SaveReport("Q2_HV_Results.txt", heading, speed, bestX, bestExpl, bestSec,
                dict(dt_coarse=args.dt_coarse, nphi_coarse=args.nphi_coarse, nz_coarse=args.nz_coarse,
                     dt_final=args.dt_final, nphi_final=args.nphi_final, nz_final=args.nz_final))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = (9,6))
        ax = fig.add_subplot(111)
        for j,h in enumerate(histAll, start=1):
            ax.plot(range(len(h)), h, linewidth = 1.1, alpha = 0.6)
        ax.axhline(y = bestSec, linestyle = '--', linewidth = 2, label = f'Final Best: {bestSec:.6f}s')
        ax.set_xlabel('Pattern iteration (per seed)')
        ax.set_ylabel('Best-so-far (s)')
        ax.set_title('Convergence (Pattern refinement, top-K seeds)')
        ax.grid(True, alpha = 0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("Q2_HV_Convergence.png", dpi = 300, bbox_inches = 'tight')
        plt.close(fig)
        print("[Plot] Saved Q2_HV_Convergence.png")
    except Exception as e:
        print(f"[Plot] Skip: {e}")

def SaveReport(filename: str, heading: float, speed: float,
                bestX: np.ndarray, bestExpl: np.ndarray, bestSec: float,
                cfg: Dict):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Q2 Optimize (Given heading & speed) - Results Report\n")
        f.write("="*80 + "\n\n")
        f.write("Fixed Inputs:\n")
        f.write("-"*40 + "\n")
        f.write(f"Heading (rad) = {heading:.9f}\n")
        f.write(f"Speed (m/s)   = {speed:.3f}  (clamped to [70,140])\n\n")

        f.write("Optimized Variables:\n")
        f.write("-"*40 + "\n")
        f.write(f"t_drop (s) = {bestX[0]:.6f}\n")
        f.write(f"tau   (s)  = {bestX[1]:.6f}\n")
        f.write(f"t0    (s)  = {bestX[0]+bestX[1]:.6f}\n")
        f.write(f"Explosion point = ({bestExpl[0]:.6f}, {bestExpl[1]:.6f}, {bestExpl[2]:.6f})\n\n")

        f.write("Evaluation Config:\n")
        f.write("-"*40 + "\n")
        f.write(f"coarse: dt={cfg['dt_coarse']}, nphi={cfg['nphi_coarse']}, nz={cfg['nz_coarse']}\n")
        f.write(f"final : dt={cfg['dt_final']}, nphi={cfg['nphi_final']}, nz={cfg['nz_final']}\n\n")

        f.write("Result:\n")
        f.write("-"*40 + "\n")
        f.write(f"Occlusion duration = {bestSec:.6f} s\n")
        f.write("="*80 + "\n")
    print(f"[Info] Results saved to {filename}")

if __name__ == "__main__":
    main()
