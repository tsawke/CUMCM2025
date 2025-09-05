import os
import math
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# constants
g = 9.8
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype = float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype = float)
FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype = float)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype = float)

DROP_TIME = 1.5
FUSE_DELAY_TIME = 3.6
EXPLODE_TIME = DROP_TIME + FUSE_DELAY_TIME

EPS = 1e-12

def Unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, mInit):
    dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
    v = MISSILE_SPEED * dirToOrigin
    return mInit + v * t, v

# Return: position and velocity vector
def UavStateHorizontal(t, uavInit, uavSpeed, headingRadius):
    vx = uavSpeed * math.cos(headingRadius)
    vy = uavSpeed * math.sin(headingRadius)
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype = uavInit.dtype), np.array([vx, vy, 0.0], dtype = uavInit.dtype)

"""
Parameters:
nPhi(int): Number of sampling points in circumferential direction
nZ(int): Number of sampling points in height direction, if less than 2 only top and bottom surfaces are calculated

Return:
array: Sampling point coordinates [x, y, z]
"""
def PreCalCylinderPoints(nPhi, nZ, dtype = np.float64):
    b = CYLINDER_BASE_CENTER.astype(dtype) # cylinder base center
    r, h = dtype(CYLINDER_R), dtype(CYLINDER_H)

    # circumferential angle sampling
    phis = np.linspace(0.0, 2.0 * math.pi, nPhi, endpoint = False, dtype = dtype)
    c, s = np.cos(phis), np.sin(phis)

    # base ring point sampling
    ring = np.stack([r * c, r * s, np.zeros_like(c)], axis = 1).astype(dtype)

    pts = [b + ring, b + np.array([0.0, 0.0, h], dtype = dtype) + ring]
    if nZ >= 2:
        for z in np.linspace(0.0, h, nZ, dtype = dtype):
            pts.append(b + np.array([0.0, 0.0, z], dtype = dtype) + ring)
            
    p = np.vstack(pts).astype(dtype)
    return p

def ExplosionPoint(headingRadius, tDrop, fuseDelayTime, dtype = np.float64):
    dropPos, uavV = UavStateHorizontal(tDrop, FY1_INIT.astype(dtype), dtype(UAV_SPEED), headingRadius)
    explXy = dropPos[:2] + uavV[:2] * dtype(fuseDelayTime)
    explZ = dropPos[2] - dtype(0.5) * dtype(g) * (dtype(fuseDelayTime) ** 2)
    return np.array([explXy[0], explXy[1], explZ], dtype = dtype)

# block processing, determine if point is within cone
def ConeAllPointsIn(m, c, p, rCloud = SMOG_R, margin = EPS, block = 8192):
    v = c - m
    l = np.linalg.norm(v)
    if l <= EPS or rCloud >= l:
        return True
    cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud / l) ** 2))

    for i in range(0, len(p), block):
        w = p[i : i + block] - m
        wn = np.linalg.norm(w, axis=1) + EPS
        lhs = w @ v
        rhs = wn * l * cosAlpha
        if not np.all(lhs + margin >= rhs):
            return False
    return True

# calculate occlusion for each point in time chunk, return bool array
def EvalChunk(args):
    idx0, tChunk, explPos, tExpl, pts, margin, block = args
    out = np.zeros_like(tChunk, dtype = bool)
    for i, t in enumerate(tChunk):
        m, _ = MissileState(float(t), M1_INIT)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - tExpl)], dtype = explPos.dtype)
        out[i] = ConeAllPointsIn(m, c, pts, rCloud = SMOG_R, margin = margin, block = block)
    return idx0, out

# multi-threading processing
def CalExtreme(dt = 0.0005, nphi = 960, nz = 13, backend = "process", workers = None, chunk = 800, fp32 = False, margin = EPS, block = 8192):
    if backend == "process":
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    dtype = np.float32 if fp32 else np.float64
    heading = math.atan2(-FY1_INIT[1], -FY1_INIT[0])
    explPos = ExplosionPoint(heading, DROP_TIME, FUSE_DELAY_TIME, dtype = dtype)
    if explPos[2] <= 0:
        return 0.0, explPos, float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

    hitTime = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    t0, t1 = EXPLODE_TIME, min(EXPLODE_TIME + SMOG_EFFECT_TIME, hitTime)
    tGrid = np.arange(t0, t1 + EPS, dt, dtype = dtype)

    pts = PreCalCylinderPoints(nphi, nz, dtype = dtype)

    # chunking
    chunks = [(i, tGrid[i : i + chunk]) for i in range(0, len(tGrid), chunk)]
    maskTotal = np.zeros_like(tGrid, dtype = bool)

    print(f"[Q1] steps = {len(tGrid)}, dt = {dt}, range = [{t0:.6f},{t1:.6f}]  pts = {len(pts)}  dtype = {dtype}")
    print(f"[Q1] backend = {backend}, workers = {workers or 'auto'}, chunk = {chunk}, block = {block}, margin = {margin:g}")

    tA = time.time()
    if backend == "thread":
        poolCls = ThreadPoolExecutor
    else:
        poolCls = ProcessPoolExecutor

    with poolCls(max_workers = workers) as pool:
        futs = {pool.submit(EvalChunk, (idx, c, explPos, EXPLODE_TIME, pts, margin, block)): idx for idx, c in chunks}
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            _, m = fut.result()
            l = len(m)
            maskTotal[idx : idx + l] = m
            done += 1
            if done % max(1, len(chunks) // 20) == 0:
                print(f"    [Q1] Progress {int(100 * done / len(chunks))}%")

    seconds = float(np.count_nonzero(maskTotal) * dt)
    tB = time.time()
    print(f"[Q1] Occlusion time (strict cone) = {seconds:.6f} s | Runtime {tB - tA:.2f}s")
    return seconds, explPos, hitTime

def main():
    ap = argparse.ArgumentParser("Extreme Parameters")

    # all parameters
    ap.add_argument("--dt", type = float, default = 0.0005)
    ap.add_argument("--nphi", type = int, default = 960)
    ap.add_argument("--nz", type = int, default = 13)
    ap.add_argument("--backend", choices = ["process", "thread"], default = "process")
    ap.add_argument("--workers", type = int, default = None)
    ap.add_argument("--chunk", type = int, default = 800)
    ap.add_argument("--block", type = int, default = 8192)
    ap.add_argument("--margin", type = float, default = EPS)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    print("=" * 80)
    print("Calculating Q1...")
    print(f"dt = {args.dt}, nPHI = {args.nphi}, nZ = {args.nz}, backend = {args.backend}, workers = {args.workers or 'auto'}, chunk = {args.chunk}, block = {args.block}, fp32 = {args.fp32}")
    print("=" * 80)

    sec, expl, hit = CalExtreme(dt = args.dt, nphi = args.nphi, nz = args.nz, backend = args.backend, workers = args.workers, chunk = args.chunk, fp32 = args.fp32, margin = args.margin, block = args.block)

    with open("Q1Results.txt", "w", encoding = "utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Q1 Calculation Results Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Calculation Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"dt = {args.dt}\n")
        f.write(f"nphi = {args.nphi}\n")
        f.write(f"nz = {args.nz}\n")
        f.write(f"backend = {args.backend}\n")
        f.write(f"workers = {args.workers or 'auto'}\n")
        f.write(f"chunk = {args.chunk}\n")
        f.write(f"block = {args.block}\n")
        f.write(f"margin = {args.margin:g}\n")
        f.write(f"fp32 = {args.fp32}\n\n")

        f.write("Calculation Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Occlusion duration = {sec:.6f} seconds\n")
        f.write(f"Explosion point = ({expl[0]:.6f}, {expl[1]:.6f}, {expl[2]:.6f})\n")
        f.write(f"Missile hit time ≈ {hit:.6f} seconds\n\n")

        f.write("=" * 80 + "\n")

    print("-" * 80)
    print(f"[Result] Occlusion duration = {sec:.6f} s")
    print(f"[Info] Explosion point = ({expl[0]:.6f}, {expl[1]:.6f}, {expl[2]:.6f})")
    print(f"[Info] Missile hit time approx= {hit:.6f} s")
    print(f"[Info] Results saved to Q1Results.txt")

if __name__ == "__main__":
    main()
