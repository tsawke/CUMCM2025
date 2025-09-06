# -*- coding: utf-8 -*-
# Q5Solver_numba.py
# Numba并行时间循环版本（prange 真正多核），带进度日志、两阶段评估与Excel/报告输出

import os, math, time, json, argparse, random
from copy import deepcopy
import numpy as np
import pandas as pd

# ========== 尝试加载 Numba ==========
NUMBA_OK = False
try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ========== 常量与场景 ==========
g = 9.8
MISSILE_SPEED = 300.0
SMOG_R = 10.0
SMOG_SINK = 3.0
SMOG_T = 20.0

FAKE = np.array([0.,0.,0.], dtype=np.float64)
CYL_R, CYL_H = 7.0, 10.0
CYL_BASE = np.array([0.,200.,0.], dtype=np.float64)

M_INITS = [
    np.array([20000.,    0., 2000.], dtype=np.float64),
    np.array([19000.,  600., 2100.], dtype=np.float64),
    np.array([18000., -600., 1900.], dtype=np.float64),
]
FY_INITS = [
    np.array([17800.,     0., 1800.], dtype=np.float64),
    np.array([12000.,  1400., 1400.], dtype=np.float64),
    np.array([ 6000., -3000.,  700.], dtype=np.float64),
    np.array([11000.,  2000., 1800.], dtype=np.float64),
    np.array([13000., -2000., 1300.], dtype=np.float64),
]

UAV_MIN, UAV_MAX = 70., 140.
MAX_BOMBS, MIN_GAP = 3, 1.0
DEFAULT_DELAY = 3.6
EPS = 1e-12

# ========== 工具 ==========
def unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v / n

def missile_pos_py(t, m0):
    return m0 + MISSILE_SPEED * unit(FAKE - m0) * t

def uav_state(t, u0, spd, hd):
    return np.array([u0[0] + spd*math.cos(hd)*t, u0[1] + spd*math.sin(hd)*t, u0[2]], dtype=np.float64)

def explosion(u0, spd, hd, t_drop, delay):
    p_drop = uav_state(t_drop, u0, spd, hd)
    p_expl = p_drop + np.array([spd*math.cos(hd)*delay, spd*math.sin(hd)*delay, -0.5*g*delay*delay], dtype=np.float64)
    return p_expl, t_drop + delay

# 采样缓存
_CYL_CACHE = {}
def sample_cylinder(nphi, nz, dtype):
    key = (nphi, nz, np.dtype(dtype).name)
    if key in _CYL_CACHE:
        return _CYL_CACHE[key]
    b = CYL_BASE.astype(dtype); r, h = dtype(CYL_R), dtype(CYL_H)
    ph = np.linspace(0.0, 2*np.pi, nphi, endpoint=False, dtype=dtype)
    ring = np.stack([r*np.cos(ph), r*np.sin(ph), np.zeros_like(ph)], axis=1)
    pts = [b + ring, b + np.array([0,0,h], dtype=dtype) + ring]
    if nz >= 2:
        for z in np.linspace(0.0, h, nz, dtype=dtype):
            pts.append(b + np.array([0,0,z], dtype=dtype) + ring)
    arr = np.vstack(pts).astype(dtype, copy=False)
    _CYL_CACHE[key] = arr
    return arr

# ========== Numba 核心：时间维并行 ==========
if NUMBA_OK:
    @njit(parallel=True, fastmath=True)
    def _blk_eval_numba(t_blk, m0, epos, et0, pts, mode, smog_r, smog_sink, smog_t, eps, fake, missile_speed):
        # t_blk: (Tb,), m0:(3,), epos:(C,3), et0:(C,), pts:(P,3)
        Tb = t_blk.shape[0]
        mask = np.zeros(Tb, dtype=np.bool_)
        # 预先计算导弹方向（常量）
        vdx = fake[0] - m0[0]
        vdy = fake[1] - m0[1]
        vdz = fake[2] - m0[2]
        vn = math.sqrt(vdx*vdx + vdy*vdy + vdz*vdz)
        if vn <= eps:
            vdx = 0.0; vdy = 0.0; vdz = 0.0
        else:
            vdx /= vn; vdy /= vn; vdz /= vn

        for i in prange(Tb):  # 并行 over 时间
            t = float(t_blk[i])

            # 统计活跃云
            cnt = 0
            for k in range(et0.shape[0]):
                if (t >= et0[k]) and (t <= et0[k] + smog_t):
                    cnt += 1
            if cnt == 0:
                mask[i] = False
                continue
            idx = np.empty(cnt, dtype=np.int64)
            p = 0
            for k in range(et0.shape[0]):
                if (t >= et0[k]) and (t <= et0[k] + smog_t):
                    idx[p] = k; p += 1

            # 导弹位置 m(t)
            mx = m0[0] + missile_speed * vdx * t
            my = m0[1] + missile_speed * vdy * t
            mz = m0[2] + missile_speed * vdz * t

            # 中心先验命中（快速路径）
            hit_center = False
            for a in range(cnt):
                k = idx[a]
                cx = epos[k,0] - mx
                cy = epos[k,1] - my
                cz = (epos[k,2] - smog_sink*(t - et0[k])) - mz
                l = math.sqrt(cx*cx + cy*cy + cz*cz)
                if l <= smog_r + eps:
                    hit_center = True
                    break
            if hit_center:
                mask[i] = True
                continue

            # 多云协同：点上 OR，点间 AND
            all_pts_ok = True
            P = pts.shape[0]
            for pi in range(P):
                wx = pts[pi,0] - mx
                wy = pts[pi,1] - my
                wz = pts[pi,2] - mz
                wn = math.sqrt(wx*wx + wy*wy + wz*wz)
                covered = False
                for a in range(cnt):
                    k = idx[a]
                    vx = epos[k,0] - mx
                    vy = epos[k,1] - my
                    vz = (epos[k,2] - smog_sink*(t - et0[k])) - mz
                    l = math.sqrt(vx*vx + vy*vy + vz*vz)
                    if l <= eps:
                        covered = True
                        break
                    # 锥体余弦判据
                    cosA = 0.0
                    tmp = smog_r / l
                    val = 1.0 - tmp*tmp
                    if val > 0.0:
                        cosA = math.sqrt(val)
                    if (wx*vx + wy*vy + wz*vz) + 1e-12 >= wn * (l * cosA):
                        covered = True
                        break
                if not covered:
                    all_pts_ok = False
                    break
            mask[i] = all_pts_ok
        return mask

# ========== 方案展开/评估 ==========
def expand_solution(sol, delays=None, dtype=np.float64):
    epos, et0 = [], []
    for u, (hd, spd, drops) in enumerate(sol):
        u0 = FY_INITS[u]
        last = -1e9
        for j, t in enumerate(drops):
            if t <= 0: 
                continue
            if t - last < MIN_GAP:
                t = last + MIN_GAP
            last = t
            dly = (delays[u][j] if (delays and delays[u][j] > 0) else DEFAULT_DELAY)
            p, t0 = explosion(u0, spd, hd, t, dly)
            if p[2] > 0:
                epos.append(p); et0.append(t0)
    if not epos:
        return np.zeros((0,3), dtype=dtype), np.zeros((0,), dtype=dtype)
    return np.ascontiguousarray(np.stack(epos, 0).astype(dtype)), np.ascontiguousarray(np.array(et0, dtype=dtype))

def eval_missile_numba(m0, epos, et0, dt, t0, t1, pts, mode):
    # 构建时间网格并调用 numba 内核
    t_grid = np.arange(t0, t1 + 1e-12, dt, dtype=pts.dtype)
    if t_grid.size == 0:
        return t_grid, np.zeros(0, dtype=bool), 0.0
    mask = _blk_eval_numba(t_grid, m0.astype(pts.dtype), epos, et0, pts, mode,
                           float(SMOG_R), float(SMOG_SINK), float(SMOG_T), float(EPS),
                           FAKE.astype(pts.dtype), float(MISSILE_SPEED))
    seconds = float(np.count_nonzero(mask) * dt)
    return t_grid, mask, seconds

def eval_missile_python(m0, epos, et0, dt, t0, t1, pts, mode):
    # 旧版纯 numpy 路径（用于无 Numba 环境）
    t_grid = np.arange(t0, t1 + 1e-12, dt, dtype=pts.dtype)
    if t_grid.size == 0:
        return t_grid, np.zeros(0, dtype=bool), 0.0

    out = np.zeros(t_grid.size, dtype=bool)
    for i, t in enumerate(t_grid):
        alive = np.where((t >= et0) & (t <= et0 + SMOG_T))[0]
        if alive.size == 0:
            continue
        m = missile_pos_py(float(t), m0)
        c = epos[alive].copy()
        c[:,2] -= SMOG_SINK * (float(t) - et0[alive])

        v = c - m
        l = np.linalg.norm(v, axis=1)
        if np.any(l <= SMOG_R + EPS):
            out[i] = True
            continue

        cosA = np.sqrt(np.maximum(0.0, 1.0 - (SMOG_R / l)**2))
        w = pts - m
        wn = np.linalg.norm(w, axis=1)
        lhs = w @ v.T
        rhs = (wn[:,None]) * (l * cosA)[None,:]
        if mode == 0:
            ok_all = np.all(lhs + 1e-12 >= rhs, axis=0)
            out[i] = bool(np.any(ok_all))
        else:
            covered_any = np.any(lhs + 1e-12 >= rhs, axis=1)
            out[i] = bool(np.all(covered_any))
    seconds = float(np.count_nonzero(out) * dt)
    return t_grid, out, seconds

def evaluate(sol, dt, nphi, nz, fp32, use_numba=True, delays=None,
             lambda_overlap=0.0, sigma=2.5):
    dtype = np.float32 if fp32 else np.float64
    pts = sample_cylinder(nphi, nz, dtype)
    epos, et0 = expand_solution(sol, delays, dtype)
    if epos.shape[0] == 0:
        return 0.0, {"per_missile_seconds":[0.,0.,0.], "tGrids":[np.array([])]*3, "masks":[np.array([],bool)]*3}

    mode = 1  # 多云协同（默认）
    per, tGs, mks = [], [], []
    for m0 in M_INITS:
        hit = float(np.linalg.norm(m0 - FAKE) / MISSILE_SPEED)
        t0 = float(np.min(et0))
        t1 = min(float(np.max(et0)) + SMOG_T, hit)
        if t1 <= t0:
            tGs.append(np.array([], dtype)); mks.append(np.array([], dtype=bool)); per.append(0.0)
            continue
        if use_numba and NUMBA_OK:
            tg, mk, sec = eval_missile_numba(m0, epos, et0, dt, t0, t1, pts, mode)
        else:
            tg, mk, sec = eval_missile_python(m0, epos, et0, dt, t0, t1, pts, mode)
        tGs.append(tg); mks.append(mk); per.append(sec)

    total = float(sum(per))
    if lambda_overlap > 0.0 and et0.size >= 2:
        d = np.diff(np.sort(et0.astype(float)))
        d = d[d <= 3.0 * sigma]
        if d.size:
            total -= lambda_overlap * float(np.sum(np.exp(-(d*d)/(2*sigma*sigma))))
    return total, {"per_missile_seconds": per, "tGrids": tGs, "masks": mks}

# ========== 构造/扰动/退火 ==========
def greedy_seed(dt=0.02, nphi=240, nz=7, fp32=False, use_numba=True):
    sol = []
    for u, u0 in enumerate(FY_INITS):
        hd = math.atan2(-u0[1], -u0[0])
        sol.append([hd, 120.0, [0.,0.,0.]])  # heading, speed, drops(3)
    hits = [np.linalg.norm(m - FAKE)/MISSILE_SPEED for m in M_INITS]
    tmin = max(0.5, min(hits) - 8.0); tmax = max(hits) - 1.0
    cand = np.arange(tmin, tmax, 1.0, dtype=float)

    base,_ = evaluate(sol, dt, nphi, nz, fp32, use_numba)
    used = 0
    while used < 15:
        best=None; gain=-1e-9
        for u in range(5):
            drops = sol[u][2]
            k = sum(1 for x in drops if x > 0)
            if k >= MAX_BOMBS: 
                continue
            prev = [x for x in drops if x > 0]
            for t in cand:
                if any(abs(t-p) < MIN_GAP for p in prev): 
                    continue
                tmp = deepcopy(sol); tmp[u][2][k] = float(t)
                val,_ = evaluate(tmp, dt, nphi, nz, fp32, use_numba)
                if val - base > gain:
                    gain, best = val - base, (u,k,float(t),val)
        if not best or gain < 0.05:
            break
        u,k,t,val = best
        sol[u][2][k] = t; base = val; used += 1
        print(f"[SEED] FY{u+1} add #{k+1}@{t:.2f}s gain=+{gain:.3f}s used={used}")
    return sol

def clamp(x,a,b): 
    return a if x < a else (b if x > b else x)

def perturb(sol):
    s = deepcopy(sol); i = random.randrange(5); hd,spd,dr = s[i]
    r = random.random()
    if r < 0.33:
        s[i][0] = (hd + math.radians(random.uniform(-15,15)) + 2*math.pi) % (2*math.pi)
    elif r < 0.66:
        s[i][1] = clamp(spd + random.uniform(-10,10), UAV_MIN, UAV_MAX)
    else:
        j = random.randrange(MAX_BOMBS); base = dr[j]
        dr[j] = random.uniform(3,10) if (base<=0 and random.random()<0.5) else max(0.0, base + random.uniform(-2,2))
        arr = sorted([x for x in dr if x>0]); fixed=[]; last=-1e9
        for t in arr:
            if t - last < MIN_GAP: t = last + MIN_GAP
            fixed.append(t); last = t
        while len(fixed) < MAX_BOMBS:
            fixed.append(0.0)
        s[i][2] = fixed[:MAX_BOMBS]
    return s

def simulated_annealing(dt, nphi, nz, fp32, iters, batch, restarts,
                        lambda_overlap=0.0, sigma=2.5,
                        progress_every=20, time_budget=None,
                        use_numba=True, numba_threads=None):
    # 设置 numba 线程数（若启用）
    if use_numba and NUMBA_OK and numba_threads and numba_threads > 0:
        try:
            set_num_threads(int(numba_threads))
            print(f"[INFO] Numba threads set to {get_num_threads()}")
        except Exception:
            pass

    bestV, bestS, bestD = None, None, None
    eval_count = 0
    t0 = time.time()

    for rs in range(restarts):
        cur = greedy_seed(0.02, 240, 7, fp32, use_numba)
        curV, curD = evaluate(cur, dt, nphi, nz, fp32, use_numba, None, lambda_overlap, sigma)
        eval_count += 1
        if bestV is None or curV > bestV:
            bestV, bestS, bestD = curV, deepcopy(cur), curD
            print(f"[INFO] Restart {rs+1}/{restarts} init best={bestV:.3f}s per={bestD['per_missile_seconds']}")
        T = max(1.0, 0.25*(bestV if bestV>0 else 10.0)+5.0)

        for it in range(1, iters+1):
            cands = [perturb(cur) for _ in range(batch)]
            results = [None]*batch
            # 串行评估（Numba 已经内部多线程，不需要外层并行）
            for bi, c in enumerate(cands):
                v, d = evaluate(c, dt, nphi, nz, fp32, use_numba, None, lambda_overlap, sigma)
                results[bi] = (v, d)

            acc = 0
            for (v,d),c in zip(results, cands):
                eval_count += 1
                dv = v - curV
                if dv >= 0 or math.exp(dv/max(T,1e-6)) > random.random():
                    cur, curV, curD = c, v, d; acc += 1
                    if curV > bestV:
                        bestV, bestS, bestD = curV, deepcopy(cur), curD
                        pm = bestD['per_missile_seconds']
                        print(f"[INFO] New best={bestV:.3f}s | M1={pm[0]:.2f} M2={pm[1]:.2f} M3={pm[2]:.2f} | eval={eval_count} | {time.time()-t0:.1f}s")
            T *= 0.985

            if it % max(1, progress_every) == 0:
                elapsed = time.time() - t0
                eps = eval_count / max(elapsed, 1e-6)
                done = ((rs*iters + it) / max(restarts*iters,1))
                eta = (elapsed/done - elapsed) if done > 0 else float('inf')
                pm = curD['per_missile_seconds']
                print(f"[SA] rs={rs+1}/{restarts} it={it}/{iters} "
                      f"T={T:.2f} cur={curV:.2f} best={bestV:.2f} acc={acc}/{batch} "
                      f"eval={eval_count} ({eps:.1f}/s) ETA≈{eta:.1f}s pm={pm}")

            if (time_budget is not None) and (time.time()-t0) > time_budget:
                print("[WARN] hit time budget, stop SA early.")
                return bestS, bestV, bestD

    return bestS, bestV, bestD

# ========== 输出 ==========
def write_excel(xlsx, sol, detail):
    from openpyxl import Workbook
    wb = Workbook()
    ws1 = wb.active; ws1.title = "Plan"
    ws1.append(["UAV","HeadingDeg","Speed","BombIndex","DropTime","ExplodeTime","ExplodeX","ExplodeY","ExplodeZ"])
    for i,(hd,spd,drops) in enumerate(sol):
        u0 = FY_INITS[i]; last = -1e9
        for j,t in enumerate(drops):
            if t <= 0:
                ws1.append([f"FY{i+1}", (hd*180/math.pi)%360.0, spd, j+1, 0, 0, None,None,None]); continue
            if t - last < MIN_GAP: t = last + MIN_GAP
            last = t
            p,t0 = explosion(u0,spd,hd,t,DEFAULT_DELAY)
            ws1.append([f"FY{i+1}", (hd*180/math.pi)%360.0, spd, j+1, t, t0,
                        None if p[2]<=0 else float(p[0]),
                        None if p[2]<=0 else float(p[1]),
                        None if p[2]<=0 else float(p[2])])
    ws2 = wb.create_sheet("Summary")
    pm = detail["per_missile_seconds"]; total = float(sum(pm))
    for r in [["Missile","Occlusion_s"],["M1",pm[0]],["M2",pm[1]],["M3",pm[2]],["Total",total]]:
        ws2.append(r)
    wb.save(xlsx)
    print(f"[INFO] result written -> {xlsx}")
    return total, pm

def save_report(path, args, total, pm, sol):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Q5 Numba Solver Report\n"+"="*70+"\n\n")
        f.write("Args:\n"+json.dumps(vars(args),indent=2,ensure_ascii=False)+"\n\n")
        f.write(f"Total Occlusion = {total:.3f}s\nPer Missile: M1={pm[0]:.3f}s  M2={pm[1]:.3f}s  M3={pm[2]:.3f}s\n\n")
        f.write("Solution:\n")
        for i,(hd,spd,drops) in enumerate(sol):
            f.write(f" FY{i+1}: heading {(hd*180/math.pi)%360:.2f}°, speed {spd:.2f}, drops {drops}\n")
    print(f"[INFO] report saved -> {path}")

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser("Q5 high-parallel solver with Numba (time-loop JIT)")
    # 粗评
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--nphi", type=int, default=720)
    ap.add_argument("--nz",   type=int, default=11)
    # 细评
    ap.add_argument("--refine_dt", type=float, default=0.005)
    ap.add_argument("--refine_nphi", type=int, default=960)
    ap.add_argument("--refine_nz",   type=int, default=13)

    # 评估与数值
    ap.add_argument("--fp32", action="store_true", help="使用 float32（更快，略降精度）")
    ap.add_argument("--use_numba", action="store_true", help="强制启用 numba 时间并行")
    ap.add_argument("--no_numba", dest="use_numba", action="store_false", help="禁用 numba（回退 numpy）")
    ap.set_defaults(use_numba=True)
    ap.add_argument("--numba_threads", type=int, default=None, help="Numba 并行线程数（缺省用默认）")

    # SA 搜索
    ap.add_argument("--sa_iters", type=int, default=1200)
    ap.add_argument("--sa_batch", type=int, default=12)
    ap.add_argument("--sa_restarts", type=int, default=2)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--time_budget", type=float, default=None)

    ap.add_argument("--lambda_overlap", type=float, default=0.0)
    ap.add_argument("--overlap_sigma", type=float, default=2.5)

    # 输出
    ap.add_argument("--xlsx", default="result3.xlsx")
    ap.add_argument("--report", default="Result3_report.txt")
    args = ap.parse_args()

    # 环境整备：启用 numba 时，默认约束 MKL 线程，避免抢核（可手工改）
    if args.use_numba and NUMBA_OK:
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        print(f"[INFO] Numba available. MKL/OMP defaulted to 1 to avoid oversubscription.")
    elif args.use_numba and not NUMBA_OK:
        print("[WARN] --use_numba 指定但未检测到 Numba，自动回退到 numpy 路径。")
        args.use_numba = False

    if args.numba_threads:
        try:
            set_num_threads(int(args.numba_threads))
            print(f"[INFO] Numba threads set to {get_num_threads()}")
        except Exception:
            print("[WARN] 设置 Numba 线程数失败，使用默认。")

    print("="*78)
    print(f"[RUN] use_numba={args.use_numba} numba_threads={args.numba_threads or 'auto'} fp32={args.fp32}")
    print(f"[RUN] SA iters={args.sa_iters} batch={args.sa_batch} restarts={args.sa_restarts}")
    print("="*78)

    t0 = time.time()

    # 阶段1：粗评 + SA
    sol, bestV, detail = simulated_annealing(
        dt=args.dt, nphi=args.nphi, nz=args.nz,
        fp32=args.fp32, iters=args.sa_iters, batch=args.sa_batch, restarts=args.sa_restarts,
        lambda_overlap=args.lambda_overlap, sigma=args.overlap_sigma,
        progress_every=args.progress_every, time_budget=args.time_budget,
        use_numba=args.use_numba, numba_threads=args.numba_threads
    )

    # 阶段2：细粒度复评与输出
    print(f"[INFO] Refining measurement dt={args.refine_dt}, nphi={args.refine_nphi}, nz={args.refine_nz} ...")
    total, detail = evaluate(sol, args.refine_dt, args.refine_nphi, args.refine_nz, args.fp32, args.use_numba)
    pm = detail["per_missile_seconds"]
    _, _ = write_excel(args.xlsx, sol, detail)
    save_report(args.report, args, float(sum(pm)), pm, sol)

    print("-"*70)
    print(f"[RESULT] Total={sum(pm):.3f}s | M1={pm[0]:.3f}  M2={pm[1]:.3f}  M3={pm[2]:.3f}")
    print(f"[INFO] Runtime = {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
