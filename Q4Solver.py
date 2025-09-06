# -*- coding: utf-8 -*-
"""
Q4Solver_visual.py
问题4：FY1/FY2/FY3 各投放一枚烟幕干扰弹，最大化对 M1 的“总联合遮蔽时间”（时间并集）。
- 判据与 Q1 一致：严格圆锥判据 + 圆柱体高密采样
- 两级优化：候选生成（并行）→ 组合并集评估（并行，粗→细）
- 并行：Windows 友好的 ProcessPoolExecutor（带 init），可控 BLAS 线程；也支持线程后端
- 输出：result2.xlsx（三架 UAV 的最终组合与总联合遮蔽时间）
"""

import os
import math
import time
import argparse
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# =========================
# 常量（与 Q1 一致）
# =========================
g = 9.8
MISSILE_SPEED = 300.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)

FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype=float)
FY2_INIT = np.array([12000.0, 1400.0, 1400.0], dtype=float)
FY3_INIT = np.array([6000.0, -3000.0, 700.0], dtype=float)

EPS = 1e-12

# =========================
# 基础函数（贴 Q1 风格）
# =========================
def Unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, mInit=M1_INIT):
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
    return np.array([explXy[0], explXy[1], explZ], dtype=float)

def ConeAllPointsIn(m, c, p, rCloud=SMOG_R, margin=EPS, block=8192):
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

# =========================
# 进程全局（在 initializer 中构造）
# =========================
PTS_GLOBAL = None
HITTIME_GLOBAL = None
INTRA_THREADS_GLOBAL = 1

def _init_worker(nphi, nz, intra_threads):
    global PTS_GLOBAL, HITTIME_GLOBAL, INTRA_THREADS_GLOBAL
    PTS_GLOBAL = PreCalCylinderPoints(nphi, nz, dtype=np.float64)
    HITTIME_GLOBAL = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    INTRA_THREADS_GLOBAL = int(max(1, intra_threads))

# =========================
# 单枚烟幕：从 T_e 起“单体遮蔽时长”（用于候选打分）
# =========================
def _single_smoke_duration(T_e, explPos, dt):
    t0, t1 = T_e, min(T_e + SMOG_EFFECT_TIME, HITTIME_GLOBAL)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc, occluded_flag = t0, 0.0, False
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, cur - T_e)], dtype=float)
        ok = ConeAllPointsIn(m, c, PTS_GLOBAL, rCloud=SMOG_R)
        if ok:
            acc += dt; occluded_flag = True
        elif occluded_flag:
            break
        cur += dt
    return float(acc)

# =========================
# 候选生成 worker（顶层，可 pickle）
# =========================
def _candidate_worker(payload):
    """
    输入：一个 (UAV, T_e, speed) 候选，输出该候选的详细参数与“单体遮蔽时长”
    仅用于候选筛选阶段（不做联合时长）
    """
    uavName = payload["uavName"]
    uavInit = payload["uavInit"]
    T_e = payload["T_e"]
    spd = payload["speed"]
    dt = payload["dt"]
    # 允许内层 BLAS 线程
    def _do():
        mPos, _ = MissileState(T_e)
        target_center = CYLINDER_BASE_CENTER + np.array([0.0, 0.0, CYLINDER_H * 0.5], dtype=float)
        # 二分 s：可达的最靠近目标爆点
        lo, hi, feasible, best_s = 0.0, 1.0, False, 0.0
        while hi - lo > 1e-4:
            mid = (lo + hi) * 0.5
            cand = mPos + mid * (target_center - mPos)
            horiz_dist = math.hypot(cand[0] - uavInit[0], cand[1] - uavInit[1])
            req_speed = horiz_dist / max(T_e, 1e-9)
            if req_speed <= spd + 1e-12:
                feasible, best_s = True, mid; lo = mid
            else:
                hi = mid
        if not feasible: return None
        cand = mPos + best_s * (target_center - mPos)
        dx, dy = (cand[0] - uavInit[0]), (cand[1] - uavInit[1])
        heading = math.atan2(dy, dx)
        drop_alt = uavInit[2]
        if drop_alt < cand[2] - 1e-9: return None
        fuse_delay = math.sqrt(max(0.0, 2.0 * (drop_alt - cand[2]) / g))
        drop_time = T_e - fuse_delay
        if drop_time < -1e-9: return None
        explPos = ExplosionPointFromPlan(uavInit, spd, heading, drop_time, fuse_delay)
        dur = _single_smoke_duration(T_e, explPos, dt)
        return {
            "uavName": uavName,
            "T_e": T_e, "speed": spd, "heading": heading,
            "drop_time": drop_time, "fuse_delay": fuse_delay,
            "expl_pos": explPos, "single_duration": dur
        }
    if threadpool_limits is None:
        return _do()
    else:
        with threadpool_limits(limits=INTRA_THREADS_GLOBAL):
            return _do()

# =========================
# 组合并集评估（粗/细两级）
# =========================
def _union_coverage_duration(triple, dt):
    """
    triple: (cand1, cand2, cand3) 每个含 {T_e, expl_pos,...}
    返回：三枚烟幕合在一起对 M1 的联合遮蔽时间（并集长度）
    """
    Te = [triple[i]["T_e"] for i in range(3)]
    # 时间窗口仅在三枚烟幕有效范围之并集内
    t0 = min(Te)
    t1 = min(HITTIME_GLOBAL, max(Te) + SMOG_EFFECT_TIME)
    if t1 <= t0 + 1e-12:
        return 0.0
    cur, acc = t0, 0.0
    # 预取参数
    def smoke_center(i, t):
        Tei = triple[i]["T_e"]
        if t < Tei - 1e-12 or t > Tei + SMOG_EFFECT_TIME + 1e-12:
            return None
        ex = triple[i]["expl_pos"]
        return np.array([ex[0], ex[1], ex[2] - SMOG_SINK_SPEED * max(0.0, t - Tei)], dtype=float)
    while cur <= t1 + 1e-12:
        m, _ = MissileState(cur)
        covered = False
        for i in (0, 1, 2):
            c = smoke_center(i, cur)
            if c is None: 
                continue
            if ConeAllPointsIn(m, c, PTS_GLOBAL, rCloud=SMOG_R):
                covered = True
                break
        if covered:
            acc += dt
        cur += dt
    return float(acc)

def _union_worker(payload):
    """
    粗评/细评共用的组合 worker
    payload = {
        "triplet": (cand1, cand2, cand3),
        "dt": dt_for_eval
    }
    """
    dt = payload["dt"]
    triple = payload["triplet"]
    if threadpool_limits is None:
        return _union_coverage_duration(triple, dt)
    else:
        with threadpool_limits(limits=INTRA_THREADS_GLOBAL):
            return _union_coverage_duration(triple, dt)

# =========================
# 参数解析与并行自适应
# =========================
def _parse_te_range(s: str):
    a, b, st = [float(x) for x in s.split(",")]
    if st <= 0: st = 0.05
    return np.arange(a, b + 1e-12, st)

def _parse_speed_grid(s: str):
    vs = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            v = float(tok)
            v = max(70.0, min(140.0, v))
            vs.append(v)
    return sorted(set(vs), reverse=True)

def _auto_balance(workers_opt, intra_opt, backend, total_tasks):
    cpu = os.cpu_count() or 1
    if backend == "thread":
        workers = cpu if (workers_opt == "auto") else int(workers_opt)
        intra = 1
        return workers, intra
    # process
    if workers_opt == "auto":
        workers = min(cpu, max(1, total_tasks))
    else:
        workers = max(1, int(workers_opt))
    if intra_opt == "auto":
        intra = max(1, cpu // workers)
        if intra * workers < cpu:
            intra = max(intra, 1 if cpu <= workers else math.ceil(cpu / workers))
    else:
        intra = max(1, int(intra_opt))
    return workers, intra

# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser("Q4 Multi-UAV Visual Solver (Union time, Hybrid CPU parallel)")
    ap.add_argument("--dt", type=float, default=0.001, help="time step for final union evaluation")
    ap.add_argument("--dt-coarse", type=float, default=0.003, help="time step for coarse union screening")
    ap.add_argument("--nphi", type=int, default=960)
    ap.add_argument("--nz", type=int, default=13)
    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--intra-threads", default="auto")
    ap.add_argument("--block", type=int, default=8192)
    ap.add_argument("--margin", type=float, default=EPS)
    ap.add_argument("--speed-grid", type=str, default="140")
    ap.add_argument("--fy1-te", type=str, default="4,12,0.02")
    ap.add_argument("--fy2-te", type=str, default="9,22,0.02")
    ap.add_argument("--fy3-te", type=str, default="18,36,0.02")
    ap.add_argument("--topk", type=int, default=24, help="候选保存数量/每UAV")
    ap.add_argument("--min-gap", type=float, default=0.30, help="同一UAV候选在 T_e 轴最小间隔(s)")
    ap.add_argument("--combo-topm", type=int, default=64, help="粗评保留的组合数，再做精评")
    args = ap.parse_args()

    # 网格
    speedGrid = _parse_speed_grid(args.speed_grid)
    te1 = _parse_te_range(args.fy1_te); te2 = _parse_te_range(args.fy2_te); te3 = _parse_te_range(args.fy3_te)
    uavs = [("FY1", FY1_INIT, te1), ("FY2", FY2_INIT, te2), ("FY3", FY3_INIT, te3)]

    # 统一候选任务池（阶段1）
    gen_tasks = []
    for name, init, teGrid in uavs:
        for T_e in teGrid:
            for spd in speedGrid:
                gen_tasks.append({
                    "uavName": name, "uavInit": init, "T_e": float(T_e), "speed": float(spd),
                    "dt": float(args.dt)  # 用最终 dt 评估单体时长以保证一致
                })

    # 自适应并行
    workers, intra = _auto_balance(args.workers, args.intra_threads, args.backend, total_tasks=len(gen_tasks))
    poolCls = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor

    print("="*110)
    print("Stage-1: Candidate generation (per UAV, single-smoke durations; parallel)")
    print(f"backend={args.backend}, workers={workers}, intra-threads={intra} | candidates={len(gen_tasks)}")
    print("="*110)

    # 阶段1：候选生成
    tA = time.time()
    best_lists = {"FY1": [], "FY2": [], "FY3": []}  # 存字典项，后续去重/截断
    if args.backend == "process":
        with poolCls(max_workers=workers, initializer=_init_worker, initargs=(args.nphi, args.nz, intra)) as pool:
            futs = { pool.submit(_candidate_worker, pl): pl for pl in gen_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    r = fut.result()
                except Exception:
                    r = None
                if r is not None and r["single_duration"] > 0.0:
                    best_lists[r["uavName"]].append(r)
                if done % max(1, total // 20) == 0:
                    print(f"    [Gen] Progress {int(100*done/total)}%")
    else:
        _init_worker(args.nphi, args.nz, 1)
        with poolCls(max_workers=workers) as pool:
            futs = { pool.submit(_candidate_worker, pl): pl for pl in gen_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    r = fut.result()
                except Exception:
                    r = None
                if r is not None and r["single_duration"] > 0.0:
                    best_lists[r["uavName"]].append(r)
                if done % max(1, total // 20) == 0:
                    print(f"    [Gen] Progress {int(100*done/total)}%")
    tB = time.time()
    print(f"[Stage-1] Generated: FY1={len(best_lists['FY1'])}, FY2={len(best_lists['FY2'])}, FY3={len(best_lists['FY3'])} | Runtime {tB-tA:.2f}s")

    # 时间去重 + topK
    def _select_topk(cands, topk, min_gap):
        if not cands:
            return []
        cands = sorted(cands, key=lambda x: (-x["single_duration"], x["T_e"]))
        selected = []
        used_times = []
        for c in cands:
            T = c["T_e"]
            if all(abs(T - uT) >= min_gap for uT in used_times):
                selected.append(c)
                used_times.append(T)
            if len(selected) >= topk:
                break
        return selected

    fy1_cands = _select_topk(best_lists["FY1"], args.topk, args.min_gap)
    fy2_cands = _select_topk(best_lists["FY2"], args.topk, args.min_gap)
    fy3_cands = _select_topk(best_lists["FY3"], args.topk, args.min_gap)

    if not (fy1_cands and fy2_cands and fy3_cands):
        print("[Warn] Some UAV has no feasible candidate; writing zeros.")
        rows = []
        for name in ("FY1","FY2","FY3"):
            rows.append({
                "UAV": name, "Speed (m/s)": 140.0, "Heading (deg)": 0.0,
                "Drop Time (s)": 0.0, "Explosion Time (s)": 0.0,
                "Explosion X (m)": 0.0, "Explosion Y (m)": 0.0, "Explosion Z (m)": 0.0,
                "Fuse Delay (s)": 0.0, "Single Occlusion (s)": 0.0
            })
        df = pd.DataFrame(rows)
        df["Total Union Occlusion (s)"] = 0.0
        df.to_excel("result2.xlsx", index=False)
        print(df.to_string(index=False))
        print("[Result] Total Union Occlusion ≈ 0.000 s")
        return

    print(f"[Stage-1] Kept topK: FY1={len(fy1_cands)}, FY2={len(fy2_cands)}, FY3={len(fy3_cands)}")

    # 阶段2：组合并集评估（粗 -> 细）
    print("="*110)
    print("Stage-2: Combination union-time evaluation (parallel, coarse -> refine)")
    print("="*110)

    # 组合粗评
    coarse_tasks = []
    for c1 in fy1_cands:
        for c2 in fy2_cands:
            for c3 in fy3_cands:
                coarse_tasks.append({"triplet": (c1, c2, c3), "dt": float(args.dt_coarse)})

    workers2, intra2 = _auto_balance(args.workers, args.intra_threads, args.backend, total_tasks=len(coarse_tasks))
    tC = time.time()
    scored = []
    if args.backend == "process":
        with ProcessPoolExecutor(max_workers=workers2, initializer=_init_worker, initargs=(args.nphi, args.nz, intra2)) as pool:
            futs = { pool.submit(_union_worker, pl): pl for pl in coarse_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    u = fut.result()
                except Exception:
                    u = 0.0
                scored.append(u)
                if done % max(1, total // 20) == 0:
                    print(f"    [Coarse] Progress {int(100*done/total)}%")
    else:
        _init_worker(args.nphi, args.nz, 1)
        with ThreadPoolExecutor(max_workers=workers2) as pool:
            futs = { pool.submit(_union_worker, pl): pl for pl in coarse_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    u = fut.result()
                except Exception:
                    u = 0.0
                scored.append(u)
                if done % max(1, total // 20) == 0:
                    print(f"    [Coarse] Progress {int(100*done/total)}%")
    tD = time.time()

    # 取 topM 组合进入精评
    indices = np.argsort(scored)[::-1][:min(args.combo_topm, len(scored))]
    refine_tasks = [ {"triplet": (coarse_tasks[i]["triplet"][0],
                                  coarse_tasks[i]["triplet"][1],
                                  coarse_tasks[i]["triplet"][2]),
                      "dt": float(args.dt)} for i in indices ]

    print(f"[Stage-2] Coarse screened {len(coarse_tasks)} combos → refine {len(refine_tasks)} | Runtime {tD-tC:.2f}s")

    # 精评
    tE = time.time()
    union_scores = []
    if args.backend == "process":
        with ProcessPoolExecutor(max_workers=workers2, initializer=_init_worker, initargs=(args.nphi, args.nz, intra2)) as pool:
            futs = { pool.submit(_union_worker, pl): pl for pl in refine_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    u = fut.result()
                except Exception:
                    u = 0.0
                union_scores.append(u)
                if done % max(1, total // 20) == 0:
                    print(f"    [Refine] Progress {int(100*done/total)}%")
    else:
        _init_worker(args.nphi, args.nz, 1)
        with ThreadPoolExecutor(max_workers=workers2) as pool:
            futs = { pool.submit(_union_worker, pl): pl for pl in refine_tasks }
            done = 0; total = len(futs)
            for fut in as_completed(futs):
                done += 1
                try:
                    u = fut.result()
                except Exception:
                    u = 0.0
                union_scores.append(u)
                if done % max(1, total // 20) == 0:
                    print(f"    [Refine] Progress {int(100*done/total)}%")
    tF = time.time()

    if not union_scores:
        print("[Error] No union scores computed.")
        return

    best_idx = int(np.argmax(union_scores))
    best_union = float(union_scores[best_idx])
    best_triple = refine_tasks[best_idx]["triplet"]

    # 输出
    rows = []
    for c in sorted(best_triple, key=lambda x: x["T_e"]):
        heading_deg = math.degrees(c["heading"])
        if heading_deg < 0: heading_deg += 360.0
        rows.append({
            "UAV": c["uavName"],
            "Speed (m/s)": round(c["speed"], 3),
            "Heading (deg)": round(heading_deg, 3),
            "Drop Time (s)": round(c["drop_time"], 3),
            "Explosion Time (s)": round(c["T_e"], 3),
            "Explosion X (m)": round(c["expl_pos"][0], 3),
            "Explosion Y (m)": round(c["expl_pos"][1], 3),
            "Explosion Z (m)": round(c["expl_pos"][2], 3),
            "Single Occlusion (s)": round(c["single_duration"], 3),  # 单体时长，仅信息
            "Fuse Delay (s)": round(c["fuse_delay"], 3),
        })
    df = pd.DataFrame(rows)
    df["Total Union Occlusion (s)"] = [None]*(len(rows)-1) + [round(best_union, 3)]
    df.to_excel("result2.xlsx", index=False)

    print("-"*110)
    print(df.to_string(index=False))
    print(f"[Result] TOTAL UNION OCCLUSION (no double count) ≈ {best_union:.3f} s")
    print(f"[Info] Stage-1 {tB-tA:.2f}s | Coarse {tD-tC:.2f}s | Refine {tF-tE:.2f}s | Saved to result2.xlsx")

if __name__ == "__main__":
    main()
