# -*- coding: utf-8 -*-
"""
Checker_Q3.py — Verify Problem 3 solution with cooperative masking.

Given `result1.xlsx` (same format as produced by Q3Solver.py), this script:
1) Reads UAV speed & heading, and per-bomb explosion (x,y,z) at the instant of detonation.
2) Reconstructs each bomb's explosion time t_expl from geometry:
      ex_xy = FY1_INIT_xy + v_xy * t_expl  =>  t_expl = dot(ex_xy - FY1_xy, v_hat) / speed
3) Builds a union time grid over [t_expl_i, t_expl_i + SMOG_EFFECT_TIME], capped at missile hit time.
4) For each time step, checks cooperative coverage:
      For every sampled point q on the target cylinder:
         covered_any(q) = OR_i [ q in infinite cone (apex = missile M(t), axis = C_i(t) - M(t),
                                  half-angle from ball radius R = SMOG_R) ]
      The time step is "covered" iff ALL_q covered_any(q) is True.
   (This matches the Q1Solver geometry; only granularity changes to pointwise.)
5) Reports the union covered time (cooperative), and single-bomb FULL-cover time (legacy notion).

Parallelism:
- Single candidate, but heavy time grid → inner time-chunk parallel with threads or processes.
- Windows-friendly (guarded by if __name__ == "__main__":)

Usage example:
  python Checker_Q3.py --excel result1.xlsx --dt 0.002 --nphi 720 --nz 11 ^
                       --backend thread --workers 16 --chunk 3000 --block 8192 --margin 1e-12
"""

import os
import re
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# --- Q1 physics & geometry (must be in same directory) ---
import Q1Solver_visual as Q1

EPS = 1e-12

def info(msg: str):
    print(f"[Q3-CHK][INFO] {msg}", flush=True)

# ------------------ target / constants helpers ------------------
def get_true_target_np() -> np.ndarray:
    for name in ("TRUE_TARGET", "TRUE_TARGET_ORIGIN", "REAL_TARGET_ORIGIN", "FAKE_TARGET_ORIGIN"):
        if hasattr(Q1, name):
            return np.array(getattr(Q1, name), dtype=float)
    return np.array([0.0, 0.0, 0.0], dtype=float)

TRUE_TGT = get_true_target_np()

def los_min_dist_to_point(m_pos: np.ndarray, tgt_pos: np.ndarray, c: np.ndarray) -> float:
    """Min distance from point c to segment M(t)→TRUE_TARGET."""
    v = tgt_pos - m_pos
    w = c - m_pos
    vv = float(np.dot(v, v)) + 1e-12
    t = float(np.dot(w, v) / vv)
    if t < 0.0:
        closest = m_pos
    elif t > 1.0:
        closest = tgt_pos
    else:
        closest = m_pos + t * v
    return float(np.linalg.norm(c - closest))

# ------------------ cooperative cone test (pointwise) ------------------
def cone_points_in(m_pos: np.ndarray, c: np.ndarray, pts: np.ndarray, R: float) -> np.ndarray:
    """
    Return (Npts,) bool mask: whether each target sample point lies within the infinite cone
    with apex m_pos, axis a = c - m_pos, and half-angle derived from ball radius R.

    This is the pointwise version of Q1.ConeAllPointsIn and uses the same inequality:
      forward: dot(b, a) >= 0
      if |a|^2 <= R^2 -> half-angle = 90°, accept forward points
      else require (dot(b,a))^2 >= ||b||^2 * (||a||^2 - R^2)
    """
    a = c - m_pos
    d2 = float(np.dot(a, a))
    b = pts - m_pos[None, :]
    dot = b @ a
    fwd = dot >= 0.0
    if d2 <= R * R:
        return fwd
    rhs = (np.sum(b * b, axis=1)) * (d2 - R * R)
    lhs = dot * dot
    return fwd & (lhs >= rhs)

# ------------------ excel reader ------------------
def _parse_heading(text) -> float:
    """
    Accepts '0.123456 rad', '0.123456', or numeric.
    Returns heading in radians (float).
    """
    if isinstance(text, (int, float, np.floating)):
        return float(text)
    if not isinstance(text, str):
        raise ValueError("Heading text not recognized.")
    m = re.search(r"([-+]?[\d\.Ee+-]+)", text)
    if not m:
        raise ValueError(f"Cannot parse heading from: {text}")
    return float(m.group(1))

def load_solution_from_excel(path: str) -> Dict[str, Any]:
    """
    Expected columns (as in Q3Solver write_excel):
      '无人机运动方向' -> string like '0.123 rad'
      '无人机运动速度 (m/s)'
      '烟幕干扰弹编号' (1..3)
      '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)'
    Drop point cols may present but not required.

    Returns:
      dict with keys: heading (rad), speed, bombs_expl: List[xyz], t_expls: List[float]
    """
    df = pd.read_excel(path, sheet_name=0)
    # Normalize column names (strip spaces)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Required cols
    col_heading = "无人机运动方向"
    col_speed = "无人机运动速度 (m/s)"
    col_id = "烟幕干扰弹编号"
    col_exx = "烟幕干扰弹起爆点的x坐标 (m)"
    col_exy = "烟幕干扰弹起爆点的y坐标 (m)"
    col_exz = "烟幕干扰弹起爆点的z坐标 (m)"

    for c in (col_heading, col_speed, col_id, col_exx, col_exy, col_exz):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in {path}")

    # Heading & speed
    heading = _parse_heading(df.iloc[0][col_heading])
    speed = float(df.iloc[0][col_speed])

    # Sort by id 1..3
    df = df.sort_values(col_id)
    exps = []
    for _, row in df.iterrows():
        exps.append(np.array([float(row[col_exx]), float(row[col_exy]), float(row[col_exz])], dtype=float))

    # Compute t_expl from projection along heading
    hv = np.array([np.cos(heading), np.sin(heading)], dtype=float)
    p0 = np.array(Q1.FY1_INIT[:2], dtype=float)
    t_expls = []
    for e in exps:
        s = float(np.dot(e[:2] - p0, hv))
        if speed <= 0:
            raise ValueError("UAV speed must be positive.")
        t_expl = s / speed
        if t_expl < -1e-6:
            info(f"Warning: inferred t_expl<0 ({t_expl:.6f}s). Check heading/speed/explosion XY.")
        t_expls.append(t_expl)

    return dict(heading=heading, speed=speed, bombs_expl=exps, t_expls=t_expls)

# ------------------ time grid helpers ------------------
def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        pa, pb = out[-1]
        if a <= pb + 1e-12:
            out[-1] = (pa, max(pb, b))
        else:
            out.append((a, b))
    return out

# ------------------ per-chunk evaluation ------------------
def eval_chunk(args):
    """
    Evaluate cooperative coverage over a given time sub-grid.
    Returns: (idx0, union_mask_chunk, per_bomb_full_mask_chunk)
    """
    (idx0, t_chunk, bombs_expl, t_expls, pts, margin, block, chunk_id, total_chunks, inner_log_every) = args
    n = len(t_chunk)
    out_union = np.zeros(n, dtype=bool)
    out_each  = [np.zeros(n, dtype=bool) for _ in bombs_expl]

    if inner_log_every > 0 and (chunk_id % inner_log_every == 0):
        info(f"Chunk {chunk_id}/{total_chunks}")

    R = float(Q1.SMOG_R)
    Rm = R + float(margin)

    for i, t in enumerate(t_chunk):
        m_pos, _ = Q1.MissileState(float(t), Q1.M1_INIT)

        # Quick necessary rejection on center line:
        all_far = True
        C_list = []
        for expl, texp in zip(bombs_expl, t_expls):
            cz = expl[2] - Q1.SMOG_SINK_SPEED * max(0.0, float(t) - texp)
            c  = np.array([expl[0], expl[1], cz], dtype=float)
            C_list.append(c)
            if los_min_dist_to_point(m_pos, TRUE_TGT, c) <= Rm:
                all_far = False
        if all_far:
            out_union[i] = False
            continue

        covered_any = None
        for bi, c in enumerate(C_list):
            # Optional skip for per-bomb full cover test
            if los_min_dist_to_point(m_pos, TRUE_TGT, c) > Rm:
                out_each[bi][i] = False
                pts_in = None
            else:
                pts_in = cone_points_in(m_pos, c, pts, R)
                out_each[bi][i] = bool(np.all(pts_in))
            if pts_in is not None:
                covered_any = pts_in if covered_any is None else (covered_any | pts_in)

        out_union[i] = (covered_any is not None) and bool(np.all(covered_any))

    return idx0, out_union, out_each

def mask_seconds(mask: np.ndarray, dt: float) -> float:
    return float(np.count_nonzero(mask) * dt)

# ------------------ main checking routine ------------------
def run_check(excel_path: str,
              dt: float, nphi: int, nz: int,
              backend: str, workers: int,
              chunk: int, block: int, margin: float,
              inner_log: bool, inner_log_every: int,
              timegrid_mode: str) -> Dict[str, Any]:

    sol = load_solution_from_excel(excel_path)
    heading = sol["heading"]; speed = sol["speed"]
    bombs_expl = sol["bombs_expl"]; t_expls = sol["t_expls"]

    info("=" * 80)
    info(f"Loaded solution from {excel_path}")
    info(f"Heading(rad)={heading:.6f}, Speed={speed:.3f} m/s")
    for k, (e, te) in enumerate(zip(bombs_expl, t_expls), 1):
        info(f"Bomb#{k}: expl=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}), t_expl={te:.6f}s")

    # Time range: union of effective intervals, capped by hit time
    hit_time = float(np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED)
    intervals = []
    for te in t_expls:
        a = float(te)
        b = float(min(te + Q1.SMOG_EFFECT_TIME, hit_time))
        if b > a:
            intervals.append((a, b))
    if not intervals:
        info("No effective intervals found. Covered time = 0.")
        return dict(total=0.0, per_bomb=[0.0, 0.0, 0.0])

    intervals = merge_intervals(intervals)
    if timegrid_mode == "union":
        t_list = [np.arange(a, b + EPS, dt, dtype=float) for (a, b) in intervals]
        t_grid = np.concatenate(t_list)
    else:
        t0 = min(t_expls)
        t1 = min(max(t_expls) + Q1.SMOG_EFFECT_TIME, hit_time)
        t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)

    info(f"Time grid size = {len(t_grid)} (dt={dt})")
    pts = Q1.PreCalCylinderPoints(nphi, nz, dtype=float)
    info(f"Cylinder samples: nphi={nphi}, nz={nz}, total={len(pts)}")

    # Chunking
    total_chunks = (len(t_grid) + chunk - 1) // chunk
    tasks = []
    for chunk_id, i0 in enumerate(range(0, len(t_grid), chunk), start=1):
        tasks.append((i0, t_grid[i0:i0+chunk], bombs_expl, t_expls, pts,
                      margin, block, chunk_id, total_chunks, inner_log_every))

    mask_union = np.zeros_like(t_grid, dtype=bool)
    masks_each = [np.zeros_like(t_grid, dtype=bool) for _ in range(len(bombs_expl))]

    if inner_log:
        info(f"Evaluate {total_chunks} time-chunks with backend={backend}, workers={workers}")

    pool_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    if backend in ("thread", "process") and workers > 1:
        with pool_cls(max_workers=workers) as pool:
            futs = {pool.submit(eval_chunk, a): a[7] for a in tasks}
            done = 0
            report_every = max(1, total_chunks // 10)
            for fut in as_completed(futs):
                _chunk_id = futs[fut]
                idx, mu, me = fut.result()
                l = len(mu)
                mask_union[idx:idx+l] = mu
                for bi in range(len(masks_each)):
                    masks_each[bi][idx:idx+l] = me[bi]
                done += 1
                if inner_log and (done % report_every == 0):
                    info(f"Progress: {done}/{total_chunks} chunks ({int(100*done/total_chunks)}%)")
    else:
        for (i0, t_sub, *rest) in tasks:
            idx, mu, me = eval_chunk((i0, t_sub, *rest))
            l = len(mu)
            mask_union[i0:i0+l] = mu
            for bi in range(len(masks_each)):
                masks_each[bi][i0:i0+l] = me[bi]

    total_seconds = mask_seconds(mask_union, dt)
    per_bomb = [mask_seconds(m, dt) for m in masks_each]

    info("-" * 80)
    info(f"TOTAL union covered (cooperative) = {total_seconds:.6f} s")
    for k, s in enumerate(per_bomb, 1):
        info(f"Bomb#{k} FULL-cover time (single-bomb) = {s:.6f} s")
    info("=" * 80)

    return dict(total=total_seconds, per_bomb=per_bomb)

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser("Checker for Q3 (cooperative masking)")
    ap.add_argument("--excel", type=str, default="result1.xlsx", help="path to result1.xlsx")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--nphi", type=int, default=720)
    ap.add_argument("--nz", type=int, default=11)
    ap.add_argument("--backend", choices=["thread", "process", "none"], default="thread")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--block", type=int, default=8192)  # kept for API symmetry; unused here
    ap.add_argument("--margin", type=float, default=1e-12)
    ap.add_argument("--inner_log", action="store_true")
    ap.add_argument("--inner_log_every", type=int, default=10)
    ap.add_argument("--timegrid_mode", choices=["union", "full"], default="union")
    args = ap.parse_args()

    # Pin BLAS threads to avoid oversubscription
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    run_check(args.excel, args.dt, args.nphi, args.nz,
              args.backend, args.workers,
              args.chunk, args.block, args.margin,
              args.inner_log, max(1, args.inner_log_every),
              args.timegrid_mode)

if __name__ == "__main__":
    main()
