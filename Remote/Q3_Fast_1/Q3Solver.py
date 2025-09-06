# -*- coding: utf-8 -*-
"""
Q3Solver.py — Problem 3: Maximize the total (not necessarily continuous) covered time
by deploying 3 smoke bombs from UAV FY1 to interfere with missile M1.

Integrated features:
1) Two-stage search (Stage1 coarse grid → Stage2 local refinement around seed)
2) Coordinate Descent micro-tuning (optional, enabled by default)
3) Time-grid only on effective smoke intervals (union of 3 bombs)
4) Cheap necessary test before expensive checking
5) Two-level parallelism (outer candidates + inner time-chunks), Windows-friendly
6) Rich [Q3][INFO] logs; Excel export `result1.xlsx`
7) Convergence plot `q3_convergence.png` (English labels; clean style; optional shading)
8) Stage2 batching + candidate cap + logs suppressed in workers to avoid stdout blocking
9) NEW: Cooperative masking — multiple smokes can jointly cover the cylinder (pointwise OR across bombs, then AND across all cylinder points).
"""

import os
import math
import time
import argparse
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ---- Matplotlib (non-interactive) ----
import matplotlib
matplotlib.use("Agg")
# Sane defaults: English-only labels, clean output
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,   # safe minus rendering
})
import matplotlib.pyplot as plt

# Your Q1 geometric/physics logic module (place in the same directory)
import Q1Solver_visual as Q1

EPS = 1e-12

def info(msg: str):
    print(f"[Q3][INFO] {msg}", flush=True)

# ------------------ Helpers ------------------
def get_true_target_np() -> np.ndarray:
    # Try common names, fall back to origin
    for name in ("TRUE_TARGET", "TRUE_TARGET_ORIGIN", "REAL_TARGET_ORIGIN", "FAKE_TARGET_ORIGIN"):
        if hasattr(Q1, name):
            return np.array(getattr(Q1, name), dtype=float)
    return np.array([0.0, 0.0, 0.0], dtype=float)

TRUE_TGT = get_true_target_np()

def heading_to_origin() -> float:
    v = Q1.FAKE_TARGET_ORIGIN - Q1.FY1_INIT
    return math.atan2(v[1], v[0])

def uav_state_horizontal(t: float, uav_init: np.ndarray, uav_speed: float, heading_rad: float):
    vx = uav_speed * math.cos(heading_rad)
    vy = uav_speed * math.sin(heading_rad)
    pos = np.array([uav_init[0] + vx * t, uav_init[1] + vy * t, uav_init[2]], dtype=uav_init.dtype)
    vel = np.array([vx, vy, 0.0], dtype=uav_init.dtype)
    return pos, vel

def explosion_point(heading_rad: float, t_drop: float, fuse_delay: float, uav_speed: float, dtype=float):
    drop_pos, vel = uav_state_horizontal(t_drop, Q1.FY1_INIT.astype(dtype), dtype(uav_speed), heading_rad)
    ex_xy = drop_pos[:2] + vel[:2] * dtype(fuse_delay)
    ex_z  = drop_pos[2]  - dtype(0.5) * dtype(Q1.g) * (dtype(fuse_delay) ** 2)
    return np.array([ex_xy[0], ex_xy[1], ex_z], dtype=dtype), drop_pos

def los_min_dist_to_point(m_pos: np.ndarray, tgt_pos: np.ndarray, c: np.ndarray) -> float:
    """Minimum distance from point c to segment (m_pos → tgt_pos)."""
    v = tgt_pos - m_pos
    w = c - m_pos
    vv = float(np.dot(v, v)) + 1e-12
    t = float(np.dot(w, v) / vv)
    if t < 0.0:   closest = m_pos
    elif t > 1.0: closest = tgt_pos
    else:         closest = m_pos + t * v
    return float(np.linalg.norm(c - closest))

def merge_intervals(intervals: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
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

# ------------------ Cooperative cone check (pointwise) ------------------
def cone_points_in(m_pos: np.ndarray, c: np.ndarray, pts: np.ndarray, R: float) -> np.ndarray:
    """
    Vectorized cone-membership for target sample points.
    Return mask of shape (Npts,): True if a point is inside the infinite cone
    with apex m_pos, axis (c - m_pos), half-angle induced by ball radius R.
    Geometric test matches Q1.ConeAllPointsIn but at point granularity.
    """
    a = c - m_pos                         # (3,)
    d2 = float(np.dot(a, a))
    b = pts - m_pos[None, :]              # (N,3)
    dot = b @ a                           # (N,)
    fwd = dot >= 0.0                      # facing toward the cloud
    if d2 <= R*R:                         # apex inside the ball => half-angle = 90°
        return fwd
    rhs = (np.sum(b*b, axis=1)) * (d2 - R*R)
    lhs = dot * dot
    cone = lhs >= rhs
    return fwd & cone

# ------------------ Inner time-chunk evaluation ------------------
def eval_chunk_union(args):
    """
    Evaluate union mask (and per-bomb masks) over a time chunk.

    NEW cooperative masking:
      For each time t, for each target point q:
        covered_any(q) = OR over bombs_i [ q in Cone(M(t), C_i(t), R) ]
      Then time t is considered covered iff ALL_q covered_any(q) is True.

    out_each[bi] still records "this single bomb alone fully covers the cylinder"
    (i.e., ALL points in its cone) — same semantics as before.
    """
    (idx0, t_chunk, bombs, t_expls, pts, margin, block, chunk_id, total_chunks, inner_log_every) = args
    n = len(t_chunk)
    out_union = np.zeros(n, dtype=bool)
    out_each  = [np.zeros(n, dtype=bool) for _ in bombs]

    if inner_log_every > 0 and (chunk_id % inner_log_every == 0):
        info(f"Candidate: time-chunk {chunk_id}/{total_chunks}")

    R = float(Q1.SMOG_R)
    Rm = R + float(margin)

    for i, t in enumerate(t_chunk):
        m_pos, _ = Q1.MissileState(float(t), Q1.M1_INIT)

        # quick reject: if even the center line to ALL clouds are farther than R => impossible
        all_far = True
        C_list = []
        for (expl, texp) in zip(bombs, t_expls):
            cz = expl[2] - Q1.SMOG_SINK_SPEED * max(0.0, float(t) - texp)
            c  = np.array([expl[0], expl[1], cz], dtype=expl.dtype)
            C_list.append(c)
            if los_min_dist_to_point(m_pos, TRUE_TGT, c) <= Rm:
                all_far = False
        if all_far:
            out_union[i] = False
            for bi in range(len(bombs)):
                out_each[bi][i] = False
            continue

        covered_any = None  # (Npts,) mask across bombs
        for bi, c in enumerate(C_list):
            # Optional cheap skip for per-bomb full-cover check
            if los_min_dist_to_point(m_pos, TRUE_TGT, c) > Rm:
                pts_in = None
                out_each[bi][i] = False  # single-bomb full cover not possible
            else:
                # point-level membership for this cloud
                pts_in = cone_points_in(m_pos, c, pts, R)
                # record whether this single bomb alone covers the entire cylinder
                out_each[bi][i] = bool(np.all(pts_in))

            if pts_in is not None:
                covered_any = pts_in if covered_any is None else (covered_any | pts_in)

        # cooperative masking requires all target points to be covered by at least one bomb
        out_union[i] = (covered_any is not None) and bool(np.all(covered_any))

    return idx0, out_union, out_each

def mask_seconds(mask, dt) -> float:
    return float(np.count_nonzero(mask) * dt)

# ------------------ Single candidate evaluation (Windows-friendly) ------------------
def evaluate_strategy(params: Dict[str, Any], sim: Dict[str, Any]):
    """
    params = {heading_offset_deg, uav_speed, drops=[t1,t2,t3], fuses=[d1,d2,d3]}
    sim    = {dt,nphi,nz,backend,workers,chunk,block,fp32,margin,inner_log,inner_log_every,timegrid_mode}
    return total_seconds, per_bomb_seconds(list), misc(dict)
    """
    heading = heading_to_origin() + math.radians(params["heading_offset_deg"])
    uav_speed = params["uav_speed"]
    t_drops   = params["drops"]
    fuses     = params["fuses"]

    # Constraints: gaps >= 1s, fuses > 0, explosion altitude >= 0
    if not (t_drops[0] >= 0.0 and t_drops[1] - t_drops[0] >= 1.0 and t_drops[2] - t_drops[1] >= 1.0):
        return -1.0, [0.0, 0.0, 0.0], None
    for fd in fuses:
        if fd <= 0.0:
            return -1.0, [0.0, 0.0, 0.0], None
        if (Q1.FY1_INIT[2] - 0.5 * Q1.g * (fd ** 2)) < 0.0:
            return -1.0, [0.0, 0.0, 0.0], None

    dtype = np.float32 if sim["fp32"] else np.float64

    # Explosion points
    bombs_expl, drops_pos, t_expls = [], [], []
    for td, fd in zip(t_drops, fuses):
        expl, drop = explosion_point(heading, td, fd, uav_speed, dtype=dtype)
        if expl[2] < 0.0:
            return -1.0, [0.0, 0.0, 0.0], None
        bombs_expl.append(expl)
        drops_pos.append(drop)
        t_expls.append(td + fd)

    # Time grid (union of effective intervals)
    hit_time = float(np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED)
    if sim.get("timegrid_mode", "union") == "union":
        intervals = []
        for te in t_expls:
            a = float(te)
            b = float(min(te + Q1.SMOG_EFFECT_TIME, hit_time))
            if b > a:
                intervals.append((a, b))
        intervals = merge_intervals(intervals)
        if not intervals:
            return 0.0, [0.0, 0.0, 0.0], None
        t_list = [np.arange(a, b + EPS, sim["dt"], dtype=dtype) for (a, b) in intervals]
        t_grid = np.concatenate(t_list)
    else:
        t0 = min(t_expls)
        t1 = min(max(t_expls) + Q1.SMOG_EFFECT_TIME, hit_time)
        if t1 <= t0:
            return 0.0, [0.0, 0.0, 0.0], None
        t_grid = np.arange(t0, t1 + EPS, sim["dt"], dtype=dtype)

    # Cylinder samples
    pts = Q1.PreCalCylinderPoints(sim["nphi"], sim["nz"], dtype=dtype)

    # Chunked evaluation
    total_chunks = (len(t_grid) + sim["chunk"] - 1) // sim["chunk"]
    chunks = []
    for chunk_id, i in enumerate(range(0, len(t_grid), sim["chunk"]), start=1):
        chunks.append((i, t_grid[i:i+sim["chunk"]], bombs_expl, t_expls, pts,
                       sim["margin"], sim["block"], chunk_id, total_chunks, sim["inner_log_every"]))

    mask_union = np.zeros_like(t_grid, dtype=bool)
    masks_each = [np.zeros_like(t_grid, dtype=bool) for _ in range(3)]

    pool_cls = ProcessPoolExecutor if sim["backend"] == "process" else ThreadPoolExecutor
    if sim["inner_log"]:
        info(f"Candidate start: {total_chunks} time-chunks (dt={sim['dt']}, nphi={sim['nphi']}, nz={sim['nz']})")

    with pool_cls(max_workers=sim["workers"]) as pool:
        futs = {pool.submit(eval_chunk_union, args): args[7] for args in chunks}
        done = 0
        report_every = max(1, total_chunks // 10)
        for fut in as_completed(futs):
            _chunk_id = futs[fut]
            idx, m_union, m_each = fut.result()
            l = len(m_union)
            mask_union[idx:idx+l] = m_union
            for bi in range(3):
                masks_each[bi][idx:idx+l] = m_each[bi]
            done += 1
            if sim["inner_log"] and (done % report_every == 0):
                info(f"Candidate progress: {done}/{total_chunks} chunks ({int(done*100/total_chunks)}%)")

    total_seconds = mask_seconds(mask_union, sim["dt"])
    secs_each     = [mask_seconds(m, sim["dt"]) for m in masks_each]

    # Small payload for inter-process passing
    drops_pos_list  = [p.tolist() for p in drops_pos]
    bombs_expl_list = [p.tolist() for p in bombs_expl]

    misc = dict(
        heading=float(heading), speed=float(uav_speed),
        drops_pos=drops_pos_list, bombs_expl=bombs_expl_list,
        t_drops=[float(x) for x in t_drops], t_expls=[float(x) for x in t_expls]
    )
    if sim["inner_log"]:
        info(f"Candidate done: union={total_seconds:.6f}s, per-bomb(single full)={secs_each}")

    return float(total_seconds), [float(x) for x in secs_each], misc

# Worker wrapper (Windows-friendly)
def eval_one_pack(pack: Tuple[Dict[str, Any], Dict[str, Any]]):
    params, sim = pack
    total, per_bomb, misc = evaluate_strategy(params, sim)
    return total, per_bomb, misc, params

# ------------------ Excel export ------------------
def write_excel(best, path="result1.xlsx"):
    cols = [
        "无人机运动方向",
        "无人机运动速度 (m/s)",
        "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)",
        "烟幕干扰弹投放点的y坐标 (m)",
        "烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)",
        "烟幕干扰弹起爆点的y坐标 (m)",
        "烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)",
    ]
    heading_text = f"{best['heading']:.6f} rad"
    rows = []
    for k in range(3):
        dp = best["drops_pos"][k]
        ep = best["bombs_expl"][k]
        rows.append({
            "无人机运动方向": heading_text,
            "无人机运动速度 (m/s)": best["speed"],
            "烟幕干扰弹编号": k + 1,
            "烟幕干扰弹投放点的x坐标 (m)": float(dp[0]),
            "烟幕干扰弹投放点的y坐标 (m)": float(dp[1]),
            "烟幕干扰弹投放点的z坐标 (m)": float(dp[2]),
            "烟幕干扰弹起爆点的x坐标 (m)": float(ep[0]),
            "烟幕干扰弹起爆点的y坐标 (m)": float(ep[1]),
            "烟幕干扰弹起爆点的z坐标 (m)": float(ep[2]),
            "有效干扰时长 (s)": best["secs_each"][k],
        })
    pd.DataFrame(rows, columns=cols).to_excel(path, index=False, sheet_name="Sheet1")

# ------------------ Convergence plot (English; clean) ------------------
def save_convergence_plot(history, path="q3_convergence.png", shade=False):
    """
    Plot convergence curve of best objective vs evaluation index.
    All labels are in English to avoid glyph issues.
    shade: if True, lightly shade Stage1/Stage2/CD spans; otherwise clean background.
    """
    if not history:
        return
    xs = [h["idx"] for h in history]
    ys = [h["best"] for h in history]
    phases = [h["phase"] for h in history]

    def phase_range(p):
        idxs = [i for i, ph in enumerate(phases) if ph == p]
        if not idxs:
            return None
        return (xs[idxs[0]], xs[idxs[-1]])

    p1 = phase_range("stage1")
    p2 = phase_range("stage2")
    p3 = phase_range("cd")

    plt.figure(figsize=(10, 5), dpi=150)
    # main line
    plt.plot(xs, ys, lw=2)

    # optional shading
    if shade:
        def shade_span(rng, color, label):
            if rng:
                plt.axvspan(rng[0], rng[1], alpha=0.08, color=color, label=label)
        shade_span(p1, "C0", "Stage1 (coarse)")
        shade_span(p2, "C1", "Stage2 (refine)")
        shade_span(p3, "C2", "Coordinate descent")

    # labels/format
    plt.xlabel("Evaluations")
    plt.ylabel("Best covered time (s)")
    plt.title("Q3 Convergence (search / refinement)")
    if ys:
        i_best = int(np.argmax(ys))
        plt.scatter([xs[i_best]], [ys[i_best]], s=36, zorder=3)
        plt.annotate(f"best={ys[i_best]:.3f}s @#{xs[i_best]}",
                     xy=(xs[i_best], ys[i_best]),
                     xytext=(6, 10), textcoords="offset points",
                     fontsize=9)
    plt.grid(True, alpha=0.25)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(frameon=False, loc="best")  # no white legend box
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ------------------ Coordinate Descent ------------------
def coordinate_descent(seed_params: Dict[str, Any],
                       sim: Dict[str, Any],
                       steps: Dict[str, float],
                       bounds: Dict[str, Tuple[float, float]],
                       rounds: int,
                       conv_hist: List[Dict[str, Any]],
                       eval_counter: List[int]):
    def within(v, low, high): return max(low, min(high, v))

    def repair_time_vars(p):
        t1, t2, t3 = p["drops"]
        g12 = max(1.0, t2 - t1)
        g23 = max(1.0, t3 - t2)
        t1 = max(0.0, t1)
        t2 = t1 + g12
        t3 = t2 + g23
        p["drops"] = [t1, t2, t3]
        for i in range(3):
            f = max(0.1, p["fuses"][i])
            z_est = Q1.FY1_INIT[2] - 0.5 * Q1.g * (f**2)
            if z_est < 0.0:
                f = math.sqrt(2*Q1.FY1_INIT[2]/Q1.g) - 1e-3
            p["fuses"][i] = f

    cur = dict(seed_params)
    total, per_bomb, _ = evaluate_strategy(cur, sim)
    best_val = total
    best_params = dict(cur)
    info(f"[CD] init: {best_val:.6f}s | {best_params}")
    eval_counter[0] += 1
    conv_hist.append(dict(idx=eval_counter[0], best=best_val, phase="cd"))

    keys = ["heading_offset_deg","uav_speed","t1","gap12","gap23","f1","f2","f3"]
    for rd in range(1, rounds+1):
        improved = False
        info(f"[CD] round {rd} start")
        for key in keys:
            step = steps.get(key, 0.0)
            if step <= 0: continue
            for sgn in (+1, -1):
                trial = dict(best_params)
                t1, t2, t3 = trial["drops"]; f1, f2, f3 = trial["fuses"]
                if key == "heading_offset_deg":
                    trial["heading_offset_deg"] = within(trial["heading_offset_deg"] + sgn*step, *bounds["heading_offset_deg"])
                elif key == "uav_speed":
                    trial["uav_speed"] = within(trial["uav_speed"] + sgn*step, *bounds["uav_speed"])
                elif key == "t1":
                    t1 = within(t1 + sgn*step, *bounds["t1"]); trial["drops"]=[t1,t2,t3]
                elif key == "gap12":
                    g12 = within((t2-t1) + sgn*step, *bounds["gap12"]); t2=t1+g12; t3=max(t3,t2+1.0); trial["drops"]=[t1,t2,t3]
                elif key == "gap23":
                    g23 = within((t3-t2) + sgn*step, *bounds["gap23"]); t3=t2+g23; trial["drops"]=[t1,t2,t3]
                elif key == "f1":
                    f1 = within(f1 + sgn*step, *bounds["fuse"]); trial["fuses"]=[f1,f2,f3]
                elif key == "f2":
                    f2 = within(f2 + sgn*step, *bounds["fuse"]); trial["fuses"]=[f1,f2,f3]
                elif key == "f3":
                    f3 = within(f3 + sgn*step, *bounds["fuse"]); trial["fuses"]=[f1,f2,f3]
                repair_time_vars(trial)
                val, _, _ = evaluate_strategy(trial, sim)
                eval_counter[0] += 1
                if val > best_val:
                    best_val = val; best_params = dict(trial); improved = True
                    info(f"[CD] improve: {best_val:.6f}s | {best_params}")
                conv_hist.append(dict(idx=eval_counter[0], best=best_val, phase="cd"))
        if not improved:
            info("[CD] converged (no improvement)")
            break
    return best_params, best_val

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser("Q3 — three-bomb coverage optimization (Windows-friendly, cooperative masking)")

    # Inner evaluation (accuracy & parallel)
    ap.add_argument("--dt", type=float, default=0.003)
    ap.add_argument("--nphi", type=int, default=360)
    ap.add_argument("--nz", type=int, default=7)
    ap.add_argument("--backend", choices=["process","thread"], default="thread")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--margin", type=float, default=1e-12)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--inner_log", action="store_true")
    ap.add_argument("--inner_log_every", type=int, default=10)
    ap.add_argument("--timegrid_mode", choices=["union","full"], default="union")

    # Outer candidates
    ap.add_argument("--outer_backend", choices=["process","thread","none"], default="process")
    ap.add_argument("--search_workers", type=int, default=max(1,(os.cpu_count() or 1)))

    # Stage1 search space
    ap.add_argument("--heading_min_deg", type=float, default=-2.0)
    ap.add_argument("--heading_max_deg", type=float, default=+2.0)
    ap.add_argument("--heading_step_deg", type=float, default=0.5)
    ap.add_argument("--speed_min", type=float, default=120.0)
    ap.add_argument("--speed_max", type=float, default=140.0)
    ap.add_argument("--speed_step", type=float, default=10.0)
    ap.add_argument("--t1_min", type=float, default=0.0)
    ap.add_argument("--t1_max", type=float, default=6.0)
    ap.add_argument("--t1_step", type=float, default=1.0)
    ap.add_argument("--gap12_min", type=float, default=1.0)
    ap.add_argument("--gap12_max", type=float, default=4.0)
    ap.add_argument("--gap12_step", type=float, default=1.0)
    ap.add_argument("--gap23_min", type=float, default=1.0)
    ap.add_argument("--gap23_max", type=float, default=4.0)
    ap.add_argument("--gap23_step", type=float, default=1.0)
    ap.add_argument("--fuse_min", type=float, default=3.0)
    ap.add_argument("--fuse_max", type=float, default=8.0)
    ap.add_argument("--fuse_step", type=float, default=1.0)

    # Stage2 refinement
    ap.add_argument("--stage2_enable", action="store_true", help="enable Stage2 refinement")
    ap.add_argument("--topk", type=int, default=20)  # reserved (currently use best as seed)
    ap.add_argument("--stage2_span_steps", type=int, default=2, help="±span_steps * step")
    ap.add_argument("--stage2_heading_step_deg", type=float, default=0.2)
    ap.add_argument("--stage2_speed_step", type=float, default=5.0)
    ap.add_argument("--stage2_t_step", type=float, default=0.5)
    ap.add_argument("--stage2_fuse_step", type=float, default=0.5)
    ap.add_argument("--stage2_batch_size", type=int, default=2000, help="batch size for Stage2 submission")
    ap.add_argument("--stage2_max_cands", type=int, default=20000, help="cap Stage2 candidates; random sample if exceeded")

    # Coordinate Descent (default ON; use --no_cd to disable)
    ap.add_argument("--cd_enable", action="store_true", default=True, help="enable coordinate descent (default ON)")
    ap.add_argument("--no_cd", dest="cd_enable", action="store_false", help="disable coordinate descent")
    ap.add_argument("--cd_rounds", type=int, default=3)
    ap.add_argument("--cd_step_heading", type=float, default=0.2)
    ap.add_argument("--cd_step_speed", type=float, default=5.0)
    ap.add_argument("--cd_step_time", type=float, default=0.5)
    ap.add_argument("--cd_step_fuse", type=float, default=0.5)

    args = ap.parse_args()

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    info("="*90)
    info("Start Q3 optimization (cooperative masking): two-stage + coordinate descent + clean visualization")
    info(f"Q1 consts: MISSILE_SPEED={Q1.MISSILE_SPEED} m/s, SMOG_R={Q1.SMOG_R} m, "
         f"SMOG_EFFECT_TIME={Q1.SMOG_EFFECT_TIME} s, SMOG_SINK_SPEED={Q1.SMOG_SINK_SPEED} m/s")
    info(f"Missile init: {Q1.M1_INIT}, Fake target: {Q1.FAKE_TARGET_ORIGIN}, FY1 init: {Q1.FY1_INIT}")
    info(f"TRUE_TARGET (for cheap test): {TRUE_TGT}")

    # Inner sim config
    sim = dict(
        dt=args.dt, nphi=args.nphi, nz=args.nz,
        backend=("thread" if args.outer_backend == "process" else args.backend),
        workers=args.workers, chunk=args.chunk, block=args.block,
        margin=args.margin, fp32=args.fp32,
        inner_log=args.inner_log, inner_log_every=max(1, args.inner_log_every),
        timegrid_mode=args.timegrid_mode
    )
    info(f"Sim: {sim}")
    info(f"Outer parallel: {args.outer_backend}, workers={args.search_workers}")

    # Stage1 candidates
    headings = np.arange(args.heading_min_deg, args.heading_max_deg + 1e-9, args.heading_step_deg)
    speeds   = np.arange(args.speed_min, args.speed_max + 1e-9, args.speed_step)
    t1s      = np.arange(args.t1_min, args.t1_max + 1e-9, args.t1_step)
    gap12s   = np.arange(args.gap12_min, args.gap12_max + 1e-9, args.gap12_step)
    gap23s   = np.arange(args.gap23_min, args.gap23_max + 1e-9, args.gap23_step)
    fuses    = np.arange(args.fuse_min, args.fuse_max + 1e-9, args.fuse_step)
    info(f"Stage1 space: heading {len(headings)} × speed {len(speeds)} × t1 {len(t1s)} × "
         f"gap12 {len(gap12s)} × gap23 {len(gap23s)} × fuse^3 {len(fuses)}^3")

    candidates = []
    t_build_a = time.time()
    for hd in headings:
        for sp in speeds:
            for t1 in t1s:
                for g12 in gap12s:
                    for g23 in gap23s:
                        t2 = t1 + g12
                        t3 = t2 + g23
                        for f1 in fuses:
                            for f2 in fuses:
                                for f3 in fuses:
                                    candidates.append(dict(
                                        heading_offset_deg=float(hd),
                                        uav_speed=float(sp),
                                        drops=[float(t1), float(t2), float(t3)],
                                        fuses=[float(f1), float(f2), float(f3)],
                                    ))
    t_build_b = time.time()
    info(f"Stage1 candidates built: {len(candidates)} (in {t_build_b - t_build_a:.2f}s)")

    conv_hist: List[Dict[str, Any]] = []
    eval_counter = [0]

    # Batched evaluation runner
    def run_pool_evaluate(sim_local: Dict[str,Any], cands: List[Dict[str,Any]], phase_label: str,
                          batch_size: int = 2000) -> Dict[str, Any]:
        nonlocal_best = None
        total = len(cands)
        info(f"{phase_label}: plan to evaluate {total} candidates in batches of {batch_size}")

        def submit_batch(executor, start):
            end = min(start + batch_size, total)
            packs = [(p, sim_local) for p in cands[start:end]]
            futs = {executor.submit(eval_one_pack, pack): i for i, pack in enumerate(packs, start)}
            return futs, end

        if args.outer_backend == "process":
            info(f"{phase_label}: ProcessPoolExecutor (Windows-friendly)")
            with ProcessPoolExecutor(max_workers=args.search_workers) as pool:
                done_total = 0
                next_start = 0
                futs, next_start = submit_batch(pool, next_start)
                report_every = max(1, total // 20)
                while futs:
                    for fut in as_completed(list(futs.keys())):
                        _idx = futs.pop(fut)
                        total_val, per_bomb, misc, p = fut.result()
                        done_total += 1; eval_counter[0] += 1
                        if (nonlocal_best is None) or (total_val > nonlocal_best["total"]):
                            nonlocal_best = dict(total=total_val, secs_each=per_bomb, **misc, params=p)
                            info(f"{phase_label}: new best {nonlocal_best['total']:.6f}s | params={p}")
                        conv_hist.append(dict(idx=eval_counter[0],
                                              best=(nonlocal_best["total"] if nonlocal_best else 0.0),
                                              phase=phase_label))
                        if done_total % report_every == 0:
                            info(f"{phase_label}: progress {done_total}/{total} ({int(100*done_total/total)}%)")
                    if next_start < total:
                        new_futs, next_start = submit_batch(pool, next_start)
                        futs.update(new_futs)
        elif args.outer_backend == "thread":
            info(f"{phase_label}: ThreadPoolExecutor")
            with ThreadPoolExecutor(max_workers=args.search_workers) as pool:
                done_total = 0
                next_start = 0
                futs, next_start = submit_batch(pool, next_start)
                report_every = max(1, total // 20)
                while futs:
                    for fut in as_completed(list(futs.keys())):
                        _idx = futs.pop(fut)
                        total_val, per_bomb, misc, p = fut.result()
                        done_total += 1; eval_counter[0] += 1
                        if (nonlocal_best is None) or (total_val > nonlocal_best["total"]):
                            nonlocal_best = dict(total=total_val, secs_each=per_bomb, **misc, params=p)
                            info(f"{phase_label}: new best {nonlocal_best['total']:.6f}s | params={p}")
                        conv_hist.append(dict(idx=eval_counter[0],
                                              best=(nonlocal_best["total"] if nonlocal_best else 0.0),
                                              phase=phase_label))
                        if done_total % report_every == 0:
                            info(f"{phase_label}: progress {done_total}/{total} ({int(100*done_total/total)}%)")
                    if next_start < total:
                        new_futs, next_start = submit_batch(pool, next_start)
                        futs.update(new_futs)
        else:
            info(f"{phase_label}: sequential")
            for i, p in enumerate(cands, 1):
                total_val, per_bomb, misc, p = eval_one_pack((p, sim_local))
                eval_counter[0] += 1
                if (nonlocal_best is None) or (total_val > nonlocal_best["total"]):
                    nonlocal_best = dict(total=total_val, secs_each=per_bomb, **misc, params=p)
                    info(f"{phase_label}: new best {nonlocal_best['total']:.6f}s | params={p}")
                conv_hist.append(dict(idx=eval_counter[0],
                                      best=(nonlocal_best["total"] if nonlocal_best else 0.0),
                                      phase=phase_label))
                if i % max(1, len(cands) // 20) == 0:
                    info(f"{phase_label}: progress {i}/{len(cands)} ({int(100*i/len(cands))}%)")
        return nonlocal_best

    # Stage1
    t1_a = time.time()
    best = run_pool_evaluate(sim, candidates, "stage1", batch_size=2000)
    t1_b = time.time()
    if best is None:
        info("Stage1 found no feasible solution. Expand ranges or refine steps.")
        return
    info(f"Stage1 done: {t1_b - t1_a:.2f}s, best={best['total']:.6f}s | {best['params']}")

    # Stage2 refinement around Stage1 best
    if args.stage2_enable:
        def neighbors(seed: Dict[str, Any]) -> List[Dict[str, Any]]:
            hd0 = seed["heading_offset_deg"]; sp0 = seed["uav_speed"]
            t1, t2, t3 = seed["drops"]; f1, f2, f3 = seed["fuses"]
            span = args.stage2_span_steps
            Hs = [hd0 + i*args.stage2_heading_step_deg for i in range(-span, span+1)]
            Ss = [sp0 + i*args.stage2_speed_step for i in range(-span, span+1)]
            T1s = [t1 + i*args.stage2_t_step for i in range(-span, span+1)]
            G12s= [max(1.0, (t2-t1) + i*args.stage2_t_step) for i in range(-span, span+1)]
            G23s= [max(1.0, (t3-t2) + i*args.stage2_t_step) for i in range(-span, span+1)]
            Fs  = [max(0.1, f1 + i*args.stage2_fuse_step) for i in range(-span, span+1)]

            cands2=[]
            for hd in Hs:
                for sp in Ss:
                    for tt1 in T1s:
                        for g12 in G12s:
                            for g23 in G23s:
                                _t1 = max(0.0, tt1); _t2 = _t1 + g12; _t3 = _t2 + g23
                                for _f1 in Fs:
                                    for _f2 in Fs:
                                        for _f3 in Fs:
                                            cands2.append(dict(
                                                heading_offset_deg=float(hd),
                                                uav_speed=float(sp),
                                                drops=[float(_t1), float(_t2), float(_t3)],
                                                fuses=[float(_f1), float(_f2), float(_f3)],
                                            ))
            return cands2

        import random
        seed = dict(best["params"])
        cands_stage2 = neighbors(seed)
        info(f"Stage2: built {len(cands_stage2)} candidates (raw)")

        if len(cands_stage2) > args.stage2_max_cands:
            info(f"Stage2 exceeds cap; sampling to {args.stage2_max_cands}")
            random.seed(0)
            cands_stage2 = random.sample(cands_stage2, args.stage2_max_cands)

        sim_stage2 = dict(sim); sim_stage2["inner_log"] = False  # quiet workers
        t2_a = time.time()
        best2 = run_pool_evaluate(sim_stage2, cands_stage2, "stage2", batch_size=args.stage2_batch_size)
        t2_b = time.time()
        if best2 and best2["total"] > best["total"]:
            best = best2
        info(f"Stage2 done: {t2_b - t2_a:.2f}s, current best={best['total']:.6f}s | {best['params']}")

    # Coordinate Descent
    if args.cd_enable:
        steps = dict(
            heading_offset_deg=args.cd_step_heading,
            uav_speed=args.cd_step_speed,
            t1=args.cd_step_time, gap12=args.cd_step_time, gap23=args.cd_step_time,
            f1=args.cd_step_fuse, f2=args.cd_step_fuse, f3=args.cd_step_fuse,
        )
        bounds = dict(
            heading_offset_deg=(args.heading_min_deg-5*args.heading_step_deg,
                                args.heading_max_deg+5*args.heading_step_deg),
            uav_speed=(min(args.speed_min, 70.0), max(args.speed_max, 140.0)),
            t1=(max(0.0, args.t1_min-5*args.t1_step), args.t1_max+5*args.t1_step),
            gap12=(1.0, args.gap12_max+5*args.gap12_step),
            gap23=(1.0, args.gap23_max+5*args.gap23_step),
            fuse=(0.1, args.fuse_max+5*args.fuse_step)
        )
        sim_cd = dict(sim); sim_cd["inner_log"] = False
        info(f"Coordinate descent start (rounds={args.cd_rounds})")
        best_params_after_cd, best_val_after_cd = coordinate_descent(
            seed_params=dict(best["params"]),
            sim=sim_cd,
            steps=steps,
            bounds=bounds,
            rounds=args.cd_rounds,
            conv_hist=conv_hist,
            eval_counter=eval_counter
        )
        val_final, per_bomb_final, misc_final = evaluate_strategy(best_params_after_cd, sim)
        eval_counter[0] += 1
        conv_hist.append(dict(idx=eval_counter[0], best=max(best["total"], val_final), phase="cd"))
        if val_final > best["total"]:
            best = dict(total=val_final, secs_each=per_bomb_final, **misc_final, params=best_params_after_cd)
            info(f"Coordinate descent improved to {best['total']:.6f}s | {best['params']}")
        else:
            info(f"Coordinate descent did not beat current best ({best['total']:.6f}s)")

    # Final outputs
    info("="*90)
    info(f"Final best UNION covered time (cooperative) = {best['total']:.6f} s | params: {best['params']}")
    for k, s in enumerate(best["secs_each"], 1):
        info(f"Bomb#{k} single-bomb FULL cover time = {s:.6f} s")
    for k in range(3):
        dp = best["drops_pos"][k]; ep = best["bombs_expl"][k]
        info(f"Bomb#{k+1} drop=({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f})  "
             f"expl=({ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f})  "
             f"t_drop={best['t_drops'][k]:.3f}s  t_expl={best['t_expls'][k]:.3f}s")

    info("Write result1.xlsx …")
    write_excel(best, path="result1.xlsx")
    info("Saved result1.xlsx")

    info("Save convergence plot q3_convergence.png …")
    # Clean style by default; set shade=True if you want colored spans
    save_convergence_plot(conv_hist, path="q3_convergence.png", shade=False)
    info("Saved convergence plot.")
    info("="*90)

if __name__ == "__main__":
    main()
