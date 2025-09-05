# -*- coding: utf-8 -*-
"""
Q2_Optimize_GivenHV.py
在仅已知 FY1 的航向与速度的前提下，优化 (t_drop, tau)，最大化对 M1 的遮蔽时间。
- 航向、速度固定；速度自动限幅 [70, 140] m/s
- 由 (heading, speed, t_drop, tau) 推导投放点与起爆点
- 评估采用与 Q1 一致的严格圆锥判据（优先导入 Q1Solver / Q1Solver_visual）
- 三阶段：LHS 粗筛 -> Pattern 局部精修 -> 终评（高精度）
- 并行评估；只按 ~5% 进度打印，不刷屏

示例（中等规模）：
  python Q2_Optimize_GivenHV.py --heading-deg 6.33 --speed 120 \
    --lhs 4096 --topk 32 --pat-iter 60 --workers auto \
    --dt-coarse 0.002 --nphi-coarse 480 --nz-coarse 9 \
    --dt-final 0.001 --nphi-final 960 --nz-final 13
"""

import os
import math
import time
import argparse
import numpy as np
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# 0) 导入 Q1 几何/判据；失败则回退一致实现
# -----------------------------
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
except Exception:
    # ---- 回退实现（与题意一致）----
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
        v = MISSILE_SPEED * Unit(FAKE_TARGET_ORIGIN - mInit)
        return mInit + v * t, v

    def UavStateHorizontal(t, uavInit, uavSpeed, headingRad):
        vx = uavSpeed * math.cos(headingRad)
        vy = uavSpeed * math.sin(headingRad)
        return np.array([uavInit[0] + vx * t,
                         uavInit[1] + vy * t,
                         uavInit[2]], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    CYL_BASE = np.array([0.0, 200.0, 0.0], dtype=float)
    CYL_R, CYL_H = 7.0, 10.0
    def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
        b = CYL_BASE.astype(dtype)
        r, h = dtype(CYL_R), dtype(CYL_H)
        phis = np.linspace(0.0, 2.0*math.pi, nPhi, endpoint=False, dtype=dtype)
        c, s = np.cos(phis), np.sin(phis)
        ring = np.stack([r*c, r*s, np.zeros_like(c)], axis=1).astype(dtype)
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
    # ---------------------------

# -----------------------------
# 1) 题面硬约束 & 工具
# -----------------------------
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DROP_MIN, DROP_MAX = 0.0, 60.0             # t_drop ∈ [0, 60]
FUSE_MIN, FUSE_MAX = 0.0, 18.0             # tau    ∈ [0, 18]

def clamp_speed(v: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(v)))

def heading_to_unit(heading_rad: float) -> np.ndarray:
    return np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0], dtype=float)

def missile_hit_time() -> float:
    return float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

def explosion_from_times(heading_rad: float, speed: float, t_drop: float, tau: float, dtype=np.float64):
    drop_pos, uavV = UavStateHorizontal(t_drop, FY1_INIT.astype(dtype), dtype(speed), heading_rad)
    expl_xy = drop_pos[:2] + uavV[:2] * dtype(tau)
    expl_z  = drop_pos[2] - dtype(0.5) * dtype(g) * (dtype(tau)**2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=dtype)

# -----------------------------
# 2) 遮蔽评估（固定 heading/speed；优化 t_drop, tau）
# -----------------------------
def evaluate_occlusion_fixed_hv(heading: float, speed: float,
                                t_drop: float, tau: float,
                                dt=0.002, nphi=480, nz=9,
                                margin=EPS, block=8192) -> Tuple[float, np.ndarray, float]:
    """
    返回：(遮蔽时长 seconds, 起爆点, 命中时间)
    """
    speed = clamp_speed(speed)
    # 起爆时刻与点
    t0 = t_drop + tau
    hit = missile_hit_time()
    if t0 >= hit:  # 起爆晚于命中，不可能遮蔽
        return 0.0, np.array([0,0,-1.0], dtype=float), hit

    expl = explosion_from_times(heading, speed, t_drop, tau, dtype=float)
    if expl[2] <= 0.0:  # 地下/地面，失效
        return 0.0, expl, hit

    # 有效窗
    t1 = min(t0 + SMOG_EFFECT_TIME, hit)
    if t1 <= t0:
        return 0.0, expl, hit

    # 网格与目标点
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)
    pts = PreCalCylinderPoints(nphi, nz, dtype=float)

    mask = np.zeros(len(t_grid), dtype=bool)
    for i, t in enumerate(t_grid):
        m_pos, _ = MissileState(float(t), M1_INIT)
        c_pos = np.array([expl[0], expl[1], expl[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype=float)
        mask[i] = ConeAllPointsIn(m_pos, c_pos, pts, rCloud=SMOG_R, margin=margin, block=block)

    return float(np.count_nonzero(mask) * dt), expl, hit

# --- 并行包装（便于批量评估） ---
def _eval_tuple(args):
    heading, speed, t_drop, tau, dt, nphi, nz, margin, block = args
    sec, expl, hit = evaluate_occlusion_fixed_hv(heading, speed, t_drop, tau, dt, nphi, nz, margin, block)
    return sec, (t_drop, tau), expl, hit

# -----------------------------
# 3) 采样 + 局部精修（2 维 Pattern）
# -----------------------------
def lhs_rect(n: int, bounds: List[Tuple[float,float]]) -> np.ndarray:
    """
    简易 2D LHS：bounds = [(t_drop_min,max), (tau_min,max)]
    """
    rng = np.random.default_rng()
    d = len(bounds)
    u = (rng.random((n,d)) + np.arange(n)[:,None]) / n
    rng.shuffle(u, axis=0)
    out = np.empty_like(u)
    for j,(lo,hi) in enumerate(bounds):
        out[:,j] = lo + u[:,j]*(hi-lo)
    # 轻微抬升极小 t_drop，避免 t0≈0 的退化
    out[:,0] = np.maximum(out[:,0], 0.02*(bounds[0][1]-bounds[0][0])/n)
    return out

def pattern_search_2d(heading: float, speed: float, x0: np.ndarray,
                      eval_kwargs: dict,
                      steps=(0.6, 0.4),
                      shrink=0.6, max_iter=60,
                      workers=None) -> Tuple[float, np.ndarray, np.ndarray, float, List[float]]:
    """
    在 (t_drop, tau) 上做 2D 模式搜索。
    返回：best_val, best_x, best_expl, hit_time, hist
    """
    # 初评
    best_val, best_expl, best_hit = evaluate_occlusion_fixed_hv(
        heading, speed, x0[0], x0[1], **eval_kwargs)
    hist = [best_val]
    s = np.array(steps, dtype=float)

    for it in range(max_iter):
        cands = np.array([
            [x0[0]+s[0], x0[1]], [x0[0]-s[0], x0[1]],
            [x0[0], x0[1]+s[1]], [x0[0], x0[1]-s[1]],
            [x0[0]+s[0], x0[1]+s[1]], [x0[0]-s[0], x0[1]-s[1]],
            [x0[0]+s[0], x0[1]-s[1]], [x0[0]-s[0], x0[1]+s[1]],
        ], dtype=float)

        # 边界裁剪
        cands[:,0] = np.clip(cands[:,0], DROP_MIN, DROP_MAX)
        cands[:,1] = np.clip(cands[:,1], FUSE_MIN, FUSE_MAX)

        tasks = [(heading, speed, c[0], c[1],
                  eval_kwargs["dt"], eval_kwargs["nphi"], eval_kwargs["nz"],
                  eval_kwargs["margin"], eval_kwargs["block"]) for c in cands]
        vals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
            for fut in as_completed(futs):
                sec, (td,tf), expl, hit = fut.result()
                vals.append((sec, np.array([td,tf], dtype=float), expl, hit))

        improved = False
        for sec, x, expl, hit in vals:
            if sec > best_val + 1e-12:
                best_val, best_expl, best_hit = sec, expl, hit
                x0 = x
                improved = True

        hist.append(best_val)
        if improved:
            continue

        s *= shrink
        if np.all(s < np.array([1e-3, 1e-3])):
            break

    return best_val, x0, best_expl, best_hit, hist

# -----------------------------
# 4) 主流程：LHS -> Pattern -> 终评
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Q2 Optimize (Given heading & speed)")
    ap.add_argument("--heading-deg", type=float, default=None, help="航向角（度）")
    ap.add_argument("--heading-rad", type=float, default=None, help="航向角（弧度）")
    ap.add_argument("--speed", type=float, required=True, help="FY1 速度（m/s），自动限幅到[70,140]")

    # 规模控制
    ap.add_argument("--lhs", type=int, default=4096, help="LHS 粗筛样本数")
    ap.add_argument("--topk", type=int, default=32, help="进入 Pattern 的候选个数")
    ap.add_argument("--pat-iter", type=int, default=60, help="Pattern 搜索最大迭代")
    ap.add_argument("--workers", default="auto", help="并行进程数，int 或 'auto'")

    # 粗精度
    ap.add_argument("--dt-coarse", type=float, default=0.002)
    ap.add_argument("--nphi-coarse", type=int, default=480)
    ap.add_argument("--nz-coarse", type=int, default=9)

    # 终评精度
    ap.add_argument("--dt-final", type=float, default=0.001)
    ap.add_argument("--nphi-final", type=int, default=960)
    ap.add_argument("--nz-final", type=int, default=13)

    args = ap.parse_args()

    # 航向解析
    if (args.heading_deg is None) == (args.heading_rad is None):
        raise SystemExit("请用 --heading-deg 或 --heading-rad 之一指定航向角。")
    heading = (args.heading_rad if args.heading_rad is not None else math.radians(args.heading_deg))
    speed = clamp_speed(args.speed)

    workers = None if str(args.workers).lower() == "auto" else int(args.workers)

    # 避免 BLAS 线程内卷
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # 阶段 1：LHS 粗筛
    tA = time.time()
    bounds = [(DROP_MIN, DROP_MAX), (FUSE_MIN, FUSE_MAX)]
    samples = lhs_rect(args.lhs, bounds)

    tasks = [(heading, speed, s[0], s[1],
              args.dt_coarse, args.nphi_coarse, args.nz_coarse, EPS, 8192) for s in samples]

    vals = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        done, stride = 0, max(1, len(tasks)//20)
        best_so_far = 0.0
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            vals.append((sec, x, expl, hit))
            done += 1
            if sec > best_so_far: best_so_far = sec
            if done % stride == 0 or done == len(tasks):
                pct = int(100*done/len(tasks))
                print(f"[LHS] 进度 {pct:3d}%  best_so_far={best_so_far:.6f}")

    vals.sort(key=lambda z: z[0], reverse=True)
    top = vals[:min(args.topk, len(vals))]
    if not top or top[0][0] <= 0.0:
        print("[Info] LHS 未找到有效遮蔽（>0），仍将尝试 Pattern 精修。")

    # 阶段 2：Pattern 精修（对 Top-K 逐个）
    refined = []
    global_best = 0.0
    hist_all = []
    for i,(sec0, x0, expl0, hit0) in enumerate(top, start=1):
        print(f"[PAT] seed#{i:02d} 初值={sec0:.6f}")
        best_sec, best_x, best_expl, best_hit, hist = pattern_search_2d(
            heading, speed, x0, eval_kwargs=dict(dt=args.dt_coarse, nphi=args.nphi_coarse, nz=args.nz_coarse, margin=EPS, block=8192),
            steps=(0.8, 0.6), shrink=0.6, max_iter=args.pat_iter, workers=workers
        )
        refined.append((best_sec, best_x, best_expl, best_hit))
        hist_all.append(hist)
        global_best = max(global_best, best_sec)
        # 每 ~5% 的 seed 输出一次进度
        if i % max(1, len(top)//20) == 0 or i == len(top):
            pct = int(100*i/len(top))
            print(f"[PAT] 进度 {pct:3d}%  global_best={global_best:.6f}")

    refined.sort(key=lambda z: z[0], reverse=True)
    candidates = refined[:min(10, len(refined))] if refined else top[:min(10, len(top))]

    # 阶段 3：终评（高精度）
    finals = []
    tasks = [(heading, speed, c[1][0], c[1][1],
              args.dt_final, args.nphi_final, args.nz_final, EPS, 8192) for c in candidates]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_tuple, t): i for i,t in enumerate(tasks)}
        done, stride = 0, max(1, len(tasks)//20)
        for fut in as_completed(futs):
            sec, x, expl, hit = fut.result()
            finals.append((sec, x, expl, hit))
            done += 1
            if done % stride == 0 or done == len(tasks):
                pct = int(100*done/len(tasks))
                print(f"[FINAL] 进度 {pct:3d}%")

    if finals:
        finals.sort(key=lambda z: z[0], reverse=True)
        best = finals[0]
        best_sec, best_x, best_expl, best_hit = best
    else:
        # 极端情况下（全部为 0），回退到 LHS 的最好一个
        best_sec, best_x, best_expl, best_hit = top[0]

    tB = time.time()
    print("="*80)
    print(f"[Best] heading={heading:.6f} rad, speed={speed:.3f} m/s")
    print(f"       t_drop={best_x[0]:.3f} s, tau={best_x[1]:.3f} s, t0={best_x[0]+best_x[1]:.3f} s")
    print(f"       explosion=({best_expl[0]:.2f},{best_expl[1]:.2f},{best_expl[2]:.2f})")
    print(f"       occlusion={best_sec:.6f} s | total time={tB - tA:.2f}s")
    print("="*80)

    # 保存报告
    save_report("Q2_HV_Results.txt", heading, speed, best_x, best_expl, best_sec,
                dict(dt_coarse=args.dt_coarse, nphi_coarse=args.nphi_coarse, nz_coarse=args.nz_coarse,
                     dt_final=args.dt_final, nphi_final=args.nphi_final, nz_final=args.nz_final))

    # 收敛图（把各 seed 的 hist 叠在一起 + 终点水平线）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        for j,h in enumerate(hist_all, start=1):
            ax.plot(range(len(h)), h, linewidth=1.1, alpha=0.6)
        ax.axhline(y=best_sec, linestyle='--', linewidth=2, label=f'Final Best: {best_sec:.6f}s')
        ax.set_xlabel('Pattern iteration (per seed)')
        ax.set_ylabel('Best-so-far (s)')
        ax.set_title('Convergence (Pattern refinement, top-K seeds)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("Q2_HV_Convergence.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[Plot] Saved Q2_HV_Convergence.png")
    except Exception as e:
        print(f"[Plot] Skip: {e}")

def save_report(filename: str, heading: float, speed: float,
                best_x: np.ndarray, best_expl: np.ndarray, best_sec: float,
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
        f.write(f"t_drop (s) = {best_x[0]:.6f}\n")
        f.write(f"tau   (s)  = {best_x[1]:.6f}\n")
        f.write(f"t0    (s)  = {best_x[0]+best_x[1]:.6f}\n")
        f.write(f"Explosion point = ({best_expl[0]:.6f}, {best_expl[1]:.6f}, {best_expl[2]:.6f})\n\n")

        f.write("Evaluation Config:\n")
        f.write("-"*40 + "\n")
        f.write(f"coarse: dt={cfg['dt_coarse']}, nphi={cfg['nphi_coarse']}, nz={cfg['nz_coarse']}\n")
        f.write(f"final : dt={cfg['dt_final']}, nphi={cfg['nphi_final']}, nz={cfg['nz_final']}\n\n")

        f.write("Result:\n")
        f.write("-"*40 + "\n")
        f.write(f"Occlusion duration = {best_sec:.6f} s\n")
        f.write("="*80 + "\n")
    print(f"[Info] Results saved to {filename}")

if __name__ == "__main__":
    main()
