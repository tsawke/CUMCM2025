# -*- coding: utf-8 -*-
"""
Q2_Verifier.py
验证：给定 FY1 的飞行方向、速度、投放点/时刻、起爆点/引信时延，计算对 M1 的遮蔽时长。
- 物理一致性校验（水平等速 + 竖直自由落体）
- 速度限幅 [70, 140] m/s
- 与 Q1 一致的严格圆锥判据
- 并行评估 + 进度提示（每~5%一次）
"""

import os
import math
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# -----------------------------
# 0) 尝试从 Q1Solver_visual 复用常量/函数；失败则回退一致实现
# -----------------------------
Q1_IMPORTED = False
try:
    import Q1Solver_visual as Q1
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
    # ---- 回退实现（保持与 Q1 文件一致的判据与参数）----
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

    # 目标圆柱（见题面）
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
    # ---------------------------------------------------

# -----------------------------
# 1) 题面硬约束 & 实用函数
# -----------------------------
SPEED_MIN, SPEED_MAX = 70.0, 140.0  # 题意限速

def clamp_speed(v: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(v)))

def heading_to_unit(heading_rad: float) -> np.ndarray:
    return np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0], dtype=float)

def missile_hit_time() -> float:
    return float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

def parse_xyz(s: str) -> np.ndarray:
    try:
        x, y, z = [float(t) for t in s.split(",")]
        return np.array([x, y, z], dtype=float)
    except Exception:
        raise ValueError(f"坐标解析失败，应为 'x,y,z'，得到：{s}")

# -----------------------------
# 2) 由“空间点”求时间参数，并做一致性校验
# -----------------------------
def times_from_points(heading_rad: float, speed: float,
                      drop_xyz: np.ndarray, expl_xyz: np.ndarray,
                      tol_xy: float = 0.5, tol_z: float = 0.2):
    dir2 = heading_to_unit(heading_rad)
    v = clamp_speed(speed)

    d_vec = drop_xyz - FY1_INIT
    t_drop = float(np.dot(d_vec, dir2) / (v + 1e-12))
    proj_xy = FY1_INIT[:2] + dir2[:2] * v * t_drop
    err_xy = float(np.linalg.norm(proj_xy - drop_xyz[:2]))
    err_z  = float(abs(drop_xyz[2] - FY1_INIT[2]))

    dz = float(drop_xyz[2] - expl_xyz[2])
    tau_z = math.sqrt(max(0.0, 2.0*dz / g))
    dxy = float(np.linalg.norm(expl_xyz[:2] - drop_xyz[:2]))
    tau_xy = dxy / (v + 1e-12)
    tau = 0.5*(tau_z + tau_xy)
    t0 = t_drop + tau

    warn = []
    if err_xy > tol_xy:
        warn.append(f"投放点与航迹偏差较大：|Δxy|≈{err_xy:.3f} m（阈值{tol_xy}），已用投影时间 t_drop={t_drop:.4f}s 近似。")
    if err_z > tol_z:
        warn.append(f"投放点高度与等高度假设不符：|Δz|≈{err_z:.3f} m（阈值{tol_z}）。")
    if abs(tau_z - tau_xy) > 0.1:
        warn.append(f"起爆点水平/竖直反解的引信时延差异较大：tau_z={tau_z:.3f}s, tau_xy={tau_xy:.3f}s，取折中 tau={tau:.3f}s。")

    return dict(speed=v, t_drop=max(0.0, t_drop), tau=max(0.0, tau), t0=max(0.0, t0), warn=warn)

# -----------------------------
# 3) 并行评估的“全局上下文” + 顶层 worker（可被 pickle）
# -----------------------------
_EVAL_CTX = None  # 每轮评估在子进程/线程里的只读上下文

def _set_eval_context(ctx: dict):
    """供进程/线程池在启动时调用，设置全局只读上下文。"""
    global _EVAL_CTX
    _EVAL_CTX = ctx

def _eval_chunk_worker(args):
    """
    顶层 worker（可被 pickle）。只从 _EVAL_CTX 取常量，不闭包任何局部变量。
    参数:
      args = (idx0, t_chunk)
    返回:
      (idx0, mask_chunk[bool])
    """
    idx0, t_chunk = args
    ctx = _EVAL_CTX
    t0 = ctx["t0"]
    expl_pos = ctx["expl_pos"]
    pts = ctx["pts"]
    margin = ctx["margin"]
    block = ctx["block"]
    dtype = ctx["dtype"]

    out = np.zeros(len(t_chunk), dtype=bool)
    for i, t in enumerate(t_chunk):
        m_pos, _ = MissileState(float(t), M1_INIT)
        c_pos = np.array([expl_pos[0], expl_pos[1],
                          expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)], dtype=dtype)
        out[i] = ConeAllPointsIn(m_pos, c_pos, pts, rCloud=SMOG_R, margin=margin, block=block)
    return idx0, out

# -----------------------------
# 4) 评估：严格圆锥判据（支持并行 + 进度）
# -----------------------------
def eval_occlusion(t0: float, dt: float, nphi: int, nz: int,
                   expl_pos: np.ndarray,
                   margin: float = EPS, block: int = 8192,
                   backend: str = "process", workers: int | None = None,
                   chunk: int = 800, fp32: bool = False):

    hit = missile_hit_time()
    t1 = min(t0 + SMOG_EFFECT_TIME, hit)
    if t1 <= t0 or expl_pos[2] <= 0.0:
        return 0.0, hit

    dtype = np.float32 if fp32 else np.float64
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=dtype)
    pts = PreCalCylinderPoints(nphi, nz, dtype=dtype)
    mask = np.zeros(len(t_grid), dtype=bool)

    # 切分任务
    chunks = [(i, t_grid[i:i+chunk]) for i in range(0, len(t_grid), chunk)]

    # 上下文（在每个子进程/线程中只存一份）
    ctx = dict(
        t0=float(t0),
        expl_pos=np.array(expl_pos, dtype=dtype),
        pts=pts, margin=float(margin), block=int(block),
        dtype=dtype
    )

    # 选择执行器
    if backend == "thread":
        # 线程不支持 initializer，就直接在主线程设置一次
        _set_eval_context(ctx)
        Pool = ThreadPoolExecutor
        pool_kwargs = dict(max_workers=workers)
    else:
        Pool = ProcessPoolExecutor
        # 避免 BLAS 线程内卷
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        pool_kwargs = dict(max_workers=workers, initializer=_set_eval_context, initargs=(ctx,))

    tA = time.time()
    with Pool(**pool_kwargs) as pool:
        futs = {pool.submit(_eval_chunk_worker, (i, c)): i for i, c in chunks}
        done = 0
        stride = max(1, len(chunks)//20)  # ~5%
        for fut in as_completed(futs):
            i0 = futs[fut]
            idx, part = fut.result()
            L = len(part)
            mask[idx:idx+L] = part
            done += 1
            if done % stride == 0 or done == len(chunks):
                pct = int(100*done/len(chunks))
                print(f"[Verify] 进度 {pct:3d}%")

    seconds = float(np.count_nonzero(mask) * dt)
    tB = time.time()
    print(f"[Verify] 遮蔽时长 = {seconds:.6f} s | 运行时间 {tB - tA:.2f}s")
    return seconds, hit

# -----------------------------
# 5) 起爆点计算（由时间参数）
# -----------------------------
def explosion_from_times(heading_rad: float, speed: float, t_drop: float, tau: float, dtype=np.float64):
    drop_pos, uavV = UavStateHorizontal(t_drop, FY1_INIT.astype(dtype), dtype(speed), heading_rad)
    expl_xy = drop_pos[:2] + uavV[:2] * dtype(tau)
    expl_z  = drop_pos[2] - dtype(0.5) * dtype(g) * (dtype(tau)**2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=dtype)

# -----------------------------
# 6) 主流程（解析两种输入形态）
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Q2 Verifier")
    # 航向：deg / rad 选一
    ap.add_argument("--heading-deg", type=float, default=None, help="航向角（度）")
    ap.add_argument("--heading-rad", type=float, default=None, help="航向角（弧度）")
    ap.add_argument("--speed", type=float, required=True, help="FY1 速度（m/s），自动限幅到[70,140]")

    # 输入方案 A：空间点
    ap.add_argument("--drop", type=str, default=None, help="投放点 'x,y,z'")
    ap.add_argument("--expl", type=str, default=None, help="起爆点 'x,y,z'")

    # 输入方案 B：时间
    ap.add_argument("--drop-time", type=float, default=None, help="投放时刻（s）")
    ap.add_argument("--fuse", type=float, default=None, help="引信时延 τ（s）")

    # 评估网格 & 并行
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--nphi", type=int, default=960)
    ap.add_argument("--nz", type=int, default=13)
    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", default="auto", help="并行进程/线程数，int 或 'auto'")
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--block", type=int, default=8192)
    ap.add_argument("--margin", type=float, default=EPS)
    ap.add_argument("--fp32", action="store_true")

    args = ap.parse_args()

    # 航向解析
    if (args.heading_deg is None) == (args.heading_rad is None):
        raise SystemExit("请用 --heading-deg 或 --heading-rad 之一指定航向角。")
    heading = (args.heading_rad if args.heading_rad is not None else math.radians(args.heading_deg))
    speed = clamp_speed(args.speed)

    # 两种输入形态互斥
    has_points = (args.drop is not None or args.expl is not None)
    has_times  = (args.drop_time is not None or args.fuse is not None)
    if has_points and has_times:
        raise SystemExit("请二选一：提供 (drop, expl) 空间点 或 (drop-time, fuse) 时间参数。")
    if not has_points and not has_times:
        raise SystemExit("请提供 (drop, expl) 或 (drop-time, fuse)。")

    # 并行工作数
    workers = None if str(args.workers).lower() == "auto" else int(args.workers)

    # 计算
    if has_points:
        if args.drop is None or args.expl is None:
            raise SystemExit("采用“空间点”输入时，必须同时提供 --drop 与 --expl。")
        drop_xyz = parse_xyz(args.drop)
        expl_xyz = parse_xyz(args.expl)

        # 由空间点反解时间 + 一致性校验
        info = times_from_points(heading, speed, drop_xyz, expl_xyz)
        if info["warn"]:
            for w in info["warn"]:
                print("[Warn]", w)

        # 由“纠正后”的时间计算严格物理一致的起爆点（用于评估）
        t_drop, tau, t0 = info["t_drop"], info["tau"], info["t0"]
        expl_eval = explosion_from_times(heading, speed, t_drop, tau, dtype=(np.float32 if args.fp32 else np.float64))
        print(f"[Use] speed={info['speed']:.3f} m/s, t_drop={t_drop:.4f} s, tau={tau:.4f} s, t0={t0:.4f} s")
        print(f"[Use] explosion(eval) = ({expl_eval[0]:.3f},{expl_eval[1]:.3f},{expl_eval[2]:.3f})")

        # 评估
        sec, hit = eval_occlusion(t0, args.dt, args.nphi, args.nz, expl_eval,
                                  margin=args.margin, block=args.block,
                                  backend=args.backend, workers=workers,
                                  chunk=args.chunk, fp32=args.fp32)
        # 报告
        save_report("Q2Verify_Results.txt", heading, speed, t_drop, tau, expl_eval, sec, hit, args)

    else:
        # 直接用时间参数
        if args.drop_time is None or args.fuse is None:
            raise SystemExit("采用“时间参数”输入时，必须同时提供 --drop-time 与 --fuse。")
        t_drop = float(args.drop_time)
        tau    = float(args.fuse)
        t0     = t_drop + tau
        # 起爆点
        expl_eval = explosion_from_times(heading, speed, t_drop, tau, dtype=(np.float32 if args.fp32 else np.float64))
        if expl_eval[2] <= 0.0:
            print("[Warn] 起爆点高度<=0，物理不可用，遮蔽=0。")
        print(f"[Use] speed={speed:.3f} m/s, t_drop={t_drop:.4f} s, tau={tau:.4f} s, t0={t0:.4f} s")
        print(f"[Use] explosion(eval) = ({expl_eval[0]:.3f},{expl_eval[1]:.3f},{expl_eval[2]:.3f})")

        sec, hit = eval_occlusion(t0, args.dt, args.nphi, args.nz, expl_eval,
                                  margin=args.margin, block=args.block,
                                  backend=args.backend, workers=workers,
                                  chunk=args.chunk, fp32=args.fp32)
        save_report("Q2Verify_Results.txt", heading, speed, t_drop, tau, expl_eval, sec, hit, args)


def save_report(filename, heading, speed, t_drop, tau, expl_pos, occl, hit, args):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("Q2 Verification Results Report\n")
        f.write("="*80 + "\n\n")
        f.write("Inputs & Parsed Parameters:\n")
        f.write("-"*40 + "\n")
        f.write(f"Heading (rad) = {heading:.9f}\n")
        f.write(f"Speed (m/s)   = {speed:.3f}  (clamped to [70,140])\n")
        f.write(f"Drop time (s) = {t_drop:.6f}\n")
        f.write(f"Fuse   tau (s)= {tau:.6f}\n")
        f.write(f"Explode t0 (s)= {t_drop + tau:.6f}\n")
        f.write(f"Explosion point (eval) = ({expl_pos[0]:.6f}, {expl_pos[1]:.6f}, {expl_pos[2]:.6f})\n\n")

        f.write("Evaluation Config:\n")
        f.write("-"*40 + "\n")
        f.write(f"dt = {args.dt}\n")
        f.write(f"nphi = {args.nphi}, nz = {args.nz}\n")
        f.write(f"backend = {args.backend}, workers = {args.workers}\n")
        f.write(f"chunk = {args.chunk}, block = {args.block}\n")
        f.write(f"margin = {args.margin:g}, fp32 = {args.fp32}\n\n")

        f.write("Results:\n")
        f.write("-"*40 + "\n")
        f.write(f"Occlusion duration = {occl:.6f} s\n")
        f.write(f"Missile hit time ≈ {hit:.6f} s\n")
        f.write("="*80 + "\n")
    print(f"[Info] Results saved to {filename}")


if __name__ == "__main__":
    main()
