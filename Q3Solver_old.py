# -*- coding: utf-8 -*-
"""
Q3Solver.py — 问题3：FY1 投放 3 枚烟幕干扰弹干扰 M1，最大化总遮蔽时间（可不连续）
多进程跑满 CPU（参考 Q1 的做法）：
- 默认 backend="process"，workers=os.cpu_count()
- 将 MKL/OMP/NumExpr 线程数固定为 1，避免过度超订阅
- 其余逻辑与先前版本一致，按时间块并行计算遮蔽布尔向量
"""
import os, math, time, argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import Q1Solver_visual as Q1

# ========== 可调参数（保持和 Q1 风格一致） ==========
EPS = 1e-12

DEFAULTS = dict(
    dt     = 0.0015,
    nphi   = 720,
    nz     = 11,
    backend= "process",                # 默认使用多进程
    workers= os.cpu_count() or 1,      # 跑满 CPU
    chunk  = 1200,
    block  = 8192,
    fp32   = False,
    margin = EPS
)

# Q3 策略参数（可按需调整/搜索）
UAV_SPEED_Q3 = 140.0
HEADING_OFFSET_DEG = 0.3
DROPS   = [0.46, 4.00, 5.80]
FUSE_DS = [3.55, 5.50, 6.00]

# ========== 工具函数 ==========
def heading_baseline_to_origin():
    v = Q1.FAKE_TARGET_ORIGIN - Q1.FY1_INIT
    return math.atan2(v[1], v[0])

def explosion_point_parametric(heading_rad, t_drop, fuse_delay, uav_speed, dtype=float):
    uav_pos, uav_v = uav_state_horizontal(t_drop, Q1.FY1_INIT.astype(dtype), dtype(uav_speed), heading_rad)
    expl_xy = uav_pos[:2] + uav_v[:2] * dtype(fuse_delay)
    expl_z  = uav_pos[2]  - dtype(0.5) * dtype(Q1.g) * (dtype(fuse_delay) ** 2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=dtype), uav_pos

def uav_state_horizontal(t, uav_init, uav_speed, heading_rad):
    vx = uav_speed * math.cos(heading_rad)
    vy = uav_speed * math.sin(heading_rad)
    return np.array([uav_init[0] + vx * t, uav_init[1] + vy * t, uav_init[2]], dtype=uav_init.dtype), \
           np.array([vx, vy, 0.0], dtype=uav_init.dtype)

def eval_chunk_union(args):
    (idx0, t_chunk, bombs, t_expl_list, pts, margin, block) = args
    n = len(t_chunk)
    out_union = np.zeros(n, dtype=bool)
    out_each  = [np.zeros(n, dtype=bool) for _ in bombs]

    for i, t in enumerate(t_chunk):
        m_pos, _ = Q1.MissileState(float(t), Q1.M1_INIT)
        any_cover = False
        for bi, (expl_pos, t_expl) in enumerate(zip(bombs, t_expl_list)):
            cz = expl_pos[2] - Q1.SMOG_SINK_SPEED * max(0.0, float(t) - t_expl)
            c  = np.array([expl_pos[0], expl_pos[1], cz], dtype=expl_pos.dtype)
            covered = Q1.ConeAllPointsIn(m_pos, c, pts, rCloud=Q1.SMOG_R, margin=margin, block=block)
            out_each[bi][i] = covered
            any_cover = any_cover or covered
        out_union[i] = any_cover
    return idx0, out_union, out_each

def seconds_from_mask(mask, dt): return float(np.count_nonzero(mask) * dt)

# ========== 主求解流程 ==========
def solve_q3(dt=DEFAULTS["dt"], nphi=DEFAULTS["nphi"], nz=DEFAULTS["nz"],
             backend=DEFAULTS["backend"], workers=DEFAULTS["workers"], chunk=DEFAULTS["chunk"],
             block=DEFAULTS["block"], fp32=DEFAULTS["fp32"], margin=DEFAULTS["margin"]):
    print("="*90)
    print("[Q3][INFO] 开始计算问题3（FY1投放3枚烟幕弹，总遮蔽时间最大化——给定方案评估）")
    print(f"[Q3][INFO] dt={dt}, nphi={nphi}, nz={nz}, backend={backend}, workers={workers}, chunk={chunk}, block={block}, fp32={fp32}")

    # 固定底层库线程数，避免超订阅（与 Q1 相同思路）
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    dtype = np.float32 if fp32 else np.float64

    heading0 = heading_baseline_to_origin()
    heading  = heading0 + math.radians(HEADING_OFFSET_DEG)
    print(f"[Q3][INFO] UAV_SPEED={UAV_SPEED_Q3} m/s, 航向角度={math.degrees(heading):.6f}° ({heading:.8f} rad)")

    print("[Q3][INFO] 计算三枚烟幕的投放/起爆点 ...")
    bombs_expl, uav_drops, t_expls = [], [], []
    for k, (td, fd) in enumerate(zip(DROPS, FUSE_DS), start=1):
        expl, drop_pos = explosion_point_parametric(heading, td, fd, UAV_SPEED_Q3, dtype=dtype)
        t_expl = td + fd
        bombs_expl.append(expl); uav_drops.append(drop_pos); t_expls.append(t_expl)
        print(f"    [Q3][INFO] 弹#{k} 投放t={td:.3f}s  起爆t={t_expl:.3f}s  投放=({drop_pos[0]:.3f},{drop_pos[1]:.3f},{drop_pos[2]:.3f})  起爆=({expl[0]:.3f},{expl[1]:.3f},{expl[2]:.3f})")

    hit_time = float(np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED)
    t0 = min(t_expls); t1 = min(max(t_expls) + Q1.SMOG_EFFECT_TIME, hit_time)
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=dtype)
    print(f"[Q3][INFO] 导弹命中假目标≈{hit_time:.3f}s, 评估区间=[{t0:.3f},{t1:.3f}]，步数={len(t_grid)}")

    print("[Q3][INFO] 预采样圆柱表面点 ...")
    pts = Q1.PreCalCylinderPoints(nphi, nz, dtype=dtype)
    print(f"[Q3][INFO] 圆柱采样点数={len(pts)}")

    chunks = [(i, t_grid[i:i+chunk], bombs_expl, t_expls, pts, margin, block) for i in range(0, len(t_grid), chunk)]
    mask_union = np.zeros_like(t_grid, dtype=bool)
    masks_each = [np.zeros_like(t_grid, dtype=bool) for _ in bombs_expl]

    print(f"[Q3][INFO] 多进程并行评估遮蔽（时间块） ...")
    tA = time.time()
    pool_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    with pool_cls(max_workers=workers) as pool:
        futs = {pool.submit(eval_chunk_union, args): args[0] for args in chunks}
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            _, m_union, m_each = fut.result()
            l = len(m_union)
            mask_union[idx:idx+l] = m_union
            for bi in range(len(masks_each)):
                masks_each[bi][idx:idx+l] = m_each[bi]
            done += 1
            if done % max(1, len(chunks)//20) == 0:
                print(f"    [Q3][INFO] 进度 {int(100*done/len(chunks))}%")
    tB = time.time()

    total_seconds = seconds_from_mask(mask_union, dt)
    secs_each     = [seconds_from_mask(m, dt) for m in masks_each]

    print(f"[Q3][INFO] 遮蔽评估完成, 用时 {tB - tA:.2f}s")
    for k, sec in enumerate(secs_each, start=1):
        print(f"[Q3][INFO] 弹#{k} 有效遮蔽 {sec:.6f} s")
    print(f"[Q3][INFO] 三弹并集总遮蔽时间 = {total_seconds:.6f} s")

    print("[Q3][INFO] 写入 result1.xlsx ...")
    heading_text = f"{heading:.6f} rad"
    rows = []
    for k in range(3):
        rows.append({
            "无人机运动方向": heading_text,
            "无人机运动速度 (m/s)": UAV_SPEED_Q3,
            "烟幕干扰弹编号": k+1,
            "烟幕干扰弹投放点的x坐标 (m)": float(uav_drops[k][0]),
            "烟幕干扰弹投放点的y坐标 (m)": float(uav_drops[k][1]),
            "烟幕干扰弹投放点的z坐标 (m)": float(uav_drops[k][2]),
            "烟幕干扰弹起爆点的x坐标 (m)": float(bombs_expl[k][0]),
            "烟幕干扰弹起爆点的y坐标 (m)": float(bombs_expl[k][1]),
            "烟幕干扰弹起爆点的z坐标 (m)": float(bombs_expl[k][2]),
            "有效干扰时长 (s)": secs_each[k],
        })
    df = pd.DataFrame(rows, columns=[
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
    ])
    df.to_excel("result1.xlsx", index=False, sheet_name="Sheet1")
    print("[Q3][INFO] 已保存 result1.xlsx")
    print("="*90)

def main():
    ap = argparse.ArgumentParser("Q3 — 三弹遮蔽方案评估 [多进程加速]")
    ap.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    ap.add_argument("--nphi", type=int, default=DEFAULTS["nphi"])
    ap.add_argument("--nz", type=int, default=DEFAULTS["nz"])
    ap.add_argument("--backend", choices=["process","thread"], default=DEFAULTS["backend"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    ap.add_argument("--chunk", type=int, default=DEFAULTS["chunk"])
    ap.add_argument("--block", type=int, default=DEFAULTS["block"])
    ap.add_argument("--margin", type=float, default=DEFAULTS["margin"])
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    solve_q3(dt=args.dt, nphi=args.nphi, nz=args.nz, backend=args.backend, workers=args.workers,
             chunk=args.chunk, block=args.block, fp32=args.fp32, margin=args.margin)

if __name__ == "__main__":
    main()
