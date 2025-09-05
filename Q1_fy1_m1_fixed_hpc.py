# -*- coding: utf-8 -*-
# Q1_fy1_m1_fixed_hpc.py
# 高精度 + 并行/可选GPU 版本
# 计算 FY1 单次投放对 M1 的遮蔽时长（严格圆锥判据：圆柱整体 ∈ 无限圆锥）

import math
import sys
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------- 固定场景参数 ----------------------
g = 9.8
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
CLOUD_R = 10.0
CLOUD_SINK = 3.0
CLOUD_EFFECT = 20.0

CYL_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYL_R = 7.0
CYL_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)
FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype=float)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)

DROP_T = 1.5
FUSE_DT = 3.6
EXPL_T = DROP_T + FUSE_DT

# ---------------------- 基础工具 ----------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def missile_state(t: float, M_init: np.ndarray):
    # 导弹直线匀速，指向原点
    dir_to_origin = unit(FAKE_TARGET_ORIGIN - M_init)
    v = MISSILE_SPEED * dir_to_origin
    return M_init + v * t, v

def uav_state_horizontal(t: float, uav_init: np.ndarray, uav_speed: float, heading_rad: float):
    vx = uav_speed * math.cos(heading_rad)
    vy = uav_speed * math.sin(heading_rad)
    x = uav_init[0] + vx * t
    y = uav_init[1] + vy * t
    z = uav_init[2]
    return np.array([x, y, z]), np.array([vx, vy, 0.0])

def precompute_cylinder_points(n_phi: int, n_z_side: int) -> np.ndarray:
    """预生成圆柱表面采样点（底/顶圆周 + 侧面若干圈），世界坐标"""
    B = CYL_BASE_CENTER
    R = CYL_R
    H = CYL_H
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    pts = []
    # 底/顶圆周
    cosv, sinv = np.cos(phis), np.sin(phis)
    ring = np.stack([R*cosv, R*sinv, np.zeros_like(cosv)], axis=1)  # [n_phi, 3]
    pts.append(B + ring)
    pts.append(B + np.array([0.0, 0.0, H]) + ring)
    # 侧面圈
    if n_z_side >= 2:
        zs = np.linspace(0.0, H, n_z_side)
        for z in zs:
            center = B + np.array([0.0, 0.0, z])
            pts.append(center + ring)
    P = np.vstack(pts).astype(float)  # [Npts, 3]
    return P

def cylinder_inside_infinite_cone_vectorized(M: np.ndarray, C: np.ndarray,
                                             P: np.ndarray, r_cloud: float = CLOUD_R) -> bool:
    """
    向量化严格判据：全部采样点 P 都满足落入以 M 为锥顶、轴向 (C-M)、半顶角 asin(r/L) 的无限圆锥。
    """
    v = C - M
    L = np.linalg.norm(v)
    if L <= 1e-9 or r_cloud >= L:
        return True
    cos_alpha = math.sqrt(max(0.0, 1.0 - (r_cloud/L)**2))
    W = P - M            # [N,3]
    Wn = np.linalg.norm(W, axis=1) + 1e-12
    cos_theta = (W @ v) / (Wn * L)
    return bool(np.all(cos_theta >= (cos_alpha - 1e-12)))

def explosion_point(heading_rad: float, t_drop: float, fuse_dt: float) -> np.ndarray:
    drop_pos, uav_v = uav_state_horizontal(t_drop, FY1_INIT, UAV_SPEED, heading_rad)
    expl_xy = drop_pos[:2] + uav_v[:2] * fuse_dt
    expl_z = drop_pos[2] - 0.5 * g * (fuse_dt ** 2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=float)

def missile_hit_time(M_init: np.ndarray) -> float:
    return float(np.linalg.norm(M_init - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)

# ---------------------- 分块并行计算 ----------------------
def _eval_chunk(t_chunk: np.ndarray, expl_pos: np.ndarray, t_expl: float,
                pts: np.ndarray) -> np.ndarray:
    """
    计算一个时间分块内的遮蔽布尔数组（CPU子进程内执行）。
    """
    out = np.zeros_like(t_chunk, dtype=bool)
    for i, t in enumerate(t_chunk):
        M, _ = missile_state(t, M1_INIT)
        C = np.array([expl_pos[0], expl_pos[1], expl_pos[2] - CLOUD_SINK * max(0.0, t - t_expl)])
        out[i] = cylinder_inside_infinite_cone_vectorized(M, C, pts, CLOUD_R)
    return out

# 可选：GPU路径（需 cupy）
def _try_import_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None

def _gpu_mask(t_grid, expl_pos, t_expl, pts_np):
    """
    GPU 单进程实现（cupy），将每个时刻独立判断。
    说明：这里按时间循环，但每次判定在 GPU 上完成（大规模点乘/范数均在 GPU）。
    """
    cp = _try_import_cupy()
    if cp is None:
        raise RuntimeError("cupy 未安装，无法使用 GPU 路径。")

    pts = cp.asarray(pts_np)  # [N,3] 常驻显存
    mask = np.zeros_like(t_grid, dtype=bool)
    for i, t in enumerate(t_grid):
        M, _ = missile_state(float(t), M1_INIT)
        # 转到GPU
        M_gpu = cp.asarray(M)
        C_gpu = cp.asarray([expl_pos[0], expl_pos[1], expl_pos[2] - CLOUD_SINK * max(0.0, float(t) - t_expl)])
        v = C_gpu - M_gpu
        L = cp.linalg.norm(v)
        # 近似或在云内：直接 True
        if float(L) <= 1e-9 or CLOUD_R >= float(L):
            mask[i] = True
            continue
        cos_alpha = cp.sqrt(cp.maximum(0.0, 1.0 - (CLOUD_R/ L)**2))
        W = pts - M_gpu  # [N,3]
        Wn = cp.linalg.norm(W, axis=1) + 1e-12
        cos_theta = (W @ v) / (Wn * L)
        ok = bool(cp.all(cos_theta >= (cos_alpha - 1e-12)))
        mask[i] = ok
    return mask

# ---------------------- 主求解函数 ----------------------
def compute_q1_shield_time(
    dt: float = 0.002,
    n_phi: int = 360,
    n_z_side: int = 9,
    chunk_size: int = 200,     # 每个子进程处理的时间步个数
    workers: int | None = None,
    backend: str = "process",  # "process" 或 "thread"
    use_gpu: bool = False
):
    """
    返回： shield_seconds, expl_pos, hit_time
    """
    # 航向朝向原点
    heading = math.atan2(0.0 - FY1_INIT[1], 0.0 - FY1_INIT[0])

    # 起爆点
    expl_pos = explosion_point(heading, DROP_T, FUSE_DT)
    if expl_pos[2] <= 0:
        print("[WARN] 起爆高度 <= 0，方案无效。")
        return 0.0, expl_pos, missile_hit_time(M1_INIT)

    # 时间轴
    hit_time = missile_hit_time(M1_INIT)
    t0 = EXPL_T
    t1 = min(EXPL_T + CLOUD_EFFECT, hit_time)
    if t0 >= t1:
        return 0.0, expl_pos, hit_time

    t_grid = np.arange(t0, t1 + 1e-12, dt)
    print(f"[Q1] t in [{t0:.3f}, {t1:.3f}]s, dt={dt}, steps={len(t_grid)}")
    print(f"[Q1] 采样：NPHI={n_phi}, NZ={n_z_side}  -> 圆柱表面点数约 {(2 + max(0,n_z_side)) * n_phi}")

    # 预生成圆柱点
    pts = precompute_cylinder_points(n_phi, n_z_side)

    # GPU 路径
    if use_gpu:
        cp = _try_import_cupy()
        if cp is None:
            print("[Q1] 未检测到 cupy，回退到 CPU。")
        else:
            tA = time.time()
            mask = _gpu_mask(t_grid, expl_pos, EXPL_T, pts)
            seconds = float(np.count_nonzero(mask) * dt)
            tB = time.time()
            print(f"[Q1][GPU] 遮蔽时长 = {seconds:.6f} s   （GPU用时 {tB - tA:.2f}s）")
            return seconds, expl_pos, hit_time

    # CPU 并行路径
    # 分块
    chunks = [t_grid[i:i+chunk_size] for i in range(0, len(t_grid), chunk_size)]
    print(f"[Q1] 并行分块数：{len(chunks)}，每块步数≈{chunk_size}，backend={backend}, workers={workers or 'auto'}")

    mask_total = np.zeros_like(t_grid, dtype=bool)
    tA = time.time()
    if backend == "thread":
        # 线程池（若 numpy 内核释放 GIL，线程也能加速；否则建议用 process）
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_eval_chunk, c, expl_pos, EXPL_T, pts) for c in chunks]
            done = 0
            offset = 0
            for fut in as_completed(futs):
                res = fut.result()
                # 将该块结果放入总掩码（按块顺序写回）
                L = len(res)
                mask_total[offset: offset+L] = res
                offset += L
                done += 1
                if done % max(1, len(chunks)//10) == 0:
                    print(f"    [Q1][thread] 进度 {int(100*done/len(chunks))}%")
    else:
        # 进程池（推荐）
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_eval_chunk, c, expl_pos, EXPL_T, pts) for c in chunks]
            done = 0
            offset = 0
            for fut in as_completed(futs):
                res = fut.result()
                L = len(res)
                mask_total[offset: offset+L] = res
                offset += L
                done += 1
                if done % max(1, len(chunks)//10) == 0:
                    print(f"    [Q1][proc] 进度 {int(100*done/len(chunks))}%")

    seconds = float(np.count_nonzero(mask_total) * dt)
    tB = time.time()
    print(f"[Q1] 遮蔽时长（严格圆锥判据）= {seconds:.6f} s   （CPU用时 {tB - tA:.2f}s）")
    return seconds, expl_pos, hit_time

# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser(description="Q1 高精度 + 并行/可选GPU 计算")
    parser.add_argument("--dt", type=float, default=0.002, help="时间步长（默认 0.002s）")
    parser.add_argument("--nphi", type=int, default=360, help="圆周采样数（默认 360）")
    parser.add_argument("--nz", type=int, default=9, help="侧面圈数（默认 9）")
    parser.add_argument("--chunk", type=int, default=200, help="每块时间步个数（默认 200）")
    parser.add_argument("--workers", type=int, default=None, help="并行进程/线程数（默认 CPU 全核）")
    parser.add_argument("--backend", choices=["process","thread"], default="process", help="并行后端（默认 process）")
    parser.add_argument("--use-gpu", action="store_true", help="若安装了 cupy，则用GPU路径（单进程）")
    args = parser.parse_args()

    print("="*72)
    print("问题1：FY1 以 120 m/s 朝原点飞行；1.5s 投弹，3.6s 后起爆 —— 高精度并行求解")
    print("="*72)
    print(f"参数：dt={args.dt}, NPHI={args.nphi}, NZ={args.nz}, chunk={args.chunk}, backend={args.backend}, workers={args.workers or 'auto'}, use_gpu={args.use_gpu}")
    print("-"*72)

    shield_time, expl_pos, hit_time = compute_q1_shield_time(
        dt=args.dt,
        n_phi=args.nphi,
        n_z_side=args.nz,
        chunk_size=args.chunk,
        workers=args.workers,
        backend=args.backend,
        use_gpu=args.use_gpu
    )
    print("-"*72)
    print(f"[结果] 遮蔽时长 = {shield_time:.6f} s")
    print(f"[信息] 起爆点 = ({expl_pos[0]:.3f}, {expl_pos[1]:.3f}, {expl_pos[2]:.3f})")
    print(f"[信息] 导弹命中原点时间 ≈ {hit_time:.6f} s")

if __name__ == "__main__":
    main()
