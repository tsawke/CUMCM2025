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

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)
FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype=float)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)

DROP_TIME = 1.5
FUSE_DELAY_TIME = 3.6
EXPLODE_TIME = DROP_TIME + FUSE_DELAY_TIME

EPS = 1e-12

def unit(v): 
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def missile_state(t, M_init):
    dir_to_origin = unit(FAKE_TARGET_ORIGIN - M_init)
    v = MISSILE_SPEED * dir_to_origin
    return M_init + v * t, v

def uav_state_horizontal(t, uav_init, uav_speed, heading_rad):
    vx = uav_speed * math.cos(heading_rad)
    vy = uav_speed * math.sin(heading_rad)
    return np.array([uav_init[0] + vx * t, uav_init[1] + vy * t, uav_init[2]], dtype=uav_init.dtype), np.array([vx, vy, 0.0], dtype=uav_init.dtype)

def precompute_cylinder_points(n_phi, n_z_side, dtype=np.float64):
    B = CYLINDER_BASE_CENTER.astype(dtype)
    R, H = dtype(CYLINDER_R), dtype(CYLINDER_H)
    phis = np.linspace(0.0, 2.0*math.pi, n_phi, endpoint=False, dtype=dtype)
    c, s = np.cos(phis), np.sin(phis)
    ring = np.stack([R*c, R*s, np.zeros_like(c)], axis=1).astype(dtype)   # [n_phi,3]
    pts = [B + ring, B + np.array([0.0,0.0,H], dtype=dtype) + ring]
    if n_z_side >= 2:
        for z in np.linspace(0.0, H, n_z_side, dtype=dtype):
            pts.append(B + np.array([0.0,0.0,z], dtype=dtype) + ring)
    P = np.vstack(pts).astype(dtype)
    return P

def explosion_point(heading_rad, t_drop, fuse_DELAY_TIME, dtype=np.float64):
    drop_pos, uav_v = uav_state_horizontal(t_drop, FY1_INIT.astype(dtype), dtype(UAV_SPEED), heading_rad)
    expl_xy = drop_pos[:2] + uav_v[:2] * dtype(fuse_DELAY_TIME)
    expl_z = drop_pos[2] - dtype(0.5)*dtype(g)*(dtype(fuse_DELAY_TIME)**2)
    return np.array([expl_xy[0], expl_xy[1], expl_z], dtype=dtype)

# —— 快速早停版：按点块判定，无需整批向量化到底 —— #
def _cone_all_points_in(M, C, P, r_cloud=SMOG_R, margin=EPS, block=8192):
    v = C - M
    L = np.linalg.norm(v)
    if L <= EPS or r_cloud >= L:
        return True
    cos_alpha = math.sqrt(max(0.0, 1.0 - (r_cloud/L)**2))
    # 分块：任一块失败直接 False
    for i in range(0, len(P), block):
        W = P[i:i+block] - M
        Wn = np.linalg.norm(W, axis=1) + EPS
        # 比较： (W·v)/(||W|| L) >= cos_alpha  ->  W·v >= ||W|| L cos_alpha
        lhs = W @ v
        rhs = Wn * L * cos_alpha
        if not np.all(lhs + margin >= rhs):
            return False
    return True

def _eval_chunk(args):
    """子进程：计算一个时间段的遮蔽布尔数组，带块索引确保回写顺序正确"""
    idx0, t_chunk, expl_pos, t_expl, pts, margin, block = args
    out = np.zeros_like(t_chunk, dtype=bool)
    for i, t in enumerate(t_chunk):
        M, _ = missile_state(float(t), M1_INIT)
        C = np.array([expl_pos[0], expl_pos[1], expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t_expl)], dtype=expl_pos.dtype)
        out[i] = _cone_all_points_in(M, C, pts, r_cloud=SMOG_R, margin=margin, block=block)
    return idx0, out

def compute_q1_extreme(
    dt=0.0005, nphi=960, nz=13,
    backend="process", workers=None, chunk=800,
    fp32=False, margin=EPS, block=8192
):
    # 限制 BLAS 线程避免过度抢核（多进程时很关键）
    if backend == "process":
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    dtype = np.float32 if fp32 else np.float64
    heading = math.atan2(-FY1_INIT[1], -FY1_INIT[0])
    expl_pos = explosion_point(heading, DROP_TIME, FUSE_DELAY_TIME, dtype=dtype)
    if expl_pos[2] <= 0:
        return 0.0, expl_pos, float(np.linalg.norm(M1_INIT-FAKE_TARGET_ORIGIN)/MISSILE_SPEED)

    hit_time = float(np.linalg.norm(M1_INIT-FAKE_TARGET_ORIGIN)/MISSILE_SPEED)
    t0, t1 = EXPLODE_TIME, min(EXPLODE_TIME + SMOG_EFFECT_TIME, hit_time)
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=dtype)

    pts = precompute_cylinder_points(nphi, nz, dtype=dtype)

    # 分块与并行
    chunks = [(i, t_grid[i:i+chunk]) for i in range(0, len(t_grid), chunk)]
    mask_total = np.zeros_like(t_grid, dtype=bool)

    print(f"[Q1] steps={len(t_grid)}, dt={dt}, range=[{t0:.6f},{t1:.6f}]  pts={len(pts)}  dtype={dtype}")
    print(f"[Q1] backend={backend}, workers={workers or 'auto'}, chunk={chunk}, block={block}, margin={margin:g}")

    tA = time.time()
    if backend == "thread":
        pool_cls = ThreadPoolExecutor
    else:
        pool_cls = ProcessPoolExecutor

    with pool_cls(max_workers=workers) as pool:
        futs = {pool.submit(_eval_chunk, (idx, c, expl_pos, EXPLODE_TIME, pts, margin, block)): idx for idx, c in chunks}
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            _, m = fut.result()
            L = len(m)
            mask_total[idx:idx+L] = m
            done += 1
            if done % max(1, len(chunks)//10) == 0:
                print(f"    [Q1] 进度 {int(100*done/len(chunks))}%")

    seconds = float(np.count_nonzero(mask_total) * dt)
    tB = time.time()
    print(f"[Q1] 遮蔽时长（严格圆锥判据）= {seconds:.6f} s  | 用时 {tB - tA:.2f}s")
    return seconds, expl_pos, hit_time

def main():
    ap = argparse.ArgumentParser("Q1 极限参数版")
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--nphi", type=int, default=960)
    ap.add_argument("--nz", type=int, default=13)
    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--block", type=int, default=8192, help="每次判定的点块大小（越大越快但内存更高）")
    ap.add_argument("--margin", type=float, default=EPS, help="圆锥判据余量（防止边界抖动）")
    ap.add_argument("--fp32", action="store_true", help="用 float32 加速（误差通常 <1e-3s）")
    args = ap.parse_args()

    print("="*70)
    print("FY1→M1 高精度遮蔽计算（严格圆锥判据｜超参数 & 快速早停 & 并行）")
    print("="*70)
    print(f"dt={args.dt}, NPHI={args.nphi}, NZ={args.nz}, backend={args.backend}, workers={args.workers or 'auto'}, chunk={args.chunk}, block={args.block}, fp32={args.fp32}")

    sec, expl, hit = compute_q1_extreme(
        dt=args.dt, nphi=args.nphi, nz=args.nz,
        backend=args.backend, workers=args.workers, chunk=args.chunk,
        fp32=args.fp32, margin=args.margin, block=args.block
    )
    print("-"*70)
    print(f"[结果] 遮蔽时长 = {sec:.9f} s")
    print(f"[信息] 起爆点 = ({expl[0]:.6f}, {expl[1]:.6f}, {expl[2]:.6f})")
    print(f"[信息] 导弹命中原点时间 ≈ {hit:.6f} s")

if __name__ == "__main__":
    main()
