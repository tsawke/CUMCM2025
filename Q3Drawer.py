# -*- coding: utf-8 -*-
"""
Q3Drawer.py — Animate & check Problem 3 answer (MP4 via imageio-ffmpeg, pure Python)

- Read result .xlsx (Chinese column names) produced by Solver
- Reconstruct UAV heading/speed and three bombs (drop & explosion points)
- Animate missile (to fake target origin), UAV FY1, bomb trajectories, smoke spheres sinking
- Physically consistent masking (default 'segment'):
    * segment–sphere for each cylinder sample (M→point)
    * union across clouds (OR), AND across cylinder points
    * ground clipping (cz<=0 ignored)
- Clean overlays on the LEFT: legend, cumulative covered timer
- Bottom monitors: three bomb body positions, three cloud centers (live)
- Live progress logs and per-frame append via imageio-ffmpeg internal binary

Requires: numpy, pandas, matplotlib, imageio, imageio-ffmpeg, openpyxl
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
import imageio_ffmpeg

# Geometry/physics from Q1
import Q1Solver_visual as Q1

EPS = 1e-12

def info(msg: str):
    print(f"[CHECK][INFO] {msg}", flush=True)

# ---------- missile / uav kinematics ----------
def missile_pos_at(t: float, p0: np.ndarray, pT: np.ndarray, speed: float) -> np.ndarray:
    vec = pT - p0
    L = float(np.linalg.norm(vec))
    if L < 1e-9:
        return pT.copy()
    u = vec / L
    hit_t = L / speed
    if t >= hit_t:
        return pT.copy()
    return p0 + u * (speed * t)

def uav_pos_vel_at(t: float, uav_init: np.ndarray, speed: float, heading_rad: float):
    vx = speed * math.cos(heading_rad)
    vy = speed * math.sin(heading_rad)
    pos = np.array([uav_init[0] + vx * t, uav_init[1] + vy * t, uav_init[2]], dtype=float)
    vel = np.array([vx, vy, 0.0], dtype=float)
    return pos, vel

# ---------- cylinder sampling ----------
def precalc_cylinder_points(nphi: int, nz: int, dtype=float):
    # True target cylinder: radius=7m, height=10m; bottom center at (0,200,0)
    R = 7.0; H = 10.0
    center = np.array([0.0, 200.0, 0.0], dtype=dtype)
    phis = np.linspace(0, 2*np.pi, num=nphi, endpoint=False, dtype=dtype)
    zs = np.linspace(0, H, num=nz, dtype=dtype)
    pts = []
    for z in zs:
        ring = center + np.stack([R*np.cos(phis), np.zeros_like(phis), np.full_like(phis, z)], axis=1)
        pts.append(ring)
    pts = np.concatenate(pts, axis=0)
    # add top/bottom centers（更严格）
    pts = np.vstack([pts, center + np.array([0,0,0],dtype=dtype), center + np.array([0,0,H],dtype=dtype)])
    return pts  # (nphi*nz+2, 3)

# ---------- masking ----------
def segment_points_mask(m_pos: np.ndarray, pts: np.ndarray, c: np.ndarray, R: float) -> np.ndarray:
    """Vectorized segment-sphere intersection, plus endpoint-inside."""
    d = pts - m_pos[None, :]
    f = m_pos - c
    a = np.sum(d*d, axis=1)
    b = 2.0 * np.sum(f[None,:]*d, axis=1)
    csc = float(np.dot(f,f) - R*R)
    disc = b*b - 4.0*a*csc
    good = (a > 0.0) & (disc >= 0.0)
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[good] = np.sqrt(np.maximum(disc[good], 0.0))
    t1 = np.empty_like(disc); t2 = np.empty_like(disc)
    t1[good] = (-b[good] - sqrt_disc[good])/(2.0*a[good])
    t2[good] = (-b[good] + sqrt_disc[good])/(2.0*a[good])
    inter = good & ( ((t1>=0.0)&(t1<=1.0)) | ((t2>=0.0)&(t2<=1.0)) )
    inside_P = (np.sum((pts - c[None,:])**2, axis=1) <= R*R + 1e-12)
    inside_M = (np.dot(m_pos - c, m_pos - c) <= R*R + 1e-12)
    return inter | inside_P | inside_M

def cone_points_in(m_pos: np.ndarray, c: np.ndarray, pts: np.ndarray, R: float) -> np.ndarray:
    # Infinite cone through sphere tangent（兼容模式）
    a = c - m_pos
    d2 = float(np.dot(a,a))
    b = pts - m_pos[None,:]
    dot = b @ a
    fwd = dot >= 0.0
    if d2 <= R*R:
        return fwd
    rhs = (np.sum(b*b, axis=1)) * (d2 - R*R)
    lhs = dot*dot
    return fwd & (lhs >= rhs)

# ---------- read solver output ----------
def read_solution(xlsx_path: str):
    df = pd.read_excel(xlsx_path, sheet_name=0)
    need = [
        "无人机运动方向", "无人机运动速度 (m/s)", "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)","烟幕干扰弹投放点的y坐标 (m)","烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)","烟幕干扰弹起爆点的y坐标 (m)","烟幕干扰弹起爆点的z坐标 (m)",
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")
    rows = [df.iloc[i] for i in range(3)]
    heading_text = str(rows[0]["无人机运动方向"])
    if "rad" in heading_text:
        heading_rad = float(heading_text.replace("rad","").strip())
    else:
        heading_rad = float(rows[0]["无人机运动方向"])
    speed = float(rows[0]["无人机运动速度 (m/s)"])
    drops, expls = [], []
    for r in rows:
        drops.append(np.array([float(r["烟幕干扰弹投放点的x坐标 (m)"]),
                               float(r["烟幕干扰弹投放点的y坐标 (m)"]),
                               float(r["烟幕干扰弹投放点的z坐标 (m)"])], dtype=float))
        expls.append(np.array([float(r["烟幕干扰弹起爆点的x坐标 (m)"]),
                               float(r["烟幕干扰弹起爆点的y坐标 (m)"]),
                               float(r["烟幕干扰弹起爆点的z坐标 (m)"])], dtype=float))
    return heading_rad, speed, drops, expls

# ---------- reconstruct fuse & drop times from points ----------
def recover_times(heading_rad: float, speed: float, drops, expls):
    vx = speed * math.cos(heading_rad); vy = speed * math.sin(heading_rad)
    vxy = np.array([vx,vy], dtype=float)
    fuses, tdrops, texpls = [], [], []
    for D, E in zip(drops, expls):
        dz = float(D[2] - E[2])
        fuse = math.sqrt(max(0.0, 2.0*dz/Q1.g))
        denom = float(np.dot(vxy, vxy)) + 1e-12
        t_drop = float(np.dot(D[:2] - Q1.FY1_INIT[:2], vxy) / denom)
        fuses.append(fuse)
        tdrops.append(max(0.0, t_drop))
        texpls.append(tdrops[-1] + fuse)
    return tdrops, fuses, texpls

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Q3 drawer (MP4 via internal ffmpeg)")
    ap.add_argument("--xlsx", type=str, default="result1.xlsx", help="solver output xlsx")
    ap.add_argument("--out", type=str, default="q3_anim.mp4")
    ap.add_argument("--anim_seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--sphere_res", type=int, default=16, help="sphere mesh resolution (higher=slower)")
    ap.add_argument("--skip", type=int, default=1, help="frame thinning (>=1)")
    ap.add_argument("--masking_mode", choices=["segment","cone"], default="segment")
    ap.add_argument("--label_fontsize", type=int, default=9)
    args = ap.parse_args()

    # internal ffmpeg binary
    ffmpexe = imageio_ffmpeg.get_ffmpeg_exe()
    info(f"Using internal ffmpeg: {ffmpexe}")

    # read xlsx
    info(f"Reading {args.xlsx} …")
    heading_rad, speed, drops, expls = read_solution(args.xlsx)

    # cylinder samples (use same method as Q3Solver_v2)
    pts_mask = Q1.PreCalCylinderPoints(360, 7, dtype=float)

    # timeline
    fps = args.fps
    T = float(args.anim_seconds)
    N = int(round(T * fps)) + 1
    ts = np.linspace(0.0, T, N, dtype=float)
    info(f"Animation length = {T:.3f} s @ {fps} fps ({len(ts)} frames)")
    info(f"Masking mode = {args.masking_mode}")

    # constants
    FY1 = Q1.FY1_INIT.astype(float)
    M1 = Q1.M1_INIT.astype(float)
    FAKE = Q1.FAKE_TARGET_ORIGIN.astype(float)  # missile always toward (0,0,0)
    MISSILE_SPEED = float(Q1.MISSILE_SPEED)
    R = float(Q1.SMOG_R)
    SINK = float(Q1.SMOG_SINK_SPEED)

    # kinematics arrays
    vx = speed * math.cos(heading_rad); vy = speed * math.sin(heading_rad)
    m_path = np.vstack([missile_pos_at(t, M1, FAKE, MISSILE_SPEED) for t in ts])
    fy_path = np.vstack([np.array([FY1[0] + vx*t, FY1[1] + vy*t, FY1[2]], dtype=float) for t in ts])

    # recovered times for labels（也用于 bomb body 轨迹）
    tdrops, fuses, texpls = recover_times(heading_rad, speed, drops, expls)

    # figure/axes
    fig = plt.figure(figsize=(10, 6), dpi=args.dpi)
    ax = fig.add_subplot(1,1,1, projection='3d')

    # bounds
    xmin = min(FY1[0], M1[0], 0.0) - 200.0
    xmax = max(FY1[0], M1[0], 0.0) + 200.0
    ymin = min(-800.0, FY1[1]-500.0, M1[1]-500.0)
    ymax = max( 800.0, FY1[1]+500.0, M1[1]+500.0)
    zmin, zmax = 0.0, max(FY1[2], M1[2], 2100.0) + 200.0
    ax.set_xlim3d([xmin, xmax]); ax.set_ylim3d([ymin, ymax]); ax.set_zlim3d([zmin, zmax])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Problem 3 — Smoke Interference (MP4)")

    # draw cylinder wire
    cyl_phi = np.linspace(0, 2*np.pi, 120)
    R_cyl, H_cyl = 7.0, 10.0
    cx, cy = 0.0, 200.0
    for z in (0.0, 10.0):
        ax.plot(cx + R_cyl*np.cos(cyl_phi), cy + 0*cyl_phi, z + 0*cyl_phi, color="k", lw=1, alpha=0.6)
    for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = cx + R_cyl*np.cos(ang); y = cy
        ax.plot([x, x], [y, y], [0.0, 10.0], color="k", lw=0.5, alpha=0.3)

    # artists
    mline, = ax.plot([], [], [], color="tab:red", lw=2)
    mpt,   = ax.plot([], [], [], "o", color="tab:red", ms=4)
    fyline,= ax.plot([], [], [], color="tab:blue", lw=1.5)
    fypt,  = ax.plot([], [], [], "o", color="tab:blue", ms=4)

    bomb_colors = ["tab:green","tab:orange","tab:purple"]
    bomb_pts = [ax.plot([], [], [], "o", color=bomb_colors[i], ms=4)[0] for i in range(3)]
    cloud_pts= [ax.plot([], [], [], "o", color=bomb_colors[i], ms=4, alpha=0.6)[0] for i in range(3)]

    # ---- LEFT overlays (legend + timer) via figure.text (avoid overlap) ----
    fsz = args.label_fontsize
    fig.text(0.015, 0.96,
        "Legend:\n"
        "  red  : M1 & path\n"
        "  blue : FY1 & path\n"
        "  green/orange/purple : Bomb bodies & cloud centers",
        ha="left", va="top", fontsize=fsz,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    cover_timer = fig.text(0.015, 0.86, "Covered time: 0.000 s",
        ha="left", va="top", fontsize=fsz+1,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    # ---- BOTTOM monitors（不再从 Line3D 读坐标；改为用缓存变量） ----
    bomb_monitor = fig.text(0.015, 0.12, "Bomb bodies:\n", ha="left", va="bottom", fontsize=fsz,
                            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    cloud_monitor= fig.text(0.015, 0.05, "Cloud centers:\n", ha="left", va="bottom", fontsize=fsz,
                            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    # coverage bookkeeping
    covered_mask = np.zeros(len(ts), dtype=bool)

    # unit sphere mesh
    def unit_sphere(res):
        u = np.linspace(0, 2*np.pi, res, endpoint=True)
        v = np.linspace(0, np.pi, res//2+1, endpoint=True)
        x = np.outer(np.cos(u), np.sin(v)).T
        y = np.outer(np.sin(u), np.sin(v)).T
        z = np.outer(np.ones_like(u), np.cos(v)).T
        return x, y, z
    sph_x, sph_y, sph_z = unit_sphere(max(8, args.sphere_res))

    # wireframe lifecycle
    active_wires = []
    def clear_wires():
        nonlocal active_wires
        for w in active_wires:
            try:
                w.remove()
            except Exception:
                pass
        active_wires = []

    # cloud center evolution
    def cloud_center_at(expl, te, tnow):
        if tnow < te:
            return None
        cz = float(expl[2]) - SINK * (tnow - te)
        if cz <= 0.0 + 1e-9:
            return None
        return np.array([expl[0], expl[1], cz], dtype=float)

    # helper
    def fmt_xyz(p):
        return f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"

    # prepare writer
    fig.canvas.draw()
    # 获取画布尺寸（兼容新旧matplotlib版本）
    try:
        w, h = fig.canvas.get_width_height()
    except AttributeError:
        # 新版本matplotlib的方法
        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)
    
    writer = imageio.get_writer(
        args.out, format="FFMPEG", mode="I",
        fps=args.fps, codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=None,
        ffmpeg_params=['-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    )

    # ---- 位置缓存（替代读取图元）----
    last_body_pos = [None, None, None]
    last_cloud_pos= [None, None, None]

    try:
        wire_every = max(10, args.fps // 2)

        for fi, t in enumerate(ts[::max(1, args.skip)]):
            i = fi * max(1, args.skip)

            # missile & uav
            mp = m_path[i]; fyp = fy_path[i]
            mpt.set_data([mp[0]],[mp[1]]); mpt.set_3d_properties([mp[2]])
            fypt.set_data([fyp[0]],[fyp[1]]); fypt.set_3d_properties([fyp[2]])
            if i == 0:
                mline.set_data([mp[0]],[mp[1]]); mline.set_3d_properties([mp[2]])
                fyline.set_data([fyp[0]],[fyp[1]]); fyline.set_3d_properties([fyp[2]])
            else:
                mline.set_data(m_path[:i+1,0], m_path[:i+1,1]); mline.set_3d_properties(m_path[:i+1,2])
                fyline.set_data(fy_path[:i+1,0], fy_path[:i+1,1]); fyline.set_3d_properties(fy_path[:i+1,2])

            # bombs body (pre-explosion)
            for bi in range(3):
                D = drops[bi]; E = expls[bi]
                # 使用预计算的时间参数
                t_drop = tdrops[bi]
                fuse = fuses[bi]
                t_expl = texpls[bi]
                if t <= t_expl + 1e-9:
                    tau = max(0.0, t - t_drop)
                    pos = np.array([D[0] + vx*tau, D[1] + vy*tau, D[2] - 0.5*Q1.g*(tau**2)], dtype=float)
                    bomb_pts[bi].set_data([pos[0]],[pos[1]]); bomb_pts[bi].set_3d_properties([pos[2]])
                    last_body_pos[bi] = pos
                else:
                    bomb_pts[bi].set_data([],[]); bomb_pts[bi].set_3d_properties([])
                    last_body_pos[bi] = None

            # clouds & masking
            any_mask_pts = None
            if (i % wire_every) == 0:
                clear_wires()

            for bi in range(3):
                D = drops[bi]; E = expls[bi]
                # 使用预计算的时间参数，而不是实时重新计算
                t_drop = tdrops[bi]
                fuse = fuses[bi] 
                t_expl = texpls[bi]
                c = cloud_center_at(E, t_expl, t)
                last_cloud_pos[bi] = c
                if c is None:
                    cloud_pts[bi].set_data([],[]); cloud_pts[bi].set_3d_properties([])
                    continue
                cloud_pts[bi].set_data([c[0]],[c[1]]); cloud_pts[bi].set_3d_properties([c[2]])

                # cooperative masking
                if args.masking_mode == "segment":
                    mask_in = segment_points_mask(mp, pts_mask, c, R)
                else:
                    mask_in = cone_points_in(mp, c, pts_mask, R)
                any_mask_pts = mask_in if any_mask_pts is None else (any_mask_pts | mask_in)

                # sparse wireframe
                if (i % wire_every) == 0:
                    xs = R * sph_x + c[0]
                    ys = R * sph_y + c[1]
                    zs = R * sph_z + c[2]
                    wf = ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2,
                                           color=bomb_colors[bi], alpha=0.18, linewidth=0.35)
                    active_wires.append(wf)

            covered = (any_mask_pts is not None) and bool(np.all(any_mask_pts))
            if covered:
                covered_mask[i] = True

            # update overlays
            tot_cov = float(np.count_nonzero(covered_mask) / fps)
            cover_timer.set_text(f"Covered time: {tot_cov:.3f} s")

            # bottom monitors（使用缓存值）
            bodies_txt = []
            for bi in range(3):
                p = last_body_pos[bi]
                bodies_txt.append(f"B{bi+1}: {fmt_xyz(p)}" if p is not None else f"B{bi+1}: (-)")
            bomb_monitor.set_text("Bomb bodies:\n" + " | ".join(bodies_txt))

            clouds_txt = []
            for bi in range(3):
                c = last_cloud_pos[bi]
                clouds_txt.append(f"C{bi+1}: {fmt_xyz(c)}" if c is not None else f"C{bi+1}: (-)")
            cloud_monitor.set_text("Cloud centers:\n" + " | ".join(clouds_txt))

            # draw & write
            fig.canvas.draw()
            # 使用新的matplotlib API获取RGB数据
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            # 转换RGBA到RGB（丢弃alpha通道）
            frame = frame[:, :, :3]
            writer.append_data(frame)

            if (fi % 25) == 0 or fi == (len(ts[::max(1,args.skip)])-1):
                info(f"frame {fi+1}/{len(ts[::max(1,args.skip)])}")

        info("Finalizing writer …")
    finally:
        writer.close()
        plt.close(fig)

    total_sec = float(np.count_nonzero(covered_mask) / fps)
    info(f"Done. Covered total time ≈ {total_sec:.3f} s")
    info(f"Saved MP4 → {args.out}")

if __name__ == "__main__":
    main()
