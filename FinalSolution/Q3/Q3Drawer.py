import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import imageio_ffmpeg

import Q1Solver_visual as Q1

EPS = 1e-12

def Info(msg: str):
    print(f"[Q3] {msg}", flush = True)

def MissilePosAt(t: float, p0: np.ndarray, pT: np.ndarray, speed: float) -> np.ndarray:
    vec = pT - p0
    L = float(np.linalg.norm(vec))
    if L < 1e-9:
        return pT.copy()
    u = vec / L
    hitT = L / speed
    if t >= hitT:
        return pT.copy()
    return p0 + u * (speed * t)

def UavPosVelAt(t: float, uavInit: np.ndarray, speed: float, headingRad: float):
    vx = speed * math.cos(headingRad)
    vy = speed * math.sin(headingRad)
    pos = np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype = float)
    vel = np.array([vx, vy, 0.0], dtype = float)
    return pos, vel

def PrecalcCylinderPoints(nphi: int, nz: int, dtype = float):
    R = 7.0
    H = 10.0
    center = np.array([0.0, 200.0, 0.0], dtype = dtype)
    phis = np.linspace(0, 2*np.pi, num = nphi, endpoint = False, dtype = dtype)
    zs = np.linspace(0, H, num = nz, dtype = dtype)
    pts = []
    for z in zs:
        ring = center + np.stack([R*np.cos(phis), np.zeros_like(phis), np.full_like(phis, z)], axis = 1)
        pts.append(ring)
    pts = np.concatenate(pts, axis = 0)
    pts = np.vstack([pts, center + np.array([0,0,0], dtype = dtype), center + np.array([0,0,H], dtype = dtype)])
    return pts

def SegmentPointsMask(mPos: np.ndarray, pts: np.ndarray, c: np.ndarray, R: float) -> np.ndarray:
    """Segment-sphere intersection"""
    d = pts - mPos[None, :]
    f = mPos - c
    a = np.sum(d*d, axis = 1)
    b = 2.0 * np.sum(f[None,:]*d, axis = 1)
    csc = float(np.dot(f,f) - R*R)
    disc = b*b - 4.0*a*csc
    good = (a > 0.0) & (disc >= 0.0)
    sqrtDisc = np.zeros_like(disc)
    sqrtDisc[good] = np.sqrt(np.maximum(disc[good], 0.0))
    t1 = np.empty_like(disc)
    t2 = np.empty_like(disc)
    t1[good] = (-b[good] - sqrtDisc[good])/(2.0*a[good])
    t2[good] = (-b[good] + sqrtDisc[good])/(2.0*a[good])
    inter = good & ( ((t1>=0.0)&(t1<=1.0)) | ((t2>=0.0)&(t2<=1.0)) )
    insideP = (np.sum((pts - c[None,:])**2, axis = 1) <= R*R + 1e-12)
    insideM = (np.dot(mPos - c, mPos - c) <= R*R + 1e-12)
    return inter | insideP | insideM

def ConePointsIn(mPos: np.ndarray, c: np.ndarray, pts: np.ndarray, R: float) -> np.ndarray:
    a = c - mPos
    d2 = float(np.dot(a,a))
    b = pts - mPos[None,:]
    dot = b @ a
    fwd = dot >= 0.0
    if d2 <= R*R:
        return fwd
    rhs = (np.sum(b*b, axis = 1)) * (d2 - R*R)
    lhs = dot*dot
    return fwd & (lhs >= rhs)

def ReadSolution(xlsxPath: str):
    df = pd.read_excel(xlsxPath, sheet_name = 0)
    need = [
        "Direction_degrees", "Speed_m_per_s", "Smoke_Bomb_Number",
        "Drop_Point_X_m", "Drop_Point_Y_m", "Drop_Point_Z_m",
        "Detonation_Point_X_m", "Detonation_Point_Y_m", "Detonation_Point_Z_m",
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    rows = [df.iloc[i] for i in range(3)]
    headingText = str(rows[0]["Direction_degrees"])
    if "rad" in headingText:
        headingRad = float(headingText.replace("rad","").strip())
    else:
        headingRad = float(rows[0]["Direction_degrees"])
    speed = float(rows[0]["Speed_m_per_s"])
    drops, expls = [], []
    for r in rows:
        drops.append(np.array([float(r["Drop_Point_X_m"]),
                               float(r["Drop_Point_Y_m"]),
                               float(r["Drop_Point_Z_m"])], dtype = float))
        expls.append(np.array([float(r["Detonation_Point_X_m"]),
                               float(r["Detonation_Point_Y_m"]),
                               float(r["Detonation_Point_Z_m"])], dtype = float))
    return headingRad, speed, drops, expls

def RecoverTimes(headingRad: float, speed: float, drops, expls):
    vx = speed * math.cos(headingRad)
    vy = speed * math.sin(headingRad)
    vxy = np.array([vx,vy], dtype = float)
    fuses, tdrops, texpls = [], [], []
    for D, E in zip(drops, expls):
        dz = float(D[2] - E[2])
        fuse = math.sqrt(max(0.0, 2.0*dz/Q1.g))
        denom = float(np.dot(vxy, vxy)) + 1e-12
        tDrop = float(np.dot(D[:2] - Q1.FY1_INIT[:2], vxy) / denom)
        fuses.append(fuse)
        tdrops.append(max(0.0, tDrop))
        texpls.append(tdrops[-1] + fuse)
    return tdrops, fuses, texpls

def main():
    ap = argparse.ArgumentParser("Q3 drawer")
    ap.add_argument("--xlsx", type = str, default = "result3.xlsx", help = "solver output xlsx")
    ap.add_argument("--out", type = str, default = "q3_anim.mp4")
    ap.add_argument("--anim_seconds", type = float, default = 40.0)
    ap.add_argument("--fps", type = int, default = 24)
    ap.add_argument("--crf", type = int, default = 22)
    ap.add_argument("--dpi", type = int, default = 120)
    ap.add_argument("--sphere_res", type = int, default = 16)
    ap.add_argument("--skip", type = int, default = 1)
    ap.add_argument("--masking_mode", choices = ["segment","cone"], default = "segment")
    ap.add_argument("--label_fontsize", type = int, default = 9)
    args = ap.parse_args()

    ffmpexe = imageio_ffmpeg.get_ffmpeg_exe()
    Info(f"Using ffmpeg: {ffmpexe}")

    Info(f"Reading {args.xlsx}")
    headingRad, speed, drops, expls = ReadSolution(args.xlsx)

    ptsMask = Q1.PreCalCylinderPoints(360, 7, dtype = float)

    fps = args.fps
    T = float(args.anim_seconds)
    N = int(round(T * fps)) + 1
    ts = np.linspace(0.0, T, N, dtype = float)
    Info(f"Animation length = {T:.3f}s @ {fps} fps ({len(ts)} frames)")
    Info(f"Masking mode = {args.masking_mode}")

    FY1 = Q1.FY1_INIT.astype(float)
    M1 = Q1.M1_INIT.astype(float)
    FAKE = Q1.FAKE_TARGET_ORIGIN.astype(float)
    MISSILE_SPEED = float(Q1.MISSILE_SPEED)
    R = float(Q1.SMOG_R)
    SINK = float(Q1.SMOG_SINK_SPEED)

    vx = speed * math.cos(headingRad)
    vy = speed * math.sin(headingRad)
    mPath = np.vstack([MissilePosAt(t, M1, FAKE, MISSILE_SPEED) for t in ts])
    fyPath = np.vstack([np.array([FY1[0] + vx*t, FY1[1] + vy*t, FY1[2]], dtype = float) for t in ts])

    tdrops, fuses, texpls = RecoverTimes(headingRad, speed, drops, expls)

    fig = plt.figure(figsize = (10, 6), dpi = args.dpi)
    ax = fig.add_subplot(1,1,1, projection = '3d')

    xmin = min(FY1[0], M1[0], 0.0) - 200.0
    xmax = max(FY1[0], M1[0], 0.0) + 200.0
    ymin = min(-800.0, FY1[1]-500.0, M1[1]-500.0)
    ymax = max( 800.0, FY1[1]+500.0, M1[1]+500.0)
    zmin, zmax = 0.0, max(FY1[2], M1[2], 2100.0) + 200.0
    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Q3 Smoke Interference")

    cylPhi = np.linspace(0, 2*np.pi, 120)
    R_cyl, H_cyl = 7.0, 10.0
    cx, cy = 0.0, 200.0
    for z in (0.0, 10.0):
        ax.plot(cx + R_cyl*np.cos(cylPhi), cy + 0*cylPhi, z + 0*cylPhi, color = "k", lw = 1, alpha = 0.6)
    for ang in np.linspace(0, 2*np.pi, 12, endpoint = False):
        x = cx + R_cyl*np.cos(ang)
        y = cy
        ax.plot([x, x], [y, y], [0.0, 10.0], color = "k", lw = 0.5, alpha = 0.3)

    mline, = ax.plot([], [], [], color = "tab:red", lw = 2)
    mpt,   = ax.plot([], [], [], "o", color = "tab:red", ms = 4)
    fyline,= ax.plot([], [], [], color = "tab:blue", lw = 1.5)
    fypt,  = ax.plot([], [], [], "o", color = "tab:blue", ms = 4)

    bombColors = ["tab:green","tab:orange","tab:purple"]
    bombPts = [ax.plot([], [], [], "o", color = bombColors[i], ms = 4)[0] for i in range(3)]
    cloudPts= [ax.plot([], [], [], "o", color = bombColors[i], ms = 4, alpha = 0.6)[0] for i in range(3)]

    fsz = args.label_fontsize
    fig.text(0.015, 0.96,
        "Legend:\n"
        "  red  : M1 & path\n"
        "  blue : FY1 & path\n"
        "  green/orange/purple : Bomb bodies & cloud centers",
        ha = "left", va = "top", fontsize = fsz,
        bbox = dict(facecolor = "white", alpha = 0.75, edgecolor = "none"))

    coverTimer = fig.text(0.015, 0.86, "Covered time: 0.000 s",
        ha = "left", va = "top", fontsize = fsz+1,
        bbox = dict(facecolor = "white", alpha = 0.75, edgecolor = "none"))

    bombMonitor = fig.text(0.015, 0.12, "Bomb bodies:\n", ha = "left", va = "bottom", fontsize = fsz,
                            bbox = dict(facecolor = "white", alpha = 0.75, edgecolor = "none"))
    cloudMonitor= fig.text(0.015, 0.05, "Cloud centers:\n", ha = "left", va = "bottom", fontsize = fsz,
                            bbox = dict(facecolor = "white", alpha = 0.75, edgecolor = "none"))

    coveredMask = np.zeros(len(ts), dtype = bool)

    def UnitSphere(res):
        u = np.linspace(0, 2*np.pi, res, endpoint = True)
        v = np.linspace(0, np.pi, res//2+1, endpoint = True)
        x = np.outer(np.cos(u), np.sin(v)).T
        y = np.outer(np.sin(u), np.sin(v)).T
        z = np.outer(np.ones_like(u), np.cos(v)).T
        return x, y, z
    sphX, sphY, sphZ = UnitSphere(max(8, args.sphere_res))

    activeWires = []
    def ClearWires():
        nonlocal activeWires
        for w in activeWires:
            try:
                w.remove()
            except Exception:
                pass
        activeWires = []

    def CloudCenterAt(expl, te, tnow):
        if tnow < te:
            return None
        cz = float(expl[2]) - SINK * (tnow - te)
        if cz <= 0.0 + 1e-9:
            return None
        return np.array([expl[0], expl[1], cz], dtype = float)

    def FmtXyz(p):
        return f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"

    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
    except AttributeError:
        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)
    
    writer = imageio.get_writer(
        args.out, format = "FFMPEG", mode = "I",
        fps = args.fps, codec = "libx264",
        ffmpeg_log_level = "error",
        macro_block_size = None,
        ffmpeg_params = ['-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    )

    lastBodyPos = [None, None, None]
    lastCloudPos= [None, None, None]

    try:
        wireEvery = max(10, args.fps // 2)

        for fi, t in enumerate(ts[::max(1, args.skip)]):
            i = fi * max(1, args.skip)

            mp = mPath[i]
            fyp = fyPath[i]
            mpt.set_data([mp[0]],[mp[1]])
            mpt.set_3d_properties([mp[2]])
            fypt.set_data([fyp[0]],[fyp[1]])
            fypt.set_3d_properties([fyp[2]])
            if i == 0:
                mline.set_data([mp[0]],[mp[1]])
                mline.set_3d_properties([mp[2]])
                fyline.set_data([fyp[0]],[fyp[1]])
                fyline.set_3d_properties([fyp[2]])
            else:
                mline.set_data(mPath[:i+1,0], mPath[:i+1,1])
                mline.set_3d_properties(mPath[:i+1,2])
                fyline.set_data(fyPath[:i+1,0], fyPath[:i+1,1])
                fyline.set_3d_properties(fyPath[:i+1,2])

            for bi in range(3):
                D = drops[bi]
                E = expls[bi]
                tDrop = tdrops[bi]
                fuse = fuses[bi]
                tExpl = texpls[bi]
                if t <= tExpl + 1e-9:
                    tau = max(0.0, t - tDrop)
                    pos = np.array([D[0] + vx*tau, D[1] + vy*tau, D[2] - 0.5*Q1.g*(tau**2)], dtype = float)
                    bombPts[bi].set_data([pos[0]],[pos[1]])
                    bombPts[bi].set_3d_properties([pos[2]])
                    lastBodyPos[bi] = pos
                else:
                    bombPts[bi].set_data([],[])
                    bombPts[bi].set_3d_properties([])
                    lastBodyPos[bi] = None

            anyMaskPts = None
            if (i % wireEvery) == 0:
                ClearWires()

            for bi in range(3):
                D = drops[bi]
                E = expls[bi]
                tDrop = tdrops[bi]
                fuse = fuses[bi] 
                tExpl = texpls[bi]
                c = CloudCenterAt(E, tExpl, t)
                lastCloudPos[bi] = c
                if c is None:
                    cloudPts[bi].set_data([],[])
                    cloudPts[bi].set_3d_properties([])
                    continue
                cloudPts[bi].set_data([c[0]],[c[1]])
                cloudPts[bi].set_3d_properties([c[2]])

                if args.masking_mode == "segment":
                    maskIn = SegmentPointsMask(mp, ptsMask, c, R)
                else:
                    maskIn = ConePointsIn(mp, c, ptsMask, R)
                anyMaskPts = maskIn if anyMaskPts is None else (anyMaskPts | maskIn)

                if (i % wireEvery) == 0:
                    xs = R * sphX + c[0]
                    ys = R * sphY + c[1]
                    zs = R * sphZ + c[2]
                    wf = ax.plot_wireframe(xs, ys, zs, rstride = 2, cstride = 2,
                                           color = bombColors[bi], alpha = 0.18, linewidth = 0.35)
                    activeWires.append(wf)

            covered = (anyMaskPts is not None) and bool(np.all(anyMaskPts))
            if covered:
                coveredMask[i] = True

            totCov = float(np.count_nonzero(coveredMask) / fps)
            coverTimer.set_text(f"Covered time: {totCov:.3f} s")

            bodiesTxt = []
            for bi in range(3):
                p = lastBodyPos[bi]
                bodiesTxt.append(f"B{bi+1}: {FmtXyz(p)}" if p is not None else f"B{bi+1}: (-)")
            bombMonitor.set_text("Bomb bodies:\n" + " | ".join(bodiesTxt))

            cloudsTxt = []
            for bi in range(3):
                c = lastCloudPos[bi]
                cloudsTxt.append(f"C{bi+1}: {FmtXyz(c)}" if c is not None else f"C{bi+1}: (-)")
            cloudMonitor.set_text("Cloud centers:\n" + " | ".join(cloudsTxt))

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]
            writer.append_data(frame)

            if (fi % 25) == 0 or fi == (len(ts[::max(1,args.skip)])-1):
                Info(f"frame {fi+1}/{len(ts[::max(1,args.skip)])}")

        Info("Finalizing writer")
    finally:
        writer.close()
        plt.close(fig)

    totalSec = float(np.count_nonzero(coveredMask) / fps)
    Info(f"Done. Covered total time ≈ {totalSec:.3f} s")
    Info(f"Saved MP4 → {args.out}")

if __name__ == "__main__":
    main()
