# -*- coding: utf-8 -*-
"""
Q4Solver_visual.py — 协同遮蔽 (线段-球相交并集) + 保留单体=0候选 + 横向偏移 + 粗细两级 + 轻量SA + 并行
新增：在候选生成阶段显式过滤“起爆瞬间在地下(z<0)”与“起爆瞬间在导弹背向半空间”的无效爆点。
输出：
  - result2.xlsx（中文10列，每行为对应UAV的单体时长；控制台打印联合并集时长）
  - Q4ConvergencePlot.png（收敛曲线，best-so-far）
"""

import os, math, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# ===== 场景常量 =====
g = 9.81
EPS = 1e-12

CYL_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYL_R, CYL_H = 7.0, 10.0

SMOKE_R = 10.0
SMOKE_SINK = 3.0
SMOKE_VALID = 20.0

M_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)     # 默认 M1
ORIGIN  = np.array([0.0, 0.0, 0.0], dtype=float)
MISSILE_SPEED = 300.0
MISSILE_DIR = (ORIGIN - M_INIT)/np.linalg.norm(ORIGIN - M_INIT)
HIT_TIME = float(np.linalg.norm(ORIGIN - M_INIT)/MISSILE_SPEED)

UAV_INITS = {
    "FY1": np.array([17800.0,     0.0, 1800.0], dtype=float),
    "FY2": np.array([12000.0,  1400.0, 1400.0], dtype=float),
    "FY3": np.array([ 6000.0, -3000.0,  700.0], dtype=float),
}
UAV_NAMES = ["FY1","FY2","FY3"]

# FY1 固定（来自 Q2）
FY1_FIXED = np.array([0.137353, 112.0298, 0.0045, 0.4950], dtype=float)

# ===== 基础工具 =====
def unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v/n

def missile_pos(t):
    return M_INIT + MISSILE_SPEED*t*MISSILE_DIR

def detonation(uav_init, theta, v, t1, t2):
    d = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    drop = uav_init + v*t1*d
    det_xy = drop[:2] + v*t2*d[:2]
    det_z  = drop[2] - 0.5*g*t2*t2
    return drop, np.array([det_xy[0], det_xy[1], det_z], dtype=float)

def cylinder_points(n_phi=480, n_z=8):
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    ring = np.stack([CYL_R*np.cos(phis), CYL_R*np.sin(phis), np.zeros_like(phis)], axis=1)
    pts = [CYL_CENTER + ring, CYL_CENTER + np.array([0,0,CYL_H], dtype=float) + ring]
    if n_z >= 2:
        for z in np.linspace(0, CYL_H, n_z, endpoint=True):
            pts.append(CYL_CENTER + np.array([0,0,z], dtype=float) + ring)
    return np.vstack(pts).astype(np.float64)

def events_time_grid(t0, t1, events, dtc, dtf, win=1.0):
    if t1 <= t0 + 1e-12: return np.array([], dtype=float)
    if not events:       return np.arange(t0, t1+dtc/2, dtc)
    seg=[]
    for e in events:
        seg.append((max(t0,e-win), min(t1,e+win)))
    seg.sort()
    merged=[list(seg[0])]
    for a,b in seg[1:]:
        if a<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1], b)
        else:
            merged.append([a,b])
    ts=[]; prev=t0
    for a,b in merged:
        if prev<a: ts.extend(np.arange(prev,a,dtc))
        ts.extend(np.arange(a,b+dtf/2,dtf))
        prev=b
    if prev<t1: ts.extend(np.arange(prev,t1+dtc/2,dtc))
    return np.unique(np.array(ts, dtype=float))

# ===== 线段-球体（协同） =====
def _seg_sphere_any_cover(M, P, centers, r=SMOKE_R):
    """逐点协同：∀点p∈P，∃球心Ck 使线段 M->p 与球体(Ck,r) 有交；若全部点满足则 True。"""
    if len(centers)==0: return False
    MP = P - M                          # (N,3)
    a  = np.sum(MP*MP, axis=1) + EPS    # (N,)
    covered_any = np.zeros(P.shape[0], dtype=bool)
    for C in centers:
        # 物理约束：必须在导弹前向半空间
        if np.dot(C - M, MISSILE_DIR) <= 0:
            continue
        MC = C - M                      # (3,)
        b  = -2.0 * (MP @ MC)           # (N,)
        c  = (MC @ MC) - r*r            # ()
        disc = b*b - 4*a*c              # (N,)
        good = disc >= 0.0
        if not np.any(good):
            continue
        sd = np.zeros_like(a)
        sd[good] = np.sqrt(disc[good])
        s1 = (-b - sd) / (2*a)
        s2 = (-b + sd) / (2*a)
        s_lo = np.minimum(s1,s2); s_hi = np.maximum(s1,s2)
        inter_len = np.minimum(s_hi,1.0) - np.maximum(s_lo,0.0)
        covered_any |= (inter_len > 0.0)
        if np.all(covered_any):  # 已全覆盖
            return True
    return bool(np.all(covered_any))

def _seg_sphere_all_cover(M, P, C, r=SMOKE_R):
    """单体能否独立全遮（用于单体时长信息/掩码）"""
    if np.dot(C - M, MISSILE_DIR) <= 0:
        return False
    MP = P - M
    a  = np.sum(MP*MP, axis=1) + EPS
    MC = C - M
    b  = -2.0 * (MP @ MC)
    c  = (MC @ MC) - r*r
    disc = b*b - 4*a*c
    if not np.all(disc >= 0.0):
        return False
    sd = np.sqrt(np.maximum(disc,0.0))
    s1 = (-b - sd) / (2*a)
    s2 = (-b + sd) / (2*a)
    s_lo = np.minimum(s1,s2); s_hi = np.maximum(s1,s2)
    inter_len = np.minimum(s_hi,1.0) - np.maximum(s_lo,0.0)
    return bool(np.all(inter_len > 0.0))

# ===== 全局（子进程） =====
PTS_GLOBAL = None
def _init_worker(nphi, nz, intra_threads):
    global PTS_GLOBAL
    PTS_GLOBAL = cylinder_points(n_phi=nphi, n_z=nz)
    if threadpool_limits is not None:
        threadpool_limits(limits=max(1,int(intra_threads)))

# ===== 单体时长 / 单体掩码 =====
def _single_duration(T_e, expl, dt):
    t0, t1 = T_e, min(T_e + SMOKE_VALID, HIT_TIME)
    if t1 <= t0 + 1e-12: return 0.0
    cur, acc = t0, 0.0
    while cur <= t1 + 1e-12:
        M = missile_pos(cur)
        z = expl[2] - SMOKE_SINK*max(0.0, cur-T_e)
        if z >= 0.0:
            C = np.array([expl[0], expl[1], z], dtype=float)
            if _seg_sphere_all_cover(M, PTS_GLOBAL, C, r=SMOKE_R):
                acc += dt
        cur += dt
    return float(acc)

def _single_mask(T_e, expl, tgrid):
    mask = np.zeros_like(tgrid, dtype=bool)
    for i,t in enumerate(tgrid):
        if t < T_e-1e-12 or t > T_e+SMOKE_VALID+1e-12: continue
        M = missile_pos(t)
        z = expl[2] - SMOKE_SINK*max(0.0, t-T_e)
        if z < 0.0: 
            continue
        C = np.array([expl[0], expl[1], z], dtype=float)
        mask[i] = _seg_sphere_all_cover(M, PTS_GLOBAL, C, r=SMOKE_R)
    return mask

# ===== 候选生成（横向偏移 + 保留单体=0 + 新增起爆瞬间约束） =====
def _best_candidate_for_Te_speed(uav_init, T_e, spd, lateral_scales=(0.0, 0.015, -0.015), dt_eval=0.001):
    # 命中前才有意义
    if T_e >= HIT_TIME - 1e-9:
        return None

    mPos = missile_pos(T_e)
    tgt  = CYL_CENTER + np.array([0.0,0.0,CYL_H*0.5])
    L = tgt - mPos
    dist = np.linalg.norm(L)
    if dist < 1e-9: return None
    eL = L / dist
    # 水平横向单位向量
    perp = np.array([-eL[1], eL[0], 0.0]); 
    nperp = np.linalg.norm(perp)
    if nperp < 1e-9: perp = np.array([1.0,0.0,0.0]); nperp=1.0
    perp /= nperp

    best = None
    s_grid = np.linspace(0.07, 1.0, 16)[::-1]  # 优先靠近目标（协同更易）
    for dscale in lateral_scales:
        for s in s_grid:
            cand = mPos + s*L + dscale*dist*perp
            # 可达性与物理约束（UAV高度≥爆点高度；水平速度可达）
            if cand[2] > uav_init[2] + 1e-9: 
                continue
            dx,dy = cand[0]-uav_init[0], cand[1]-uav_init[1]
            req_v = math.hypot(dx,dy)/max(T_e,1e-9)
            if req_v > spd + 1e-12: 
                continue
            heading = math.atan2(dy, dx)
            fuse = math.sqrt(max(0.0, 2.0*(uav_init[2]-cand[2])/g))
            drop  = T_e - fuse
            if drop < -1e-9: 
                continue
            # 真正起爆点
            dropPos = uav_init + np.array([spd*drop*math.cos(heading),
                                           spd*drop*math.sin(heading), 0.0])
            expl = np.array([dropPos[0]+spd*fuse*math.cos(heading),
                             dropPos[1]+spd*fuse*math.sin(heading),
                             uav_init[2]-0.5*g*fuse*fuse], dtype=float)

            # ===== 新增硬约束：起爆瞬间必须在地上且处于导弹前向半空间 =====
            if expl[2] < 0.5:           # 留0.5m安全裕度
                continue
            if np.dot(expl - mPos, MISSILE_DIR) <= 0:
                continue

            dur = _single_duration(T_e, expl, dt_eval)  # 允许=0（用于协同）
            c = {
                "uavName": None, "T_e": float(T_e), "speed": float(spd), "heading": float(heading),
                "drop_time": float(drop), "fuse_delay": float(fuse),
                "expl_pos": expl, "single_duration": float(dur),
                "score_s": float(s)  # 协同偏好
            }
            if (best is None) or (dur > best["single_duration"] + 1e-12) or \
               (abs(dur - best["single_duration"]) <= 1e-12 and s > best["score_s"]):
                best = c
    return best

def _candidate_worker(payload):
    uavName = payload["uavName"]; uavInit = payload["uavInit"]
    T_e = payload["T_e"]; spd = payload["speed"]
    lat = payload["lat_scales"]; dt = payload["dt"]
    if threadpool_limits is None:
        cand = _best_candidate_for_Te_speed(uavInit, T_e, spd, lateral_scales=lat, dt_eval=dt)
    else:
        with threadpool_limits(limits=1):
            cand = _best_candidate_for_Te_speed(uavInit, T_e, spd, lateral_scales=lat, dt_eval=dt)
    if cand is None: return None
    cand["uavName"] = uavName
    return cand

# ===== 联合并集（真·协同；时刻内再次检查 z>=0 & 前向半空间） =====
def _union_duration_coop(triple, dt):
    Te = [triple[i]["T_e"] for i in range(3)]
    t0 = min(Te); t1 = min(HIT_TIME, max(Te) + SMOKE_VALID)
    if t1 <= t0 + 1e-12: return 0.0
    ts = events_time_grid(t0, t1, Te, max(dt*3, 0.003), dt, win=1.0)
    if len(ts) == 0: return 0.0
    occ = 0.0; prev = ts[0]
    for t in ts[1:]:
        M = missile_pos(t)
        centers=[]
        for i in (0,1,2):
            Tei = triple[i]["T_e"]
            if t < Tei-1e-12 or t > Tei+SMOKE_VALID+1e-12: 
                continue
            z = triple[i]["expl_pos"][2] - SMOKE_SINK*max(0.0, t-Tei)
            if z < 0.0: 
                continue
            C = np.array([triple[i]["expl_pos"][0], triple[i]["expl_pos"][1], z], dtype=float)
            if np.dot(C - M, MISSILE_DIR) <= 0:   # 前向半空间
                continue
            centers.append(C)
        if centers and _seg_sphere_any_cover(M, PTS_GLOBAL, centers, r=SMOKE_R):
            occ += (t - prev)
        prev = t
    return float(occ)

def _union_worker(payload):
    tri, dt = payload["triplet"], payload["dt"]
    if threadpool_limits is None:
        return _union_duration_coop(tri, dt)
    else:
        with threadpool_limits(limits=1):
            return _union_duration_coop(tri, dt)

# ===== 粗评掩码 =====
def _coarse_mask_for(cand, tgrid):
    return _single_mask(cand["T_e"], cand["expl_pos"], tgrid)

# ===== TopK 混合保留（含单体=0） =====
def _select_topk_mixed(cands, topk, min_gap, pos_ratio=0.75):
    if not cands: return []
    pos = [c for c in cands if c["single_duration"] > 1e-12]
    zer = [c for c in cands if c["single_duration"] <= 1e-12]
    pos = sorted(pos, key=lambda x:(-x["single_duration"], x["T_e"]))
    zer = sorted(zer, key=lambda x:(-x["score_s"], x["T_e"]))
    sel, used, want_pos = [], [], int(round(topk*pos_ratio))
    for c in pos:
        T=c["T_e"]
        if all(abs(T-u)>=min_gap for u in used):
            sel.append(c); used.append(T)
        if len(sel)>=want_pos: break
    for c in zer:
        if len(sel)>=topk: break
        T=c["T_e"]
        if all(abs(T-u)>=min_gap for u in used):
            sel.append(c); used.append(T)
    if len(sel)<topk:
        for c in pos:
            if c in sel: continue
            T=c["T_e"]
            if all(abs(T-u)>=min_gap for u in used):
                sel.append(c); used.append(T)
            if len(sel)>=topk: break
    return sel

# ===== 打印 / Excel / 图 =====
def _deg(rad):
    d = math.degrees(rad); 
    return (d+360.0)%360.0

def _print_best(prefix, union_time, tri):
    print(f"[Best↑][{prefix}] 联合遮蔽 ≈ {union_time:.6f} s")
    for c in sorted(tri, key=lambda x:x["uavName"]):
        ex=c["expl_pos"]; print(
          f"  - {c['uavName']}: v={c['speed']:.3f} m/s, heading={_deg(c['heading']):.3f}°, "
          f"T_e={c['T_e']:.3f}s, drop={c['drop_time']:.3f}s, fuse={c['fuse_delay']:.3f}s, "
          f"expl=({ex[0]:.3f},{ex[1]:.3f},{ex[2]:.3f}), single={c['single_duration']:.6f}s, s_bias={c['score_s']:.3f}"
        )

def to_excel(triple, total_union, fname="result2.xlsx"):
    rows=[]
    for c in sorted(triple, key=lambda x:x["uavName"]):
        h=_deg(c["heading"]); ex=c["expl_pos"]
        d = np.array([math.cos(c["heading"]), math.sin(c["heading"]), 0.0], dtype=float)
        drop = UAV_INITS[c["uavName"]] + c["speed"]*c["drop_time"]*d
        rows.append({
            "无人机编号": c["uavName"],
            "无人机运动方向": round(h,6),
            "无人机运动速度 (m/s)": round(c["speed"],6),
            "烟幕干扰弹投放点的x坐标 (m)": round(drop[0],6),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop[1],6),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop[2],6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(ex[0],6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(ex[1],6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(ex[2],6),
            "有效干扰时长 (s)": round(c["single_duration"],6)  # 单体
        })
    df = pd.DataFrame(rows)[[
        '无人机编号','无人机运动方向','无人机运动速度 (m/s)',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)'
    ]]
    df.to_excel(fname, index=False)
    print("-"*110)
    print(df.to_string(index=False))
    print(f"[Result] TOTAL UNION (cooperative, no double count) ≈ {total_union:.6f} s | 已保存: {fname}")

def plot_conv(hist, out="Q4ConvergencePlot.png"):
    xs=np.arange(len(hist)); best=np.maximum.accumulate(hist)
    plt.figure(figsize=(9,5),dpi=120)
    plt.plot(xs,best,lw=2)
    plt.xlabel("Iteration"); plt.ylabel("Best Union (s)")
    plt.title("Q4 Hybrid (cooperative) convergence")
    plt.grid(alpha=0.3)
    plt.savefig(out,dpi=300,bbox_inches='tight')
    print(f"[Plot] saved {out}")

# ===== 自适应并行 =====
def _auto_balance(workers_opt, intra_opt, backend, total_tasks):
    cpu = os.cpu_count() or 1
    if backend == "thread":
        workers = cpu if (workers_opt == "auto") else int(workers_opt)
        return workers, 1
    workers = (cpu if workers_opt=="auto" else max(1,int(workers_opt)))
    if total_tasks < workers: workers = max(1,total_tasks)
    intra = (1 if intra_opt=="auto" else max(1,int(intra_opt)))
    return workers, intra

# ===== main =====
def main():
    ap = argparse.ArgumentParser("Q4 Solver (cooperative union + hybrid parallel)")
    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--intra-threads", default="auto")

    ap.add_argument("--dt", type=float, default=0.0012)
    ap.add_argument("--dt-coarse", type=float, default=0.004)
    ap.add_argument("--nphi", type=int, default=600)
    ap.add_argument("--nz", type=int, default=8)

    ap.add_argument("--speed-grid", type=str, default="140,130,120,110,100")
    ap.add_argument("--fy1-te", type=str, default="6,10,0.03")
    ap.add_argument("--fy2-te", type=str, default="12,20,0.03")
    ap.add_argument("--fy3-te", type=str, default="22,34,0.03")
    ap.add_argument("--lat-scales", type=str, default="0.0,0.018,-0.018")

    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min-gap", type=float, default=0.30)
    ap.add_argument("--combo-topm", type=int, default=48)

    ap.add_argument("--sa-iters", type=int, default=80)
    ap.add_argument("--sa-batch", default="auto")
    ap.add_argument("--sa-T0", type=float, default=1.0)
    ap.add_argument("--sa-alpha", type=float, default=0.92)
    ap.add_argument("--sigma-Te", type=float, default=0.55)
    ap.add_argument("--sigma-v", type=float, default=10.0)
    ap.add_argument("--local-iters", type=int, default=30)

    args = ap.parse_args()

    # 采样点
    global PTS_GLOBAL
    PTS_GLOBAL = cylinder_points(n_phi=args.nphi, n_z=args.nz)
    print(f"[Info] target samples = {len(PTS_GLOBAL)}")

    # grids
    def _parse_te(s):
        a,b,st=[float(x) for x in s.split(",")]
        if st<=0: st=0.05
        return np.arange(a,b+1e-12,st)
    def _parse_speed(s):
        vs=[]
        for tok in s.split(","):
            tok=tok.strip()
            if tok:
                v=float(tok); v=max(70.0, min(140.0, v)); vs.append(v)
        return sorted(set(vs), reverse=True)
    def _parse_lat(s):
        arr=[]
        for tok in s.split(","):
            tok=tok.strip()
            if tok: arr.append(float(tok))
        return arr if arr else [0.0]

    te1=_parse_te(args.fy1_te); te2=_parse_te(args.fy2_te); te3=_parse_te(args.fy3_te)
    speed_set=_parse_speed(args.speed_grid)
    lateral_list=_parse_lat(args.lat_scales)

    # 阶段1：候选生成
    gen_tasks=[]
    for name,init,grid in [("FY1",UAV_INITS["FY1"],te1), ("FY2",UAV_INITS["FY2"],te2), ("FY3",UAV_INITS["FY3"],te3)]:
        for T_e in grid:
            for v in speed_set:
                gen_tasks.append({"uavName":name, "uavInit":init, "T_e":float(T_e),
                                  "speed":float(v), "lat_scales":lateral_list, "dt":float(args.dt)})

    workers,intra=_auto_balance(args.workers, args.intra_threads, args.backend, len(gen_tasks))
    poolCls = ThreadPoolExecutor if args.backend=="thread" else ProcessPoolExecutor
    print("="*110)
    print(f"Stage-1: candidates | backend={args.backend}, workers={workers}, intra-threads={intra}, tasks={len(gen_tasks)}")
    print("="*110)

    best_lists={"FY1":[], "FY2":[], "FY3":[]}
    tA=time.time()
    if args.backend=="process":
        with poolCls(max_workers=workers, initializer=_init_worker, initargs=(args.nphi,args.nz,intra)) as pool:
            futs={ pool.submit(_candidate_worker, pl):i for i,pl in enumerate(gen_tasks) }
            done,total=0,len(futs)
            for fut in as_completed(futs):
                done+=1
                try: r=fut.result()
                except Exception: r=None
                if r is not None:
                    best_lists[r["uavName"]].append(r)   # 不丢单体=0
                if done%max(1,total//20)==0:
                    print(f"  [Gen] {int(100*done/total)}%")
    else:
        _init_worker(args.nphi,args.nz,1)
        with poolCls(max_workers=workers) as pool:
            futs={ pool.submit(_candidate_worker, pl):i for i,pl in enumerate(gen_tasks) }
            done,total=0,len(futs)
            for fut in as_completed(futs):
                done+=1
                try: r=fut.result()
                except Exception: r=None
                if r is not None:
                    best_lists[r["uavName"]].append(r)
                if done%max(1,total//20)==0:
                    print(f"  [Gen] {int(100*done/total)}%")
    tB=time.time()
    print(f"[Stage-1] FY1={len(best_lists['FY1'])}, FY2={len(best_lists['FY2'])}, FY3={len(best_lists['FY3'])} | {tB-tA:.2f}s")

    # TopK 混合保留
    def _select_topk_mixed(cands, topk, min_gap, pos_ratio=0.75):
        if not cands: return []
        pos = [c for c in cands if c["single_duration"] > 1e-12]
        zer = [c for c in cands if c["single_duration"] <= 1e-12]
        pos = sorted(pos, key=lambda x:(-x["single_duration"], x["T_e"]))
        zer = sorted(zer, key=lambda x:(-x["score_s"], x["T_e"]))
        sel, used, want_pos = [], [], int(round(topk*pos_ratio))
        for c in pos:
            T=c["T_e"]
            if all(abs(T-u)>=min_gap for u in used):
                sel.append(c); used.append(T)
            if len(sel)>=want_pos: break
        for c in zer:
            if len(sel)>=topk: break
            T=c["T_e"]
            if all(abs(T-u)>=min_gap for u in used):
                sel.append(c); used.append(T)
        if len(sel)<topk:
            for c in pos:
                if c in sel: continue
                T=c["T_e"]
                if all(abs(T-u)>=min_gap for u in used):
                    sel.append(c); used.append(T)
                if len(sel)>=topk: break
        return sel

    fy1 = _select_topk_mixed(best_lists["FY1"], args.topk, args.min_gap, pos_ratio=0.75)
    fy2 = _select_topk_mixed(best_lists["FY2"], args.topk, args.min_gap, pos_ratio=0.75)
    fy3 = _select_topk_mixed(best_lists["FY3"], args.topk, args.min_gap, pos_ratio=0.75)
    if not (fy1 and fy2 and fy3):
        print("[Warn] 无可行候选，输出全0。")
        rows=[]
        for n in ("FY1","FY2","FY3"):
            rows.append({ "无人机编号":n, "无人机运动方向":0.0, "无人机运动速度 (m/s)":0.0,
                          "烟幕干扰弹投放点的x坐标 (m)":0.0, "烟幕干扰弹投放点的y坐标 (m)":0.0, "烟幕干扰弹投放点的z坐标 (m)":0.0,
                          "烟幕干扰弹起爆点的x坐标 (m)":0.0, "烟幕干扰弹起爆点的y坐标 (m)":0.0, "烟幕干扰弹起爆点的z坐标 (m)":0.0,
                          "有效干扰时长 (s)":0.0 })
        pd.DataFrame(rows).to_excel("result2.xlsx", index=False)
        print("[Result] 0.000 s | Saved result2.xlsx")
        return

    print(f"[Stage-1] Kept topK: FY1={len(fy1)}, FY2={len(fy2)}, FY3={len(fy3)}")

    # 阶段2：粗评（掩码 OR + 重叠惩罚）
    print("="*110); print("Stage-2: coarse combos (mask OR + overlap penalty)"); print("="*110)
    tGrid_coarse = np.arange(0.0, HIT_TIME+1e-12, args.dt_coarse, dtype=float)
    for lst in (fy1,fy2,fy3):
        for c in lst:
            c["_mask"] = _coarse_mask_for(c, tGrid_coarse)

    def _coarse_score(c1,c2,c3, lam=0.22):
        m1,m2,m3 = c1["_mask"], c2["_mask"], c3["_mask"]
        union = (m1 | m2 | m3).sum()*args.dt_coarse
        overlap=((m1&m2).sum()+(m1&m3).sum()+(m2&m3).sum())*args.dt_coarse
        return float(union - lam*overlap)

    coarse_tasks=[]
    for c1 in fy1:
        for c2 in fy2:
            for c3 in fy3:
                coarse_tasks.append((c1,c2,c3))

    scores=np.empty(len(coarse_tasks), dtype=float)
    best_tri=None; best_s=-1e9
    for i,(c1,c2,c3) in enumerate(coarse_tasks):
        s=_coarse_score(c1,c2,c3)
        scores[i]=s
        if s>best_s:
            best_s=s; best_tri=(c1,c2,c3)
            _print_best("Coarse", best_s, best_tri)

    idxs = np.argsort(scores)[::-1][:min(args.combo_topm, len(scores))]
    seeds = [ coarse_tasks[i] for i in idxs ]
    print(f"[Stage-2] Coarse best≈{best_s:.6f}s | seeds={len(seeds)}")

    # 阶段3：精评 + 轻量SA（真·协同）
    print("="*110); print("Stage-3: exact cooperative + light SA"); print("="*110)

    # 先精评种子
    workers2,intra2=_auto_balance(args.workers, args.intra_threads, args.backend, len(seeds))
    poolCls = ThreadPoolExecutor if args.backend=="thread" else ProcessPoolExecutor
    exact_scores=[]
    if args.backend=="process":
        with poolCls(max_workers=workers2, initializer=_init_worker, initargs=(args.nphi,args.nz,intra2)) as pool:
            futs={ pool.submit(_union_worker, {"triplet": tri, "dt":float(args.dt)}):i for i,tri in enumerate(seeds) }
            for fut in as_completed(futs):
                try: exact_scores.append( (futs[fut], fut.result()) )
                except Exception: pass
    else:
        _init_worker(args.nphi,args.nz,1)
        with poolCls(max_workers=workers2) as pool:
            futs={ pool.submit(_union_worker, {"triplet": tri, "dt":float(args.dt)}):i for i,tri in enumerate(seeds) }
            for fut in as_completed(futs):
                try: exact_scores.append( (futs[fut], fut.result()) )
                except Exception: pass
    if not exact_scores:
        print("[Error] no exact scores"); return
    exact_scores.sort(key=lambda x:x[1], reverse=True)
    best_triple = seeds[exact_scores[0][0]]
    best_exact  = float(exact_scores[0][1])
    _print_best("SeedExact", best_exact, best_triple)
    hist=[best_exact]

    # 轻量 SA：变量只微调 Te 和 v
    sa_iters=int(args.sa_iters)
    sa_batch = (max((os.cpu_count() or 1), 16) if (isinstance(args.sa_batch,str) and args.sa_batch.lower()=="auto") else max(8,int(args.sa_batch)))
    T=float(args.sa_T0); alpha=float(args.sa_alpha)
    sigTe=float(args.sigma_Te); sigV=float(args.sigma_v)

    def _parse_te(s):
        a,b,st=[float(x) for x in s.split(",")]
        if st<=0: st=0.05
        return np.arange(a,b+1e-12,st)
    te1=_parse_te(args.fy1_te); te2=_parse_te(args.fy2_te); te3=_parse_te(args.fy3_te)

    def _x_of(tri):
        return np.array([tri[0]["T_e"], tri[1]["T_e"], tri[2]["T_e"],
                         tri[0]["speed"],tri[1]["speed"],tri[2]["speed"]], dtype=float)
    def _clip_x(x):
        lohi=[ (te1[0], te1[-1]), (te2[0], te2[-1]), (te3[0], te3[-1]) ]
        y=x.copy()
        for i,(lo,hi) in enumerate(lohi): y[i]=min(hi,max(lo,y[i]))
        for i in range(3,6): y[i]=min(140.0,max(70.0,y[i]))
        return y
    def _tri_from_x(x):
        inits=[UAV_INITS["FY1"],UAV_INITS["FY2"],UAV_INITS["FY3"]]
        out=[]
        for k in range(3):
            cand=_best_candidate_for_Te_speed(inits[k], x[k], x[3+k], lateral_scales=[0.0,0.018,-0.018], dt_eval=args.dt)
            if cand is None: return None
            cand["uavName"]=f"FY{k+1}"
            out.append(cand)
        return tuple(out)

    x_cur=_x_of(best_triple); J_cur=best_exact
    workers3,intra3=_auto_balance(args.workers, args.intra_threads, args.backend, sa_batch)
    if args.backend=="process":
        pool = ProcessPoolExecutor(max_workers=workers3, initializer=_init_worker, initargs=(args.nphi,args.nz,intra3))
    else:
        _init_worker(args.nphi,args.nz,1)
        pool = ThreadPoolExecutor(max_workers=workers3)

    try:
        for it in range(1, sa_iters+1):
            props=[]
            for _ in range(sa_batch):
                noise=np.array([np.random.normal(0,sigTe),np.random.normal(0,sigTe),np.random.normal(0,sigTe),
                                np.random.normal(0,sigV), np.random.normal(0,sigV), np.random.normal(0,sigV)], dtype=float)
                props.append(_clip_x(x_cur + noise))
            triples=[_tri_from_x(x) for x in props]
            futures={}
            for j,tri in enumerate(triples):
                if tri is None: continue
                futures[pool.submit(_union_worker, {"triplet": tri, "dt":float(args.dt)})]=j
            best_idx=None; best_val=-1e9; best_tri=None
            for fut in as_completed(futures):
                j=futures[fut]
                try: val=float(fut.result())
                except Exception: val=0.0
                if val>best_val:
                    best_val=val; best_idx=j; best_tri=triples[j]
            if best_idx is not None:
                dJ = best_val - J_cur
                if (dJ >= 0) or (np.random.rand() < math.exp(dJ/max(T,1e-9))):
                    x_cur = props[best_idx]; J_cur = best_val; best_triple = best_tri
                    hist.append(max(hist[-1], J_cur))
                    _print_best(f"SA@{it}", J_cur, best_triple)
                else:
                    hist.append(hist[-1])
            else:
                hist.append(hist[-1])
            T *= alpha
            if it % max(1, sa_iters//10)==0:
                print(f"   [SA] iter {it}/{sa_iters}, best≈{hist[-1]:.4f}s, T={T:.4f}")
    finally:
        pool.shutdown(wait=True)

    # 局部微调
    print("="*110); print("Stage-4: Local polish"); print("="*110)
    x_best = _x_of(best_triple); best_exact = J_cur
    for k in range(30):
        improved=False
        for j in range(6):
            step = (args.sigma_Te*0.25) if j<3 else (args.sigma_v*0.25)
            for sgn in (+1,-1):
                x_try = x_best.copy(); x_try[j]+= sgn*step
                x_try[:3]=np.clip(x_try[:3], [te1[0],te2[0],te3[0]],[te1[-1],te2[-1],te3[-1]])
                x_try[3:]=np.clip(x_try[3:], 70.0, 140.0)
                tri=_tri_from_x(x_try)
                if tri is None: continue
                val=_union_duration_coop(tri, args.dt)
                if val > best_exact + 1e-12:
                    best_exact=val; best_triple=tri; x_best=x_try; improved=True
                    _print_best(f"Local@{k+1}", best_exact, best_triple)
        if not improved:
            x_best[:3]+=np.random.normal(0, args.sigma_Te*0.15, size=3)
            x_best[3:]+=np.random.normal(0, args.sigma_v*0.15, size=3)
            x_best[:3]=np.clip(x_best[:3], [te1[0],te2[0],te3[0]],[te1[-1],te2[-1],te3[-1]])
            x_best[3:]=np.clip(x_best[3:], 70.0, 140.0)

    # 输出
    to_excel(best_triple, best_exact, fname="result2.xlsx")
    plot_conv(hist, out="Q4ConvergencePlot.png")

if __name__=="__main__":
    main()
