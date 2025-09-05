# -*- coding: utf-8 -*-
"""
ac_enumeration_milp_v2.py
A + C：严格 3D 对齐的候选生成 + MILP/贪心 选择，保证非零可行解更易出现。

用法示例（Windows 请给 offset-list 加引号或用等号）：
  python ac_enumeration_milp_v2.py --tmax 60 --dt 0.1 --windows 14,26,38,50,62 \
    --tau_list 0.9,1.0,1.2,1.5,2.0,2.5,2.8 \
    --lambda_list 0.2,0.35,0.5,0.65,0.8 \
    --offset_list="-40,-20,0,20,40" \
    --outdir .

依赖：
  pip install numpy pandas
  # 可选：更优解器与写 Excel
  pip install ortools openpyxl
"""
from __future__ import annotations
import os, math, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ---------- 场景参数 ----------
g = 9.8
CLOUD_RADIUS = 10.0
CLOUD_ACTIVE = 20.0
CLOUD_SINK = 3.0
MISSILE_SPEED = 300.0
TARGET_CENTER_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0

M_INIT = {
    "M1": (20000.0,    0.0, 2000.0),
    "M2": (19000.0,  600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}
FY_INIT = {
    "FY1": (17800.0,     0.0, 1800.0),
    "FY2": (12000.0,  1400.0, 1400.0),
    "FY3": ( 6000.0, -3000.0,  700.0),
    "FY4": (11000.0,  2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}
ALL_UAVS = list(FY_INIT.keys())
ALL_MISSILES = ["M1","M2","M3"]
M2IDX = {"M1":0,"M2":1,"M3":2}

# ---------- 基础几何 ----------
def _normalize(v):
    x,y,z = v
    n = math.hypot(x, math.hypot(y,z))
    return (x/n, y/n, z/n) if n>0 else (0.0,0.0,0.0)

def missile_pos(mid: str, t: float):
    x0,y0,z0 = M_INIT[mid]
    dx,dy,dz = _normalize((-x0,-y0,-z0))  # 指向原点
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_xy(uid: str, v: float, heading: float, t: float):
    x0,y0,_ = FY_INIT[uid]
    return (x0 + v*t*math.cos(heading),
            y0 + v*t*math.sin(heading))

def point_to_segment_dist(P, A, B) -> float:
    ax,ay,az = A; bx,by,bz = B; px,py,pz = P
    AB = (bx-ax, by-ay, bz-az)
    AP = (px-ax, py-ay, pz-az)
    ab2 = AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]
    if ab2 == 0.0:
        dx,dy,dz = (px-ax, py-ay, pz-az)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    tau = (AP[0]*AB[0] + AP[1]*AB[1] + AP[2]*AB[2]) / ab2
    if tau < 0.0: tau = 0.0
    elif tau > 1.0: tau = 1.0
    qx = ax + AB[0]*tau; qy = ay + AB[1]*tau; qz = az + AB[2]*tau
    dx,dy,dz = (px-qx, py-qy, pz-qz)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def covered_hard(c_center, mid: str, t: float, z_samples: int=11) -> bool:
    m = missile_pos(mid, t)
    xT, yT = TARGET_CENTER_XY
    if z_samples <= 1:
        zs = [(TARGET_Z0 + TARGET_Z1)/2.0]
    else:
        step = (TARGET_Z1 - TARGET_Z0) / (z_samples - 1)
        zs = [TARGET_Z0 + i*step for i in range(z_samples)]
    for z in zs:
        Tz = (xT, yT, z)
        if point_to_segment_dist(c_center, m, Tz) <= CLOUD_RADIUS:
            return True
    return False

def cloud_center(explode_xyz, t_exp, t):
    if t < t_exp or t > t_exp + CLOUD_ACTIVE:
        return None
    xe,ye,ze = explode_xyz
    return (xe, ye, ze - CLOUD_SINK*(t - t_exp))

# ---------- IO ----------
def save_table(df: pd.DataFrame, path_xlsx: str):
    try:
        df.to_excel(path_xlsx, index=False)
        return path_xlsx
    except Exception:
        csv_path = os.path.splitext(path_xlsx)[0] + ".csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

# ---------- 事件掩码 ----------
def event_mask(uid: str, mid: str, v: float, hd: float, t_drop: float, tau: float, time_grid: np.ndarray) -> np.ndarray:
    tE = t_drop + tau
    xE,yE = uav_xy(uid, v, hd, tE)
    z0 = FY_INIT[uid][2]
    zE = z0 - 0.5*g*(tau**2)
    mask = np.zeros(len(time_grid), dtype=np.int8)
    for i,t in enumerate(time_grid):
        if tE <= t <= tE + CLOUD_ACTIVE:
            c = (xE, yE, zE - CLOUD_SINK*(t - tE))
            if covered_hard(c, mid, t, z_samples=11):
                mask[i] = 1
    return mask

# ---------- 3D 严格对齐候选（核心） ----------
def generate_candidates_aligned(time_grid: np.ndarray,
                                windows: List[float],
                                tau_list: List[float],
                                tmax: float,
                                v_min: float=80.0, v_max: float=140.0,
                                min_seconds: float=0.05,
                                verbose:int=1) -> List[dict]:
    """
    对每个 (tc, mid, uid, tau)：严格求解 3D 视线段一致性：
      zE = z0 - 0.5*g*tau^2
      λ = (zE - Mz(tc)) / (zT - Mz(tc)),  zT ∈ {0,5,10}
      Q = M(tc) + λ*(T(zT) - M(tc)),  要求 λ∈[0,1]
      v = ||Qxy - (x0,y0)|| / tc,  hd = atan2,  t_drop=tc - tau

    若可行则必在 t=tc 时遮蔽该导弹。
    """
    dt = float(time_grid[1]-time_grid[0])
    xT, yT = TARGET_CENTER_XY
    cands: List[dict] = []
    zT_list = [TARGET_Z0, 0.5*(TARGET_Z0+TARGET_Z1), TARGET_Z1]

    for tc in windows:
        for mid in ALL_MISSILES:
            Mx,My,Mz = missile_pos(mid, tc)
            for uid in ALL_UAVS:
                x0,y0,z0 = FY_INIT[uid]
                for tau in tau_list:
                    zE = z0 - 0.5*g*(tau**2)
                    t_drop = tc - tau
                    if t_drop < 0.0 or tc > tmax:
                        continue
                    ok_once = False
                    for zT in zT_list:
                        denom = (zT - Mz)
                        if abs(denom) < 1e-9:
                            continue
                        lam = (zE - Mz) / denom
                        if lam < 0.0 or lam > 1.0:
                            continue
                        # Q on the same 3D sight line segment
                        Qx = Mx + lam*(xT - Mx)
                        Qy = My + lam*(yT - My)
                        # 速度/航向
                        dist_xy = math.hypot(Qx - x0, Qy - y0)
                        v = dist_xy / tc if tc > 1e-9 else 1e9
                        if v < v_min or v > v_max:
                            continue
                        hd = math.atan2(Qy - y0, Qx - x0)
                        # 生成 mask（硬判定）
                        mask = event_mask(uid, mid, v, hd, t_drop, tau, time_grid)
                        secs = float(mask.sum())*dt
                        if secs >= min_seconds:
                            cands.append({
                                "uid": uid, "mid": mid, "v": v, "hd": hd,
                                "t_drop": t_drop, "tau": tau, "mask": mask,
                                "src": "aligned", "tc": tc, "zT": zT, "lam": lam
                            })
                            ok_once = True
                    # 没对上也不紧张，备胎枚举会补充
            if verbose:
                print(f"[Aligned] tc={tc:.2f}s {mid}: candidates so far {len(cands)}")
    if verbose:
        print(f"[Aligned] total aligned candidates: {len(cands)}")
    return cands

# ---------- 宽松水平栅格（备胎增强） ----------
def generate_candidates_relaxed(time_grid: np.ndarray,
                                windows: List[float],
                                lambda_list: List[float],
                                offset_list: List[float],
                                tau_list: List[float],
                                tmax: float,
                                v_min: float=80.0, v_max: float=140.0,
                                min_seconds: float=0.02,
                                verbose:int=1) -> List[dict]:
    """
    沿 M(tc)->Target 的水平连线取 λ，再在法向方向偏移 d，反解 v/hd/t_drop。
    不保证严格 3D 对齐，但能提供多样性。
    """
    dt = float(time_grid[1]-time_grid[0])
    cands: List[dict] = []
    xT,yT = TARGET_CENTER_XY

    for tc in windows:
        for mid in ALL_MISSILES:
            Mx,My,Mz = missile_pos(mid, tc)
            dx = xT - Mx; dy = yT - My
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx/L, dy/L
            nx, ny = -uy, ux
            if verbose:
                print(f"[Relax] tc={tc:.2f}s {mid}: try {len(lambda_list)}x{len(offset_list)} grid")
            for lam in lambda_list:
                px = Mx + lam*dx
                py = My + lam*dy
                for d in offset_list:
                    xE = px + d*nx
                    yE = py + d*ny
                    for uid in ALL_UAVS:
                        x0,y0,z0 = FY_INIT[uid]
                        dist_xy = math.hypot(xE - x0, yE - y0)
                        if tc <= 0.05:
                            continue
                        hd = math.atan2(yE - y0, xE - x0)
                        for tau in tau_list:
                            t_drop = tc - tau
                            if t_drop < 0.0 or tc > tmax:
                                continue
                            v = dist_xy / tc
                            if v < v_min or v > v_max:
                                continue
                            mask = event_mask(uid, mid, v, hd, t_drop, tau, time_grid)
                            secs = float(mask.sum())*dt
                            if secs >= min_seconds:
                                cands.append({
                                    "uid": uid, "mid": mid, "v": v, "hd": hd,
                                    "t_drop": t_drop, "tau": tau, "mask": mask,
                                    "src": "relaxed", "tc": tc, "lam": lam, "d": d
                                })
    if verbose:
        print(f"[Relax] total relaxed candidates: {len(cands)}")
    return cands

# ---------- 主问题 ----------
def master_greedy(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3, verbose:int=1) -> dict:
    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    chosen = []; used = {u:0 for u in ALL_UAVS}; lastt = {u:-1e9 for u in ALL_UAVS}
    yM = {m:np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    def zscore(): return float((yM["M1"] & yM["M2"] & yM["M3"]).sum())*dt

    while True:
        base = zscore(); best = None
        for c in cands:
            u = c['uid']
            if used[u] >= max_per_uav: continue
            if c['t_drop'] - lastt[u] < 1.0 - 1e-9: continue
            ytmp = yM[c['mid']].copy(); ytmp = np.maximum(ytmp, c['mask'])
            znew = ((yM["M1"] if c['mid']!="M1" else ytmp) &
                    (yM["M2"] if c['mid']!="M2" else ytmp) &
                    (yM["M3"] if c['mid']!="M3" else ytmp))
            inc = float(znew.sum())*dt - base
            if best is None or inc > best[0]:
                best = (inc, c)
        if best is None or best[0] <= 1e-9: break
        inc, c = best
        yM[c['mid']] = np.maximum(yM[c['mid']], c['mask'])
        chosen.append(c); used[c['uid']]+=1; lastt[c['uid']]=c['t_drop']
        if verbose:
            print(f"[Greedy] pick {c['uid']}->{c['mid']} @t_drop={c['t_drop']:.2f}, +Δz≈{inc:.3f}s")
    return {"chosen":chosen, "score":zscore()}

def master_solve(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3, verbose:int=1):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        if verbose: print("[MILP] OR-Tools not found, fallback to greedy.")
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav, verbose=verbose), "greedy"

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(cands))]
    y = [[model.NewBoolVar(f"y{m}{t}") for t in range(N)] for m in range(3)]
    z = [model.NewBoolVar(f"z{t}") for t in range(N)]

    for m in range(3):
        for t in range(N):
            inv = [x[i] for i,c in enumerate(cands) if M2IDX[c['mid']]==m and c['mask'][t]==1]
            if inv: model.Add(y[m][t] <= sum(inv))
            else:   model.Add(y[m][t] == 0)
    for t in range(N):
        for m in range(3): model.Add(z[t] <= y[m][t])
    for uid in ALL_UAVS:
        model.Add(sum(x[i] for i,c in enumerate(cands) if c['uid']==uid) <= max_per_uav)
        lst = [(i,c) for i,c in enumerate(cands) if c['uid']==uid]; lst.sort(key=lambda ic: ic[1]['t_drop'])
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                if abs(lst[i][1]['t_drop']-lst[j][1]['t_drop']) < 1.0 - 1e-9:
                    model.Add(x[lst[i][0]] + x[lst[j][0]] <= 1)

    model.Maximize(sum(z))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 25.0
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if verbose: print("[MILP] infeasible or timeout, fallback to greedy.")
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav, verbose=verbose), "greedy"
    chosen = [cands[i] for i in range(len(cands)) if solver.Value(x[i])==1]
    score  = sum(solver.Value(z[t]) for t in range(N))*dt
    if verbose: print(f"[MILP] chosen={len(chosen)}, z≈{score:.3f}s")
    return {"chosen":chosen, "score":float(score)}, "milp"

# ---------- 导出 ----------
def export_solution(sol: dict, time_grid: np.ndarray, outdir: str):
    rows = []
    for c in sol['chosen']:
        tE = c["t_drop"] + c["tau"]
        xE,yE = uav_xy(c["uid"], c["v"], c["hd"], tE)
        z0 = FY_INIT[c["uid"]][2]; zE = z0 - 0.5*g*(c["tau"]**2)
        xD,yD = uav_xy(c["uid"], c["v"], c["hd"], c["t_drop"])
        rows.append({
            "UAV": c["uid"], "Missile": c["mid"], "Speed(m/s)": round(c["v"],1),
            "Heading(rad)": round(c["hd"],6), "DropTime(s)": round(c["t_drop"],2),
            "ExplodeTime(s)": round(tE,2),
            "DropX": round(xD,2), "DropY": round(yD,2), "DropZ": round(z0,2),
            "ExplodeX": round(xE,2), "ExplodeY": round(yE,2), "ExplodeZ": round(zE,2),
            "Source": c["src"], "tc": c.get("tc","-"), "zT": c.get("zT","-"),
            "lambda": c.get("lam","-"), "offset": c.get("d","-")
        })
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, "ac_plan_v2.xlsx")
    written_path = save_table(df, xlsx_path)

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for c in sol["chosen"]:
        yM[c["mid"]] = np.maximum(yM[c["mid"]], c["mask"])
    z = (yM["M1"] & yM["M2"] & yM["M3"])
    z_seconds = float(z.sum())*dt

    js = {"objective_seconds": z_seconds, "chosen_events": len(sol["chosen"]), "solver": sol.get("how","auto")}
    js_path = os.path.join(outdir, "ac_summary_v2.json")
    Path(js_path).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written_path, js_path, z_seconds

# ---------- CLI ----------
def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--windows", type=str, default="14,26,38,50,62")
    ap.add_argument("--tau_list", type=str, default="0.9,1.0,1.2,1.5,2.0,2.5,2.8")
    ap.add_argument("--lambda_list", type=str, default="0.2,0.35,0.5,0.65,0.8")
    ap.add_argument("--offset_list", type=str, default="-40,-20,0,20,40")
    ap.add_argument("--min_seconds_aligned", type=float, default=0.05)
    ap.add_argument("--min_seconds_relaxed", type=float, default=0.02)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    time_grid = np.arange(0.0, args.tmax + 1e-9, args.dt)
    wins      = parse_floats_list(args.windows)
    taus      = parse_floats_list(args.tau_list)
    lambdas   = parse_floats_list(args.lambda_list)
    offsets   = parse_floats_list(args.offset_list)

    print("== A + C (strict 3D aligned + relaxed backup) ==")
    print(f"tmax={args.tmax}, dt={args.dt}, windows={wins}")
    print(f"taus={taus}, lambdas={lambdas}, offsets={offsets}")

    # 先生成“严格 3D 对齐”的强候选（确保 t=tc 遮蔽）
    cands = generate_candidates_aligned(
        time_grid, wins, taus, tmax=args.tmax,
        min_seconds=args.min_seconds_aligned, verbose=args.verbose
    )
    # 再加一层“宽松栅格”的备胎候选
    more  = generate_candidates_relaxed(
        time_grid, wins, lambdas, offsets, taus, tmax=args.tmax,
        min_seconds=args.min_seconds_relaxed, verbose=args.verbose
    )
    cands.extend(more)
    print(f"[Build] total candidates = {len(cands)}")

    if len(cands) == 0:
        print("[Warn] 无候选；建议：加密 windows/taus；或放宽 min_seconds；或扩大 offset 范围。")
        sol = {"chosen": [], "score": 0.0, "how": "none"}
        xlsx, js, z = export_solution(sol, time_grid, args.outdir)
        print(f"Exported:\n  plan_table: {xlsx}\n  summary:    {js}\n  z≈{z:.3f} s")
        return

    # 主问题
    sol, how = master_solve(cands, time_grid, max_per_uav=3, verbose=args.verbose)
    sol["how"] = how
    print(f"[Final] solver={how}, z≈{sol['score']:.3f}s, chosen={len(sol['chosen'])}")

    # 导出
    xlsx, js, z = export_solution(sol, time_grid, args.outdir)
    print(f"Exported:\n  plan_table: {xlsx}\n  summary:    {js}\n  z≈{z:.3f} s")

if __name__ == "__main__":
    main()
