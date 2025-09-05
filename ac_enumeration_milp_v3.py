# -*- coding: utf-8 -*-
"""
ac_enumeration_milp_v3.py
A + C：严格 3D 对齐 + 宽松枚举 + 窗口内三元组组合保底 + MILP/贪心全局选择
- 强化“非零保底”：每个时间窗内先把 M1/M2/M3 的候选做小范围组合，挑出能形成 z>0 的“三朵云同窗”组合；
- 全局交给 CP-SAT（若可用）或贪心；若仍返回 0，则直接输出“最佳本地组合”的方案（保证 z>0）。

用法示例（Windows 请给 offset-list 加引号或用等号）：
  python ac_enumeration_milp_v3.py --tmax 60 --dt 0.1 --windows 14,26,38,50,62 \
    --tau_list 0.9,1.0,1.2,1.5,2.0,2.5,2.8 \
    --lambda_list 0.1,0.2,0.35,0.5,0.65,0.8 \
    --offset_list="-60,-40,-20,0,20,40,60" \
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
    dx,dy,dz = _normalize((-x0,-y0,-z0))  # 指向原点的单位向量（题面如需改，可改为指向 (0,200,0)）
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

# ---------- 掩码 ----------
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

# ---------- 对齐/枚举候选 ----------
def generate_candidates_aligned(time_grid: np.ndarray,
                                windows: List[float],
                                tau_list: List[float],
                                tmax: float,
                                v_min: float=80.0, v_max: float=140.0,
                                min_seconds: float=0.04,
                                verbose:int=1) -> List[dict]:
    dt = float(time_grid[1]-time_grid[0])
    xT, yT = TARGET_CENTER_XY
    cands: List[dict] = []
    zT_list = [TARGET_Z0, 0.5*(TARGET_Z0+TARGET_Z1), TARGET_Z1]  # 0/5/10

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
                    for zT in zT_list:
                        denom = (zT - Mz)
                        if abs(denom) < 1e-9:
                            continue
                        lam = (zE - Mz) / denom
                        if lam < 0.0 or lam > 1.0:
                            continue
                        # Q 在 M(tc)->T(zT) 的直线上
                        Qx = Mx + lam*(xT - Mx)
                        Qy = My + lam*(yT - My)
                        hd = math.atan2(Qy - y0, Qx - x0)
                        dist_xy = math.hypot(Qx - x0, Qy - y0)
                        v = dist_xy / tc if tc > 1e-9 else 1e9
                        if not (v_min <= v <= v_max):
                            continue
                        mask = event_mask(uid, mid, v, hd, t_drop, tau, time_grid)
                        secs = float(mask.sum())*dt
                        if secs >= min_seconds:
                            cands.append({
                                "uid": uid, "mid": mid, "v": v, "hd": hd,
                                "t_drop": t_drop, "tau": tau, "mask": mask,
                                "src": "aligned", "tc": tc, "zT": zT, "lam": lam
                            })
            if verbose:
                print(f"[Aligned] tc={tc:.2f}s {mid}: candidates so far {len(cands)}")
    if verbose:
        print(f"[Aligned] total aligned candidates: {len(cands)}")
    return cands

def generate_candidates_relaxed(time_grid: np.ndarray,
                                windows: List[float],
                                lambda_list: List[float],
                                offset_list: List[float],
                                tau_list: List[float],
                                tmax: float,
                                v_min: float=80.0, v_max: float=140.0,
                                min_seconds: float=0.02,
                                verbose:int=1) -> List[dict]:
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
                            if not (v_min <= v <= v_max):
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

# ---------- 窗口内三元组组合（保底核心） ----------
def best_triple_per_window(cands: List[dict], time_grid: np.ndarray, tc: float,
                           topk_per_missile: int = 8, require_distinct_uav: bool = True):
    """
    在给定 tc 内，为每个导弹取覆盖秒数 Top-K 的候选，枚举三元组（必要时放宽不同 UAV），
    用按位与评估 z 秒数，返回最好的一组（三朵云）和 z 值。
    """
    dt = float(time_grid[1]-time_grid[0])
    groups = {m: [] for m in ALL_MISSILES}
    for c in cands:
        if abs(c.get("tc", tc) - tc) < 1e-6:  # 属于该窗
            groups[c["mid"]].append(c)
    for m in ALL_MISSILES:
        groups[m].sort(key=lambda c: int(c["mask"].sum()), reverse=True)
        groups[m] = groups[m][:topk_per_missile]

    if any(len(groups[m]) == 0 for m in ALL_MISSILES):
        return None, 0.0

    best = None; best_z = 0.0
    A, B, C = groups["M1"], groups["M2"], groups["M3"]
    for a in A:
        for b in B:
            if require_distinct_uav and a["uid"] == b["uid"]:
                continue
            for c in C:
                if require_distinct_uav and (c["uid"] in (a["uid"], b["uid"])):
                    continue
                zmask = (a["mask"] & b["mask"] & c["mask"])
                zsec = float(zmask.sum()) * dt
                if zsec > best_z:
                    best_z = zsec
                    best = (a, b, c)
    # 如果严格不同 UAV 找不到 z>0，放宽 UAV 再试一次
    if best is None:
        return best_triple_per_window(cands, time_grid, tc, topk_per_missile, require_distinct_uav=False)
    return best, best_z

def build_window_triples(cands: List[dict], time_grid: np.ndarray, windows: List[float],
                         topk_per_missile: int = 8):
    triples = []
    for tc in windows:
        triple, z = best_triple_per_window(cands, time_grid, tc, topk_per_missile)
        if triple is not None and z > 0.0:
            triples.append({"tc": tc, "triple": triple, "z": z})
    return triples

# ---------- 主问题：MILP（优先）/ 贪心（回退） ----------
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
    solver.parameters.max_time_in_seconds = 30.0
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
def export_solution(sol: dict, time_grid: np.ndarray, outdir: str, stem: str="ac_plan_v3"):
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
            "Source": c.get("src","-"), "tc": c.get("tc","-"),
            "lambda": c.get("lam","-"), "offset": c.get("d","-"), "zT": c.get("zT","-")
        })
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, f"{stem}.xlsx")
    try:
        df.to_excel(xlsx_path, index=False)
        written = xlsx_path
    except Exception:
        csv_path = os.path.join(outdir, f"{stem}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written = csv_path

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for c in sol["chosen"]:
        yM[c["mid"]] = np.maximum(yM[c["mid"]], c["mask"])
    z = (yM["M1"] & yM["M2"] & yM["M3"])
    z_seconds = float(z.sum())*dt

    js = {"objective_seconds": z_seconds, "chosen_events": len(sol["chosen"]), "solver": sol.get("how","auto")}
    js_path = os.path.join(outdir, f"{stem}.json")
    Path(js_path).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written, js_path, z_seconds

# ---------- 主流程 ----------
def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--windows", type=str, default="14,26,38,50,62")
    ap.add_argument("--tau_list", type=str, default="0.9,1.0,1.2,1.5,2.0,2.5,2.8")
    ap.add_argument("--lambda_list", type=str, default="0.1,0.2,0.35,0.5,0.65,0.8")
    ap.add_argument("--offset_list", type=str, default="-60,-40,-20,0,20,40,60")
    ap.add_argument("--min_seconds_aligned", type=float, default=0.04)
    ap.add_argument("--min_seconds_relaxed", type=float, default=0.02)
    ap.add_argument("--topk_per_missile", type=int, default=8, help="窗口内组合时每个导弹保留的TopK候选")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    time_grid = np.arange(0.0, args.tmax + 1e-9, args.dt)
    wins      = parse_floats_list(args.windows)
    taus      = parse_floats_list(args.tau_list)
    lambdas   = parse_floats_list(args.lambda_list)
    offsets   = parse_floats_list(args.offset_list)

    print("== A + C v3 (aligned + relaxed + window triples + MILP) ==")
    print(f"tmax={args.tmax}, dt={args.dt}, windows={wins}")
    print(f"taus={taus}, lambdas={lambdas}, offsets={offsets}")
    print("[Step 1] Enumerating aligned candidates ...")
    cands = generate_candidates_aligned(
        time_grid, wins, taus, tmax=args.tmax,
        min_seconds=args.min_seconds_aligned, verbose=args.verbose
    )
    print("[Step 2] Enumerating relaxed candidates ...")
    cands += generate_candidates_relaxed(
        time_grid, wins, lambdas, offsets, taus, tmax=args.tmax,
        min_seconds=args.min_seconds_relaxed, verbose=args.verbose
    )
    print(f"[Build] total candidates = {len(cands)}")

    # 窗口内先做三元组组合，保证有 z>0 的三朵云同窗
    print("[Step 3] Building window triples (local combos) ...")
    triples = build_window_triples(cands, time_grid, wins, topk_per_missile=args.topk_per_missile)
    triples.sort(key=lambda x: x["z"], reverse=True)
    if len(triples) == 0:
        print("[Warn] 未找到任何同窗 z>0 的三元组组合；建议增加 windows/taus/offsets 或减小 dt。")
    else:
        best_tc = triples[0]["tc"]; best_z = triples[0]["z"]
        print(f"[Triples] best window={best_tc}, triple-z≈{best_z:.3f}s; total triples={len(triples)}")

    # 交给主问题
    print("[Step 4] Solving global master (MILP or greedy) ...")
    sol, how = master_solve(cands, time_grid, max_per_uav=3, verbose=args.verbose)
    sol["how"] = how
    print(f"[Final] solver={how}, z≈{sol['score']:.3f}s, chosen={len(sol['chosen'])}")

    # 若仍为 0，则回退到“最佳本地组合”
    if sol["score"] <= 1e-9 and len(triples) > 0:
        print("[Fallback] Global solver got z≈0; using best local triple combo as final plan.")
        chosen = list(triples[0]["triple"])
        sol = {"chosen": chosen, "score": triples[0]["z"], "how": "local_triple_fallback"}

    # 导出
    written, js, z = export_solution(sol, time_grid, args.outdir, stem="ac_plan_v3")
    print(f"Exported:\n  plan_table: {written}\n  summary:    {js}\n  z≈{z:.3f} s")

if __name__ == "__main__":
    main()
