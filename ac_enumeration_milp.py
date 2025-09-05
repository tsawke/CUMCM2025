# -*- coding: utf-8 -*-
"""
ac_enumeration_milp.py
A + C：几何热区枚举 + MILP 全局选择（无 OR-Tools 时回退贪心）。
- 按 (导弹->目标) 的水平连线 + 垂直偏移，枚举爆点 (xE,yE)；
- 枚举 UAV & 引信 tau，反解速度/航向/投放时刻，过滤不可达/超界；
- 硬判定生成掩码 F(s,t)，进入 CP-SAT（或贪心）选择，最大化三导弹同时遮蔽总秒数；
- 输出 Excel/CSV + JSON，带详细进度打印。

用法：
  pip install numpy pandas
  # 可选：更优解器 & 写 Excel
  pip install ortools openpyxl
  python ac_enumeration_milp.py --tmax 60 --dt 0.2 --windows 14,26,38,50,62 --outdir .

建议先用默认参数跑通，再加密 dt 或扩大枚举以抬分。
"""
from __future__ import annotations
import os, math, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# -------------------- 题面常量（可按题调） --------------------
g = 9.8
CLOUD_RADIUS = 10.0
CLOUD_ACTIVE = 20.0
CLOUD_SINK = 3.0
MISSILE_SPEED = 300.0

TARGET_CENTER_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0

# 导弹初始（示例符合之前脚本）
M_INIT = {
    "M1": (20000.0,    0.0, 2000.0),
    "M2": (19000.0,  600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}
# 五架 UAV 初始
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

# -------------------- 数学/几何工具 --------------------
def _normalize(v):
    x,y,z = v
    n = math.hypot(x, math.hypot(y,z))
    return (x/n, y/n, z/n) if n>0 else (0.0,0.0,0.0)

def missile_pos(mid: str, t: float):
    x0,y0,z0 = M_INIT[mid]
    dx,dy,dz = _normalize((-x0,-y0,-z0))
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_xy(uid: str, v: float, hd: float, t: float):
    x0,y0,_ = FY_INIT[uid]
    return (x0 + v*t*math.cos(hd),
            y0 + v*t*math.sin(hd))

def point_to_segment_dist(P, A, B) -> float:
    """点 P 到线段 AB 的距离"""
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
    """硬判定：云心到“导弹->目标条”的最小距离 ≤ 半径"""
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
    """时刻 t 的云心；不活跃返回 None"""
    if t < t_exp or t > t_exp + CLOUD_ACTIVE:
        return None
    xe,ye,ze = explode_xyz
    return (xe, ye, ze - CLOUD_SINK*(t - t_exp))

# -------------------- IO 工具 --------------------
def save_table(df: pd.DataFrame, path_xlsx: str):
    """优先写 Excel；若无 openpyxl 则写 CSV"""
    try:
        df.to_excel(path_xlsx, index=False)
        return path_xlsx
    except Exception:
        csv_path = os.path.splitext(path_xlsx)[0] + ".csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

# -------------------- 候选生成（核心：几何热区 + 反解） --------------------
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

def generate_candidates(time_grid: np.ndarray,
                        windows: List[float],
                        lambda_list: List[float],
                        offset_list: List[float],
                        tau_list: List[float],
                        v_min: float=80.0, v_max: float=140.0,
                        min_seconds: float=0.02,
                        tmax: float=60.0,
                        verbose:int=1) -> List[dict]:
    """
    在每个窗 tc、每枚导弹 mid：
      - 沿 M(tc)->Target 水平线取 λ 点；
      - 在水平法向方向加偏移 d；
      - 对每个 UAV、每个 tau，反解 v、hd、t_drop；
      - 若可行则生成 mask（硬判定），mask 秒数≥阈值就加入候选。
    """
    dt = float(time_grid[1]-time_grid[0])
    cands: List[dict] = []
    xT,yT = TARGET_CENTER_XY

    for tc in windows:
        for mid in ALL_MISSILES:
            Mx,My,Mz = missile_pos(mid, tc)
            # 水平连线向量（朝向目标）
            dx = xT - Mx; dy = yT - My
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx/L, dy/L  # 单位切向
            nx, ny = -uy, ux     # 单位法向（左手）

            if verbose:
                print(f"[Enum] tc={tc:.1f}s {mid}: segment length≈{L:.1f} m, try {len(lambda_list)}×{len(offset_list)} grid")

            for lam in lambda_list:
                # 沿线点
                px = Mx + lam*dx
                py = My + lam*dy
                # 目标高度用中值代表（可改进为多 z）
                pz = 0.5*(TARGET_Z0 + TARGET_Z1)
                for d in offset_list:
                    xE = px + d*nx
                    yE = py + d*ny
                    # 反解：对每个 UAV、每个 tau，构造候选
                    for uid in ALL_UAVS:
                        x0,y0,z0 = FY_INIT[uid]
                        dist_xy = math.hypot(xE - x0, yE - y0)
                        if tc <= 0.05:  # 起爆接近 0 不现实
                            continue
                        # 航向
                        hd = math.atan2(yE - y0, xE - x0)
                        for tau in tau_list:
                            # 引信决定起爆高度：zE = z0 - 0.5 g tau^2
                            zE = z0 - 0.5*g*(tau**2)
                            # 与代表高度 pz 的差不用于可行性，但用于后面 mask 真实判断
                            # 投放时间与速度
                            t_drop = tc - tau
                            if t_drop < 0.0 or tc > tmax:
                                continue
                            v = dist_xy / tc
                            if v < v_min or v > v_max:
                                continue
                            # 生成 mask（硬判定）
                            mask = event_mask(uid, mid, v, hd, t_drop, tau, time_grid)
                            secs = float(mask.sum())*dt
                            if secs >= min_seconds:
                                cands.append({
                                    "uid": uid, "mid": mid, "v": v, "hd": hd,
                                    "t_drop": t_drop, "tau": tau, "mask": mask,
                                    "src": "enum", "tc": tc, "lam": lam, "d": d
                                })
    if verbose:
        print(f"[Enum] total candidates: {len(cands)}")
    return cands

# -------------------- 主问题：MILP（优先）/ 贪心（回退） --------------------
def master_greedy(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3, verbose:int=1) -> dict:
    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    chosen = []; used = {u:0 for u in ALL_UAVS}; lastt = {u:-1e9 for u in ALL_UAVS}
    yM = {m:np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}

    def zscore():
        return float((yM["M1"] & yM["M2"] & yM["M3"]).sum())*dt

    while True:
        base = zscore(); best = None
        for c in cands:
            u = c['uid']
            if used[u] >= max_per_uav: continue
            if c['t_drop'] - lastt[u] < 1.0 - 1e-9: continue
            ytmp = yM[c['mid']].copy()
            ytmp = np.maximum(ytmp, c['mask'])
            znew = ((yM["M1"] if c['mid']!="M1" else ytmp) &
                    (yM["M2"] if c['mid']!="M2" else ytmp) &
                    (yM["M3"] if c['mid']!="M3" else ytmp))
            inc = float(znew.sum())*dt - base
            if best is None or inc > best[0]:
                best = (inc, c)
        if best is None or best[0] <= 1e-9:
            break
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
        if verbose:
            print("[MILP] OR-Tools not found, fallback to greedy.")
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav, verbose=verbose), "greedy"

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    model = cp_model.CpModel()

    x = [model.NewBoolVar(f"x{i}") for i in range(len(cands))]
    y = [[model.NewBoolVar(f"y{m}{t}") for t in range(N)] for m in range(3)]
    z = [model.NewBoolVar(f"z{t}") for t in range(N)]

    # link y: y[m,t] ≤ sum_{s covers (m,t)} x_s
    for m in range(3):
        for t in range(N):
            inv = [x[i] for i,c in enumerate(cands) if M2IDX[c['mid']]==m and c['mask'][t]==1]
            if inv:
                model.Add(y[m][t] <= sum(inv))
            else:
                model.Add(y[m][t] == 0)
    # AND: z[t] ≤ y[m,t] (对每个 m)
    for t in range(N):
        for m in range(3):
            model.Add(z[t] <= y[m][t])

    # 每 UAV ≤ 3；同 UAV 间隔 ≥1s（仅近邻互斥）
    for uid in ALL_UAVS:
        model.Add(sum(x[i] for i,c in enumerate(cands) if c['uid']==uid) <= max_per_uav)
        lst = [(i,c) for i,c in enumerate(cands) if c['uid']==uid]
        lst.sort(key=lambda ic: ic[1]['t_drop'])
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                if abs(lst[i][1]['t_drop'] - lst[j][1]['t_drop']) < 1.0 - 1e-9:
                    model.Add(x[lst[i][0]] + x[lst[j][0]] <= 1)

    model.Maximize(sum(z))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 25.0
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if verbose:
            print("[MILP] infeasible or timeout, fallback to greedy.")
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav, verbose=verbose), "greedy"

    chosen = [cands[i] for i in range(len(cands)) if solver.Value(x[i])==1]
    score  = sum(solver.Value(z[t]) for t in range(N))*dt
    if verbose:
        print(f"[MILP] chosen={len(chosen)}, z≈{score:.3f}s")
    return {"chosen":chosen, "score":float(score)}, "milp"

# -------------------- 导出 --------------------
def export_solution(sol: dict, time_grid: np.ndarray, outdir: str):
    rows = []
    for c in sol['chosen']:
        tE = c["t_drop"] + c["tau"]
        xE,yE = uav_xy(c["uid"], c["v"], c["hd"], tE)
        z0 = FY_INIT[c["uid"]][2]; zE = z0 - 0.5*g*(c["tau"]**2)
        xD,yD = uav_xy(c["uid"], c["v"], c["hd"], c["t_drop"])
        rows.append({
            "UAV": c["uid"], "Missile": c["mid"], "Speed(m/s)": round(c["v"],1),
            "Heading(rad)": round(c["hd"],6), "DropTime(s)": round(c["t_drop"] ,2),
            "ExplodeTime(s)": round(tE ,2),
            "DropX": round(xD,2), "DropY": round(yD,2), "DropZ": round(z0,2),
            "ExplodeX": round(xE,2), "ExplodeY": round(yE,2), "ExplodeZ": round(zE,2),
            "Source": c["src"], "tc": c.get("tc","-"), "lambda": c.get("lam","-"), "offset": c.get("d","-")
        })
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, "ac_plan.xlsx")
    written_path = save_table(df, xlsx_path)

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for c in sol["chosen"]:
        yM[c["mid"]] = np.maximum(yM[c["mid"]], c["mask"])
    z = (yM["M1"] & yM["M2"] & yM["M3"])
    z_seconds = float(z.sum())*dt

    js = {"objective_seconds": z_seconds, "chosen_events": len(sol["chosen"]), "solver": sol.get("how","auto")}
    js_path = os.path.join(outdir, "ac_summary.json")
    Path(js_path).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written_path, js_path, z_seconds

# -------------------- CLI --------------------
def parse_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=60.0, help="总时长 s")
    ap.add_argument("--dt", type=float, default=0.2, help="时间步 s")
    ap.add_argument("--windows", type=str, default="14,26,38,50", help="爆点时刻候选（逗号分隔）")
    ap.add_argument("--lambda_list", type=str, default="0.1,0.2,0.35,0.5,0.65,0.8", help="沿线比例（靠导弹→靠目标）")
    ap.add_argument("--offset_list", type=str, default="-40,-20,0,20,40", help="法向偏移（米）")
    ap.add_argument("--tau_list", type=str, default="1.0,1.5,2.0,2.5", help="引信候选（秒）")
    ap.add_argument("--min_seconds", type=float, default=0.02, help="单候选最小覆盖秒数阈值")
    ap.add_argument("--outdir", type=str, default=".", help="输出目录")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    time_grid = np.arange(0.0, args.tmax + 1e-9, args.dt)
    wins = parse_floats_list(args.windows)
    lambdas = parse_floats_list(args.lambda_list)
    offsets = parse_floats_list(args.offset_list)
    taus = parse_floats_list(args.tau_list)

    print("== A + C (Enumeration + MILP) ==")
    print(f"tmax={args.tmax}, dt={args.dt}, windows={wins}")
    print(f"λ-list={lambdas}, offset-list={offsets}, tau-list={taus}")
    # 生成候选
    cands = generate_candidates(time_grid, wins, lambdas, offsets, taus,
                                min_seconds=args.min_seconds, tmax=args.tmax, verbose=args.verbose)
    if len(cands) == 0:
        print("[Warn] 无候选；请增大搜索（更多 λ/offset/tau 或更密时间窗），或放宽 min_seconds。")
        # 仍导出空表与 0 结果
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

#python ac_enumeration_milp.py --tmax 60 --dt 0.1 --windows 14,26,38,50,62 --lambda_list 0.05,0.1,0.2,0.35,0.5,0.65,0.8,0.9 --offset_list '-80,-60,-40,-20,0,20,40,60,80' --tau_list 0.9,1.0,1.2,1.5,2.0,2.5,2.8 --outdir . 
