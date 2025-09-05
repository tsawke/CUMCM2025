# -*- coding: utf-8 -*-
"""
pso_colgen_v2.py
列生成 + 粒子群定价（PSO），带“解析对齐”初始候选、自举权重、详尽进度打印。
改进点：
- 初始候选新增“解析对齐”：把起爆点严格放到 M(tc)->Target 的3D线段上，确保 t=tc 时遮蔽该导弹。
- PSO 搜索空间加宽（绝对航向、t_drop 在 tc±5s）。
- 更稳的数值（稳定 sigmoid），更积极的阈值（避免空解）。
- 详尽进度输出（每轮、每窗×导弹、PSO阶段）。
依赖：
  必需：numpy pandas
  可选：ortools（更优主问题）、openpyxl（写 Excel）
用法示例：
  python pso_colgen_v2.py --tmax 60 --dt 0.2 --windows 14,26,38,50 --rounds 4 --swarm 50 --iters 60 --verbose 1 --outdir .
"""
from __future__ import annotations
import os, math, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# -------------------- 场景常量（按题面如需可微调） --------------------
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

# -------------------- 几何基础 --------------------
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

def uav_xy(uid: str, v: float, heading: float, t: float):
    x0,y0,_ = FY_INIT[uid]
    return (x0 + v*t*math.cos(heading),
            y0 + v*t*math.sin(heading))

def point_to_segment_dist(P, A, B) -> float:
    ax,ay,az = A; bx,by,bz = B; px,py,pz = P
    AB = (bx-ax, by-ay, bz-bz+az-az)  # placeholder to keep line length?
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

# -------------------- 工具：保存 Excel/CSV --------------------
def save_table(df: pd.DataFrame, path_xlsx: str):
    try:
        df.to_excel(path_xlsx, index=False)
        return path_xlsx
    except Exception:
        csv_path = os.path.splitext(path_xlsx)[0] + ".csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

# -------------------- 掩码与候选 --------------------
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

# -------------------- 主问题：MILP（优先）/ 贪心（回退） --------------------
def master_greedy(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3) -> dict:
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
            if best is None or inc > best[0]: best = (inc, c)
        if best is None or best[0] <= 1e-9: break
        inc, c = best
        yM[c['mid']] = np.maximum(yM[c['mid']], c['mask'])
        chosen.append(c); used[c['uid']]+=1; lastt[c['uid']]=c['t_drop']
    return {"chosen":chosen, "score":zscore()}

def master_solve(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav), "greedy"
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
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav), "greedy"
    chosen = [cands[i] for i in range(len(cands)) if solver.Value(x[i])==1]
    score  = sum(solver.Value(z[t]) for t in range(N))*dt
    return {"chosen":chosen, "score":float(score)}, "milp"

# -------------------- 粒子群定价器（PSO 对单朵云） --------------------
def stable_sigmoid(x): return 0.5*(1.0 + np.tanh(0.5*x))

def price_pso_for_missile(mid: str, t_center: float, time_grid: np.ndarray,
                          yM: Dict[str,np.ndarray], W: np.ndarray,
                          swarm: int=40, iters: int=50, seed: int=0, verbose:int=0):
    rng = np.random.RandomState(seed + int(t_center*1000))
    N = len(time_grid); dt = float(time_grid[1]-time_grid[0])
    # 变量：u_logits(5), hd_raw(1), v_raw(1), tD_raw(1), tau_raw(1)  → 总 9 维
    D = 9
    X = rng.randn(swarm, D)*0.1
    V = np.zeros((swarm, D))
    w0, w1 = 0.9, 0.4
    c1, c2 = 1.8, 1.8
    vclip = np.array([0.6]*5 + [0.8, 0.8, 1.2, 0.8], dtype=float)

    def decode(xrow):
        lo = xrow[:5]
        pu = np.exp(lo - np.max(lo)); pu /= pu.sum()
        uidx = int(np.argmax(pu)); uid = ALL_UAVS[uidx]
        # 绝对航向：(-pi, pi)
        hd = (2.0*math.pi*stable_sigmoid(xrow[5])) - math.pi
        v  = 80.0 + 60.0*stable_sigmoid(xrow[6])
        # t_drop 在 [tc-5, tc+5]，再截断到 [0, tmax-0.8]在上层做
        tD = (t_center - 5.0) + 10.0*stable_sigmoid(xrow[7])
        tau = 0.8 + 2.2*stable_sigmoid(xrow[8])
        return uid, v, hd, tD, tau

    def fitness(xrow):
        uid, v, hd, tD, tau = decode(xrow)
        # 边际（硬）：只在 mid 未被覆盖且 W=1 时有分
        mask = event_mask(uid, mid, v, hd, tD, tau, time_grid)
        margin = (1 - yM[mid]) * mask * W
        return float(margin.sum())*dt, (uid, v, hd, tD, tau)

    pbest_val = np.full(swarm, -1e18, dtype=float); pbest_x = X.copy(); pbest_aux = [None]*swarm
    gbest_val = -1e18; gbest_x = None; gbest_aux = None

    for it in range(iters):
        w = w0 + (w1 - w0)*it/max(1, iters-1)
        for i in range(swarm):
            val, aux = fitness(X[i])
            if val > pbest_val[i]:
                pbest_val[i] = val; pbest_x[i] = X[i].copy(); pbest_aux[i] = aux
            if val > gbest_val:
                gbest_val = val; gbest_x = X[i].copy(); gbest_aux = aux
        if verbose >= 2 and (it % 10 == 0 or it == iters-1):
            print(f"      PSO iter {it+1}/{iters}: gbest_gain≈{gbest_val:.4f}s")
        r1 = rng.rand(swarm, D); r2 = rng.rand(swarm, D)
        V = w*V + c1*r1*(pbest_x - X) + c2*r2*(gbest_x - X)
        V = np.clip(V, -vclip, vclip)
        X = X + V

    if gbest_aux is None:
        uid = ALL_UAVS[0]; v=120.0; tau=2.0; tD=max(0.0, t_center-2.0); hd=0.0
        return {"uid":uid,"mid":mid,"v":v,"hd":hd,"t_drop":tD,"tau":tau}, 0.0
    uid, v, hd, tD, tau = gbest_aux
    return {"uid":uid,"mid":mid,"v":float(v),"hd":float(hd),"t_drop":float(tD),"tau":float(tau)}, float(gbest_val)

# -------------------- 解析对齐：构造强初始列 --------------------
def seed_analytic_aligned(cands: List[dict], time_grid: np.ndarray, tmax: float, dt: float,
                          windows: List[float], verbose:int=0):
    """
    在每个 tc、每枚导弹，沿 M(tc)->Target 线段按 λ 取点 Qλ，
    令起爆点=Qλ，反解(UID, v, hd, t_drop, tau)：
      v = ||Qxy - UAV_xy0|| / tE
      hd = atan2(Qy - y0, Qx - x0)
      tau = sqrt( 2*(z0 - Qz) / g )    （若越界则跳过）
      t_drop = tE - tau
    这样能保证 t=tE 时对该导弹一定遮蔽（几何上点落在视线段上）。
    """
    xT,yT = TARGET_CENTER_XY
    for tc in windows:
        for mid in ALL_MISSILES:
            Mx,My,Mz = missile_pos(mid, tc)
            # 目标高度取中值以代表（也可多取几个 z ）
            Tz = 0.5*(TARGET_Z0 + TARGET_Z1)
            # 线段端点
            Ax,Ay,Az = Mx,My,Mz
            Bx,By,Bz = xT,yT,Tz
            for lam in (0.05, 0.1, 0.2, 0.4, 0.7):  # 从靠近导弹到靠近目标
                Qx = (1-lam)*Ax + lam*Bx
                Qy = (1-lam)*Ay + lam*By
                Qz = (1-lam)*Az + lam*Bz
                for uid in ALL_UAVS:
                    x0,y0,z0 = FY_INIT[uid]
                    # 速度/航向
                    dist_xy = math.hypot(Qx - x0, Qy - y0)
                    if tc <= 0.05: continue
                    v = dist_xy / tc
                    if v < 80.0 or v > 140.0: continue
                    hd = math.atan2(Qy - y0, Qx - x0)
                    # 引信
                    if z0 < Qz: continue
                    tau = math.sqrt( max(0.0, 2.0*(z0 - Qz)/g) )
                    if tau < 0.8 or tau > 3.0: continue
                    t_drop = tc - tau
                    if t_drop < 0.0 or (t_drop + tau) > tmax: continue
                    mask = event_mask(uid, mid, v, hd, t_drop, tau, time_grid)
                    if mask.sum()*float(dt) >= 0.01:
                        cands.append({"uid":uid,"mid":mid,"v":v,"hd":hd,
                                      "t_drop":t_drop,"tau":tau,"mask":mask,"src":"seed"})
                        if verbose >= 2:
                            print(f"    seed ok: tc={tc}, {mid}, {uid}, lam={lam}, gain≈{float(mask.sum())*dt:.3f}s")

# -------------------- 列生成（PSO 定价）主流程 --------------------
def colgen_pso(tmax: float, dt: float, windows: List[float],
               rounds: int=4, swarm: int=50, iters: int=60,
               gain_thresh: float=0.03, seed: int=0, verbose:int=1):
    rng = np.random.RandomState(seed)
    time_grid = np.arange(0.0, tmax + 1e-9, dt)
    dtf = float(dt)
    cands: List[dict] = []

    # （1）解析对齐的强初始列
    if verbose: print("[Init] building analytic-aligned seeds ...")
    seed_analytic_aligned(cands, time_grid, tmax, dt, windows, verbose=verbose)

    # （2）常规备胎初始列（更积极）
    if verbose: print("[Init] adding aggressive backup seeds ...")
    for tc in windows:
        for mid in ALL_MISSILES:
            for uid in ALL_UAVS:
                base_hd = math.atan2(-FY_INIT[uid][1], -FY_INIT[uid][0])
                for dh in (-math.radians(20), -math.radians(10), 0.0, math.radians(10), math.radians(20)):
                    for v in (90.0, 110.0, 130.0, 140.0):
                        for tau in (1.0, 1.5, 2.0, 2.5):
                            tD = max(0.0, min(tmax-0.8, tc - tau))
                            hd = base_hd + dh
                            mask = event_mask(uid, mid, v, hd, tD, tau, time_grid)
                            if mask.sum()*dtf >= 0.01:
                                cands.append({"uid":uid,"mid":mid,"v":v,"hd":hd,
                                              "t_drop":tD,"tau":tau,"mask":mask,"src":"init"})
    if verbose:
        print(f"[Init] initial candidates: {len(cands)}")

    # （3）列生成循环
    for rd in range(rounds):
        sol, how = master_solve(cands, time_grid, max_per_uav=3)
        print(f"[Round {rd+1}] solver={how}, z≈{sol['score']:.3f}s, chosen={len(sol['chosen'])}, cands={len(cands)}")

        N = len(time_grid)
        yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
        for c in sol['chosen']:
            yM[c['mid']] = np.maximum(yM[c['mid']], c['mask'])

        W = {
            "M1": (yM["M2"] & yM["M3"]).astype(float),
            "M2": (yM["M1"] & yM["M3"]).astype(float),
            "M3": (yM["M1"] & yM["M2"]).astype(float),
        }
        # 自举：若全 0，则给全 1
        for mid in ALL_MISSILES:
            if float(W[mid].sum()) == 0.0:
                W[mid] = np.ones(N, dtype=float)

        added = 0
        for tc in windows:
            for mid in ALL_MISSILES:
                cand, est = price_pso_for_missile(
                    mid, tc, time_grid, yM, W[mid],
                    swarm=swarm, iters=iters, seed=seed+rd*100, verbose=verbose
                )
                mask = event_mask(cand["uid"], cand["mid"], cand["v"], cand["hd"], cand["t_drop"], cand["tau"], time_grid)
                margin = (1 - yM[mid]) * mask * W[mid]
                gain = float(margin.sum())*dtf
                if verbose:
                    print(f"  [Price] tc={tc:>4}, {mid}: best_gain≈{gain:.4f}s ({cand['uid']}, v={cand['v']:.1f}, tau={cand['tau']:.2f})")
                if gain >= gain_thresh:
                    cand.update({"mask":mask, "src":"pso"})
                    cands.append(cand); added += 1
        print(f"  [Round {rd+1}] added columns: {added}")
        if added == 0:
            break

    # 最终主问题
    sol, how = master_solve(cands, time_grid, max_per_uav=3)
    sol["how"] = how
    return sol, time_grid

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
            "Heading(rad)": round(c["hd"],6), "DropTime(s)": round(c["t_drop"],2),
            "ExplodeTime(s)": round(tE,2), "DropX": round(xD,2), "DropY": round(yD,2),
            "DropZ": round(z0,2), "ExplodeX": round(xE,2), "ExplodeY": round(yE,2),
            "ExplodeZ": round(zE,2), "Source": c["src"]
        })
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, "pso_colgen_plan.xlsx")
    written_path = save_table(df, xlsx_path)

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for c in sol["chosen"]:
        yM[c["mid"]] = np.maximum(yM[c["mid"]], c["mask"])
    z = (yM["M1"] & yM["M2"] & yM["M3"])
    z_seconds = float(z.sum())*dt

    js = {"objective_seconds": z_seconds, "chosen_events": len(sol["chosen"]), "solver": sol.get("how","auto")}
    js_path = os.path.join(outdir, "pso_colgen_plan.json")
    Path(js_path).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written_path, js_path, z_seconds

# -------------------- CLI --------------------
def parse_windows(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--windows", type=str, default="14,26,38,50")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--swarm", type=int, default=50)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--gain_thresh", type=float, default=0.03)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1, help="0:quiet, 1:round+pricing, 2:+PSO inner logs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wins = parse_windows(args.windows)
    print(f"== PSO-ColGen v2 ==")
    print(f"tmax={args.tmax}, dt={args.dt}, windows={wins}, rounds={args.rounds}, swarm={args.swarm}, iters={args.iters}")
    sol, time_grid = colgen_pso(
        tmax=args.tmax, dt=args.dt, windows=wins,
        rounds=args.rounds, swarm=args.swarm, iters=args.iters,
        gain_thresh=args.gain_thresh, seed=args.seed, verbose=args.verbose
    )
    xlsx, js, z = export_solution(sol, time_grid, args.outdir)
    print(f"Exported:\n  plan_table: {xlsx}\n  summary:    {js}\n  z≈{z:.3f} s")

if __name__ == "__main__":
    main()
