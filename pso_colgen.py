# -*- coding: utf-8 -*-
"""
pso_colgen.py
最优方案：列生成 + 粒子群定价（PSO 定价器）
- 主问题（Master）：优先 OR-Tools CP-SAT；若无，则贪心近似。
- 定价（Pricing）：对“导弹 × 时间窗”用 PSO 搜单朵云参数，最大化边际增益。
- 输出：Excel/CSV + JSON 到 --outdir。
用法示例：
  python pso_colgen.py --tmax 60 --dt 0.2 --windows 14,26,38,50 --rounds 4 --swarm 40 --iters 50 --outdir .
依赖：
  必需：numpy pandas
  可选：ortools（更优的主问题选择），openpyxl（写 Excel）
"""
from __future__ import annotations
import os, math, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# -------------------- 场景常量（按题面，可自行微调） --------------------
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
    dx,dy,dz = _normalize((-x0,-y0,-z0))  # 指向原点单位向量
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

def covered_hard(c_center, mid: str, t: float, z_samples: int=9) -> bool:
    """硬判定：点到“导弹→目标条”的最小距离 ≤ R 即遮蔽"""
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

# -------------------- 工具：保存 Excel/CSV --------------------
def save_table(df: pd.DataFrame, path_xlsx: str):
    """优先写 Excel；若无 openpyxl 则写 CSV 到同名路径（后缀改 .csv）"""
    try:
        df.to_excel(path_xlsx, index=False)  # 需要 openpyxl
        return path_xlsx
    except Exception:
        csv_path = os.path.splitext(path_xlsx)[0] + ".csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

# -------------------- 候选事件与掩码 --------------------
def event_mask(uid: str, mid: str, v: float, hd: float, t_drop: float, tau: float, time_grid: np.ndarray) -> np.ndarray:
    tE = t_drop + tau
    xE,yE = uav_xy(uid, v, hd, tE)
    z0 = FY_INIT[uid][2]
    zE = z0 - 0.5*g*(tau**2)
    mask = np.zeros(len(time_grid), dtype=np.int8)
    for i,t in enumerate(time_grid):
        if tE <= t <= tE + CLOUD_ACTIVE:
            c = (xE, yE, zE - CLOUD_SINK*(t - tE))
            if covered_hard(c, mid, t, z_samples=9):
                mask[i] = 1
    return mask

# -------------------- 主问题：MILP（优先）/ 贪心（回退） --------------------
def master_greedy(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3) -> dict:
    """简单贪心：每次选增益最大的候选，直到受限。"""
    dt = float(time_grid[1]-time_grid[0])
    N  = len(time_grid)
    chosen = []
    used  = {u:0 for u in ALL_UAVS}
    lastt = {u:-1e9 for u in ALL_UAVS}
    yM    = {m:np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}

    def zscore():
        return float((yM["M1"] & yM["M2"] & yM["M3"]).sum())*dt

    while True:
        base = zscore()
        best = None
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
    return {"chosen":chosen, "score":zscore()}

def master_solve(cands: List[dict], time_grid: np.ndarray, max_per_uav: int=3) -> Tuple[dict,str]:
    """优先 OR-Tools CP-SAT；否则回退到贪心。"""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return master_greedy(cands, time_grid, max_per_uav=max_per_uav), "greedy"

    dt = float(time_grid[1]-time_grid[0]); N = len(time_grid)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(cands))]
    y = [[model.NewBoolVar(f"y{m}{t}") for t in range(N)] for m in range(3)]
    z = [model.NewBoolVar(f"z{t}") for t in range(N)]

    # link y
    for m in range(3):
        for t in range(N):
            inv = [x[i] for i,c in enumerate(cands) if M2IDX[c['mid']]==m and c['mask'][t]==1]
            if inv:
                model.Add(y[m][t] <= sum(inv))
            else:
                model.Add(y[m][t] == 0)
    # AND for z
    for t in range(N):
        for m in range(3):
            model.Add(z[t] <= y[m][t])
    # per UAV ≤ 3
    for uid in ALL_UAVS:
        model.Add(sum(x[i] for i,c in enumerate(cands) if c['uid']==uid) <= max_per_uav)
        # gap ≥ 1s （仅对近邻对加互斥）
        lst = [(i,c) for i,c in enumerate(cands) if c['uid']==uid]
        lst.sort(key=lambda ic: ic[1]['t_drop'])
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if abs(lst[i][1]['t_drop'] - lst[j][1]['t_drop']) < 1.0 - 1e-9:
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
def stable_sigmoid(x: np.ndarray | float):
    # 数值稳定版：sigmoid(x) = 0.5 * (1 + tanh(x/2))
    return 0.5*(1.0 + np.tanh(0.5*x))

def price_pso_for_missile(mid: str, t_center: float, time_grid: np.ndarray,
                          yM: Dict[str,np.ndarray], W: np.ndarray,
                          swarm: int=40, iters: int=50, seed: int=0):
    """
    用 PSO 在窗口中心 t_center 附近搜索一朵云，使得该导弹 mid 在权重 W(t) 下的“边际增益”最大。
    - yM: 当前解中各导弹的覆盖 0/1 掩码
    - W: 权重 = 1{其它两枚导弹已覆盖}
    返回：最佳候选 dict（不含 mask），若增益很小可由上层过滤。
    """
    rng = np.random.RandomState(seed + int(t_center*1000))
    N = len(time_grid); dt = float(time_grid[1]-time_grid[0])

    # 变量编码（连续）：
    # u_logits[5], v_raw, dh_raw, tD_raw, tau_raw
    D = 5 + 4
    X = rng.randn(swarm, D)*0.1  # 位置
    V = np.zeros((swarm, D))     # 速度

    # PSO 参数
    w0, w1 = 0.9, 0.4           # 惯性线性衰减
    c1, c2 = 1.8, 1.8
    vclip = np.array([0.6]*5 + [0.8, 0.8, 1.2, 0.8], dtype=float)  # 各维速度上限

    def decode(xrow):
        # UAV 概率（softmax），然后 argmax 离散
        lo = xrow[:5]
        pu = np.exp(lo - np.max(lo)); pu /= pu.sum()
        uidx = int(np.argmax(pu)); uid = ALL_UAVS[uidx]
        # 连续变量映射
        v  = 80.0 + 60.0*stable_sigmoid(xrow[5])          # [80,140]
        dh = math.radians(15.0)*np.tanh(xrow[6])          # [-15°,15°]
        # 把投放时刻限制在窗口 ±2.5s，再截断到 [0, tmax-0.8]
        tD = (t_center - 2.5) + 5.0*stable_sigmoid(xrow[7])
        tau = 0.8 + 2.2*stable_sigmoid(xrow[8])           # [0.8,3.0]
        return uid, v, dh, tD, tau

    def fitness(xrow):
        uid, v, dh, tD, tau = decode(xrow)
        # 计算航向：以 UAV 初始点指向原点为基准 + dh
        base_hd = math.atan2(-FY_INIT[uid][1], -FY_INIT[uid][0])
        hd = base_hd + dh
        # 边际增益（硬判定）：只在当前 mid 未被覆盖且 W=1 的时刻有用
        mask = event_mask(uid, mid, v, hd, tD, tau, time_grid)
        # 边际： (1 - yM[mid]) * mask * W
        margin = (1 - yM[mid]) * mask * W
        return float(margin.sum())*dt, (uid, v, hd, tD, tau)

    # 初始化个人最好 / 全局最好
    pbest_val = np.full(swarm, -1e18, dtype=float)
    pbest_x   = X.copy()
    pbest_aux = [None]*swarm
    gbest_val = -1e18
    gbest_x   = None
    gbest_aux = None

    for it in range(iters):
        w = w0 + (w1 - w0)*it/max(1, iters-1)
        for i in range(swarm):
            val, aux = fitness(X[i])
            if val > pbest_val[i]:
                pbest_val[i] = val; pbest_x[i] = X[i].copy(); pbest_aux[i] = aux
            if val > gbest_val:
                gbest_val = val; gbest_x = X[i].copy(); gbest_aux = aux
        # 更新速度与位置
        r1 = rng.rand(swarm, D); r2 = rng.rand(swarm, D)
        V = w*V + c1*r1*(pbest_x - X) + c2*r2*(gbest_x - X)
        V = np.clip(V, -vclip, vclip)
        X = X + V

    # 返回最佳（离散化后再由上层计算 mask）
    if gbest_aux is None:
        # PSO 没有成功更新，返回一个极保守的候选占位
        uid = ALL_UAVS[0]; v=120.0; tau=2.0; tD=max(0.0, t_center-2.0)
        base_hd = math.atan2(-FY_INIT[uid][1], -FY_INIT[uid][0]); hd=base_hd
        return {"uid":uid,"mid":mid,"v":v,"hd":hd,"t_drop":tD,"tau":tau}, 0.0
    uid, v, hd, tD, tau = gbest_aux
    return {"uid":uid,"mid":mid,"v":float(v),"hd":float(hd),"t_drop":float(tD),"tau":float(tau)}, float(gbest_val)

# -------------------- 列生成（PSO 定价）主流程 --------------------
def colgen_pso(tmax: float, dt: float, windows: List[float],
               rounds: int=4, swarm: int=40, iters: int=50,
               gain_thresh: float=0.05, seed: int=0):
    """列生成主循环：初始列 → 主问题 → PSO 定价加列 → 迭代。"""
    rng = np.random.RandomState(seed)
    time_grid = np.arange(0.0, tmax + 1e-9, dt)
    dtf = float(dt)
    cands: List[dict] = []

    # 初始候选（积极一些，保证第一轮能动起来）
    for tc in windows:
        for mid in ALL_MISSILES:
            for uid in ALL_UAVS:
                base_hd = math.atan2(-FY_INIT[uid][1], -FY_INIT[uid][0])
                for dh in (-math.radians(12), -math.radians(6), 0.0, math.radians(6), math.radians(12)):
                    for v in (100.0, 120.0, 140.0):
                        for tau in (1.5, 2.0, 2.5):
                            tD = max(0.0, min(tmax-0.8, tc - tau))  # 爆点贴近窗口中心
                            hd = base_hd + dh
                            mask = event_mask(uid, mid, v, hd, tD, tau, time_grid)
                            if mask.sum()*dtf >= 0.03:  # 降低阈值，利于启动
                                cands.append({"uid":uid,"mid":mid,"v":v,"hd":hd,
                                              "t_drop":tD,"tau":tau,"mask":mask,"src":"init"})

    # 列生成迭代
    for rd in range(rounds):
        sol, how = master_solve(cands, time_grid, max_per_uav=3)
        print(f"[Round {rd+1}] master={how}, z≈{sol['score']:.3f}, chosen={len(sol['chosen'])}")

        # 当前解的三导弹覆盖
        N = len(time_grid)
        yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
        for c in sol['chosen']:
            yM[c['mid']] = np.maximum(yM[c['mid']], c['mask'])

        # 构造权重：在“其它两导弹已覆盖”的时刻才有边际价值
        W = {
            "M1": (yM["M2"] & yM["M3"]).astype(float),
            "M2": (yM["M1"] & yM["M3"]).astype(float),
            "M3": (yM["M1"] & yM["M2"]).astype(float),
        }
        # 自举：若某导弹权重全 0，则先给全 1（启动后续交集）
        for mid in ALL_MISSILES:
            if float(W[mid].sum()) == 0.0:
                W[mid] = np.ones(N, dtype=float)

        # PSO 定价：为每个“窗×导弹”加 1 条新列（若增益达标）
        added = 0
        for tc in windows:
            for mid in ALL_MISSILES:
                cand, est_gain = price_pso_for_missile(
                    mid, tc, time_grid, yM, W[mid],
                    swarm=swarm, iters=iters, seed=seed+rd*100
                )
                # 以硬判定 + 当前 yM 计算真实边际（与 PSO 评估一致）
                mask = event_mask(cand["uid"], cand["mid"], cand["v"], cand["hd"], cand["t_drop"], cand["tau"], time_grid)
                margin = (1 - yM[mid]) * mask * W[mid]
                gain = float(margin.sum())*dtf
                if gain >= gain_thresh:
                    cand.update({"mask":mask, "src":"pso"})
                    cands.append(cand); added += 1
        print(f"  added columns: {added} (cands total: {len(cands)})")
        if added == 0:
            break

    # 最终主问题
    sol, how = master_solve(cands, time_grid, max_per_uav=3)
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
            "UAV": c["uid"],
            "Missile": c["mid"],
            "Speed(m/s)": round(c["v"],1),
            "Heading(rad)": round(c["hd"],6),
            "DropTime(s)": round(c["t_drop"],2),
            "ExplodeTime(s)": round(tE,2),
            "DropX": round(xD,2), "DropY": round(yD,2), "DropZ": round(z0,2),
            "ExplodeX": round(xE,2), "ExplodeY": round(yE,2), "ExplodeZ": round(zE,2),
            "Source": c["src"]
        })
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    xlsx_path = os.path.join(outdir, "pso_colgen_plan.xlsx")
    written_path = save_table(df, xlsx_path)

    # 估计同时遮蔽秒数（按位 AND）
    dt = float(time_grid[1]-time_grid[0])
    N  = len(time_grid)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for c in sol["chosen"]:
        yM[c["mid"]] = np.maximum(yM[c["mid"]], c["mask"])
    z = (yM["M1"] & yM["M2"] & yM["M3"])
    z_seconds = float(z.sum())*dt

    js = {
        "objective_seconds": z_seconds,
        "chosen_events": len(sol["chosen"]),
        "solver": "MILP" if "milp" in written_path.lower() else "auto",
    }
    js_path = os.path.join(outdir, "pso_colgen_plan.json")
    Path(js_path).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written_path, js_path, z_seconds

# -------------------- CLI --------------------
def parse_windows(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=60.0, help="总时长（秒）")
    ap.add_argument("--dt", type=float, default=0.2, help="时间步（秒）")
    ap.add_argument("--windows", type=str, default="14,26,38,50", help="时间窗中心，逗号分隔")
    ap.add_argument("--rounds", type=int, default=4, help="列生成轮数（3~6 常见）")
    ap.add_argument("--swarm", type=int, default=40, help="PSO 粒子数")
    ap.add_argument("--iters", type=int, default=50, help="PSO 迭代数")
    ap.add_argument("--gain_thresh", type=float, default=0.05, help="加入新列最小边际（秒）")
    ap.add_argument("--outdir", type=str, default=".", help="输出目录（默认当前目录）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wins = parse_windows(args.windows)
    sol, time_grid = colgen_pso(
        tmax=args.tmax, dt=args.dt, windows=wins,
        rounds=args.rounds, swarm=args.swarm, iters=args.iters,
        gain_thresh=args.gain_thresh, seed=args.seed
    )
    xlsx, js, z = export_solution(sol, time_grid, args.outdir)
    print(f"Exported:\n  plan_table: {xlsx}\n  summary:    {js}\n  z≈{z:.3f} s")

if __name__ == "__main__":
    main()
