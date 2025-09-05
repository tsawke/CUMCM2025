# -*- coding: utf-8 -*-
"""
q5_ml_surrogate_cma.py
方案A：代理模型（MLP）+ CMA-ES 全局优化 + 真仿真验收
- 物理：UAV 水平匀速；投放后至起爆，烟幕水平随机；起爆后水平静止、竖直匀速下沉；
- 几何：判定“完全遮蔽”= 对目标圆柱(z∈[0,10])上采样点，导弹M(t)->点的线段与球相交（等价：球心到线段距离≤R），全部成立；
- 参数化（每朵云一行）：
    uav_id ∈ {0..4}，missile_id ∈ {0..2}，tc∈[8,62]（起爆时刻），tau∈[0.8,3.0]（引信），zT_id∈{0,1,2}（目标高度端点/中点）
  通过“严格3D对齐”反解 v/heading/t_drop（若不可行则丢弃该云）。
- 流程：
    1) 用对齐生成器 + 轻噪声合成训练集（几千条）；仿真打真分；
    2) 训练 MLP 回归分数；
    3) CMA-ES 在代理上搜若干候选，回投真仿真选最优；
    4) 导出计划表与 JSON 摘要。
"""

import math, os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# 可选：代理模型与CMA
import torch
import torch.nn as nn
import torch.optim as optim
import cma

# ============ 场景常量（按题面自行调整） ============
g = 9.8
CLOUD_RADIUS = 10.0     # 烟幕半径(m)
CLOUD_ACTIVE = 20.0     # 起爆后有效期(s)
CLOUD_SINK = 3.0        # 起爆后下沉速度(m/s)
MISSILE_SPEED = 300.0   # 导弹速度(m/s)
TARGET_CENTER_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0

# 初始位置（如与PDF不同请改）
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
ALL_UAVS = list(FY_INIT.keys())         # 固定顺序
ALL_MISSILES = ["M1","M2","M3"]
M2IDX = {"M1":0,"M2":1,"M3":2}

# UAV 速度限制（按题面）
V_MIN, V_MAX = 80.0, 140.0
TMAX = 60.0

# ============ 几何/仿真基础 ============
def _normalize(v):
    x,y,z = v
    n = math.hypot(x, math.hypot(y,z))
    return (x/n, y/n, z/n) if n>0 else (0.0,0.0,0.0)

def missile_pos(mid: str, t: float):
    # 指向“目标中心(z取中值)”的直线等速
    x0,y0,z0 = M_INIT[mid]
    xT,yT = TARGET_CENTER_XY
    zT = 0.5*(TARGET_Z0+TARGET_Z1)
    dx,dy,dz = _normalize((xT-x0, yT-y0, zT-z0))
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_xy(uid: str, v: float, hd: float, t: float):
    x0,y0,_ = FY_INIT[uid]
    return (x0 + v*t*math.cos(hd),
            y0 + v*t*math.sin(hd))

def point_to_segment_dist(P, A, B) -> float:
    ax,ay,az = A; bx,by,bz = B; px,py,pz = P
    AB = (bx-ax, by-ay, bz-az)
    AP = (px-ax, py-ay, pz-az)
    ab2 = AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]
    if ab2 == 0.0:
        dx,dy,dz = (px-ax, py-ay, pz-az)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    s = (AP[0]*AB[0] + AP[1]*AB[1] + AP[2]*AB[2]) / ab2
    if s < 0.0: s = 0.0
    elif s > 1.0: s = 1.0
    qx = ax + AB[0]*s; qy = ay + AB[1]*s; qz = az + AB[2]*s
    dx,dy,dz = (px-qx, py-qy, pz-qz)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def covered_at_t(C, mid: str, t: float, z_samples: int=11, eps: float=1e-8) -> bool:
    """时刻 t：烟幕球心 C 覆盖导弹 mid → 目标条(z∈[0,10])？"""
    if C is None:
        return False
    m = missile_pos(mid, t)
    xT,yT = TARGET_CENTER_XY
    if z_samples <= 1:
        zs = [(TARGET_Z0 + TARGET_Z1)/2.0]
    else:
        step = (TARGET_Z1 - TARGET_Z0) / (z_samples - 1)
        zs = [TARGET_Z0 + i*step for i in range(z_samples)]
    for z in zs:
        if point_to_segment_dist(C, m, (xT,yT,z)) > CLOUD_RADIUS + eps:
            return False
    return True

def cloud_center_from_event(uid, v, hd, t_drop, tau, t):
    """返回时刻 t 的云心；t < tE 未起爆 → None；t > tE+CLOUD_ACTIVE 视作失效 → None。"""
    tE = t_drop + tau
    if t < tE or t > tE + CLOUD_ACTIVE:
        return None
    xE,yE = uav_xy(uid, v, hd, tE)
    z0 = FY_INIT[uid][2]
    zE = z0 - 0.5*g*(tau**2)
    return (xE, yE, zE - CLOUD_SINK*(t - tE))

# ============ 事件“严格3D对齐”反解 ============
def strict_align(mid: str, uid: str, tc: float, tau: float, zT: float, tmax=TMAX):
    """
    令起爆点落在 M(tc)->T(zT) 的同一条直线上，且高度= zE = z0 - 0.5 g tau^2。
    反解 UAV 的 v/hd/t_drop；若不可行返回 None。
    """
    Mx,My,Mz = missile_pos(mid, tc)
    xT,yT = TARGET_CENTER_XY
    x0,y0,z0 = FY_INIT[uid]
    zE = z0 - 0.5*g*(tau**2)
    denom = (zT - Mz)
    if abs(denom) < 1e-9:
        return None
    lam = (zE - Mz)/denom
    if lam < 0.0 or lam > 1.0:
        return None
    Qx = Mx + lam*(xT - Mx)
    Qy = My + lam*(yT - My)
    dist_xy = math.hypot(Qx - x0, Qy - y0)
    if tc <= 0.05:
        return None
    v = dist_xy / tc
    if not (V_MIN <= v <= V_MAX):
        return None
    hd = math.atan2(Qy - y0, Qx - x0)
    t_drop = tc - tau
    if t_drop < 0.0 or tc > tmax:
        return None
    return {"uid": uid, "mid": mid, "v": v, "hd": hd, "t_drop": t_drop, "tau": tau, "tc": tc, "zT": zT}

# ============ 计划/仿真：目标分数 ============
def simulate_plan(events: List[dict], tmax=TMAX, dt=0.1) -> Tuple[float, dict]:
    """
    输入 events（每个含 uid,mid,v,hd,t_drop,tau），返回
    - score = ∑_t 1[三导弹同时被至少一朵云覆盖] * dt
    - 附加统计（每导弹覆盖秒数、掩码）
    约束：每 UAV ≤3，且同 UAV 的投放间隔 ≥1s（不满足者该事件作废）
    """
    # 约束过滤（按 t_drop 排序）
    per_uav = {u: [] for u in ALL_UAVS}
    for e in events:
        per_uav[e["uid"]].append(e)
    pruned = []
    for u, lst in per_uav.items():
        lst.sort(key=lambda x: x["t_drop"])
        taken = []
        for e in lst:
            if len(taken) >= 3:
                continue
            if taken and e["t_drop"] - taken[-1]["t_drop"] < 1.0 - 1e-9:
                continue
            taken.append(e)
        pruned.extend(taken)

    ts = np.arange(0.0, tmax + 1e-9, dt)
    N = len(ts)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for e in pruned:
        for i,t in enumerate(ts):
            C = cloud_center_from_event(e["uid"], e["v"], e["hd"], e["t_drop"], e["tau"], t)
            if C is None:
                continue
            if covered_at_t(C, e["mid"], t, z_samples=11):
                yM[e["mid"]][i] |= 1
    zmask = yM["M1"] & yM["M2"] & yM["M3"]
    score = float(zmask.sum()) * dt
    per_m = {m: float(yM[m].sum())*dt for m in ALL_MISSILES}
    return score, {"per_m": per_m, "zmask": zmask, "events": pruned, "dt": dt}

# ============ 参数化：连续向量 <-> 事件（对齐生成） ============
# 每个云的5维参数： [uav_s, mis_s, tc_s, tau_s, zT_s]
# uav_id = floor(sigmoid(uav_s)*5) 或 clamp到[0,4]
# mis_id = floor(sigmoid(mis_s)*3)
# tc = 8 + sigmoid(tc_s) * (62-8)
# tau = 0.8 + sigmoid(tau_s) * (3.0-0.8)
# zT_id = floor(sigmoid(zT_s)*3) ∈ {0:0,1:5,2:10}

def sigmoid(x): return 1/(1+math.exp(-x))

def vec_to_events(x: np.ndarray, K: int) -> List[dict]:
    """将长度=5K 的向量解析为最多K朵云的对齐事件（不可行的会丢弃）。"""
    events = []
    for k in range(K):
        uav_s, mis_s, tc_s, tau_s, zT_s = x[5*k:5*(k+1)]
        uidx = int(np.clip(int(sigmoid(uav_s)*5.0), 0, 4))
        midx = int(np.clip(int(sigmoid(mis_s)*3.0), 0, 2))
        tc = 8.0 + sigmoid(tc_s)*(62.0 - 8.0)
        tau = 0.8 + sigmoid(tau_s)*(3.0 - 0.8)
        zT_id = int(np.clip(int(sigmoid(zT_s)*3.0), 0, 2))
        zT = [TARGET_Z0, 0.5*(TARGET_Z0+TARGET_Z1), TARGET_Z1][zT_id]

        uid = ALL_UAVS[uidx]
        mid = ALL_MISSILES[midx]
        e = strict_align(mid, uid, tc, tau, zT, tmax=TMAX)
        if e is not None:
            events.append(e)
    return events

def random_vec(K:int) -> np.ndarray:
    """N(0,1) 初始化（用sigmoid做有界映射）；"""
    return np.random.randn(5*K).astype(np.float32)

def jitter(x: np.ndarray, scale=0.2):
    return x + np.random.randn(*x.shape).astype(np.float32)*scale

# ============ 训练集合成（对齐+噪声） ============
def synthesize_dataset(N: int, K: int, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 N 条样本：随机 x → 事件 → 仿真得分 y
    为了提升“非零样本比率”，使用对齐解码；并对x做轻噪声抖动。
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 5*K), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        base = random_vec(K)
        # 提高命中率：让部分维度更接近“可行域中心”（比如 tc 中域、tau中域）
        for k in range(K):
            base[5*k+2] += rng.normal(0, 0.5)  # tc_s
            base[5*k+3] += rng.normal(0, 0.3)  # tau_s
        if rng.random() < 0.5:
            base = jitter(base, 0.15)
        events = vec_to_events(base, K)
        score, _ = simulate_plan(events, tmax=TMAX, dt=0.1)
        X[i] = base
        y[i] = score
        if (i+1) % max(1, N//10) == 0:
            print(f"[Data] {i+1}/{N}, last z≈{score:.3f}")
    return X, y

# ============ 代理模型（MLP） ============
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256,256,128]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_surrogate(X, y, lr=1e-3, epochs=50, batch=128, val_split=0.1, seed=0):
    torch.manual_seed(seed)
    N = X.shape[0]; idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = int(N*val_split)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    xtr = torch.tensor(X[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    xva = torch.tensor(X[val_idx], dtype=torch.float32)
    yva = torch.tensor(y[val_idx], dtype=torch.float32)

    model = MLP(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    best = (1e9, None)
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(xtr.size(0))
        losses = []
        for i in range(0, xtr.size(0), batch):
            sel = perm[i:i+batch]
            pred = model(xtr[sel])
            loss = ((pred - ytr[sel])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vpred = model(xva); vloss = ((vpred - yva)**2).mean().item()
        if vloss < best[0]:
            best = (vloss, model.state_dict())
        if ep % 5 == 0:
            print(f"[Surrogate] ep={ep} train_mse={np.mean(losses):.4f} val_mse={vloss:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model

# ============ CMA-ES 在代理上搜解，回投真仿真选最优 ============
def optimize_with_surrogate(model, K:int, iters=300, pop=32, sigma=0.8, topM=20):
    dim = 5*K
    es = cma.CMAEvolutionStrategy(np.zeros(dim), sigma,
                                  {'popsize': pop, 'verb_disp': 1})
    cand_bank = []  # 保存若干最好候选
    def f_sur(x):
        x = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            val = model(torch.tensor(x[None,:], dtype=torch.float32)).item()
        # CMA-ES 是最小化，这里取 -score
        return -val
    while not es.stop():
        X = es.ask()
        F = [f_sur(x) for x in X]
        es.tell(X, F)
        es.disp()
        # 收集当前最好几个
        for x, f in sorted(zip(X,F), key=lambda xf: xf[1])[:5]:
            cand_bank.append(np.asarray(x, dtype=np.float32))
        if es.countiter >= iters:
            break
    # 从候选池选 topM，回投真仿真
    cand_bank = cand_bank[-(topM*5):]  # 取后段（更新）
    evals = []
    for x in cand_bank[-topM:]:
        events = vec_to_events(x, K)
        z, extra = simulate_plan(events, tmax=TMAX, dt=0.1)
        evals.append((z, x, events, extra))
    evals.sort(key=lambda e: -e[0])
    if len(evals)==0:
        return None
    return evals[0]  # 最优真分的 (z, x, events, extra)

# ============ 导出 ============
def export_plan(events: List[dict], score: float, extra: dict, outdir: str, stem="q5_solution"):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for i,e in enumerate(events,1):
        tE = e["t_drop"] + e["tau"]
        xD,yD = uav_xy(e["uid"], e["v"], e["hd"], e["t_drop"])
        z0 = FY_INIT[e["uid"]][2]
        xE,yE = uav_xy(e["uid"], e["v"], e["hd"], tE)
        zE = z0 - 0.5*g*(e["tau"]**2)
        rows.append({
            "UAV": e["uid"], "TargetMissile": e["mid"],
            "Speed(m/s)": round(e["v"],2), "Heading(rad)": round(e["hd"],6),
            "DropTime(s)": round(e["t_drop"],3), "ExplodeTime(s)": round(tE,3),
            "DropX": round(xD,2), "DropY": round(yD,2), "DropZ": round(z0,2),
            "ExplodeX": round(xE,2), "ExplodeY": round(yE,2), "ExplodeZ": round(zE,2),
            "zT": e.get("zT","-"), "tc": e.get("tc","-")
        })
    df = pd.DataFrame(rows)
    xlsx = os.path.join(outdir, f"{stem}.xlsx")
    try:
        df.to_excel(xlsx, index=False)
        written = xlsx
    except Exception:
        csv = os.path.join(outdir, f"{stem}.csv")
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        written = csv

    js = {
        "objective_seconds": score,
        "per_missile_seconds": extra["per_m"],
        "dt": extra["dt"], "chosen_events": len(events)
    }
    jpath = os.path.join(outdir, f"{stem}.json")
    Path(jpath).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written, jpath

# ============ 主程序 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", type=int, default=6, help="最大烟幕弹数K（上限15更慢）")
    ap.add_argument("--trainN", type=int, default=3000, help="合成训练样本数")
    ap.add_argument("--epochs", type=int, default=60, help="代理训练轮数")
    ap.add_argument("--cma_iters", type=int, default=300, help="CMA迭代轮数")
    ap.add_argument("--cma_pop", type=int, default=32, help="CMA种群规模")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    K = args.clouds
    print(f"== Q5: Surrogate + CMA-ES (K={K}) ==")

    # 1) 合成训练集
    X, y = synthesize_dataset(args.trainN, K, seed=args.seed)
    print(f"[Data] mean z={y.mean():.3f}, >0比例={(y>0).mean():.3f}")

    # 2) 训练代理
    model = train_surrogate(X, y, epochs=args.epochs, lr=1e-3, batch=256, val_split=0.1, seed=args.seed)

    # 3) 代理上做 CMA-ES，回投真仿真选最优
    best = optimize_with_surrogate(model, K, iters=args.cma_iters, pop=args.cma_pop, sigma=0.8, topM=20)
    if best is None:
        print("[Warn] CMA 未获得候选；改用随机若干解挑最好（保底）。")
        evals=[]
        for _ in range(50):
            x = random_vec(5*K)
            events = vec_to_events(x, K)
            z, extra = simulate_plan(events, tmax=TMAX, dt=0.1)
            evals.append((z, x, events, extra))
        evals.sort(key=lambda e:-e[0])
        best = evals[0]

    zbest, xbest, evbest, extra = best
    print(f"[Best] true z≈{zbest:.3f}s with {len(evbest)} events; per-missile={extra['per_m']}")

    # 4) 导出
    xlsx, jpath = export_plan(evbest, zbest, extra, outdir=args.outdir, stem="q5_solution")
    print("Exported:", xlsx, jpath)

if __name__ == "__main__":
    main()
