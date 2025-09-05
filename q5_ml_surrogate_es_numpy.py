# -*- coding: utf-8 -*-
"""
q5_ml_surrogate_es_numpy.py
纯 NumPy 版本：代理(随机傅里叶特征+岭回归) + 进化搜索(ES) + 真仿真验收
依赖：numpy, pandas（Windows 自带环境一般都装得起；如写 xlsx 失败会自动写 csv）

参数示例：
  python q5_ml_surrogate_es_numpy.py --clouds 6 --trainN 3000 --es_iters 300 --outdir .
"""

import math, os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# -------------- 场景常量（按题面替换即可） ---------------
g = 9.8
CLOUD_RADIUS = 10.0     # 烟幕半径(m)
CLOUD_ACTIVE = 20.0     # 起爆后有效期(s)
CLOUD_SINK = 3.0        # 起爆后下沉速度(m/s)
MISSILE_SPEED = 300.0   # 导弹速度(m/s)
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
V_MIN, V_MAX = 80.0, 140.0
TMAX = 60.0

# -------------- 几何/物理 ---------------
def _normalize(v):
    x,y,z = v
    n = math.hypot(x, math.hypot(y,z))
    return (x/n, y/n, z/n) if n>0 else (0.0,0.0,0.0)

def missile_pos(mid: str, t: float):
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
    tE = t_drop + tau
    if t < tE or t > tE + CLOUD_ACTIVE:
        return None
    xE,yE = uav_xy(uid, v, hd, tE)
    z0 = FY_INIT[uid][2]
    zE = z0 - 0.5*g*(tau**2)
    return (xE, yE, zE - CLOUD_SINK*(t - tE))

# -------------- 严格 3D 对齐生成 ---------------
def strict_align(mid: str, uid: str, tc: float, tau: float, zT: float, tmax=TMAX):
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

# -------------- 计划仿真得分 ---------------
def simulate_plan(events: List[dict], tmax=TMAX, dt=0.1) -> Tuple[float, dict]:
    per_uav = {u: [] for u in ALL_UAVS}
    for e in events:
        per_uav[e["uid"]].append(e)
    pruned = []
    for u, lst in per_uav.items():
        lst.sort(key=lambda x: x["t_drop"])
        taken = []
        for e in lst:
            if len(taken) >= 3:  # 每机 <=3
                continue
            if taken and e["t_drop"] - taken[-1]["t_drop"] < 1.0 - 1e-9:  # 间隔 >=1s
                continue
            taken.append(e)
        pruned.extend(taken)

    ts = np.arange(0.0, tmax + 1e-9, dt)
    N = len(ts)
    yM = {m: np.zeros(N, dtype=np.int8) for m in ALL_MISSILES}
    for e in pruned:
        for i,t in enumerate(ts):
            C = cloud_center_from_event(e["uid"], e["v"], e["hd"], e["t_drop"], e["tau"], t)
            if C is None: continue
            if covered_at_t(C, e["mid"], t, z_samples=11):
                yM[e["mid"]][i] |= 1
    zmask = yM["M1"] & yM["M2"] & yM["M3"]
    score = float(zmask.sum()) * dt
    per_m = {m: float(yM[m].sum())*dt for m in ALL_MISSILES}
    return score, {"per_m": per_m, "zmask": zmask, "events": pruned, "dt": dt}

# -------------- 连续向量 <-> 事件（对齐解码） ---------------
def sigmoid(x): return 1/(1+np.exp(-x))

def vec_to_events(x: np.ndarray, K: int) -> List[dict]:
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
    return np.random.randn(5*K).astype(np.float32)

def jitter(x: np.ndarray, scale=0.2):
    return x + np.random.randn(*x.shape).astype(np.float32)*scale

# -------------- 合成训练集（真仿真打分） ---------------
def synthesize_dataset(N: int, K: int, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 5*K), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        base = random_vec(K)
        # 提高命中率：让 tc/tau 更靠近中域，并加轻噪声
        for k in range(K):
            base[5*k+2] += rng.normal(0, 0.5)  # tc_s
            base[5*k+3] += rng.normal(0, 0.3)  # tau_s
        if rng.random() < 0.5:
            base = jitter(base, 0.15)
        events = vec_to_events(base, K)
        score, _ = simulate_plan(events, tmax=TMAX, dt=0.1)
        X[i] = base; y[i] = score
        if (i+1) % max(1, N//10) == 0:
            print(f"[Data] {i+1}/{N}, last z≈{score:.3f}")
    return X, y

# -------------- 代理模型：随机傅里叶特征 + 岭回归（纯 NumPy） ---------------
class RFFRidge:
    def __init__(self, in_dim, num_feat=1024, gamma=0.5, ridge=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, math.sqrt(2*gamma), size=(in_dim, num_feat)).astype(np.float32)
        self.b = rng.uniform(0, 2*np.pi, size=(num_feat,)).astype(np.float32)
        self.ridge = ridge
        self.mean_x = None
        self.std_x = None
        self.w = None  # 线性权重
        self.c = 0.0   # 偏置

    def _phi(self, X):
        # ϕ(x) = sqrt(2/D) * cos(xW + b)
        Z = X @ self.W + self.b
        return (np.sqrt(2.0 / Z.shape[1]) * np.cos(Z)).astype(np.float32)

    def fit(self, X, y):
        # 标准化输入
        self.mean_x = X.mean(axis=0, keepdims=True)
        self.std_x  = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - self.mean_x)/self.std_x
        Phi = self._phi(Xn)   # N x D
        # 岭回归闭式解：w = (Phi^T Phi + λI)^(-1) Phi^T y
        D = Phi.shape[1]
        A = Phi.T @ Phi + self.ridge*np.eye(D, dtype=np.float32)
        b = Phi.T @ y
        self.w = np.linalg.solve(A, b).astype(np.float32)
        self.c = 0.0

    def predict(self, X):
        Xn = (X - self.mean_x)/self.std_x
        Phi = self._phi(Xn)
        return Phi @ self.w + self.c

# -------------- 进化搜索（ES，纯 NumPy） ---------------
def es_optimize(predict_fn, K:int, iters=300, pop=40, sigma=0.8, keep=10, seed=0):
    """
    在代理上最大化预测分数的简易 (μ,λ)-ES。
    返回若干最优候选向量 x（用于回投真仿真）。
    """
    rng = np.random.default_rng(seed)
    dim = 5*K
    m = np.zeros(dim, dtype=np.float32)  # 初始均值 0
    s = sigma
    hall = []  # Hall of fame

    for it in range(1, iters+1):
        X = m + s * rng.normal(0, 1, size=(pop, dim)).astype(np.float32)
        # 加一个中心点
        X[0] = m
        # 评估（要最大化 → 取负号最小化也行，这里直接最大化）
        preds = predict_fn(X)  # -> shape (pop,)
        order = np.argsort(-preds)
        top = X[order[:max(2, pop//4)]]
        m = top.mean(axis=0).astype(np.float32)
        # 简单自适应：若前 1/3 迭代分数没提升就轻微扩散，否则温和衰减
        best = float(preds[order[0]])
        hall.append((best, X[order[0]].copy()))
        if it % 20 == 0 and len(hall) >= 20:
            prev = np.mean([h[0] for h in hall[-20:-10]])
            curr = np.mean([h[0] for h in hall[-10:]])
            if curr <= prev + 1e-6:
                s *= 1.05
            else:
                s *= 0.98
        if it % max(1, iters//10) == 0:
            print(f"[ES] iter {it}/{iters}, proxy-best≈{best:.3f}, sigma={s:.3f}")

    hall.sort(key=lambda z: -z[0])
    xs = [x for _,x in hall[:keep]]
    # 去重
    uniq = []
    for x in xs:
        if not any(np.allclose(x, y, atol=1e-6) for y in uniq):
            uniq.append(x)
    return uniq[:keep]

# -------------- 导出 ---------------
def export_plan(events: List[dict], score: float, extra: dict, outdir: str, stem="q5_solution_numpy"):
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

    js = {"objective_seconds": score, "per_missile_seconds": extra["per_m"], "dt": extra["dt"], "chosen_events": len(events)}
    jpath = os.path.join(outdir, f"{stem}.json")
    Path(jpath).write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    return written, jpath

# -------------- 主流程 ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", type=int, default=6)
    ap.add_argument("--trainN", type=int, default=3000)
    ap.add_argument("--es_iters", type=int, default=300)
    ap.add_argument("--es_pop", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)
    K = args.clouds
    print(f"== Q5: NumPy Surrogate + ES (K={K}) ==")

    # 1) 合成训练集
    X, y = synthesize_dataset(args.trainN, K, seed=args.seed)
    print(f"[Data] mean z={y.mean():.3f}, >0比例={(y>0).mean():.3f}")

    # 2) 训练代理（RFF+Ridge）
    rff = RFFRidge(in_dim=X.shape[1], num_feat=1024, gamma=0.5, ridge=1e-3, seed=args.seed)
    rff.fit(X, y)
    def proxy_score(batch_X):
        return rff.predict(batch_X.astype(np.float32))

    # 3) ES 在代理上搜，拿若干候选回投真仿真
    cand_vecs = es_optimize(proxy_score, K, iters=args.es_iters, pop=args.es_pop, sigma=0.8, keep=20, seed=args.seed)
    evals = []
    for x in cand_vecs:
        events = vec_to_events(x, K)
        z, extra = simulate_plan(events, tmax=TMAX, dt=0.1)
        evals.append((z, x, events, extra))
    # 若没有候选或真分都 0，做若干随机保底
    if len(evals)==0 or all(e[0] <= 1e-9 for e in evals):
        for _ in range(50):
            x = random_vec(5*K)
            events = vec_to_events(x, K)
            z, extra = simulate_plan(events, tmax=TMAX, dt=0.1)
            evals.append((z, x, events, extra))

    evals.sort(key=lambda e: -e[0])
    zbest, xbest, evbest, extra = evals[0]
    print(f"[Best] true z≈{zbest:.3f}s, events={len(evbest)}, per-missile={extra['per_m']}")

    # 4) 导出
    xlsx, jpath = export_plan(evbest, zbest, extra, outdir=args.outdir, stem="q5_solution_numpy")
    print("Exported:", xlsx, jpath)

if __name__ == "__main__":
    main()
