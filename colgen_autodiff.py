# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, math, json, numpy as np, pandas as pd
from pathlib import Path
from geo_core import *

def event_mask(uid, mid, v, hd, tD, tau, time_grid):
    tE=tD+tau; N=len(time_grid); mask=np.zeros(N,dtype=np.int8)
    xE,yE=uav_xy(uid,v,hd,tE); z0=FY_INIT[uid][2]; zE=z0-0.5*g*(tau**2)
    for i,t in enumerate(time_grid):
        if tE<=t<=tE+20.0:
            c=(xE,yE, zE-3.0*(t-tE))
            if covered_hard(c, mid, t, z_samples=7): mask[i]=1
    return mask

def master_greedy(cands, time_grid, max_per_uav=3):
    dt=float(time_grid[1]-time_grid[0]); N=len(time_grid)
    chosen=[]; used={u:0 for u in ALL_UAVS}; last={u:-1e9 for u in ALL_UAVS}
    yM={m:np.zeros(N,dtype=np.int8) for m in ALL_MISSILES}
    def zscore(): return float((yM["M1"] & yM["M2"] & yM["M3"]).sum())*dt
    while True:
        base=zscore(); best=None
        for c in cands:
            if used[c['uid']]>=max_per_uav: continue
            if c['t_drop']-last[c['uid']]<1.0-1e-9: continue
            ytmp=yM[c['mid']].copy(); ytmp=np.maximum(ytmp,c['mask'])
            znew = ((yM["M1"] if c['mid']!="M1" else ytmp) & (yM["M2"] if c['mid']!="M2" else ytmp) & (yM["M3"] if c['mid']!="M3" else ytmp))
            inc=float(znew.sum())*dt - base
            if best is None or inc>best[0]: best=(inc,c)
        if best is None or best[0]<=1e-9: break
        inc,c=best; yM[c['mid']]=np.maximum(yM[c['mid']],c['mask']); chosen.append(c); used[c['uid']]+=1; last[c['uid']]=c['t_drop']
    return {"chosen":chosen, "score":zscore()}

def master_solve(cands, time_grid, max_per_uav=3):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return master_greedy(cands,time_grid,max_per_uav=max_per_uav), "greedy"
    dt=float(time_grid[1]-time_grid[0]); N=len(time_grid)
    model=cp_model.CpModel()
    x=[model.NewBoolVar(f"x{i}") for i in range(len(cands))]
    y=[[model.NewBoolVar(f"y{m}{t}") for t in range(N)] for m in range(3)]
    z=[model.NewBoolVar(f"z{t}") for t in range(N)]
    m2i={"M1":0,"M2":1,"M3":2}
    for m in range(3):
        for t in range(N):
            involved=[x[i] for i,c in enumerate(cands) if m2i[c['mid']]==m and c['mask'][t]==1]
            if involved: model.Add(y[m][t] <= sum(involved))
            else: model.Add(y[m][t]==0)
    for t in range(N):
        for m in range(3): model.Add(z[t] <= y[m][t])
    for uid in ALL_UAVS:
        model.Add(sum(x[i] for i,c in enumerate(cands) if c['uid']==uid) <= max_per_uav)
        lst=[(i,c) for i,c in enumerate(cands) if c['uid']==uid]; lst.sort(key=lambda ic: ic[1]['t_drop'])
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                if abs(lst[i][1]['t_drop']-lst[j][1]['t_drop'])<1.0-1e-9:
                    model.Add(x[lst[i][0]] + x[lst[j][0]] <= 1)
    model.Maximize(sum(z))
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=12.0; solver.parameters.num_search_workers=8
    st=solver.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return master_greedy(cands,time_grid,max_per_uav=max_per_uav), "greedy"
    chosen=[cands[i] for i in range(len(cands)) if solver.Value(x[i])==1]
    score=sum(solver.Value(z[t]) for t in range(N))*dt
    return {"chosen":chosen,"score":float(score)}, "milp"

def price_for_missile(mid, t_center, time_grid, w_t, rounds=30):
    # 软 UAV + 连续参数，简易爬山
    rng=np.random.RandomState(int(1000*t_center))
    logits_u=rng.randn(5)*0.1; theta=rng.randn(4)*0.1
    def unpack():
        pu=np.exp(logits_u-np.max(logits_u)); pu/=pu.sum()
        sig=lambda x:1/(1+np.exp(-x))
        v=80+60*sig(theta[0]); dh=math.radians(15.0)*np.tanh(theta[1])
        tD=t_center-1.5+3.0*sig(theta[2]); tau=0.8+2.2*sig(theta[3])
        x0=sum(pu[i]*FY_INIT[ALL_UAVS[i]][0] for i in range(5)); y0=sum(pu[i]*FY_INIT[ALL_UAVS[i]][1] for i in range(5))
        base_hd=math.atan2(-y0,-x0); hd=base_hd+dh
        return pu,v,hd,tD,tau
    def score(pu,v,hd,tD,tau,eps=2.0):
        s=0.0
        for i,t in enumerate(time_grid):
            c=cloud_center_from_params(pu,v,hd,tD,tau,t)
            if c is None: continue
            s += w_t[i]*soft_cover_value(c, mid, t, eps=eps, z_samples=7)
        return s
    best=None
    for it in range(rounds):
        pu,v,hd,tD,tau=unpack()
        sc=score(pu,v,hd,tD,tau); 
        if best is None or sc>best[0]: best=(sc,(pu,v,hd,tD,tau))
        # greedy coordinate steps
        for k in range(5):
            old=logits_u[k]; logits_u[k]+=0.2; pu2,v2,hd2,tD2,tau2=unpack(); sc2=score(pu2,v2,hd2,tD2,tau2)
            if sc2<=sc: logits_u[k]=old
        for k in range(4):
            old=theta[k]; theta[k]+=0.2; pu2,v2,hd2,tD2,tau2=unpack(); sc2=score(pu2,v2,hd2,tD2,tau2)
            if sc2<=sc: theta[k]=old
    pu,v,hd,tD,tau=unpack()
    u_idx=int(np.argmax(pu)); uid=ALL_UAVS[u_idx]
    return {"uid":uid,"mid":mid,"v":float(v),"hd":float(hd),"t_drop":float(tD),"tau":float(tau)}

def run(tmax=50.0, dt=0.5, windows=[14,28,42], rounds=2):
    time_grid=np.arange(0.0,tmax+1e-9,dt)
    cands=[]
    # 初始列
    for tc in windows:
        for mid in ALL_MISSILES:
            for uid in ALL_UAVS:
                base_hd=math.atan2(-FY_INIT[uid][1], -FY_INIT[uid][0])
                for dh in (-math.radians(8), math.radians(8)):
                    v=120.0; tau=2.0; tD=tc-2.0; hd=base_hd+dh
                    mask=event_mask(uid, mid, v, hd, tD, tau, time_grid)
                    if mask.sum()*float(dt)>=0.1:
                        cands.append({"uid":uid,"mid":mid,"v":v,"hd":hd,"t_drop":tD,"tau":tau,"mask":mask,"src":"init"})
    for r in range(rounds):
        sol,how=master_solve(cands,time_grid,max_per_uav=3)
        print(f"[Round {r+1}] {how} z≈{sol['score']:.2f}, chosen={len(sol['chosen'])}")
        # weights
        N=len(time_grid); yM={m:np.zeros(N,dtype=np.int8) for m in ALL_MISSILES}
        for c in sol['chosen']:
            yM[c['mid']]=np.maximum(yM[c['mid']], c['mask'])
        W={"M1":(yM["M2"] & yM["M3"]).astype(float),
           "M2":(yM["M1"] & yM["M3"]).astype(float),
           "M3":(yM["M1"] & yM["M2"]).astype(float)}
        added=0
        for tc in windows:
            for mid in ALL_MISSILES:
                cand=price_for_missile(mid, tc, time_grid, W[mid], rounds=20)
                mask=event_mask(cand["uid"], cand["mid"], cand["v"], cand["hd"], cand["t_drop"], cand["tau"], time_grid)
                gain=float((np.minimum(1, yM[mid]+mask) * W[mid]).sum())*float(dt)
                if gain>=0.15:
                    cand["mask"]=mask; cand["src"]="priced"; cands.append(cand); added+=1
        print("  added columns:", added, "total:", len(cands))
        if added==0: break
    # final solve
    sol,how=master_solve(cands,time_grid,max_per_uav=3)
    rows=[]
    for c in sol['chosen']:
        tE=c["t_drop"]+c["tau"]; xE,yE=uav_xy(c["uid"], c["v"], c["hd"], tE); z0=FY_INIT[c["uid"]][2]; zE=z0-0.5*g*(c["tau"]**2)
        xD,yD=uav_xy(c["uid"], c["v"], c["hd"], c["t_drop"])
        rows.append({"UAV":c["uid"],"Missile":c["mid"],"Speed(m/s)":round(c["v"],1),"Heading(rad)":round(c["hd"],6),
                     "DropTime(s)":round(c["t_drop"],2),"ExplodeTime(s)":round(tE,2),
                     "DropX":round(xD,2),"DropY":round(yD,2),"DropZ":round(z0,2),
                     "ExplodeX":round(xE,2),"ExplodeY":round(yE,2),"ExplodeZ":round(zE,2),"Source":c["src"]})
    df=pd.DataFrame(rows); xlsx="/mnt/data/colgen_demo.xlsx"; df.to_excel(xlsx,index=False)
    Path("/mnt/data/colgen_demo.json").write_text(json.dumps({"z_seconds":sol["score"],"chosen":len(sol["chosen"])},ensure_ascii=False,indent=2),encoding="utf-8")
    return xlsx, "/mnt/data/colgen_demo.json", sol["score"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tmax",type=float,default=50.0); ap.add_argument("--dt",type=float,default=0.5); ap.add_argument("--windows",type=str,default="14,28,42"); ap.add_argument("--rounds",type=int,default=2)
    args=ap.parse_args(); wins=[float(x) for x in args.windows.split(",")]; xlsx,js,score=run(tmax=args.tmax,dt=args.dt,windows=wins,rounds=args.rounds); print("Export:", xlsx, js, "z≈", round(score,2))
if __name__=="__main__": main()


