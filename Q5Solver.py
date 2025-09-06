# -*- coding: utf-8 -*-
# Q5Solver.py  (fixed argparse + full-CPU parallel + progress)
import os, math, time, json, argparse, random
from copy import deepcopy
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from scipy.optimize import differential_evolution
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------- constants --------------------
g = 9.8
MISSILE_SPEED = 300.0
SMOG_R = 10.0
SMOG_SINK = 3.0
SMOG_T = 20.0

FAKE = np.array([0.,0.,0.], float)
CYL_R, CYL_H = 7.0, 10.0
CYL_BASE = np.array([0.,200.,0.], float)

M_INITS = [
    np.array([20000.,    0., 2000.], float),
    np.array([19000.,  600., 2100.], float),
    np.array([18000., -600., 1900.], float),
]
FY_INITS = [
    np.array([17800.,     0., 1800.], float),
    np.array([12000.,  1400., 1400.], float),
    np.array([ 6000., -3000.,  700.], float),
    np.array([11000.,  2000., 1800.], float),
    np.array([13000., -2000., 1300.], float),
]

UAV_MIN, UAV_MAX = 70., 140.
MAX_BOMBS, MIN_GAP = 3, 1.0
DEFAULT_DELAY = 3.6
EPS = 1e-12

# -------------------- utils --------------------
def unit(v): 
    n=np.linalg.norm(v); 
    return v if n<EPS else v/n

def missile_pos(t, m0):
    return m0 + MISSILE_SPEED*unit(FAKE - m0)*t

def uav_state(t, u0, spd, hd):
    return np.array([u0[0]+spd*math.cos(hd)*t, u0[1]+spd*math.sin(hd)*t, u0[2]], float)

def explosion(u0, spd, hd, t_drop, delay):
    p_drop = uav_state(t_drop, u0, spd, hd)
    p_expl = p_drop + np.array([spd*math.cos(hd)*delay, spd*math.sin(hd)*delay, -0.5*g*delay*delay], float)
    return p_expl, t_drop+delay

_CYL_CACHE={}
def sample_cylinder(nphi, nz, dtype):
    key=(nphi,nz,np.dtype(dtype).name)
    if key in _CYL_CACHE: return _CYL_CACHE[key]
    b=CYL_BASE.astype(dtype); r,h=dtype(CYL_R),dtype(CYL_H)
    ph=np.linspace(0,2*np.pi,nphi,endpoint=False,dtype=dtype)
    ring=np.stack([r*np.cos(ph), r*np.sin(ph), np.zeros_like(ph)],1)
    pts=[b+ring, b+np.array([0,0,h],dtype=dtype)+ring]
    if nz>=2:
        for z in np.linspace(0,h,nz,dtype=dtype):
            pts.append(b+np.array([0,0,z],dtype=dtype)+ring)
    arr=np.vstack(pts).astype(dtype); _CYL_CACHE[key]=arr; return arr

# -------------------- block eval --------------------
def _blk_eval(args):
    start, t_blk, m0, epos, et0, pts, mode = args
    Tb=t_blk.size; mask=np.zeros(Tb,bool)
    for i in range(Tb):
        t=float(t_blk[i])
        alive=np.where((t>=et0) & (t<=et0+SMOG_T))[0]
        if alive.size==0: continue
        m=missile_pos(t,m0)
        c=epos[alive].copy(); c[:,2] -= SMOG_SINK*(t-et0[alive])
        v=c-m; l=np.linalg.norm(v,axis=1)
        if np.any(l<=SMOG_R+EPS): mask[i]=True; continue
        cosA=np.sqrt(np.maximum(0.0, 1.0-(SMOG_R/l)**2))
        w=pts-m; wn=np.linalg.norm(w,axis=1)
        lhs = w @ v.T
        rhs = (wn[:,None])*(l*cosA)[None,:]
        if mode==0:
            mask[i] = bool(np.any(np.all(lhs+1e-12>=rhs, axis=0)))
        else:
            mask[i] = bool(np.all(np.any(lhs+1e-12>=rhs, axis=1)))
    return start, mask

class TimeBlockEvaluator:
    def __init__(self, backend="process", workers=None, base_chunk=1400, min_tasks_factor=3, mkl_threads=None):
        self.backend=backend
        self.workers=workers or os.cpu_count()
        self.base_chunk=base_chunk
        self.min_tasks_factor=max(1,int(min_tasks_factor))
        if backend=="process":
            os.environ["MKL_NUM_THREADS"]="1"
            os.environ["OMP_NUM_THREADS"]="1"
            self.pool=ProcessPoolExecutor(max_workers=self.workers)
        else:
            if mkl_threads is not None and mkl_threads>0:
                os.environ["MKL_NUM_THREADS"]=str(int(mkl_threads))
                os.environ["OMP_NUM_THREADS"]=str(int(mkl_threads))
            self.pool=ThreadPoolExecutor(max_workers=self.workers)

    def shutdown(self):
        try: self.pool.shutdown(wait=True, cancel_futures=True)
        except Exception: pass

    def eval_union_missile(self, m0, epos, et0, dt, t0, t1, pts, mode):
        t_grid=np.arange(t0,t1+1e-12,dt,dtype=pts.dtype)
        if t_grid.size==0: return t_grid, np.zeros(0,bool), 0.0
        min_tasks = self.workers * self.min_tasks_factor
        chunk_eff = max(self.base_chunk, max(1, int(math.ceil(t_grid.size / max(min_tasks,1)))))
        tasks=[]
        for s in range(0, t_grid.size, chunk_eff):
            tasks.append((s, t_grid[s:s+chunk_eff], m0, epos, et0, pts, mode))
        out=np.zeros(t_grid.size,bool)
        futs={ self.pool.submit(_blk_eval, t): t[0] for t in tasks }
        for fu in as_completed(futs):
            s=futs[fu]; s2,m=fu.result(); out[s:s+m.size]=m
        return t_grid, out, float(np.count_nonzero(out)*dt)

# -------------------- evaluate --------------------
def expand_solution(sol, delays=None, dtype=np.float64):
    epos,et0=[],[]
    for u,(hd,spd,drops) in enumerate(sol):
        u0=FY_INITS[u]; last=-1e9
        for j,t in enumerate(drops):
            if t<=0: continue
            if t-last<MIN_GAP: t=last+MIN_GAP
            last=t
            dly = (delays[u][j] if (delays and delays[u][j]>0) else DEFAULT_DELAY)
            p,t0=explosion(u0,spd,hd,t,dly)
            if p[2]>0: epos.append(p); et0.append(t0)
    if not epos: return np.zeros((0,3)), np.zeros((0,))
    return np.stack(epos,0).astype(dtype), np.array(et0,dtype=dtype)

def evaluate(sol, dt, nphi, nz, evaluator: TimeBlockEvaluator, fp32,
             mode_anycloud=True, delays=None, lambda_overlap=0.0, sigma=2.5):
    dtype=np.float32 if fp32 else np.float64
    pts=sample_cylinder(nphi,nz,dtype)
    epos,et0=expand_solution(sol,delays,dtype)
    if epos.shape[0]==0:
        return 0., {"per_missile_seconds":[0.,0.,0.], "tGrids":[np.array([])]*3, "masks":[np.array([],bool)]*3}
    mode=1 if mode_anycloud else 0
    per,tGs,mks=[],[],[]
    for m0 in M_INITS:
        hit=float(np.linalg.norm(m0-FAKE)/MISSILE_SPEED)
        t0=float(np.min(et0)); t1=min(float(np.max(et0))+SMOG_T, hit)
        if t1<=t0:
            tGs.append(np.array([],dtype)); mks.append(np.array([],bool)); per.append(0.); continue
        tg,mk,sec = evaluator.eval_union_missile(m0, epos, et0, dt, t0, t1, pts, mode)
        tGs.append(tg); mks.append(mk); per.append(sec)
    total=float(sum(per))
    if lambda_overlap>0 and et0.size>=2:
        d=np.diff(np.sort(et0.astype(float))); d=d[d<=3.0*sigma]
        if d.size: total -= lambda_overlap*float(np.sum(np.exp(-(d*d)/(2*sigma*sigma))))
    return total, {"per_missile_seconds":per, "tGrids":tGs, "masks":mks}

# -------------------- seed & SA --------------------
def greedy_seed(evaluator, dt=0.02, nphi=240, nz=7, fp32=False):
    sol=[]
    for u,u0 in enumerate(FY_INITS):
        hd=math.atan2(-u0[1], -u0[0]); sol.append([hd,120.0,[0.,0.,0.]])
    hits=[np.linalg.norm(m-FAKE)/MISSILE_SPEED for m in M_INITS]
    tmin=max(0.5, min(hits)-8.0); tmax=max(hits)-1.0
    cand=np.arange(tmin,tmax,1.0,float)
    base,_=evaluate(sol,dt,nphi,nz,evaluator,fp32,True); used=0
    while used<15:
        best=None; gain=-1e-9
        for u in range(5):
            drops=sol[u][2]; k=sum(1 for x in drops if x>0)
            if k>=MAX_BOMBS: continue
            prev=[x for x in drops if x>0]
            for t in cand:
                if any(abs(t-p)<MIN_GAP for p in prev): continue
                tmp=deepcopy(sol); tmp[u][2][k]=float(t)
                val,_=evaluate(tmp,dt,nphi,nz,evaluator,fp32,True)
                if val-base>gain: gain,best = val-base,(u,k,float(t),val)
        if not best or gain<0.05: break
        u,k,t,val=best; sol[u][2][k]=t; base=val; used+=1
        print(f"[SEED] FY{u+1} add #{k+1}@{t:.2f}s gain=+{gain:.3f}s used={used}")
    return sol

def clamp(x,a,b): return a if x<a else (b if x>b else x)

def perturb(sol):
    s=deepcopy(sol); i=random.randrange(5); hd,spd,dr=s[i]
    r=random.random()
    if r<0.33:
        s[i][0]=(hd+math.radians(random.uniform(-15,15))+2*math.pi)%(2*math.pi)
    elif r<0.66:
        s[i][1]=clamp(spd+random.uniform(-10,10),UAV_MIN,UAV_MAX)
    else:
        j=random.randrange(MAX_BOMBS); base=dr[j]
        dr[j] = random.uniform(3,10) if (base<=0 and random.random()<0.5) else max(0.0, base+random.uniform(-2,2))
        arr=sorted([x for x in dr if x>0]); fixed=[]; last=-1e9
        for t in arr:
            if t-last<MIN_GAP: t=last+MIN_GAP
            fixed.append(t); last=t
        while len(fixed)<MAX_BOMBS: fixed.append(0.0)
        s[i][2]=fixed[:MAX_BOMBS]
    return s

def simulated_annealing(evaluator, dt,nphi,nz, fp32, iters,batch,restarts,
                        lambda_overlap=0.0, sigma=2.5,
                        progress_every=20, time_budget=None,
                        batch_parallel=True, outer_workers=None):
    bestV,bestS,bestD=None,None,None
    eval_count=0; t0=time.time()
    outer_workers = outer_workers or max(2, (os.cpu_count()//2))
    outer_pool = ThreadPoolExecutor(max_workers=outer_workers) if batch_parallel else None
    try:
        for rs in range(restarts):
            cur=greedy_seed(evaluator,0.02,240,7,fp32)
            curV,curD=evaluate(cur,dt,nphi,nz,evaluator,fp32,True,None,lambda_overlap,sigma); eval_count+=1
            if bestV is None or curV>bestV:
                bestV,bestS,bestD=curV,deepcopy(cur),curD
                print(f"[INFO] Restart {rs+1}/{restarts} init best={bestV:.3f}s per={bestD['per_missile_seconds']}")
            T=max(1.0,0.25*(bestV if bestV>0 else 10.0)+5.0)
            for it in range(1,iters+1):
                cands=[perturb(cur) for _ in range(batch)]
                results=[None]*batch
                if batch_parallel and outer_pool:
                    futs={ outer_pool.submit(evaluate, c, dt,nphi,nz,evaluator,fp32,True,None,lambda_overlap,sigma): idx
                           for idx,c in enumerate(cands) }
                    for fu in as_completed(futs):
                        idx=futs[fu]; v,d=fu.result(); results[idx]=(v,d)
                else:
                    for bi,c in enumerate(cands):
                        v,d=evaluate(c,dt,nphi,nz,evaluator,fp32,True,None,lambda_overlap,sigma)
                        results[bi]=(v,d)

                acc=0
                for (v,d),c in zip(results,cands):
                    eval_count+=1
                    dv=v-curV
                    if dv>=0 or math.exp(dv/max(T,1e-6))>random.random():
                        cur,curV,curD=c,v,d; acc+=1
                        if curV>bestV:
                            bestV,bestS,bestD=curV,deepcopy(cur),curD
                            pm=bestD['per_missile_seconds']
                            print(f"[INFO] New best={bestV:.3f}s | M1={pm[0]:.2f} M2={pm[1]:.2f} M3={pm[2]:.2f} | eval={eval_count} | {time.time()-t0:.1f}s")
                T*=0.985

                if it % max(1,progress_every)==0:
                    elapsed=time.time()-t0; eps=eval_count/max(elapsed,1e-6)
                    done=(rs*iters + it)/max(restarts*iters,1); eta=(elapsed/done - elapsed) if done>0 else float('inf')
                    pm=curD['per_missile_seconds']
                    print(f"[SA] rs={rs+1}/{restarts} it={it}/{iters} "
                          f"T={T:.2f} cur={curV:.2f} best={bestV:.2f} acc={acc}/{batch} "
                          f"eval={eval_count} ({eps:.1f}/s) ETA≈{eta:.1f}s pm={pm}")
                if (time_budget is not None) and (time.time()-t0)>time_budget:
                    print("[WARN] hit time budget, stop SA early.")
                    return bestS,bestV,bestD
        return bestS,bestV,bestD
    finally:
        if outer_pool: outer_pool.shutdown(wait=True, cancel_futures=True)

# -------------------- optional DE --------------------
def local_refine_DE(evaluator, sol, dt,nphi,nz, fp32, use_delay=True):
    if not SCIPY_OK:
        print("[WARN] SciPy 不可用，跳过 DE 精修"); return sol
    def bounds_u(u):
        hd,spd,dr=sol[u]
        B=[((hd-math.radians(12))%(2*np.pi),(hd+math.radians(12))%(2*np.pi)),
           (max(UAV_MIN,spd-8), min(UAV_MAX,spd+8))]
        for k in range(MAX_BOMBS):
            t=dr[k]
            B.append((0,0) if t<=0 else (max(0.5,t-1.5), t+1.5))
        if use_delay: B.append((0.2,12.0))
        return B
    def pack(u,x):
        hd,sp=x[0],x[1]; drops=sol[u][2].copy(); idx=2
        for k in range(MAX_BOMBS):
            if drops[k]>0: drops[k]=x[idx]; idx+=1
        dly=x[idx] if use_delay else DEFAULT_DELAY
        return hd,sp,drops,dly
    def obj(u,x):
        hd,sp,drops,dly=pack(u,x)
        tmp=deepcopy(sol); tmp[u]=[hd,sp,list(drops)]
        val,_=evaluate(tmp,dt,nphi,nz,evaluator,fp32,True,[[dly]*MAX_BOMBS for _ in range(5)])
        return -val
    out=deepcopy(sol)
    with ProcessPoolExecutor(max_workers=min(5, os.cpu_count())) as ex:
        futs=[ex.submit(lambda uu: (uu, differential_evolution(lambda xx: obj(uu,xx),
                                                               bounds_u(uu), workers=-1, updating='deferred',
                                                               popsize=20, maxiter=50,
                                                               mutation=(0.6,1.0), recombination=0.9,
                                                               tol=1e-3, disp=False)), u) for u in range(5)]
        for fu in as_completed([f[0] for f in futs]):
            pass
    return out

# -------------------- output --------------------
def write_excel(xlsx, sol, detail):
    from openpyxl import Workbook
    wb=Workbook(); ws1=wb.active; ws1.title="Plan"
    ws1.append(["UAV","HeadingDeg","Speed","BombIndex","DropTime","ExplodeTime","ExplodeX","ExplodeY","ExplodeZ"])
    for i,(hd,spd,drops) in enumerate(sol):
        u0=FY_INITS[i]; last=-1e9
        for j,t in enumerate(drops):
            if t<=0:
                ws1.append([f"FY{i+1}", (hd*180/math.pi)%360.0, spd, j+1, 0, 0, None,None,None]); continue
            if t-last<MIN_GAP: t=last+MIN_GAP
            last=t
            p,t0=explosion(u0,spd,hd,t,DEFAULT_DELAY)
            ws1.append([f"FY{i+1}", (hd*180/math.pi)%360.0, spd, j+1, t, t0,
                        None if p[2]<=0 else float(p[0]),
                        None if p[2]<=0 else float(p[1]),
                        None if p[2]<=0 else float(p[2])])
    ws2=wb.create_sheet("Summary")
    pm=detail["per_missile_seconds"]; total=float(sum(pm))
    for r in [["Missile","Occlusion_s"],["M1",pm[0]],["M2",pm[1]],["M3",pm[2]],["Total",total]]:
        ws2.append(r)
    wb.save(xlsx); print(f"[INFO] result written -> {xlsx}")
    return total, pm

def save_report(path, args, total, pm, sol):
    with open(path,"w",encoding="utf-8") as f:
        f.write("Q5 Progress MAX Report\n"+"="*70+"\n\n")
        f.write("Args:\n"+json.dumps(vars(args),indent=2,ensure_ascii=False)+"\n\n")
        f.write(f"Total Occlusion = {total:.3f}s\nPer Missile: M1={pm[0]:.3f}s  M2={pm[1]:.3f}s  M3={pm[2]:.3f}s\n\n")
        f.write("Solution:\n")
        for i,(hd,spd,drops) in enumerate(sol):
            f.write(f" FY{i+1}: heading {(hd*180/math.pi)%360:.2f}°, speed {spd:.2f}, drops {drops}\n")
    print(f"[INFO] report saved -> {path}")

# -------------------- CLI --------------------
def main():
    ap=argparse.ArgumentParser("Q5 high-parallel solver with progress")
    # 粗评
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--nphi", type=int, default=720)
    ap.add_argument("--nz",   type=int, default=11)
    # 细评
    ap.add_argument("--refine_dt", type=float, default=0.005)
    ap.add_argument("--refine_nphi", type=int, default=960)
    ap.add_argument("--refine_nz",   type=int, default=13)

    ap.add_argument("--backend", choices=["process","thread"], default="process")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1400)
    ap.add_argument("--min_tasks_factor", type=int, default=3, help="每次评估最少任务数≈workers×factor")
    ap.add_argument("--mkl_threads", type=int, default=None, help="仅 thread 模式有效")
    ap.add_argument("--fp32", action="store_true")

    ap.add_argument("--sa_iters", type=int, default=1200)
    ap.add_argument("--sa_batch", type=int, default=12)
    ap.add_argument("--sa_restarts", type=int, default=2)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--time_budget", type=float, default=None)

    ap.add_argument("--lambda_overlap", type=float, default=0.0)
    ap.add_argument("--overlap_sigma", type=float, default=2.5)

    # ✅ 修正布尔开关：store_true/store_false
    ap.add_argument("--batch_parallel", action="store_true", help="外层并行开启")
    ap.add_argument("--no_batch_parallel", dest="batch_parallel", action="store_false", help="关闭外层并行")
    ap.set_defaults(batch_parallel=True)

    ap.add_argument("--outer_workers", type=int, default=None, help="外层并行线程数（默认=CPU/2）")
    ap.add_argument("--de_refine", action="store_true")

    ap.add_argument("--xlsx", default="result3.xlsx")
    ap.add_argument("--report", default="Result3_report.txt")
    args=ap.parse_args()

    if args.backend=="thread" and (args.mkl_threads is None):
        print("[INFO] thread 后端：未指定 --mkl_threads，MKL 将自动调度线程。")

    print("="*78)
    print(f"[RUN] backend={args.backend} workers={args.workers or 'auto'} "
          f"chunk={args.chunk} min_tasks_factor={args.min_tasks_factor} mkl_threads={args.mkl_threads or 'auto'} fp32={args.fp32}")
    print(f"[RUN] SA iters={args.sa_iters} batch={args.sa_batch} restarts={args.sa_restarts} batch_parallel={args.batch_parallel} outer_workers={args.outer_workers or 'auto'}")
    print("="*78)

    t0=time.time()
    evaluator = TimeBlockEvaluator(args.backend, args.workers, args.chunk, args.min_tasks_factor, args.mkl_threads)

    try:
        sol, bestV, detail = simulated_annealing(
            evaluator, args.dt, args.nphi, args.nz, args.fp32,
            iters=args.sa_iters, batch=args.sa_batch, restarts=args.sa_restarts,
            lambda_overlap=args.lambda_overlap, sigma=args.overlap_sigma,
            progress_every=args.progress_every, time_budget=args.time_budget,
            batch_parallel=args.batch_parallel, outer_workers=args.outer_workers
        )

        if args.de_refine:
            print("[INFO] DE refinement ...")
            sol = local_refine_DE(evaluator, sol, args.dt, args.nphi, args.nz, args.fp32, True)

        print(f"[INFO] Refining measurement dt={args.refine_dt}, nphi={args.refine_nphi}, nz={args.refine_nz}")
        total, detail = evaluate(sol, args.refine_dt, args.refine_nphi, args.refine_nz, evaluator, args.fp32, True)
        pm = detail["per_missile_seconds"]
        _, _ = write_excel(args.xlsx, sol, detail)
        save_report(args.report, args, float(sum(pm)), pm, sol)

        print("-"*70)
        print(f"[RESULT] Total={sum(pm):.3f}s | M1={pm[0]:.3f}  M2={pm[1]:.3f}  M3={pm[2]:.3f}")
        print(f"[INFO] Runtime = {time.time()-t0:.2f}s")
    finally:
        evaluator.shutdown()

if __name__=="__main__":
    main()
