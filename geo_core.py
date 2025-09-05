# -*- coding: utf-8 -*-
from __future__ import annotations
import math, numpy as np
g=9.8; CLOUD_RADIUS=10.0; CLOUD_ACTIVE=20.0; CLOUD_SINK=3.0; MISSILE_SPEED=300.0
TARGET_CENTER_XY=(0.0,200.0); TARGET_Z0, TARGET_Z1=0.0,10.0
M_INIT={"M1":(20000.0,0.0,2000.0),"M2":(19000.0,600.0,2100.0),"M3":(18000.0,-600.0,1900.0)}
FY_INIT={"FY1":(17800.0,0.0,1800.0),"FY2":(12000.0,1400.0,1400.0),"FY3":(6000.0,-3000.0,700.0),"FY4":(11000.0,2000.0,1800.0),"FY5":(13000.0,-2000.0,1300.0)}
ALL_UAVS=list(FY_INIT.keys()); ALL_MISSILES=["M1","M2","M3"]
def normalize(v): x,y,z=v; n=math.hypot(x,math.hypot(y,z)); return (x/n,y/n,z/n) if n>0 else (0.0,0.0,0.0)
def missile_pos(mid,t): x0,y0,z0=M_INIT[mid]; dx,dy,dz=normalize((-x0,-y0,-z0)); return (x0+dx*MISSILE_SPEED*t,y0+dy*MISSILE_SPEED*t,z0+dz*MISSILE_SPEED*t)
def uav_xy(uid,v,hd,t): x0,y0,_=FY_INIT[uid]; return (x0+v*t*math.cos(hd), y0+v*t*math.sin(hd))
def point_seg_dist(p,a,b):
    ax,ay,az=a; bx,by,bz=b; px,py,pz=p; ab=(bx-ax,by-ay,bz-az); ap=(px-ax,py-ay,pz-az)
    ab2=ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]
    if ab2==0: dx,dy,dz=(px-ax,py-ay,pz-az); return math.sqrt(dx*dx+dy*dy+dz*dz)
    t=(ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/ab2; t=0.0 if t<0 else (1.0 if t>1 else t)
    q=(ax+ab[0]*t, ay+ab[1]*t, az+ab[2]*t); dx,dy,dz=(px-q[0],py-q[1],pz-q[2]); return math.sqrt(dx*dx+dy*dy+dz*dz)
def covered_hard(c_center, mid, t, z_samples=7):
    m=missile_pos(mid,t)
    for k in range(z_samples):
        z=TARGET_Z0+(TARGET_Z1-TARGET_Z0)*(k/(z_samples-1) if z_samples>1 else 0.5)
        tgt=(TARGET_CENTER_XY[0],TARGET_CENTER_XY[1],z)
        if point_seg_dist(c_center,m,tgt)<=CLOUD_RADIUS: return True
    return False
def sigmoid(x): import numpy as _np; return 1.0/(1.0+_np.exp(-x))
def soft_cover_value(c_center, mid, t, eps=2.0, z_samples=7):
    m=missile_pos(mid,t); dmin=1e9
    for k in range(z_samples):
        z=TARGET_Z0+(TARGET_Z1-TARGET_Z0)*(k/(z_samples-1) if z_samples>1 else 0.5)
        tgt=(TARGET_CENTER_XY[0],TARGET_CENTER_XY[1],z)
        d=point_seg_dist(c_center,m,tgt); dmin=min(dmin,d)
    return sigmoid((CLOUD_RADIUS-dmin)/max(1e-6,eps))
def cloud_center_from_params(u_probs, v, heading, t_drop, tau, t):
    xy=np.zeros(2,dtype=float); z0=0.0; t_exp=t_drop+tau
    for w,uid in zip(u_probs, ALL_UAVS):
        x0,y0,z=FY_INIT[uid]
        x=x0+v*t_exp*math.cos(heading); y=y0+v*t_exp*math.sin(heading)
        xy += w*np.array([x,y]); z0 += w*z
    if not (t_exp <= t <= t_exp+20.0): return None
    z_e = z0 - 0.5*g*(tau**2)
    return (xy[0], xy[1], z_e - 3.0*(t - t_exp))
