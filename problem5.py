# -*- coding: utf-8 -*-
"""
问题五：5 架无人机（每机至多 3 枚）对 M1/M2/M3 同时遮蔽的排程草案（完整、可运行、注释详尽）

目标理解：
- 希望在同一时刻，三枚导弹同时被各自的一朵云遮蔽（“同时遮蔽”）。
- 安排若干“时间槽”，每个槽内为 M1/M2/M3 各匹配一朵云，尽量保证在该槽区间内的交集时长最大。

实现：
- 选择槽中心 15/35/55/67 s；
- 优先使用 FY1–FY3，FY4–FY5 补位；
- 生成一份可执行排程表（result3.xlsx），用于后续在更高精度判定下迭代微调。
"""
from __future__ import annotations
import math
from typing import Tuple, List
import pandas as pd

# —— 常量与初始条件（完整自包含）——
g = 9.8
CLOUD_RADIUS = 10.0
CLOUD_ACTIVE = 20.0
CLOUD_SINK = 3.0
MISSILE_SPEED = 300.0
TARGET_CENTER_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0
M_INIT = {
    "M1": (20000.0, 0.0, 2000.0),
    "M2": (19000.0, 600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}
FY_INIT = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}

def uav_xy_pos(uid: str, v: float, hd: float, t: float):
    x0,y0,_ = FY_INIT[uid]
    return (x0 + v*t*math.cos(hd), y0 + v*t*math.sin(hd))

def explosion_point(uid: str, v: float, hd: float, td: float, te: float):
    z0 = FY_INIT[uid][2]
    tau = max(0.0, te - td)
    ze = z0 - 0.5 * g * tau * tau
    xe, ye = uav_xy_pos(uid, v, hd, te)
    return (xe, ye, ze)

def make_plan():
    # —— 槽中心时刻（可按需要插入/移动）——
    slots = [15.0, 35.0, 55.0, 67.0]
    # —— 每槽的 (UAV, Missile, Speed) 三元组 ——
    allocation = [
        (("FY1","M1",120.0), ("FY2","M2",120.0), ("FY3","M3",100.0)),
        (("FY1","M1",120.0), ("FY2","M2",120.0), ("FY3","M3",100.0)),
        (("FY4","M1",120.0), ("FY5","M2",120.0), ("FY1","M3",120.0)),
        (("FY4","M3",120.0), ("FY5","M1",120.0), ("FY2","M2",120.0)),
    ]
    # —— 引信延时（统一使用 2 s；可针对不同 UAV/Missile 微调）——
    tau = 2.0

    rows = []
    for idx, slot in enumerate(slots):
        triple = allocation[idx]
        for (uid, mid, v) in triple:
            x0,y0,_ = FY_INIT[uid]
            hd = math.atan2(-y0, -x0)  # 航向默认“指向原点”
            te = slot
            td = max(0.0, te - tau)
            xe, ye, ze = explosion_point(uid, v, hd, td, te)
            dx, dy = uav_xy_pos(uid, v, hd, td)
            rows.append({
                "UAV": uid,
                "Missile": mid,
                "Speed(m/s)": v,
                "Heading(rad)": round(hd,6),
                "DropTime(s)": round(td,2),
                "ExplodeTime(s)": round(te,2),
                "DropX": round(dx,3),
                "DropY": round(dy,3),
                "DropZ": FY_INIT[uid][2],
                "ExplodeX": round(xe,3),
                "ExplodeY": round(ye,3),
                "ExplodeZ": round(ze,3),
            })
    return rows

if __name__ == "__main__":
    rows = make_plan()
    df = pd.DataFrame(rows)
    df.index = df.index + 1
    out = "result3.xlsx"
    df.to_excel(out, index=True, index_label="index")
    print(f"[问题五] 已输出排程草案到 {out}")
    print("说明：该表用于驱动更高精度几何判定与进一步微调（按槽对三导弹实现同时遮蔽）。")
