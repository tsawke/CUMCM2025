# -*- coding: utf-8 -*-
"""
2025年全国大学生数学建模竞赛A题最终完整解决方案
无人机烟幕干扰优化问题

本文件包含三个问题的完整代码实现：
1. 问题一：FY1单次投放对M1的有效遮蔽时长计算
2. 问题二：多无人机协同干扰优化
3. 问题三：多导弹同时干扰的复杂优化问题

作者：AI助手
日期：2025年
"""

import math
import json
import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict
import copy
import time

# =====================
# 一、问题参数定义
# =====================

# 物理常量
g = 9.8
CLOUD_RADIUS = 10.0
CLOUD_ACTIVE = 20.0
CLOUD_SINK = 3.0
MISSILE_SPEED = 300.0

# 目标参数
TRUE_TARGET_XY = (0.0, 200.0)
TARGET_Z0, TARGET_Z1 = 0.0, 10.0
FAKE_TARGET_XY = (0.0, -200.0)

# 初始位置
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

# =====================
# 二、几何与物理计算函数
# =====================

def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """向量单位化"""
    x, y, z = v
    n = math.hypot(x, math.hypot(y, z))
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def missile_pos(missile_id: str, t: float) -> Tuple[float, float, float]:
    """导弹在时刻t的位置"""
    x0, y0, z0 = M_INIT[missile_id]
    dx, dy, dz = normalize((-x0, -y0, -z0))
    return (x0 + dx*MISSILE_SPEED*t,
            y0 + dy*MISSILE_SPEED*t,
            z0 + dz*MISSILE_SPEED*t)

def uav_xy_pos(uav_id: str, v: float, heading_rad: float, t: float) -> Tuple[float, float]:
    """无人机在时刻t的水平位置"""
    x0, y0, _ = FY_INIT[uav_id]
    return (x0 + v * t * math.cos(heading_rad),
            y0 + v * t * math.sin(heading_rad))

def explosion_point(uav_id: str, v: float, heading_rad: float, t_drop: float, t_explode: float) -> Tuple[float, float, float]:
    """计算干扰弹起爆点坐标"""
    xe, ye = uav_xy_pos(uav_id, v, heading_rad, t_explode)
    z0 = FY_INIT[uav_id][2]
    tau = max(0.0, t_explode - t_drop)
    ze = z0 - 0.5 * g * tau * tau
    return (xe, ye, ze)

def cloud_center_at(cE: Tuple[float, float, float], t_explode: float, t: float) -> Tuple[float, float, float]:
    """起爆后的云团中心位置"""
    return (cE[0], cE[1], cE[2] - CLOUD_SINK * (t - t_explode))

def point_seg_dist(p: Tuple[float, float, float],
                   a: Tuple[float, float, float],
                   b: Tuple[float, float, float]) -> float:
    """点p到线段ab的最短距离"""
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    ab = (bx - ax, by - ay, bz - az)
    ap = (px - ax, py - ay, pz - az)
    ab2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    if ab2 == 0.0:
        dx, dy, dz = (px - ax, py - ay, pz - az)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab2
    t = max(0.0, min(1.0, t))
    q = (ax + ab[0]*t, ay + ab[1]*t, az + ab[2]*t)
    dx, dy, dz = (px - q[0], py - q[1], pz - q[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def covered_at_time(c_center: Tuple[float, float, float],
                    missile_id: str,
                    t: float,
                    z_samples: int = 5) -> bool:
    """判断在时刻t是否有效遮蔽"""
    m = missile_pos(missile_id, t)
    for k in range(z_samples):
        z = TARGET_Z0 + (TARGET_Z1 - TARGET_Z0) * (k / (z_samples - 1) if z_samples > 1 else 0.5)
        tgt = (TRUE_TARGET_XY[0], TRUE_TARGET_XY[1], z)
        if point_seg_dist(c_center, m, tgt) <= CLOUD_RADIUS:
            return True
    return False

def coverage_time_for_plan(uav_id: str,
                           missile_id: str,
                           v: float,
                           heading_rad: float,
                           t_drop: float,
                           t_explode: float,
                           t_start: float = 0.0,
                           t_end: float = 100.0,
                           dt: float = 0.05,
                           z_samples: int = 5) -> float:
    """计算有效遮蔽时长"""
    cE = explosion_point(uav_id, v, heading_rad, t_drop, t_explode)
    t0 = max(t_start, t_explode)
    t1 = min(t_end, t_explode + CLOUD_ACTIVE)
    if t0 >= t1:
        return 0.0
    t = t0
    covered = 0.0
    while t <= t1 + 1e-12:
        c = cloud_center_at(cE, t_explode, t)
        if covered_at_time(c, missile_id, t, z_samples=z_samples):
            covered += dt
        t += dt
    return covered

# =====================
# 三、问题一：FY1单次投放对M1的有效遮蔽时长
# =====================

def solve_problem1():
    """问题一：FY1以120m/s朝假目标方向飞行，t=1.5s投放，3.6s后起爆"""
    print("="*60)
    print("问题一：FY1单次投放对M1的有效遮蔽时长计算")
    print("="*60)
    
    # 参数设置
    uav_id = "FY1"
    v = 120.0
    x0, y0, _ = FY_INIT[uav_id]
    heading = math.atan2(-y0, -x0)  # 指向原点的航向角
    
    t_drop = 1.5
    t_explode = t_drop + 3.6
    
    print(f"无人机：{uav_id}")
    print(f"速度：{v} m/s")
    print(f"航向角：{math.degrees(heading):.2f}°")
    print(f"投放时刻：{t_drop} s")
    print(f"起爆时刻：{t_explode} s")
    
    # 计算有效遮蔽时长
    seconds = coverage_time_for_plan(
        uav_id=uav_id,
        missile_id="M1",
        v=v,
        heading_rad=heading,
        t_drop=t_drop,
        t_explode=t_explode,
        t_start=0.0,
        t_end=100.0,
        dt=0.05,
        z_samples=5
    )
    
    print(f"有效遮蔽时长：{seconds:.3f} s")
    
    # 计算关键位置
    explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
    drop_pos = uav_xy_pos(uav_id, v, heading, t_drop)
    drop_z = FY_INIT[uav_id][2]
    
    print(f"投放点：({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_z:.1f})")
    print(f"起爆点：({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
    
    # 保存结果
    result_data = {
        "无人机运动方向": math.degrees(heading) % 360,
        "无人机运动速度 (m/s)": v,
        "烟幕干扰弹编号": 1,
        "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
        "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
        "烟幕干扰弹投放点的z坐标 (m)": drop_z,
        "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
        "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
        "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
        "有效干扰时长 (s)": seconds
    }
    
    # 导出到CSV
    df = pd.DataFrame([result_data])
    df.to_csv("result1_final.csv", index=False, encoding='utf-8-sig')
    print(f"结果已保存到 result1_final.csv")
    
    return seconds, result_data

# =====================
# 四、问题二：多无人机协同干扰优化
# =====================

def solve_problem2():
    """问题二：多无人机协同干扰优化"""
    print("\n" + "="*60)
    print("问题二：多无人机协同干扰优化")
    print("="*60)
    
    # 使用改进的网格搜索优化
    best_total_coverage = 0.0
    best_solution = None
    
    # 为每架无人机尝试不同的航向和投放时机
    uav_ids = ["FY1", "FY2", "FY3"]
    v = 120.0
    
    # 更精细的搜索空间
    headings = np.linspace(0, 2*math.pi, 16)  # 16个方向
    drop_times = np.linspace(0.5, 8.0, 16)   # 16个时间点
    
    print("正在搜索最优解...")
    total_combinations = len(headings)**3 * len(drop_times)**3
    print(f"总搜索空间：{total_combinations} 种组合")
    
    count = 0
    for h1 in headings:
        for t1 in drop_times:
            for h2 in headings:
                for t2 in drop_times:
                    for h3 in headings:
                        for t3 in drop_times:
                            count += 1
                            if count % 10000 == 0:
                                print(f"已搜索 {count}/{total_combinations} 种组合")
                            
                            total_coverage = 0.0
                            solution = []
                            
                            for i, (uav_id, heading, t_drop) in enumerate([("FY1", h1, t1), ("FY2", h2, t2), ("FY3", h3, t3)]):
                                t_explode = t_drop + 3.6
                                
                                coverage = coverage_time_for_plan(
                                    uav_id=uav_id,
                                    missile_id="M1",
                                    v=v,
                                    heading_rad=heading,
                                    t_drop=t_drop,
                                    t_explode=t_explode,
                                    dt=0.1,  # 降低精度以提高速度
                                    z_samples=3
                                )
                                
                                total_coverage += coverage
                                solution.append({
                                    "uav_id": uav_id,
                                    "heading": heading,
                                    "t_drop": t_drop,
                                    "coverage": coverage
                                })
                            
                            if total_coverage > best_total_coverage:
                                best_total_coverage = total_coverage
                                best_solution = solution
    
    print(f"最优总遮蔽时长：{best_total_coverage:.3f} s")
    
    # 生成结果数据
    results_data = []
    for sol in best_solution:
        uav_id = sol["uav_id"]
        heading = sol["heading"]
        t_drop = sol["t_drop"]
        t_explode = t_drop + 3.6
        v = 120.0
        
        # 计算位置
        drop_pos = uav_xy_pos(uav_id, v, heading, t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
        drop_z = FY_INIT[uav_id][2]
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向": math.degrees(heading) % 360,
            "无人机运动速度 (m/s)": v,
            "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标 (m)": drop_z,
            "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
            "有效干扰时长 (s)": sol["coverage"]
        }
        
        results_data.append(result_data)
        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={sol['coverage']:.3f}s")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    df.to_csv("result2_final.csv", index=False, encoding='utf-8-sig')
    print(f"结果已保存到 result2_final.csv")
    
    return results_data

# =====================
# 五、问题三：多导弹同时干扰的复杂优化
# =====================

def solve_problem3():
    """问题三：多导弹同时干扰的复杂优化"""
    print("\n" + "="*60)
    print("问题三：多导弹同时干扰的复杂优化")
    print("="*60)
    
    # 使用启发式方法：为每架无人机分配主要干扰的导弹
    uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    missile_ids = ["M1", "M2", "M3"]
    v = 120.0
    
    results_data = []
    
    # 为每架无人机计算最优参数
    for i, uav_id in enumerate(uav_ids):
        print(f"正在优化 {uav_id}...")
        
        # 计算指向假目标的航向
        x0, y0, _ = FY_INIT[uav_id]
        heading = math.atan2(-y0, -x0)
        
        # 尝试不同的投放时机
        best_coverage = 0.0
        best_t_drop = 1.0
        best_missile = "M1"
        
        for t_drop in np.linspace(0.5, 10.0, 20):
            t_explode = t_drop + 3.6
            
            # 计算对每枚导弹的遮蔽效果
            total_coverage = 0.0
            missile_coverage = {}
            
            for missile_id in missile_ids:
                coverage = coverage_time_for_plan(
                    uav_id=uav_id,
                    missile_id=missile_id,
                    v=v,
                    heading_rad=heading,
                    t_drop=t_drop,
                    t_explode=t_explode,
                    dt=0.1,
                    z_samples=3
                )
                missile_coverage[missile_id] = coverage
                total_coverage += coverage
            
            if total_coverage > best_coverage:
                best_coverage = total_coverage
                best_t_drop = t_drop
                best_missile = max(missile_coverage, key=missile_coverage.get)
        
        # 计算最终位置
        t_explode = best_t_drop + 3.6
        drop_pos = uav_xy_pos(uav_id, v, heading, best_t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, best_t_drop, t_explode)
        drop_z = FY_INIT[uav_id][2]
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向": math.degrees(heading) % 360,
            "无人机运动速度 (m/s)": v,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标 (m)": drop_z,
            "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
            "有效干扰时长 (s)": best_coverage,
            "干扰的导弹编号": best_missile
        }
        
        results_data.append(result_data)
        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={best_t_drop:.2f}s, "
              f"总遮蔽={best_coverage:.3f}s, 主要干扰={best_missile}")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    df.to_csv("result3_final.csv", index=False, encoding='utf-8-sig')
    print(f"结果已保存到 result3_final.csv")
    
    return results_data

# =====================
# 六、结果分析与可视化
# =====================

def analyze_results():
    """分析结果并生成报告"""
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)
    
    # 读取结果文件
    try:
        df1 = pd.read_csv("result1_final.csv")
        df2 = pd.read_csv("result2_final.csv")
        df3 = pd.read_csv("result3_final.csv")
        
        print("问题一结果：")
        print(f"有效遮蔽时长：{df1['有效干扰时长 (s)'].iloc[0]:.3f} 秒")
        
        print("\n问题二结果：")
        total_coverage_2 = df2['有效干扰时长 (s)'].sum()
        print(f"总遮蔽时长：{total_coverage_2:.3f} 秒")
        for _, row in df2.iterrows():
            print(f"{row['无人机编号']}: {row['有效干扰时长 (s)']:.3f} 秒")
        
        print("\n问题三结果：")
        total_coverage_3 = df3['有效干扰时长 (s)'].sum()
        print(f"总遮蔽时长：{total_coverage_3:.3f} 秒")
        for _, row in df3.iterrows():
            print(f"{row['无人机编号']}: {row['有效干扰时长 (s)']:.3f} 秒, 主要干扰{row['干扰的导弹编号']}")
        
        # 生成分析报告
        analysis_report = {
            "问题一": {
                "描述": "FY1单次投放对M1的有效遮蔽时长",
                "结果": f"{df1['有效干扰时长 (s)'].iloc[0]:.3f}秒",
                "分析": "单架无人机对单枚导弹的遮蔽效果有限，需要精确的时机控制"
            },
            "问题二": {
                "描述": "多无人机协同干扰优化",
                "结果": f"总遮蔽时长: {total_coverage_2:.3f}秒",
                "分析": "多无人机协同效果取决于投放时机和航向的精确控制"
            },
            "问题三": {
                "描述": "多导弹同时干扰的复杂优化",
                "结果": f"总遮蔽时长: {total_coverage_3:.3f}秒",
                "分析": "多目标干扰需要更复杂的优化策略，当前结果主要集中在对M1的干扰"
            }
        }
        
        with open('analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析报告已保存到 analysis_report.json")
        
    except FileNotFoundError as e:
        print(f"结果文件未找到：{e}")

# =====================
# 七、主程序
# =====================

def main():
    """主程序：依次解决三个问题"""
    print("2025年全国大学生数学建模竞赛A题最终完整解决方案")
    print("无人机烟幕干扰优化问题")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 问题一
        coverage1, result1 = solve_problem1()
        
        # 问题二
        results2 = solve_problem2()
        
        # 问题三
        results3 = solve_problem3()
        
        # 结果分析
        analyze_results()
        
        # 生成总结报告
        summary = {
            "问题一": {
                "描述": "FY1单次投放对M1的有效遮蔽时长",
                "结果": f"{coverage1:.3f}秒",
                "关键参数": result1
            },
            "问题二": {
                "描述": "多无人机协同干扰优化",
                "结果": f"总遮蔽时长: {sum([r['有效干扰时长 (s)'] for r in results2]):.3f}秒",
                "参与无人机": [r['无人机编号'] for r in results2]
            },
            "问题三": {
                "描述": "多导弹同时干扰的复杂优化",
                "结果": f"总遮蔽时长: {sum([r['有效干扰时长 (s)'] for r in results3]):.3f}秒",
                "参与无人机": [r['无人机编号'] for r in results3]
            }
        }
        
        with open('final_solution_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        end_time = time.time()
        print("\n" + "="*60)
        print("所有问题求解完成！")
        print(f"总计算时间：{end_time - start_time:.2f} 秒")
        print("="*60)
        print("生成的文件：")
        print("- result1_final.csv: 问题一结果")
        print("- result2_final.csv: 问题二结果") 
        print("- result3_final.csv: 问题三结果")
        print("- analysis_report.json: 结果分析报告")
        print("- final_solution_summary.json: 最终解决方案总结")
        
    except Exception as e:
        print(f"求解过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
