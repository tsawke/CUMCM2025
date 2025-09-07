# -*- coding: utf-8 -*-
"""
Q3_final_correct.py - 问题3的最终正确解决方案

核心理解修正：
1. 导弹飞向假目标(0,0,0)，但我们要保护真目标(0,200,0)
2. 需要在导弹轨迹的早期就进行拦截
3. 烟幕球要放置在导弹→真目标的视线上
4. 使用更合理的搜索策略和参数范围
"""

import math
import numpy as np
import pandas as pd
import time
from typing import List, Tuple

# 物理常量
g = 9.8
CLOUD_R = 10.0
CLOUD_SINK = 3.0
CLOUD_EFFECT = 20.0
MISSILE_SPEED = 300.0

# 位置
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BASE = np.array([0.0, 200.0, 0.0])
CYL_R = 7.0
CYL_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

def missile_pos(t):
    """导弹位置：直线飞向假目标"""
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    return M1_INIT + MISSILE_SPEED * t * direction

def uav_pos(t, speed, heading):
    """无人机位置"""
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([FY1_INIT[0] + vx * t, FY1_INIT[1] + vy * t, FY1_INIT[2]])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置"""
    drop_pos = uav_pos(t_drop, speed, heading)
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * g * (fuse_delay ** 2)
    
    return np.array([expl_x, expl_y, expl_z])

def cloud_center(expl_pos, t_expl, t):
    """云团中心：考虑下沉"""
    if t < t_expl or t > t_expl + CLOUD_EFFECT:
        return None
    
    center_z = expl_pos[2] - CLOUD_SINK * (t - t_expl)
    if center_z < -CLOUD_R:
        return None
    
    return np.array([expl_pos[0], expl_pos[1], center_z])

def generate_target_points(n_phi=12, n_z=3):
    """生成目标圆柱的关键点"""
    points = []
    
    # 底面圆周
    for i in range(n_phi):
        angle = 2 * math.pi * i / n_phi
        x = TRUE_TARGET_BASE[0] + CYL_R * math.cos(angle)
        y = TRUE_TARGET_BASE[1] + CYL_R * math.sin(angle)
        points.append([x, y, TRUE_TARGET_BASE[2]])
    
    # 顶面圆周
    for i in range(n_phi):
        angle = 2 * math.pi * i / n_phi
        x = TRUE_TARGET_BASE[0] + CYL_R * math.cos(angle)
        y = TRUE_TARGET_BASE[1] + CYL_R * math.sin(angle)
        points.append([x, y, TRUE_TARGET_BASE[2] + CYL_H])
    
    # 中间层
    for k in range(1, n_z):
        z = TRUE_TARGET_BASE[2] + CYL_H * k / n_z
        for i in range(0, n_phi, 2):  # 稀疏采样
            angle = 2 * math.pi * i / n_phi
            x = TRUE_TARGET_BASE[0] + CYL_R * math.cos(angle)
            y = TRUE_TARGET_BASE[1] + CYL_R * math.sin(angle)
            points.append([x, y, z])
    
    return np.array(points)

def point_to_line_distance(point, line_start, line_end):
    """点到线段的最短距离"""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        return np.linalg.norm(point - line_start)
    
    line_unit = line_vec / line_len
    point_vec = point - line_start
    
    # 投影到线段上的参数
    t = np.dot(point_vec, line_unit)
    t = max(0, min(line_len, t))  # 限制在线段范围内
    
    # 最近点
    closest = line_start + t * line_unit
    return np.linalg.norm(point - closest)

def is_line_blocked_by_sphere(line_start, line_end, sphere_center, sphere_radius):
    """检查线段是否被球体遮挡"""
    distance = point_to_line_distance(sphere_center, line_start, line_end)
    return distance <= sphere_radius

def calculate_coverage_union(bombs_params, dt=0.05):
    """计算联合遮蔽时间"""
    target_points = generate_target_points()
    
    # 计算爆炸参数
    explosions = []
    for t_drop, fuse_delay, speed, heading in bombs_params:
        t_expl = t_drop + fuse_delay
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        if expl_pos[2] < 0:  # 无效爆炸点
            return 0.0
        explosions.append((t_expl, expl_pos))
    
    # 时间范围
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    t_start = min(t_expl for t_expl, _ in explosions)
    t_end = min(hit_time, max(t_expl + CLOUD_EFFECT for t_expl, _ in explosions))
    
    if t_end <= t_start:
        return 0.0
    
    # 时间步进
    coverage_time = 0.0
    t = t_start
    
    while t <= t_end:
        missile_position = missile_pos(t)
        
        # 收集有效云团
        active_clouds = []
        for t_expl, expl_pos in explosions:
            cloud_pos = cloud_center(expl_pos, t_expl, t)
            if cloud_pos is not None:
                active_clouds.append(cloud_pos)
        
        if not active_clouds:
            t += dt
            continue
        
        # 检查每个目标点的视线是否被遮挡
        all_blocked = True
        for target_point in target_points:
            point_blocked = False
            for cloud_pos in active_clouds:
                if is_line_blocked_by_sphere(missile_position, target_point, cloud_pos, CLOUD_R):
                    point_blocked = True
                    break
            
            if not point_blocked:
                all_blocked = False
                break
        
        if all_blocked:
            coverage_time += dt
        
        t += dt
    
    return coverage_time

def smart_search():
    """智能搜索策略"""
    print("开始智能搜索...")
    
    best_params = None
    best_coverage = 0.0
    
    # 基础航向：指向真目标附近
    base_heading = math.atan2(TRUE_TARGET_BASE[1] - FY1_INIT[1], 
                             TRUE_TARGET_BASE[0] - FY1_INIT[0])
    
    # 搜索范围
    speeds = [100, 120, 140]
    heading_offsets = np.arange(-10, 11, 2)  # ±10度
    
    eval_count = 0
    
    for speed in speeds:
        for offset_deg in heading_offsets:
            heading = base_heading + math.radians(offset_deg)
            
            print(f"测试: 速度={speed}m/s, 航向偏移={offset_deg}°")
            
            # 早期拦截策略：在导弹飞行前半程进行拦截
            for t1 in np.arange(2.0, 20.0, 2.0):
                for gap12 in [1.5, 2.0, 3.0]:
                    for gap23 in [1.5, 2.0, 3.0]:
                        t2 = t1 + gap12
                        t3 = t2 + gap23
                        
                        for f1 in [3.0, 4.0, 5.0]:
                            for f2 in [3.0, 4.0, 5.0]:
                                for f3 in [3.0, 4.0, 5.0]:
                                    params = [(t1, f1, speed, heading),
                                             (t2, f2, speed, heading),
                                             (t3, f3, speed, heading)]
                                    
                                    # 快速有效性检查
                                    valid = True
                                    for t_drop, fuse, spd, hdg in params:
                                        expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
                                        if expl_pos[2] < 100:  # 爆炸点太低
                                            valid = False
                                            break
                                    
                                    if not valid:
                                        continue
                                    
                                    coverage = calculate_coverage_union(params)
                                    eval_count += 1
                                    
                                    if coverage > best_coverage:
                                        best_coverage = coverage
                                        best_params = params
                                        print(f"  ⭐ 新最优: {coverage:.6f}s")
                                        for j, p in enumerate(params):
                                            print(f"     弹{j+1}: drop={p[0]:.1f}s, fuse={p[1]:.1f}s")
                                    
                                    if eval_count >= 3000:  # 限制搜索
                                        break
                                if eval_count >= 3000:
                                    break
                            if eval_count >= 3000:
                                break
                        if eval_count >= 3000:
                            break
                    if eval_count >= 3000:
                        break
                if eval_count >= 3000:
                    break
            if eval_count >= 3000:
                break
        if eval_count >= 3000:
            break
    
    print(f"\n搜索完成，评估了{eval_count}个组合")
    return best_params, best_coverage

def save_result(params, coverage, filename="result1_final.xlsx"):
    """保存最终结果"""
    if not params:
        print("无有效解")
        return
    
    rows = []
    for i, (t_drop, fuse_delay, speed, heading) in enumerate(params):
        drop_pos = uav_pos(t_drop, speed, heading)
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        
        # 计算单体遮蔽时间
        single_coverage = calculate_coverage_union([(t_drop, fuse_delay, speed, heading)])
        
        rows.append({
            "无人机运动方向": f"{heading:.6f} rad",
            "无人机运动速度 (m/s)": speed,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_pos[0], 6),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_pos[1], 6),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_pos[2], 6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(expl_pos[0], 6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(expl_pos[1], 6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(expl_pos[2], 6),
            "有效干扰时长 (s)": round(single_coverage, 6),
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    
    print(f"\n最终结果:")
    print(df.to_string(index=False))
    print(f"\n联合遮蔽时间: {coverage:.6f}s")
    print(f"结果已保存到: {filename}")

def main():
    print("="*80)
    print("问题3最终正确解决方案")
    print("="*80)
    
    # 基本信息
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    print(f"导弹命中时间: {hit_time:.3f}s")
    print(f"导弹轨迹: {M1_INIT} → {FAKE_TARGET}")
    print(f"保护目标: 圆柱体，底面中心{TRUE_TARGET_BASE}，半径{CYL_R}m，高{CYL_H}m")
    
    # 智能搜索
    start_time = time.time()
    best_params, best_coverage = smart_search()
    end_time = time.time()
    
    print(f"\n优化耗时: {end_time - start_time:.2f}s")
    
    if best_coverage > 0:
        save_result(best_params, best_coverage)
    else:
        print("未找到有效解，可能需要调整搜索参数或物理模型")
        
        # 提供诊断信息
        print("\n诊断建议:")
        print("1. 检查目标点生成是否合理")
        print("2. 检查遮蔽判定算法")
        print("3. 扩大搜索范围或调整约束")

if __name__ == "__main__":
    main()
