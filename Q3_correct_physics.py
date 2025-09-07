# -*- coding: utf-8 -*-
"""
Q3_correct_physics.py - 问题3的正确物理模型实现

基于对题目的重新理解：
1. 假目标在原点(0,0,0)
2. 真目标圆柱下底圆心在(0,200,0)，半径7m，高10m  
3. 导弹直指假目标，但我们要遮蔽导弹看向真目标的视线
4. 使用FY1投放3枚烟幕弹，间隔≥1s

关键修正：
- 导弹轨迹：直线飞向假目标(0,0,0)
- 遮蔽判定：导弹→真目标圆柱的视线被烟幕球遮挡
- 联合遮蔽：多个烟幕球的并集效果
"""

import math
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict, Any

# 物理常量
g = 9.8
CLOUD_R = 10.0          # 烟幕球半径
CLOUD_SINK = 3.0        # 下沉速度
CLOUD_EFFECT = 20.0     # 有效时间
MISSILE_SPEED = 300.0   # 导弹速度

# 目标位置（关键修正）
FAKE_TARGET = np.array([0.0, 0.0, 0.0])           # 假目标在原点
TRUE_TARGET_BASE_CENTER = np.array([0.0, 200.0, 0.0])  # 真目标下底面圆心
CYL_R = 7.0             # 圆柱半径
CYL_H = 10.0            # 圆柱高度

# 初始位置
M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

def missile_pos(t):
    """导弹位置：直线飞向假目标"""
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    return M1_INIT + MISSILE_SPEED * t * direction

def uav_pos(t, speed, heading):
    """无人机位置：水平直线飞行"""
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([
        FY1_INIT[0] + vx * t,
        FY1_INIT[1] + vy * t,
        FY1_INIT[2]  # 高度不变
    ])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置：考虑水平漂移和自由落体"""
    # 投放位置
    drop_pos = uav_pos(t_drop, speed, heading)
    
    # 水平漂移（无人机继续飞行）
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    # 爆炸位置
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * g * (fuse_delay ** 2)
    
    return np.array([expl_x, expl_y, expl_z])

def cloud_center(expl_pos, t_expl, t):
    """云团中心位置：考虑下沉"""
    if t < t_expl:
        return None
    if t > t_expl + CLOUD_EFFECT:
        return None
    
    sink_dist = CLOUD_SINK * (t - t_expl)
    center_z = expl_pos[2] - sink_dist
    
    if center_z < -CLOUD_R:  # 完全落地
        return None
    
    return np.array([expl_pos[0], expl_pos[1], center_z])

def generate_cylinder_points(n_phi=24, n_z=5):
    """生成圆柱体表面采样点"""
    points = []
    
    # 底面和顶面
    for z in [0, CYL_H]:
        for i in range(n_phi):
            angle = 2 * math.pi * i / n_phi
            x = TRUE_TARGET_BASE_CENTER[0] + CYL_R * math.cos(angle)
            y = TRUE_TARGET_BASE_CENTER[1] + CYL_R * math.sin(angle)
            points.append([x, y, TRUE_TARGET_BASE_CENTER[2] + z])
    
    # 侧面
    for k in range(1, n_z-1):
        z = CYL_H * k / (n_z-1)
        for i in range(n_phi):
            angle = 2 * math.pi * i / n_phi
            x = TRUE_TARGET_BASE_CENTER[0] + CYL_R * math.cos(angle)
            y = TRUE_TARGET_BASE_CENTER[1] + CYL_R * math.sin(angle)
            points.append([x, y, TRUE_TARGET_BASE_CENTER[2] + z])
    
    return np.array(points)

def line_sphere_intersect(p1, p2, center, radius):
    """线段与球体相交检测"""
    # 线段向量
    d = p2 - p1
    f = p1 - center
    
    # 求解二次方程 |p1 + t*d - center|^2 = radius^2
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False  # 无交点
    
    # 计算交点参数
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    
    # 检查交点是否在线段上
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    # 检查端点是否在球内
    if np.linalg.norm(p1 - center) <= radius:
        return True
    if np.linalg.norm(p2 - center) <= radius:
        return True
    
    return False

def is_target_blocked(missile_pos, target_points, cloud_centers):
    """检查目标是否被云团遮蔽
    
    Args:
        missile_pos: 导弹位置
        target_points: 目标采样点列表
        cloud_centers: 云团中心列表
    
    Returns:
        bool: 如果所有目标点的视线都被至少一个云团遮挡，返回True
    """
    if not cloud_centers:
        return False
    
    for target_point in target_points:
        # 检查这个目标点是否被任一云团遮挡
        blocked = False
        for cloud_center in cloud_centers:
            if line_sphere_intersect(missile_pos, target_point, cloud_center, CLOUD_R):
                blocked = True
                break
        
        if not blocked:
            return False  # 有目标点未被遮挡
    
    return True  # 所有目标点都被遮挡

def calculate_coverage_time(bombs_params, dt=0.02):
    """计算联合遮蔽时间
    
    Args:
        bombs_params: 三枚烟幕弹参数列表，每个包含 (t_drop, fuse_delay, speed, heading)
        dt: 时间步长
    
    Returns:
        float: 联合遮蔽时间（秒）
    """
    # 生成目标采样点
    target_points = generate_cylinder_points()
    
    # 计算爆炸时间和位置
    explosions = []
    for t_drop, fuse_delay, speed, heading in bombs_params:
        t_expl = t_drop + fuse_delay
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        explosions.append((t_expl, expl_pos))
    
    # 计算导弹命中时间
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    # 时间范围
    t_start = min(t_expl for t_expl, _ in explosions)
    t_end = min(hit_time, max(t_expl + CLOUD_EFFECT for t_expl, _ in explosions))
    
    if t_end <= t_start:
        return 0.0
    
    # 时间步进计算
    coverage_time = 0.0
    t = t_start
    
    while t <= t_end:
        # 当前导弹位置
        missile_position = missile_pos(t)
        
        # 收集当前有效的云团中心
        active_clouds = []
        for t_expl, expl_pos in explosions:
            cloud_pos = cloud_center(expl_pos, t_expl, t)
            if cloud_pos is not None:
                active_clouds.append(cloud_pos)
        
        # 检查是否被遮蔽
        if is_target_blocked(missile_position, target_points, active_clouds):
            coverage_time += dt
        
        t += dt
    
    return coverage_time

def optimize_three_bombs():
    """优化三枚烟幕弹的投放策略"""
    print("开始优化三枚烟幕弹投放策略...")
    
    best_params = None
    best_coverage = 0.0
    
    # 搜索参数
    speeds = [120, 130, 140]
    headings = [math.atan2(-FY1_INIT[1], -FY1_INIT[0]) + math.radians(d) 
                for d in range(-3, 4)]  # ±3度范围
    
    total_combinations = 0
    
    for speed in speeds:
        for heading in headings:
            print(f"测试速度={speed} m/s, 航向={math.degrees(heading):.1f}°")
            
            # 三枚弹的时间参数搜索
            for t1 in np.arange(0.5, 6.0, 0.5):
                for gap12 in np.arange(1.0, 4.0, 0.5):
                    for gap23 in np.arange(1.0, 4.0, 0.5):
                        t2 = t1 + gap12
                        t3 = t2 + gap23
                        
                        for f1 in np.arange(1.0, 6.0, 0.5):
                            for f2 in np.arange(1.0, 6.0, 0.5):
                                for f3 in np.arange(1.0, 6.0, 0.5):
                                    # 检查物理约束
                                    params = [(t1, f1, speed, heading),
                                             (t2, f2, speed, heading),
                                             (t3, f3, speed, heading)]
                                    
                                    # 验证爆炸点高度
                                    valid = True
                                    for t_drop, fuse, spd, hdg in params:
                                        expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
                                        if expl_pos[2] < 0:  # 爆炸点在地下
                                            valid = False
                                            break
                                    
                                    if not valid:
                                        continue
                                    
                                    # 计算遮蔽时间
                                    coverage = calculate_coverage_time(params)
                                    total_combinations += 1
                                    
                                    if coverage > best_coverage:
                                        best_coverage = coverage
                                        best_params = params
                                        print(f"  新最优: {coverage:.6f}s, 参数={params}")
                                    
                                    # 限制搜索时间
                                    if total_combinations >= 5000:
                                        break
                                if total_combinations >= 5000:
                                    break
                            if total_combinations >= 5000:
                                break
                        if total_combinations >= 5000:
                            break
                    if total_combinations >= 5000:
                        break
                if total_combinations >= 5000:
                    break
            if total_combinations >= 5000:
                break
        if total_combinations >= 5000:
            break
    
    print(f"\n优化完成！总共评估了{total_combinations}个组合")
    print(f"最优遮蔽时间: {best_coverage:.6f}s")
    
    return best_params, best_coverage

def save_to_excel(best_params, filename="result1_correct.xlsx"):
    """保存结果到Excel"""
    if best_params is None:
        print("无有效解，保存空结果")
        return
    
    rows = []
    for i, (t_drop, fuse_delay, speed, heading) in enumerate(best_params):
        # 计算位置
        drop_pos = uav_pos(t_drop, speed, heading)
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        
        # 计算单体遮蔽时间
        single_coverage = calculate_coverage_time([(t_drop, fuse_delay, speed, heading)])
        
        rows.append({
            "无人机运动方向": f"{heading:.6f} rad",
            "无人机运动速度 (m/s)": speed,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标 (m)": drop_pos[2],
            "烟幕干扰弹起爆点的x坐标 (m)": expl_pos[0],
            "烟幕干扰弹起爆点的y坐标 (m)": expl_pos[1],
            "烟幕干扰弹起爆点的z坐标 (m)": expl_pos[2],
            "有效干扰时长 (s)": single_coverage,
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"结果已保存到 {filename}")
    
    return df

def main():
    print("="*80)
    print("问题3正确物理模型求解")
    print("="*80)
    print(f"假目标位置: {FAKE_TARGET}")
    print(f"真目标中心: {TRUE_TARGET_CENTER}")
    print(f"导弹初始位置: {M1_INIT}")
    print(f"无人机初始位置: {FY1_INIT}")
    
    # 计算导弹命中时间
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    print(f"导弹命中时间: {hit_time:.3f}s")
    
    # 优化求解
    start_time = time.time()
    best_params, best_coverage = optimize_three_bombs()
    end_time = time.time()
    
    print(f"\n求解耗时: {end_time - start_time:.2f}s")
    print(f"最优联合遮蔽时间: {best_coverage:.6f}s")
    
    if best_params:
        print("\n最优参数:")
        for i, (t_drop, fuse, speed, heading) in enumerate(best_params):
            print(f"  烟幕弹{i+1}: t_drop={t_drop:.3f}s, fuse={fuse:.3f}s, "
                  f"speed={speed:.1f}m/s, heading={math.degrees(heading):.2f}°")
    
    # 保存结果
    df = save_to_excel(best_params)
    
    # 验证计算
    if best_params:
        verify_coverage = calculate_coverage_time(best_params, dt=0.01)  # 更精细验证
        print(f"\n验证结果（dt=0.01）: {verify_coverage:.6f}s")
    
    return best_params, best_coverage

if __name__ == "__main__":
    main()
