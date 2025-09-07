# -*- coding: utf-8 -*-
"""
Q3_analysis_correct.py - 重新分析问题3的几何关系

关键发现：
1. 导弹飞向假目标(0,0,0)
2. 真目标在(0,200,0)，相对导弹轨迹是侧方目标
3. 需要阻挡导弹的侧向"视线"，而不是前向飞行路径
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# 基本参数
MISSILE_SPEED = 300.0
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BASE = np.array([0.0, 200.0, 0.0])
M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

def missile_pos(t):
    """导弹位置"""
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    return M1_INIT + MISSILE_SPEED * t * direction

def analyze_sight_lines():
    """分析导弹到真目标的视线"""
    print("="*80)
    print("导弹视线分析")
    print("="*80)
    
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    print("导弹轨迹上各点到真目标的视线分析:")
    print("时间(s) | 导弹位置 | 到真目标距离(m) | 视线角度(°)")
    print("-" * 70)
    
    sight_lines = []
    
    for t in np.arange(0, hit_time, 5):
        missile_position = missile_pos(t)
        
        # 计算视线向量
        sight_vector = TRUE_TARGET_BASE - missile_position
        sight_distance = np.linalg.norm(sight_vector)
        
        # 计算视线与导弹飞行方向的夹角
        missile_direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
        sight_direction = sight_vector / sight_distance
        
        # 夹角
        cos_angle = np.dot(missile_direction, sight_direction)
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        
        print(f"{t:6.1f} | ({missile_position[0]:8.0f},{missile_position[1]:6.0f},{missile_position[2]:6.0f}) | "
              f"{sight_distance:10.0f} | {angle_deg:8.2f}")
        
        sight_lines.append((t, missile_position.copy(), sight_vector.copy(), sight_distance))
    
    return sight_lines

def find_optimal_interception_points(sight_lines):
    """找到最佳拦截点"""
    print("\n" + "="*80)
    print("最佳拦截点分析")
    print("="*80)
    
    best_points = []
    
    for t, missile_pos, sight_vec, sight_dist in sight_lines:
        if t < 5 or t > 50:  # 排除太早或太晚的时机
            continue
        
        # 在视线中点附近放置云团
        midpoint = missile_pos + sight_vec * 0.5
        
        # 检查无人机是否能到达
        uav_to_midpoint = np.linalg.norm(midpoint[:2] - FY1_INIT[:2])
        required_speed = uav_to_midpoint / t if t > 0 else float('inf')
        
        if 70 <= required_speed <= 140:
            best_points.append({
                'time': t,
                'missile_pos': missile_pos,
                'intercept_point': midpoint,
                'required_speed': required_speed,
                'sight_distance': sight_dist
            })
            
            print(f"t={t:5.1f}s: 拦截点=({midpoint[0]:7.0f},{midpoint[1]:6.0f},{midpoint[2]:6.0f}), "
                  f"需要速度={required_speed:6.1f}m/s ✅")
    
    return best_points

def calculate_simple_coverage(intercept_point, t_expl):
    """简化的遮蔽时间计算"""
    # 生成少量关键目标点
    key_points = [
        TRUE_TARGET_BASE + np.array([CYL_R, 0, 0]),      # 东侧
        TRUE_TARGET_BASE + np.array([-CYL_R, 0, 0]),     # 西侧
        TRUE_TARGET_BASE + np.array([0, CYL_R, 0]),      # 北侧
        TRUE_TARGET_BASE + np.array([0, -CYL_R, 0]),     # 南侧
        TRUE_TARGET_BASE + np.array([0, 0, CYL_H]),      # 顶部中心
    ]
    
    coverage_time = 0.0
    dt = 0.1
    
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    for t in np.arange(t_expl, min(t_expl + 20, hit_time), dt):
        missile_position = missile_pos(t)
        
        # 云团位置（考虑下沉）
        cloud_z = intercept_point[2] - 3.0 * (t - t_expl)
        if cloud_z < -10:
            break
        
        cloud_pos = np.array([intercept_point[0], intercept_point[1], cloud_z])
        
        # 检查关键点是否被遮蔽
        blocked_count = 0
        for target_point in key_points:
            # 简化判断：云团是否在视线附近
            sight_vec = target_point - missile_position
            sight_dist = np.linalg.norm(sight_vec)
            
            if sight_dist < 1e-6:
                continue
            
            # 计算云团到视线的距离
            sight_unit = sight_vec / sight_dist
            missile_to_cloud = cloud_pos - missile_position
            proj_length = np.dot(missile_to_cloud, sight_unit)
            
            if 0 <= proj_length <= sight_dist:
                proj_point = missile_position + proj_length * sight_unit
                dist_to_line = np.linalg.norm(cloud_pos - proj_point)
                
                if dist_to_line <= 10.0:  # 云团半径
                    blocked_count += 1
        
        if blocked_count >= len(key_points) * 0.8:  # 80%以上被遮蔽
            coverage_time += dt
    
    return coverage_time

def test_realistic_scenario():
    """测试现实可行的场景"""
    print("\n" + "="*80)
    print("现实可行场景测试")
    print("="*80)
    
    sight_lines = analyze_sight_lines()
    best_points = find_optimal_interception_points(sight_lines)
    
    if not best_points:
        print("❌ 未找到可行的拦截点")
        return None, 0.0
    
    # 选择最佳拦截点
    best_point = min(best_points, key=lambda x: x['required_speed'])
    print(f"\n选择最佳拦截点:")
    print(f"时间: {best_point['time']:.1f}s")
    print(f"位置: {best_point['intercept_point']}")
    print(f"需要速度: {best_point['required_speed']:.1f}m/s")
    
    # 设计三枚弹的投放策略
    base_time = best_point['time'] - 8  # 提前8秒开始投放
    speed = best_point['required_speed']
    heading = math.atan2(best_point['intercept_point'][1] - FY1_INIT[1],
                        best_point['intercept_point'][0] - FY1_INIT[0])
    
    params = [
        (base_time, 6.0, speed, heading),      # 第一枚：早投放，长引信
        (base_time + 2.0, 4.0, speed, heading),  # 第二枚：中等
        (base_time + 4.0, 2.0, speed, heading),  # 第三枚：晚投放，短引信
    ]
    
    # 验证参数
    print(f"\n投放策略:")
    for i, (t_drop, fuse, spd, hdg) in enumerate(params):
        expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
        print(f"弹{i+1}: drop={t_drop:.1f}s, fuse={fuse:.1f}s, expl=({expl_pos[0]:.0f},{expl_pos[1]:.0f},{expl_pos[2]:.0f})")
    
    # 计算遮蔽时间
    total_coverage = 0.0
    for i, (t_drop, fuse, spd, hdg) in enumerate(params):
        t_expl = t_drop + fuse
        expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
        coverage = calculate_simple_coverage(expl_pos, t_expl)
        total_coverage += coverage  # 简单加和（实际应该用并集）
        print(f"弹{i+1}遮蔽时间: {coverage:.3f}s")
    
    print(f"总遮蔽时间（近似）: {total_coverage:.3f}s")
    
    return params, total_coverage

if __name__ == "__main__":
    test_realistic_scenario()
