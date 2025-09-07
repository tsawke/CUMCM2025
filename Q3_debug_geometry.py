# -*- coding: utf-8 -*-
"""
Q3_debug_geometry.py - 调试几何关系和物理模型

专门用于验证：
1. 目标位置关系是否正确
2. 导弹轨迹是否合理
3. 遮蔽判定逻辑是否正确
"""

import math
import numpy as np

# 基本参数
MISSILE_SPEED = 300.0
CLOUD_R = 10.0

# 位置（根据题目重新确认）
FAKE_TARGET = np.array([0.0, 0.0, 0.0])           # 假目标在原点
TRUE_TARGET_BASE_CENTER = np.array([0.0, 200.0, 0.0])  # 真目标下底圆心
M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

def missile_pos(t):
    """导弹位置：直线飞向假目标"""
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    return M1_INIT + MISSILE_SPEED * t * direction

def debug_geometry():
    """调试几何关系"""
    print("="*80)
    print("几何关系调试")
    print("="*80)
    
    print(f"假目标位置: {FAKE_TARGET}")
    print(f"真目标下底圆心: {TRUE_TARGET_BASE_CENTER}")
    print(f"导弹初始位置: {M1_INIT}")
    print(f"无人机初始位置: {FY1_INIT}")
    
    # 计算导弹轨迹
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    print(f"导弹飞行方向向量: {direction}")
    
    hit_time = np.linalg.norm(FAKE_TARGET - M1_INIT) / MISSILE_SPEED
    print(f"导弹命中假目标时间: {hit_time:.3f}s")
    
    # 检查导弹在几个关键时刻的位置
    print("\n导弹轨迹关键点:")
    for t in [0, 10, 20, 30, 40, 50, 60, hit_time]:
        if t <= hit_time:
            pos = missile_pos(t)
            dist_to_true = np.linalg.norm(pos - TRUE_TARGET_BASE_CENTER)
            print(f"  t={t:5.1f}s: 位置=({pos[0]:8.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}), "
                  f"到真目标距离={dist_to_true:8.1f}m")
    
    # 分析最佳拦截时机
    print("\n最佳拦截时机分析:")
    min_dist = float('inf')
    best_t = 0
    
    for t in np.arange(0, hit_time, 1.0):
        pos = missile_pos(t)
        dist = np.linalg.norm(pos - TRUE_TARGET_BASE_CENTER)
        if dist < min_dist:
            min_dist = dist
            best_t = t
    
    print(f"导弹离真目标最近时刻: t={best_t:.1f}s, 距离={min_dist:.1f}m")
    
    # 分析无人机到拦截点的可达性
    best_missile_pos = missile_pos(best_t)
    print(f"最佳拦截点: {best_missile_pos}")
    
    # 计算无人机需要的速度和时间
    uav_to_intercept = np.linalg.norm(best_missile_pos[:2] - FY1_INIT[:2])
    required_speed = uav_to_intercept / best_t if best_t > 0 else float('inf')
    
    print(f"无人机到拦截点距离: {uav_to_intercept:.1f}m")
    print(f"需要的水平速度: {required_speed:.1f}m/s")
    
    if 70 <= required_speed <= 140:
        print("✅ 无人机可以到达拦截点")
    else:
        print("❌ 无人机无法到达拦截点")
    
    return best_t, min_dist, required_speed

def test_simple_interception():
    """测试简单拦截场景"""
    print("\n" + "="*80)
    print("简单拦截场景测试")
    print("="*80)
    
    # 使用合理的参数
    speed = 120.0  # m/s
    heading = math.atan2(TRUE_TARGET_BASE_CENTER[1] - FY1_INIT[1], 
                        TRUE_TARGET_BASE_CENTER[0] - FY1_INIT[0])  # 飞向真目标方向
    
    print(f"无人机速度: {speed} m/s")
    print(f"无人机航向: {math.degrees(heading):.2f}°")
    
    # 测试不同的投放时机
    for t_drop in [5, 10, 15, 20, 25, 30]:
        for fuse_delay in [2, 4, 6, 8]:
            t_expl = t_drop + fuse_delay
            
            # 计算爆炸点
            drop_pos = uav_pos(t_drop, speed, heading)
            expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
            
            if expl_pos[2] < 0:  # 爆炸点在地下
                continue
            
            # 计算此时导弹位置
            missile_position = missile_pos(t_expl)
            
            # 计算导弹到真目标和爆炸点的距离
            dist_missile_target = np.linalg.norm(missile_position - TRUE_TARGET_BASE_CENTER)
            dist_missile_cloud = np.linalg.norm(missile_position - expl_pos)
            dist_cloud_target = np.linalg.norm(expl_pos - TRUE_TARGET_BASE_CENTER)
            
            print(f"t_drop={t_drop:2d}s, fuse={fuse_delay}s: "
                  f"爆炸点=({expl_pos[0]:6.0f},{expl_pos[1]:6.0f},{expl_pos[2]:6.0f}), "
                  f"导弹距真目标={dist_missile_target:6.0f}m, "
                  f"导弹距云团={dist_missile_cloud:6.0f}m, "
                  f"云团距真目标={dist_cloud_target:6.0f}m")
            
            # 简单判断是否可能有效
            if dist_cloud_target < 100 and expl_pos[2] > 50:  # 云团靠近目标且高度合理
                print(f"  ⭐ 潜在有效拦截点")

def uav_pos(t, speed, heading):
    """无人机位置"""
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([
        FY1_INIT[0] + vx * t,
        FY1_INIT[1] + vy * t,
        FY1_INIT[2]
    ])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置"""
    drop_pos = uav_pos(t_drop, speed, heading)
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * 9.8 * (fuse_delay ** 2)
    
    return np.array([expl_x, expl_y, expl_z])

def main():
    # 基础几何调试
    best_t, min_dist, required_speed = debug_geometry()
    
    # 简单拦截测试
    test_simple_interception()
    
    print(f"\n总结:")
    print(f"- 最佳拦截时刻: {best_t:.1f}s")
    print(f"- 最小距离: {min_dist:.1f}m") 
    print(f"- 需要速度: {required_speed:.1f}m/s")

if __name__ == "__main__":
    main()
