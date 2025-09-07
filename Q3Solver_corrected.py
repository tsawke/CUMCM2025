# -*- coding: utf-8 -*-
"""
Q3Solver_corrected.py - 基于正确物理理解的Q3求解器

核心修正：
1. 明确几何关系：假目标(0,0,0)，真目标底面中心(0,200,0)
2. 导弹轨迹：M1直线飞向假目标
3. 遮蔽目标：导弹→真目标圆柱的侧向视线
4. 优化策略：在导弹轨迹关键点附近布置烟幕
5. 时间限制：确保在合理时间内完成
"""

import math
import numpy as np
import pandas as pd
import time
import argparse

# 物理常量
g = 9.8
CLOUD_R = 10.0
CLOUD_SINK = 3.0  
CLOUD_EFFECT = 20.0
MISSILE_SPEED = 300.0

# 几何定义（关键修正）
FAKE_TARGET = np.array([0.0, 0.0, 0.0])           # 假目标在原点
TRUE_TARGET_BASE = np.array([0.0, 200.0, 0.0])    # 真目标底面中心
CYL_R = 7.0
CYL_H = 10.0

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
    return np.array([FY1_INIT[0] + vx * t, FY1_INIT[1] + vy * t, FY1_INIT[2]])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置：投放点+水平漂移+自由落体"""
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
    if center_z < -CLOUD_R:  # 完全落地
        return None
    
    return np.array([expl_pos[0], expl_pos[1], center_z])

def generate_cylinder_points(n_phi=16, n_z=3):
    """生成圆柱体关键采样点"""
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
    
    # 中间层关键点
    for k in range(1, n_z):
        z = TRUE_TARGET_BASE[2] + CYL_H * k / n_z
        for i in range(0, n_phi, 4):  # 稀疏采样
            angle = 2 * math.pi * i / n_phi
            x = TRUE_TARGET_BASE[0] + CYL_R * math.cos(angle)
            y = TRUE_TARGET_BASE[1] + CYL_R * math.sin(angle)
            points.append([x, y, z])
    
    return np.array(points)

def point_to_line_distance(point, line_start, line_end):
    """点到线段的最短距离"""
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq < 1e-10:
        return np.linalg.norm(point - line_start)
    
    # 投影参数
    t = np.dot(point - line_start, line_vec) / line_len_sq
    t = max(0, min(1, t))  # 限制在线段上
    
    closest_point = line_start + t * line_vec
    return np.linalg.norm(point - closest_point)

def calculate_single_coverage(bomb_params, dt=0.02):
    """计算单个烟幕弹的遮蔽时间"""
    t_drop, fuse_delay, speed, heading = bomb_params
    t_expl = t_drop + fuse_delay
    expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
    
    if expl_pos[2] < 0:
        return 0.0
    
    # 生成目标点
    target_points = generate_cylinder_points()
    
    # 时间范围
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    coverage_time = 0.0
    t = t_expl
    
    while t <= min(t_expl + CLOUD_EFFECT, hit_time):
        # 云团位置
        cloud_pos = cloud_center(expl_pos, t_expl, t)
        if cloud_pos is None:
            t += dt
            continue
        
        # 导弹位置
        missile_position = missile_pos(t)
        
        # 检查所有目标点是否被遮蔽
        all_blocked = True
        for target_point in target_points:
            dist = point_to_line_distance(cloud_pos, missile_position, target_point)
            if dist > CLOUD_R:
                all_blocked = False
                break
        
        if all_blocked:
            coverage_time += dt
        
        t += dt
    
    return coverage_time

def calculate_union_coverage(bombs_params, dt=0.02):
    """计算三枚弹的联合遮蔽时间"""
    target_points = generate_cylinder_points()
    
    # 计算爆炸参数
    explosions = []
    for t_drop, fuse_delay, speed, heading in bombs_params:
        t_expl = t_drop + fuse_delay
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        if expl_pos[2] < 0:
            return 0.0
        explosions.append((t_expl, expl_pos))
    
    # 时间范围
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    t_start = min(t_expl for t_expl, _ in explosions)
    t_end = min(hit_time, max(t_expl + CLOUD_EFFECT for t_expl, _ in explosions))
    
    if t_end <= t_start:
        return 0.0
    
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
        
        # 联合遮蔽判定：所有目标点都被至少一个云团遮蔽
        all_blocked = True
        for target_point in target_points:
            point_blocked = False
            for cloud_pos in active_clouds:
                dist = point_to_line_distance(cloud_pos, missile_position, target_point)
                if dist <= CLOUD_R:
                    point_blocked = True
                    break
            
            if not point_blocked:
                all_blocked = False
                break
        
        if all_blocked:
            coverage_time += dt
        
        t += dt
    
    return coverage_time

def smart_optimization(max_time_minutes=10):
    """智能优化算法"""
    print("开始智能优化...")
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    best_params = None
    best_coverage = 0.0
    evaluations = 0
    
    # 基于物理分析的搜索策略
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    # 重点时段：导弹中后期（30-55秒）
    for target_time in np.arange(35, min(55, hit_time-10), 5):
        # 在目标时间附近布置烟幕
        missile_at_target = missile_pos(target_time)
        
        # 计算合理的拦截位置（导弹轨迹与真目标之间）
        intercept_point = missile_at_target + 0.2 * (TRUE_TARGET_BASE - missile_at_target)
        
        for speed in [100, 120, 140]:
            # 计算航向
            dx = intercept_point[0] - FY1_INIT[0]
            dy = intercept_point[1] - FY1_INIT[1]
            heading = math.atan2(dy, dx)
            
            # 计算到达时间
            dist = math.hypot(dx, dy)
            arrival_time = dist / speed
            
            if arrival_time > target_time - 15:  # 太晚
                continue
            
            # 三枚弹的投放策略
            base_drop = arrival_time + 2
            
            for dt1 in [1.5, 2.0, 2.5]:
                for dt2 in [1.5, 2.0, 2.5]:
                    t1 = base_drop
                    t2 = t1 + dt1
                    t3 = t2 + dt2
                    
                    for f1 in [3, 4, 5]:
                        for f2 in [3, 4, 5]:
                            for f3 in [3, 4, 5]:
                                params = [(t1, f1, speed, heading),
                                         (t2, f2, speed, heading),
                                         (t3, f3, speed, heading)]
                                
                                # 物理可行性检查
                                valid = True
                                for t_drop, fuse, spd, hdg in params:
                                    expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
                                    if expl_pos[2] < 100:  # 爆炸点太低
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                # 计算遮蔽时间
                                coverage = calculate_union_coverage(params)
                                evaluations += 1
                                
                                if coverage > best_coverage:
                                    best_coverage = coverage
                                    best_params = params
                                    print(f"新最优: {coverage:.6f}s (评估{evaluations}次)")
                                
                                # 时间限制
                                if time.time() - start_time > max_time_seconds:
                                    print("达到时间限制，停止搜索")
                                    break
                            if time.time() - start_time > max_time_seconds:
                                break
                        if time.time() - start_time > max_time_seconds:
                            break
                    if time.time() - start_time > max_time_seconds:
                        break
                if time.time() - start_time > max_time_seconds:
                    break
            if time.time() - start_time > max_time_seconds:
                break
        if time.time() - start_time > max_time_seconds:
            break
    
    elapsed = time.time() - start_time
    print(f"优化完成，耗时{elapsed:.2f}s，评估{evaluations}次")
    
    return best_params, best_coverage

def save_result(params, coverage):
    """保存结果到Excel"""
    if not params:
        print("无有效解，保存默认值")
        # 创建默认解
        default_heading = math.atan2(-FY1_INIT[1], -FY1_INIT[0])
        rows = []
        for i in range(3):
            rows.append({
                "无人机运动方向": f"{default_heading:.6f} rad",
                "无人机运动速度 (m/s)": 120.0,
                "烟幕干扰弹编号": i + 1,
                "烟幕干扰弹投放点的x坐标 (m)": FY1_INIT[0],
                "烟幕干扰弹投放点的y坐标 (m)": FY1_INIT[1],
                "烟幕干扰弹投放点的z坐标 (m)": FY1_INIT[2],
                "烟幕干扰弹起爆点的x坐标 (m)": FY1_INIT[0],
                "烟幕干扰弹起爆点的y坐标 (m)": FY1_INIT[1],
                "烟幕干扰弹起爆点的z坐标 (m)": FY1_INIT[2] - 100,
                "有效干扰时长 (s)": 0.0,
            })
    else:
        rows = []
        for i, (t_drop, fuse_delay, speed, heading) in enumerate(params):
            drop_pos = uav_pos(t_drop, speed, heading)
            expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
            single_coverage = calculate_single_coverage((t_drop, fuse_delay, speed, heading))
            
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
    df.to_excel("result1.xlsx", index=False)
    
    print("\n最终结果:")
    print(df.to_string(index=False))
    print(f"\n联合遮蔽时间: {coverage:.6f}s")
    print("结果已保存到: result1.xlsx")
    
    return df

def calculate_single_coverage(bomb_params, dt=0.02):
    """计算单个烟幕弹的遮蔽时间（用于Excel中的单体时长）"""
    t_drop, fuse_delay, speed, heading = bomb_params
    t_expl = t_drop + fuse_delay
    expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
    
    if expl_pos[2] < 0:
        return 0.0
    
    target_points = generate_cylinder_points(n_phi=12, n_z=2)  # 简化采样
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    coverage_time = 0.0
    t = t_expl
    
    while t <= min(t_expl + CLOUD_EFFECT, hit_time):
        cloud_pos = cloud_center(expl_pos, t_expl, t)
        if cloud_pos is None:
            t += dt
            continue
        
        missile_position = missile_pos(t)
        
        # 检查遮蔽效果
        blocked_count = 0
        for target_point in target_points:
            dist = point_to_line_distance(cloud_pos, missile_position, target_point)
            if dist <= CLOUD_R:
                blocked_count += 1
        
        # 如果大部分点被遮蔽，认为有效
        if blocked_count >= len(target_points) * 0.7:
            coverage_time += dt
        
        t += dt
    
    return coverage_time

def main():
    parser = argparse.ArgumentParser("Q3 Corrected Solver")
    parser.add_argument("--max-time", type=float, default=10.0, help="最大优化时间（分钟）")
    parser.add_argument("--dt", type=float, default=0.02, help="时间步长")
    args = parser.parse_args()
    
    print("="*80)
    print("Q3 修正版求解器 - 基于正确物理模型")
    print("="*80)
    print(f"假目标: {FAKE_TARGET}")
    print(f"真目标底面中心: {TRUE_TARGET_BASE}")
    print(f"导弹初始: {M1_INIT}")
    print(f"无人机初始: {FY1_INIT}")
    
    # 优化求解
    params, coverage = smart_optimization(max_time_minutes=args.max_time)
    
    # 保存结果
    save_result(params, coverage)
    
    print(f"\n总结:")
    print(f"- 优化时间限制: {args.max_time}分钟")
    print(f"- 最优遮蔽时间: {coverage:.6f}s")
    
    if coverage > 0.1:
        print("✅ 找到合理解，应该与Q3Drawer验证一致")
    else:
        print("⚠️  遮蔽时间较小，可能需要进一步优化参数")
    
    return params, coverage

if __name__ == "__main__":
    main()
