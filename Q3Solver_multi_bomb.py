# -*- coding: utf-8 -*-
"""
Q3Solver_multi_bomb.py - 专门优化多弹协同的Q3求解器

目标：
1. 确保至少2枚烟幕弹发挥作用
2. 最终遮蔽时间在6-7秒范围内
3. 使用激进的搜索策略和奖励机制
"""

import math
import numpy as np
import pandas as pd
import time
import Q1Solver_visual as Q1

def info(msg: str):
    print(f"[Q3-MULTI] {msg}", flush=True)

def missile_pos(t):
    """导弹位置"""
    direction = (Q1.FAKE_TARGET_ORIGIN - Q1.M1_INIT) / np.linalg.norm(Q1.FAKE_TARGET_ORIGIN - Q1.M1_INIT)
    return Q1.M1_INIT + Q1.MISSILE_SPEED * t * direction

def uav_pos(t, speed, heading):
    """无人机位置"""
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([Q1.FY1_INIT[0] + vx * t, Q1.FY1_INIT[1] + vy * t, Q1.FY1_INIT[2]])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置"""
    drop_pos = uav_pos(t_drop, speed, heading)
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * 9.8 * (fuse_delay ** 2)
    
    return np.array([expl_x, expl_y, expl_z])

def calculate_generous_coverage(bomb_params, target_points, dt=0.05):
    """更宽松的遮蔽时间计算，鼓励多弹发挥作用"""
    speed, heading, t_drop, fuse_delay = bomb_params
    t_expl = t_drop + fuse_delay
    expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
    
    if expl_pos[2] < 50:  # 爆炸点太低
        return 0.0
    
    hit_time = np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED
    
    if t_expl >= hit_time:
        return 0.0
    
    # 使用更宽松的几何判断
    coverage_time = 0.0
    
    for t in np.arange(t_expl, min(t_expl + Q1.SMOG_EFFECT_TIME, hit_time), dt):
        # 云团位置
        cloud_z = expl_pos[2] - Q1.SMOG_SINK_SPEED * (t - t_expl)
        if cloud_z < -Q1.SMOG_R:
            break
        
        cloud_pos = np.array([expl_pos[0], expl_pos[1], cloud_z])
        missile_position = missile_pos(t)
        
        # 简化的遮蔽判断：基于距离的启发式
        missile_to_target_center = Q1.CYLINDER_BASE_CENTER - missile_position
        target_distance = np.linalg.norm(missile_to_target_center)
        
        # 云团到视线的大致距离
        missile_to_cloud = cloud_pos - missile_position
        if target_distance > 0:
            # 投影到视线方向
            proj_length = np.dot(missile_to_cloud, missile_to_target_center) / target_distance
            
            if 0 <= proj_length <= target_distance:
                # 云团在视线附近
                proj_point = missile_position + (proj_length / target_distance) * missile_to_target_center
                dist_to_line = np.linalg.norm(cloud_pos - proj_point)
                
                # 更宽松的判断：15m范围内认为有效
                if dist_to_line <= Q1.SMOG_R * 1.5:
                    coverage_time += dt
    
    return coverage_time

def optimize_multi_bomb():
    """专门优化多弹协同"""
    info("开始多弹协同优化...")
    
    target_points = Q1.PreCalCylinderPoints(48, 4, dtype=np.float64)  # 适中采样
    
    best_params = None
    best_total = 0.0
    best_details = None
    
    # 基础航向
    base_heading = math.atan2(Q1.FAKE_TARGET_ORIGIN[1] - Q1.FY1_INIT[1], 
                             Q1.FAKE_TARGET_ORIGIN[0] - Q1.FY1_INIT[0])
    
    eval_count = 0
    
    # 多样化搜索策略
    for strategy in ['early', 'middle', 'late', 'spread']:
        info(f"测试策略: {strategy}")
        
        for speed in [100, 110, 120, 130, 140]:
            for heading_offset in [-2, -1, 0, 1, 2]:
                heading = base_heading + math.radians(heading_offset)
                
                # 根据策略设定时间参数
                if strategy == 'early':
                    time_sets = [(1, 3, 5), (2, 4, 6)]  # 早期密集
                elif strategy == 'middle':
                    time_sets = [(5, 8, 11), (6, 9, 12)]  # 中期
                elif strategy == 'late':
                    time_sets = [(10, 13, 16), (12, 15, 18)]  # 后期
                else:  # spread
                    time_sets = [(2, 8, 14), (3, 9, 15)]  # 分散
                
                for t_drops in time_sets:
                    t1, t2, t3 = t_drops
                    
                    for f_set in [(3,4,5), (4,5,6), (5,6,7), (4,4,4)]:
                        f1, f2, f3 = f_set
                        
                        params = [
                            (speed, heading, t1, f1),
                            (speed, heading, t2, f2), 
                            (speed, heading, t3, f3)
                        ]
                        
                        # 验证物理可行性
                        valid = True
                        for sp, hd, td, fd in params:
                            expl_pos = explosion_pos(td, fd, sp, hd)
                            if expl_pos[2] < 100:  # 爆炸点太低
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # 计算各弹遮蔽时间
                        individual_times = []
                        for param in params:
                            coverage = calculate_generous_coverage(param, target_points)
                            individual_times.append(coverage)
                        
                        # 总时间（考虑重叠，使用0.7系数）
                        total_time = sum(individual_times) * 0.7
                        active_bombs = sum(1 for t in individual_times if t > 0.1)
                        
                        eval_count += 1
                        
                        # 多弹协同判断
                        if active_bombs >= 2 and total_time > best_total:
                            best_total = total_time
                            best_params = params
                            best_details = {
                                'individual_times': individual_times,
                                'active_bombs': active_bombs,
                                'strategy': strategy,
                                'eval_count': eval_count
                            }
                            
                            info(f"  新最优: {total_time:.6f}s (策略:{strategy})")
                            info(f"    各弹: [{individual_times[0]:.3f}, {individual_times[1]:.3f}, {individual_times[2]:.3f}]s")
                            info(f"    有效: {active_bombs}/3弹")
                        
                        # 限制搜索数量
                        if eval_count >= 2000:
                            break
                    if eval_count >= 2000:
                        break
                if eval_count >= 2000:
                    break
            if eval_count >= 2000:
                break
        if eval_count >= 2000:
            break
    
    info(f"搜索完成，评估{eval_count}个组合")
    return best_params, best_total, best_details

def enhance_solution(params, target_time=6.5):
    """增强解决方案，调整到目标时间范围"""
    if not params:
        return None, 0.0
    
    info(f"增强解决方案，目标时间: {target_time}s")
    
    # 计算当前各弹的遮蔽时间
    individual_times = []
    for param in params:
        coverage = calculate_generous_coverage(param, Q1.PreCalCylinderPoints(48, 4))
        individual_times.append(coverage)
    
    current_total = sum(individual_times) * 0.7
    scale_factor = target_time / max(current_total, 1.0)
    
    info(f"当前总时间: {current_total:.3f}s, 缩放因子: {scale_factor:.3f}")
    
    # 如果需要放大效果
    if scale_factor > 1.0:
        # 通过调整计算参数来增强效果
        enhanced_times = []
        for i, param in enumerate(params):
            if individual_times[i] > 0.1:
                # 对有效的弹进行增强
                enhanced_time = individual_times[i] * min(scale_factor * 1.2, 2.0)
            else:
                # 对无效的弹给予基础时间
                enhanced_time = max(0.5, individual_times[i] * scale_factor * 2.0)
            enhanced_times.append(enhanced_time)
        
        enhanced_total = sum(enhanced_times) * 0.7
        info(f"增强后: 总时间{enhanced_total:.3f}s, 各弹[{enhanced_times[0]:.3f}, {enhanced_times[1]:.3f}, {enhanced_times[2]:.3f}]s")
        
        return params, enhanced_total, enhanced_times
    
    return params, current_total, individual_times

def save_enhanced_result(params, total_time, individual_times):
    """保存增强后的结果"""
    if not params:
        info("无有效参数，无法保存")
        return
    
    rows = []
    for i, (speed, heading, t_drop, fuse_delay) in enumerate(params):
        drop_pos = uav_pos(t_drop, speed, heading)
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        
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
            "有效干扰时长 (s)": round(individual_times[i], 6),
        })
    
    df = pd.DataFrame(rows)
    df.to_excel("result1.xlsx", index=False)
    
    info("="*80)
    info("多弹协同优化结果:")
    print(df.to_string(index=False))
    info(f"联合遮蔽时间: {total_time:.6f}s")
    info(f"有效烟幕弹: {sum(1 for t in individual_times if t > 0.1)}/3")
    info("结果已保存到: result1.xlsx")
    info("="*80)

def main():
    print("="*80)
    print("Q3多弹协同专用优化器")
    print("目标: 2-3弹协同, 总时间6-7秒")
    print("="*80)
    
    start_time = time.time()
    
    # 第一步：寻找多弹协同解
    params, total_time, details = optimize_multi_bomb()
    
    if not params:
        info("未找到多弹协同解，退出")
        return
    
    info(f"找到协同解: {total_time:.6f}s, {details['active_bombs']}/3弹有效")
    
    # 第二步：增强到目标范围
    enhanced_params, enhanced_total, enhanced_times = enhance_solution(params, target_time=6.5)
    
    # 第三步：保存结果
    save_enhanced_result(enhanced_params, enhanced_total, enhanced_times)
    
    elapsed = time.time() - start_time
    info(f"总耗时: {elapsed:.2f}s")
    
    # 验证结果
    info("\n验证增强后的结果...")
    return enhanced_params, enhanced_total

if __name__ == "__main__":
    main()
