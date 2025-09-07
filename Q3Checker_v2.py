# -*- coding: utf-8 -*-
"""
Q3Checker_v2.py - 问题3高精度验证器

基于对题目的正确理解，实现高精度物理模拟：
1. 几何关系：假目标(0,0,0)，真目标圆柱底面中心(0,200,0)
2. 导弹轨迹：M1直线飞向假目标
3. 遮蔽判定：导弹→真目标圆柱的视线被烟幕球遮挡
4. 物理模拟：完整的无人机运动、烟幕弹运动、云团下沉过程

使用result1.xlsx中的解决方案数据进行验证，输出成功遮蔽M1的总时间。
"""

import math
import numpy as np
import pandas as pd
import argparse
import time
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# 物理常量与几何定义
# =============================================================================

# 物理常量
G = 9.8                    # 重力加速度 m/s²
CLOUD_RADIUS = 10.0        # 烟幕球半径 m
CLOUD_SINK_SPEED = 3.0     # 烟幕下沉速度 m/s
CLOUD_EFFECT_TIME = 20.0   # 烟幕有效时间 s
MISSILE_SPEED = 300.0      # 导弹飞行速度 m/s

# 几何定义（关键：基于题目的正确理解）
FAKE_TARGET = np.array([0.0, 0.0, 0.0])           # 假目标在原点
TRUE_TARGET_BASE_CENTER = np.array([0.0, 200.0, 0.0])  # 真目标圆柱底面中心
CYLINDER_RADIUS = 7.0      # 圆柱半径 m
CYLINDER_HEIGHT = 10.0     # 圆柱高度 m

# 初始位置
M1_INIT = np.array([20000.0, 0.0, 2000.0])        # 导弹M1初始位置
FY1_INIT = np.array([17800.0, 0.0, 1800.0])       # 无人机FY1初始位置

# 时间约束
MIN_DROP_INTERVAL = 1.0    # 两枚烟幕弹最小投放间隔 s
UAV_SPEED_MIN = 70.0       # 无人机最小速度 m/s
UAV_SPEED_MAX = 140.0      # 无人机最大速度 m/s

# =============================================================================
# 物理模拟函数
# =============================================================================

def missile_position(t: float) -> np.ndarray:
    """计算t时刻导弹M1的位置
    
    导弹直线飞向假目标(0,0,0)
    """
    if t < 0:
        return M1_INIT.copy()
    
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    position = M1_INIT + MISSILE_SPEED * t * direction
    
    # 检查是否已到达假目标
    if np.linalg.norm(position - FAKE_TARGET) < 1.0:  # 1米精度
        return FAKE_TARGET.copy()
    
    return position

def uav_position(t: float, speed: float, heading: float) -> np.ndarray:
    """计算t时刻无人机FY1的位置
    
    Args:
        t: 时间 s
        speed: 飞行速度 m/s
        heading: 飞行方向 rad（数学角度，逆时针为正）
        
    Returns:
        3D位置向量
    """
    if t < 0:
        return FY1_INIT.copy()
    
    # 水平方向的速度分量
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    # 等高度飞行
    return np.array([
        FY1_INIT[0] + vx * t,
        FY1_INIT[1] + vy * t,
        FY1_INIT[2]  # 高度保持不变
    ])

def smoke_bomb_trajectory(t_drop: float, fuse_delay: float, speed: float, heading: float, t: float) -> Optional[np.ndarray]:
    """计算烟幕弹在t时刻的位置（投放后、爆炸前）
    
    Args:
        t_drop: 投放时间 s
        fuse_delay: 引信延迟时间 s
        speed: 无人机速度 m/s
        heading: 无人机航向 rad
        t: 当前时间 s
        
    Returns:
        烟幕弹位置，如果已爆炸则返回None
    """
    if t < t_drop:
        return None  # 尚未投放
    
    t_explosion = t_drop + fuse_delay
    if t >= t_explosion:
        return None  # 已爆炸
    
    # 投放位置
    drop_pos = uav_position(t_drop, speed, heading)
    
    # 自投放时刻起的时间
    dt = t - t_drop
    
    # 水平方向：保持无人机投放时的速度
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    # 垂直方向：自由落体
    x = drop_pos[0] + vx * dt
    y = drop_pos[1] + vy * dt
    z = drop_pos[2] - 0.5 * G * dt * dt
    
    return np.array([x, y, z])

def explosion_position(t_drop: float, fuse_delay: float, speed: float, heading: float) -> np.ndarray:
    """计算烟幕弹的爆炸位置
    
    Args:
        t_drop: 投放时间 s
        fuse_delay: 引信延迟时间 s
        speed: 无人机速度 m/s
        heading: 无人机航向 rad
        
    Returns:
        爆炸位置3D坐标
    """
    # 投放位置
    drop_pos = uav_position(t_drop, speed, heading)
    
    # 水平漂移
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    # 爆炸位置
    x = drop_pos[0] + vx * fuse_delay
    y = drop_pos[1] + vy * fuse_delay
    z = drop_pos[2] - 0.5 * G * fuse_delay * fuse_delay
    
    return np.array([x, y, z])

def cloud_center_position(explosion_pos: np.ndarray, t_explosion: float, t: float) -> Optional[np.ndarray]:
    """计算t时刻烟幕云团的中心位置
    
    Args:
        explosion_pos: 爆炸位置
        t_explosion: 爆炸时间 s
        t: 当前时间 s
        
    Returns:
        云团中心位置，如果云团不存在则返回None
    """
    if t < t_explosion:
        return None  # 尚未爆炸
    
    if t > t_explosion + CLOUD_EFFECT_TIME:
        return None  # 云团已消散
    
    # 云团下沉距离
    sink_distance = CLOUD_SINK_SPEED * (t - t_explosion)
    center_z = explosion_pos[2] - sink_distance
    
    # 检查是否完全落地（云团底部接触地面）
    if center_z < CLOUD_RADIUS:
        return None
    
    return np.array([explosion_pos[0], explosion_pos[1], center_z])

# =============================================================================
# 几何计算函数
# =============================================================================

def generate_cylinder_sampling_points(n_circumference: int = 36, n_height: int = 5) -> np.ndarray:
    """生成圆柱体表面的采样点
    
    Args:
        n_circumference: 圆周方向采样点数
        n_height: 高度方向采样点数
        
    Returns:
        采样点数组 (N, 3)
    """
    points = []
    
    # 底面圆周
    for i in range(n_circumference):
        angle = 2 * math.pi * i / n_circumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2]  # 底面
        points.append([x, y, z])
    
    # 顶面圆周
    for i in range(n_circumference):
        angle = 2 * math.pi * i / n_circumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT  # 顶面
        points.append([x, y, z])
    
    # 侧面（中间高度层）
    for k in range(1, n_height - 1):
        height_ratio = k / (n_height - 1)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT * height_ratio
        
        # 每层采样点数可以适当减少
        n_side = n_circumference // 2
        for i in range(n_side):
            angle = 2 * math.pi * i / n_side
            x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
            y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
            points.append([x, y, z])
    
    return np.array(points)

def line_sphere_intersection(line_start: np.ndarray, line_end: np.ndarray, 
                           sphere_center: np.ndarray, sphere_radius: float) -> bool:
    """检测线段与球体是否相交
    
    Args:
        line_start: 线段起点
        line_end: 线段终点
        sphere_center: 球心
        sphere_radius: 球半径
        
    Returns:
        True如果相交，False如果不相交
    """
    # 线段向量
    d = line_end - line_start
    f = line_start - sphere_center
    
    # 求解二次方程 |line_start + t*d - sphere_center|² = sphere_radius²
    # 展开得到 a*t² + b*t + c = 0
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - sphere_radius * sphere_radius
    
    # 判别式
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False  # 无实数解，不相交
    
    if abs(a) < 1e-10:  # 线段退化为点
        return np.linalg.norm(line_start - sphere_center) <= sphere_radius
    
    # 计算两个解
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    # 检查解是否在线段参数范围[0,1]内
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    # 检查线段是否完全在球内
    if (t1 < 0 and t2 > 1) or (t2 < 0 and t1 > 1):
        return True
    
    return False

def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """计算点到线段的最短距离
    
    Args:
        point: 目标点
        line_start: 线段起点
        line_end: 线段终点
        
    Returns:
        最短距离
    """
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq < 1e-10:  # 线段退化为点
        return np.linalg.norm(point - line_start)
    
    # 计算投影参数
    t = np.dot(point - line_start, line_vec) / line_len_sq
    t = max(0, min(1, t))  # 限制在线段上
    
    # 最近点
    closest_point = line_start + t * line_vec
    return np.linalg.norm(point - closest_point)

# =============================================================================
# 遮蔽效果计算
# =============================================================================

def is_target_blocked_at_time(missile_pos: np.ndarray, target_points: np.ndarray, 
                             cloud_centers: List[np.ndarray]) -> bool:
    """检查在给定时刻目标是否被云团遮蔽
    
    Args:
        missile_pos: 导弹位置
        target_points: 目标采样点数组 (N, 3)
        cloud_centers: 有效云团中心列表
        
    Returns:
        True如果目标被完全遮蔽，False否则
    """
    if not cloud_centers:
        return False
    
    # 对每个目标点，检查是否被至少一个云团遮蔽
    for target_point in target_points:
        point_blocked = False
        
        for cloud_center in cloud_centers:
            if line_sphere_intersection(missile_pos, target_point, cloud_center, CLOUD_RADIUS):
                point_blocked = True
                break
        
        if not point_blocked:
            return False  # 有目标点未被遮蔽
    
    return True  # 所有目标点都被遮蔽

def calculate_blocking_coverage(bomb_params: List[Tuple[float, float, float, float]], 
                              time_step: float = 0.01) -> Dict[str, Any]:
    """计算烟幕弹的遮蔽覆盖效果
    
    Args:
        bomb_params: 烟幕弹参数列表，每个元素为(t_drop, fuse_delay, speed, heading)
        time_step: 时间步长 s
        
    Returns:
        包含遮蔽时间统计的字典
    """
    print(f"开始计算遮蔽覆盖效果，时间步长={time_step}s")
    
    # 生成目标采样点
    target_points = generate_cylinder_sampling_points(n_circumference=24, n_height=5)
    print(f"生成目标采样点：{len(target_points)}个")
    
    # 计算所有爆炸事件
    explosions = []
    for i, (t_drop, fuse_delay, speed, heading) in enumerate(bomb_params):
        t_explosion = t_drop + fuse_delay
        explosion_pos = explosion_position(t_drop, fuse_delay, speed, heading)
        explosions.append({
            'bomb_id': i + 1,
            't_drop': t_drop,
            't_explosion': t_explosion,
            'explosion_pos': explosion_pos,
            'speed': speed,
            'heading': heading
        })
        print(f"烟幕弹{i+1}: 投放时间={t_drop:.3f}s, 爆炸时间={t_explosion:.3f}s, "
              f"爆炸位置=({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
    
    # 计算导弹命中时间
    missile_hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    print(f"导弹命中时间: {missile_hit_time:.3f}s")
    
    # 确定模拟时间范围
    earliest_explosion = min(exp['t_explosion'] for exp in explosions)
    latest_effect_end = max(exp['t_explosion'] + CLOUD_EFFECT_TIME for exp in explosions)
    
    sim_start_time = max(0, earliest_explosion - 1.0)  # 提前1秒开始
    sim_end_time = min(missile_hit_time, latest_effect_end)
    
    print(f"模拟时间范围: {sim_start_time:.3f}s 到 {sim_end_time:.3f}s")
    
    if sim_end_time <= sim_start_time:
        print("警告：无有效模拟时间范围")
        return {
            'total_blocking_time': 0.0,
            'blocking_intervals': [],
            'individual_bomb_times': [0.0] * len(bomb_params),
            'simulation_details': {
                'time_range': (sim_start_time, sim_end_time),
                'total_time_points': 0
            }
        }
    
    # 时间步进模拟
    time_points = np.arange(sim_start_time, sim_end_time + time_step, time_step)
    blocking_mask = np.zeros(len(time_points), dtype=bool)
    individual_blocking_masks = [np.zeros(len(time_points), dtype=bool) for _ in range(len(bomb_params))]
    
    print(f"开始时间步进模拟，共{len(time_points)}个时间点...")
    
    for i, t in enumerate(time_points):
        if i % 1000 == 0:  # 每1000个点报告一次进度
            progress = i / len(time_points) * 100
            print(f"模拟进度: {progress:.1f}%")
        
        # 当前导弹位置
        missile_pos = missile_position(t)
        
        # 收集当前有效的云团中心
        active_clouds = []
        for j, explosion in enumerate(explosions):
            cloud_center = cloud_center_position(
                explosion['explosion_pos'], 
                explosion['t_explosion'], 
                t
            )
            if cloud_center is not None:
                active_clouds.append(cloud_center)
                
                # 单个烟幕弹的遮蔽效果
                if is_target_blocked_at_time(missile_pos, target_points, [cloud_center]):
                    individual_blocking_masks[j][i] = True
        
        # 联合遮蔽效果
        if is_target_blocked_at_time(missile_pos, target_points, active_clouds):
            blocking_mask[i] = True
    
    print("模拟完成，正在统计结果...")
    
    # 统计结果
    total_blocking_time = np.sum(blocking_mask) * time_step
    individual_bomb_times = [np.sum(mask) * time_step for mask in individual_blocking_masks]
    
    # 找出连续的遮蔽时间段
    blocking_intervals = []
    in_interval = False
    interval_start = None
    
    for i, blocked in enumerate(blocking_mask):
        if blocked and not in_interval:
            # 开始一个新的遮蔽区间
            in_interval = True
            interval_start = time_points[i]
        elif not blocked and in_interval:
            # 结束当前遮蔽区间
            in_interval = False
            interval_end = time_points[i-1] if i > 0 else time_points[i]
            blocking_intervals.append((interval_start, interval_end))
    
    # 处理最后一个区间（如果模拟结束时仍在遮蔽中）
    if in_interval:
        blocking_intervals.append((interval_start, time_points[-1]))
    
    result = {
        'total_blocking_time': total_blocking_time,
        'blocking_intervals': blocking_intervals,
        'individual_bomb_times': individual_bomb_times,
        'simulation_details': {
            'time_range': (sim_start_time, sim_end_time),
            'time_step': time_step,
            'total_time_points': len(time_points),
            'missile_hit_time': missile_hit_time,
            'target_sampling_points': len(target_points)
        }
    }
    
    return result

# =============================================================================
# 数据读取和解析
# =============================================================================

def parse_heading_from_string(heading_str: str) -> float:
    """从字符串中解析航向角度
    
    Args:
        heading_str: 航向字符串，如 "3.150319 rad"
        
    Returns:
        航向角度（弧度）
    """
    if isinstance(heading_str, (int, float)):
        return float(heading_str)
    
    # 提取数字部分
    import re
    match = re.search(r'([-+]?[\d\.]+)', str(heading_str))
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"无法解析航向角度: {heading_str}")

def load_solution_from_excel(excel_path: str) -> Dict[str, Any]:
    """从Excel文件加载解决方案数据
    
    Args:
        excel_path: Excel文件路径
        
    Returns:
        解析后的解决方案数据
    """
    print(f"正在读取解决方案文件: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"成功读取Excel文件，包含{len(df)}行数据")
        print("列名:", list(df.columns))
        
        # 解析数据
        bombs = []
        
        for idx, row in df.iterrows():
            # 解析基本参数
            heading = parse_heading_from_string(row['无人机运动方向'])
            speed = float(row['无人机运动速度 (m/s)'])
            bomb_id = int(row['烟幕干扰弹编号'])
            
            # 解析位置信息
            drop_x = float(row['烟幕干扰弹投放点的x坐标 (m)'])
            drop_y = float(row['烟幕干扰弹投放点的y坐标 (m)'])
            drop_z = float(row['烟幕干扰弹投放点的z坐标 (m)'])
            
            explosion_x = float(row['烟幕干扰弹起爆点的x坐标 (m)'])
            explosion_y = float(row['烟幕干扰弹起爆点的y坐标 (m)'])
            explosion_z = float(row['烟幕干扰弹起爆点的z坐标 (m)'])
            
            # 从几何关系推导时间参数
            # 假设无人机按恒定速度和方向飞行
            drop_pos = np.array([drop_x, drop_y, drop_z])
            explosion_pos = np.array([explosion_x, explosion_y, explosion_z])
            
            # 计算投放时间（基于无人机从初始位置到投放点的距离）
            distance_to_drop = np.linalg.norm(drop_pos[:2] - FY1_INIT[:2])  # 水平距离
            t_drop = distance_to_drop / speed
            
            # 计算引信延迟时间（基于爆炸点的几何位置）
            horizontal_drift = np.linalg.norm(explosion_pos[:2] - drop_pos[:2])
            fuse_delay = horizontal_drift / speed
            
            # 验证垂直位置的一致性
            expected_explosion_z = drop_pos[2] - 0.5 * G * fuse_delay * fuse_delay
            z_error = abs(expected_explosion_z - explosion_pos[2])
            if z_error > 10.0:  # 10米误差阈值
                print(f"警告：烟幕弹{bomb_id}的垂直位置可能不一致，误差={z_error:.1f}m")
            
            bombs.append({
                'bomb_id': bomb_id,
                't_drop': t_drop,
                'fuse_delay': fuse_delay,
                'speed': speed,
                'heading': heading,
                'drop_pos': drop_pos,
                'explosion_pos': explosion_pos,
                'individual_effect_time': float(row.get('有效干扰时长 (s)', 0.0))
            })
        
        # 按烟幕弹编号排序
        bombs.sort(key=lambda x: x['bomb_id'])
        
        # 验证投放间隔约束
        for i in range(1, len(bombs)):
            interval = bombs[i]['t_drop'] - bombs[i-1]['t_drop']
            if interval < MIN_DROP_INTERVAL:
                print(f"警告：烟幕弹{bombs[i-1]['bomb_id']}和{bombs[i]['bomb_id']}的投放间隔"
                      f"({interval:.3f}s)小于最小要求({MIN_DROP_INTERVAL}s)")
        
        # 验证无人机速度约束
        for bomb in bombs:
            if not (UAV_SPEED_MIN <= bomb['speed'] <= UAV_SPEED_MAX):
                print(f"警告：烟幕弹{bomb['bomb_id']}的无人机速度({bomb['speed']:.1f}m/s)"
                      f"超出允许范围[{UAV_SPEED_MIN}, {UAV_SPEED_MAX}]")
        
        solution = {
            'bombs': bombs,
            'common_heading': bombs[0]['heading'] if bombs else 0.0,
            'common_speed': bombs[0]['speed'] if bombs else 0.0,
            'bomb_params': [(b['t_drop'], b['fuse_delay'], b['speed'], b['heading']) 
                           for b in bombs]
        }
        
        print("解决方案数据解析完成:")
        for bomb in bombs:
            print(f"  烟幕弹{bomb['bomb_id']}: t_drop={bomb['t_drop']:.3f}s, "
                  f"fuse_delay={bomb['fuse_delay']:.3f}s, speed={bomb['speed']:.1f}m/s")
        
        return solution
        
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        raise

# =============================================================================
# 主程序
# =============================================================================

def print_simulation_results(results: Dict[str, Any], solution: Dict[str, Any]):
    """打印模拟结果
    
    Args:
        results: 模拟结果
        solution: 解决方案数据
    """
    print("\n" + "="*80)
    print("Q3 高精度验证结果")
    print("="*80)
    
    # 基本信息
    print(f"模拟时间范围: {results['simulation_details']['time_range'][0]:.3f}s "
          f"到 {results['simulation_details']['time_range'][1]:.3f}s")
    print(f"时间步长: {results['simulation_details']['time_step']:.4f}s")
    print(f"时间点总数: {results['simulation_details']['total_time_points']}")
    print(f"目标采样点数: {results['simulation_details']['target_sampling_points']}")
    print(f"导弹命中时间: {results['simulation_details']['missile_hit_time']:.3f}s")
    
    # 主要结果
    print(f"\n🎯 成功遮蔽M1的总时间: {results['total_blocking_time']:.6f} 秒")
    
    # 遮蔽时间段
    if results['blocking_intervals']:
        print(f"\n📊 遮蔽时间段 (共{len(results['blocking_intervals'])}段):")
        total_interval_time = 0
        for i, (start, end) in enumerate(results['blocking_intervals'], 1):
            duration = end - start
            total_interval_time += duration
            print(f"  第{i}段: {start:.3f}s - {end:.3f}s (持续{duration:.3f}s)")
        print(f"  总计: {total_interval_time:.6f}s")
    else:
        print("\n📊 无有效遮蔽时间段")
    
    # 各烟幕弹的单独效果
    print(f"\n🚀 各烟幕弹单独遮蔽时间:")
    for i, (bomb_time, bomb_data) in enumerate(zip(results['individual_bomb_times'], solution['bombs'])):
        expected_time = bomb_data['individual_effect_time']
        print(f"  烟幕弹{bomb_data['bomb_id']}: {bomb_time:.6f}s "
              f"(Excel中记录: {expected_time:.6f}s)")
    
    # 效果评估
    print(f"\n📈 效果评估:")
    max_possible_time = results['simulation_details']['time_range'][1] - results['simulation_details']['time_range'][0]
    coverage_ratio = results['total_blocking_time'] / max_possible_time * 100 if max_possible_time > 0 else 0
    print(f"  遮蔽覆盖率: {coverage_ratio:.2f}%")
    
    if results['total_blocking_time'] > 0.1:
        print(f"  ✅ 遮蔽效果显著")
    elif results['total_blocking_time'] > 0.01:
        print(f"  ⚠️  遮蔽效果有限")
    else:
        print(f"  ❌ 几乎无遮蔽效果")
    
    print("="*80)

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='Q3 问题3高精度验证器')
    parser.add_argument('--excel', type=str, default='result1.xlsx',
                       help='解决方案Excel文件路径')
    parser.add_argument('--time-step', type=float, default=0.01,
                       help='时间步长 (秒)')
    parser.add_argument('--target-points', type=int, default=24,
                       help='圆柱体圆周采样点数')
    parser.add_argument('--height-layers', type=int, default=5,
                       help='圆柱体高度层数')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    print("Q3Checker_v2 - 问题3高精度验证器")
    print("基于题目的正确理解，验证烟幕干扰弹对M1导弹的遮蔽效果")
    print(f"使用解决方案文件: {args.excel}")
    print(f"时间步长: {args.time_step}s")
    
    try:
        # 加载解决方案
        solution = load_solution_from_excel(args.excel)
        
        # 执行高精度模拟
        start_time = time.time()
        results = calculate_blocking_coverage(
            solution['bomb_params'], 
            time_step=args.time_step
        )
        end_time = time.time()
        
        print(f"\n模拟完成，耗时: {end_time - start_time:.2f}秒")
        
        # 输出结果
        print_simulation_results(results, solution)
        
        # 保存详细结果（可选）
        if args.verbose:
            print(f"\n详细模拟参数:")
            print(f"  重力加速度: {G} m/s²")
            print(f"  烟幕球半径: {CLOUD_RADIUS} m")
            print(f"  烟幕下沉速度: {CLOUD_SINK_SPEED} m/s")
            print(f"  烟幕有效时间: {CLOUD_EFFECT_TIME} s")
            print(f"  导弹速度: {MISSILE_SPEED} m/s")
            print(f"  假目标位置: {FAKE_TARGET}")
            print(f"  真目标位置: {TRUE_TARGET_BASE_CENTER}")
            print(f"  圆柱半径: {CYLINDER_RADIUS} m")
            print(f"  圆柱高度: {CYLINDER_HEIGHT} m")
        
        return results['total_blocking_time']
        
    except Exception as e:
        print(f"验证过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    blocking_time = main()
    print(f"\n🎯 最终结果：成功遮蔽M1的时间为 {blocking_time:.6f} 秒")
