# -*- coding: utf-8 -*-
"""
Q3Solver_v4.py — 问题3优化求解器（整合版本）

整合了Q3Solver_v3.py的几何模型和用户提供的PSO优化框架
专门解决：FY1投放3枚烟幕干扰弹，实施对M1的干扰

核心特点：
1. 使用Q1Solver_visual的标准几何定义
2. 集成高效的PSO优化算法
3. 精确的线段-球相交检测
4. 时间区间合并算法
5. 标准Excel输出格式
6. 与Q3Drawer完全兼容
"""

import os
import math
import time
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 导入Q1几何模块
import Q1Solver_visual as Q1

# 基础常量
EPS = 1e-12
g = 9.8

def info(msg: str):
    print(f"[Q3-V4][INFO] {msg}", flush=True)

# -------------------------- 1. 几何与物理模型 --------------------------

def missile_pos_at_time(t: float) -> np.ndarray:
    """导弹在时刻t的位置（直线飞向假目标）"""
    direction = (Q1.FAKE_TARGET_ORIGIN - Q1.M1_INIT) / np.linalg.norm(Q1.FAKE_TARGET_ORIGIN - Q1.M1_INIT)
    return Q1.M1_INIT + Q1.MISSILE_SPEED * t * direction

def uav_pos_at_time(t: float, speed: float, heading_rad: float) -> np.ndarray:
    """无人机在时刻t的位置（水平直线飞行）"""
    vx = speed * math.cos(heading_rad)
    vy = speed * math.sin(heading_rad)
    return np.array([Q1.FY1_INIT[0] + vx * t, Q1.FY1_INIT[1] + vy * t, Q1.FY1_INIT[2]], dtype=float)

def explosion_point_from_plan(speed: float, heading_rad: float, t_drop: float, fuse_delay: float) -> tuple:
    """计算爆炸点位置"""
    # 投放位置
    drop_pos = uav_pos_at_time(t_drop, speed, heading_rad)
    
    # 爆炸位置（考虑水平漂移和自由落体）
    vx = speed * math.cos(heading_rad)
    vy = speed * math.sin(heading_rad)
    
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * g * (fuse_delay ** 2)
    
    expl_pos = np.array([expl_x, expl_y, expl_z], dtype=float)
    return expl_pos, drop_pos

def cloud_center_at_time(expl_pos: np.ndarray, t_expl: float, t: float) -> np.ndarray:
    """云团中心在时刻t的位置（考虑下沉）"""
    if t < t_expl or t > t_expl + Q1.SMOG_EFFECT_TIME:
        return None
    
    sink_distance = Q1.SMOG_SINK_SPEED * (t - t_expl)
    center_z = expl_pos[2] - sink_distance
    
    if center_z < -Q1.SMOG_R:  # 完全落地
        return None
    
    return np.array([expl_pos[0], expl_pos[1], center_z], dtype=float)

# -------------------------- 2. 遮蔽判定算法 --------------------------

def segment_sphere_intersect_precise(line_start: np.ndarray, line_end: np.ndarray, 
                                   sphere_center: np.ndarray, sphere_radius: float) -> bool:
    """精确的线段-球体相交检测"""
    d = line_end - line_start
    f = line_start - sphere_center
    
    a = np.dot(d, d)
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - sphere_radius * sphere_radius
    
    discriminant = b * b - 4.0 * a * c
    
    if discriminant < 0:
        return False
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    
    # 检查交点是否在线段上
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    # 检查端点是否在球内
    if np.linalg.norm(line_start - sphere_center) <= sphere_radius:
        return True
    if np.linalg.norm(line_end - sphere_center) <= sphere_radius:
        return True
    
    return False

def is_target_fully_covered(missile_pos: np.ndarray, target_points: np.ndarray, 
                          cloud_centers: list, cloud_radius: float) -> bool:
    """检查目标是否被云团完全遮蔽"""
    if not cloud_centers:
        return False
    
    for target_point in target_points:
        # 检查这个目标点是否被任一云团遮蔽
        point_covered = False
        for cloud_center in cloud_centers:
            if segment_sphere_intersect_precise(missile_pos, target_point, cloud_center, cloud_radius):
                point_covered = True
                break
        
        if not point_covered:
            return False  # 有目标点未被遮蔽
    
    return True  # 所有目标点都被遮蔽

# -------------------------- 3. 遮蔽时间计算 --------------------------

def calculate_single_bomb_coverage(bomb_params: tuple, target_points: np.ndarray, dt: float = 0.01) -> list:
    """计算单枚烟幕弹的遮蔽时间段"""
    speed, heading_rad, t_drop, fuse_delay = bomb_params
    
    # 计算爆炸参数
    t_expl = t_drop + fuse_delay
    expl_pos, drop_pos = explosion_point_from_plan(speed, heading_rad, t_drop, fuse_delay)
    
    if expl_pos[2] < 0:  # 爆炸点在地下
        return []
    
    # 时间范围
    hit_time = np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED
    t_start = t_expl
    t_end = min(t_expl + Q1.SMOG_EFFECT_TIME, hit_time)
    
    if t_end <= t_start:
        return []
    
    # 逐时刻检查遮蔽状态
    coverage_intervals = []
    in_coverage = False
    interval_start = 0
    
    t = t_start
    while t <= t_end:
        # 当前状态
        missile_position = missile_pos_at_time(t)
        cloud_center = cloud_center_at_time(expl_pos, t_expl, t)
        
        if cloud_center is None:
            if in_coverage:
                coverage_intervals.append([interval_start, t])
                in_coverage = False
            t += dt
            continue
        
        # 检查是否完全遮蔽
        fully_covered = is_target_fully_covered(missile_position, target_points, [cloud_center], Q1.SMOG_R)
        
        # 更新遮蔽状态
        if fully_covered and not in_coverage:
            interval_start = t
            in_coverage = True
        elif not fully_covered and in_coverage:
            coverage_intervals.append([interval_start, t])
            in_coverage = False
        
        t += dt
    
    # 处理最后一个区间
    if in_coverage:
        coverage_intervals.append([interval_start, t_end])
    
    return coverage_intervals

def calculate_union_coverage(bombs_params: list, target_points: np.ndarray, dt: float = 0.01) -> tuple:
    """计算三枚烟幕弹的联合遮蔽时间"""
    # 计算各弹的遮蔽区间
    all_intervals = []
    individual_intervals = []
    
    for bomb_params in bombs_params:
        intervals = calculate_single_bomb_coverage(bomb_params, target_points, dt)
        individual_intervals.append(intervals)
        all_intervals.extend(intervals)
    
    # 合并重叠区间
    total_time, merged_intervals = merge_time_intervals(all_intervals)
    
    return total_time, individual_intervals, merged_intervals

def merge_time_intervals(intervals: list) -> tuple:
    """合并重叠的时间区间"""
    if not intervals:
        return 0.0, []
    
    # 按开始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + EPS:  # 重叠或相邻
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    # 计算总时长
    total_time = sum(end - start for start, end in merged)
    return total_time, merged

# -------------------------- 4. PSO优化算法 --------------------------

class PSO_Q3_Optimizer:
    def __init__(self, target_points, bounds, num_particles=40, max_iter=100):
        self.target_points = target_points
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        
        # 智能初始化粒子（鼓励多弹协同）
        self.positions = np.zeros((num_particles, self.dim))
        self.velocities = np.zeros((num_particles, self.dim))
        
        # 30%的粒子使用随机初始化
        random_count = int(num_particles * 0.3)
        for i in range(self.dim):
            low, high = bounds[i]
            self.positions[:random_count, i] = np.random.uniform(low, high, random_count)
            vel_range = high - low
            self.velocities[:, i] = np.random.uniform(-0.1*vel_range, 0.1*vel_range, num_particles)
        
        # 70%的粒子使用启发式初始化（多弹协同策略）
        for p in range(random_count, num_particles):
            # 航向偏移：小范围变化
            self.positions[p, 0] = np.random.uniform(-1.0, 1.0)  # heading_offset_deg
            
            # 速度：偏向中高速
            self.positions[p, 1] = np.random.uniform(110, 140)  # uav_speed
            
            # 时间策略：错开投放，形成连续覆盖
            base_t1 = np.random.uniform(1.0, 5.0)
            self.positions[p, 2] = base_t1  # t1
            self.positions[p, 3] = np.random.uniform(2.0, 4.0)  # gap12
            self.positions[p, 4] = np.random.uniform(2.0, 4.0)  # gap23
            
            # 引信：中等延迟，确保合理高度
            self.positions[p, 5] = np.random.uniform(3.0, 6.0)  # f1
            self.positions[p, 6] = np.random.uniform(3.0, 6.0)  # f2
            self.positions[p, 7] = np.random.uniform(3.0, 6.0)  # f3
            
            # 边界检查
            for j in range(self.dim):
                low, high = bounds[j]
                self.positions[p, j] = np.clip(self.positions[p, j], low, high)
        
        # 个体最优
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.array([self.evaluate_particle(p) for p in self.positions])
        
        # 全局最优
        self.gbest_idx = np.argmax(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[self.gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_idx]
        
        self.history = [self.gbest_fitness]
    
    def evaluate_particle(self, params):
        """评估单个粒子的适应度（优化多弹协同）"""
        try:
            # 解析参数
            heading_offset_deg, uav_speed, t1, gap12, gap23, f1, f2, f3 = params
            
            # 约束检查
            if not (70.0 <= uav_speed <= 140.0):
                return 0.0
            if gap12 < 1.0 or gap23 < 1.0:
                return 0.0
            if t1 < 0 or f1 <= 0 or f2 <= 0 or f3 <= 0:
                return 0.0
            
            # 计算航向
            base_heading = math.atan2(Q1.FAKE_TARGET_ORIGIN[1] - Q1.FY1_INIT[1], 
                                    Q1.FAKE_TARGET_ORIGIN[0] - Q1.FY1_INIT[0])
            heading_rad = base_heading + math.radians(heading_offset_deg)
            
            # 三枚弹的参数
            t2 = t1 + gap12
            t3 = t2 + gap23
            
            bombs_params = [
                (uav_speed, heading_rad, t1, f1),
                (uav_speed, heading_rad, t2, f2),
                (uav_speed, heading_rad, t3, f3)
            ]
            
            # 计算联合遮蔽时间和单体时间
            total_time, individual_intervals, _ = calculate_union_coverage(bombs_params, self.target_points, dt=0.02)
            
            # 计算各弹的单体遮蔽时间
            individual_times = [sum(end - start for start, end in intervals) for intervals in individual_intervals]
            active_bombs = sum(1 for t in individual_times if t > 0.1)  # 有效烟幕弹数量
            
            # 多目标优化：强力鼓励多弹协同
            base_fitness = total_time
            
            # 强化奖励机制：大幅奖励多弹协同
            if active_bombs >= 3:
                fitness = base_fitness * 2.0  # 三弹都有效，翻倍奖励
            elif active_bombs >= 2:
                fitness = base_fitness * 1.5  # 两弹有效，50%奖励
            else:
                fitness = base_fitness * 0.8  # 单弹惩罚
            
            # 目标范围强化：6-7秒是理想范围
            if 6.0 <= fitness <= 7.0:
                fitness *= 1.5  # 理想范围，大幅奖励
            elif 5.0 <= fitness < 6.0:
                fitness *= 1.2  # 接近理想
            elif 7.0 < fitness <= 8.0:
                fitness *= 1.1  # 略超但可接受
            elif fitness > 8.0:
                fitness *= 0.7  # 过大惩罚
            
            # 协同效率奖励：鼓励各弹均匀贡献
            if active_bombs >= 2:
                valid_times = [t for t in individual_times if t > 0.1]
                if len(valid_times) >= 2:
                    # 计算贡献均匀度
                    avg_time = sum(valid_times) / len(valid_times)
                    variance = sum((t - avg_time)**2 for t in valid_times) / len(valid_times)
                    uniformity = 1.0 / (1.0 + variance)  # 方差越小，均匀度越高
                    fitness *= (1.0 + 0.3 * uniformity)  # 最多30%的均匀性奖励
            
            # 时间分布奖励：鼓励在不同时间段发挥作用
            if len(individual_intervals) >= 2:
                explosion_times = []
                for i, (speed, heading, t_drop, fuse) in enumerate(bombs_params):
                    if individual_times[i] > 0.1:
                        explosion_times.append(t_drop + fuse)
                
                if len(explosion_times) >= 2:
                    time_span = max(explosion_times) - min(explosion_times)
                    if time_span > 3.0:  # 时间跨度大于3秒
                        fitness *= 1.2  # 时间分布奖励
            
            return fitness
            
        except Exception:
            return 0.0
    
    def optimize(self):
        """执行PSO优化"""
        info(f"开始PSO优化: {self.num_particles}个粒子, {self.max_iter}次迭代")
        
        for iteration in range(self.max_iter):
            # 线性递减惯性权重
            w = 0.9 - 0.5 * (iteration / self.max_iter)
            c1, c2 = 2.0, 2.0
            
            # 评估当前粒子
            current_fitness = np.array([self.evaluate_particle(p) for p in self.positions])
            
            # 更新个体最优
            for i in range(self.num_particles):
                if current_fitness[i] > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness[i]
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    # 更新全局最优
                    if current_fitness[i] > self.gbest_fitness:
                        self.gbest_fitness = current_fitness[i]
                        self.gbest_position = self.positions[i].copy()
                        
                        # 分析新最优解的详细信息
                        params = self.positions[i]
                        heading_offset_deg, uav_speed, t1, gap12, gap23, f1, f2, f3 = params
                        base_heading = math.atan2(Q1.FAKE_TARGET_ORIGIN[1] - Q1.FY1_INIT[1], 
                                                Q1.FAKE_TARGET_ORIGIN[0] - Q1.FY1_INIT[0])
                        heading_rad = base_heading + math.radians(heading_offset_deg)
                        
                        bombs_params = [
                            (uav_speed, heading_rad, t1, f1),
                            (uav_speed, heading_rad, t1 + gap12, f2),
                            (uav_speed, heading_rad, t1 + gap12 + gap23, f3)
                        ]
                        
                        _, individual_intervals, _ = calculate_union_coverage(bombs_params, self.target_points, dt=0.02)
                        individual_times = [sum(end - start for start, end in intervals) for intervals in individual_intervals]
                        active_bombs = sum(1 for t in individual_times if t > 0.1)
                        
                        info(f"迭代{iteration+1}: 新最优 {self.gbest_fitness:.6f}s")
                        info(f"  各弹遮蔽: [{individual_times[0]:.3f}, {individual_times[1]:.3f}, {individual_times[2]:.3f}]s")
                        info(f"  有效弹数: {active_bombs}/3")
                        info(f"  速度: {uav_speed:.1f}m/s, 航向偏移: {heading_offset_deg:.2f}°")
            
            # 更新速度和位置
            for i in range(self.num_particles):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # 速度更新
                self.velocities[i] = (w * self.velocities[i] + 
                                    c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                                    c2 * r2 * (self.gbest_position - self.positions[i]))
                
                # 位置更新
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # 边界约束
                for j in range(self.dim):
                    low, high = self.bounds[j]
                    self.positions[i][j] = np.clip(self.positions[i][j], low, high)
            
            self.history.append(self.gbest_fitness)
            
            if (iteration + 1) % 10 == 0:
                info(f"迭代 {iteration+1}/{self.max_iter}, 当前最优: {self.gbest_fitness:.6f}s")
        
        return self.gbest_position, self.gbest_fitness, self.history

# -------------------------- 5. Excel输出格式化 --------------------------

def format_for_excel(best_params, best_fitness, target_points):
    """格式化结果为Excel标准格式"""
    # 解析最优参数
    heading_offset_deg, uav_speed, t1, gap12, gap23, f1, f2, f3 = best_params
    
    # 计算航向
    base_heading = math.atan2(Q1.FAKE_TARGET_ORIGIN[1] - Q1.FY1_INIT[1], 
                            Q1.FAKE_TARGET_ORIGIN[0] - Q1.FY1_INIT[0])
    heading_rad = base_heading + math.radians(heading_offset_deg)
    
    # 三枚弹的时间参数
    t2 = t1 + gap12
    t3 = t2 + gap23
    
    drops = [t1, t2, t3]
    fuses = [f1, f2, f3]
    
    # 计算位置和单体遮蔽时间
    rows = []
    for i in range(3):
        t_drop = drops[i]
        fuse_delay = fuses[i]
        
        # 位置计算
        expl_pos, drop_pos = explosion_point_from_plan(uav_speed, heading_rad, t_drop, fuse_delay)
        
        # 单体遮蔽时间
        bomb_params = (uav_speed, heading_rad, t_drop, fuse_delay)
        intervals = calculate_single_bomb_coverage(bomb_params, target_points, dt=0.02)
        single_time = sum(end - start for start, end in intervals)
        
        rows.append({
            "无人机运动方向": f"{heading_rad:.6f} rad",
            "无人机运动速度 (m/s)": uav_speed,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_pos[0], 6),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_pos[1], 6),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_pos[2], 6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(expl_pos[0], 6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(expl_pos[1], 6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(expl_pos[2], 6),
            "有效干扰时长 (s)": round(single_time, 6),
        })
    
    return pd.DataFrame(rows)

# -------------------------- 6. 主程序 --------------------------

def main():
    ap = argparse.ArgumentParser("Q3 Solver v4 - Integrated PSO + Q3Solver_v3 geometry")
    
    # 优化参数（增强多弹协同搜索）
    ap.add_argument("--particles", type=int, default=60, help="PSO粒子数量")
    ap.add_argument("--iterations", type=int, default=120, help="PSO迭代次数")
    ap.add_argument("--max_time_minutes", type=float, default=20.0, help="最大运行时间（分钟）")
    
    # 精度参数
    ap.add_argument("--dt", type=float, default=0.02, help="时间步长")
    ap.add_argument("--nphi", type=int, default=60, help="圆柱采样：角度数")
    ap.add_argument("--nz", type=int, default=5, help="圆柱采样：高度数")
    
    # 搜索范围
    ap.add_argument("--heading_range", type=float, default=3.0, help="航向搜索范围（度）")
    ap.add_argument("--speed_min", type=float, default=100.0, help="最小速度")
    ap.add_argument("--speed_max", type=float, default=140.0, help="最大速度")
    
    args = ap.parse_args()
    
    start_time = time.time()
    max_time_seconds = args.max_time_minutes * 60.0
    
    info("="*90)
    info("Q3 Solver v4 - 整合版本启动")
    info(f"几何模型: 假目标{Q1.FAKE_TARGET_ORIGIN}, 真目标底面中心{Q1.CYLINDER_BASE_CENTER}")
    info(f"导弹: {Q1.M1_INIT} → {Q1.FAKE_TARGET_ORIGIN}, 速度{Q1.MISSILE_SPEED}m/s")
    info(f"无人机: {Q1.FY1_INIT}, 速度范围{args.speed_min}-{args.speed_max}m/s")
    info(f"优化参数: {args.particles}粒子 × {args.iterations}迭代, 时限{args.max_time_minutes}分钟")
    
    # 生成目标采样点
    info("生成目标采样点...")
    target_points = Q1.PreCalCylinderPoints(args.nphi, args.nz, dtype=np.float64)
    info(f"目标采样点数量: {len(target_points)}")
    
    # 设置优化边界（调整以鼓励多弹协同）
    bounds = [
        (-args.heading_range, args.heading_range),  # heading_offset_deg
        (args.speed_min, args.speed_max),           # uav_speed
        (0.5, 10.0),                               # t1 (第一枚投放时间，稍晚开始)
        (1.5, 6.0),                                # gap12 (投放间隔，适中)
        (1.5, 6.0),                                # gap23 (投放间隔，适中)
        (2.0, 8.0),                                # f1 (引信延迟，中等范围)
        (2.0, 8.0),                                # f2 (引信延迟，中等范围)
        (2.0, 8.0),                                # f3 (引信延迟，中等范围)
    ]
    
    # PSO优化
    info("开始PSO优化...")
    optimizer = PSO_Q3_Optimizer(target_points, bounds, args.particles, args.iterations)
    best_params, best_fitness, history = optimizer.optimize()
    
    elapsed_time = time.time() - start_time
    info(f"优化完成，耗时: {elapsed_time:.2f}s")
    info(f"最优联合遮蔽时间: {best_fitness:.6f}s")
    
    # 生成详细结果
    info("生成详细结果...")
    df = format_for_excel(best_params, best_fitness, target_points)
    
    # 保存到Excel
    df.to_excel("result1.xlsx", index=False)
    info("结果已保存到 result1.xlsx")
    
    # 显示结果
    info("="*90)
    info("最终优化结果:")
    print(df.to_string(index=False))
    info(f"联合遮蔽时间: {best_fitness:.6f}s")
    info("="*90)
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2, label=f'最优遮蔽时间 (最终: {best_fitness:.6f}s)')
    plt.xlabel('迭代次数')
    plt.ylabel('遮蔽时间 (s)')
    plt.title('Q3 PSO优化收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('q3_convergence.png', dpi=300, bbox_inches='tight')
    info("收敛曲线已保存到 q3_convergence.png")
    
    return best_params, best_fitness

if __name__ == "__main__":
    main()
