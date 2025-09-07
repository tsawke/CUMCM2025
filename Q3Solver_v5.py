# -*- coding: utf-8 -*-
"""
Q3Solver_v5.py - 问题3高效混合优化求解器

基于Q3Checker_v2.py的精确物理模型，使用混合优化算法（PSO+SA）寻找最优遮蔽方案：
1. 多线程并行计算提高效率
2. 粒子群优化(PSO) + 模拟退火(SA)混合算法
3. 智能约束处理和参数空间缩减
4. 与Q3Checker_v2.py完全兼容的物理模型

目标：最大化对M1导弹的遮蔽时间
"""

import math
import numpy as np
import pandas as pd
import time
import argparse
import random
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from copy import deepcopy

# 导入Q3Checker_v2的物理模型（确保完全一致）
# 物理常量
G = 9.8                    # 重力加速度 m/s²
CLOUD_RADIUS = 10.0        # 烟幕球半径 m
CLOUD_SINK_SPEED = 3.0     # 烟幕下沉速度 m/s
CLOUD_EFFECT_TIME = 20.0   # 烟幕有效时间 s
MISSILE_SPEED = 300.0      # 导弹飞行速度 m/s

# 几何定义
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BASE_CENTER = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# 初始位置
M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

# 约束条件
MIN_DROP_INTERVAL = 1.0
UAV_SPEED_MIN = 70.0
UAV_SPEED_MAX = 140.0

# =============================================================================
# 物理模拟函数（与Q3Checker_v2.py完全一致）
# =============================================================================

def missile_position(t: float) -> np.ndarray:
    """计算t时刻导弹M1的位置"""
    if t < 0:
        return M1_INIT.copy()
    
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    position = M1_INIT + MISSILE_SPEED * t * direction
    
    if np.linalg.norm(position - FAKE_TARGET) < 1.0:
        return FAKE_TARGET.copy()
    
    return position

def uav_position(t: float, speed: float, heading: float) -> np.ndarray:
    """计算t时刻无人机FY1的位置"""
    if t < 0:
        return FY1_INIT.copy()
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    return np.array([
        FY1_INIT[0] + vx * t,
        FY1_INIT[1] + vy * t,
        FY1_INIT[2]
    ])

def explosion_position(t_drop: float, fuse_delay: float, speed: float, heading: float) -> np.ndarray:
    """计算烟幕弹的爆炸位置"""
    drop_pos = uav_position(t_drop, speed, heading)
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    x = drop_pos[0] + vx * fuse_delay
    y = drop_pos[1] + vy * fuse_delay
    z = drop_pos[2] - 0.5 * G * fuse_delay * fuse_delay
    
    return np.array([x, y, z])

def cloud_center_position(explosion_pos: np.ndarray, t_explosion: float, t: float) -> Optional[np.ndarray]:
    """计算t时刻烟幕云团的中心位置"""
    if t < t_explosion or t > t_explosion + CLOUD_EFFECT_TIME:
        return None
    
    sink_distance = CLOUD_SINK_SPEED * (t - t_explosion)
    center_z = explosion_pos[2] - sink_distance
    
    if center_z < CLOUD_RADIUS:
        return None
    
    return np.array([explosion_pos[0], explosion_pos[1], center_z])

def generate_cylinder_sampling_points(n_circumference: int = 24, n_height: int = 5) -> np.ndarray:
    """生成圆柱体表面的采样点"""
    points = []
    
    # 底面圆周
    for i in range(n_circumference):
        angle = 2 * math.pi * i / n_circumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2]
        points.append([x, y, z])
    
    # 顶面圆周
    for i in range(n_circumference):
        angle = 2 * math.pi * i / n_circumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT
        points.append([x, y, z])
    
    # 侧面
    for k in range(1, n_height - 1):
        height_ratio = k / (n_height - 1)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT * height_ratio
        
        n_side = n_circumference // 2
        for i in range(n_side):
            angle = 2 * math.pi * i / n_side
            x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
            y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
            points.append([x, y, z])
    
    return np.array(points)

def line_sphere_intersection(line_start: np.ndarray, line_end: np.ndarray, 
                           sphere_center: np.ndarray, sphere_radius: float) -> bool:
    """检测线段与球体是否相交"""
    d = line_end - line_start
    f = line_start - sphere_center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - sphere_radius * sphere_radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False
    
    if abs(a) < 1e-10:
        return np.linalg.norm(line_start - sphere_center) <= sphere_radius
    
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    if (t1 < 0 and t2 > 1) or (t2 < 0 and t1 > 1):
        return True
    
    return False

def is_target_blocked_at_time(missile_pos: np.ndarray, target_points: np.ndarray, 
                             cloud_centers: List[np.ndarray]) -> bool:
    """检查在给定时刻目标是否被云团遮蔽"""
    if not cloud_centers:
        return False
    
    for target_point in target_points:
        point_blocked = False
        
        for cloud_center in cloud_centers:
            if line_sphere_intersection(missile_pos, target_point, cloud_center, CLOUD_RADIUS):
                point_blocked = True
                break
        
        if not point_blocked:
            return False
    
    return True

# =============================================================================
# 优化相关数据结构
# =============================================================================

@dataclass
class BombConfig:
    """单个烟幕弹配置"""
    t_drop: float      # 投放时间
    fuse_delay: float  # 引信延迟
    speed: float       # 无人机速度
    heading: float     # 无人机航向

@dataclass
class Solution:
    """完整解决方案"""
    bombs: List[BombConfig]
    fitness: float = 0.0  # 遮蔽时间
    
    def to_bomb_params(self) -> List[Tuple[float, float, float, float]]:
        """转换为bomb_params格式"""
        return [(b.t_drop, b.fuse_delay, b.speed, b.heading) for b in self.bombs]

# =============================================================================
# 快速适应度评估函数
# =============================================================================

def evaluate_solution_fitness(solution: Solution, target_points: np.ndarray, 
                             time_step: float = 0.02, quick_eval: bool = True) -> float:
    """快速评估解决方案的适应度（遮蔽时间）"""
    try:
        bomb_params = solution.to_bomb_params()
        
        # 计算爆炸事件
        explosions = []
        for i, (t_drop, fuse_delay, speed, heading) in enumerate(bomb_params):
            t_explosion = t_drop + fuse_delay
            explosion_pos = explosion_position(t_drop, fuse_delay, speed, heading)
            
            # 快速可行性检查
            if explosion_pos[2] < 0:  # 爆炸在地下
                return 0.0
            
            explosions.append({
                't_explosion': t_explosion,
                'explosion_pos': explosion_pos
            })
        
        # 计算导弹命中时间
        missile_hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
        
        # 时间范围
        earliest_explosion = min(exp['t_explosion'] for exp in explosions)
        latest_effect_end = max(exp['t_explosion'] + CLOUD_EFFECT_TIME for exp in explosions)
        
        sim_start_time = max(0, earliest_explosion - 0.5)
        sim_end_time = min(missile_hit_time, latest_effect_end)
        
        if sim_end_time <= sim_start_time:
            return 0.0
        
        # 快速评估模式：使用较大的时间步长
        if quick_eval:
            time_step = max(time_step, 0.05)
        
        # 时间步进模拟
        time_points = np.arange(sim_start_time, sim_end_time + time_step, time_step)
        blocking_count = 0
        
        for t in time_points:
            missile_pos = missile_position(t)
            
            # 收集有效云团
            active_clouds = []
            for explosion in explosions:
                cloud_center = cloud_center_position(
                    explosion['explosion_pos'], 
                    explosion['t_explosion'], 
                    t
                )
                if cloud_center is not None:
                    active_clouds.append(cloud_center)
            
            # 检查遮蔽
            if is_target_blocked_at_time(missile_pos, target_points, active_clouds):
                blocking_count += 1
        
        return blocking_count * time_step
        
    except Exception:
        return 0.0

# =============================================================================
# 粒子群优化算法
# =============================================================================

class Particle:
    """PSO粒子"""
    def __init__(self, dim: int, bounds: List[Tuple[float, float]]):
        self.dim = dim
        self.bounds = bounds
        
        # 初始化位置和速度
        self.position = np.array([
            random.uniform(bounds[i][0], bounds[i][1]) 
            for i in range(dim)
        ])
        self.velocity = np.array([
            random.uniform(-0.1 * (bounds[i][1] - bounds[i][0]), 
                          0.1 * (bounds[i][1] - bounds[i][0]))
            for i in range(dim)
        ])
        
        self.best_position = self.position.copy()
        self.best_fitness = 0.0
        self.fitness = 0.0
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """更新速度"""
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # 速度限制
        for i in range(self.dim):
            max_vel = 0.2 * (self.bounds[i][1] - self.bounds[i][0])
            self.velocity[i] = np.clip(self.velocity[i], -max_vel, max_vel)
    
    def update_position(self):
        """更新位置"""
        self.position += self.velocity
        
        # 边界处理
        for i in range(self.dim):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = 0
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = 0
    
    def to_solution(self) -> Solution:
        """转换为Solution对象"""
        bombs = []
        for i in range(3):  # 3枚烟幕弹
            idx = i * 4
            bombs.append(BombConfig(
                t_drop=self.position[idx],
                fuse_delay=self.position[idx + 1],
                speed=self.position[idx + 2],
                heading=self.position[idx + 3]
            ))
        return Solution(bombs=bombs, fitness=self.fitness)

# =============================================================================
# 模拟退火算法
# =============================================================================

def simulated_annealing(initial_solution: Solution, target_points: np.ndarray,
                       max_iterations: int = 1000, initial_temp: float = 10.0) -> Solution:
    """模拟退火优化"""
    current = deepcopy(initial_solution)
    current.fitness = evaluate_solution_fitness(current, target_points, quick_eval=False)
    
    best = deepcopy(current)
    
    temperature = initial_temp
    cooling_rate = 0.95
    
    for iteration in range(max_iterations):
        # 生成邻居解
        neighbor = deepcopy(current)
        
        # 随机扰动一个参数
        bomb_idx = random.randint(0, 2)
        param_idx = random.randint(0, 3)
        
        if param_idx == 0:  # t_drop
            neighbor.bombs[bomb_idx].t_drop += random.uniform(-1.0, 1.0)
            neighbor.bombs[bomb_idx].t_drop = max(0.5, min(10.0, neighbor.bombs[bomb_idx].t_drop))
        elif param_idx == 1:  # fuse_delay
            neighbor.bombs[bomb_idx].fuse_delay += random.uniform(-0.5, 0.5)
            neighbor.bombs[bomb_idx].fuse_delay = max(1.0, min(8.0, neighbor.bombs[bomb_idx].fuse_delay))
        elif param_idx == 2:  # speed
            neighbor.bombs[bomb_idx].speed += random.uniform(-10.0, 10.0)
            neighbor.bombs[bomb_idx].speed = max(UAV_SPEED_MIN, min(UAV_SPEED_MAX, neighbor.bombs[bomb_idx].speed))
        else:  # heading
            neighbor.bombs[bomb_idx].heading += random.uniform(-0.2, 0.2)
            neighbor.bombs[bomb_idx].heading = max(-math.pi, min(math.pi, neighbor.bombs[bomb_idx].heading))
        
        # 检查约束
        if not is_solution_valid(neighbor):
            continue
        
        # 评估邻居解
        neighbor.fitness = evaluate_solution_fitness(neighbor, target_points, quick_eval=False)
        
        # 接受准则
        delta = neighbor.fitness - current.fitness
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current = neighbor
            
            if current.fitness > best.fitness:
                best = deepcopy(current)
        
        # 降温
        temperature *= cooling_rate
        
        if iteration % 100 == 0 and iteration > 0:
            print(f"  SA迭代 {iteration}: 当前={current.fitness:.6f}, 最优={best.fitness:.6f}, T={temperature:.4f}")
    
    return best

# =============================================================================
# 约束检查
# =============================================================================

def is_solution_valid(solution: Solution) -> bool:
    """检查解决方案是否满足约束条件"""
    bombs = solution.bombs
    
    # 检查投放间隔
    drop_times = [b.t_drop for b in bombs]
    drop_times.sort()
    
    for i in range(1, len(drop_times)):
        if drop_times[i] - drop_times[i-1] < MIN_DROP_INTERVAL:
            return False
    
    # 检查速度约束
    for bomb in bombs:
        if not (UAV_SPEED_MIN <= bomb.speed <= UAV_SPEED_MAX):
            return False
    
    # 检查物理可行性
    for bomb in bombs:
        explosion_pos = explosion_position(bomb.t_drop, bomb.fuse_delay, bomb.speed, bomb.heading)
        if explosion_pos[2] < 0:  # 爆炸在地下
            return False
    
    return True

# =============================================================================
# 多线程优化引擎
# =============================================================================

class OptimizationEngine:
    """多线程混合优化引擎"""
    
    def __init__(self, n_threads: int = 8, n_particles: int = 50):
        self.n_threads = n_threads
        self.n_particles = n_particles
        self.target_points = generate_cylinder_sampling_points(n_circumference=24, n_height=5)
        
        # 参数边界 [t_drop1, fuse1, speed1, heading1, t_drop2, fuse2, speed2, heading2, t_drop3, fuse3, speed3, heading3]
        self.bounds = []
        for i in range(3):
            self.bounds.extend([
                (0.5, 10.0),           # t_drop
                (1.0, 8.0),            # fuse_delay
                (UAV_SPEED_MIN, UAV_SPEED_MAX),  # speed
                (-math.pi, math.pi)    # heading
            ])
        
        self.best_solution = None
        self.best_fitness = 0.0
        self.evaluation_count = 0
        self.lock = threading.Lock()
    
    def evaluate_particle_batch(self, particles: List[Particle]) -> List[float]:
        """批量评估粒子适应度"""
        results = []
        
        for particle in particles:
            solution = particle.to_solution()
            if is_solution_valid(solution):
                fitness = evaluate_solution_fitness(solution, self.target_points, quick_eval=True)
            else:
                fitness = 0.0
            
            particle.fitness = fitness
            results.append(fitness)
            
            # 更新个体最优
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # 更新全局最优
            with self.lock:
                self.evaluation_count += 1
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = deepcopy(solution)
        
        return results
    
    def run_pso_phase(self, max_iterations: int = 100) -> Solution:
        """运行PSO阶段"""
        print(f"开始PSO阶段，粒子数={self.n_particles}，最大迭代={max_iterations}")
        
        # 初始化粒子群
        particles = [Particle(12, self.bounds) for _ in range(self.n_particles)]
        
        # 初始评估
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            batch_size = max(1, self.n_particles // self.n_threads)
            batches = [particles[i:i+batch_size] for i in range(0, len(particles), batch_size)]
            
            futures = [executor.submit(self.evaluate_particle_batch, batch) for batch in batches]
            for future in as_completed(futures):
                future.result()
        
        # 找到全局最优位置
        global_best_position = max(particles, key=lambda p: p.best_fitness).best_position
        
        # PSO迭代
        for iteration in range(max_iterations):
            # 更新粒子
            for particle in particles:
                particle.update_velocity(global_best_position)
                particle.update_position()
            
            # 重新评估
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self.evaluate_particle_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    future.result()
            
            # 更新全局最优位置
            best_particle = max(particles, key=lambda p: p.best_fitness)
            if best_particle.best_fitness > max(particles, key=lambda p: p.fitness).fitness:
                global_best_position = best_particle.best_position
            
            if iteration % 10 == 0:
                print(f"  PSO迭代 {iteration}: 最优适应度={self.best_fitness:.6f}, 评估次数={self.evaluation_count}")
        
        return deepcopy(self.best_solution)
    
    def run_sa_phase(self, initial_solution: Solution, max_iterations: int = 500) -> Solution:
        """运行SA阶段"""
        print(f"开始SA阶段，初始适应度={initial_solution.fitness:.6f}")
        
        sa_result = simulated_annealing(initial_solution, self.target_points, max_iterations)
        
        with self.lock:
            if sa_result.fitness > self.best_fitness:
                self.best_fitness = sa_result.fitness
                self.best_solution = deepcopy(sa_result)
        
        return sa_result
    
    def optimize(self, pso_iterations: int = 100, sa_iterations: int = 500) -> Solution:
        """运行完整的混合优化"""
        start_time = time.time()
        
        print("="*80)
        print("Q3Solver_v5 混合优化引擎启动")
        print("="*80)
        print(f"线程数: {self.n_threads}")
        print(f"粒子数: {self.n_particles}")
        print(f"目标采样点: {len(self.target_points)}")
        
        # 阶段1: PSO全局搜索
        pso_result = self.run_pso_phase(pso_iterations)
        
        # 阶段2: SA局部精化
        if pso_result and pso_result.fitness > 0:
            sa_result = self.run_sa_phase(pso_result, sa_iterations)
            final_result = sa_result if sa_result.fitness > pso_result.fitness else pso_result
        else:
            final_result = pso_result
        
        end_time = time.time()
        
        print("\n" + "="*80)
        print("优化完成")
        print("="*80)
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"总评估次数: {self.evaluation_count}")
        print(f"最优遮蔽时间: {final_result.fitness:.6f}秒")
        
        # 精确重新评估最优解
        if final_result:
            print("\n正在进行最终精确评估...")
            final_result.fitness = evaluate_solution_fitness(final_result, self.target_points, 
                                                           time_step=0.01, quick_eval=False)
            print(f"精确遮蔽时间: {final_result.fitness:.6f}秒")
        
        return final_result

# =============================================================================
# 结果保存
# =============================================================================

def save_solution_to_excel(solution: Solution, filename: str = "result1.xlsx"):
    """保存解决方案到Excel文件"""
    if not solution or not solution.bombs:
        print("警告：无有效解决方案可保存")
        return
    
    rows = []
    for i, bomb in enumerate(solution.bombs):
        # 计算位置
        drop_pos = uav_position(bomb.t_drop, bomb.speed, bomb.heading)
        explosion_pos = explosion_position(bomb.t_drop, bomb.fuse_delay, bomb.speed, bomb.heading)
        
        # 计算单体遮蔽时间（用于Excel记录）
        single_solution = Solution(bombs=[bomb])
        target_points = generate_cylinder_sampling_points()
        single_time = evaluate_solution_fitness(single_solution, target_points, 
                                               time_step=0.01, quick_eval=False)
        
        rows.append({
            "无人机运动方向": f"{bomb.heading:.6f} rad",
            "无人机运动速度 (m/s)": bomb.speed,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_pos[0], 6),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_pos[1], 6),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_pos[2], 6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(explosion_pos[0], 6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(explosion_pos[1], 6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(explosion_pos[2], 6),
            "有效干扰时长 (s)": round(single_time, 6),
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    
    print(f"\n解决方案已保存到: {filename}")
    print("\n最终方案详情:")
    print(df.to_string(index=False))
    print(f"\n联合遮蔽时间: {solution.fitness:.6f}秒")

# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='Q3Solver_v5 混合优化求解器')
    parser.add_argument('--threads', type=int, default=8, help='线程数')
    parser.add_argument('--particles', type=int, default=50, help='PSO粒子数')
    parser.add_argument('--pso-iter', type=int, default=100, help='PSO迭代次数')
    parser.add_argument('--sa-iter', type=int, default=500, help='SA迭代次数')
    parser.add_argument('--output', type=str, default='result1.xlsx', help='输出文件名')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print("Q3Solver_v5 - 基于混合优化算法的高效求解器")
    print("与Q3Checker_v2.py使用完全一致的物理模型")
    
    # 创建优化引擎
    engine = OptimizationEngine(n_threads=args.threads, n_particles=args.particles)
    
    # 运行优化
    best_solution = engine.optimize(pso_iterations=args.pso_iter, sa_iterations=args.sa_iter)
    
    if best_solution and best_solution.fitness > 0:
        # 保存结果
        save_solution_to_excel(best_solution, args.output)
        
        print(f"\n🎯 优化成功完成！")
        print(f"   最优遮蔽时间: {best_solution.fitness:.6f}秒")
        print(f"   结果文件: {args.output}")
        print(f"\n💡 建议使用以下命令验证结果:")
        print(f"   python Q3Checker_v2.py --excel {args.output} --time-step 0.01")
        
        return best_solution.fitness
    else:
        print("\n❌ 优化失败，未找到有效解决方案")
        return 0.0

if __name__ == "__main__":
    result = main()
    print(f"\n🎯 最终遮蔽时间: {result:.6f}秒")
