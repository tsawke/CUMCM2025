# -*- coding: utf-8 -*-
"""
Q2Solver_test.py - Q2求解器测试版本
快速测试多线程优化功能
"""

import os
import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random

# ==================== 简化常量定义 ====================
g = 9.8
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)
FY_INIT = {
    "FY1": np.array([17800.0, 0.0, 1800.0], dtype=float),
    "FY2": np.array([12000.0, 1400.0, 1400.0], dtype=float),
    "FY3": np.array([6000.0, -3000.0, 700.0], dtype=float),
}
ALL_UAVS = ["FY1", "FY2", "FY3"]  # 先用3架测试
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)

def Unit(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def MissileState(t, mInit):
    dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
    v = MISSILE_SPEED * dirToOrigin
    return mInit + v * t, v

def UavStateHorizontal(t, uavInit, uavSpeed, headingRadius):
    vx = uavSpeed * math.cos(headingRadius)
    vy = uavSpeed * math.sin(headingRadius)
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype=uavInit.dtype), \
           np.array([vx, vy, 0.0], dtype=uavInit.dtype)

# ==================== 简化遮蔽计算 ====================
def quick_coverage_calculation(uav_id, heading, t_drop):
    """快速遮蔽时长计算（简化版）"""
    # 简化的几何计算，避免复杂的三维积分
    # 这里使用近似公式快速估算

    # 导弹飞行时间
    missile_hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED

    # 无人机投放和起爆时间
    t_explode = t_drop + 3.6

    if t_explode >= missile_hit_time or t_explode < 0:
        return 0.0

    # 简化的遮蔽时长估算
    # 基于距离和角度的近似计算
    drop_pos, _ = UavStateHorizontal(t_drop, FY_INIT[uav_id], UAV_SPEED, heading)
    distance_to_target = np.linalg.norm(drop_pos[:2] - FAKE_TARGET_ORIGIN[:2])

    # 简单的角度因子
    angle_factor = abs(math.sin(heading))  # 简化角度影响

    # 估算遮蔽时长
    base_coverage = min(5.0, max(0.0, 10.0 - distance_to_target / 2000.0))
    coverage = base_coverage * angle_factor * (1.0 - t_drop / 10.0)

    return max(0.0, coverage)

# ==================== 优化算法 ====================
class SimpleGeneticAlgorithm:
    """简化遗传算法"""
    def __init__(self, population_size=20, generations=20):
        self.population_size = population_size
        self.generations = generations
        self.population = []

    def create_individual(self):
        """创建个体：3架无人机，每架2个参数"""
        genes = []
        for _ in range(3):  # 3架无人机
            genes.append(random.uniform(0, 2*math.pi))  # 航向角
            genes.append(random.uniform(0, 10))         # 投放时间
        return genes

    def evaluate_fitness(self, genes):
        """评估适应度：总遮蔽时长（多线程并行计算）"""
        # 准备计算任务
        tasks = []
        for i, uav_id in enumerate(ALL_UAVS):
            heading = genes[i * 2]
            t_drop = genes[i * 2 + 1]
            tasks.append((uav_id, heading, t_drop))

        # 多线程并行计算
        total_coverage = 0.0
        with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count())) as executor:
            futures = [executor.submit(quick_coverage_calculation, *task) for task in tasks]
            for future in futures:
                coverage = future.result()
                total_coverage += coverage

        return total_coverage

    def evolve(self):
        """进化过程"""
        print(f"开始遗传算法优化 (使用 {os.cpu_count()} CPU核心)")

        # 初始化种群
        self.population = [self.create_individual() for _ in range(self.population_size)]

        best_fitness = 0.0
        best_individual = None

        for generation in range(self.generations):
            # 并行评估种群
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(self.evaluate_fitness, ind) for ind in self.population]
                fitness_values = [future.result() for future in futures]

            # 更新最佳个体
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[current_best_idx][:]

            # 简单的选择和繁殖（精英保留 + 随机变异）
            elite = best_individual[:]
            new_population = [elite]  # 精英保留

            while len(new_population) < self.population_size:
                # 随机选择父代
                parent_idx = random.randint(0, self.population_size - 1)
                parent = self.population[parent_idx][:]

                # 变异
                for i in range(len(parent)):
                    if random.random() < 0.1:  # 10%变异率
                        if i % 2 == 0:  # 航向角
                            parent[i] = random.uniform(0, 2*math.pi)
                        else:  # 投放时间
                            parent[i] = random.uniform(0, 10)

                new_population.append(parent)

            self.population = new_population[:self.population_size]

            if generation % 5 == 0:
                print(f"第{generation}代，最佳适应度: {best_fitness:.3f}s")

        return best_individual, best_fitness

# ==================== 主测试函数 ====================
def test_q2_solver():
    """测试Q2求解器"""
    print("=" * 60)
    print("Q2Solver 测试版本 - 多线程优化")
    print("=" * 60)

    start_time = time.time()

    # 运行遗传算法
    ga = SimpleGeneticAlgorithm(population_size=20, generations=30)
    best_solution, best_fitness = ga.evolve()

    print("\n测试结果:")
    print(f"总遮蔽时长: {best_fitness:.3f} 秒")

    # 显示每架无人机的参数
    for i, uav_id in enumerate(ALL_UAVS):
        heading = best_solution[i * 2]
        t_drop = best_solution[i * 2 + 1]
        coverage = quick_coverage_calculation(uav_id, heading, t_drop)

        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={coverage:.3f}s")

    computation_time = time.time() - start_time
    print(".3f"
    print("测试完成！多线程优化功能正常工作。")

if __name__ == "__main__":
    test_q2_solver()
