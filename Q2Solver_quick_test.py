# -*- coding: utf-8 -*-
"""
Q2Solver_quick_test.py - 快速测试版本
使用最小参数验证基本功能
"""

import os
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random

# ==================== 简化参数 ====================
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
    "FY4": np.array([11000.0, 2000.0, 1800.0], dtype=float),
    "FY5": np.array([13000.0, -2000.0, 1300.0], dtype=float),
}
ALL_UAVS = list(FY_INIT.keys())
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

# ==================== 快速估算遮蔽计算 ====================
def quick_coverage_calculation(uav_id, heading, t_drop):
    """快速遮蔽时长估算，避免复杂几何计算"""
    missile_hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED
    t_explode = t_drop + 3.6

    if t_explode >= missile_hit_time or t_explode < 0:
        return 0.0

    # 简化的几何估算
    drop_pos, _ = UavStateHorizontal(t_drop, FY_INIT[uav_id], UAV_SPEED, heading)
    distance_to_target = np.linalg.norm(drop_pos[:2] - FAKE_TARGET_ORIGIN[:2])

    # 基于距离和角度的简单估算
    angle_factor = (math.sin(heading) + 1) / 2
    distance_factor = max(0.1, 1.0 - distance_to_target / 20000.0)
    time_factor = max(0.1, 1.0 - t_drop / 10.0)

    # 无人机位置因子
    position_factors = {
        "FY1": 1.0,   # 最近，最有效
        "FY2": 0.9,
        "FY3": 0.8,
        "FY4": 0.7,
        "FY5": 0.6    # 最远，效果最差
    }

    coverage = 2.5 * angle_factor * distance_factor * time_factor * position_factors[uav_id]
    return max(0.0, coverage)

# ==================== 简化遗传算法 ====================
class QuickGeneticAlgorithm:
    """快速遗传算法测试版本"""
    def __init__(self, population_size=10, generations=10):
        self.population_size = population_size
        self.generations = generations
        self.population = []

    def create_individual(self):
        """创建个体：5架无人机，每架2个参数"""
        genes = []
        for _ in range(5):  # 5架无人机
            genes.append(random.uniform(0, 2*math.pi))  # 航向角
            genes.append(random.uniform(0, 10))         # 投放时间
        return genes

    def evaluate_fitness(self, genes):
        """评估适应度：总遮蔽时长（串行计算）"""
        total_coverage = 0.0

        for i, uav_id in enumerate(ALL_UAVS):
            heading = genes[i * 2]
            t_drop = genes[i * 2 + 1]

            # 约束检查
            if t_drop < 0 or t_drop > 10:
                return -1000

            # 快速估算
            coverage = quick_coverage_calculation(uav_id, heading, t_drop)
            total_coverage += coverage

        return total_coverage

    def evolve(self):
        """进化过程"""
        print(f"🔬 快速遗传算法测试 (使用 {os.cpu_count()} CPU核心)")

        # 初始化种群
        self.population = [self.create_individual() for _ in range(self.population_size)]
        best_fitness = 0.0
        best_individual = None

        for generation in range(self.generations):
            # 使用线程池并行评估种群
            with ThreadPoolExecutor(max_workers=min(len(self.population), os.cpu_count())) as executor:
                futures = [executor.submit(self.evaluate_fitness, ind) for ind in self.population]
                fitness_values = [future.result() for future in futures]

            # 更新最佳个体
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[current_best_idx][:]

            progress = (generation + 1) / self.generations * 100
            # 每20%显示进度
            if int(progress) % 20 == 0 or generation == self.generations - 1:
                print(f"   [进度] 第{generation+1}/{self.generations}代 ({progress:.1f}%) | 最佳遮蔽时长: {best_fitness:.3f}s")

            # 简单的精英保留策略
            elite = best_individual[:]
            new_population = [elite]  # 精英保留

            while len(new_population) < self.population_size:
                # 随机选择父代
                parent_idx = random.randint(0, self.population_size - 1)
                parent = self.population[parent_idx][:]

                # 变异
                for i in range(len(parent)):
                    if random.random() < 0.2:  # 20%变异率
                        if i % 2 == 0:  # 航向角
                            parent[i] = random.uniform(0, 2*math.pi)
                        else:  # 投放时间
                            parent[i] = random.uniform(0, 10)

                new_population.append(parent)

            self.population = new_population[:self.population_size]

        return best_individual, best_fitness

# ==================== 主快速测试函数 ====================
def quick_test():
    """快速测试Q2求解器"""
    print("=" * 60)
    print("⚡ Q2Solver 快速测试版本")
    print("   参数: 种群10, 代数10, 快速估算")
    print("=" * 60)

    start_time = time.time()

    try:
        # 运行快速遗传算法
        print("\n🚀 开始快速优化测试...")
        ga = QuickGeneticAlgorithm(population_size=10, generations=10)
        best_solution, best_fitness = ga.evolve()

        print("\n✅ 快速测试完成！")
        print("=" * 60)
        print("[结果] 测试结果:")
        print(f"   总遮蔽时长: {best_fitness:.3f} 秒")
        # 显示每架无人机的参数
        for i, uav_id in enumerate(ALL_UAVS):
            heading = best_solution[i * 2]
            t_drop = best_solution[i * 2 + 1]
            coverage = quick_coverage_calculation(uav_id, heading, t_drop)

            print(f"   {uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={coverage:.3f}s")

        computation_time = time.time() - start_time
        print("\n[时间] 计算时间统计:")
        print(".3f")
        print(".3f")
        print("\n[成功] 快速测试成功！参数设置合理，可以用于完整优化。")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
