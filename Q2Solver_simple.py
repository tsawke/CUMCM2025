# -*- coding: utf-8 -*-
"""
Q2Solver_simple.py - Q2求解器简化版本
单线程测试基本功能
"""

import math
import time
import random

# ==================== 简化参数 ====================
ALL_UAVS = ["FY1", "FY2", "FY3"]

def simple_coverage_calculation(uav_id, heading, t_drop):
    """极简化的遮蔽时长计算"""
    # 基于距离和角度的简单估算
    base_coverage = 2.0  # 基础遮蔽时长

    # 角度因子 (0-1之间)
    angle_factor = (math.sin(heading) + 1) / 2

    # 时间因子 (越早投放效果越好)
    time_factor = max(0.1, 1.0 - t_drop / 10.0)

    # 无人机位置因子 (FY1最近，效果最好)
    position_factor = {"FY1": 1.0, "FY2": 0.8, "FY3": 0.6}[uav_id]

    coverage = base_coverage * angle_factor * time_factor * position_factor
    return coverage

# ==================== 简化遗传算法 ====================
class SimpleGA:
    def __init__(self, population_size=10, generations=10):
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
        """评估适应度"""
        total_coverage = 0.0
        for i, uav_id in enumerate(ALL_UAVS):
            heading = genes[i * 2]
            t_drop = genes[i * 2 + 1]
            coverage = simple_coverage_calculation(uav_id, heading, t_drop)
            total_coverage += coverage
        return total_coverage

    def evolve(self):
        """进化过程"""
        print("开始简化遗传算法优化")

        # 初始化种群
        self.population = [self.create_individual() for _ in range(self.population_size)]

        best_fitness = 0.0
        best_individual = None

        for generation in range(self.generations):
            # 评估种群
            fitness_values = [self.evaluate_fitness(ind) for ind in self.population]

            # 更新最佳个体
            current_best_idx = max(range(len(fitness_values)), key=lambda i: fitness_values[i])
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[current_best_idx][:]

            progress = (generation + 1) / self.generations * 100
            print(f"   📊 第{generation+1}/{self.generations}代 ({progress:.1f}%) | 最佳遮蔽时长: {best_fitness:.3f}s")

            # 简单的选择和繁殖
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

# ==================== 主函数 ====================
def test_simple_q2():
    """测试简化版Q2求解器"""
    print("=" * 50)
    print("Q2Solver 简化测试版本")
    print("=" * 50)

    start_time = time.time()

    # 运行遗传算法
    ga = SimpleGA(population_size=10, generations=20)
    best_solution, best_fitness = ga.evolve()

    print("\n测试结果:")
    print(f"总遮蔽时长: {best_fitness:.3f} 秒")

    # 显示每架无人机的参数
    for i, uav_id in enumerate(ALL_UAVS):
        heading = best_solution[i * 2]
        t_drop = best_solution[i * 2 + 1]
        coverage = simple_coverage_calculation(uav_id, heading, t_drop)

        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={coverage:.3f}s")

    computation_time = time.time() - start_time
    print(f"计算时间: {computation_time:.3f}秒")
    print("\n测试完成！基本优化功能正常工作。")
    print("现在可以扩展到多线程和更精确的计算。")

if __name__ == "__main__":
    test_simple_q2()
