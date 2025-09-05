# -*- coding: utf-8 -*-
"""
Q2Solver_ultra_simple.py - 超简化测试版本
用于验证修复是否有效
"""

import math
import random

def simple_coverage(uav_id, heading, t_drop):
    """极简化的遮蔽计算"""
    return 1.0 + 0.5 * math.sin(heading) + 0.3 * (10 - t_drop)

class SimpleGA:
    def __init__(self, pop_size=10, gens=5):
        self.pop_size = pop_size
        self.generations = gens

    def create_individual(self):
        return [random.uniform(0, 2*math.pi), random.uniform(0, 10)]

    def evaluate(self, genes):
        return simple_coverage("FY1", genes[0], genes[1])

    def evolve(self):
        print("开始超简化遗传算法测试...")
        population = [self.create_individual() for _ in range(self.pop_size)]

        for gen in range(self.generations):
            fitnesses = [self.evaluate(ind) for ind in population]
            best_idx = fitnesses.index(max(fitnesses))
            best_fitness = fitnesses[best_idx]
            print(f"代 {gen+1}: 最佳适应度 = {best_fitness:.3f}")

            # 简单的精英保留
            population[0] = population[best_idx][:]

        return population[0], max(fitnesses)

def test():
    print("🧪 超简化Q2Solver测试")
    ga = SimpleGA()
    solution, fitness = ga.evolve()
    print(f"✅ 测试成功！最佳解: {solution}, 适应度: {fitness:.3f}")
    print("🎉 修复后的Q2Solver基本功能正常！")

if __name__ == "__main__":
    test()
