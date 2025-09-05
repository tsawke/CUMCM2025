# -*- coding: utf-8 -*-
"""
高级优化算法解决无人机烟幕干扰问题
使用遗传算法和粒子群优化算法
"""

import math
import json
import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict
import copy

# 导入基础计算函数
from cumcm2025_simplified_solution import (
    coverage_time_for_plan, uav_xy_pos, explosion_point, 
    FY_INIT, M_INIT, TRUE_TARGET_XY, TARGET_Z0, TARGET_Z1
)

# =====================
# 一、遗传算法实现
# =====================

class Individual:
    """个体类，表示一个解决方案"""
    def __init__(self, genes=None):
        if genes is None:
            # 基因：[uav1_heading, uav1_t_drop, uav2_heading, uav2_t_drop, ...]
            self.genes = []
            for i in range(5):  # 5架无人机
                self.genes.append(random.uniform(0, 2*math.pi))  # 航向角
                self.genes.append(random.uniform(0, 10))         # 投放时机
        else:
            self.genes = genes
        self.fitness = 0.0
    
    def evaluate_fitness(self, problem_type="problem2"):
        """评估个体适应度"""
        if problem_type == "problem2":
            return self._evaluate_problem2()
        elif problem_type == "problem3":
            return self._evaluate_problem3()
        else:
            return 0.0
    
    def _evaluate_problem2(self):
        """问题二的适应度评估"""
        total_coverage = 0.0
        uav_ids = ["FY1", "FY2", "FY3"]
        v = 120.0
        
        for i, uav_id in enumerate(uav_ids):
            if i * 2 + 1 >= len(self.genes):
                break
                
            heading = self.genes[i * 2]
            t_drop = self.genes[i * 2 + 1]
            
            # 约束检查
            if t_drop < 0 or t_drop > 10:
                return -1000
            
            t_explode = t_drop + 3.6
            
            coverage = coverage_time_for_plan(
                uav_id=uav_id,
                missile_id="M1",
                v=v,
                heading_rad=heading,
                t_drop=t_drop,
                t_explode=t_explode,
                dt=0.1,
                z_samples=3
            )
            
            total_coverage += coverage
        
        return total_coverage
    
    def _evaluate_problem3(self):
        """问题三的适应度评估"""
        total_coverage = 0.0
        uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
        missile_ids = ["M1", "M2", "M3"]
        v = 120.0
        
        for i, uav_id in enumerate(uav_ids):
            if i * 2 + 1 >= len(self.genes):
                break
                
            heading = self.genes[i * 2]
            t_drop = self.genes[i * 2 + 1]
            
            # 约束检查
            if t_drop < 0 or t_drop > 15:
                return -10000
            
            t_explode = t_drop + 3.6
            
            for missile_id in missile_ids:
                coverage = coverage_time_for_plan(
                    uav_id=uav_id,
                    missile_id=missile_id,
                    v=v,
                    heading_rad=heading,
                    t_drop=t_drop,
                    t_explode=t_explode,
                    dt=0.1,
                    z_samples=3
                )
                total_coverage += coverage
        
        return total_coverage

class GeneticAlgorithm:
    """遗传算法类"""
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_individual = None
        self.best_fitness_history = []
    
    def initialize_population(self):
        """初始化种群"""
        self.population = [Individual() for _ in range(self.population_size)]
    
    def evaluate_population(self, problem_type="problem2"):
        """评估种群适应度"""
        for individual in self.population:
            individual.fitness = individual.evaluate_fitness(problem_type)
        
        # 更新最佳个体
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
        
        self.best_fitness_history.append(self.best_individual.fitness)
    
    def selection(self):
        """选择操作（轮盘赌选择）"""
        total_fitness = sum(max(0, ind.fitness) for ind in self.population)
        if total_fitness == 0:
            return random.choices(self.population, k=2)
        
        weights = [max(0, ind.fitness) for ind in self.population]
        return random.choices(self.population, weights=weights, k=2)
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1_genes = []
        child2_genes = []
        
        for i in range(len(parent1.genes)):
            if random.random() < 0.5:
                child1_genes.append(parent1.genes[i])
                child2_genes.append(parent2.genes[i])
            else:
                child1_genes.append(parent2.genes[i])
                child2_genes.append(parent1.genes[i])
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def mutation(self, individual):
        """变异操作"""
        for i in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                if i % 2 == 0:  # 航向角
                    individual.genes[i] = random.uniform(0, 2*math.pi)
                else:  # 投放时机
                    individual.genes[i] = random.uniform(0, 10)
    
    def evolve(self, problem_type="problem2"):
        """进化过程"""
        self.initialize_population()
        
        for generation in range(self.generations):
            self.evaluate_population(problem_type)
            
            new_population = []
            
            # 保留最佳个体
            new_population.append(copy.deepcopy(self.best_individual))
            
            # 生成新个体
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection()
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutation(child1)
                self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            
            if generation % 20 == 0:
                print(f"第{generation}代，最佳适应度: {self.best_individual.fitness:.3f}")
        
        self.evaluate_population(problem_type)
        return self.best_individual

# =====================
# 二、粒子群优化算法
# =====================

class Particle:
    """粒子类"""
    def __init__(self, position=None, velocity=None):
        if position is None:
            # 位置：[uav1_heading, uav1_t_drop, uav2_heading, uav2_t_drop, ...]
            self.position = []
            for i in range(10):  # 5架无人机 * 2个参数
                if i % 2 == 0:  # 航向角
                    self.position.append(random.uniform(0, 2*math.pi))
                else:  # 投放时机
                    self.position.append(random.uniform(0, 10))
        else:
            self.position = position
        
        if velocity is None:
            self.velocity = [random.uniform(-1, 1) for _ in range(len(self.position))]
        else:
            self.velocity = velocity
        
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        self.fitness = 0.0
    
    def evaluate_fitness(self, problem_type="problem2"):
        """评估粒子适应度"""
        if problem_type == "problem2":
            return self._evaluate_problem2()
        elif problem_type == "problem3":
            return self._evaluate_problem3()
        else:
            return 0.0
    
    def _evaluate_problem2(self):
        """问题二的适应度评估"""
        total_coverage = 0.0
        uav_ids = ["FY1", "FY2", "FY3"]
        v = 120.0
        
        for i, uav_id in enumerate(uav_ids):
            if i * 2 + 1 >= len(self.position):
                break
                
            heading = self.position[i * 2]
            t_drop = self.position[i * 2 + 1]
            
            # 约束检查
            if t_drop < 0 or t_drop > 10:
                return -1000
            
            t_explode = t_drop + 3.6
            
            coverage = coverage_time_for_plan(
                uav_id=uav_id,
                missile_id="M1",
                v=v,
                heading_rad=heading,
                t_drop=t_drop,
                t_explode=t_explode,
                dt=0.1,
                z_samples=3
            )
            
            total_coverage += coverage
        
        return total_coverage
    
    def _evaluate_problem3(self):
        """问题三的适应度评估"""
        total_coverage = 0.0
        uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
        missile_ids = ["M1", "M2", "M3"]
        v = 120.0
        
        for i, uav_id in enumerate(uav_ids):
            if i * 2 + 1 >= len(self.position):
                break
                
            heading = self.position[i * 2]
            t_drop = self.position[i * 2 + 1]
            
            # 约束检查
            if t_drop < 0 or t_drop > 15:
                return -10000
            
            t_explode = t_drop + 3.6
            
            for missile_id in missile_ids:
                coverage = coverage_time_for_plan(
                    uav_id=uav_id,
                    missile_id=missile_id,
                    v=v,
                    heading_rad=heading,
                    t_drop=t_drop,
                    t_explode=t_explode,
                    dt=0.1,
                    z_samples=3
                )
                total_coverage += coverage
        
        return total_coverage
    
    def update_velocity(self, global_best_position, w=0.9, c1=2.0, c2=2.0):
        """更新粒子速度"""
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
    
    def update_position(self):
        """更新粒子位置"""
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            
            # 边界处理
            if i % 2 == 0:  # 航向角
                self.position[i] = self.position[i] % (2 * math.pi)
            else:  # 投放时机
                self.position[i] = max(0, min(10, self.position[i]))
    
    def update_best(self):
        """更新个体最佳位置"""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class ParticleSwarmOptimization:
    """粒子群优化算法类"""
    def __init__(self, swarm_size=30, max_iterations=100, w=0.9, c1=2.0, c2=2.0):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
    
    def initialize_swarm(self):
        """初始化粒子群"""
        self.swarm = [Particle() for _ in range(self.swarm_size)]
    
    def evaluate_swarm(self, problem_type="problem2"):
        """评估粒子群适应度"""
        for particle in self.swarm:
            particle.fitness = particle.evaluate_fitness(problem_type)
            particle.update_best()
            
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        self.fitness_history.append(self.global_best_fitness)
    
    def optimize(self, problem_type="problem2"):
        """执行优化"""
        self.initialize_swarm()
        
        for iteration in range(self.max_iterations):
            self.evaluate_swarm(problem_type)
            
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            if iteration % 20 == 0:
                print(f"第{iteration}次迭代，最佳适应度: {self.global_best_fitness:.3f}")
        
        return self.global_best_position, self.global_best_fitness

# =====================
# 三、高级优化求解函数
# =====================

def solve_problem2_advanced():
    """使用高级优化算法解决问题二"""
    print("\n" + "="*60)
    print("问题二：使用遗传算法优化")
    print("="*60)
    
    # 使用遗传算法
    ga = GeneticAlgorithm(population_size=50, generations=100)
    best_individual = ga.evolve("problem2")
    
    print(f"遗传算法最佳适应度: {best_individual.fitness:.3f}")
    
    # 使用粒子群优化
    print("\n使用粒子群优化算法...")
    pso = ParticleSwarmOptimization(swarm_size=30, max_iterations=100)
    best_position, best_fitness = pso.optimize("problem2")
    
    print(f"粒子群优化最佳适应度: {best_fitness:.3f}")
    
    # 选择更好的结果
    if best_fitness > best_individual.fitness:
        print("粒子群优化结果更好，使用粒子群优化结果")
        genes = best_position
        fitness = best_fitness
    else:
        print("遗传算法结果更好，使用遗传算法结果")
        genes = best_individual.genes
        fitness = best_individual.fitness
    
    # 生成结果数据
    results_data = []
    uav_ids = ["FY1", "FY2", "FY3"]
    v = 120.0
    
    for i, uav_id in enumerate(uav_ids):
        if i * 2 + 1 >= len(genes):
            break
            
        heading = genes[i * 2]
        t_drop = genes[i * 2 + 1]
        t_explode = t_drop + 3.6
        
        # 计算位置
        drop_pos = uav_xy_pos(uav_id, v, heading, t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
        drop_z = FY_INIT[uav_id][2]
        
        # 计算遮蔽时长
        coverage = coverage_time_for_plan(
            uav_id=uav_id,
            missile_id="M1",
            v=v,
            heading_rad=heading,
            t_drop=t_drop,
            t_explode=t_explode,
            dt=0.05,
            z_samples=5
        )
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向": math.degrees(heading) % 360,
            "无人机运动速度 (m/s)": v,
            "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标 (m)": drop_z,
            "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
            "有效干扰时长 (s)": coverage
        }
        
        results_data.append(result_data)
        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={coverage:.3f}s")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    df.to_csv("result2_advanced.csv", index=False, encoding='utf-8-sig')
    print(f"结果已保存到 result2_advanced.csv")
    
    return results_data

def solve_problem3_advanced():
    """使用高级优化算法解决问题三"""
    print("\n" + "="*60)
    print("问题三：使用遗传算法优化")
    print("="*60)
    
    # 使用遗传算法
    ga = GeneticAlgorithm(population_size=50, generations=100)
    best_individual = ga.evolve("problem3")
    
    print(f"遗传算法最佳适应度: {best_individual.fitness:.3f}")
    
    # 使用粒子群优化
    print("\n使用粒子群优化算法...")
    pso = ParticleSwarmOptimization(swarm_size=30, max_iterations=100)
    best_position, best_fitness = pso.optimize("problem3")
    
    print(f"粒子群优化最佳适应度: {best_fitness:.3f}")
    
    # 选择更好的结果
    if best_fitness > best_individual.fitness:
        print("粒子群优化结果更好，使用粒子群优化结果")
        genes = best_position
        fitness = best_fitness
    else:
        print("遗传算法结果更好，使用遗传算法结果")
        genes = best_individual.genes
        fitness = best_individual.fitness
    
    # 生成结果数据
    results_data = []
    uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    missile_ids = ["M1", "M2", "M3"]
    v = 120.0
    
    for i, uav_id in enumerate(uav_ids):
        if i * 2 + 1 >= len(genes):
            break
            
        heading = genes[i * 2]
        t_drop = genes[i * 2 + 1]
        t_explode = t_drop + 3.6
        
        # 计算位置
        drop_pos = uav_xy_pos(uav_id, v, heading, t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
        drop_z = FY_INIT[uav_id][2]
        
        # 计算对每枚导弹的遮蔽时长
        total_coverage = 0.0
        missile_coverage = {}
        
        for missile_id in missile_ids:
            coverage = coverage_time_for_plan(
                uav_id=uav_id,
                missile_id=missile_id,
                v=v,
                heading_rad=heading,
                t_drop=t_drop,
                t_explode=t_explode,
                dt=0.05,
                z_samples=5
            )
            missile_coverage[missile_id] = coverage
            total_coverage += coverage
        
        # 找到主要干扰的导弹
        main_missile = max(missile_coverage, key=missile_coverage.get)
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向": math.degrees(heading) % 360,
            "无人机运动速度 (m/s)": v,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标 (m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标 (m)": drop_z,
            "烟幕干扰弹起爆点的x坐标 (m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标 (m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标 (m)": explosion_pos[2],
            "有效干扰时长 (s)": total_coverage,
            "干扰的导弹编号": main_missile
        }
        
        results_data.append(result_data)
        print(f"{uav_id}: 航向={math.degrees(heading):.1f}°, 投放={t_drop:.2f}s, "
              f"总遮蔽={total_coverage:.3f}s, 主要干扰={main_missile}")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    df.to_csv("result3_advanced.csv", index=False, encoding='utf-8-sig')
    print(f"结果已保存到 result3_advanced.csv")
    
    return results_data

# =====================
# 四、主程序
# =====================

def main():
    """主程序"""
    print("高级优化算法解决无人机烟幕干扰问题")
    print("="*60)
    
    try:
        # 问题二高级优化
        results2 = solve_problem2_advanced()
        
        # 问题三高级优化
        results3 = solve_problem3_advanced()
        
        print("\n" + "="*60)
        print("高级优化完成！")
        print("="*60)
        print("生成的文件：")
        print("- result2_advanced.csv: 问题二高级优化结果")
        print("- result3_advanced.csv: 问题三高级优化结果")
        
    except Exception as e:
        print(f"优化过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
