# -*- coding: utf-8 -*-
"""
Q2Solver.py - CUMCM2025第二问求解器
多无人机协同干扰优化，最大化对M1的总遮蔽效果

优化目标：max Σ(i=1 to 5) T_cover_i 对于导弹M1
变量：每架无人机的投放时间t_drop和航向角heading
约束：0 ≤ t_drop_i ≤ 10s，0 ≤ heading_i ≤ 2π
"""

import os
import math
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
import multiprocessing as mp
from typing import List, Tuple, Dict

# ==================== 常量定义 ====================
g = 9.8
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
SMOG_R = 10.0
SMOG_SINK_SPEED = 3.0
SMOG_EFFECT_TIME = 20.0

CYLINDER_BASE_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)
CYLINDER_R = 7.0
CYLINDER_H = 10.0

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

DROP_TIME = 1.5
FUSE_DELAY_TIME = 3.6
EXPLODE_TIME = DROP_TIME + FUSE_DELAY_TIME
EPS = 1e-12

def Unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, mInit):
    dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
    v = MISSILE_SPEED * dirToOrigin
    return mInit + v * t, v

def UavStateHorizontal(t, uavInit, uavSpeed, headingRadius):
    vx = uavSpeed * math.cos(headingRadius)
    vy = uavSpeed * math.sin(headingRadius)
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype=uavInit.dtype), \
           np.array([vx, vy, 0.0], dtype=uavInit.dtype)

def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
    b = CYLINDER_BASE_CENTER.astype(dtype)
    r, h = dtype(CYLINDER_R), dtype(CYLINDER_H)

    phis = np.linspace(0.0, 2.0 * math.pi, nPhi, endpoint=False, dtype=dtype)
    c, s = np.cos(phis), np.sin(phis)
    ring = np.stack([r * c, r * s, np.zeros_like(c)], axis=1).astype(dtype)

    pts = [b + ring, b + np.array([0.0, 0.0, h], dtype=dtype) + ring]
    if nZ >= 2:
        for z in np.linspace(0.0, h, nZ, dtype=dtype):
            pts.append(b + np.array([0.0, 0.0, z], dtype=dtype) + ring)

    p = np.vstack(pts).astype(dtype)
    return p

def ExplosionPoint(headingRadius, tDrop, fuseDelayTime, uav_id="FY1", dtype=np.float64):
    dropPos, uavV = UavStateHorizontal(tDrop, FY_INIT[uav_id].astype(dtype), dtype(UAV_SPEED), headingRadius)
    explXy = dropPos[:2] + uavV[:2] * dtype(fuseDelayTime)
    explZ = dropPos[2] - dtype(0.5) * dtype(g) * (dtype(fuseDelayTime) ** 2)
    return np.array([explXy[0], explXy[1], explZ], dtype=dtype)

def ConeAllPointsIn(m, c, p, rCloud=SMOG_R, margin=EPS, block=8192):
    v = c - m
    l = np.linalg.norm(v)
    if l <= EPS or rCloud >= l:
        return True
    cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud / l) ** 2))

    for i in range(0, len(p), block):
        w = p[i: i + block] - m
        wn = np.linalg.norm(w, axis=1) + EPS
        lhs = w @ v
        rhs = wn * l * cosAlpha
        if not np.all(lhs + margin >= rhs):
            return False
    return True

def EvalChunk(args):
    idx0, tChunk, explPos, tExpl, pts, margin, block = args
    out = np.zeros_like(tChunk, dtype=bool)
    for i, t in enumerate(tChunk):
        m, _ = MissileState(float(t), M1_INIT)
        c = np.array([explPos[0], explPos[1], explPos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - tExpl)], dtype=explPos.dtype)
        out[i] = ConeAllPointsIn(m, c, pts, rCloud=SMOG_R, margin=margin, block=block)
    return idx0, out

def CalSingleUavOcclusion(headingRadius, tDrop, fuseDelayTime, uav_id="FY1", dt=0.0005, nphi=960, nz=13,
                         backend="process", workers=None, chunk=800, fp32=False, margin=EPS, block=8192):
    if backend == "process":
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    dtype = np.float32 if fp32 else np.float64
    explPos = ExplosionPoint(headingRadius, tDrop, fuseDelayTime, uav_id, dtype=dtype)
    if explPos[2] <= 0:
        return 0.0

    tExpl = tDrop + fuseDelayTime
    t0, t1 = tExpl, min(tExpl + SMOG_EFFECT_TIME, np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    tGrid = np.arange(t0, t1 + EPS, dt, dtype=dtype)

    pts = PreCalCylinderPoints(nphi, nz, dtype=dtype)

    chunks = [(i, tGrid[i: i + chunk]) for i in range(0, len(tGrid), chunk)]
    maskTotal = np.zeros_like(tGrid, dtype=bool)

    if backend == "thread":
        from concurrent.futures import ThreadPoolExecutor
        poolCls = ThreadPoolExecutor
    else:
        poolCls = ProcessPoolExecutor

    with poolCls(max_workers=workers) as poolCls:
        futs = {poolCls.submit(EvalChunk, (idx, c, explPos, tExpl, pts, margin, block)): idx for idx, c in chunks}
        for fut in futs:
            idx = futs[fut]
            _, m = fut.result()
            l = len(m)
            maskTotal[idx: idx + l] = m

    seconds = float(np.count_nonzero(maskTotal) * dt)
    return seconds

# ==================== 优化算法 ====================

class UAVPlan:
    """无人机投放计划"""
    def __init__(self, uav_id, heading, t_drop, fuse_delay=3.6):
        self.uav_id = uav_id
        self.heading = heading
        self.t_drop = t_drop
        self.fuse_delay = fuse_delay
        self.t_explode = t_drop + fuse_delay
        self.coverage_time = 0.0

    def calculate_coverage(self):
        """计算该计划的遮蔽时长（优化版本）"""
        self.coverage_time = CalSingleUavOcclusion(
            self.heading, self.t_drop, self.fuse_delay, self.uav_id,
            dt=0.001, nphi=480, nz=7, backend="process", workers=os.cpu_count()
        )
        return self.coverage_time

class GeneticAlgorithmQ2:
    """遗传算法求解第二问"""
    def __init__(self, population_size=20, generations=30, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_individual = None
        self.best_fitness_history = []

    def create_individual(self):
        """创建个体：5架无人机，每架2个参数（航向角，投放时间）"""
        genes = []
        for _ in range(5):  # 5架无人机
            genes.append(random.uniform(0, 2*math.pi))  # 航向角
            genes.append(random.uniform(0, 10))         # 投放时间
        return genes

    def evaluate_fitness(self, genes):
        """评估适应度：总遮蔽时长（串行计算，避免多进程嵌套）"""
        total_coverage = 0.0

        for i, uav_id in enumerate(ALL_UAVS):
            heading = genes[i * 2]
            t_drop = genes[i * 2 + 1]

            # 约束检查
            if t_drop < 0 or t_drop > 10:
                return -1000

            # 串行计算每个无人机的遮蔽时长
            plan = UAVPlan(uav_id, heading, t_drop)
            coverage = plan.calculate_coverage()
            total_coverage += coverage

        return total_coverage

    def selection(self, fitness_values=None):
        """轮盘赌选择（使用已计算的适应度值）"""
        if fitness_values is None:
            fitness_values = [self.evaluate_fitness(ind) for ind in self.population]

        total_fitness = sum(max(0, f) for f in fitness_values)
        if total_fitness == 0:
            return random.choices(self.population, k=2)

        weights = [max(0, f) for f in fitness_values]
        return random.choices(self.population, weights=weights, k=2)

    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

        return child1, child2

    def mutation(self, individual):
        """变异操作"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                if i % 2 == 0:  # 航向角
                    individual[i] = random.uniform(0, 2*math.pi)
                else:  # 投放时间
                    individual[i] = random.uniform(0, 10)

    def evolve(self):
        """进化过程"""
        # 初始化种群
        self.population = [self.create_individual() for _ in range(self.population_size)]

        for generation in range(self.generations):
            # 使用线程池并行评估种群（避免多进程嵌套）
            with ThreadPoolExecutor(max_workers=min(len(self.population), os.cpu_count())) as executor:
                futures = [executor.submit(self.evaluate_fitness, ind) for ind in self.population]
                fitness_values = [future.result() for future in futures]

            # 更新最佳个体
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]

            if self.best_individual is None or current_best_fitness > self.evaluate_fitness(self.best_individual):
                self.best_individual = self.population[current_best_idx][:]

            self.best_fitness_history.append(current_best_fitness)

            # 新种群
            new_population = [self.best_individual[:]]  # 精英保留

            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(fitness_values)  # 传递已计算的fitness_values
                child1, child2 = self.crossover(parent1, parent2)
                self.mutation(child1)
                self.mutation(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

            progress = (generation + 1) / self.generations * 100
            # 每5%显示进度
            if int(progress) % 5 == 0 or generation == self.generations - 1:
                print(f"   [进度] 第{generation+1}/{self.generations}代 ({progress:.1f}%) | 最佳遮蔽时长: {current_best_fitness:.3f}s")

        return self.best_individual, self.evaluate_fitness(self.best_individual)

class ParticleSwarmQ2:
    """粒子群优化求解第二问"""
    def __init__(self, swarm_size=15, max_iterations=30, w=0.9, c1=2.0, c2=2.0):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []

    def evaluate_fitness(self, genes):
        """评估适应度：总遮蔽时长（串行计算）"""
        total_coverage = 0.0

        for i, uav_id in enumerate(ALL_UAVS):
            heading = genes[i * 2]
            t_drop = genes[i * 2 + 1]

            # 约束检查
            if t_drop < 0 or t_drop > 10:
                return -1000

            # 串行计算每个无人机的遮蔽时长
            plan = UAVPlan(uav_id, heading, t_drop)
            coverage = plan.calculate_coverage()
            total_coverage += coverage

        return total_coverage

    def create_particle(self):
        """创建粒子：10个维度（5架无人机×2参数）"""
        position = []
        velocity = []
        for _ in range(5):  # 5架无人机
            position.append(random.uniform(0, 2*math.pi))  # 航向角
            position.append(random.uniform(0, 10))         # 投放时间
            velocity.append(random.uniform(-1, 1))
            velocity.append(random.uniform(-1, 1))
        return {'position': position, 'velocity': velocity,
                'best_position': position[:], 'best_fitness': float('-inf')}

    def evaluate_particle(self, particle):
        """评估粒子适应度"""
        return self.evaluate_fitness(particle['position'])

    def update_velocity(self, particle):
        """更新粒子速度"""
        for i in range(len(particle['velocity'])):
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (particle['best_position'][i] - particle['position'][i])
            social = self.c2 * r2 * (self.global_best_position[i] - particle['position'][i])
            particle['velocity'][i] = self.w * particle['velocity'][i] + cognitive + social

    def update_position(self, particle):
        """更新粒子位置"""
        for i in range(len(particle['position'])):
            particle['position'][i] += particle['velocity'][i]

            # 边界处理
            if i % 2 == 0:  # 航向角
                particle['position'][i] = particle['position'][i] % (2 * math.pi)
            else:  # 投放时间
                particle['position'][i] = max(0, min(10, particle['position'][i]))

    def optimize(self):
        """执行优化"""
        # 初始化粒子群
        self.swarm = [self.create_particle() for _ in range(self.swarm_size)]

        for iteration in range(self.max_iterations):
            # 使用线程池并行评估粒子群（避免多进程嵌套）
            with ThreadPoolExecutor(max_workers=min(len(self.swarm), os.cpu_count())) as executor:
                futures = [executor.submit(self.evaluate_particle, particle) for particle in self.swarm]
                fitness_values = [future.result() for future in futures]

                for i, (particle, fitness) in enumerate(zip(self.swarm, fitness_values)):
                    # 更新个体最佳
                    if fitness > particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'][:]

                    # 更新全局最佳
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle['position'][:]

            # 更新所有粒子
            for particle in self.swarm:
                self.update_velocity(particle)
                self.update_position(particle)

            self.fitness_history.append(self.global_best_fitness)

            progress = (iteration + 1) / self.max_iterations * 100
            # 每5%显示进度
            if int(progress) % 5 == 0 or iteration == self.max_iterations - 1:
                print(f"   [进度] 第{iteration+1}/{self.max_iterations}次迭代 ({progress:.1f}%) | 全局最佳遮蔽时长: {self.global_best_fitness:.3f}s")

        return self.global_best_position, self.global_best_fitness

class GridSearchQ2:
    """网格搜索求解第二问"""
    def __init__(self, n_heading=8, n_time=5):
        self.n_heading = n_heading  # 航向角网格点数
        self.n_time = n_time        # 时间网格点数

    def search(self):
        """网格搜索（并行版本）"""
        print("开始网格搜索...")

        # 生成所有可能的组合
        heading_values = np.linspace(0, 2*math.pi, self.n_heading, endpoint=False)
        time_values = np.linspace(0, 10, self.n_time)

        total_combinations = len(heading_values) ** 5 * len(time_values) ** 5
        print(f"总搜索空间: {total_combinations}")

        # 生成所有组合
        combinations = []
        for h1 in heading_values:
            for h2 in heading_values:
                for h3 in heading_values:
                    for h4 in heading_values:
                        for h5 in heading_values:
                            for t1 in time_values:
                                for t2 in time_values:
                                    for t3 in time_values:
                                        for t4 in time_values:
                                            for t5 in time_values:
                                                genes = [h1, t1, h2, t2, h3, t3, h4, t4, h5, t5]
                                                combinations.append(genes)

        print(f"生成组合数量: {len(combinations)}")

        # 串行评估（避免多进程嵌套和内存问题）
        best_fitness = 0.0
        best_solution = None

        ga_temp = GeneticAlgorithmQ2()

        # 减小批次大小以避免内存问题
        batch_size = min(100, len(combinations))

        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i + batch_size]

            # 串行评估当前批次
            for j, genes in enumerate(batch):
                fitness = ga_temp.evaluate_fitness(genes)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = genes[:]

            progress = min(i + batch_size, len(combinations)) / len(combinations) * 100
            # 每5%显示进度
            if int(progress) % 5 == 0 or i + batch_size >= len(combinations):
                print(f"   [进度] 已处理 {min(i + batch_size, len(combinations))}/{len(combinations)} 组合 ({progress:.1f}%) | 当前最佳遮蔽时长: {best_fitness:.3f}s")

        return best_solution, best_fitness

# ==================== 主求解函数 ====================

def solve_problem2():
    """解决问题二：多无人机协同干扰优化"""
    print("=" * 80)
    print("CUMCM2025 第二问：多无人机协同干扰优化")
    print("=" * 80)
    print("[目标] 优化目标：最大化5架无人机对M1导弹的总遮蔽时长")
    print("[变量] 优化变量：每架无人机的航向角和投放时间")
    print(f"[系统] 检测到 {os.cpu_count()} 个CPU核心")
    print("[计算] 使用多线程并行计算，充分利用CPU资源")
    print("=" * 80)

    # 参数验证
    if len(ALL_UAVS) != 5:
        print(f"[警告] 无人机数量为{len(ALL_UAVS)}，期望为5架")
    if MISSILE_SPEED <= 0:
        raise ValueError("[错误] 导弹速度必须大于0")
    if UAV_SPEED <= 0:
        raise ValueError("[错误] 无人机速度必须大于0")

    start_time = time.time()
    total_steps = 3  # 三种优化算法
    current_step = 0

    try:
        # 方法1：遗传算法
        current_step += 1
        print(f"\n[算法 {current_step}] 方法1：遗传算法优化")
        print(f"   使用 {os.cpu_count()} CPU核心 | 种群大小: 20 | 迭代代数: 30")
        ga_start = time.time()
        ga = GeneticAlgorithmQ2(population_size=20, generations=30)
        ga_solution, ga_fitness = ga.evolve()
        ga_time = time.time() - ga_start
        print(".3f")
        print(".3f")

        # 方法2：粒子群优化
        current_step += 1
        print(f"\n[算法 {current_step}] 方法2：粒子群优化")
        print(f"   使用 {os.cpu_count()} CPU核心 | 粒子数量: 15 | 迭代代数: 30")
        pso_start = time.time()
        pso = ParticleSwarmQ2(swarm_size=15, max_iterations=30)
        pso_solution, pso_fitness = pso.optimize()
        pso_time = time.time() - pso_start
        print(".3f")
        print(".3f")

        # 方法3：网格搜索（优化规模）
        current_step += 1
        print(f"\n[算法 {current_step}] 方法3：网格搜索")
        print(f"   使用 {os.cpu_count()} CPU核心 | 航向网格: 4×4×4×4×4 | 时间网格: 3×3×3×3×3")
        gs_start = time.time()
        gs = GridSearchQ2(n_heading=4, n_time=3)  # 简化的网格密度
        gs_solution, gs_fitness = gs.search()
        gs_time = time.time() - gs_start
        print(".3f")
        print(".3f")

        # 选择最佳结果
        print("\n[比较] 正在比较三种算法结果...")
        results = [
            ("遗传算法", ga_solution, ga_fitness),
            ("粒子群优化", pso_solution, pso_fitness),
            ("网格搜索", gs_solution, gs_fitness)
        ]

        best_method, best_solution, best_fitness = max(results, key=lambda x: x[2])

        print("\n[结果] 算法比较结果:")
        print(f"   遗传算法: {ga_fitness:.3f}s (耗时: {ga_time:.1f}s)")
        print(f"   粒子群优化: {pso_fitness:.3f}s (耗时: {pso_time:.1f}s)")
        print(f"   网格搜索: {gs_fitness:.3f}s (耗时: {gs_time:.1f}s)")
        print(f"\n[最优] 最优算法: {best_method}")
        print(".3f")

        # 生成详细结果
        print("\n[生成] 正在生成最终优化结果...")
        results_data = []
        total_coverage = 0.0

        for i, uav_id in enumerate(ALL_UAVS):
            print(f"   [计算] 正在计算{uav_id}的遮蔽效果...")
            heading = best_solution[i * 2]
            t_drop = best_solution[i * 2 + 1]

            plan = UAVPlan(uav_id, heading, t_drop)
            coverage = plan.calculate_coverage()
            total_coverage += coverage

            # 计算投放和起爆位置
            drop_pos, _ = UavStateHorizontal(t_drop, FY_INIT[uav_id], UAV_SPEED, heading)
            expl_pos = ExplosionPoint(heading, t_drop, FUSE_DELAY_TIME, uav_id)

            result = {
                "无人机编号": uav_id,
                "航向角(度)": math.degrees(heading) % 360,
                "投放时间(s)": t_drop,
                "起爆时间(s)": plan.t_explode,
                "投放点x(m)": drop_pos[0],
                "投放点y(m)": drop_pos[1],
                "投放点z(m)": FY_INIT[uav_id][2],
                "起爆点x(m)": expl_pos[0],
                "起爆点y(m)": expl_pos[1],
                "起爆点z(m)": expl_pos[2],
                "遮蔽时长(s)": coverage
            }
            results_data.append(result)

            progress = (i + 1) / len(ALL_UAVS) * 100
            # 每25%显示进度（因为只有5个无人机）
            if int(progress) % 25 == 0 or i == len(ALL_UAVS) - 1:
                print(f"   [进度] 已完成 {i+1}/{len(ALL_UAVS)} 架无人机 ({progress:.1f}%) | 总遮蔽时长: {total_coverage:.3f}s")

        print("\n[完成] 所有无人机结果生成完成！")

        # 保存结果
        print("\n[保存] 正在保存优化结果...")
        print("   [文件] 保存CSV文件...")
        df = pd.DataFrame(results_data)
        df.to_csv("q2_solution.csv", index=False, encoding='utf-8-sig')

        print("   [文件] 保存Excel文件...")
        df.to_excel("q2_solution.xlsx", index=False)

        print("   [文件] 保存汇总信息...")
        summary = {
            "optimization_method": best_method,
            "total_coverage_time": best_fitness,
            "computation_time": time.time() - start_time,
            "uav_count": 5,
            "target_missile": "M1",
            "constraints": {
                "heading_range": "[0, 2π)",
                "time_range": "[0, 10]s"
            }
        }

        with open("q2_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        total_time = time.time() - start_time
        print("[完成] 所有结果保存完成！")
        print("=" * 80)
        print(f"[输出] 生成文件:")
        print("- q2_solution.csv (详细结果表格)")
        print("- q2_solution.xlsx (Excel格式)")
        print("- q2_summary.json (汇总信息)")
        print(".3f")
        print("=" * 80)

        return results_data, summary

    except KeyboardInterrupt:
        print("\n[中断] 用户中断了优化过程")
        return [], {"status": "interrupted"}
    except Exception as e:
        print(f"\n[错误] 优化过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return [], {"status": "error", "message": str(e)}

if __name__ == "__main__":
    solve_problem2()
