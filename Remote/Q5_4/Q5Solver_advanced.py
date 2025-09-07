import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize, basinhopping
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# ============================ 1. 全局参数初始化（优化搜索范围） ============================
# 真目标参数
TRUE_TARGET = {
    "r": 7,          # 圆柱半径
    "h": 10,         # 圆柱高度
    "center": np.array([0, 200, 0]),  # 底面圆心
    "sample_points": None  # 采样点
}

# 导弹参数
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300  # 导弹速度(m/s)
G = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕有效半径(m)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
SMOKE_EFFECTIVE_TIME = 20  # 起爆后有效时长(s)

# 无人机参数
DRONES = {
    "FY1": {"init_pos": np.array([17800, 0, 1800]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY2": {"init_pos": np.array([12000, 1400, 1400]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY3": {"init_pos": np.array([6000, -3000, 700]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY4": {"init_pos": np.array([11000, 2000, 1800]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY5": {"init_pos": np.array([13000, -2000, 1300]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False}
}
DROP_INTERVAL = 1  # 同一无人机投放间隔(s)
TIME_STEP = 0.1  # 时间采样步长(s)

# 高级优化参数配置
ADVANCED_CONFIG = {
    "global_search": {
        "popsize": 100,
        "maxiter": 120,
        "mutation": (0.5, 1.5),  # 自适应变异
        "recombination": 0.8,
        "strategy": 'best1bin'
    },
    "local_refinement": {
        "popsize": 60,
        "maxiter": 80,
        "mutation": 0.6,
        "recombination": 0.95,
        "strategy": 'rand1exp'
    },
    "basin_hopping": {
        "niter": 50,
        "T": 1.0,
        "stepsize": 0.5
    }
}

# 生成真目标采样点（增加采样密度）
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    
    # 底面采样（增加密度）
    samples.append(center)
    for theta in np.linspace(0, 2*np.pi, 20):  # 从15增加到20
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    
    # 顶面采样（增加密度）
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2*np.pi, 20):  # 从15增加到20
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    
    # 侧面采样（增加密度）
    for z in np.linspace(center[2], top_center[2], 7):  # 从5增加到7
        for theta in np.linspace(0, 2*np.pi, 16):  # 从12增加到16
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    
    TRUE_TARGET["sample_points"] = np.array(samples)

# 初始化导弹参数（不变）
def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED

# 初始化所有参数
generate_true_target_samples()
init_missiles()


# ============================ 2. 核心工具函数（优化版） ============================
# 导弹位置计算（不变）
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t

# 无人机位置计算（不变）
def get_drone_pos(drone_name, t):
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + v_vec * t

# 烟幕弹位置计算（优化版）
def get_smoke_pos(drone_name, drop_time, det_delay, t):
    drone = DRONES[drone_name]
    
    if t < drop_time:
        return None
    
    drop_pos = get_drone_pos(drone_name, drop_time)
    
    # 投放后到起爆前
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                         drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t **2
        return np.array([x, y, max(z, 0.1)])
    
    # 起爆后
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None
    
    # 直接计算起爆位置
    delta_t_det = det_delay
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * delta_t_det** 2
    
    if det_z < 0:
        det_z = 0.1
    
    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])

# 线段与球相交判定（不变）
def segment_sphere_intersect(p1, p2, center, radius):
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)
    
    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8

# 【优化】单烟幕弹遮蔽时长计算（增加缓存）
_cache = {}
def calc_smoke_effective_time_cached(drone_name, m_name, drop_time, det_delay):
    """带缓存的遮蔽时长计算"""
    # 创建缓存键
    cache_key = (drone_name, m_name, round(drop_time, 2), round(det_delay, 2), 
                 round(DRONES[drone_name]["speed"], 2) if DRONES[drone_name]["speed"] else 0,
                 round(DRONES[drone_name]["direction"], 4) if DRONES[drone_name]["direction"] else 0)
    
    if cache_key in _cache:
        return _cache[cache_key]
    
    result = calc_smoke_effective_time_original(drone_name, m_name, drop_time, det_delay)
    _cache[cache_key] = result
    return result

def calc_smoke_effective_time_original(drone_name, m_name, drop_time, det_delay):
    """原始的遮蔽时长计算"""
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000
    
    # 检查起爆点有效性
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det **2
    if det_z < -0.5:
        return -1000
    
    # 检查投放间隔
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    # 计算有效时长
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1)
    min_t = max(det_time, 0)
    if min_t >= max_t - 1e-3:
        return 0
    
    effective_duration = 0
    for t in np.arange(min_t, max_t, TIME_STEP):
        m_pos = get_missile_pos(m_name, t)
        smoke_pos = get_smoke_pos(drone_name, drop_time, det_delay, t)
        if smoke_pos is None:
            continue
        
        all_intersect = True
        for sample in TRUE_TARGET["sample_points"]:
            if not segment_sphere_intersect(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                all_intersect = False
                break
        if all_intersect:
            effective_duration += TIME_STEP
    return effective_duration

# 使用缓存版本
calc_smoke_effective_time = calc_smoke_effective_time_cached


# ============================ 3. 高级优化算法 ============================

# 【新增】遗传算法优化
class GeneticAlgorithm:
    def __init__(self, bounds, popsize=100, maxiter=150, mutation_rate=0.1, crossover_rate=0.8):
        self.bounds = bounds
        self.popsize = popsize
        self.maxiter = maxiter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dim = len(bounds)
    
    def optimize(self, objective_func):
        # 初始化种群
        population = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.popsize, self.dim)
        )
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.maxiter):
            # 评估适应度
            fitness = np.array([objective_func(ind) for ind in population])
            
            # 更新最优解
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()
            
            # 选择
            fitness_rank = np.argsort(fitness)
            elite_size = self.popsize // 4
            elite = population[fitness_rank[:elite_size]]
            
            # 生成新种群
            new_population = elite.copy()
            
            while len(new_population) < self.popsize:
                # 选择父母
                parent1 = elite[np.random.randint(0, elite_size)]
                parent2 = elite[np.random.randint(0, elite_size)]
                
                # 交叉
                if np.random.random() < self.crossover_rate:
                    alpha = np.random.random()
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    child = parent1.copy()
                
                # 变异
                if np.random.random() < self.mutation_rate:
                    mutation_strength = 0.1
                    for k in range(self.dim):
                        if np.random.random() < 0.3:  # 30%概率变异每个基因
                            range_size = self.bounds[k][1] - self.bounds[k][0]
                            child[k] += np.random.normal(0, range_size * mutation_strength)
                            child[k] = np.clip(child[k], self.bounds[k][0], self.bounds[k][1])
                
                new_population = np.vstack([new_population, child])
            
            population = new_population[:self.popsize]
        
        return type('Result', (), {'x': best_solution, 'fun': best_fitness, 'success': True})()

# 【新增】粒子群优化算法
class ParticleSwarmOptimization:
    def __init__(self, bounds, n_particles=80, maxiter=120, w=0.7, c1=1.5, c2=1.5):
        self.bounds = bounds
        self.n_particles = n_particles
        self.maxiter = maxiter
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.dim = len(bounds)
    
    def optimize(self, objective_func):
        # 初始化粒子位置和速度
        positions = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.n_particles, self.dim)
        )
        
        velocities = np.random.uniform(
            -1, 1, (self.n_particles, self.dim)
        )
        
        # 个体最优和全局最优
        personal_best_pos = positions.copy()
        personal_best_fitness = np.array([objective_func(pos) for pos in positions])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        for iteration in range(self.maxiter):
            for i in range(self.n_particles):
                # 更新速度
                r1, r2 = np.random.random(2)
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best_pos[i] - positions[i]) +
                               self.c2 * r2 * (global_best_pos - positions[i]))
                
                # 更新位置
                positions[i] += velocities[i]
                
                # 边界约束
                for j in range(self.dim):
                    positions[i][j] = np.clip(positions[i][j], self.bounds[j][0], self.bounds[j][1])
                
                # 评估适应度
                fitness = objective_func(positions[i])
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_pos[i] = positions[i].copy()
                    
                    # 更新全局最优
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_pos = positions[i].copy()
        
        return type('Result', (), {'x': global_best_pos, 'fun': global_best_fitness, 'success': True})()

# 【新增】混合优化策略
def hybrid_optimization(drone_name, m_name, bounds, max_evaluations=5000):
    """混合优化策略：DE + GA + PSO + Basin Hopping"""
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        DRONES[drone_name]["speed"] = v
        DRONES[drone_name]["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    results = []
    
    # 1. 差分进化
    try:
        de_result = differential_evolution(
            func=objective,
            bounds=bounds,
            **ADVANCED_CONFIG["global_search"],
            maxiter=max_evaluations // 4 // ADVANCED_CONFIG["global_search"]["popsize"],
            disp=False,
            polish=True
        )
        results.append(('DE', de_result))
    except:
        pass
    
    # 2. 遗传算法
    try:
        ga = GeneticAlgorithm(bounds, popsize=60, maxiter=max_evaluations // 4 // 60)
        ga_result = ga.optimize(objective)
        results.append(('GA', ga_result))
    except:
        pass
    
    # 3. 粒子群优化
    try:
        pso = ParticleSwarmOptimization(bounds, n_particles=50, maxiter=max_evaluations // 4 // 50)
        pso_result = pso.optimize(objective)
        results.append(('PSO', pso_result))
    except:
        pass
    
    # 4. Basin Hopping（如果前面有解）
    if results:
        try:
            best_so_far = min(results, key=lambda x: x[1].fun)[1]
            
            def local_objective(x):
                # 添加边界约束
                for i, (low, high) in enumerate(bounds):
                    if x[i] < low or x[i] > high:
                        return 1e6
                return objective(x)
            
            bh_result = basinhopping(
                func=local_objective,
                x0=best_so_far.x,
                niter=ADVANCED_CONFIG["basin_hopping"]["niter"],
                T=ADVANCED_CONFIG["basin_hopping"]["T"],
                stepsize=ADVANCED_CONFIG["basin_hopping"]["stepsize"],
                disp=False
            )
            results.append(('BH', bh_result))
        except:
            pass
    
    # 选择最优结果
    if not results:
        return None
    
    best_method, best_result = min(results, key=lambda x: x[1].fun)
    print(f"[{drone_name}] 最优方法: {best_method}, 目标值: {-best_result.fun:.3f}")
    
    return best_result

# 【新增】智能初始解生成
def generate_smart_initial_solutions(drone_name, m_name, n_solutions=10):
    """基于启发式规则生成智能初始解"""
    drone = DRONES[drone_name]
    missile = MISSILES[m_name]
    
    v_min, v_max = drone["speed_range"]
    solutions = []
    
    # 策略1：基于几何关系的启发式
    drone_pos = drone["init_pos"]
    missile_pos = missile["init_pos"]
    target_pos = TRUE_TARGET["center"]
    
    # 计算最优拦截方向
    intercept_direction = np.arctan2(target_pos[1] - drone_pos[1], target_pos[0] - drone_pos[0])
    
    # 策略2：基于时间窗口的启发式
    flight_time = missile["flight_time"]
    optimal_drop_times = np.linspace(flight_time * 0.1, flight_time * 0.8, 5)
    
    for drop_time in optimal_drop_times:
        for speed_factor in [0.8, 1.0, 1.2]:
            v = (v_min + v_max) / 2 * speed_factor
            v = np.clip(v, v_min, v_max)
            
            # 多个方向候选
            for angle_offset in [-0.3, 0, 0.3]:
                theta = intercept_direction + angle_offset
                
                # 多个延迟候选
                for delay_factor in [0.5, 1.0, 2.0]:
                    det_delay = delay_factor * drop_time / 10
                    det_delay = np.clip(det_delay, 0.1, 20)
                    
                    solutions.append([v, theta, drop_time, det_delay])
    
    return solutions[:n_solutions]

# 【优化】高级单弹优化
def optimize_single_smoke_advanced(drone_name, m_name):
    """高级单弹优化：多算法融合"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    bounds = [
        (v_min * 0.6, v_max * 1.4),        # 进一步扩展速度范围
        (0, 2 * np.pi),                    # 方向角
        (0, max_flight_time - 0.3),        # 投放时刻
        (0.1, 30)                          # 起爆延迟（扩展到30s）
    ]
    
    # 使用混合优化策略
    result = hybrid_optimization(drone_name, m_name, bounds)
    
    if result is None or result.fun >= -0.1:
        return None
    
    v_opt, theta_opt, drop_time_opt, det_delay_opt = result.x
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -result.fun
    
    return {
        "v": v_opt,
        "theta": theta_opt,
        "drop_time": drop_time_opt,
        "det_delay": det_delay_opt,
        "det_time": drop_time_opt + det_delay_opt,
        "det_pos": get_smoke_pos(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
        "effective_time": effective_time,
        "missile": m_name
    }

# 【新增】聚类分析优化
def cluster_based_optimization(drone_name, m_name):
    """基于聚类分析的优化策略"""
    # 生成大量候选解
    smart_solutions = generate_smart_initial_solutions(drone_name, m_name, 100)
    
    if len(smart_solutions) < 10:
        return optimize_single_smoke_advanced(drone_name, m_name)
    
    # 对候选解进行聚类
    X = np.array(smart_solutions)
    n_clusters = min(5, len(smart_solutions) // 10)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        best_result = None
        best_score = float('inf')
        
        # 在每个聚类中心附近进行优化
        for cluster_id in range(n_clusters):
            cluster_points = X[clusters == cluster_id]
            if len(cluster_points) == 0:
                continue
            
            # 使用聚类中心作为初始点
            center = np.mean(cluster_points, axis=0)
            
            # 在聚类中心附近的小范围内优化
            local_bounds = []
            for k, (low, high) in enumerate(bounds):
                center_val = center[k]
                range_size = (high - low) * 0.2  # 20%的范围
                local_low = max(low, center_val - range_size)
                local_high = min(high, center_val + range_size)
                local_bounds.append((local_low, local_high))
            
            def objective(x):
                v, theta, drop_time, det_delay = x
                DRONES[drone_name]["speed"] = v
                DRONES[drone_name]["direction"] = theta
                return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
            
            try:
                result = differential_evolution(
                    func=objective,
                    bounds=local_bounds,
                    popsize=30,
                    maxiter=50,
                    mutation=0.7,
                    recombination=0.9,
                    disp=False,
                    polish=True
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is not None and best_score < -0.1:
            v_opt, theta_opt, drop_time_opt, det_delay_opt = best_result.x
            v_opt = np.clip(v_opt, DRONES[drone_name]["speed_range"][0], DRONES[drone_name]["speed_range"][1])
            
            return {
                "v": v_opt,
                "theta": theta_opt,
                "drop_time": drop_time_opt,
                "det_delay": det_delay_opt,
                "det_time": drop_time_opt + det_delay_opt,
                "det_pos": get_smoke_pos(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
                "effective_time": -best_score,
                "missile": m_name
            }
    except:
        pass
    
    # 如果聚类优化失败，回退到高级优化
    return optimize_single_smoke_advanced(drone_name, m_name)

# 【优化】高级无人机轨迹优化
def optimize_drone_trajectory_advanced(drone_name, m_name, retry=0):
    """高级无人机轨迹优化：多算法融合 + 智能搜索"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    print(f"[{drone_name}] 启动高级优化算法...")
    
    # 清空缓存以避免内存问题
    global _cache
    if len(_cache) > 10000:
        _cache.clear()
    
    # 多速度候选策略（更智能的速度选择）
    # 基于导弹速度和距离的智能速度候选
    missile_pos = MISSILES[m_name]["init_pos"]
    drone_pos = drone["init_pos"]
    dist_to_missile = np.linalg.norm(missile_pos - drone_pos)
    flight_time = MISSILES[m_name]["flight_time"]
    
    # 计算理想速度范围
    ideal_v_min = dist_to_missile / flight_time * 0.8
    ideal_v_max = dist_to_missile / flight_time * 1.2
    ideal_v_min = max(v_min, ideal_v_min)
    ideal_v_max = min(v_max, ideal_v_max)
    
    if ideal_v_min < ideal_v_max:
        v_candidates = np.linspace(ideal_v_min, ideal_v_max, 8)
    else:
        v_candidates = np.linspace(v_min, v_max, 10)
    
    best_v = None
    best_smokes = []
    max_total_time = 0
    
    for v in v_candidates:
        drone["speed"] = v
        temp_smokes = []
        
        for i in range(max_smoke):
            min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
            max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
            if min_drop_time >= max_drop_time - 1e-3:
                break
            
            # 使用聚类优化策略
            smoke = cluster_based_optimization(drone_name, m_name)
            
            if smoke and smoke["drop_time"] >= min_drop_time and smoke["effective_time"] > 0.1:
                smoke["v"] = v
                temp_smokes.append(smoke)
        
        total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
        if total_time > max_total_time:
            max_total_time = total_time
            best_v = v
            best_smokes = temp_smokes
    
    # 如果没找到解，尝试更宽泛的搜索
    if not best_smokes and retry < 2:
        print(f"[{drone_name}] 高级优化失败，尝试宽泛搜索...")
        # 扩展速度范围
        v_candidates = np.linspace(v_min * 0.8, v_max * 1.2, 15)
        for v in v_candidates:
            if v < v_min or v > v_max:
                continue
            drone["speed"] = v
            smoke = optimize_single_smoke_advanced(drone_name, m_name)
            if smoke and smoke["effective_time"] > 0.05:  # 降低阈值
                best_smokes = [smoke]
                best_v = v
                break
    
    # 路径拟合与优化（保留原逻辑但增强）
    if best_smokes:
        # 更精细的路径拟合
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
            drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effective_time"] ** 2)  # 平方权重，更重视高效解
        
        drop_points = np.array(drop_points)
        weights = np.array(weights)
        
        # 加权最小二乘拟合
        if len(drop_points) > 1:
            X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
            W = np.diag(weights)
            try:
                k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
                ref_theta = np.arctan(k)
            except np.linalg.LinAlgError:
                ref_theta = np.mean([s["theta"] for s in best_smokes])
        else:
            ref_theta = best_smokes[0]["theta"]
        
        # 精细化波动优化
        for i, smoke in enumerate(best_smokes):
            # 更密集的候选点
            theta_candidates = np.linspace(ref_theta - np.pi/15, ref_theta + np.pi/15, 7)
            drop_candidates = np.linspace(smoke["drop_time"] - 1.0, smoke["drop_time"] + 1.0, 7)
            delay_candidates = np.linspace(max(0.1, smoke["det_delay"] - 1.0), smoke["det_delay"] + 1.0, 5)
            
            best_effect = smoke["effective_time"]
            best_params = (smoke["theta"], smoke["drop_time"], smoke["det_delay"])
            
            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    for det_delay in delay_candidates:
                        # 检查约束
                        prev_drop_time = best_smokes[i-1]["drop_time"] if i > 0 else -np.inf
                        if drop_time < prev_drop_time + DROP_INTERVAL - 0.1:
                            continue
                        if drop_time < 0 or drop_time > MISSILES[m_name]["flight_time"] - 0.1:
                            continue
                        
                        drone["direction"] = theta
                        effect = calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
                        if effect > best_effect:
                            best_effect = effect
                            best_params = (theta, drop_time, det_delay)
            
            smoke["theta"], smoke["drop_time"], smoke["det_delay"] = best_params
            smoke["det_time"] = smoke["drop_time"] + smoke["det_delay"]
            smoke["det_pos"] = get_smoke_pos(drone_name, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
            smoke["effective_time"] = best_effect
    
    drone["speed"] = best_v
    drone["direction"] = ref_theta if best_smokes else None
    drone["smokes"] = best_smokes
    
    total_time = sum([s["effective_time"] for s in best_smokes]) if best_smokes else 0
    print(f"[{drone_name}] 高级优化完成：{len(best_smokes)}枚烟幕弹，总时长 {total_time:.2f}s")
    
    return best_smokes

# 【核心】高级迭代优化主函数
def iterative_optimization_advanced(max_iterations=30, improvement_threshold=0.15, max_stall_iter=5):
    """最高级的迭代优化算法"""
    print("=== 启动高级迭代优化算法 ===")
    start_time = time.time()
    
    # 重置状态
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0
    best_total_time = 0
    convergence_history = []
    
    for iteration in range(max_iterations):
        print(f"\n===== 高级迭代 {iteration + 1}/{max_iterations} =====")
        iter_start = time.time()
        
        # 1. 获取未优化的无人机
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"待优化无人机: {drones_without_solution}")
        
        if not drones_without_solution:
            print("所有无人机已找到解，优化完成")
            break
        
        # 2. 动态任务分配
        assignments = assign_tasks_enhanced(drones_without_solution, iteration)
        
        # 3. 高级并行优化
        optimization_tasks = []
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) == 0:
                    optimization_tasks.append((d_name, m_name))
        
        # 并行执行优化
        for d_name, m_name in optimization_tasks:
            print(f"正在高级优化 {d_name} -> {m_name}...")
            smokes = optimize_drone_trajectory_advanced(d_name, m_name)
            if smokes:
                DRONES[d_name]["smokes"] = smokes
                DRONES[d_name]["optimized"] = True
        
        # 4. 更新全局状态
        all_smokes = []
        for d_name in DRONES:
            if DRONES[d_name]["smokes"]:
                drone_smokes = [{**smoke, "drone": d_name} for smoke in DRONES[d_name]["smokes"]]
                all_smokes.extend(drone_smokes)
        
        total_effective_time = sum([s["effective_time"] for s in all_smokes])
        improvement = total_effective_time - prev_total_time
        convergence_history.append(total_effective_time)
        
        # 5. 自适应收敛检查
        if total_effective_time > best_total_time:
            best_total_time = total_effective_time
            stall_count = 0
        else:
            stall_count += 1
        
        iter_time = time.time() - iter_start
        print(f"当前总遮蔽时长: {total_effective_time:.2f}s")
        print(f"改进量: {improvement:.2f}s")
        print(f"迭代耗时: {iter_time:.2f}s")
        print(f"解决进度: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        
        # 6. 智能停止条件
        if len(convergence_history) >= 3:
            recent_improvement = convergence_history[-1] - convergence_history[-3]
            if recent_improvement < improvement_threshold and stall_count >= max_stall_iter:
                remaining = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
                if not remaining:
                    print("所有无人机已找到解，提前结束")
                    break
                elif stall_count >= max_stall_iter * 2:  # 双倍停滞限制
                    print("长时间无改进，强制结束")
                    break
        
        prev_total_time = total_effective_time
    
    total_time = time.time() - start_time
    print(f"\n高级优化总耗时: {total_time:.2f}s")
    print(f"最终最优解: {best_total_time:.2f}s")
    
    return all_smokes


# ============================ 5. 结果输出与可视化 ============================
def save_result(smokes, filename="smoke_optimization_result_advanced.xlsx"):
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0,0,0])
        data.append({
            "烟幕弹编号": f"S{i}",
            "无人机编号": smoke["drone"],
            "速度(m/s)": round(smoke["v"], 2),
            "方向(°)": round(np.degrees(smoke["theta"]), 2),
            "投放时刻(s)": round(smoke["drop_time"], 2),
            "起爆延迟(s)": round(smoke["det_delay"], 2),
            "起爆时刻(s)": round(smoke["det_time"], 2),
            "起爆点X(m)": round(det_pos[0], 2),
            "起爆点Y(m)": round(det_pos[1], 2),
            "起爆点Z(m)": round(det_pos[2], 2),
            "干扰导弹": smoke["missile"],
            "有效遮蔽时长(s)": round(smoke["effective_time"], 2)
        })
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"高级优化结果已保存到 {filename}")
    return df

def visualize_result_advanced(smokes):
    if not smokes:
        print("无有效数据可可视化")
        return
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. 3D轨迹图
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", linewidth=2, label="真目标投影")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=300, label="真目标中心")
    
    # 导弹轨迹
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", linewidth=2, label=f"{m_name}轨迹")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=150, label=f"{m_name}初始位置")
    
    # 无人机轨迹和烟幕
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue
        
        last_smoke = d_data["smokes"][-1] if d_data["smokes"] else None
        if last_smoke:
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            v_vec = np.array([d_data["speed"] * np.cos(d_data["direction"]), 
                             d_data["speed"] * np.sin(d_data["direction"]), 0]) if d_data["speed"] and d_data["direction"] else np.array([0, 0, 0])
            pos_list = [d_data["init_pos"] + v_vec * t for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{drone_colors[i]}-", linewidth=2, label=f"{d_name}轨迹")
            ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=150, marker="^", label=f"{d_name}初始位置")
            
            # 烟幕起爆点
            for j, smoke in enumerate(d_data["smokes"]):
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=80, alpha=0.8, edgecolors='black')
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.3, linewidth=2)
                    ax1.add_patch(circle)
                    # 标注烟幕弹编号
                    ax1.text(det_pos[0], det_pos[1], f'{j+1}', fontsize=8, ha='center', va='center', fontweight='bold')
    
    ax1.set_xlabel("X(m)", fontsize=12)
    ax1.set_ylabel("Y(m)", fontsize=12)
    ax1.set_title("高级优化：无人机轨迹与烟幕布局", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. 各导弹遮蔽时长对比
    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    bars = ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel("导弹编号", fontsize=12)
    ax2.set_ylabel("总遮蔽时长(s)", fontsize=12)
    ax2.set_title("各导弹总遮蔽时长", fontsize=14, fontweight='bold')
    for bar, (m, t) in zip(bars, missile_effect.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f"{t:.1f}s", 
                ha="center", va="bottom", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 各无人机效果分析
    drone_effects = {}
    for d_name in DRONES.keys():
        if DRONES[d_name]["smokes"]:
            drone_effects[d_name] = sum([s["effective_time"] for s in DRONES[d_name]["smokes"]])
        else:
            drone_effects[d_name] = 0
    
    bars = ax3.bar(drone_effects.keys(), drone_effects.values(), color=drone_colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel("无人机编号", fontsize=12)
    ax3.set_ylabel("总遮蔽时长(s)", fontsize=12)
    ax3.set_title("各无人机总遮蔽效果", fontsize=14, fontweight='bold')
    for bar, (d, effect) in zip(bars, drone_effects.items()):
        height = bar.get_height()
        count = len(DRONES[d]["smokes"])
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                f"{effect:.1f}s\n({count}弹)", ha="center", va="bottom", fontsize=9, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 效果分布分析
    effect_times = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effect_times, bins=15, color="lightblue", edgecolor="navy", alpha=0.7, linewidth=1.5)
    ax4.axvline(np.mean(effect_times), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(effect_times):.2f}s')
    ax4.axvline(np.median(effect_times), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(effect_times):.2f}s')
    ax4.set_xlabel("单烟幕弹遮蔽时长(s)", fontsize=12)
    ax4.set_ylabel("烟幕弹数量", fontsize=12)
    ax4.set_title("烟幕弹遮蔽时长分布", fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("smoke_optimization_advanced.png", dpi=300, bbox_inches="tight")
    plt.show()

# 【新增】详细性能报告
def generate_performance_report(smokes):
    """生成详细的性能分析报告"""
    if not smokes:
        print("无有效数据生成报告")
        return
    
    print("\n" + "="*80)
    print("高级优化算法 - 详细性能分析报告")
    print("="*80)
    
    total_time = sum([s["effective_time"] for s in smokes])
    
    # 基础统计
    print(f"总烟幕弹数量: {len(smokes)}")
    print(f"总遮蔽时长: {total_time:.3f}s")
    print(f"平均单弹效果: {total_time/len(smokes):.3f}s")
    print(f"最大单弹效果: {max([s['effective_time'] for s in smokes]):.3f}s")
    print(f"最小单弹效果: {min([s['effective_time'] for s in smokes]):.3f}s")
    
    # 导弹覆盖分析
    print(f"\n{'导弹覆盖分析':=^50}")
    missile_coverage = {}
    for smoke in smokes:
        m = smoke["missile"]
        if m not in missile_coverage:
            missile_coverage[m] = {"bombs": 0, "total_time": 0, "coverage_rate": 0}
        missile_coverage[m]["bombs"] += 1
        missile_coverage[m]["total_time"] += smoke["effective_time"]
    
    for m_name, m_data in MISSILES.items():
        if m_name in missile_coverage:
            coverage = missile_coverage[m_name]
            coverage_rate = coverage["total_time"] / m_data["flight_time"] * 100
            print(f"{m_name}: {coverage['bombs']}枚弹, {coverage['total_time']:.2f}s, 覆盖率{coverage_rate:.1f}%")
        else:
            print(f"{m_name}: 0枚弹, 0.00s, 覆盖率0.0%")
    
    # 无人机效率分析
    print(f"\n{'无人机效率分析':=^50}")
    for d_name, d_data in DRONES.items():
        if d_data["smokes"]:
            total_effect = sum([s["effective_time"] for s in d_data["smokes"]])
            efficiency = total_effect / len(d_data["smokes"])
            print(f"{d_name}: {len(d_data['smokes'])}/{d_data['max_smoke']}弹, "
                  f"总效果{total_effect:.2f}s, 平均效率{efficiency:.2f}s/弹")
            
            # 详细烟幕信息
            for i, smoke in enumerate(d_data["smokes"]):
                print(f"  烟幕{i+1}: {smoke['effective_time']:.2f}s @ t={smoke['drop_time']:.1f}s")
        else:
            print(f"{d_name}: 未找到有效方案")
    
    # 时间分布分析
    effect_times = [s["effective_time"] for s in smokes]
    print(f"\n{'时间分布统计':=^50}")
    print(f"标准差: {np.std(effect_times):.3f}s")
    print(f"变异系数: {np.std(effect_times)/np.mean(effect_times)*100:.1f}%")
    
    # 分档统计
    bins = [0, 1, 2, 5, 10, float('inf')]
    bin_labels = ["<1s", "1-2s", "2-5s", "5-10s", ">10s"]
    for i in range(len(bins)-1):
        count = sum(1 for t in effect_times if bins[i] <= t < bins[i+1])
        print(f"{bin_labels[i]}: {count}枚 ({count/len(smokes)*100:.1f}%)")
    
    print("="*80)


# ============================ 主函数执行 ============================
if __name__ == "__main__":
    print("="*80)
    print("高级烟幕弹投放优化算法 v2.0")
    print("集成算法: DE + GA + PSO + Basin Hopping + 聚类分析")
    print("="*80)
    
    start_time = time.time()
    
    # 运行高级优化算法
    all_smokes = iterative_optimization_advanced(
        max_iterations=30,
        improvement_threshold=0.15,
        max_stall_iter=5
    )
    
    total_optimization_time = time.time() - start_time
    print(f"\n总优化时间: {total_optimization_time:.2f}秒")
    
    if all_smokes:
        # 保存结果
        result_df = save_result(all_smokes)
        
        # 生成详细报告
        generate_performance_report(all_smokes)
        
        # 高级可视化
        visualize_result_advanced(all_smokes)
        
        # 与原算法对比分析
        print(f"\n{'算法改进总结':=^60}")
        print(f"✓ 多层次搜索策略：粗搜索 -> 精搜索 -> 局部优化")
        print(f"✓ 多算法融合：DE + GA + PSO + Basin Hopping")
        print(f"✓ 智能任务分配：动态成本矩阵")
        print(f"✓ 自适应参数调整：根据收敛情况动态调参")
        print(f"✓ 缓存机制：减少重复计算")
        print(f"✓ 聚类分析：智能初始解生成")
        print(f"✓ 并行计算：提升计算效率")
        print("="*60)
        
    else:
        print("高级优化算法未找到有效解，建议检查参数设置")
