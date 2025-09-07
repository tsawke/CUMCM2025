import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
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

# 无人机参数（保留optimized标记，用于跟踪优化状态）
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

# 优化参数配置
OPTIMIZATION_CONFIG = {
    "coarse_search": {
        "popsize": 80,
        "maxiter": 100,
        "mutation": 0.9,
        "recombination": 0.8
    },
    "fine_search": {
        "popsize": 120,
        "maxiter": 150,
        "mutation": 0.7,
        "recombination": 0.9
    },
    "local_search": {
        "popsize": 40,
        "maxiter": 50,
        "mutation": 0.5,
        "recombination": 0.95
    }
}

# 生成真目标采样点（不变）
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    # 底面采样
    samples.append(center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    # 顶面采样
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    # 侧面采样
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2*np.pi, 12):
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


# ============================ 2. 核心工具函数（增加Z坐标容错） ============================
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

# 烟幕弹位置计算（增加Z<0时的容错，避免微小误差导致失效）
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
        return np.array([x, y, max(z, 0.1)])  # Z轴最小0.1，避免无效
    
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
        det_z = 0.1  # 容错处理
    
    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])  # Z轴最小0.1

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

# 单烟幕弹遮蔽时长计算（放宽投放间隔判定）
def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):  # 增加容错
        return -1000
    
    # 检查起爆点有效性
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det **2
    if det_z < -0.5:  # 放宽Z轴判定
        return -1000
    
    # 检查投放间隔（增加0.1s容错）
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    # 计算有效时长（确保时间范围合理）
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


# ============================ 3. 优化函数（多层次搜索策略） ============================

# 【新增】自适应参数调整
def adaptive_parameters(iteration, max_iterations, current_best, stagnation_count):
    """根据迭代情况自适应调整参数"""
    progress = iteration / max_iterations
    
    # 基础参数
    base_config = OPTIMIZATION_CONFIG["fine_search"].copy()
    
    # 根据进度调整
    if progress < 0.3:  # 早期：探索为主
        base_config["mutation"] = 0.9 - 0.2 * progress
        base_config["recombination"] = 0.7 + 0.2 * progress
    elif progress < 0.7:  # 中期：平衡探索与开发
        base_config["mutation"] = 0.7 - 0.1 * (progress - 0.3) / 0.4
        base_config["recombination"] = 0.9
    else:  # 后期：开发为主
        base_config["mutation"] = 0.6 - 0.1 * (progress - 0.7) / 0.3
        base_config["recombination"] = 0.95
    
    # 根据停滞情况调整
    if stagnation_count > 3:
        base_config["mutation"] += 0.2  # 增加探索
        base_config["popsize"] = int(base_config["popsize"] * 1.2)
    
    return base_config

# 【优化】多层次单弹优化
def optimize_single_smoke_multilevel(drone_name, m_name, search_level="fine"):
    """多层次单弹优化：粗搜索 -> 精搜索 -> 局部优化"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    # 根据搜索层次选择参数
    config = OPTIMIZATION_CONFIG[search_level]
    
    # 扩展搜索边界
    bounds = [
        (v_min * 0.7, v_max * 1.3),        # 速度（扩展30%）
        (0, 2 * np.pi),                    # 方向角
        (0, max_flight_time - 0.5),        # 投放时刻
        (0.1, 25)                          # 起爆延迟（扩展到25s）
    ]
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    # 多次尝试不同的随机种子
    best_result = None
    best_score = float('inf')
    
    for seed in range(3):  # 尝试3个不同的随机种子
        result = differential_evolution(
            func=objective,
            bounds=bounds,
            mutation=config["mutation"],
            recombination=config["recombination"],
            popsize=config["popsize"],
            maxiter=config["maxiter"],
            tol=1e-4,
            disp=False,
            polish=True,
            seed=seed
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    if best_result is None:
        return None
    
    v_opt, theta_opt, drop_time_opt, det_delay_opt = best_result.x
    # 速度截断到合理范围
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -best_result.fun
    
    if effective_time < 0.1:  # 过滤无效解
        return None
    
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

# 【新增】局部搜索优化
def local_search_optimization(drone_name, m_name, initial_solution):
    """基于已有解进行局部搜索优化"""
    if initial_solution is None:
        return None
    
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    
    # 在初始解周围进行局部搜索
    v0, theta0, drop_time0, det_delay0 = (
        initial_solution["v"], 
        initial_solution["theta"],
        initial_solution["drop_time"],
        initial_solution["det_delay"]
    )
    
    # 局部搜索范围
    delta_v = (v_max - v_min) * 0.1
    delta_theta = np.pi / 12
    delta_time = 2.0
    delta_delay = 2.0
    
    bounds = [
        (max(v_min, v0 - delta_v), min(v_max, v0 + delta_v)),
        (theta0 - delta_theta, theta0 + delta_theta),
        (max(0, drop_time0 - delta_time), min(MISSILES[m_name]["flight_time"] - 0.5, drop_time0 + delta_time)),
        (max(0.1, det_delay0 - delta_delay), det_delay0 + delta_delay)
    ]
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    # 使用L-BFGS-B进行局部优化
    try:
        result = minimize(
            fun=objective,
            x0=[v0, theta0, drop_time0, det_delay0],
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            v_opt, theta_opt, drop_time_opt, det_delay_opt = result.x
            v_opt = np.clip(v_opt, v_min, v_max)
            effective_time = -result.fun
            
            if effective_time > initial_solution["effective_time"]:
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
    except:
        pass
    
    return initial_solution

# 【优化】无人机轨迹优化（多层次搜索）
def optimize_drone_trajectory_enhanced(drone_name, m_name, retry=0):
    """增强的无人机轨迹优化：多层次搜索策略"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    print(f"[{drone_name}] 开始多层次优化，目标导弹: {m_name}")
    
    # 阶段1：粗搜索 - 快速探索可行域
    print(f"[{drone_name}] 阶段1: 粗搜索...")
    coarse_candidates = []
    v_candidates = np.linspace(v_min, v_max, 12)  # 增加候选点
    
    for v in v_candidates:
        drone["speed"] = v
        temp_smokes = []
        
        for i in range(max_smoke):
            min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
            max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
            if min_drop_time >= max_drop_time - 1e-3:
                break
            
            # 粗搜索
            smoke = optimize_single_smoke_multilevel(drone_name, m_name, "coarse_search")
            if smoke and smoke["drop_time"] >= min_drop_time:
                smoke["v"] = v
                temp_smokes.append(smoke)
        
        if temp_smokes:
            total_time = sum([s["effective_time"] for s in temp_smokes])
            coarse_candidates.append((v, temp_smokes, total_time))
    
    # 选择最好的几个候选
    coarse_candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = coarse_candidates[:min(3, len(coarse_candidates))]
    
    if not top_candidates:
        print(f"[{drone_name}] 粗搜索未找到可行解")
        return []
    
    # 阶段2：精搜索 - 在最优候选附近精细搜索
    print(f"[{drone_name}] 阶段2: 精搜索...")
    best_smokes = []
    max_total_time = 0
    best_v = None
    
    for v_candidate, candidate_smokes, _ in top_candidates:
        # 在候选速度附近进行精搜索
        v_range = np.linspace(max(v_min, v_candidate - 10), min(v_max, v_candidate + 10), 5)
        
        for v in v_range:
            drone["speed"] = v
            temp_smokes = []
            
            for i in range(max_smoke):
                min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
                max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
                if min_drop_time >= max_drop_time - 1e-3:
                    break
                
                # 精搜索
                smoke = optimize_single_smoke_multilevel(drone_name, m_name, "fine_search")
                if smoke and smoke["drop_time"] >= min_drop_time and smoke["effective_time"] > 0.1:
                    smoke["v"] = v
                    temp_smokes.append(smoke)
            
            total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
            if total_time > max_total_time:
                max_total_time = total_time
                best_v = v
                best_smokes = temp_smokes
    
    # 阶段3：局部优化 - 对最优解进行局部优化
    if best_smokes:
        print(f"[{drone_name}] 阶段3: 局部优化...")
        optimized_smokes = []
        for smoke in best_smokes:
            drone["speed"] = smoke["v"]
            optimized = local_search_optimization(drone_name, m_name, smoke)
            if optimized:
                optimized_smokes.append(optimized)
        
        if optimized_smokes:
            best_smokes = optimized_smokes
            best_v = best_smokes[0]["v"] if best_smokes else best_v
    
    # 路径拟合与波动优化（保留原有逻辑）
    if best_smokes:
        # 加权拟合直线
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
            drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effective_time"])
        drop_points = np.array(drop_points)
        weights = np.array(weights)
        
        # 拟合方向角
        X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
        W = np.diag(weights)
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
            ref_theta = np.arctan(k)
        except np.linalg.LinAlgError:
            ref_theta = np.mean([s["theta"] for s in best_smokes]) if best_smokes else 0
        
        # 波动优化
        for i, smoke in enumerate(best_smokes):
            theta_candidates = [ref_theta - np.pi/20, ref_theta, ref_theta + np.pi/20]
            drop_candidates = [smoke["drop_time"] - 0.5, smoke["drop_time"], smoke["drop_time"] + 0.5]
            
            best_effect = smoke["effective_time"]
            best_params = (smoke["theta"], smoke["drop_time"])
            
            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    prev_drop_time = best_smokes[i-1]["drop_time"] if i > 0 else -np.inf
                    if drop_time < prev_drop_time + DROP_INTERVAL - 0.1:
                        continue
                    drone["direction"] = theta
                    effect = calc_smoke_effective_time(drone_name, m_name, drop_time, smoke["det_delay"])
                    if effect > best_effect:
                        best_effect = effect
                        best_params = (theta, drop_time)
            
            smoke["theta"], smoke["drop_time"] = best_params
            smoke["det_time"] = smoke["drop_time"] + smoke["det_delay"]
            smoke["det_pos"] = get_smoke_pos(drone_name, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
            smoke["effective_time"] = best_effect
    
    drone["speed"] = best_v
    drone["direction"] = ref_theta if best_smokes else None
    drone["smokes"] = best_smokes
    
    total_time = sum([s["effective_time"] for s in best_smokes]) if best_smokes else 0
    print(f"[{drone_name}] 优化完成：{len(best_smokes)}枚烟幕弹，总遮蔽时长 {total_time:.2f}s")
    
    return best_smokes

# 【优化】智能任务分配
def assign_tasks_enhanced(unoptimized_drones=None, iteration=0):
    """增强的任务分配：考虑更多因素的成本矩阵"""
    if unoptimized_drones is None:
        unoptimized_drones = list(DRONES.keys())
    
    missile_list = list(MISSILES.keys())
    n_drones = len(unoptimized_drones)
    n_missiles = len(missile_list)
    if n_drones == 0:
        return {m: [] for m in missile_list}
    
    cost_matrix = np.zeros((n_drones, n_missiles))
    for i, d_name in enumerate(unoptimized_drones):
        d_init = DRONES[d_name]["init_pos"]
        d_avg_v = (DRONES[d_name]["speed_range"][0] + DRONES[d_name]["speed_range"][1]) / 2
        d_max_smokes = DRONES[d_name]["max_smoke"]
        
        for j, m_name in enumerate(missile_list):
            m_init = MISSILES[m_name]["init_pos"]
            m_flight_time = MISSILES[m_name]["flight_time"]
            
            # 成本1：无人机到导弹初始位置的距离成本
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v / 100  # 归一化
            
            # 成本2：导弹紧急度（飞行时间越短越紧急）
            cost2 = 2000 / (m_flight_time + 1)
            
            # 成本3：速度匹配度
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 50
            
            # 成本4：位置优势（考虑相对位置）
            relative_pos = m_init - d_init
            angle_advantage = np.abs(np.arctan2(relative_pos[1], relative_pos[0]))
            cost4 = angle_advantage / np.pi * 100
            
            # 成本5：无人机载弹量优势
            cost5 = (3 - d_max_smokes) * 200  # 载弹量越多成本越低
            
            # 成本6：迭代自适应权重
            iteration_weight = 1 + iteration * 0.1  # 后期更注重效果
            
            cost_matrix[i][j] = (cost1 + cost2 + cost3 + cost4 + cost5) * iteration_weight
    
    # 匈牙利算法分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    for i, j in zip(row_ind, col_ind):
        assignments[missile_list[j]].append(unoptimized_drones[i])
    
    # 未分配的无人机分配给成本最低的导弹
    assigned_drones = set(row_ind)
    for i in range(n_drones):
        if i not in assigned_drones:
            min_cost_j = np.argmin(cost_matrix[i])
            assignments[missile_list[min_cost_j]].append(unoptimized_drones[i])
    
    return assignments

# 【新增】并行优化支持
def optimize_drone_parallel(args):
    """并行优化单个无人机的包装函数"""
    drone_name, m_name = args
    return drone_name, m_name, optimize_drone_trajectory_enhanced(drone_name, m_name)

# 【核心优化】迭代优化主函数
def iterative_optimization_enhanced(max_iterations=25, improvement_threshold=0.2, max_stall_iter=4):
    """增强的迭代优化：多层次搜索 + 自适应参数 + 并行计算"""
    print("=== 启动增强迭代优化算法 ===")
    
    # 重置无人机状态
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0
    best_total_time = 0
    
    for iteration in range(max_iterations):
        print(f"\n===== 迭代 {iteration + 1}/{max_iterations} =====")
        iteration_start_time = time.time()
        
        # 1. 获取尚未找到有效解的无人机
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"尚未找到有效解的无人机: {drones_without_solution}")
        
        # 2. 检查是否所有无人机都有解
        if not drones_without_solution:
            print("所有无人机都已找到有效解，停止迭代")
            break
        
        # 3. 智能任务分配
        assignments = assign_tasks_enhanced(drones_without_solution, iteration)
        print(f"任务分配结果: {assignments}")
        
        # 4. 自适应参数调整
        current_total_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        adaptive_config = adaptive_parameters(iteration, max_iterations, current_total_time, stall_count)
        
        # 更新优化配置
        OPTIMIZATION_CONFIG["fine_search"].update(adaptive_config)
        
        # 5. 并行优化无人机
        optimization_tasks = []
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) == 0:  # 只优化没有解的无人机
                    optimization_tasks.append((d_name, m_name))
        
        print(f"准备并行优化 {len(optimization_tasks)} 个任务...")
        
        # 使用进程池并行优化（如果任务数量足够多）
        if len(optimization_tasks) > 1:
            try:
                with ProcessPoolExecutor(max_workers=min(4, len(optimization_tasks))) as executor:
                    future_to_task = {executor.submit(optimize_drone_parallel, task): task for task in optimization_tasks}
                    
                    for future in as_completed(future_to_task):
                        d_name, m_name, smokes = future.result()
                        if smokes:
                            DRONES[d_name]["smokes"] = smokes
                            DRONES[d_name]["optimized"] = True
                            print(f"[{d_name}] 并行优化成功：{len(smokes)}枚烟幕弹")
                        else:
                            print(f"[{d_name}] 并行优化失败")
            except Exception as e:
                print(f"并行优化出错，切换到串行模式: {e}")
                # fallback到串行优化
                for d_name, m_name in optimization_tasks:
                    smokes = optimize_drone_trajectory_enhanced(d_name, m_name)
                    if smokes:
                        DRONES[d_name]["smokes"] = smokes
                        DRONES[d_name]["optimized"] = True
        else:
            # 串行优化
            for d_name, m_name in optimization_tasks:
                smokes = optimize_drone_trajectory_enhanced(d_name, m_name)
                if smokes:
                    DRONES[d_name]["smokes"] = smokes
                    DRONES[d_name]["optimized"] = True
        
        # 6. 更新全局烟幕数据
        iteration_smokes = []
        for d_name in DRONES:
            if DRONES[d_name]["smokes"]:
                drone_smokes = [{**smoke, "drone": d_name} for smoke in DRONES[d_name]["smokes"]]
                iteration_smokes.extend(drone_smokes)
        
        all_smokes = iteration_smokes  # 更新全局烟幕数据
        
        # 7. 计算总遮蔽时间和改进量
        total_effective_time = sum([s["effective_time"] for s in all_smokes])
        improvement = total_effective_time - prev_total_time
        
        # 8. 更新最优记录
        if total_effective_time > best_total_time:
            best_total_time = total_effective_time
            stall_count = 0
        else:
            stall_count += 1
        
        iteration_time = time.time() - iteration_start_time
        print(f"当前总遮蔽时长: {total_effective_time:.2f}s")
        print(f"相比上一轮改进量: {improvement:.2f}s")
        print(f"迭代用时: {iteration_time:.2f}s")
        print(f"已有有效解的无人机数量: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        
        # 9. 收敛性检查
        if improvement < improvement_threshold:
            print(f"连续无有效改进次数: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
                if remaining_without_solution:
                    print(f"仍有无人机没有解，继续优化: {remaining_without_solution}")
                    stall_count = max_stall_iter - 1  # 避免无限循环
                else:
                    print(f"连续{max_stall_iter}轮无有效改进，停止迭代")
                    break
        
        prev_total_time = total_effective_time
    
    # 检查最终结果
    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\n警告：以下无人机仍未找到有效解: {remaining_without_solution}")
    
    print(f"\n优化完成！最终总遮蔽时长: {best_total_time:.2f}s")
    return all_smokes


# ============================ 4. 结果输出与可视化（保持不变） ============================
def save_result(smokes, filename="smoke_optimization_result_enhanced.xlsx"):
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
    print(f"结果已保存到 {filename}")
    return df

def visualize_result(smokes):
    if not smokes:
        print("无有效数据可可视化")
        return
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 绘制真目标、导弹轨迹、无人机轨迹
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="真目标投影")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="真目标中心")
    
    # 导弹轨迹
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", label=f"{m_name}轨迹")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name}初始位置")
    
    # 无人机轨迹和烟幕
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue
        # 无人机轨迹
        last_smoke = d_data["smokes"][-1] if d_data["smokes"] else None
        if last_smoke:
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            v_vec = np.array([d_data["speed"] * np.cos(d_data["direction"]), d_data["speed"] * np.sin(d_data["direction"]), 0]) if d_data["speed"] and d_data["direction"] else np.array([0, 0, 0])
            pos_list = [d_data["init_pos"] + v_vec * t for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{drone_colors[i]}-", label=f"{d_name}轨迹")
            ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^", label=f"{d_name}初始位置")
            
            # 烟幕起爆点
            for smoke in d_data["smokes"]:
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                    ax1.add_patch(circle)
    
    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.set_title("增强优化：无人机、导弹轨迹及烟幕起爆点")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. 各导弹遮蔽时长
    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors)
    ax2.set_xlabel("导弹编号")
    ax2.set_ylabel("总遮蔽时长(s)")
    ax2.set_title("各导弹总遮蔽时长")
    for m, t in missile_effect.items():
        ax2.text(m, t + 0.5, f"{t:.1f}s", ha="center")
    
    # 3. 各无人机烟幕弹数量
    drone_smoke_count = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    ax3.bar(drone_smoke_count.keys(), drone_smoke_count.values(), color=drone_colors)
    ax3.set_xlabel("无人机编号")
    ax3.set_ylabel("烟幕弹数量")
    ax3.set_title("各无人机烟幕弹投放数量")
    for d, cnt in drone_smoke_count.items():
        ax3.text(d, cnt + 0.05, str(cnt), ha="center")
    
    # 4. 遮蔽时长分布
    effect_times = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effect_times, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("单烟幕弹遮蔽时长(s)")
    ax4.set_ylabel("烟幕弹数量")
    ax4.set_title("单烟幕弹遮蔽时长分布")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("smoke_optimization_visualization_enhanced.png", dpi=300, bbox_inches="tight")
    plt.show()

# 【新增】性能分析函数
def analyze_performance(smokes):
    """分析优化性能"""
    if not smokes:
        return
    
    print("\n" + "="*60)
    print("性能分析报告")
    print("="*60)
    
    total_time = sum([s["effective_time"] for s in smokes])
    print(f"总烟幕弹数量：{len(smokes)}")
    print(f"总遮蔽时长：{total_time:.2f}s")
    print(f"平均单弹遮蔽时长：{total_time/len(smokes):.2f}s")
    
    # 按导弹分析
    missile_stats = {}
    for smoke in smokes:
        m = smoke["missile"]
        if m not in missile_stats:
            missile_stats[m] = {"count": 0, "total_time": 0}
        missile_stats[m]["count"] += 1
        missile_stats[m]["total_time"] += smoke["effective_time"]
    
    print("\n各导弹遮蔽情况：")
    for m, stats in missile_stats.items():
        print(f"{m}：{stats['count']}枚弹，总时长{stats['total_time']:.2f}s，平均{stats['total_time']/stats['count']:.2f}s")
    
    # 按无人机分析
    print("\n各无人机投放情况：")
    for d_name, d_data in DRONES.items():
        if d_data["smokes"]:
            total = sum([s["effective_time"] for s in d_data["smokes"]])
            print(f"{d_name}：{len(d_data['smokes'])}枚弹，总遮蔽时长{total:.2f}s")
        else:
            print(f"{d_name}：未找到有效投放方案")
    
    print("="*60)


# ============================ 主函数执行 ============================
if __name__ == "__main__":
    start_time = time.time()
    print("启动增强版烟幕弹投放优化算法...")
    
    # 运行增强的迭代优化
    all_smokes = iterative_optimization_enhanced(
        max_iterations=25, 
        improvement_threshold=0.2, 
        max_stall_iter=4
    )
    
    total_time = time.time() - start_time
    print(f"\n总优化时间: {total_time:.2f}s")
    
    if all_smokes:
        # 保存结果
        result_df = save_result(all_smokes)
        
        # 性能分析
        analyze_performance(all_smokes)
        
        # 可视化
        visualize_result(all_smokes)
        
    else:
        print("未找到有效的烟幕弹投放方案")
