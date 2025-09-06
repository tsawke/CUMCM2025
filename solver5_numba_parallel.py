import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, prange, types
from numba.typed import Dict, List
from numba.core import types as nb_types
import numba as nb
from scipy.optimize import differential_evolution, linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

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

# 为numba优化准备的全局数组
MISSILE_INIT_POSITIONS = np.array([MISSILES[m]["init_pos"] for m in ["M1", "M2", "M3"]])
MISSILE_DIRECTIONS = np.array([MISSILES[m]["dir"] for m in ["M1", "M2", "M3"]])
MISSILE_FLIGHT_TIMES = np.array([MISSILES[m]["flight_time"] for m in ["M1", "M2", "M3"]])
DRONE_INIT_POSITIONS = np.array([DRONES[d]["init_pos"] for d in ["FY1", "FY2", "FY3", "FY4", "FY5"]])
DRONE_SPEED_RANGES = np.array([DRONES[d]["speed_range"] for d in ["FY1", "FY2", "FY3", "FY4", "FY5"]])
TARGET_SAMPLES = TRUE_TARGET["sample_points"]

# ============================ 2. 核心工具函数（numba优化） ============================

@jit(nopython=True, cache=True)
def get_missile_pos_numba(missile_idx, t):
    """导弹位置计算（numba优化版）"""
    flight_time = MISSILE_FLIGHT_TIMES[missile_idx]
    if t > flight_time:
        return MISSILE_INIT_POSITIONS[missile_idx] + MISSILE_DIRECTIONS[missile_idx] * flight_time
    return MISSILE_INIT_POSITIONS[missile_idx] + MISSILE_DIRECTIONS[missile_idx] * t

@jit(nopython=True, cache=True)
def get_drone_pos_numba(drone_idx, t, speed, direction):
    """无人机位置计算（numba优化版）"""
    init_pos = DRONE_INIT_POSITIONS[drone_idx]
    if speed <= 0:
        return init_pos + np.array([0.0, 0.0, 0.0])
    
    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])
    return init_pos + v_vec * t

@jit(nopython=True, cache=True)
def get_smoke_pos_numba(drone_idx, drop_time, det_delay, t, speed, direction):
    """烟幕弹位置计算（numba优化版）"""
    if t < drop_time:
        return np.array([0.0, 0.0, -1.0])  # 无效标记
    
    drop_pos = get_drone_pos_numba(drone_idx, drop_time, speed, direction)
    
    # 投放后到起爆前
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t * delta_t
        return np.array([x, y, max(z, 0.1)])
    
    # 起爆后
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return np.array([0.0, 0.0, -1.0])  # 无效标记
    
    # 计算起爆位置
    delta_t_det = det_delay
    v_vec = np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * delta_t_det * delta_t_det
    
    if det_z < 0.0:
        det_z = 0.1
    
    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])

@jit(nopython=True, cache=True)
def segment_sphere_intersect_numba(p1, p2, center, radius):
    """线段与球相交判定（numba优化版）"""
    vec_p = p2 - p1
    vec_c = center - p1
    dot_cp = np.dot(vec_c, vec_p)
    dot_pp = np.dot(vec_p, vec_p)
    
    if dot_pp < 1e-8:
        return np.linalg.norm(p1 - center) <= radius + 1e-8
    
    t = dot_cp / dot_pp
    
    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8

@jit(nopython=True, cache=True)
def calc_smoke_effective_time_numba(drone_idx, missile_idx, drop_time, det_delay, speed, direction, 
                                  existing_drop_times, n_existing):
    """单烟幕弹遮蔽时长计算（numba优化版）"""
    # 检查速度范围
    v_min, v_max = DRONE_SPEED_RANGES[drone_idx]
    if not (v_min - 1e-3 <= speed <= v_max + 1e-3):
        return -1000.0
    
    # 检查起爆点有效性
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos_numba(drone_idx, drop_time, speed, direction)
    det_z = drop_pos[2] - 0.5 * G * det_delay * det_delay
    if det_z < -0.5:
        return -1000.0
    
    # 检查投放间隔
    for i in range(n_existing):
        if abs(drop_time - existing_drop_times[i]) < DROP_INTERVAL - 0.1:
            return -1000.0
    
    # 计算有效时长
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILE_FLIGHT_TIMES[missile_idx] + 1)
    min_t = max(det_time, 0.0)
    if min_t >= max_t - 1e-3:
        return 0.0
    
    effective_duration = 0.0
    t = min_t
    while t < max_t:
        m_pos = get_missile_pos_numba(missile_idx, t)
        smoke_pos = get_smoke_pos_numba(drone_idx, drop_time, det_delay, t, speed, direction)
        
        if smoke_pos[2] > 0:  # 有效烟幕
            all_intersect = True
            for sample_idx in range(TARGET_SAMPLES.shape[0]):
                sample = TARGET_SAMPLES[sample_idx]
                if not segment_sphere_intersect_numba(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                    all_intersect = False
                    break
            if all_intersect:
                effective_duration += TIME_STEP
        
        t += TIME_STEP
    
    return effective_duration

@jit(nopython=True, parallel=True, cache=True)
def parallel_smoke_evaluation(drone_idx, missile_idx, param_array, existing_drop_times, n_existing):
    """并行评估多个参数组合（numba并行优化）"""
    n_params = param_array.shape[0]
    results = np.zeros(n_params)
    
    for i in prange(n_params):
        speed, direction, drop_time, det_delay = param_array[i]
        results[i] = calc_smoke_effective_time_numba(drone_idx, missile_idx, drop_time, det_delay, 
                                                   speed, direction, existing_drop_times, n_existing)
    
    return results

# ============================ 3. 优化函数（多线程差分进化） ============================

def optimize_single_smoke_parallel(drone_name, m_name):
    """单弹优化（并行版本）"""
    drone_idx = list(DRONES.keys()).index(drone_name)
    missile_idx = list(MISSILES.keys()).index(m_name)
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    # 准备现有投放时间数组
    existing_times = np.array([s["drop_time"] for s in drone["smokes"]], dtype=np.float64)
    if len(existing_times) == 0:
        existing_times = np.array([0.0])
    
    bounds = [
        (v_min * 0.8, v_max * 1.2),
        (0, 2 * np.pi),
        (0, max_flight_time - 1),
        (0.1, 20)
    ]
    
    def objective_parallel(x):
        """并行目标函数"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        results = parallel_smoke_evaluation(drone_idx, missile_idx, x, existing_times, len(existing_times))
        return -results[0] if len(results) == 1 else -results
    
    # 使用更大的种群和并行评估
    result = differential_evolution(
        func=objective_parallel,
        bounds=bounds,
        mutation=0.8,
        recombination=0.9,
        popsize=80,  # 增大种群
        maxiter=100,  # 增大迭代次数
        tol=1e-3,
        disp=False,
        polish=True,
        workers=1  # scipy内部并行，我们用numba并行
    )
    
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
        "effective_time": effective_time if effective_time > 1e-3 else 0,
        "missile": m_name
    }

def optimize_drone_trajectory_parallel(args):
    """无人机轨迹优化（并行版本）"""
    drone_name, m_name, retry = args
    drone_idx = list(DRONES.keys()).index(drone_name)
    missile_idx = list(MISSILES.keys()).index(m_name)
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    # 使用更多速度候选点
    v_candidates = np.linspace(v_min, v_max, 12)
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
            
            # 准备现有投放时间
            existing_times = np.array([s["drop_time"] for s in temp_smokes], dtype=np.float64)
            if len(existing_times) == 0:
                existing_times = np.array([0.0])
            
            def objective_parallel(x):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                # 添加速度列
                x_with_speed = np.column_stack([np.full(x.shape[0], v), x])
                results = parallel_smoke_evaluation(drone_idx, missile_idx, x_with_speed, 
                                                  existing_times, len(existing_times))
                return -results[0] if len(results) == 1 else -results
            
            result = differential_evolution(
                func=objective_parallel,
                bounds=[(0, 2*np.pi), (min_drop_time, max_drop_time), (0.1, 10)],
                mutation=0.7,
                recombination=0.8,
                popsize=60,
                maxiter=80,
                disp=False,
                workers=1
            )
            
            theta_opt, drop_time_opt, det_delay_opt = result.x
            drone["direction"] = theta_opt
            effective_time = calc_smoke_effective_time(drone_name, m_name, drop_time_opt, det_delay_opt)
            
            if effective_time > 0.1:
                smoke = {
                    "v": v,
                    "theta": theta_opt,
                    "drop_time": drop_time_opt,
                    "det_delay": det_delay_opt,
                    "det_time": drop_time_opt + det_delay_opt,
                    "det_pos": get_smoke_pos(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
                    "effective_time": effective_time,
                    "missile": m_name
                }
                temp_smokes.append(smoke)
        
        total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
        if total_time > max_total_time:
            max_total_time = total_time
            best_v = v
            best_smokes = temp_smokes
    
    if not best_smokes and retry < 3:
        return optimize_drone_trajectory_parallel((drone_name, m_name, retry + 1))
    
    # 路径优化（保持原逻辑）
    if best_smokes:
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
            drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effective_time"])
        drop_points = np.array(drop_points)
        weights = np.array(weights)
        
        X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
        W = np.diag(weights)
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
            ref_theta = np.arctan(k)
        except np.linalg.LinAlgError:
            ref_theta = np.mean([s["theta"] for s in best_smokes]) if best_smokes else 0
        
        # 波动优化
        for i, smoke in enumerate(best_smokes):
            theta_candidates = [ref_theta - np.pi/24, ref_theta, ref_theta + np.pi/24]
            drop_candidates = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]
            
            best_effect = smoke["effective_time"]
            best_params = (smoke["theta"], smoke["drop_time"])
            
            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    prev_drop_time = best_smokes[i-1]["drop_time"] if i > 0 and i-1 < len(best_smokes) else -np.inf
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
    return (drone_name, best_smokes)

# 保持原有的非numba版本函数以兼容性
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t

def get_drone_pos(drone_name, t):
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + v_vec * t

def get_smoke_pos(drone_name, drop_time, det_delay, t):
    drone = DRONES[drone_name]
    
    if t < drop_time:
        return None
    
    drop_pos = get_drone_pos(drone_name, drop_time)
    
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                         drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t **2
        return np.array([x, y, max(z, 0.1)])
    
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None
    
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

def segment_sphere_intersect(p1, p2, center, radius):
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)
    
    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8

def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000
    
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det **2
    if det_z < -0.5:
        return -1000
    
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
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

# ============================ 4. 任务分配与迭代优化（并行版本） ============================

def assign_tasks(unoptimized_drones=None):
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
        for j, m_name in enumerate(missile_list):
            m_init = MISSILES[m_name]["init_pos"]
            m_flight_time = MISSILES[m_name]["flight_time"]
            
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v
            cost2 = 1000 / (m_flight_time + 1)
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 100
            
            cost_matrix[i][j] = cost1 + cost2 + cost3
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    for i, j in zip(row_ind, col_ind):
        assignments[missile_list[j]].append(unoptimized_drones[i])
    
    assigned_drones = set(row_ind)
    for i in range(n_drones):
        if i not in assigned_drones:
            min_cost_j = np.argmin(cost_matrix[i])
            assignments[missile_list[min_cost_j]].append(unoptimized_drones[i])
    
    return assignments

def iterative_optimization_parallel(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    """迭代优化主函数（并行版本）"""
    # 重置无人机状态
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0
    
    # 获取CPU核心数
    n_cores = mp.cpu_count()
    print(f"使用 {n_cores} 个CPU核心进行并行计算")
    
    for iteration in range(max_iterations):
        print(f"\n===== 迭代 {iteration + 1}/{max_iterations} =====")
        
        # 1. 获取尚未找到有效解的无人机
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"尚未找到有效解的无人机: {drones_without_solution}")
        
        # 2. 检查是否所有无人机都有解
        if not drones_without_solution:
            print("所有无人机都已找到有效解，停止迭代")
            break
        
        # 3. 任务分配
        assignments = assign_tasks(drones_without_solution)
        
        # 4. 并行优化无人机
        current_total_time = 0
        iteration_smokes = []
        optimized_this_iter = []
        
        # 准备并行任务
        optimization_tasks = []
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                optimization_tasks.append((d_name, m_name, 0))
        
        print(f"并行优化 {len(optimization_tasks)} 个任务...")
        
        # 使用进程池并行执行优化
        with ProcessPoolExecutor(max_workers=min(n_cores, len(optimization_tasks))) as executor:
            results = list(executor.map(optimize_drone_trajectory_parallel, optimization_tasks))
        
        # 处理优化结果
        for drone_name, smokes in results:
            if smokes:
                drone_smokes = [{**smoke, "drone": drone_name} for smoke in smokes]
                iteration_smokes.extend(drone_smokes)
                current_total_time += sum([s["effective_time"] for s in smokes])
                print(f"[{drone_name}] 优化成功：{len(smokes)}枚烟幕弹，总遮蔽时长 {sum([s['effective_time'] for s in smokes]):.2f}s")
            else:
                print(f"[{drone_name}] 仍未找到有效投放方案")
            
            DRONES[drone_name]["optimized"] = True
            optimized_this_iter.append(drone_name)
        
        # 5. 更新全局烟幕数据
        all_smokes.extend(iteration_smokes)
        
        # 计算总遮蔽时间
        total_effective_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        print(f"当前总遮蔽时长: {total_effective_time:.2f}s")
        print(f"本轮优化无人机: {optimized_this_iter}")
        print(f"已有有效解的无人机数量: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        
        # 6. 检查改进量
        improvement = total_effective_time - prev_total_time
        print(f"相比上一轮改进量: {improvement:.2f}s")
        
        if improvement < improvement_threshold:
            stall_count += 1
            print(f"连续无有效改进次数: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                if drones_without_solution:
                    print(f"连续{max_stall_iter}轮无有效改进，但仍有无人机没有解，继续优化...")
                    stall_count = max_stall_iter - 1
                else:
                    print(f"连续{max_stall_iter}轮无有效改进，停止迭代")
                    break
        else:
            stall_count = 0
        
        prev_total_time = total_effective_time
    
    # 检查结果
    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\n警告：达到最大迭代次数，以下无人机仍未找到有效解: {remaining_without_solution}")
    
    return all_smokes

# ============================ 5. 结果输出与可视化（不变） ============================
def save_result(smokes, filename="smoke_optimization_result_numba_parallel.xlsx"):
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
    ax1.set_title("无人机、导弹轨迹及烟幕起爆点（Numba并行优化）")
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
    plt.savefig("smoke_optimization_visualization_numba_parallel.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================ 主函数执行 ============================
if __name__ == "__main__":
    import time
    
    print("启动Numba并行优化版本...")
    print(f"检测到 {mp.cpu_count()} 个CPU核心")
    
    start_time = time.time()
    
    # 使用并行版本的迭代优化
    all_smokes = iterative_optimization_parallel(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3)
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    if all_smokes:
        result_df = save_result(all_smokes)
        visualize_result(all_smokes)
        print("\n" + "="*50)
        print("最终结果汇总（Numba并行优化版）：")
        print(f"总烟幕弹数量：{len(all_smokes)}")
        print(f"总遮蔽时长：{sum([s['effective_time'] for s in all_smokes]):.2f}s")
        print(f"优化总耗时：{optimization_time:.2f}秒")
        print(f"平均每轮耗时：{optimization_time/20:.2f}秒")
        print("\n各无人机投放详情：")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"{d_name}：{len(d_data['smokes'])}枚弹，总遮蔽时长{total:.2f}s")
            else:
                print(f"{d_name}：未找到有效投放方案")
        print("="*50)
        print("性能提升说明：")
        print("- 使用Numba JIT编译加速核心计算函数")
        print("- 使用numba.prange实现并行计算")
        print("- 使用ProcessPoolExecutor进行多进程并行优化")
        print("- 增大差分进化算法的种群规模和迭代次数")
        print("="*50)
    else:
        print("未找到有效的烟幕弹投放方案")
