import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment
import time

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


# ============================ 2. 核心工具函数 ============================
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

# 烟幕弹位置计算
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

# 单烟幕弹遮蔽时长计算
def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
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


# ============================ 3. 快速优化函数（重点优化参数和策略） ============================

# 【优化】多策略单弹优化
def optimize_single_smoke_fast(drone_name, m_name, strategy="enhanced"):
    """快速但高效的单弹优化"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    # 扩展搜索边界
    bounds = [
        (v_min * 0.7, v_max * 1.3),        # 速度范围扩展
        (0, 2 * np.pi),                    # 方向角
        (0, max_flight_time - 0.5),        # 投放时刻
        (0.1, 25)                          # 起爆延迟
    ]
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    # 根据策略选择参数
    if strategy == "fast":
        config = {"popsize": 50, "maxiter": 60, "mutation": 0.8, "recombination": 0.8}
    elif strategy == "enhanced":
        config = {"popsize": 100, "maxiter": 120, "mutation": 0.9, "recombination": 0.85}
    else:  # "intensive"
        config = {"popsize": 150, "maxiter": 200, "mutation": 1.0, "recombination": 0.9}
    
    best_result = None
    best_score = float('inf')
    
    # 多次运行取最优（增加成功率）
    for run in range(3):
        try:
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
                seed=run * 42  # 不同随机种子
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None or best_score >= -0.1:
        return None
    
    v_opt, theta_opt, drop_time_opt, det_delay_opt = best_result.x
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -best_result.fun
    
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

# 【优化】智能速度选择
def get_smart_speed_candidates(drone_name, m_name, num_candidates=12):
    """基于几何和物理约束的智能速度候选"""
    drone = DRONES[drone_name]
    missile = MISSILES[m_name]
    v_min, v_max = drone["speed_range"]
    
    # 基础候选速度
    base_candidates = np.linspace(v_min, v_max, num_candidates // 2)
    
    # 基于导弹距离的推荐速度
    dist_to_missile = np.linalg.norm(missile["init_pos"] - drone["init_pos"])
    flight_time = missile["flight_time"]
    
    # 推荐速度：能在合理时间内到达拦截位置
    recommended_v = dist_to_missile / (flight_time * 0.7)  # 70%飞行时间内到达
    recommended_v = np.clip(recommended_v, v_min, v_max)
    
    # 在推荐速度附近增加更多候选
    v_range = (v_max - v_min) * 0.3
    smart_candidates = np.linspace(
        max(v_min, recommended_v - v_range), 
        min(v_max, recommended_v + v_range), 
        num_candidates // 2
    )
    
    # 合并并去重
    all_candidates = np.concatenate([base_candidates, smart_candidates])
    return np.unique(np.round(all_candidates, 1))

# 【优化】无人机轨迹优化
def optimize_drone_trajectory_fast(drone_name, m_name, retry=0):
    """快速但高效的无人机轨迹优化"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    print(f"[{drone_name}] 快速优化中...")
    
    # 智能速度候选
    v_candidates = get_smart_speed_candidates(drone_name, m_name, 15)
    
    best_v = None
    best_smokes = []
    max_total_time = 0
    
    # 分阶段优化：先快速搜索，再精细优化
    strategies = ["fast", "enhanced"] if retry == 0 else ["enhanced", "intensive"]
    
    for strategy in strategies:
        current_best_time = 0
        current_best_v = None
        current_best_smokes = []
        
        for v in v_candidates:
            drone["speed"] = v
            temp_smokes = []
            
            for i in range(max_smoke):
                min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
                max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
                if min_drop_time >= max_drop_time - 1e-3:
                    break
                
                smoke = optimize_single_smoke_fast(drone_name, m_name, strategy)
                
                if smoke and smoke["drop_time"] >= min_drop_time and smoke["effective_time"] > 0.1:
                    smoke["v"] = v
                    temp_smokes.append(smoke)
            
            total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
            if total_time > current_best_time:
                current_best_time = total_time
                current_best_v = v
                current_best_smokes = temp_smokes
        
        # 如果这一轮有改进，更新全局最优
        if current_best_time > max_total_time:
            max_total_time = current_best_time
            best_v = current_best_v
            best_smokes = current_best_smokes
            print(f"[{drone_name}] {strategy}策略找到更优解: {max_total_time:.2f}s")
    
    # 若仍未找到解，降低要求重试
    if not best_smokes and retry < 2:
        print(f"[{drone_name}] 重试优化 {retry+1}/2...")
        return optimize_drone_trajectory_fast(drone_name, m_name, retry+1)
    
    # 路径拟合与波动优化（简化版）
    if best_smokes:
        # 简化的方向拟合
        thetas = [s["theta"] for s in best_smokes]
        weights = [s["effective_time"] for s in best_smokes]
        ref_theta = np.average(thetas, weights=weights)
        
        # 快速波动优化
        for smoke in best_smokes:
            best_effect = smoke["effective_time"]
            best_theta = smoke["theta"]
            
            # 只在小范围内微调
            for theta_offset in [-0.1, 0, 0.1]:
                test_theta = ref_theta + theta_offset
                drone["direction"] = test_theta
                effect = calc_smoke_effective_time(drone_name, m_name, smoke["drop_time"], smoke["det_delay"])
                if effect > best_effect:
                    best_effect = effect
                    best_theta = test_theta
            
            smoke["theta"] = best_theta
            smoke["effective_time"] = best_effect
            smoke["det_pos"] = get_smoke_pos(drone_name, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
    
    drone["speed"] = best_v
    drone["direction"] = ref_theta if best_smokes else None
    drone["smokes"] = best_smokes
    
    total_time = sum([s["effective_time"] for s in best_smokes]) if best_smokes else 0
    print(f"[{drone_name}] 快速优化完成: {len(best_smokes)}枚弹, {total_time:.2f}s")
    
    return best_smokes

# 【优化】增强任务分配
def assign_tasks_smart(unoptimized_drones=None, iteration=0):
    """智能任务分配算法"""
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
            
            # 成本1：距离成本（归一化）
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v / 100
            
            # 成本2：时间紧急度（飞行时间越短成本越高）
            cost2 = 1500 / (m_flight_time + 1)
            
            # 成本3：速度匹配度
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 50
            
            # 成本4：角度优势
            drone_to_missile = m_init - d_init
            drone_to_target = TRUE_TARGET["center"] - d_init
            angle_diff = np.abs(np.arccos(np.clip(
                np.dot(drone_to_missile[:2], drone_to_target[:2]) / 
                (np.linalg.norm(drone_to_missile[:2]) * np.linalg.norm(drone_to_target[:2]) + 1e-8),
                -1, 1
            )))
            cost4 = angle_diff / np.pi * 200
            
            # 成本5：高度优势
            height_diff = abs(d_init[2] - m_init[2])
            cost5 = height_diff / 1000 * 50
            
            # 迭代权重：后期更注重效果
            iteration_weight = 1 + iteration * 0.05
            
            cost_matrix[i][j] = (cost1 + cost2 + cost3 + cost4 + cost5) * iteration_weight
    
    # 匈牙利算法分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    for i, j in zip(row_ind, col_ind):
        assignments[missile_list[j]].append(unoptimized_drones[i])
    
    # 处理未分配的无人机
    assigned_drones = set(row_ind)
    for i in range(n_drones):
        if i not in assigned_drones:
            min_cost_j = np.argmin(cost_matrix[i])
            assignments[missile_list[min_cost_j]].append(unoptimized_drones[i])
    
    return assignments

# 【核心】快速迭代优化主函数
def iterative_optimization_fast(max_iterations=25, improvement_threshold=0.2, max_stall_iter=4):
    """快速迭代优化算法"""
    print("=== 启动快速优化算法 ===")
    
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
        print(f"\n===== 快速迭代 {iteration + 1}/{max_iterations} =====")
        iter_start = time.time()
        
        # 1. 获取尚未找到有效解的无人机
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"待优化无人机: {drones_without_solution}")
        
        if not drones_without_solution:
            print("所有无人机都已找到有效解，停止迭代")
            break
        
        # 2. 智能任务分配
        assignments = assign_tasks_smart(drones_without_solution, iteration)
        print(f"任务分配: {assignments}")
        
        # 3. 优化执行
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                
                print(f"正在优化 {d_name} -> {m_name}...")
                smokes = optimize_drone_trajectory_fast(d_name, m_name)
                
                if smokes:
                    DRONES[d_name]["smokes"] = smokes
                    DRONES[d_name]["optimized"] = True
                else:
                    print(f"[{d_name}] 优化失败")
        
        # 4. 更新全局烟幕数据
        all_smokes = []
        for d_name in DRONES:
            if DRONES[d_name]["smokes"]:
                drone_smokes = [{**smoke, "drone": d_name} for smoke in DRONES[d_name]["smokes"]]
                all_smokes.extend(drone_smokes)
        
        # 5. 计算性能指标
        total_effective_time = sum([s["effective_time"] for s in all_smokes])
        improvement = total_effective_time - prev_total_time
        
        if total_effective_time > best_total_time:
            best_total_time = total_effective_time
            stall_count = 0
        else:
            stall_count += 1
        
        iter_time = time.time() - iter_start
        print(f"当前总遮蔽时长: {total_effective_time:.2f}s")
        print(f"改进量: {improvement:.2f}s")
        print(f"迭代用时: {iter_time:.1f}s")
        print(f"完成进度: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        
        # 6. 收敛判断
        if improvement < improvement_threshold:
            print(f"连续无改进次数: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                remaining = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
                if remaining:
                    print(f"仍有无人机无解，继续优化: {remaining}")
                    stall_count = max_stall_iter - 1
                else:
                    print(f"连续{max_stall_iter}轮无改进，停止迭代")
                    break
        
        prev_total_time = total_effective_time
    
    # 最终结果检查
    remaining = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining:
        print(f"\n警告：以下无人机仍未找到解: {remaining}")
    
    print(f"\n快速优化完成！最终总遮蔽时长: {best_total_time:.2f}s")
    return all_smokes


# ============================ 4. 结果输出与分析 ============================
def save_result(smokes, filename="smoke_optimization_result_fast.xlsx"):
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
    print(f"快速优化结果已保存到 {filename}")
    return df

def analyze_and_visualize(smokes):
    """分析和可视化结果"""
    if not smokes:
        print("无有效数据可分析")
        return
    
    # 性能分析
    total_time = sum([s["effective_time"] for s in smokes])
    print(f"\n{'快速算法性能分析':=^50}")
    print(f"总烟幕弹数量：{len(smokes)}")
    print(f"总遮蔽时长：{total_time:.2f}s")
    print(f"平均单弹效果：{total_time/len(smokes):.2f}s")
    
    # 按导弹分析
    missile_stats = {}
    for smoke in smokes:
        m = smoke["missile"]
        if m not in missile_stats:
            missile_stats[m] = {"count": 0, "total_time": 0}
        missile_stats[m]["count"] += 1
        missile_stats[m]["total_time"] += smoke["effective_time"]
    
    print(f"\n各导弹遮蔽情况：")
    for m_name in MISSILES.keys():
        if m_name in missile_stats:
            stats = missile_stats[m_name]
            coverage_rate = stats["total_time"] / MISSILES[m_name]["flight_time"] * 100
            print(f"{m_name}：{stats['count']}枚弹，{stats['total_time']:.2f}s，覆盖率{coverage_rate:.1f}%")
        else:
            print(f"{m_name}：0枚弹，0.00s，覆盖率0.0%")
    
    print(f"\n各无人机投放情况：")
    for d_name, d_data in DRONES.items():
        if d_data["smokes"]:
            total = sum([s["effective_time"] for s in d_data["smokes"]])
            efficiency = total / len(d_data["smokes"])
            print(f"{d_name}：{len(d_data['smokes'])}枚弹，总时长{total:.2f}s，效率{efficiency:.2f}s/弹")
        else:
            print(f"{d_name}：未找到有效投放方案")
    
    print("="*50)
    
    # 简化可视化
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 轨迹图
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="真目标")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200)
    
    # 导弹轨迹
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name}")
    
    # 无人机和烟幕
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^", label=f"{d_name}")
        
        if d_data["smokes"]:
            for smoke in d_data["smokes"]:
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=60, alpha=0.7)
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                    ax1.add_patch(circle)
    
    ax1.set_title("快速优化：布局图")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 导弹遮蔽时长
    missile_times = [missile_stats.get(m, {"total_time": 0})["total_time"] for m in MISSILES.keys()]
    ax2.bar(MISSILES.keys(), missile_times, color=colors)
    ax2.set_title("各导弹遮蔽时长")
    ax2.set_ylabel("时长(s)")
    for m, t in zip(MISSILES.keys(), missile_times):
        ax2.text(m, t + 0.3, f"{t:.1f}s", ha="center")
    
    # 3. 无人机效果
    drone_effects = []
    drone_names = []
    for d_name, d_data in DRONES.items():
        if d_data["smokes"]:
            drone_effects.append(sum([s["effective_time"] for s in d_data["smokes"]]))
            drone_names.append(d_name)
    
    if drone_effects:
        ax3.bar(drone_names, drone_effects, color=drone_colors[:len(drone_names)])
        ax3.set_title("各无人机总效果")
        ax3.set_ylabel("时长(s)")
        for name, effect in zip(drone_names, drone_effects):
            ax3.text(name, effect + 0.1, f"{effect:.1f}s", ha="center")
    
    # 4. 效果分布
    effect_times = [s["effective_time"] for s in smokes]
    ax4.hist(effect_times, bins=10, alpha=0.7, color="lightblue")
    ax4.set_title("遮蔽时长分布")
    ax4.set_xlabel("时长(s)")
    ax4.set_ylabel("数量")
    ax4.axvline(np.mean(effect_times), color='red', linestyle='--', label=f'平均: {np.mean(effect_times):.2f}s')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("fast_optimization_result.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================ 主函数执行 ============================
if __name__ == "__main__":
    print("="*60)
    print("快速烟幕弹投放优化算法")
    print("特点：高效参数 + 智能搜索 + 快速收敛")
    print("="*60)
    
    start_time = time.time()
    
    # 运行快速优化
    all_smokes = iterative_optimization_fast(
        max_iterations=25,
        improvement_threshold=0.2,
        max_stall_iter=4
    )
    
    total_time = time.time() - start_time
    print(f"\n快速优化总耗时: {total_time:.2f}s")
    
    if all_smokes:
        # 保存和分析结果
        result_df = save_result(all_smokes)
        analyze_and_visualize(all_smokes)
        
        print(f"\n{'快速算法特点':=^50}")
        print("✓ 智能速度候选生成")
        print("✓ 多策略分阶段搜索")
        print("✓ 增强的任务分配算法")
        print("✓ 快速收敛检测")
        print("✓ 简化但有效的参数调优")
        print("="*50)
        
    else:
        print("快速优化算法未找到有效解")
