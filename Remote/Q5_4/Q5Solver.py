import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment

# ============================ 1. 全局参数初始化（优化搜索范围） ============================
TRUE_TARGET = {
    "r": 7,
    "h": 10,
    "center": np.array([0, 200, 0]),
    "sample_points": None
}

MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300
G = 9.8
SMOKE_RADIUS = 10
SMOKE_SINK_SPEED = 3
SMOKE_EFFECTIVE_TIME = 20

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
DROP_INTERVAL = 1
TIME_STEP = 0.1

def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    samples.append(center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2*np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    TRUE_TARGET["sample_points"] = np.array(samples)

def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED

generate_true_target_samples()
init_missiles()

# ============================ 2. 核心工具函数（关键修正） ============================
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
    """
    修正要点：
    1) 不再把 z 负值“抬高”；一旦 z<0，直接视为无烟雾（返回 None）。
    2) 起爆高度 det_z<0 也视为无效（None）。
    """
    drone = DRONES[drone_name]
    if t < drop_time:
        return None

    drop_pos = get_drone_pos(drone_name, drop_time)

    # 投放后到起爆前：烟雾尚未生成，用弹道 z 计算；若<0 直接无效
    if t < drop_time + det_delay:
        dt = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                          drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + v_vec[0] * dt
        y = drop_pos[1] + v_vec[1] * dt
        z = drop_pos[2] - 0.5 * G * dt**2
        return None if z < 0.0 else np.array([x, y, z])

    # 起爆后：若起爆高度<0 直接无效；随时间下沉到<0 也无效
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None

    dt_det = det_delay
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                      drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_pos[0] + v_vec[0] * dt_det
    det_y = drop_pos[1] + v_vec[1] * dt_det
    det_z = drop_pos[2] - 0.5 * G * dt_det**2
    if det_z < 0.0:
        return None

    z = det_z - SMOKE_SINK_SPEED * (t - det_time)
    return None if z < 0.0 else np.array([det_x, det_y, z])

def segment_sphere_intersect(p1, p2, center, radius):
    """
    关键修正：
    只承认“在导弹→目标点的线段内部”的相交。
    之前的实现允许用端点（尤其是目标采样点）作为最近点，从而把“在目标背面但球覆盖了采样点”的情况误判为命中。
    现在要求投影参数 t ∈ [0,1] 才可能 True；否则直接 False。
    """
    v = p2 - p1
    w = center - p1
    vv = np.dot(v, v) + 1e-8
    t = np.dot(w, v) / vv
    if t < 0.0 or t > 1.0:
        return False  # 背面/越界：不可能遮挡
    nearest = p1 + t * v
    return np.linalg.norm(nearest - center) <= radius + 1e-8

def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    """
    关键修正：
    1) 起爆高度 det_z < 0 => 直接判无效（返回 -1000）。
    2) get_smoke_pos() 已经在 z<0 时返回 None，彻底杜绝“落地后还有效”的假象。
    其余判定维持原逻辑。
    """
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    if v is None or theta is None:
        return -1000
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000

    # 起爆高度必须 >= 0
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    det_z = drop_pos[2] - 0.5 * G * det_delay**2
    if det_z < 0.0:
        return -1000

    # 间隔约束
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000

    # 时间窗
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1)
    min_t = max(det_time, 0)
    if min_t >= max_t - 1e-3:
        return 0.0

    effective_duration = 0.0
    for t in np.arange(min_t, max_t, TIME_STEP):
        m_pos = get_missile_pos(m_name, t)
        smoke_pos = get_smoke_pos(drone_name, drop_time, det_delay, t)
        if smoke_pos is None:
            continue
        # 必须“在导弹→目标点线段内部”命中（背面不计）
        all_intersect = True
        for sample in TRUE_TARGET["sample_points"]:
            if not segment_sphere_intersect(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                all_intersect = False
                break
        if all_intersect:
            effective_duration += TIME_STEP
    return effective_duration

# ============================ 3. 优化函数（保持原策略） ============================
def optimize_single_smoke(drone_name, m_name):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    bounds = [
        (v_min * 0.8, v_max * 1.2),
        (0, 2 * np.pi),
        (0, max_flight_time - 1),
        (0.1, 20)
    ]
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        mutation=0.8,
        recombination=0.9,
        popsize=60,
        maxiter=80,
        tol=1e-3,
        disp=False,
        polish=True
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

def optimize_drone_trajectory(drone_name, m_name, retry=0):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    v_candidates = np.linspace(v_min, v_max, 8)
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

            def objective(x):
                theta, drop_time, det_delay = x
                drone["direction"] = theta
                return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)

            result = differential_evolution(
                func=objective,
                bounds=[(0, 2*np.pi), (min_drop_time, max_drop_time), (0.1, 10)],
                mutation=0.7,
                recombination=0.8,
                popsize=50,
                maxiter=60,
                tol=1e-3,
                disp=False
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
        print(f"[{drone_name}] 优化失败，重试 {retry+1}/3...")
        return optimize_drone_trajectory(drone_name, m_name, retry+1)

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
        W = np.diag(weights) if len(weights) else np.eye(len(drop_points))
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
            ref_theta = np.arctan(k)
        except np.linalg.LinAlgError:
            ref_theta = np.mean([s["theta"] for s in best_smokes]) if best_smokes else 0

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
    return best_smokes

# ============================ 4. 任务分配与迭代优化（保持不变） ============================
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

def iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None

    all_smokes = []
    prev_total_time = 0
    stall_count = 0

    for iteration in range(max_iterations):
        print(f"\n===== 迭代 {iteration + 1}/{max_iterations} =====")
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"尚未找到有效解的无人机: {drones_without_solution}")

        if not drones_without_solution:
            print("所有无人机都已找到有效解，停止迭代")
            break

        assignments = assign_tasks(drones_without_solution)

        current_total_time = 0
        iteration_smokes = []
        optimized_this_iter = []

        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                print(f"正在优化无人机 {d_name} 干扰 {m_name}...")
                smokes = optimize_drone_trajectory(d_name, m_name)

                if smokes:
                    drone_smokes = [{**smoke, "drone": d_name} for smoke in smokes]
                    iteration_smokes.extend(drone_smokes)
                    current_total_time += sum([s["effective_time"] for s in smokes])
                    print(f"[{d_name}] 优化成功：{len(smokes)}枚烟幕弹，总遮蔽时长 {current_total_time:.2f}s")
                else:
                    print(f"[{d_name}] 仍未找到有效投放方案，将在下次迭代继续尝试")

                DRONES[d_name]["optimized"] = True
                optimized_this_iter.append(d_name)

        all_smokes.extend(iteration_smokes)

        total_effective_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        print(f"当前总遮蔽时长: {total_effective_time:.2f}s")
        print(f"本轮优化无人机: {optimized_this_iter}")
        print(f"已有有效解的无人机数量: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")

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

    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\n警告：达到最大迭代次数，以下无人机仍未找到有效解: {remaining_without_solution}")

    return all_smokes

# ============================ 5. 结果输出与可视化（不变） ============================
def save_result(smokes, filename="smoke_optimization_result.xlsx"):
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

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="真目标投影")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="真目标中心")

    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", label=f"{m_name}轨迹")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name}初始位置")

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
            for smoke in d_data["smokes"]:
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                    ax1.add_patch(circle)

    ax1.set_xlabel("X(m)")
    ax1.set_ylabel("Y(m)")
    ax1.set_title("无人机、导弹轨迹及烟幕起爆点")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors)
    ax2.set_xlabel("导弹编号")
    ax2.set_ylabel("总遮蔽时长(s)")
    ax2.set_title("各导弹总遮蔽时长")
    for m, t in missile_effect.items():
        ax2.text(m, t + 0.5, f"{t:.1f}s", ha="center")

    drone_smoke_count = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    ax3.bar(drone_smoke_count.keys(), drone_smoke_count.values(), color=drone_colors)
    ax3.set_xlabel("无人机编号")
    ax3.set_ylabel("烟幕弹数量")
    ax3.set_title("各无人机烟幕弹投放数量")
    for d, cnt in drone_smoke_count.items():
        ax3.text(d, cnt + 0.05, str(cnt), ha="center")

    effect_times = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effect_times, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("单烟幕弹遮蔽时长(s)")
    ax4.set_ylabel("烟幕弹数量")
    ax4.set_title("单烟幕弹遮蔽时长分布")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("smoke_optimization_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================ 主函数执行 ============================
if __name__ == "__main__":
    all_smokes = iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3)
    if all_smokes:
        result_df = save_result(all_smokes)
        visualize_result(all_smokes)
        print("\n" + "="*50)
        print("最终结果汇总：")
        print(f"总烟幕弹数量：{len(all_smokes)}")
        print(f"总遮蔽时长：{sum([s['effective_time'] for s in all_smokes]):.2f}s")
        print("\n各无人机投放详情：")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"{d_name}：{len(d_data['smokes'])}枚弹，总遮蔽时长{total:.2f}s")
            else:
                print(f"{d_name}：未找到有效投放方案")
        print("="*50)
    else:
        print("未找到有效的烟幕弹投放方案")
