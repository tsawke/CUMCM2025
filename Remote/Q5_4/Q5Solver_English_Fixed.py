import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend for headless operation
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment

# ============================ 1. Global Parameter Initialization (Optimized Search Range) ============================
# Real target parameters
TRUE_TARGET = {
    "r": 7,          # Cylinder radius
    "h": 10,         # Cylinder height
    "center": np.array([0, 200, 0]),  # Base center coordinates
    "sample_points": None  # Sampling points
}

# Missile parameters
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300  # Missile speed (m/s)
G = 9.8  # Gravitational acceleration (m/s²)
SMOKE_RADIUS = 10  # Smoke effective radius (m)
SMOKE_SINK_SPEED = 3  # Sinking speed after detonation (m/s)
SMOKE_EFFECTIVE_TIME = 20  # Effective duration after detonation (s)

# UAV parameters (keep optimized flag for tracking optimization status)
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
DROP_INTERVAL = 1  # Same UAV drop interval (s)
TIME_STEP = 0.1  # Time sampling step (s)

# Generate real target sampling points (unchanged)
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    # Base sampling
    samples.append(center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    # Top sampling
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    # Side sampling
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2*np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    TRUE_TARGET["sample_points"] = np.array(samples)

# Initialize missile parameters (unchanged)
def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED

# Initialize all parameters
generate_true_target_samples()
init_missiles()


# ============================ 2. Core Tool Functions (Add Z-coordinate Tolerance) ============================
# Missile position calculation (unchanged)
def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t

# UAV position calculation (unchanged)
def get_drone_pos(drone_name, t):
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + v_vec * t

# Smoke bomb position calculation (add Z<0 tolerance to avoid minor errors causing failure)
def get_smoke_pos(drone_name, drop_time, det_delay, t):
    drone = DRONES[drone_name]
    
    if t < drop_time:
        return None
    
    drop_pos = get_drone_pos(drone_name, drop_time)
    
    # After drop but before detonation
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                         drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t **2
        return np.array([x, y, max(z, 0.1)])  # Minimum Z = 0.1 to avoid invalid positions
    
    # After detonation
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None
    
    # Calculate detonation position directly
    delta_t_det = det_delay
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * delta_t_det** 2
    
    if det_z < 0:
        det_z = 0.1  # Error tolerance
    
    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])  # Minimum Z = 0.1

# Line segment and sphere intersection test (unchanged)
def segment_sphere_intersect(p1, p2, center, radius):
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)
    
    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8

# Single smoke bomb shielding duration calculation (relax drop interval constraint)
def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):  # Add tolerance
        return -1000
    
    # Check detonation point validity
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * G * delta_t_det **2
    if det_z < -0.5:  # Relax Z-axis constraint
        return -1000
    
    # Check drop interval (add 0.1s tolerance)
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    # Calculate effective duration (ensure reasonable time range)
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


# ============================ 3. Optimization Functions (Enhanced Differential Evolution Search Capability) ============================
# Single bomb optimization (increase population size and iteration count)
def optimize_single_smoke(drone_name, m_name):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    # Extended search boundaries
    bounds = [
        (v_min * 0.7, v_max * 1.3),        # Speed (30% extension)
        (0, 2 * np.pi),                    # Direction angle
        (0, max_flight_time - 0.5),        # Drop time (relaxed)
        (0.1, 25)                          # Detonation delay (expanded range)
    ]
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    # Enhanced differential evolution parameters: increase population and iteration count, adjust mutation coefficient
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        mutation=0.8,          # Increase mutation coefficient for enhanced exploration
        recombination=0.9,     # Increase crossover coefficient for faster convergence
        popsize=60,            # Population size from 40 to 60
        maxiter=80,            # Iteration count from 50 to 80
        tol=1e-3,              # Relax tolerance
        disp=False,
        polish=True            # Add local optimization
    )
    
    v_opt, theta_opt, drop_time_opt, det_delay_opt = result.x
    # Clip speed to reasonable range
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -result.fun
    return {
        "v": v_opt,
        "theta": theta_opt,
        "drop_time": drop_time_opt,
        "det_delay": det_delay_opt,
        "det_time": drop_time_opt + det_delay_opt,
        "det_pos": get_smoke_pos(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
        "effective_time": effective_time if effective_time > 1e-3 else 0,  # Filter tiny values
        "missile": m_name
    }

# UAV trajectory optimization (add speed candidates, allow second optimization)
def optimize_drone_trajectory(drone_name, m_name, retry=0):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    # Increase speed candidate count for better exploration
    v_candidates = np.linspace(v_min, v_max, 12)
    best_v = None
    best_smokes = []
    max_total_time = 0
    
    for v in v_candidates:
        drone["speed"] = v
        temp_smokes = []
        
        for i in range(max_smoke):
            # Fix index out of bounds issue: check if temp_smokes is empty instead of just checking i>0
            min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
            max_drop_time = MISSILES[m_name]["flight_time"] - 0.1  # Relax upper bound
            if min_drop_time >= max_drop_time - 1e-3:
                break
            
            def objective(x):
                theta, drop_time, det_delay = x
                drone["direction"] = theta
                return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
            
            # Enhanced differential evolution parameters
            result = differential_evolution(
                func=objective, 
                bounds=[(0, 2*np.pi), (min_drop_time, max_drop_time), (0.1, 10)],
                mutation=0.7, 
                recombination=0.8, 
                popsize=50,        # Population from 30 to 50
                maxiter=60,        # Iterations from 30 to 60
                disp=False
            )
            
            theta_opt, drop_time_opt, det_delay_opt = result.x
            drone["direction"] = theta_opt
            effective_time = calc_smoke_effective_time(drone_name, m_name, drop_time_opt, det_delay_opt)
            
            if effective_time > 0.1:  # Only keep effective solutions (>0.1s)
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
        
        # Update optimal solution
        total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
        if total_time > max_total_time:
            max_total_time = total_time
            best_v = v
            best_smokes = temp_smokes
    
    # If first optimization fails, increase retry count
    if not best_smokes and retry < 5:  # Increase to 5 retries
        print(f"[{drone_name}] Optimization failed, retrying {retry+1}/5...")
        return optimize_drone_trajectory(drone_name, m_name, retry+1)
    
    # Path fitting and fluctuation optimization
    if best_smokes:
        # Weighted line fitting
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
            drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effective_time"])
        drop_points = np.array(drop_points)
        weights = np.array(weights)
        
        # Fit direction angle
        X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
        W = np.diag(weights)
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
            ref_theta = np.arctan(k)
        except np.linalg.LinAlgError:
            ref_theta = np.mean([s["theta"] for s in best_smokes]) if best_smokes else 0
        
        # Fluctuation optimization
        for i, smoke in enumerate(best_smokes):
            theta_candidates = [ref_theta - np.pi/24, ref_theta, ref_theta + np.pi/24]
            drop_candidates = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]
            
            best_effect = smoke["effective_time"]
            best_params = (smoke["theta"], smoke["drop_time"])
            
            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    # Safety check: ensure previous smoke bomb exists before checking interval
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


# ============================ 4. Task Assignment and Iterative Optimization (Core Fix: Stop Condition) ============================
# Task assignment (optimize cost matrix, add missile urgency weight)
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
            
            # Cost 1: UAV to missile initial position flight time
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v
            
            # Cost 2: Missile remaining flight time (urgency, shorter time = lower cost)
            cost2 = 1000 / (m_flight_time + 1)  # Invert urgency
            
            # Cost 3: Speed matching (UAV speed vs missile speed difference)
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 100
            
            cost_matrix[i][j] = cost1 + cost2 + cost3
    
    # Hungarian algorithm assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    for i, j in zip(row_ind, col_ind):
        assignments[missile_list[j]].append(unoptimized_drones[i])
    
    # Assign unassigned UAVs to lowest cost missiles
    assigned_drones = set(row_ind)
    for i in range(n_drones):
        if i not in assigned_drones:
            min_cost_j = np.argmin(cost_matrix[i])
            assignments[missile_list[min_cost_j]].append(unoptimized_drones[i])
    
    return assignments

# Iterative optimization main function (core modification: stop condition changed to all UAVs having solutions)
def iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    # Reset UAV states
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0  # Track consecutive non-improvement iterations
    
    for iteration in range(max_iterations):
        print(f"\n===== Iteration {iteration + 1}/{max_iterations} =====")
        
        # 1. Get UAVs that haven't found valid solutions yet
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"UAVs without valid solutions: {drones_without_solution}")
        
        # 2. Check if all UAVs have solutions, if so stop iteration
        if not drones_without_solution:
            print("All UAVs have found valid solutions, stopping iteration")
            break
        
        # 3. Task assignment (assign missiles to UAVs without solutions)
        assignments = assign_tasks(drones_without_solution)
        
        # 4. Optimize UAVs without valid solutions
        current_total_time = 0
        iteration_smokes = []
        optimized_this_iter = []
        
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                # Only optimize UAVs that don't have valid solutions yet
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                    
                print(f"Optimizing UAV {d_name} to interfere with {m_name}...")
                smokes = optimize_drone_trajectory(d_name, m_name)
                
                # If solution found, update information
                if smokes:
                    # Save current UAV's smoke data
                    drone_smokes = [{**smoke, "drone": d_name} for smoke in smokes]
                    iteration_smokes.extend(drone_smokes)
                    current_total_time += sum([s["effective_time"] for s in smokes])
                    print(f"[{d_name}] Optimization successful: {len(smokes)} smoke bombs, total shielding duration {current_total_time:.2f}s")
                else:
                    print(f"[{d_name}] Still no valid deployment strategy found, will continue trying in next iteration")
                
                # Mark as optimization attempted
                DRONES[d_name]["optimized"] = True
                optimized_this_iter.append(d_name)
        
        # 5. Update global smoke data
        all_smokes.extend(iteration_smokes)
        
        # Calculate total shielding time for all UAVs
        total_effective_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        print(f"Current total shielding duration: {total_effective_time:.2f}s")
        print(f"UAVs optimized this round: {optimized_this_iter}")
        print(f"UAVs with valid solutions count: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        
        # 6. Check improvement amount
        improvement = total_effective_time - prev_total_time
        print(f"Improvement from previous round: {improvement:.2f}s")
        
        if improvement < improvement_threshold:
            stall_count += 1
            print(f"Consecutive non-improvement count: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                # Check if there are still UAVs without solutions
                if drones_without_solution:
                    print(f"Consecutive {max_stall_iter} rounds without effective improvement, but UAVs still need solutions, continuing optimization...")
                    stall_count = max_stall_iter - 1  # Reset counter but keep some count to avoid infinite loop
                else:
                    print(f"Consecutive {max_stall_iter} rounds without effective improvement, stopping iteration")
                    break
        else:
            stall_count = 0  # Reset consecutive non-improvement counter
        
        prev_total_time = total_effective_time
    
    # Check if all UAVs have solutions
    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\nWarning: Maximum iteration count reached, UAVs still without valid solutions: {remaining_without_solution}")
    
    return all_smokes


# ============================ 5. Result Output and Visualization (English Labels) ============================
def save_result(smokes, filename="smoke_optimization_result.xlsx"):
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0,0,0])
        data.append({
            "Smoke_Bomb_Number": f"S{i}",
            "UAV_Number": smoke["drone"],
            "Speed_m_per_s": round(smoke["v"], 2),
            "Direction_degrees": round(np.degrees(smoke["theta"]), 2),
            "Drop_Time_s": round(smoke["drop_time"], 2),
            "Detonation_Delay_s": round(smoke["det_delay"], 2),
            "Detonation_Time_s": round(smoke["det_time"], 2),
            "Detonation_Point_X_m": round(det_pos[0], 2),
            "Detonation_Point_Y_m": round(det_pos[1], 2),
            "Detonation_Point_Z_m": round(det_pos[2], 2),
            "Target_Missile": smoke["missile"],
            "Effective_Shielding_Duration_s": round(smoke["effective_time"], 2)
        })
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, engine="openpyxl")
    print(f"Results saved to {filename}")
    return df

def visualize_result(smokes):
    if not smokes:
        print("No valid data for visualization")
        return
    
    # Set English labels
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Draw real target, missile trajectories, UAV trajectories
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="Real Target Projection")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="Real Target Center")
    
    # Missile trajectories
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_pos(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", label=f"{m_name} Trajectory")
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name} Initial Position")
    
    # UAV trajectories and smoke clouds
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue
        # UAV trajectory
        last_smoke = d_data["smokes"][-1] if d_data["smokes"] else None
        if last_smoke:
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            v_vec = np.array([d_data["speed"] * np.cos(d_data["direction"]), d_data["speed"] * np.sin(d_data["direction"]), 0]) if d_data["speed"] and d_data["direction"] else np.array([0, 0, 0])
            pos_list = [d_data["init_pos"] + v_vec * t for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{drone_colors[i]}-", label=f"{d_name} Trajectory")
            ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^", label=f"{d_name} Initial Position")
            
            # Smoke detonation points
            for smoke in d_data["smokes"]:
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                    ax1.add_patch(circle)
    
    ax1.set_xlabel("X Coordinate (m)")
    ax1.set_ylabel("Y Coordinate (m)")
    ax1.set_title("UAV, Missile Trajectories and Smoke Detonation Points")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Shielding duration for each missile
    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors)
    ax2.set_xlabel("Missile ID")
    ax2.set_ylabel("Total Shielding Duration (s)")
    ax2.set_title("Total Shielding Duration per Missile")
    for m, t in missile_effect.items():
        ax2.text(m, t + 0.5, f"{t:.1f}s", ha="center")
    
    # 3. Number of smoke bombs per UAV
    drone_smoke_count = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    ax3.bar(drone_smoke_count.keys(), drone_smoke_count.values(), color=drone_colors)
    ax3.set_xlabel("UAV ID")
    ax3.set_ylabel("Number of Smoke Bombs")
    ax3.set_title("Smoke Bomb Deployment Count per UAV")
    for d, cnt in drone_smoke_count.items():
        ax3.text(d, cnt + 0.05, str(cnt), ha="center")
    
    # 4. Shielding duration distribution
    effect_times = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effect_times, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Single Smoke Bomb Shielding Duration (s)")
    ax4.set_ylabel("Number of Smoke Bombs")
    ax4.set_title("Single Smoke Bomb Shielding Duration Distribution")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("smoke_optimization_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

# Save comprehensive optimization report to txt file
def save_comprehensive_report(smokes, filename="optimization_comprehensive_report.txt"):
    """Save all optimization information and optimal solution to txt file as backup"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION - COMPREHENSIVE REPORT\n")
        f.write("Problem 5: 5 UAVs (FY1-FY5) vs 3 Missiles (M1, M2, M3)\n")
        f.write("Algorithm: Enhanced Iterative Differential Evolution\n")
        f.write("="*100 + "\n")
        f.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Smoke Bombs Deployed: {len(smokes)}\n")
        f.write(f"Total System Effectiveness: {sum([s['effective_time'] for s in smokes]):.6f} seconds\n")
        f.write("\n")
        
        # Problem configuration
        f.write("PROBLEM CONFIGURATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Real Target: Cylindrical target at {TRUE_TARGET['center']} with radius {TRUE_TARGET['r']}m, height {TRUE_TARGET['h']}m\n")
        f.write(f"Fake Target: Point at origin [0, 0, 0] (missile destination)\n")
        f.write(f"Missiles:\n")
        for m_name, m_data in MISSILES.items():
            f.write(f"  {m_name}: Initial position {m_data['init_pos']}, flight time {m_data['flight_time']:.2f}s\n")
        f.write(f"UAVs:\n")
        for d_name, d_data in DRONES.items():
            f.write(f"  {d_name}: Initial position {d_data['init_pos']}, speed range {d_data['speed_range']} m/s\n")
        f.write(f"Smoke Parameters: Radius {SMOKE_RADIUS}m, sink speed {SMOKE_SINK_SPEED}m/s, duration {SMOKE_EFFECTIVE_TIME}s\n")
        f.write("\n")
        
        # Detailed deployment strategy
        f.write("DETAILED DEPLOYMENT STRATEGY:\n")
        f.write("-" * 50 + "\n")
        
        for d_name, d_data in DRONES.items():
            f.write(f"UAV {d_name}:\n")
            if d_data["smokes"]:
                uav_total = sum([s["effective_time"] for s in d_data["smokes"]])
                f.write(f"  Status: ACTIVE DEPLOYMENT\n")
                f.write(f"  Optimized Speed: {d_data['speed']:.3f} m/s\n")
                f.write(f"  Optimized Direction: {d_data['direction']:.6f} rad ({np.degrees(d_data['direction']):.2f}°)\n")
                f.write(f"  Number of Smoke Bombs: {len(d_data['smokes'])}\n")
                f.write(f"  Total Effectiveness: {uav_total:.6f} s\n")
                
                for j, smoke in enumerate(d_data["smokes"], 1):
                    det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0,0,0])
                    f.write(f"    Smoke Bomb {j}:\n")
                    f.write(f"      Drop Time: {smoke['drop_time']:.3f}s\n")
                    f.write(f"      Detonation Delay: {smoke['det_delay']:.3f}s\n")
                    f.write(f"      Detonation Position: ({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f}) m\n")
                    f.write(f"      Effectiveness: {smoke['effective_time']:.6f}s\n")
                    f.write(f"      Target Missile: {smoke['missile']}\n")
            else:
                f.write(f"  Status: NO VALID DEPLOYMENT FOUND\n")
            f.write("\n")
        
        # Performance analysis
        total_effectiveness = sum([s["effective_time"] for s in smokes])
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total System Effectiveness: {total_effectiveness:.6f} seconds\n")
        f.write(f"Average Effectiveness per Bomb: {total_effectiveness/len(smokes):.6f} seconds\n")
        
        # Missile coverage analysis
        missile_coverage = {m: 0 for m in MISSILES.keys()}
        for smoke in smokes:
            missile_coverage[smoke["missile"]] += smoke["effective_time"]
        
        f.write(f"\nMissile Coverage Analysis:\n")
        for m_name, coverage_time in missile_coverage.items():
            flight_time = MISSILES[m_name]["flight_time"]
            coverage_percentage = coverage_time / flight_time * 100
            f.write(f"  {m_name}: {coverage_time:.3f}s shielding ({coverage_percentage:.1f}% of flight time)\n")
        
        f.write("\n")
        f.write("="*100 + "\n")
        f.write("END OF COMPREHENSIVE REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"Comprehensive report saved to {filename}")
    return filename


# ============================ Main Function Execution ============================
if __name__ == "__main__":
    # Increase maximum iteration count to ensure sufficient opportunity for all UAVs to find solutions
    all_smokes = iterative_optimization(max_iterations=30, improvement_threshold=0.2, max_stall_iter=5)
    
    if all_smokes:
        result_df = save_result(all_smokes)
        visualize_result(all_smokes)
        
        # Save comprehensive report
        comprehensive_report_file = save_comprehensive_report(all_smokes)
        
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY:")
        print(f"Total smoke bomb count: {len(all_smokes)}")
        print(f"Total shielding duration: {sum([s['effective_time'] for s in all_smokes]):.2f}s")
        print("\nUAV deployment details:")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"{d_name}: {len(d_data['smokes'])} bombs, total shielding duration {total:.2f}s")
            else:
                print(f"{d_name}: No valid deployment strategy found")
        print("="*50)
    else:
        print("No valid smoke bomb deployment strategy found")
