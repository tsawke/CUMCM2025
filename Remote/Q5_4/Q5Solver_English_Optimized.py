#!/usr/bin/env python3
"""
English Version: Multi-UAV Smoke Bomb Deployment Optimization
Enhanced iterative optimization algorithm with convergence tracking
Problem 5: 5 UAVs, up to 3 smoke bombs each, intercept 3 missiles M1/M2/M3
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for server environment
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, linear_sum_assignment
import time
import warnings
warnings.filterwarnings('ignore')

# ========================== 1. GLOBAL PARAMETERS INITIALIZATION ==========================
# Real target parameters
TRUE_TARGET = {
    "radius": 7,          # Cylinder radius (m)
    "height": 10,         # Cylinder height (m)  
    "center": np.array([0, 200, 0]),  # Base center coordinates (m)
    "sample_points": None  # Sampling points
}

# Missile parameters
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "direction": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "direction": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "direction": None, "flight_time": None}
}
MISSILE_SPEED = 300  # Missile speed (m/s)
GRAVITY = 9.8  # Gravitational acceleration (m/s²)
SMOKE_RADIUS = 10  # Smoke effective radius (m)
SMOKE_SINK_SPEED = 3  # Sinking speed after detonation (m/s)
SMOKE_EFFECTIVE_TIME = 20  # Effective duration after detonation (s)

# UAV parameters
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
DROP_INTERVAL = 1  # Drop interval between bombs from same UAV (s)
TIME_STEP = 0.1  # Time sampling step (s)

# Optimization tracking
OPTIMIZATION_HISTORY = {
    "iteration_times": [],
    "total_durations": [],
    "improvements": [],
    "drone_solutions": []
}

# Generate real target sampling points
def generate_true_target_samples():
    """Generate sampling points on the cylindrical target surface and interior"""
    samples = []
    r, h, center = TRUE_TARGET["radius"], TRUE_TARGET["height"], TRUE_TARGET["center"]
    
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
    
    # Side surface sampling
    for z in np.linspace(center[2], top_center[2], 5):
        for theta in np.linspace(0, 2*np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    
    TRUE_TARGET["sample_points"] = np.array(samples)
    print(f"Generated {len(samples)} target sampling points")

# Initialize missile parameters
def init_missiles():
    """Initialize missile direction vectors and flight times"""
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / np.linalg.norm(init_pos)  # Direction towards origin (fake target)
        m_data["direction"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED
        print(f"Missile {m_name}: flight time = {m_data['flight_time']:.2f}s")

# Initialize all parameters
generate_true_target_samples()
init_missiles()


# ========================== 2. CORE UTILITY FUNCTIONS ==========================
def get_missile_position(m_name, t):
    """Calculate missile position at time t"""
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["direction"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["direction"] * t

def get_drone_position(drone_name, t):
    """Calculate UAV position at time t"""
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                               drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + velocity_vector * t

def get_smoke_position(drone_name, drop_time, det_delay, t):
    """Calculate smoke bomb position at time t (with Z-coordinate tolerance)"""
    drone = DRONES[drone_name]
    
    if t < drop_time:
        return None
    
    drop_pos = get_drone_position(drone_name, drop_time)
    
    # Before detonation (free fall trajectory)
    if t < drop_time + det_delay:
        delta_t = t - drop_time
        velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                                   drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_pos[0] + velocity_vector[0] * delta_t
        y = drop_pos[1] + velocity_vector[1] * delta_t
        z = drop_pos[2] - 0.5 * GRAVITY * delta_t**2
        return np.array([x, y, max(z, 0.1)])  # Minimum Z = 0.1 to avoid invalid positions
    
    # After detonation (sinking smoke cloud)
    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None
    
    # Calculate detonation position
    delta_t_det = det_delay
    velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                               drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_pos[0] + velocity_vector[0] * delta_t_det
    det_y = drop_pos[1] + velocity_vector[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * GRAVITY * delta_t_det**2
    
    if det_z < 0:
        det_z = 0.1  # Error tolerance
    
    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])  # Minimum Z = 0.1

def segment_sphere_intersect(p1, p2, center, radius):
    """Check if line segment intersects with sphere"""
    vec_p = p2 - p1
    vec_c = center - p1
    t = np.dot(vec_c, vec_p) / (np.dot(vec_p, vec_p) + 1e-8)
    
    if 0 <= t <= 1:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8

def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    """Calculate effective shielding time for single smoke bomb"""
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    # Speed constraint check (with tolerance)
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000
    
    # Check detonation point validity
    det_time = drop_time + det_delay
    drop_pos = get_drone_position(drone_name, drop_time)
    delta_t_det = det_delay
    det_z = drop_pos[2] - 0.5 * GRAVITY * delta_t_det**2
    if det_z < -0.5:  # Relaxed Z-axis constraint
        return -1000
    
    # Check drop interval constraint (with 0.1s tolerance)
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    # Calculate effective duration
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1)
    min_t = max(det_time, 0)
    if min_t >= max_t - 1e-3:
        return 0
    
    effective_duration = 0
    for t in np.arange(min_t, max_t, TIME_STEP):
        m_pos = get_missile_position(m_name, t)
        smoke_pos = get_smoke_position(drone_name, drop_time, det_delay, t)
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


# ========================== 3. OPTIMIZATION FUNCTIONS ==========================
def optimize_single_smoke(drone_name, m_name):
    """Optimize single smoke bomb deployment using differential evolution"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_flight_time = MISSILES[m_name]["flight_time"]
    
    # Extended search boundaries
    bounds = [
        (v_min * 0.8, v_max * 1.2),        # Speed (extended by 20%)
        (0, 2 * np.pi),                    # Direction angle
        (0, max_flight_time - 1),          # Drop time
        (0.1, 20)                          # Detonation delay (0.1 to 20s)
    ]
    
    def objective(x):
        v, theta, drop_time, det_delay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay)
    
    # Enhanced differential evolution parameters
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        mutation=0.8,          # Increased mutation for better exploration
        recombination=0.9,     # Increased recombination for faster convergence
        popsize=60,            # Population size increased from 40 to 60
        maxiter=80,            # Iterations increased from 50 to 80
        tol=1e-3,              # Relaxed tolerance
        disp=False,
        polish=True            # Add local optimization
    )
    
    v_opt, theta_opt, drop_time_opt, det_delay_opt = result.x
    # Clip speed to valid range
    v_opt = np.clip(v_opt, v_min, v_max)
    effective_time = -result.fun
    
    return {
        "speed": v_opt,
        "direction": theta_opt,
        "drop_time": drop_time_opt,
        "det_delay": det_delay_opt,
        "det_time": drop_time_opt + det_delay_opt,
        "det_pos": get_smoke_position(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
        "effective_time": effective_time if effective_time > 1e-3 else 0,  # Filter small values
        "missile": m_name
    }

def optimize_drone_trajectory(drone_name, m_name, retry=0):
    """Optimize UAV trajectory with multiple smoke bombs"""
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]
    
    # Increased speed candidates (from 5 to 8)
    v_candidates = np.linspace(v_min, v_max, 8)
    best_v = None
    best_smokes = []
    max_total_time = 0
    
    for v in v_candidates:
        drone["speed"] = v
        temp_smokes = []
        
        for i in range(max_smoke):
            # Fix index bounds issue: check if temp_smokes is empty
            min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0
            max_drop_time = MISSILES[m_name]["flight_time"] - 0.1  # Relaxed upper bound
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
                popsize=50,        # Population increased from 30 to 50
                maxiter=60,        # Iterations increased from 30 to 60
                disp=False
            )
            
            theta_opt, drop_time_opt, det_delay_opt = result.x
            drone["direction"] = theta_opt
            effective_time = calc_smoke_effective_time(drone_name, m_name, drop_time_opt, det_delay_opt)
            
            if effective_time > 0.1:  # Only keep valid solutions (>0.1s)
                smoke = {
                    "speed": v,
                    "direction": theta_opt,
                    "drop_time": drop_time_opt,
                    "det_delay": det_delay_opt,
                    "det_time": drop_time_opt + det_delay_opt,
                    "det_pos": get_smoke_position(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
                    "effective_time": effective_time,
                    "missile": m_name
                }
                temp_smokes.append(smoke)
        
        # Update best solution
        total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0
        if total_time > max_total_time:
            max_total_time = total_time
            best_v = v
            best_smokes = temp_smokes
    
    # Retry if first optimization fails
    if not best_smokes and retry < 3:  # Increased to 3 retries
        print(f"[{drone_name}] Optimization failed, retrying {retry+1}/3...")
        return optimize_drone_trajectory(drone_name, m_name, retry+1)
    
    # Path fitting and fluctuation optimization
    if best_smokes:
        # Weighted line fitting
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["speed"] * np.cos(smoke["direction"]), 
                             smoke["speed"] * np.sin(smoke["direction"]), 0])
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
            ref_theta = np.mean([s["direction"] for s in best_smokes]) if best_smokes else 0
        
        # Fluctuation optimization
        for i, smoke in enumerate(best_smokes):
            theta_candidates = [ref_theta - np.pi/24, ref_theta, ref_theta + np.pi/24]
            drop_candidates = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]
            
            best_effect = smoke["effective_time"]
            best_params = (smoke["direction"], smoke["drop_time"])
            
            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    # Safety check: ensure previous smoke exists before checking interval
                    prev_drop_time = best_smokes[i-1]["drop_time"] if i > 0 and i-1 < len(best_smokes) else -np.inf
                    if drop_time < prev_drop_time + DROP_INTERVAL - 0.1:
                        continue
                    drone["direction"] = theta
                    effect = calc_smoke_effective_time(drone_name, m_name, drop_time, smoke["det_delay"])
                    if effect > best_effect:
                        best_effect = effect
                        best_params = (theta, drop_time)
            
            smoke["direction"], smoke["drop_time"] = best_params
            smoke["det_time"] = smoke["drop_time"] + smoke["det_delay"]
            smoke["det_pos"] = get_smoke_position(drone_name, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
            smoke["effective_time"] = best_effect
    
    drone["speed"] = best_v
    drone["direction"] = ref_theta if best_smokes else None
    drone["smokes"] = best_smokes
    return best_smokes

def assign_tasks(unoptimized_drones=None):
    """Task assignment using Hungarian algorithm with optimized cost matrix"""
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
            cost2 = 1000 / (m_flight_time + 1)  # Inverted urgency
            
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

def iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    """Main iterative optimization function with convergence tracking"""
    print("=== Starting Enhanced Iterative Optimization ===")
    
    # Reset UAV states
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    # Reset optimization history
    OPTIMIZATION_HISTORY["iteration_times"] = []
    OPTIMIZATION_HISTORY["total_durations"] = []
    OPTIMIZATION_HISTORY["improvements"] = []
    OPTIMIZATION_HISTORY["drone_solutions"] = []
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0
    
    for iteration in range(max_iterations):
        iteration_start = time.time()
        print(f"\n===== Iteration {iteration + 1}/{max_iterations} =====")
        
        # 1. Get UAVs without valid solutions
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"UAVs without solutions: {drones_without_solution}")
        
        # 2. Check if all UAVs have solutions
        if not drones_without_solution:
            print("All UAVs have found valid solutions, stopping iteration")
            break
        
        # 3. Task assignment
        assignments = assign_tasks(drones_without_solution)
        print(f"Task assignments: {assignments}")
        
        # 4. Optimize UAVs without valid solutions
        current_total_time = 0
        iteration_smokes = []
        optimized_this_iter = []
        
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                # Only optimize UAVs that don't have valid solutions yet
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                    
                print(f"Optimizing UAV {d_name} for missile {m_name}...")
                smokes = optimize_drone_trajectory(d_name, m_name)
                
                # If solution found, update information
                if smokes:
                    # Save current UAV's smoke data
                    drone_smokes = [{{**smoke, "drone": d_name}} for smoke in smokes]
                    iteration_smokes.extend(drone_smokes)
                    current_total_time += sum([s["effective_time"] for s in smokes])
                    print(f"[{d_name}] Optimization successful: {len(smokes)} smoke bombs, total shielding time {current_total_time:.2f}s")
                else:
                    print(f"[{d_name}] Still no valid deployment strategy found, will retry in next iteration")
                
                # Mark as optimization attempted
                DRONES[d_name]["optimized"] = True
                optimized_this_iter.append(d_name)
        
        # 5. Update global smoke data
        all_smokes.extend(iteration_smokes)
        
        # Calculate total shielding time for all UAVs
        total_effective_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        improvement = total_effective_time - prev_total_time
        
        # Record optimization history
        iteration_time = time.time() - iteration_start
        OPTIMIZATION_HISTORY["iteration_times"].append(iteration_time)
        OPTIMIZATION_HISTORY["total_durations"].append(total_effective_time)
        OPTIMIZATION_HISTORY["improvements"].append(improvement)
        OPTIMIZATION_HISTORY["drone_solutions"].append(len(DRONES) - len(drones_without_solution))
        
        print(f"Current total shielding time: {total_effective_time:.2f}s")
        print(f"Improvement from previous iteration: {improvement:.2f}s")
        print(f"UAVs optimized this round: {optimized_this_iter}")
        print(f"UAVs with valid solutions: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        print(f"Iteration time: {iteration_time:.2f}s")
        
        # 6. Check improvement
        if improvement < improvement_threshold:
            stall_count += 1
            print(f"Consecutive iterations without significant improvement: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                # Check if there are still UAVs without solutions
                if drones_without_solution:
                    print(f"Consecutive {max_stall_iter} iterations without improvement, but UAVs still need solutions, continuing...")
                    stall_count = max_stall_iter - 1  # Reset counter but keep some, avoid infinite loop
                else:
                    print(f"Consecutive {max_stall_iter} iterations without improvement, stopping iteration")
                    break
        else:
            stall_count = 0  # Reset consecutive no-improvement counter
        
        prev_total_time = total_effective_time
    
    # Check if all UAVs have solutions
    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\nWarning: Maximum iterations reached, following UAVs still have no valid solutions: {remaining_without_solution}")
    
    print(f"\nOptimization completed! Final total shielding time: {total_effective_time:.2f}s")
    return all_smokes
    
    # Reset UAV states
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None
    
    # Reset optimization history
    OPTIMIZATION_HISTORY["iteration_times"] = []
    OPTIMIZATION_HISTORY["total_durations"] = []
    OPTIMIZATION_HISTORY["improvements"] = []
    OPTIMIZATION_HISTORY["drone_solutions"] = []
    
    all_smokes = []
    prev_total_time = 0
    stall_count = 0
    
    for iteration in range(max_iterations):
        iteration_start = time.time()
        print(f"\n===== Iteration {iteration + 1}/{max_iterations} =====")
        
        # 1. Get UAVs without valid solutions
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"UAVs without solutions: {drones_without_solution}")
        
        # 2. Check if all UAVs have solutions
        if not drones_without_solution:
            print("All UAVs have found valid solutions, stopping iteration")
            break
        
        # 3. Task assignment
        assignments = assign_tasks(drones_without_solution)
        print(f"Task assignments: {assignments}")
        
        # 4. Optimize UAVs without valid solutions
        current_total_time = 0
        iteration_smokes = []
        optimized_this_iter = []
        
        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                # Only optimize UAVs that don't have valid solutions yet
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                    
                print(f"Optimizing UAV {d_name} for missile {m_name}...")
                smokes = optimize_drone_trajectory(d_name, m_name)
                
                # If solution found, update information
                if smokes:
                    # Save current UAV's smoke data
                    drone_smokes = [{**smoke, "drone": d_name} for smoke in smokes]
                    iteration_smokes.extend(drone_smokes)
                    current_total_time += sum([s["effective_time"] for s in smokes])
                    print(f"[{d_name}] Optimization successful: {len(smokes)} smoke bombs, total shielding time {current_total_time:.2f}s")
                else:
                    print(f"[{d_name}] Still no valid deployment strategy found, will retry in next iteration")
                
                # Mark as optimization attempted
                DRONES[d_name]["optimized"] = True
                optimized_this_iter.append(d_name)
        
        # 5. Update global smoke data
        all_smokes.extend(iteration_smokes)
        
        # Calculate total shielding time for all UAVs
        total_effective_time = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        improvement = total_effective_time - prev_total_time
        
        # Record optimization history
        iteration_time = time.time() - iteration_start
        OPTIMIZATION_HISTORY["iteration_times"].append(iteration_time)
        OPTIMIZATION_HISTORY["total_durations"].append(total_effective_time)
        OPTIMIZATION_HISTORY["improvements"].append(improvement)
        OPTIMIZATION_HISTORY["drone_solutions"].append(len(DRONES) - len(drones_without_solution))
        
        print(f"Current total shielding time: {total_effective_time:.2f}s")
        print(f"Improvement from previous iteration: {improvement:.2f}s")
        print(f"UAVs optimized this round: {optimized_this_iter}")
        print(f"UAVs with valid solutions: {len(DRONES) - len(drones_without_solution)}/{len(DRONES)}")
        print(f"Iteration time: {iteration_time:.2f}s")
        
        # 6. Check improvement
        if improvement < improvement_threshold:
            stall_count += 1
            print(f"Consecutive iterations without significant improvement: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                # Check if there are still UAVs without solutions
                if drones_without_solution:
                    print(f"Consecutive {max_stall_iter} iterations without improvement, but UAVs still need solutions, continuing...")
                    stall_count = max_stall_iter - 1  # Reset counter but keep some, avoid infinite loop
                else:
                    print(f"Consecutive {max_stall_iter} iterations without improvement, stopping iteration")
                    break
        else:
            stall_count = 0  # Reset consecutive no-improvement counter
        
        prev_total_time = total_effective_time
    
    # Check if all UAVs have solutions
    remaining_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining_without_solution:
        print(f"\nWarning: Maximum iterations reached, following UAVs still have no valid solutions: {remaining_without_solution}")
    
    print(f"\nOptimization completed! Final total shielding time: {total_effective_time:.2f}s")
    return all_smokes


# ========================== 4. RESULT OUTPUT AND VISUALIZATION ==========================
def save_result_standard_format(smokes, filename="result3.xlsx"):
    """Save results in standard format matching result3.xlsx template"""
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0,0,0])
        data.append({
            "Serial_Number": f"S{i}",
            "UAV_Number": smoke["drone"],
            "Speed_m_per_s": round(smoke["speed"], 2),
            "Direction_degrees": round(np.degrees(smoke["direction"]), 2),
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

def save_comprehensive_optimization_report(smokes, filename="optimization_comprehensive_report.txt"):
    """Save comprehensive optimization results and analysis to txt file"""
    
    with open(filename, 'w', encoding='utf-8') as file:
        # Header
        file.write("=" * 100 + "\n")
        file.write("COMPREHENSIVE MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION REPORT\n")
        file.write("Problem 5: 5 UAVs vs 3 Missiles (M1, M2, M3)\n")
        file.write("Algorithm: Enhanced Iterative Differential Evolution\n")
        file.write("=" * 100 + "\n")
        file.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Total Computation Time: {sum(OPTIMIZATION_HISTORY['iteration_times']):.3f} seconds\n")
        file.write(f"Number of Iterations: {len(OPTIMIZATION_HISTORY['total_durations'])}\n")
        file.write("\n")
        
        # Problem setup
        file.write("PROBLEM SETUP:\n")
        file.write("-" * 50 + "\n")
        file.write(f"Real Target: Cylinder at {TRUE_TARGET['center']} with radius {TRUE_TARGET['radius']}m, height {TRUE_TARGET['height']}m\n")
        file.write(f"Missiles: M1{MISSILES['M1']['init_pos']}, M2{MISSILES['M2']['init_pos']}, M3{MISSILES['M3']['init_pos']}\n")
        file.write(f"Missile Speed: {MISSILE_SPEED} m/s\n")
        file.write(f"UAVs: FY1{DRONES['FY1']['init_pos']}, FY2{DRONES['FY2']['init_pos']}, FY3{DRONES['FY3']['init_pos']}, FY4{DRONES['FY4']['init_pos']}, FY5{DRONES['FY5']['init_pos']}\n")
        file.write(f"UAV Speed Range: {DRONES['FY1']['speed_range']} m/s\n")
        file.write(f"Max Smoke Bombs per UAV: {DRONES['FY1']['max_smoke']}\n")
        file.write(f"Smoke Parameters: Radius {SMOKE_RADIUS}m, sink speed {SMOKE_SINK_SPEED}m/s, duration {SMOKE_EFFECTIVE_TIME}s\n")
        file.write("\n")
        
        # Optimization convergence history
        file.write("OPTIMIZATION CONVERGENCE HISTORY:\n")
        file.write("-" * 50 + "\n")
        file.write("Iteration | Total Duration (s) | Improvement (s) | UAVs Solved | Time (s)\n")
        file.write("-" * 70 + "\n")
        for i, (duration, improvement, solved, iter_time) in enumerate(zip(
            OPTIMIZATION_HISTORY["total_durations"],
            OPTIMIZATION_HISTORY["improvements"], 
            OPTIMIZATION_HISTORY["drone_solutions"],
            OPTIMIZATION_HISTORY["iteration_times"]
        )):
            file.write(f"{i+1:8d} | {duration:14.6f} | {improvement:12.6f} | {solved:10d} | {iter_time:7.2f}\n")
        file.write("\n")
        
        # Final solution summary
        total_duration = sum([s["effective_time"] for s in smokes])
        file.write("FINAL OPTIMIZATION SOLUTION:\n")
        file.write("-" * 50 + "\n")
        file.write(f"Total Smoke Bombs Deployed: {len(smokes)}\n")
        file.write(f"Total Effective Shielding Duration: {total_duration:.6f} seconds\n")
        file.write(f"Average Effectiveness per Bomb: {total_duration/len(smokes):.6f} seconds\n")
        file.write("\n")
        
        # UAV-wise detailed results
        file.write("DETAILED UAV DEPLOYMENT PLANS:\n")
        file.write("-" * 50 + "\n")
        
        for d_name, d_data in DRONES.items():
            file.write(f"UAV {d_name}:\n")
            if d_data["smokes"]:
                uav_total = sum([s["effective_time"] for s in d_data["smokes"]])
                file.write(f"  Status: DEPLOYED\n")
                file.write(f"  Fixed Speed: {d_data['speed']:.3f} m/s\n")
                file.write(f"  Fixed Heading: {d_data['direction']:.6f} rad ({np.degrees(d_data['direction']):.2f}°)\n")
                file.write(f"  Total Smoke Bombs: {len(d_data['smokes'])}\n")
                file.write(f"  Total Shielding Duration: {uav_total:.6f} s\n")
                file.write(f"  Average per Bomb: {uav_total/len(d_data['smokes']):.6f} s\n")
                
                for bomb_id, smoke in enumerate(d_data["smokes"], 1):
                    det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0,0,0])
                    file.write(f"    Bomb {bomb_id}: Drop={smoke['drop_time']:.3f}s, Delay={smoke['det_delay']:.3f}s, "
                              f"Detonation=({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f}), "
                              f"Effect={smoke['effective_time']:.6f}s, Target={smoke['missile']}\n")
            else:
                file.write(f"  Status: NO VALID SOLUTION FOUND\n")
            file.write("\n")
        
        # Missile-wise coverage analysis
        file.write("MISSILE COVERAGE ANALYSIS:\n")
        file.write("-" * 50 + "\n")
        
        missile_coverage = {m: {"bombs": 0, "total_time": 0, "coverage_rate": 0} for m in MISSILES.keys()}
        for smoke in smokes:
            m = smoke["missile"]
            missile_coverage[m]["bombs"] += 1
            missile_coverage[m]["total_time"] += smoke["effective_time"]
        
        for m_name, m_data in MISSILES.items():
            coverage = missile_coverage[m_name]
            coverage_rate = coverage["total_time"] / m_data["flight_time"] * 100
            file.write(f"Missile {m_name}: {coverage['bombs']} bombs, {coverage['total_time']:.6f}s shielding, {coverage_rate:.2f}% coverage\n")
        file.write("\n")
        
        # Performance statistics
        if smokes:
            effect_times = [s["effective_time"] for s in smokes]
            file.write("PERFORMANCE STATISTICS:\n")
            file.write("-" * 50 + "\n")
            file.write(f"Maximum Single Bomb Effect: {max(effect_times):.6f} s\n")
            file.write(f"Minimum Single Bomb Effect: {min(effect_times):.6f} s\n")
            file.write(f"Standard Deviation: {np.std(effect_times):.6f} s\n")
            file.write(f"Coefficient of Variation: {np.std(effect_times)/np.mean(effect_times)*100:.2f}%\n")
        
        file.write("\n")
        file.write("=" * 100 + "\n")
        file.write("END OF COMPREHENSIVE REPORT\n")
        file.write("=" * 100 + "\n")
    
    print(f"Comprehensive optimization report saved to: {filename}")
    return filename

def create_optimization_convergence_plot(save_path="q5_optimization_convergence.png"):
    """Create convergence plot for the optimization process"""
    if not OPTIMIZATION_HISTORY["total_durations"]:
        print("No convergence data available")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    iterations = range(1, len(OPTIMIZATION_HISTORY["total_durations"]) + 1)
    
    # 1. Total shielding duration convergence
    ax1.plot(iterations, OPTIMIZATION_HISTORY["total_durations"], 'b-o', linewidth=2, markersize=6, label="Total Shielding Duration")
    ax1.fill_between(iterations, OPTIMIZATION_HISTORY["total_durations"], alpha=0.3, color='lightblue')
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Total Shielding Duration (s)", fontsize=12)
    ax1.set_title("Convergence: Total Shielding Duration", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add best point annotation
    if OPTIMIZATION_HISTORY["total_durations"]:
        max_duration = max(OPTIMIZATION_HISTORY["total_durations"])
        max_iter = OPTIMIZATION_HISTORY["total_durations"].index(max_duration) + 1
        ax1.annotate(f'Best: {max_duration:.3f}s\nat iteration {max_iter}', 
                    xy=(max_iter, max_duration), xytext=(max_iter + 1, max_duration + max_duration*0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, color='red', fontweight='bold')
    
    # 2. Improvement per iteration
    ax2.bar(iterations, OPTIMIZATION_HISTORY["improvements"], color='green', alpha=0.7, label="Improvement per Iteration")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Improvement (s)", fontsize=12)
    ax2.set_title("Iteration-wise Improvement", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # 3. Number of solved UAVs
    ax3.plot(iterations, OPTIMIZATION_HISTORY["drone_solutions"], 'r-s', linewidth=2, markersize=6, label="UAVs with Solutions")
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("Number of UAVs", fontsize=12)
    ax3.set_title("UAVs Solution Progress", fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 6)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Iteration time analysis
    ax4.bar(iterations, OPTIMIZATION_HISTORY["iteration_times"], color='orange', alpha=0.7, label="Iteration Time")
    ax4.set_xlabel("Iteration", fontsize=12)
    ax4.set_ylabel("Time (s)", fontsize=12)
    ax4.set_title("Computational Time per Iteration", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    # Overall title
    fig.suptitle("Multi-UAV Smoke Bomb Deployment Optimization Convergence Analysis", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization convergence plot saved to: {save_path}")

def visualize_final_deployment(smokes, save_path="q5_final_deployment.png"):
    """Create visualization of final deployment strategy"""
    if not smokes:
        print("No valid data for visualization")
        return
    
    plt.figure(figsize=(16, 12))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Draw real target, missile trajectories, UAV trajectories
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["radius"] * np.cos(theta)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["radius"] * np.sin(theta)
    ax1.plot(x_circle, y_circle, "r-", label="Real Target Projection", linewidth=2)
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=200, label="Real Target Center")
    
    # Missile trajectories
    colors = ["red", "green", "blue"]
    for i, (m_name, m_data) in enumerate(MISSILES.items()):
        t_range = np.linspace(0, m_data["flight_time"], 100)
        pos_list = [get_missile_position(m_name, t)[:2] for t in t_range]
        pos_arr = np.array(pos_list)
        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{colors[i]}--", label=f"{m_name} Trajectory", linewidth=2)
        ax1.scatter(m_data["init_pos"][0], m_data["init_pos"][1], c=colors[i], s=100, label=f"{m_name} Initial Position")
    
    # UAV trajectories and smoke bombs
    drone_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (d_name, d_data) in enumerate(DRONES.items()):
        if not d_data["smokes"]:
            continue
        
        # UAV trajectory
        last_smoke = d_data["smokes"][-1] if d_data["smokes"] else None
        if last_smoke:
            t_range = np.linspace(0, last_smoke["drop_time"], 50)
            v_vec = np.array([d_data["speed"] * np.cos(d_data["direction"]), 
                             d_data["speed"] * np.sin(d_data["direction"]), 0]) if d_data["speed"] and d_data["direction"] else np.array([0, 0, 0])
            pos_list = [d_data["init_pos"] + v_vec * t for t in t_range]
            pos_arr = np.array(pos_list)
            ax1.plot(pos_arr[:, 0], pos_arr[:, 1], f"{drone_colors[i]}-", label=f"{d_name} Trajectory", linewidth=2)
            ax1.scatter(d_data["init_pos"][0], d_data["init_pos"][1], c=drone_colors[i], s=100, marker="^", label=f"{d_name} Initial Position")
            
            # Smoke detonation points
            for smoke in d_data["smokes"]:
                det_pos = smoke["det_pos"]
                if det_pos is not None:
                    ax1.scatter(det_pos[0], det_pos[1], c=drone_colors[i], s=50, alpha=0.7)
                    circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, color=drone_colors[i], alpha=0.2)
                    ax1.add_patch(circle)
    
    ax1.set_xlabel("X Coordinate (m)", fontsize=12)
    ax1.set_ylabel("Y Coordinate (m)", fontsize=12)
    ax1.set_title("UAV Trajectories, Missile Paths and Smoke Detonation Points", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Missile shielding duration
    missile_effect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missile_effect[smoke["missile"]] += smoke["effective_time"]
    bars2 = ax2.bar(missile_effect.keys(), missile_effect.values(), color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel("Missile ID", fontsize=12)
    ax2.set_ylabel("Total Shielding Duration (s)", fontsize=12)
    ax2.set_title("Total Shielding Duration per Missile", fontsize=14, fontweight='bold')
    for bar, (m, t) in zip(bars2, missile_effect.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f"{t:.1f}s", ha="center", va="bottom", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. UAV smoke bomb count
    drone_smoke_count = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    bars3 = ax3.bar(drone_smoke_count.keys(), drone_smoke_count.values(), color=drone_colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel("UAV ID", fontsize=12)
    ax3.set_ylabel("Number of Smoke Bombs", fontsize=12)
    ax3.set_title("Smoke Bomb Deployment per UAV", fontsize=14, fontweight='bold')
    for bar, (d, cnt) in zip(bars3, drone_smoke_count.items()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05, str(cnt), ha="center", va="bottom", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Shielding duration distribution
    effect_times = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effect_times, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Single Smoke Bomb Shielding Duration (s)", fontsize=12)
    ax4.set_ylabel("Number of Smoke Bombs", fontsize=12)
    ax4.set_title("Shielding Duration Distribution", fontsize=14, fontweight='bold')
    ax4.axvline(np.mean(effect_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(effect_times):.3f}s')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Final deployment visualization saved to: {save_path}")


# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    start_time = time.time()
    
    print("=" * 80)
    print("MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION")
    print("Problem 5: 5 UAVs (FY1-FY5) vs 3 Missiles (M1-M3)")
    print("Algorithm: Enhanced Iterative Differential Evolution")
    print("=" * 80)
    
    # Run enhanced iterative optimization with increased parameters
    all_smokes = iterative_optimization(
        max_iterations=20, 
        improvement_threshold=0.3, 
        max_stall_iter=3
    )
    
    total_computation_time = time.time() - start_time
    
    if all_smokes:
        print(f"\nOptimization completed in {total_computation_time:.2f} seconds")
        
        # Save results in standard format (result3.xlsx compatible)
        result_df = save_result_standard_format(all_smokes, filename="result3.xlsx")
        
        # Create comprehensive txt backup
        txt_report = save_comprehensive_optimization_report(all_smokes)
        
        # Generate convergence visualization
        create_optimization_convergence_plot()
        
        # Generate final deployment visualization  
        visualize_final_deployment(all_smokes)
        
        # Final summary output
        print("\n" + "="*80)
        print("FINAL OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        print(f"Total smoke bombs deployed: {len(all_smokes)}")
        print(f"Total shielding duration: {sum([s['effective_time'] for s in all_smokes]):.3f}s")
        print(f"Average effectiveness per bomb: {sum([s['effective_time'] for s in all_smokes])/len(all_smokes):.3f}s")
        print(f"Total computation time: {total_computation_time:.2f}s")
        
        print(f"\nUAV deployment details:")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"  {d_name}: {len(d_data['smokes'])} bombs, total shielding time {total:.3f}s")
            else:
                print(f"  {d_name}: No valid deployment strategy found")
        
        print(f"\nGenerated files:")
        print(f"  - Standard results: result3.xlsx")
        print(f"  - Comprehensive report: {txt_report}")
        print(f"  - Convergence plot: q5_optimization_convergence.png")
        print(f"  - Deployment visualization: q5_final_deployment.png")
        
        print("="*80)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
    else:
        print("No valid smoke bomb deployment strategy found")
        print("Consider adjusting optimization parameters or constraints")
