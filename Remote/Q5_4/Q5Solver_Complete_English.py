#!/usr/bin/env python3
"""
Complete English Version: Multi-UAV Smoke Bomb Deployment Optimization
Problem 5: 5 UAVs (FY1-FY5) intercept 3 missiles (M1-M3) using up to 3 smoke bombs each
Enhanced algorithm with convergence tracking and standard output format
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

# ========================== 1. GLOBAL PARAMETERS AND CONSTANTS ==========================
# Real target parameters (cylindrical target)
TRUE_TARGET = {
    "radius": 7,          # Cylinder radius (m)
    "height": 10,         # Cylinder height (m)  
    "center": np.array([0, 200, 0]),  # Base center coordinates (m)
    "sample_points": None  # Will be populated with sampling points
}

# Missile parameters (3 incoming missiles)
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "direction": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "direction": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "direction": None, "flight_time": None}
}

# Physical constants
MISSILE_SPEED = 300  # Missile speed (m/s)
GRAVITY = 9.8  # Gravitational acceleration (m/s²)
SMOKE_RADIUS = 10  # Smoke effective radius (m)
SMOKE_SINK_SPEED = 3  # Sinking speed after detonation (m/s)
SMOKE_EFFECTIVE_TIME = 20  # Effective duration after detonation (s)

# UAV parameters (5 UAVs, each can deploy up to 3 smoke bombs)
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

# Operational constraints
DROP_INTERVAL = 1  # Minimum interval between drops from same UAV (s)
TIME_STEP = 0.1  # Time sampling step for calculations (s)

# Optimization tracking
OPTIMIZATION_HISTORY = {
    "iteration_times": [],
    "total_durations": [],
    "improvements": [],
    "drone_solutions": []
}


# ========================== 2. INITIALIZATION FUNCTIONS ==========================
def generate_target_sampling_points():
    """Generate sampling points on the cylindrical target surface and interior"""
    samples = []
    r, h, center = TRUE_TARGET["radius"], TRUE_TARGET["height"], TRUE_TARGET["center"]
    
    # Base circle sampling
    samples.append(center)
    for theta in np.linspace(0, 2*np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    
    # Top circle sampling
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

def initialize_missiles():
    """Initialize missile direction vectors and flight times"""
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        # Direction towards origin (fake target at [0,0,0])
        dir_vec = -init_pos / np.linalg.norm(init_pos)
        m_data["direction"] = dir_vec * MISSILE_SPEED
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED
        print(f"Missile {m_name}: flight time = {m_data['flight_time']:.2f}s")


# ========================== 3. CORE CALCULATION FUNCTIONS ==========================
def get_missile_position(missile_name, time_t):
    """Calculate missile position at time t"""
    missile_data = MISSILES[missile_name]
    if time_t > missile_data["flight_time"]:
        return missile_data["init_pos"] + missile_data["direction"] * missile_data["flight_time"]
    return missile_data["init_pos"] + missile_data["direction"] * time_t

def get_drone_position(drone_name, time_t):
    """Calculate UAV position at time t"""
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                               drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + velocity_vector * time_t

def get_smoke_position(drone_name, drop_time, detonation_delay, time_t):
    """Calculate smoke bomb position at time t (includes free fall and sinking phases)"""
    drone = DRONES[drone_name]
    
    if time_t < drop_time:
        return None
    
    drop_position = get_drone_position(drone_name, drop_time)
    
    # Phase 1: Before detonation (free fall with horizontal motion)
    if time_t < drop_time + detonation_delay:
        delta_t = time_t - drop_time
        velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                                   drone["speed"] * np.sin(drone["direction"]), 0])
        x = drop_position[0] + velocity_vector[0] * delta_t
        y = drop_position[1] + velocity_vector[1] * delta_t
        z = drop_position[2] - 0.5 * GRAVITY * delta_t**2
        return np.array([x, y, max(z, 0.1)])  # Minimum Z = 0.1 to avoid invalid positions
    
    # Phase 2: After detonation (sinking smoke cloud)
    detonation_time = drop_time + detonation_delay
    if time_t > detonation_time + SMOKE_EFFECTIVE_TIME:
        return None
    
    # Calculate detonation position
    delta_t_detonation = detonation_delay
    velocity_vector = np.array([drone["speed"] * np.cos(drone["direction"]), 
                               drone["speed"] * np.sin(drone["direction"]), 0])
    det_x = drop_position[0] + velocity_vector[0] * delta_t_detonation
    det_y = drop_position[1] + velocity_vector[1] * delta_t_detonation
    det_z = drop_position[2] - 0.5 * GRAVITY * delta_t_detonation**2
    
    if det_z < 0:
        det_z = 0.1  # Error tolerance
    
    # Apply sinking effect
    delta_t_after_detonation = time_t - detonation_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after_detonation
    return np.array([det_x, det_y, max(z, 0.1)])

def line_segment_sphere_intersection(point1, point2, sphere_center, sphere_radius):
    """Check if line segment intersects with sphere (geometric intersection test)"""
    vector_p = point2 - point1
    vector_c = sphere_center - point1
    t = np.dot(vector_c, vector_p) / (np.dot(vector_p, vector_p) + 1e-8)
    
    if 0 <= t <= 1:
        nearest_point = point1 + t * vector_p
    else:
        nearest_point = point1 if t < 0 else point2
    
    return np.linalg.norm(nearest_point - sphere_center) <= sphere_radius + 1e-8

def calculate_smoke_effectiveness(drone_name, missile_name, drop_time, detonation_delay):
    """Calculate effective shielding time for single smoke bomb against specific missile"""
    drone = DRONES[drone_name]
    velocity, direction = drone["speed"], drone["direction"]
    
    if velocity is None or direction is None:
        return -1000
    
    # Speed constraint check (with numerical tolerance)
    if not (drone["speed_range"][0] - 1e-3 <= velocity <= drone["speed_range"][1] + 1e-3):
        return -1000
    
    # Check detonation point validity (must be above ground)
    detonation_time = drop_time + detonation_delay
    drop_position = get_drone_position(drone_name, drop_time)
    detonation_z = drop_position[2] - 0.5 * GRAVITY * detonation_delay**2
    if detonation_z < -0.5:  # Relaxed ground constraint
        return -1000
    
    # Check drop interval constraint (minimum 1s between drops from same UAV)
    for existing_smoke in drone["smokes"]:
        if abs(drop_time - existing_smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    # Calculate effective shielding duration
    max_time = min(detonation_time + SMOKE_EFFECTIVE_TIME, MISSILES[missile_name]["flight_time"] + 1)
    min_time = max(detonation_time, 0)
    if min_time >= max_time - 1e-3:
        return 0
    
    effective_duration = 0
    for t in np.arange(min_time, max_time, TIME_STEP):
        missile_position = get_missile_position(missile_name, t)
        smoke_position = get_smoke_position(drone_name, drop_time, detonation_delay, t)
        if smoke_position is None:
            continue
        
        # Check if all target sample points are shielded
        all_points_shielded = True
        for sample_point in TRUE_TARGET["sample_points"]:
            if not line_segment_sphere_intersection(missile_position, sample_point, smoke_position, SMOKE_RADIUS):
                all_points_shielded = False
                break
        
        if all_points_shielded:
            effective_duration += TIME_STEP
    
    return effective_duration


# ========================== 4. OPTIMIZATION ALGORITHMS ==========================
def optimize_single_smoke_deployment(drone_name, missile_name):
    """Optimize single smoke bomb deployment using enhanced differential evolution"""
    drone = DRONES[drone_name]
    speed_min, speed_max = drone["speed_range"]
    max_flight_time = MISSILES[missile_name]["flight_time"]
    
    # Extended search boundaries for better exploration
    optimization_bounds = [
        (speed_min * 0.8, speed_max * 1.2),    # Speed (20% extension)
        (0, 2 * np.pi),                        # Direction angle (0-360°)
        (0, max_flight_time - 1),              # Drop time
        (0.1, 20)                              # Detonation delay (0.1-20s)
    ]
    
    def objective_function(parameters):
        velocity, direction, drop_time, det_delay = parameters
        drone["speed"] = velocity
        drone["direction"] = direction
        return -calculate_smoke_effectiveness(drone_name, missile_name, drop_time, det_delay)
    
    # Enhanced differential evolution with optimized parameters
    optimization_result = differential_evolution(
        func=objective_function,
        bounds=optimization_bounds,
        mutation=0.8,          # Increased mutation for better exploration
        recombination=0.9,     # High recombination for faster convergence
        popsize=60,            # Larger population size
        maxiter=80,            # More iterations
        tol=1e-3,              # Relaxed tolerance
        disp=False,
        polish=True,           # Local optimization refinement
        seed=42                # Fixed seed for reproducibility
    )
    
    velocity_opt, direction_opt, drop_time_opt, det_delay_opt = optimization_result.x
    # Ensure speed is within valid range
    velocity_opt = np.clip(velocity_opt, speed_min, speed_max)
    effectiveness = -optimization_result.fun
    
    return {
        "speed": velocity_opt,
        "direction": direction_opt,
        "drop_time": drop_time_opt,
        "detonation_delay": det_delay_opt,
        "detonation_time": drop_time_opt + det_delay_opt,
        "detonation_position": get_smoke_position(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
        "effectiveness": effectiveness if effectiveness > 1e-3 else 0,
        "target_missile": missile_name
    }

def optimize_uav_trajectory(drone_name, missile_name, retry_count=0):
    """Optimize complete UAV trajectory with multiple smoke bomb deployments"""
    drone = DRONES[drone_name]
    speed_min, speed_max = drone["speed_range"]
    max_smoke_bombs = drone["max_smoke"]
    
    print(f"[{drone_name}] Optimizing trajectory for missile {missile_name}...")
    
    # Test multiple speed candidates (increased from 5 to 8)
    speed_candidates = np.linspace(speed_min, speed_max, 8)
    best_speed = None
    best_smoke_deployment = []
    maximum_total_effectiveness = 0
    
    for test_speed in speed_candidates:
        drone["speed"] = test_speed
        temporary_smoke_deployment = []
        
        for smoke_index in range(max_smoke_bombs):
            # Calculate minimum drop time based on previous deployments
            min_drop_time = temporary_smoke_deployment[-1]["drop_time"] + DROP_INTERVAL if temporary_smoke_deployment else 0
            max_drop_time = MISSILES[missile_name]["flight_time"] - 0.1
            if min_drop_time >= max_drop_time - 1e-3:
                break
            
            def trajectory_objective(parameters):
                direction, drop_time, det_delay = parameters
                drone["direction"] = direction
                return -calculate_smoke_effectiveness(drone_name, missile_name, drop_time, det_delay)
            
            # Enhanced differential evolution for trajectory optimization
            trajectory_result = differential_evolution(
                func=trajectory_objective, 
                bounds=[(0, 2*np.pi), (min_drop_time, max_drop_time), (0.1, 10)],
                mutation=0.7, 
                recombination=0.8, 
                popsize=50,        # Increased population size
                maxiter=60,        # More iterations
                disp=False,
                seed=smoke_index * 17  # Different seed for each smoke bomb
            )
            
            direction_opt, drop_time_opt, det_delay_opt = trajectory_result.x
            drone["direction"] = direction_opt
            effectiveness = calculate_smoke_effectiveness(drone_name, missile_name, drop_time_opt, det_delay_opt)
            
            if effectiveness > 0.1:  # Only keep solutions with significant effectiveness
                smoke_deployment = {
                    "speed": test_speed,
                    "direction": direction_opt,
                    "drop_time": drop_time_opt,
                    "detonation_delay": det_delay_opt,
                    "detonation_time": drop_time_opt + det_delay_opt,
                    "detonation_position": get_smoke_position(drone_name, drop_time_opt, det_delay_opt, drop_time_opt + det_delay_opt),
                    "effectiveness": effectiveness,
                    "target_missile": missile_name
                }
                temporary_smoke_deployment.append(smoke_deployment)
        
        # Update best solution if current is better
        total_effectiveness = sum([s["effectiveness"] for s in temporary_smoke_deployment]) if temporary_smoke_deployment else 0
        if total_effectiveness > maximum_total_effectiveness:
            maximum_total_effectiveness = total_effectiveness
            best_speed = test_speed
            best_smoke_deployment = temporary_smoke_deployment
    
    # Retry mechanism if no solution found
    if not best_smoke_deployment and retry_count < 3:
        print(f"[{drone_name}] Optimization failed, retrying {retry_count+1}/3...")
        return optimize_uav_trajectory(drone_name, missile_name, retry_count+1)
    
    # Path fitting and fine-tuning optimization
    if best_smoke_deployment:
        # Weighted direction fitting based on effectiveness
        drop_points = []
        weights = []
        for smoke in best_smoke_deployment:
            velocity_vector = np.array([smoke["speed"] * np.cos(smoke["direction"]), 
                                       smoke["speed"] * np.sin(smoke["direction"]), 0])
            drop_pos = drone["init_pos"] + velocity_vector * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effectiveness"])
        
        drop_points = np.array(drop_points)
        weights = np.array(weights)
        
        # Weighted least squares fitting for direction
        if len(drop_points) > 1:
            X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
            W = np.diag(weights)
            try:
                k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
                reference_direction = np.arctan(k)
            except np.linalg.LinAlgError:
                reference_direction = np.mean([s["direction"] for s in best_smoke_deployment])
        else:
            reference_direction = best_smoke_deployment[0]["direction"]
        
        # Fine-tuning optimization around best solution
        for i, smoke in enumerate(best_smoke_deployment):
            direction_candidates = [reference_direction - np.pi/24, reference_direction, reference_direction + np.pi/24]
            drop_time_candidates = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]
            
            best_effectiveness = smoke["effectiveness"]
            best_parameters = (smoke["direction"], smoke["drop_time"])
            
            for test_direction in direction_candidates:
                for test_drop_time in drop_time_candidates:
                    # Check constraints
                    previous_drop_time = best_smoke_deployment[i-1]["drop_time"] if i > 0 else -np.inf
                    if test_drop_time < previous_drop_time + DROP_INTERVAL - 0.1:
                        continue
                    
                    drone["direction"] = test_direction
                    test_effectiveness = calculate_smoke_effectiveness(drone_name, missile_name, test_drop_time, smoke["detonation_delay"])
                    if test_effectiveness > best_effectiveness:
                        best_effectiveness = test_effectiveness
                        best_parameters = (test_direction, test_drop_time)
            
            # Update smoke parameters with best found
            smoke["direction"], smoke["drop_time"] = best_parameters
            smoke["detonation_time"] = smoke["drop_time"] + smoke["detonation_delay"]
            smoke["detonation_position"] = get_smoke_position(drone_name, smoke["drop_time"], smoke["detonation_delay"], smoke["detonation_time"])
            smoke["effectiveness"] = best_effectiveness
    
    # Update drone parameters
    drone["speed"] = best_speed
    drone["direction"] = reference_direction if best_smoke_deployment else None
    drone["smokes"] = best_smoke_deployment
    
    total_effectiveness = sum([s["effectiveness"] for s in best_smoke_deployment]) if best_smoke_deployment else 0
    print(f"[{drone_name}] Optimization complete: {len(best_smoke_deployment)} smoke bombs, total effectiveness {total_effectiveness:.3f}s")
    
    return best_smoke_deployment

def assign_tasks_to_uavs(unoptimized_drones=None):
    """Assign missiles to UAVs using Hungarian algorithm with enhanced cost matrix"""
    if unoptimized_drones is None:
        unoptimized_drones = list(DRONES.keys())
    
    missile_list = list(MISSILES.keys())
    n_drones = len(unoptimized_drones)
    n_missiles = len(missile_list)
    
    if n_drones == 0:
        return {m: [] for m in missile_list}
    
    # Build cost matrix considering multiple factors
    cost_matrix = np.zeros((n_drones, n_missiles))
    
    for i, drone_name in enumerate(unoptimized_drones):
        drone_init_pos = DRONES[drone_name]["init_pos"]
        drone_avg_speed = (DRONES[drone_name]["speed_range"][0] + DRONES[drone_name]["speed_range"][1]) / 2
        
        for j, missile_name in enumerate(missile_list):
            missile_init_pos = MISSILES[missile_name]["init_pos"]
            missile_flight_time = MISSILES[missile_name]["flight_time"]
            
            # Cost factor 1: Distance-based flight time
            distance = np.linalg.norm(drone_init_pos - missile_init_pos)
            cost_distance = distance / drone_avg_speed
            
            # Cost factor 2: Missile urgency (shorter flight time = higher urgency = lower cost)
            cost_urgency = 1000 / (missile_flight_time + 1)
            
            # Cost factor 3: Speed matching with missile
            cost_speed_match = abs(drone_avg_speed - MISSILE_SPEED) / 100
            
            # Combined cost
            cost_matrix[i][j] = cost_distance + cost_urgency + cost_speed_match
    
    # Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    
    for i, j in zip(row_indices, col_indices):
        assignments[missile_list[j]].append(unoptimized_drones[i])
    
    # Assign remaining UAVs to lowest cost missiles
    assigned_drones = set(row_indices)
    for i in range(n_drones):
        if i not in assigned_drones:
            min_cost_missile_index = np.argmin(cost_matrix[i])
            assignments[missile_list[min_cost_missile_index]].append(unoptimized_drones[i])
    
    return assignments

def enhanced_iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stagnation_iterations=3):
    """Main iterative optimization with convergence tracking and adaptive parameters"""
    print("=== Starting Enhanced Iterative Optimization Algorithm ===")
    
    # Initialize UAV states
    for drone_name in DRONES:
        DRONES[drone_name]["optimized"] = False
        DRONES[drone_name]["smokes"] = []
        DRONES[drone_name]["speed"] = None
        DRONES[drone_name]["direction"] = None
    
    # Initialize optimization tracking
    OPTIMIZATION_HISTORY["iteration_times"] = []
    OPTIMIZATION_HISTORY["total_durations"] = []
    OPTIMIZATION_HISTORY["improvements"] = []
    OPTIMIZATION_HISTORY["drone_solutions"] = []
    
    all_smoke_deployments = []
    previous_total_effectiveness = 0
    stagnation_counter = 0
    
    for iteration in range(max_iterations):
        iteration_start_time = time.time()
        print(f"\n===== Iteration {iteration + 1}/{max_iterations} =====")
        
        # Step 1: Identify UAVs without solutions
        drones_needing_optimization = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"UAVs requiring optimization: {drones_needing_optimization}")
        
        # Step 2: Early termination if all UAVs have solutions
        if not drones_needing_optimization:
            print("All UAVs have found valid deployment strategies, terminating optimization")
            break
        
        # Step 3: Intelligent task assignment
        task_assignments = assign_tasks_to_uavs(drones_needing_optimization)
        print(f"Task assignment result: {task_assignments}")
        
        # Step 4: Optimize assigned UAVs
        iteration_smoke_deployments = []
        uavs_optimized_this_iteration = []
        
        for missile_name, assigned_drone_names in task_assignments.items():
            for drone_name in assigned_drone_names:
                # Skip UAVs that already have solutions
                if len(DRONES[drone_name]["smokes"]) > 0:
                    continue
                    
                print(f"Optimizing UAV {drone_name} against missile {missile_name}...")
                smoke_deployments = optimize_uav_trajectory(drone_name, missile_name)
                
                if smoke_deployments:
                    # Add drone identifier to each smoke deployment
                    drone_smoke_deployments = [{**smoke, "drone": drone_name} for smoke in smoke_deployments]
                    iteration_smoke_deployments.extend(drone_smoke_deployments)
                    total_effectiveness = sum([s["effectiveness"] for s in smoke_deployments])
                    print(f"[{drone_name}] Success: {len(smoke_deployments)} smoke bombs, total effectiveness {total_effectiveness:.3f}s")
                else:
                    print(f"[{drone_name}] No valid deployment strategy found, will retry in next iteration")
                
                # Mark UAV as having been optimized
                DRONES[drone_name]["optimized"] = True
                uavs_optimized_this_iteration.append(drone_name)
        
        # Step 5: Update global deployment data
        all_smoke_deployments.extend(iteration_smoke_deployments)
        
        # Step 6: Calculate performance metrics
        total_system_effectiveness = sum([sum([s["effectiveness"] for s in d["smokes"]]) for d in DRONES.values()])
        effectiveness_improvement = total_system_effectiveness - previous_total_effectiveness
        
        # Record iteration data for convergence analysis
        iteration_duration = time.time() - iteration_start_time
        OPTIMIZATION_HISTORY["iteration_times"].append(iteration_duration)
        OPTIMIZATION_HISTORY["total_durations"].append(total_system_effectiveness)
        OPTIMIZATION_HISTORY["improvements"].append(effectiveness_improvement)
        OPTIMIZATION_HISTORY["drone_solutions"].append(len(DRONES) - len(drones_needing_optimization))
        
        # Step 7: Progress reporting
        print(f"Current total system effectiveness: {total_system_effectiveness:.3f}s")
        print(f"Improvement from previous iteration: {effectiveness_improvement:.3f}s")
        print(f"UAVs optimized this iteration: {uavs_optimized_this_iteration}")
        print(f"UAVs with valid solutions: {len(DRONES) - len(drones_needing_optimization)}/{len(DRONES)}")
        print(f"Iteration computation time: {iteration_duration:.2f}s")
        
        # Step 8: Convergence and stagnation analysis
        if effectiveness_improvement < improvement_threshold:
            stagnation_counter += 1
            print(f"Consecutive stagnation iterations: {stagnation_counter}/{max_stagnation_iterations}")
            
            if stagnation_counter >= max_stagnation_iterations:
                if drones_needing_optimization:
                    print(f"Stagnation detected but UAVs still need optimization, continuing with modified parameters...")
                    stagnation_counter = max_stagnation_iterations - 1  # Prevent infinite loop
                else:
                    print(f"Convergence achieved after {stagnation_counter} stagnation iterations, terminating")
                    break
        else:
            stagnation_counter = 0  # Reset stagnation counter
        
        previous_total_effectiveness = total_system_effectiveness
    
    # Final validation and reporting
    final_unoptimized_uavs = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if final_unoptimized_uavs:
        print(f"\nWarning: Optimization limit reached, UAVs without solutions: {final_unoptimized_uavs}")
    
    final_total_effectiveness = sum([sum([s["effectiveness"] for s in d["smokes"]]) for d in DRONES.values()])
    print(f"\nOptimization algorithm completed! Final total effectiveness: {final_total_effectiveness:.3f}s")
    
    return all_smoke_deployments


# ========================== 5. RESULT OUTPUT AND ANALYSIS ==========================
def save_result_standard_format(smoke_deployments, filename="result3.xlsx"):
    """Save optimization results in standard format compatible with result3.xlsx template"""
    output_data = []
    
    for index, smoke in enumerate(smoke_deployments, 1):
        detonation_pos = smoke["detonation_position"] if smoke["detonation_position"] is not None else np.array([0,0,0])
        
        output_data.append({
            "Serial_Number": f"S{index}",
            "UAV_Number": smoke["drone"],
            "Speed_m_per_s": round(smoke["speed"], 2),
            "Direction_degrees": round(np.degrees(smoke["direction"]), 2),
            "Drop_Time_s": round(smoke["drop_time"], 2),
            "Detonation_Delay_s": round(smoke["detonation_delay"], 2),
            "Detonation_Time_s": round(smoke["detonation_time"], 2),
            "Detonation_Point_X_m": round(detonation_pos[0], 2),
            "Detonation_Point_Y_m": round(detonation_pos[1], 2),
            "Detonation_Point_Z_m": round(detonation_pos[2], 2),
            "Target_Missile": smoke["target_missile"],
            "Effective_Shielding_Duration_s": round(smoke["effectiveness"], 3)
        })
    
    # Create DataFrame and export to Excel
    results_dataframe = pd.DataFrame(output_data)
    results_dataframe.to_excel(filename, index=False, engine="openpyxl")
    print(f"Standard format results saved to: {filename}")
    return results_dataframe

def save_comprehensive_optimization_report(smoke_deployments, filename="q5_optimization_comprehensive_report.txt"):
    """Save detailed optimization results and analysis to text file"""
    
    with open(filename, 'w', encoding='utf-8') as report_file:
        # Report header
        report_file.write("=" * 120 + "\n")
        report_file.write("COMPREHENSIVE MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION REPORT\n")
        report_file.write("Problem 5: 5 UAVs (FY1-FY5) vs 3 Missiles (M1, M2, M3)\n")
        report_file.write("Algorithm: Enhanced Iterative Differential Evolution with Convergence Tracking\n")
        report_file.write("=" * 120 + "\n")
        report_file.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"Total Computation Time: {sum(OPTIMIZATION_HISTORY['iteration_times']):.3f} seconds\n")
        report_file.write(f"Number of Optimization Iterations: {len(OPTIMIZATION_HISTORY['total_durations'])}\n")
        report_file.write(f"Final Total Effectiveness: {sum([s['effectiveness'] for s in smoke_deployments]):.6f} seconds\n")
        report_file.write("\n")
        
        # Problem configuration
        report_file.write("PROBLEM CONFIGURATION:\n")
        report_file.write("-" * 60 + "\n")
        report_file.write(f"Real Target: Cylindrical target at {TRUE_TARGET['center']} with radius {TRUE_TARGET['radius']}m, height {TRUE_TARGET['height']}m\n")
        report_file.write(f"Fake Target: Point at origin [0, 0, 0] (missile destination)\n")
        report_file.write(f"Missiles:\n")
        for m_name, m_data in MISSILES.items():
            report_file.write(f"  {m_name}: Initial position {m_data['init_pos']}, flight time {m_data['flight_time']:.2f}s\n")
        report_file.write(f"Missile Speed: {MISSILE_SPEED} m/s\n")
        report_file.write(f"UAVs:\n")
        for d_name, d_data in DRONES.items():
            report_file.write(f"  {d_name}: Initial position {d_data['init_pos']}, speed range {d_data['speed_range']} m/s, max bombs {d_data['max_smoke']}\n")
        report_file.write(f"Smoke Parameters: Radius {SMOKE_RADIUS}m, sink speed {SMOKE_SINK_SPEED}m/s, duration {SMOKE_EFFECTIVE_TIME}s\n")
        report_file.write(f"Drop Interval Constraint: {DROP_INTERVAL}s minimum between drops from same UAV\n")
        report_file.write("\n")
        
        # Optimization convergence analysis
        report_file.write("OPTIMIZATION CONVERGENCE ANALYSIS:\n")
        report_file.write("-" * 60 + "\n")
        report_file.write("Iter | Total Effectiveness (s) | Improvement (s) | UAVs Solved | Computation Time (s)\n")
        report_file.write("-" * 85 + "\n")
        
        for i, (total_eff, improvement, solved_count, iter_time) in enumerate(zip(
            OPTIMIZATION_HISTORY["total_durations"],
            OPTIMIZATION_HISTORY["improvements"], 
            OPTIMIZATION_HISTORY["drone_solutions"],
            OPTIMIZATION_HISTORY["iteration_times"]
        )):
            report_file.write(f"{i+1:4d} | {total_eff:20.6f} | {improvement:13.6f} | {solved_count:10d} | {iter_time:16.3f}\n")
        report_file.write("\n")
        
        # Final deployment strategy details
        total_effectiveness = sum([s["effectiveness"] for s in smoke_deployments])
        report_file.write("FINAL DEPLOYMENT STRATEGY:\n")
        report_file.write("-" * 60 + "\n")
        report_file.write(f"Total Smoke Bombs Deployed: {len(smoke_deployments)}\n")
        report_file.write(f"Total System Effectiveness: {total_effectiveness:.6f} seconds\n")
        report_file.write(f"Average Effectiveness per Bomb: {total_effectiveness/len(smoke_deployments):.6f} seconds\n")
        report_file.write("\n")
        
        # UAV-specific deployment details
        report_file.write("UAV-SPECIFIC DEPLOYMENT DETAILS:\n")
        report_file.write("-" * 60 + "\n")
        
        for drone_name, drone_data in DRONES.items():
            report_file.write(f"UAV {drone_name}:\n")
            if drone_data["smokes"]:
                uav_total_effectiveness = sum([s["effectiveness"] for s in drone_data["smokes"]])
                report_file.write(f"  Status: ACTIVE DEPLOYMENT\n")
                report_file.write(f"  Optimized Speed: {drone_data['speed']:.3f} m/s\n")
                report_file.write(f"  Optimized Heading: {drone_data['direction']:.6f} rad ({np.degrees(drone_data['direction']):.2f}°)\n")
                report_file.write(f"  Number of Smoke Bombs: {len(drone_data['smokes'])}\n")
                report_file.write(f"  Total Effectiveness: {uav_total_effectiveness:.6f} s\n")
                report_file.write(f"  Average per Bomb: {uav_total_effectiveness/len(drone_data['smokes']):.6f} s\n")
                
                for bomb_index, smoke in enumerate(drone_data["smokes"], 1):
                    det_pos = smoke["detonation_position"] if smoke["detonation_position"] is not None else np.array([0,0,0])
                    report_file.write(f"    Smoke Bomb {bomb_index}:\n")
                    report_file.write(f"      Drop Time: {smoke['drop_time']:.3f}s\n")
                    report_file.write(f"      Detonation Delay: {smoke['detonation_delay']:.3f}s\n")
                    report_file.write(f"      Detonation Position: ({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f}) m\n")
                    report_file.write(f"      Effectiveness: {smoke['effectiveness']:.6f}s\n")
                    report_file.write(f"      Target Missile: {smoke['target_missile']}\n")
            else:
                report_file.write(f"  Status: NO VALID DEPLOYMENT FOUND\n")
            report_file.write("\n")
        
        # Missile coverage analysis
        report_file.write("MISSILE COVERAGE ANALYSIS:\n")
        report_file.write("-" * 60 + "\n")
        
        missile_coverage_stats = {m: {"bomb_count": 0, "total_effectiveness": 0, "coverage_percentage": 0} for m in MISSILES.keys()}
        for smoke in smoke_deployments:
            missile_id = smoke["target_missile"]
            missile_coverage_stats[missile_id]["bomb_count"] += 1
            missile_coverage_stats[missile_id]["total_effectiveness"] += smoke["effectiveness"]
        
        for missile_name, missile_data in MISSILES.items():
            coverage_stats = missile_coverage_stats[missile_name]
            coverage_percentage = coverage_stats["total_effectiveness"] / missile_data["flight_time"] * 100
            report_file.write(f"Missile {missile_name}:\n")
            report_file.write(f"  Assigned Smoke Bombs: {coverage_stats['bomb_count']}\n")
            report_file.write(f"  Total Shielding Time: {coverage_stats['total_effectiveness']:.6f}s\n")
            report_file.write(f"  Flight Time Coverage: {coverage_percentage:.2f}%\n")
            report_file.write(f"  Flight Duration: {missile_data['flight_time']:.2f}s\n")
        report_file.write("\n")
        
        # Statistical performance analysis
        if smoke_deployments:
            effectiveness_values = [s["effectiveness"] for s in smoke_deployments]
            report_file.write("STATISTICAL PERFORMANCE ANALYSIS:\n")
            report_file.write("-" * 60 + "\n")
            report_file.write(f"Maximum Single Bomb Effectiveness: {max(effectiveness_values):.6f} s\n")
            report_file.write(f"Minimum Single Bomb Effectiveness: {min(effectiveness_values):.6f} s\n")
            report_file.write(f"Mean Effectiveness: {np.mean(effectiveness_values):.6f} s\n")
            report_file.write(f"Standard Deviation: {np.std(effectiveness_values):.6f} s\n")
            report_file.write(f"Coefficient of Variation: {np.std(effectiveness_values)/np.mean(effectiveness_values)*100:.2f}%\n")
            
            # Effectiveness distribution analysis
            report_file.write(f"\nEffectiveness Distribution:\n")
            bins = [0, 1, 2, 5, 10, float('inf')]
            bin_labels = ["<1s", "1-2s", "2-5s", "5-10s", ">10s"]
            for i in range(len(bins)-1):
                count = sum(1 for eff in effectiveness_values if bins[i] <= eff < bins[i+1])
                percentage = count/len(effectiveness_values)*100
                report_file.write(f"  {bin_labels[i]}: {count} bombs ({percentage:.1f}%)\n")
        
        report_file.write("\n")
        report_file.write("=" * 120 + "\n")
        report_file.write("END OF COMPREHENSIVE OPTIMIZATION REPORT\n")
        report_file.write("=" * 120 + "\n")
    
    print(f"Comprehensive optimization report saved to: {filename}")
    return filename

def create_optimization_convergence_visualization(save_path="q5_optimization_convergence.png"):
    """Create comprehensive convergence visualization for the optimization process"""
    if not OPTIMIZATION_HISTORY["total_durations"]:
        print("No convergence data available for visualization")
        return
    
    # Create multi-panel convergence analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    iterations = range(1, len(OPTIMIZATION_HISTORY["total_durations"]) + 1)
    
    # Panel 1: Total effectiveness convergence
    ax1.plot(iterations, OPTIMIZATION_HISTORY["total_durations"], 'b-o', linewidth=2.5, markersize=6, label="Total System Effectiveness")
    ax1.fill_between(iterations, OPTIMIZATION_HISTORY["total_durations"], alpha=0.3, color='lightblue')
    ax1.set_xlabel("Iteration Number", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Total Shielding Duration (s)", fontsize=12, fontweight='bold')
    ax1.set_title("Convergence: Total System Effectiveness", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add peak performance annotation
    if OPTIMIZATION_HISTORY["total_durations"]:
        max_effectiveness = max(OPTIMIZATION_HISTORY["total_durations"])
        max_iteration = OPTIMIZATION_HISTORY["total_durations"].index(max_effectiveness) + 1
        ax1.annotate(f'Peak: {max_effectiveness:.3f}s\nat iteration {max_iteration}', 
                    xy=(max_iteration, max_effectiveness), 
                    xytext=(max_iteration + 1, max_effectiveness + max_effectiveness*0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Panel 2: Iteration-wise improvement
    colors = ['green' if imp >= 0 else 'red' for imp in OPTIMIZATION_HISTORY["improvements"]]
    bars = ax2.bar(iterations, OPTIMIZATION_HISTORY["improvements"], color=colors, alpha=0.7, 
                   label="Effectiveness Improvement")
    ax2.set_xlabel("Iteration Number", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Improvement (s)", fontsize=12, fontweight='bold')
    ax2.set_title("Iteration-wise Effectiveness Improvement", fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=11)
    
    # Panel 3: UAV solution progress
    ax3.plot(iterations, OPTIMIZATION_HISTORY["drone_solutions"], 'r-s', linewidth=2.5, markersize=6, label="UAVs with Valid Solutions")
    ax3.fill_between(iterations, OPTIMIZATION_HISTORY["drone_solutions"], alpha=0.3, color='lightcoral')
    ax3.set_xlabel("Iteration Number", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Number of UAVs", fontsize=12, fontweight='bold')
    ax3.set_title("UAV Solution Progress", fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 6)
    ax3.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target: All 5 UAVs')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    # Panel 4: Computational efficiency analysis
    ax4.bar(iterations, OPTIMIZATION_HISTORY["iteration_times"], color='orange', alpha=0.7, 
            label="Iteration Computation Time")
    ax4.set_xlabel("Iteration Number", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Computation Time (s)", fontsize=12, fontweight='bold')
    ax4.set_title("Computational Efficiency per Iteration", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=11)
    
    # Add average time line
    if OPTIMIZATION_HISTORY["iteration_times"]:
        avg_time = np.mean(OPTIMIZATION_HISTORY["iteration_times"])
        ax4.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_time:.2f}s')
        ax4.legend(fontsize=11)
    
    # Overall figure title and layout
    fig.suptitle("Multi-UAV Smoke Bomb Deployment Optimization: Convergence Analysis", 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization convergence visualization saved to: {save_path}")

def create_final_deployment_visualization(smoke_deployments, save_path="q5_final_deployment_layout.png"):
    """Create comprehensive visualization of the final deployment strategy"""
    if not smoke_deployments:
        print("No deployment data available for visualization")
        return
    
    # Create multi-panel deployment visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Panel 1: Spatial layout (top view)
    # Draw real target
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = TRUE_TARGET["center"][0] + TRUE_TARGET["radius"] * np.cos(theta_circle)
    y_circle = TRUE_TARGET["center"][1] + TRUE_TARGET["radius"] * np.sin(theta_circle)
    ax1.plot(x_circle, y_circle, "r-", linewidth=3, label="Real Target")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c="r", marker="*", s=300, label="Target Center")
    
    # Draw missile trajectories
    missile_colors = ["red", "green", "blue"]
    for i, (missile_name, missile_data) in enumerate(MISSILES.items()):
        time_range = np.linspace(0, missile_data["flight_time"], 100)
        trajectory_points = [get_missile_position(missile_name, t)[:2] for t in time_range]
        trajectory_array = np.array(trajectory_points)
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color=missile_colors[i], linestyle='--', linewidth=2, label=f"{missile_name} Trajectory")
        ax1.scatter(missile_data["init_pos"][0], missile_data["init_pos"][1], 
                   c=missile_colors[i], s=150, marker='v', label=f"{missile_name} Initial Position")
    
    # Draw UAV trajectories and smoke deployments
    uav_colors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (drone_name, drone_data) in enumerate(DRONES.items()):
        if not drone_data["smokes"]:
            # Show inactive UAVs
            ax1.scatter(drone_data["init_pos"][0], drone_data["init_pos"][1], 
                       c='gray', s=100, marker="^", alpha=0.5, label=f"{drone_name} (Inactive)")
            continue
        
        # Active UAV trajectory
        last_smoke = drone_data["smokes"][-1]
        time_range = np.linspace(0, last_smoke["drop_time"], 50)
        velocity_vector = np.array([drone_data["speed"] * np.cos(drone_data["direction"]), 
                                   drone_data["speed"] * np.sin(drone_data["direction"]), 0])
        trajectory_points = [drone_data["init_pos"] + velocity_vector * t for t in time_range]
        trajectory_array = np.array(trajectory_points)
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], color=uav_colors[i], 
                linewidth=2, label=f"{drone_name} Trajectory")
        ax1.scatter(drone_data["init_pos"][0], drone_data["init_pos"][1], 
                   c=uav_colors[i], s=120, marker="^", label=f"{drone_name} Initial Position")
        
        # Smoke bomb detonation points and effective areas
        for bomb_index, smoke in enumerate(drone_data["smokes"]):
            det_pos = smoke["detonation_position"]
            if det_pos is not None:
                ax1.scatter(det_pos[0], det_pos[1], c=uav_colors[i], s=80, alpha=0.8, edgecolors='black')
                # Draw effective radius circle
                effectiveness_circle = plt.Circle((det_pos[0], det_pos[1]), SMOKE_RADIUS, 
                                                 color=uav_colors[i], alpha=0.2, linewidth=1)
                ax1.add_patch(effectiveness_circle)
                # Label with bomb number
                ax1.text(det_pos[0], det_pos[1], f'{bomb_index+1}', fontsize=8, ha='center', va='center', 
                        fontweight='bold', color='white', 
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor=uav_colors[i], alpha=0.8))
    
    ax1.set_xlabel("X Coordinate (m)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Y Coordinate (m)", fontsize=12, fontweight='bold')
    ax1.set_title("Spatial Deployment Layout (Top View)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Missile shielding effectiveness
    missile_effectiveness = {m: 0 for m in MISSILES.keys()}
    for smoke in smoke_deployments:
        missile_effectiveness[smoke["target_missile"]] += smoke["effectiveness"]
    
    bars2 = ax2.bar(missile_effectiveness.keys(), missile_effectiveness.values(), 
                    color=missile_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel("Missile ID", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Total Shielding Duration (s)", fontsize=12, fontweight='bold')
    ax2.set_title("Total Shielding Effectiveness per Missile", fontsize=14, fontweight='bold')
    
    for bar, (missile_name, effectiveness) in zip(bars2, missile_effectiveness.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                f'{effectiveness:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: UAV deployment statistics
    uav_deployment_stats = {}
    for drone_name in DRONES.keys():
        if DRONES[drone_name]["smokes"]:
            uav_deployment_stats[drone_name] = sum([s["effectiveness"] for s in DRONES[drone_name]["smokes"]])
        else:
            uav_deployment_stats[drone_name] = 0
    
    bars3 = ax3.bar(uav_deployment_stats.keys(), uav_deployment_stats.values(), 
                    color=uav_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel("UAV ID", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Total Effectiveness (s)", fontsize=12, fontweight='bold')
    ax3.set_title("UAV Deployment Effectiveness", fontsize=14, fontweight='bold')
    
    for bar, (drone_name, effectiveness) in zip(bars3, uav_deployment_stats.items()):
        height = bar.get_height()
        bomb_count = len(DRONES[drone_name]["smokes"])
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05, 
                f'{effectiveness:.2f}s\n({bomb_count} bombs)', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Effectiveness distribution histogram
    effectiveness_values = [smoke["effectiveness"] for smoke in smoke_deployments]
    ax4.hist(effectiveness_values, bins=12, color="skyblue", edgecolor="navy", alpha=0.7, linewidth=1.5)
    ax4.axvline(np.mean(effectiveness_values), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(effectiveness_values):.3f}s')
    ax4.axvline(np.median(effectiveness_values), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(effectiveness_values):.3f}s')
    ax4.set_xlabel("Single Smoke Bomb Effectiveness (s)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Number of Smoke Bombs", fontsize=12, fontweight='bold')
    ax4.set_title("Effectiveness Distribution Analysis", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Overall figure formatting
    fig.suptitle("Multi-UAV Smoke Bomb Deployment: Final Strategy Analysis", 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Final deployment visualization saved to: {save_path}")


# ========================== 6. MAIN EXECUTION ==========================
if __name__ == "__main__":
    execution_start_time = time.time()
    
    print("=" * 100)
    print("MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION SYSTEM")
    print("Problem 5: 5 UAVs (FY1-FY5) vs 3 Missiles (M1-M3)")
    print("Algorithm: Enhanced Iterative Differential Evolution with Convergence Tracking")
    print("=" * 100)
    
    # Initialize system parameters
    print("Initializing system parameters...")
    generate_target_sampling_points()
    initialize_missiles()
    
    # Execute enhanced iterative optimization
    print("Starting enhanced iterative optimization...")
    all_smoke_deployments = enhanced_iterative_optimization(
        max_iterations=20, 
        improvement_threshold=0.3, 
        max_stagnation_iterations=3
    )
    
    total_execution_time = time.time() - execution_start_time
    
    if all_smoke_deployments:
        print(f"\nOptimization completed successfully in {total_execution_time:.2f} seconds")
        
        # Save results in standard format (result3.xlsx compatible)
        results_dataframe = save_result_standard_format(all_smoke_deployments, filename="result3.xlsx")
        
        # Generate comprehensive text report
        comprehensive_report = save_comprehensive_optimization_report(all_smoke_deployments)
        
        # Create convergence analysis visualization
        create_optimization_convergence_visualization()
        
        # Create final deployment visualization  
        create_final_deployment_visualization(all_smoke_deployments)
        
        # Display final summary
        total_effectiveness = sum([s['effectiveness'] for s in all_smoke_deployments])
        print("\n" + "="*100)
        print("FINAL OPTIMIZATION RESULTS SUMMARY")
        print("="*100)
        print(f"Total smoke bombs deployed: {len(all_smoke_deployments)}")
        print(f"Total system effectiveness: {total_effectiveness:.6f} seconds")
        print(f"Average effectiveness per bomb: {total_effectiveness/len(all_smoke_deployments):.6f} seconds")
        print(f"Total computation time: {total_execution_time:.2f} seconds")
        print(f"Optimization efficiency: {total_effectiveness/total_execution_time:.4f} effectiveness/second")
        
        print(f"\nUAV deployment summary:")
        for drone_name, drone_data in DRONES.items():
            if drone_data["smokes"]:
                uav_total = sum([s["effectiveness"] for s in drone_data["smokes"]])
                print(f"  {drone_name}: {len(drone_data['smokes'])}/{drone_data['max_smoke']} bombs deployed, "
                      f"total effectiveness {uav_total:.3f}s, efficiency {uav_total/len(drone_data['smokes']):.3f}s/bomb")
            else:
                print(f"  {drone_name}: No valid deployment strategy found")
        
        print(f"\nMissile coverage summary:")
        missile_coverage = {m: 0 for m in MISSILES.keys()}
        for smoke in all_smoke_deployments:
            missile_coverage[smoke["target_missile"]] += smoke["effectiveness"]
        
        for missile_name, coverage_time in missile_coverage.items():
            flight_time = MISSILES[missile_name]["flight_time"]
            coverage_percentage = coverage_time / flight_time * 100
            print(f"  {missile_name}: {coverage_time:.3f}s shielding ({coverage_percentage:.1f}% of flight time)")
        
        print(f"\nGenerated output files:")
        print(f"  - Standard results (Excel): result3.xlsx")
        print(f"  - Comprehensive report (Text): {comprehensive_report}")
        print(f"  - Convergence analysis (PNG): q5_optimization_convergence.png")
        print(f"  - Deployment layout (PNG): q5_final_deployment_layout.png")
        
        print("="*100)
        print("MULTI-UAV OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("All results saved in standard format compatible with problem requirements.")
        print("="*100)
        
    else:
        print("\nOptimization failed: No valid smoke bomb deployment strategy found")
        print("Recommendations:")
        print("  - Check missile and UAV initial positions")
        print("  - Verify speed range and timing constraints") 
        print("  - Consider relaxing optimization parameters")
        print("  - Increase maximum iteration count")
