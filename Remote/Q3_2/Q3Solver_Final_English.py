#!/usr/bin/env python3
"""
Final English Version: Smoke Bomb Deployment Optimization
Optimized PSO algorithm with English comments and standard output format
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for server environment
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ========================== 1. GLOBAL CONSTANTS AND PARAMETERS ==========================
# Physical constants
G = 9.81  # Gravitational acceleration (m/s²)
EPSILON = 1e-12  # Numerical computation protection threshold
DT_FINE = 0.01  # Time step for shielding determination (s)

# Target parameters
FAKE_TARGET = np.array([0.0, 0.0, 0.0])  # Fake target (missile destination)
REAL_TARGET = {
    "center": np.array([0.0, 200.0, 0.0]),  # Real target base center coordinates
    "radius": 7.0,  # Cylinder radius (m)
    "height": 10.0   # Cylinder height (m)
}

# UAV FY1 parameters
FY1_INIT_POS = np.array([17800.0, 0.0, 1800.0])  # Initial position (m)
FY1_SPEED_RANGE = [70.0, 140.0]  # Speed range (m/s)

# Smoke bomb parameters
SMOKE_PARAMS = {
    "radius": 10.0,  # Effective shielding radius (m)
    "sink_speed": 3.0,  # Sinking speed after detonation (m/s)
    "effective_time": 20.0  # Effective duration after detonation (s)
}

# Missile M1 parameters
MISSILE_M1 = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # Initial position (m)
    "speed": 300.0,  # Flight speed (m/s)
}
# Calculate missile direction and arrival time
missile_direction = (FAKE_TARGET - MISSILE_M1["init_pos"]) / np.linalg.norm(FAKE_TARGET - MISSILE_M1["init_pos"])
MISSILE_M1["direction"] = missile_direction
MISSILE_ARRIVAL_TIME = np.linalg.norm(FAKE_TARGET - MISSILE_M1["init_pos"]) / MISSILE_M1["speed"]

print(f"Missile M1 arrival time at fake target: {MISSILE_ARRIVAL_TIME:.2f} seconds")


# ========================== 2. TARGET SAMPLING FUNCTIONS ==========================
def generate_target_samples(target_params, theta_samples=40, height_samples=15):
    """Generate high-density sampling points on and inside the cylindrical target"""
    samples = []
    center_xy = target_params["center"][:2]
    base_z = target_params["center"][2]
    top_z = target_params["center"][2] + target_params["height"]
    radius = target_params["radius"]
    
    # 1. Cylinder surface sampling
    theta_values = np.linspace(0, 2*np.pi, theta_samples, endpoint=False)
    
    # Bottom circle (z = base_z)
    for theta in theta_values:
        x = center_xy[0] + radius * np.cos(theta)
        y = center_xy[1] + radius * np.sin(theta)
        samples.append([x, y, base_z])
    
    # Top circle (z = top_z)
    for theta in theta_values:
        x = center_xy[0] + radius * np.cos(theta)
        y = center_xy[1] + radius * np.sin(theta)
        samples.append([x, y, top_z])
    
    # Side surface (multiple height levels)
    height_values = np.linspace(base_z, top_z, height_samples)
    for z in height_values:
        for theta in theta_values:
            x = center_xy[0] + radius * np.cos(theta)
            y = center_xy[1] + radius * np.sin(theta)
            samples.append([x, y, z])
    
    # 2. Interior sampling (grid points inside cylinder)
    interior_radii = np.linspace(0, radius, 4)
    interior_heights = np.linspace(base_z, top_z, 8)
    interior_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    
    for z in interior_heights:
        for r in interior_radii:
            for theta in interior_angles:
                x = center_xy[0] + r * np.cos(theta)
                y = center_xy[1] + r * np.sin(theta)
                samples.append([x, y, z])
    
    # Remove duplicate points and return
    unique_samples = np.unique(np.array(samples), axis=0)
    print(f"Generated {len(unique_samples)} unique target sampling points")
    return unique_samples


# ========================== 3. GEOMETRIC INTERSECTION FUNCTIONS ==========================
def line_segment_sphere_intersection(point_M, point_P, sphere_center, sphere_radius):
    """
    Check if line segment MP intersects with sphere
    M: missile position, P: target sample point, C: smoke center, r: smoke radius
    """
    MP_vector = point_P - point_M
    MC_vector = sphere_center - point_M
    
    # Dot product calculations
    a = np.dot(MP_vector, MP_vector)
    
    # Handle zero-length segment
    if a < EPSILON:
        return np.linalg.norm(MC_vector) <= sphere_radius + EPSILON
    
    # Quadratic equation coefficients
    b = -2 * np.dot(MP_vector, MC_vector)
    c = np.dot(MC_vector, MC_vector) - sphere_radius**2
    discriminant = b**2 - 4*a*c
    
    # No real solution (no intersection)
    if discriminant < -EPSILON:
        return False
    
    # Handle numerical precision
    discriminant = max(discriminant, 0)
    sqrt_discriminant = np.sqrt(discriminant)
    
    # Calculate intersection parameters
    t1 = (-b - sqrt_discriminant) / (2*a)
    t2 = (-b + sqrt_discriminant) / (2*a)
    
    # Check if intersection occurs within line segment [0,1]
    return (t1 <= 1.0 + EPSILON) and (t2 >= -EPSILON)


def calculate_smoke_shielding_intervals(bomb_parameters, target_sample_points):
    """Calculate time intervals when smoke bomb provides complete shielding"""
    theta, velocity, drop_delay, detonation_delay = bomb_parameters
    
    # 1. Calculate UAV trajectory and drop point
    uav_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_position = FY1_INIT_POS + velocity * drop_delay * uav_direction
    
    # 2. Calculate detonation point (horizontal motion + vertical free fall)
    detonation_xy = drop_position[:2] + velocity * detonation_delay * uav_direction[:2]
    detonation_z = drop_position[2] - 0.5 * G * detonation_delay**2
    
    # Invalid if detonation occurs below ground
    if detonation_z < 0:
        return []
    
    detonation_position = np.array([detonation_xy[0], detonation_xy[1], detonation_z])
    
    # 3. Define smoke effective time window
    detonation_time = drop_delay + detonation_delay
    smoke_start_time = detonation_time
    smoke_end_time = min(detonation_time + SMOKE_PARAMS["effective_time"], MISSILE_ARRIVAL_TIME)
    
    if smoke_start_time >= smoke_end_time:
        return []
    
    # 4. Check shielding status at each time step
    time_points = np.arange(smoke_start_time, smoke_end_time + DT_FINE, DT_FINE)
    shielding_intervals = []
    currently_shielded = False
    interval_start_time = 0
    
    for current_time in time_points:
        # Calculate current missile position
        missile_position = MISSILE_M1["init_pos"] + MISSILE_M1["speed"] * current_time * MISSILE_M1["direction"]
        
        # Calculate current smoke center (sinking effect)
        sink_duration = current_time - detonation_time
        smoke_center = np.array([
            detonation_position[0], 
            detonation_position[1], 
            detonation_position[2] - SMOKE_PARAMS["sink_speed"] * sink_duration
        ])
        
        # Skip if smoke has sunk below ground
        if smoke_center[2] < 0:
            if currently_shielded:
                shielding_intervals.append([interval_start_time, current_time])
                currently_shielded = False
            continue
        
        # Check if all target points are shielded
        complete_shielding = True
        for target_point in target_sample_points:
            if not line_segment_sphere_intersection(missile_position, target_point, 
                                                   smoke_center, SMOKE_PARAMS["radius"]):
                complete_shielding = False
                break
        
        # Update shielding status
        if complete_shielding and not currently_shielded:
            interval_start_time = current_time
            currently_shielded = True
        elif not complete_shielding and currently_shielded:
            shielding_intervals.append([interval_start_time, current_time])
            currently_shielded = False
    
    # Handle final interval if still shielded at end
    if currently_shielded:
        shielding_intervals.append([interval_start_time, smoke_end_time])
    
    return shielding_intervals


def merge_time_intervals(interval_list):
    """Merge overlapping or adjacent time intervals"""
    if not interval_list:
        return 0.0, []
    
    # Sort intervals by start time
    sorted_intervals = sorted(interval_list, key=lambda x: x[0])
    merged_intervals = [sorted_intervals[0]]
    
    for current_interval in sorted_intervals[1:]:
        last_interval = merged_intervals[-1]
        # Merge if overlapping or adjacent
        if current_interval[0] <= last_interval[1] + EPSILON:
            merged_intervals[-1] = [last_interval[0], max(last_interval[1], current_interval[1])]
        else:
            merged_intervals.append(current_interval)
    
    # Calculate total duration
    total_duration = sum([end_time - start_time for start_time, end_time in merged_intervals])
    return total_duration, merged_intervals


# ========================== 4. FITNESS FUNCTION ==========================
def objective_function(optimization_parameters, target_sampling_points):
    """
    Objective function: Total shielding duration of three smoke bombs
    Parameters: [theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
    """
    theta, velocity, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = optimization_parameters
    
    # Constraint checks
    if not (FY1_SPEED_RANGE[0] - EPSILON <= velocity <= FY1_SPEED_RANGE[1] + EPSILON):
        return 0.0  # Speed out of range
    if delta_t2 < 1.0 - EPSILON or delta_t3 < 1.0 - EPSILON:
        return 0.0  # Drop intervals too short
    if any(param < -EPSILON for param in [t1_1, t2_1, t2_2, t2_3]):
        return 0.0  # Negative time parameters
    
    # Calculate timing for three bombs
    t1_2 = t1_1 + delta_t2  # Second bomb drop time
    t1_3 = t1_2 + delta_t3  # Third bomb drop time
    
    # Parameters for each bomb
    bomb1_params = [theta, velocity, t1_1, t2_1]
    bomb2_params = [theta, velocity, t1_2, t2_2]
    bomb3_params = [theta, velocity, t1_3, t2_3]
    
    # Calculate shielding intervals for each bomb
    all_intervals = []
    all_intervals.extend(calculate_smoke_shielding_intervals(bomb1_params, target_sampling_points))
    all_intervals.extend(calculate_smoke_shielding_intervals(bomb2_params, target_sampling_points))
    all_intervals.extend(calculate_smoke_shielding_intervals(bomb3_params, target_sampling_points))
    
    # Merge overlapping intervals and return total duration
    total_duration, _ = merge_time_intervals(all_intervals)
    return total_duration


# ========================== 5. PARTICLE SWARM OPTIMIZATION ==========================
class ParticleSwarmOptimizer:
    """Particle Swarm Optimization implementation for smoke bomb deployment"""
    
    def __init__(self, objective_func, parameter_bounds, num_particles=50, max_iterations=100):
        self.objective_func = objective_func
        self.bounds = parameter_bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.dimension = len(parameter_bounds)
        
        # Initialize particle positions and velocities
        self.positions = np.zeros((num_particles, self.dimension))
        self.velocities = np.zeros((num_particles, self.dimension))
        
        for i in range(self.dimension):
            # Random initial positions within bounds
            self.positions[:, i] = np.random.uniform(
                parameter_bounds[i][0], parameter_bounds[i][1], num_particles
            )
            # Initialize velocities as small random values
            velocity_range = parameter_bounds[i][1] - parameter_bounds[i][0]
            self.velocities[:, i] = 0.1 * np.random.uniform(
                -velocity_range, velocity_range, num_particles
            )
        
        # Initialize personal and global best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([self.objective_func(pos) for pos in self.positions])
        
        # Find global best
        self.global_best_index = np.argmax(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[self.global_best_index].copy()
        self.global_best_fitness = self.personal_best_fitness[self.global_best_index]
        
        # Track optimization history
        self.fitness_history = [self.global_best_fitness]
        
        print(f"PSO initialized with {num_particles} particles")
        print(f"Initial best fitness: {self.global_best_fitness:.4f}")
    
    def optimize(self):
        """Run PSO optimization"""
        print("Starting PSO optimization...")
        
        for iteration in range(self.max_iterations):
            # Dynamic inertia weight (decreases over time)
            inertia_weight = 0.9 - 0.5 * (iteration / self.max_iterations)
            cognitive_factor = 2.0  # Personal best attraction
            social_factor = 2.0     # Global best attraction
            
            # Update each particle
            for i in range(self.num_particles):
                # Calculate current fitness
                current_fitness = self.objective_func(self.positions[i])
                
                # Update personal best
                if current_fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                # Update global best
                if current_fitness > self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = self.positions[i].copy()
                
                # Update velocity
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)
                
                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    cognitive_factor * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                    social_factor * r2 * (self.global_best_position - self.positions[i])
                )
                
                # Update position
                self.positions[i] += self.velocities[i]
                
                # Apply boundary constraints
                for j in range(self.dimension):
                    if self.positions[i][j] < self.bounds[j][0]:
                        self.positions[i][j] = self.bounds[j][0]
                    elif self.positions[i][j] > self.bounds[j][1]:
                        self.positions[i][j] = self.bounds[j][1]
            
            # Record iteration history
            self.fitness_history.append(self.global_best_fitness)
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Fitness: {self.global_best_fitness:.6f} | "
                      f"Inertia Weight: {inertia_weight:.3f}")
        
        print(f"PSO optimization completed. Final best fitness: {self.global_best_fitness:.6f}")
        return self.global_best_position, self.global_best_fitness, self.fitness_history


# ========================== 6. RESULT PROCESSING AND OUTPUT ==========================
def save_standard_excel_format(optimal_parameters, total_shielding_duration, bomb_intervals_data, filename="result_english"):
    """Save optimization results in standard Excel format"""
    
    # Parse optimal parameters
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = optimal_parameters
    
    # Calculate timing for each bomb
    t1_2_opt = t1_1_opt + delta_t2_opt  # Second bomb drop time
    t1_3_opt = t1_2_opt + delta_t3_opt  # Third bomb drop time
    
    # UAV direction vector
    uav_direction = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    
    # Prepare data for Excel export
    excel_data = []
    bomb_timings = [(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]
    
    for bomb_index, (drop_time, detonation_delay) in enumerate(bomb_timings):
        # Calculate positions
        drop_position = FY1_INIT_POS + v_opt * drop_time * uav_direction
        detonation_xy = drop_position[:2] + v_opt * detonation_delay * uav_direction[:2]
        detonation_z = drop_position[2] - 0.5 * G * detonation_delay**2
        detonation_position = np.array([detonation_xy[0], detonation_xy[1], detonation_z])
        
        # Calculate individual shielding duration
        individual_duration = 0.0
        if bomb_index < len(bomb_intervals_data) and bomb_intervals_data[bomb_index]:
            individual_duration = sum([end - start for start, end in bomb_intervals_data[bomb_index]])
        
        # Standard format matching result1.xlsx structure
        excel_data.append({
            "Smoke_Bomb_Number": f"S{bomb_index + 1}",
            "UAV_Number": "FY1",
            "Speed_m_per_s": round(v_opt, 2),
            "Direction_degrees": round(np.degrees(theta_opt), 2),
            "Drop_Time_s": round(drop_time, 2),
            "Detonation_Delay_s": round(detonation_delay, 2),
            "Detonation_Time_s": round(drop_time + detonation_delay, 2),
            "Detonation_Point_X_m": round(detonation_position[0], 2),
            "Detonation_Point_Y_m": round(detonation_position[1], 2),
            "Detonation_Point_Z_m": round(detonation_position[2], 2),
            "Target_Missile": "M1",
            "Effective_Shielding_Duration_s": round(individual_duration, 3)
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    excel_filename = f"{filename}.xlsx"
    df.to_excel(excel_filename, index=False, engine="openpyxl")
    
    print(f"Standard format results saved to: {excel_filename}")
    return df, excel_filename


def save_comprehensive_txt_report(optimal_parameters, total_duration, intervals_data, 
                                 optimization_history, computation_time, filename="comprehensive_results.txt"):
    """Save comprehensive optimization results and analysis to txt file"""
    
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = optimal_parameters
    t1_2_opt = t1_1_opt + delta_t2_opt
    t1_3_opt = t1_2_opt + delta_t3_opt
    uav_direction = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    
    with open(filename, 'w', encoding='utf-8') as file:
        # Header
        file.write("=" * 100 + "\n")
        file.write("COMPREHENSIVE SMOKE BOMB DEPLOYMENT OPTIMIZATION REPORT\n")
        file.write("Algorithm: Enhanced Particle Swarm Optimization (PSO)\n")
        file.write("=" * 100 + "\n")
        file.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Total Computation Time: {computation_time:.3f} seconds\n")
        file.write(f"Total Effective Shielding Duration: {total_duration:.6f} seconds\n")
        file.write("\n")
        
        # Problem setup
        file.write("PROBLEM SETUP:\n")
        file.write("-" * 50 + "\n")
        file.write(f"Real Target: Cylinder at {REAL_TARGET['center']} with radius {REAL_TARGET['radius']}m, height {REAL_TARGET['height']}m\n")
        file.write(f"Fake Target: Point at {FAKE_TARGET}\n")
        file.write(f"Missile M1: Initial position {MISSILE_M1['init_pos']}, speed {MISSILE_M1['speed']}m/s\n")
        file.write(f"UAV FY1: Initial position {FY1_INIT_POS}, speed range {FY1_SPEED_RANGE}m/s\n")
        file.write(f"Smoke Parameters: Radius {SMOKE_PARAMS['radius']}m, sink speed {SMOKE_PARAMS['sink_speed']}m/s, duration {SMOKE_PARAMS['effective_time']}s\n")
        file.write("\n")
        
        # Optimal solution
        file.write("OPTIMAL SOLUTION:\n")
        file.write("-" * 50 + "\n")
        file.write(f"UAV Fixed Speed: {v_opt:.6f} m/s\n")
        file.write(f"UAV Fixed Heading: {theta_opt:.6f} radians ({np.degrees(theta_opt):.3f} degrees)\n")
        file.write(f"UAV Direction Vector: [{uav_direction[0]:.6f}, {uav_direction[1]:.6f}, {uav_direction[2]:.6f}]\n")
        file.write("\n")
        
        # Detailed bomb information
        file.write("DETAILED BOMB DEPLOYMENT PLAN:\n")
        file.write("-" * 50 + "\n")
        
        bomb_timings = [(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]
        total_individual_duration = 0
        
        for bomb_id, (drop_time, det_delay) in enumerate(bomb_timings, 1):
            # Calculate positions
            drop_pos = FY1_INIT_POS + v_opt * drop_time * uav_direction
            det_xy = drop_pos[:2] + v_opt * det_delay * uav_direction[:2]
            det_z = drop_pos[2] - 0.5 * G * det_delay**2
            det_pos = np.array([det_xy[0], det_xy[1], det_z])
            
            # Individual shielding duration
            individual_duration = 0
            if bomb_id - 1 < len(intervals_data):
                individual_duration = sum([end - start for start, end in intervals_data[bomb_id - 1]])
                total_individual_duration += individual_duration
            
            file.write(f"Smoke Bomb {bomb_id}:\n")
            file.write(f"  Drop Time: {drop_time:.6f} s\n")
            file.write(f"  Detonation Delay: {det_delay:.6f} s\n")
            file.write(f"  Detonation Time: {drop_time + det_delay:.6f} s\n")
            file.write(f"  Drop Position: [{drop_pos[0]:.3f}, {drop_pos[1]:.3f}, {drop_pos[2]:.3f}] m\n")
            file.write(f"  Detonation Position: [{det_pos[0]:.3f}, {det_pos[1]:.3f}, {det_pos[2]:.3f}] m\n")
            file.write(f"  Individual Shielding Duration: {individual_duration:.6f} s\n")
            
            if bomb_id - 1 < len(intervals_data):
                file.write(f"  Number of Shielding Intervals: {len(intervals_data[bomb_id - 1])}\n")
                for interval_id, (start, end) in enumerate(intervals_data[bomb_id - 1], 1):
                    file.write(f"    Interval {interval_id}: [{start:.4f}, {end:.4f}] s (duration: {end-start:.4f}s)\n")
            file.write("\n")
        
        # Performance analysis
        file.write("PERFORMANCE ANALYSIS:\n")
        file.write("-" * 50 + "\n")
        file.write(f"Sum of Individual Durations: {total_individual_duration:.6f} s\n")
        file.write(f"Merged Total Duration: {total_duration:.6f} s\n")
        file.write(f"Overlap Reduction: {total_individual_duration - total_duration:.6f} s\n")
        if total_individual_duration > 0:
            overlap_percentage = (total_individual_duration - total_duration) / total_individual_duration * 100
            file.write(f"Overlap Percentage: {overlap_percentage:.2f}%\n")
        file.write("\n")
        
        # Optimization convergence
        file.write("OPTIMIZATION CONVERGENCE HISTORY:\n")
        file.write("-" * 50 + "\n")
        file.write("Iteration | Best Fitness (s)\n")
        file.write("-" * 30 + "\n")
        for i in range(0, len(optimization_history), max(1, len(optimization_history) // 20)):
            file.write(f"{i:8d} | {optimization_history[i]:12.6f}\n")
        file.write("\n")
        
        # Constraint verification
        file.write("CONSTRAINT VERIFICATION:\n")
        file.write("-" * 50 + "\n")
        file.write(f"UAV Speed: {v_opt:.3f} m/s ∈ [{FY1_SPEED_RANGE[0]}, {FY1_SPEED_RANGE[1]}] → ")
        file.write("SATISFIED\n" if FY1_SPEED_RANGE[0] <= v_opt <= FY1_SPEED_RANGE[1] else "VIOLATED\n")
        file.write(f"Drop Interval 1→2: {delta_t2_opt:.3f} s ≥ 1.0 → ")
        file.write("SATISFIED\n" if delta_t2_opt >= 1.0 else "VIOLATED\n")
        file.write(f"Drop Interval 2→3: {delta_t3_opt:.3f} s ≥ 1.0 → ")
        file.write("SATISFIED\n" if delta_t3_opt >= 1.0 else "VIOLATED\n")
        
        # Check detonation heights
        detonation_heights = []
        for drop_time, det_delay in bomb_timings:
            drop_pos = FY1_INIT_POS + v_opt * drop_time * uav_direction
            det_z = drop_pos[2] - 0.5 * G * det_delay**2
            detonation_heights.append(det_z)
        
        all_above_ground = all(z > 0 for z in detonation_heights)
        file.write(f"All Detonations Above Ground: {detonation_heights} → ")
        file.write("SATISFIED\n" if all_above_ground else "VIOLATED\n")
        file.write("\n")
        
        # Footer
        file.write("=" * 100 + "\n")
        file.write("END OF COMPREHENSIVE REPORT\n")
        file.write("=" * 100 + "\n")
    
    print(f"Comprehensive report saved to: {filename}")
    return filename


def create_convergence_plot(fitness_history, save_path="convergence_english.png"):
    """Create optimization convergence plot with English labels"""
    plt.figure(figsize=(12, 8))
    
    # Plot main convergence curve
    iterations = range(len(fitness_history))
    plt.plot(iterations, fitness_history, 'b-', linewidth=2.5, label="Global Best Fitness")
    plt.fill_between(iterations, fitness_history, alpha=0.3, color='lightblue')
    
    # Add performance annotations
    max_fitness = max(fitness_history)
    max_iteration = fitness_history.index(max_fitness)
    final_fitness = fitness_history[-1]
    
    plt.annotate(f'Peak Performance\n{max_fitness:.4f}s at iteration {max_iteration}', 
                xy=(max_iteration, max_fitness), 
                xytext=(max_iteration + len(fitness_history)*0.1, max_fitness + max_fitness*0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Labels and formatting
    plt.xlabel("Iteration Number", fontsize=14, fontweight='bold')
    plt.ylabel("Shielding Duration (seconds)", fontsize=14, fontweight='bold')
    plt.title("PSO Optimization Convergence Curve\nThree Smoke Bombs Deployment Strategy", 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12, loc='lower right')
    
    # Add statistics box
    initial_fitness = fitness_history[0]
    improvement = final_fitness - initial_fitness
    improvement_percentage = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
    
    stats_text = (f"Initial: {initial_fitness:.4f}s\n"
                 f"Final: {final_fitness:.4f}s\n"
                 f"Improvement: {improvement:.4f}s ({improvement_percentage:.1f}%)\n"
                 f"Peak: {max_fitness:.4f}s")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to: {save_path}")


# ========================== 7. MAIN EXECUTION ==========================
def main():
    """Main execution function"""
    print("=" * 80)
    print("SMOKE BOMB DEPLOYMENT OPTIMIZATION - ENGLISH VERSION")
    print("Algorithm: Enhanced Particle Swarm Optimization")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Generate target sampling points
    print("Step 1: Generating target sampling points...")
    target_samples = generate_target_samples(REAL_TARGET, theta_samples=40, height_samples=12)
    
    # Step 2: Define optimization bounds
    print("Step 2: Setting up optimization parameters...")
    optimization_bounds = [
        (0.0, 2*np.pi),      # theta: UAV heading angle (0~360°)
        (70.0, 140.0),       # v: UAV speed (70~140 m/s)
        (0.0, 50.0),         # t1_1: First bomb drop delay (0~50s)
        (0.0, 15.0),         # t2_1: First bomb detonation delay (0~15s)
        (1.0, 25.0),         # delta_t2: Drop interval between bomb 1 and 2 (≥1s)
        (0.0, 15.0),         # t2_2: Second bomb detonation delay (0~15s)
        (1.0, 25.0),         # delta_t3: Drop interval between bomb 2 and 3 (≥1s)
        (0.0, 15.0)          # t2_3: Third bomb detonation delay (0~15s)
    ]
    
    # Step 3: Run PSO optimization
    print("Step 3: Running PSO optimization...")
    
    pso_optimizer = ParticleSwarmOptimizer(
        objective_func=lambda params: objective_function(params, target_samples),
        parameter_bounds=optimization_bounds,
        num_particles=60,    # Balanced particle count
        max_iterations=100   # Sufficient iterations for convergence
    )
    
    optimal_params, best_fitness, fitness_history = pso_optimizer.optimize()
    optimization_time = time.time() - start_time
    
    # Step 4: Calculate detailed results
    print("Step 4: Processing optimization results...")
    
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = optimal_params
    t1_2_opt = t1_1_opt + delta_t2_opt
    t1_3_opt = t1_2_opt + delta_t3_opt
    
    # Calculate shielding intervals for each bomb
    bomb_parameters_list = [
        [theta_opt, v_opt, t1_1_opt, t2_1_opt],
        [theta_opt, v_opt, t1_2_opt, t2_2_opt],
        [theta_opt, v_opt, t1_3_opt, t2_3_opt]
    ]
    
    intervals_data = []
    for bomb_params in bomb_parameters_list:
        intervals = calculate_smoke_shielding_intervals(bomb_params, target_samples)
        intervals_data.append(intervals)
    
    # Calculate total merged duration
    all_intervals = []
    for intervals in intervals_data:
        all_intervals.extend(intervals)
    total_duration, merged_intervals = merge_time_intervals(all_intervals)
    
    # Step 5: Output results
    print("Step 5: Saving results...")
    
    # Save Excel format
    df, excel_file = save_standard_excel_format(
        optimal_params, total_duration, intervals_data, filename="result_final_english"
    )
    
    # Save comprehensive txt report
    txt_file = save_comprehensive_txt_report(
        optimal_params, total_duration, intervals_data, fitness_history, optimization_time,
        filename="optimization_comprehensive_report.txt"
    )
    
    # Create convergence plot
    create_convergence_plot(fitness_history, save_path="convergence_final_english.png")
    
    # Step 6: Display summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Total Computation Time: {optimization_time:.2f} seconds")
    print(f"Final Shielding Duration: {total_duration:.6f} seconds")
    print(f"Optimal UAV Speed: {v_opt:.3f} m/s")
    print(f"Optimal UAV Heading: {np.degrees(theta_opt):.2f}°")
    print(f"Number of Smoke Bombs: 3")
    
    # Individual bomb performance
    print(f"\nIndividual Bomb Performance:")
    for i, intervals in enumerate(intervals_data, 1):
        individual_duration = sum([end - start for start, end in intervals])
        print(f"  Bomb {i}: {individual_duration:.4f}s ({len(intervals)} intervals)")
    
    # Calculate overlap statistics
    individual_sum = sum([sum([end - start for start, end in intervals]) for intervals in intervals_data])
    overlap_reduction = individual_sum - total_duration
    overlap_percentage = (overlap_reduction / individual_sum * 100) if individual_sum > 0 else 0
    
    print(f"\nOverlap Analysis:")
    print(f"  Sum of Individual Durations: {individual_sum:.6f}s")
    print(f"  Merged Total Duration: {total_duration:.6f}s")
    print(f"  Overlap Reduction: {overlap_reduction:.6f}s ({overlap_percentage:.2f}%)")
    
    print(f"\nGenerated Files:")
    print(f"  - Excel Results: {excel_file}")
    print(f"  - Detailed Report: {txt_file}")
    print(f"  - Convergence Plot: convergence_final_english.png")
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return {
        "total_duration": total_duration,
        "optimization_time": optimization_time,
        "optimal_params": optimal_params,
        "excel_file": excel_file,
        "txt_file": txt_file
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\nFinal Result: {results['total_duration']:.6f} seconds shielding duration")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
