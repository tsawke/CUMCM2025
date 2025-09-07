import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend for headless operation
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import time
import os

# -------------------------- 1. Constants and Basic Parameters --------------------------
g = 9.81  # Gravitational acceleration (m/s²)
epsilon = 1e-12  # Numerical computation protection threshold
dt_fine = 0.01  # Time step for shielding determination
n_jobs = max(1, multiprocessing.cpu_count() - 2)  # Reserve 2 cores to avoid system lag

# Target parameters
fake_target = np.array([0.0, 0.0, 0.0])  # Fake target (missile direction)
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # Real target base center
    "r": 7.0,  # Cylinder radius
    "h": 10.0   # Cylinder height
}

# UAV FY1 initial parameters
fy1_init_pos = np.array([17800.0, 0.0, 1800.0])  # Initial position

# Smoke parameters
smoke_param = {
    "r": 10.0,  # Effective shielding radius (m)
    "sink_speed": 3.0,  # Sinking speed (m/s)
    "valid_time": 20.0  # Single smoke effective time (s)
}

# Missile M1 parameters
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # Initial position
    "speed": 300.0,  # Flight speed (m/s)
    "dir": (fake_target - np.array([20000.0, 0.0, 2000.0])) / 
           np.linalg.norm(fake_target - np.array([20000.0, 0.0, 2000.0]))  # Flight direction
}
missile_arrival_time = np.linalg.norm(fake_target - missile_param["init_pos"]) / missile_param["speed"]  # Missile arrival time at fake target


# -------------------------- 2. High-density Target Sampling --------------------------
def generate_target_samples(target, num_theta=60, num_height=20):
    """Generate real target surface + internal sampling points for complete shielding determination"""
    samples = []
    center_xy = target["center"][:2]
    min_z = target["center"][2]
    max_z = target["center"][2] + target["h"]
    
    # 1. Cylinder outer surface (bottom, top, side)
    thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    # Bottom surface (z=min_z)
    for th in thetas:
        x = center_xy[0] + target["r"] * np.cos(th)
        y = center_xy[1] + target["r"] * np.sin(th)
        samples.append([x, y, min_z])
    # Top surface (z=max_z)
    for th in thetas:
        x = center_xy[0] + target["r"] * np.cos(th)
        y = center_xy[1] + target["r"] * np.sin(th)
        samples.append([x, y, max_z])
    # Side surface (uniform height layers)
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    for z in heights:
        for th in thetas:
            x = center_xy[0] + target["r"] * np.cos(th)
            y = center_xy[1] + target["r"] * np.sin(th)
            samples.append([x, y, z])
    
    # 2. Cylinder interior (grid sampling)
    inner_radii = np.linspace(0, target["r"], 5, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    for z in inner_heights:
        for rad in inner_radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    return np.unique(np.array(samples), axis=0)  # Remove duplicates


# -------------------------- 3. Core Geometric Calculation and Shielding Determination --------------------------
def segment_sphere_intersect(M, P, C, r):
    """Determine if line segment MP (missile position M → target sample point P) intersects with sphere C(r) (smoke)"""
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)
    
    # Zero-length segment (M=P)
    if a < epsilon:
        return np.linalg.norm(MC) <= r + epsilon
    
    # Solve quadratic equation
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c
    
    # No real roots (no intersection)
    if discriminant < -epsilon:
        return False
    # Handle numerical error
    discriminant = max(discriminant, 0)
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)
    
    # Intersection exists within [0,1] interval
    return (s1 <= 1.0 + epsilon) and (s2 >= -epsilon)


def get_smoke_shield_interval(bomb_params, target_samples):
    """Calculate effective shielding time interval [start, end] for single smoke bomb"""
    theta, v, t1, t2 = bomb_params
    
    # 1. Calculate drop point (UAV fixed heading θ, speed v)
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])  # UAV flight direction
    drop_point = fy1_init_pos + v * t1 * uav_dir
    
    # 2. Calculate detonation point (horizontal along UAV direction after drop, vertical free fall)
    det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t2**2
    # Detonation height too low (invalid)
    if det_z < 0:
        return []
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    
    # 3. Smoke effective time window (20s after detonation, not exceeding missile arrival time at fake target)
    t_det = t1 + t2  # Detonation time
    t_smoke_start = t_det
    t_smoke_end = min(t_det + smoke_param["valid_time"], missile_arrival_time)
    if t_smoke_start >= t_smoke_end:
        return []
    
    # 4. Determine shielding status at each time step, record effective time intervals
    t_list = np.arange(t_smoke_start, t_smoke_end + dt_fine, dt_fine)
    shield_intervals = []
    in_shield = False
    interval_start = 0
    
    for t in t_list:
        # Current missile position
        missile_pos = missile_param["init_pos"] + missile_param["speed"] * t * missile_param["dir"]
        
        # Current smoke center (xy fixed, z sinking)
        sink_time = t - t_det
        smoke_center = np.array([det_point[0], det_point[1], det_point[2] - smoke_param["sink_speed"]*sink_time])
        # Smoke hits ground (invalid)
        if smoke_center[2] < 0:
            if in_shield:
                shield_intervals.append([interval_start, t])
                in_shield = False
            continue
        
        # Determine if completely shielded (all sample points blocked by smoke)
        fully_shielded = True
        for p in target_samples:
            if not segment_sphere_intersect(missile_pos, p, smoke_center, smoke_param["r"]):
                fully_shielded = False
                break
        
        # Update shielding status
        if fully_shielded and not in_shield:
            interval_start = t
            in_shield = True
        elif not fully_shielded and in_shield:
            shield_intervals.append([interval_start, t])
            in_shield = False
    
    # Handle last unfinished interval
    if in_shield:
        shield_intervals.append([interval_start, t_smoke_end])
    
    return shield_intervals


def merge_intervals(intervals):
    """Merge overlapping/adjacent shielding time intervals, calculate total duration"""
    if not intervals:
        return 0.0, []  # Always return tuple (total duration, merged intervals)
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + epsilon:  # Overlapping or adjacent
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    # Calculate total duration
    total = sum([end - start for start, end in merged])
    return total, merged  # Ensure tuple return


# -------------------------- 4. Fitness Function (3 bombs total shielding duration) --------------------------
def fitness_function(params, target_samples):
    """
    Fitness = Total shielding duration after merging 3 bombs
    params: [theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
    """
    theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
    
    # Constraint 1: UAV speed within 70~140m/s
    if not (70.0 - epsilon <= v <= 140.0 + epsilon):
        return 0.0
    # Constraint 2: Drop interval ≥1s
    if delta_t2 < 1.0 - epsilon or delta_t3 < 1.0 - epsilon:
        return 0.0
    # Constraint 3: Drop/detonation delays non-negative
    if t1_1 < -epsilon or t2_1 < -epsilon or t2_2 < -epsilon or t2_3 < -epsilon:
        return 0.0
    
    # Calculate parameters for 3 bombs (fixed theta and v)
    t1_2 = t1_1 + delta_t2  # Second bomb drop delay
    t1_3 = t1_2 + delta_t3  # Third bomb drop delay
    bomb1_params = [theta, v, t1_1, t2_1]
    bomb2_params = [theta, v, t1_2, t2_2]
    bomb3_params = [theta, v, t1_3, t2_3]
    
    # Calculate shielding intervals for 3 bombs (serial computation to avoid memory overflow)
    all_intervals = []
    all_intervals.extend(get_smoke_shield_interval(bomb1_params, target_samples))
    all_intervals.extend(get_smoke_shield_interval(bomb2_params, target_samples))
    all_intervals.extend(get_smoke_shield_interval(bomb3_params, target_samples))
    
    # Merge intervals and return total duration
    total_duration, _ = merge_intervals(all_intervals)
    return total_duration


# -------------------------- 5. Particle Swarm Optimization (PSO) Implementation --------------------------
class PSOOptimizer:
    def __init__(self, obj_func, bounds, num_particles=50, max_iter=120):
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        
        # Initialize particle positions and velocities
        self.pos = np.zeros((num_particles, self.dim))
        self.vel = np.zeros((num_particles, self.dim))
        for i in range(self.dim):
            self.pos[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], num_particles)
            vel_range = bounds[i][1] - bounds[i][0]
            self.vel[:, i] = 0.1 * np.random.uniform(-vel_range, vel_range, num_particles)
        
        # Initialize personal best (batch computation to avoid memory peaks)
        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.zeros(num_particles)
        batch_size = 10  # Batch processing to reduce memory usage
        for i in range(0, num_particles, batch_size):
            end_idx = min(i + batch_size, num_particles)
            self.pbest_fit[i:end_idx] = Parallel(n_jobs=n_jobs)(
                delayed(self.obj_func)(self.pos[j]) for j in range(i, end_idx)
            )
        
        # Initialize global best
        self.gbest_idx = np.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_fit = self.pbest_fit[self.gbest_idx]
        
        # Record iteration history
        self.gbest_history = [self.gbest_fit]
    
    def update(self):
        """Iterative particle update"""
        for iter in range(self.max_iter):
            # Linear decrease of inertia weight (0.9→0.4)
            w = 0.9 - 0.5 * (iter / self.max_iter)
            c1, c2 = 2.0, 2.0  # Cognitive/social factors
            
            # Batch fitness calculation to avoid memory overflow
            fit_values = np.zeros(self.num_particles)
            batch_size = 10
            for i in range(0, self.num_particles, batch_size):
                end_idx = min(i + batch_size, self.num_particles)
                fit_values[i:end_idx] = Parallel(n_jobs=n_jobs)(
                    delayed(self.obj_func)(self.pos[j]) for j in range(i, end_idx)
                )
            
            # Update personal best and global best
            for i in range(self.num_particles):
                if fit_values[i] > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit_values[i]
                    self.pbest_pos[i] = self.pos[i].copy()
                if fit_values[i] > self.gbest_fit:
                    self.gbest_fit = fit_values[i]
                    self.gbest_pos = self.pos[i].copy()
            
            # Update velocity and position
            for i in range(self.num_particles):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                # Velocity update formula
                self.vel[i] = w * self.vel[i] + c1*r1*(self.pbest_pos[i]-self.pos[i]) + c2*r2*(self.gbest_pos-self.pos[i])
                # Position update and boundary constraint
                self.pos[i] = self.pos[i] + self.vel[i]
                for j in range(self.dim):
                    self.pos[i][j] = max(self.bounds[j][0], min(self.pos[i][j], self.bounds[j][1]))
            
            # Record history and print information
            self.gbest_history.append(self.gbest_fit)
            if (iter + 1) % 10 == 0:
                print(f"Iteration {iter+1:3d}/{self.max_iter} | Best Fitness: {self.gbest_fit:.4f} | Global Best Duration: {self.gbest_history[-1]:.4f}s")
        
        return self.gbest_pos, self.gbest_fit, self.gbest_history


# -------------------------- 6. Result Output Functions --------------------------
def save_results_standard_format(best_params, total_duration, intervals_data, filename_base="result"):
    """Save results in standard format (compatible with result1_v5.xlsx format)"""
    
    # Parse optimal parameters
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
    t1_2_opt = t1_1_opt + delta_t2_opt  # Second bomb drop delay
    t1_3_opt = t1_2_opt + delta_t3_opt  # Third bomb drop delay
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])  # Optimal UAV heading
    
    # Calculate drop points and detonation points for 3 bombs
    bombs_data = []
    for i, (t1, t2) in enumerate([(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]):
        drop_point = fy1_init_pos + v_opt * t1 * uav_dir_opt
        det_xy = drop_point[:2] + v_opt * t2 * uav_dir_opt[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        det_point = np.array([det_xy[0], det_xy[1], det_z])
        
        # Calculate individual shielding duration
        individual_duration = 0
        if i < len(intervals_data):
            individual_duration = sum([end - start for start, end in intervals_data[i]])
        
        bombs_data.append({
            "Serial_Number": i + 1,
            "UAV_Number": "FY1", 
            "Speed_ms": round(v_opt, 2),
            "Direction_deg": round(np.degrees(theta_opt), 2),
            "Drop_Time_s": round(t1, 2),
            "Detonation_Delay_s": round(t2, 2),
            "Detonation_Time_s": round(t1 + t2, 2),
            "Detonation_X_m": round(det_point[0], 2),
            "Detonation_Y_m": round(det_point[1], 2),
            "Detonation_Z_m": round(det_point[2], 2),
            "Target_Missile": "M1",
            "Effective_Duration_s": round(individual_duration, 3)
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(bombs_data)
    excel_filename = f"{filename_base}.xlsx"
    df.to_excel(excel_filename, index=False, engine="openpyxl")
    print(f"Results saved to {excel_filename}")
    
    return df, excel_filename


def save_detailed_results_to_txt(best_params, total_duration, intervals_data, gbest_history, optimization_time, filename="optimization_results_detailed.txt"):
    """Save all optimization information and optimal solution to txt file as backup"""
    
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
    t1_2_opt = t1_1_opt + delta_t2_opt
    t1_3_opt = t1_2_opt + delta_t3_opt
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SMOKE BOMB DEPLOYMENT OPTIMIZATION RESULTS - DETAILED REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Optimization Time: {optimization_time:.2f} seconds\n")
        f.write(f"Total Effective Shielding Duration: {total_duration:.6f} seconds\n")
        f.write(f"Algorithm: Particle Swarm Optimization (PSO)\n")
        f.write(f"Target: Real cylindrical target (center=[0, 200, 0], radius=7m, height=10m)\n")
        f.write(f"Missile: M1 (initial_pos=[20000, 0, 2000], speed=300m/s)\n")
        f.write(f"UAV: FY1 (initial_pos=[17800, 0, 1800], speed_range=[70, 140]m/s)\n")
        f.write("\n")
        
        f.write("OPTIMAL UAV PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Fixed Speed: {v_opt:.6f} m/s\n")
        f.write(f"Fixed Heading: {theta_opt:.6f} rad ({np.degrees(theta_opt):.3f}°)\n")
        f.write(f"Flight Direction Vector: [{uav_dir_opt[0]:.6f}, {uav_dir_opt[1]:.6f}, {uav_dir_opt[2]:.6f}]\n")
        f.write("\n")
        
        f.write("DETAILED BOMB PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        
        bomb_times = [(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]
        for i, (t1, t2) in enumerate(bomb_times):
            f.write(f"Bomb {i+1}:\n")
            
            # Calculate positions
            drop_point = fy1_init_pos + v_opt * t1 * uav_dir_opt
            det_xy = drop_point[:2] + v_opt * t2 * uav_dir_opt[:2]
            det_z = drop_point[2] - 0.5 * g * t2**2
            det_point = np.array([det_xy[0], det_xy[1], det_z])
            
            f.write(f"  Drop Time: {t1:.6f} s\n")
            f.write(f"  Detonation Delay: {t2:.6f} s\n")
            f.write(f"  Detonation Time: {t1 + t2:.6f} s\n")
            f.write(f"  Drop Point: [{drop_point[0]:.3f}, {drop_point[1]:.3f}, {drop_point[2]:.3f}] m\n")
            f.write(f"  Detonation Point: [{det_point[0]:.3f}, {det_point[1]:.3f}, {det_point[2]:.3f}] m\n")
            
            if i < len(intervals_data):
                f.write(f"  Shielding Intervals: {len(intervals_data[i])}\n")
                for j, interval in enumerate(intervals_data[i]):
                    f.write(f"    Interval {j+1}: [{interval[0]:.3f}, {interval[1]:.3f}] s (duration: {interval[1]-interval[0]:.3f}s)\n")
            f.write("\n")
        
        f.write("OPTIMIZATION CONVERGENCE HISTORY:\n")
        f.write("-" * 40 + "\n")
        f.write("Iteration | Best Fitness\n")
        for i, fitness in enumerate(gbest_history[::10]):  # Every 10th iteration
            f.write(f"{i*10:8d} | {fitness:.6f}\n")
        f.write("\n")
        
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        individual_durations = []
        for intervals in intervals_data:
            duration, _ = merge_intervals(intervals)
            individual_durations.append(duration)
        
        f.write(f"Individual Bomb Durations: {[f'{d:.3f}s' for d in individual_durations]}\n")
        f.write(f"Sum of Individual Durations: {sum(individual_durations):.6f} s\n")
        f.write(f"Merged Total Duration: {total_duration:.6f} s\n")
        f.write(f"Overlap Reduction: {sum(individual_durations) - total_duration:.6f} s\n")
        if sum(individual_durations) > 0:
            overlap_rate = (sum(individual_durations) - total_duration) / sum(individual_durations) * 100
            f.write(f"Overlap Rate: {overlap_rate:.2f}%\n")
        f.write("\n")
        
        f.write("CONSTRAINT VERIFICATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"UAV Speed Constraint: 70 ≤ {v_opt:.3f} ≤ 140 m/s → {'PASS' if 70 <= v_opt <= 140 else 'FAIL'}\n")
        f.write(f"Drop Interval 1-2: {delta_t2_opt:.3f} ≥ 1.0 s → {'PASS' if delta_t2_opt >= 1.0 else 'FAIL'}\n")
        f.write(f"Drop Interval 2-3: {delta_t3_opt:.3f} ≥ 1.0 s → {'PASS' if delta_t3_opt >= 1.0 else 'FAIL'}\n")
        f.write(f"All Detonation Heights > 0 → {'PASS' if all(det_z > 0 for det_z in [drop_point[2] - 0.5*g*t2**2 for t2 in [t2_1_opt, t2_2_opt, t2_3_opt]]) else 'FAIL'}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF DETAILED REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Detailed results saved to {filename}")
    return filename


# -------------------------- 7. Visualization Functions --------------------------
def plot_convergence_curve(gbest_history, save_path="convergence_curve_english.png"):
    """Plot optimization convergence curve with English labels"""
    plt.figure(figsize=(12, 6))
    
    # Set English font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot convergence curve
    iterations = range(len(gbest_history))
    plt.plot(iterations, gbest_history, 'b-', linewidth=2, label="Global Best Shielding Duration")
    plt.fill_between(iterations, gbest_history, alpha=0.3, color='lightblue')
    
    # Add annotations
    max_fitness = max(gbest_history)
    max_iter = gbest_history.index(max_fitness)
    plt.annotate(f'Best: {max_fitness:.4f}s\nat iteration {max_iter}', 
                xy=(max_iter, max_fitness), xytext=(max_iter + 10, max_fitness + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Shielding Duration (seconds)", fontsize=12)
    plt.title("PSO Optimization Convergence Curve\nThree Smoke Bombs Deployment Strategy", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add statistics text box
    final_fitness = gbest_history[-1]
    improvement = final_fitness - gbest_history[0] if gbest_history[0] > 0 else 0
    stats_text = f"Initial: {gbest_history[0]:.4f}s\nFinal: {final_fitness:.4f}s\nImprovement: {improvement:.4f}s"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to save memory
    print(f"Convergence curve saved to {save_path}")


def plot_deployment_layout(best_params, intervals_data, save_path="deployment_layout_english.png"):
    """Plot UAV and smoke bomb deployment layout with English labels"""
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
    t1_2_opt = t1_1_opt + delta_t2_opt
    t1_3_opt = t1_2_opt + delta_t3_opt
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    
    # Set English font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Top view (X-Y plane)
    ax1.set_aspect('equal')
    
    # Draw real target
    target_circle = plt.Circle((real_target["center"][0], real_target["center"][1]), 
                              real_target["r"], fill=False, color='red', linewidth=2, label='Real Target')
    ax1.add_patch(target_circle)
    ax1.scatter(real_target["center"][0], real_target["center"][1], c='red', s=100, marker='*', label='Target Center')
    
    # Draw fake target
    ax1.scatter(fake_target[0], fake_target[1], c='black', s=80, marker='x', label='Fake Target')
    
    # Draw missile trajectory
    missile_traj_x = [missile_param["init_pos"][0], fake_target[0]]
    missile_traj_y = [missile_param["init_pos"][1], fake_target[1]]
    ax1.plot(missile_traj_x, missile_traj_y, 'g--', linewidth=2, label='Missile M1 Trajectory')
    ax1.scatter(missile_param["init_pos"][0], missile_param["init_pos"][1], c='green', s=100, marker='^', label='Missile M1 Start')
    
    # Draw UAV trajectory
    max_flight_time = max(t1_1_opt, t1_2_opt, t1_3_opt) + 5  # Add some buffer
    uav_end_pos = fy1_init_pos + v_opt * max_flight_time * uav_dir_opt
    ax1.plot([fy1_init_pos[0], uav_end_pos[0]], [fy1_init_pos[1], uav_end_pos[1]], 
             'b-', linewidth=2, label='UAV FY1 Trajectory')
    ax1.scatter(fy1_init_pos[0], fy1_init_pos[1], c='blue', s=100, marker='s', label='UAV FY1 Start')
    
    # Draw smoke bomb positions
    colors = ['orange', 'purple', 'brown']
    bomb_times = [(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]
    for i, (t1, t2) in enumerate(bomb_times):
        drop_point = fy1_init_pos + v_opt * t1 * uav_dir_opt
        det_xy = drop_point[:2] + v_opt * t2 * uav_dir_opt[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        
        # Drop point
        ax1.scatter(drop_point[0], drop_point[1], c=colors[i], s=60, marker='o', 
                   label=f'Bomb {i+1} Drop Point')
        # Detonation point
        ax1.scatter(det_xy[0], det_xy[1], c=colors[i], s=80, marker='D', 
                   label=f'Bomb {i+1} Detonation Point')
        # Smoke effective radius
        smoke_circle = plt.Circle((det_xy[0], det_xy[1]), smoke_param["r"], 
                                 fill=False, color=colors[i], linestyle=':', alpha=0.7)
        ax1.add_patch(smoke_circle)
    
    ax1.set_xlabel("X Coordinate (m)", fontsize=12)
    ax1.set_ylabel("Y Coordinate (m)", fontsize=12)
    ax1.set_title("Smoke Bomb Deployment Layout (Top View)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Shielding time analysis
    bomb_labels = [f'Bomb {i+1}' for i in range(3)]
    individual_durations = []
    for intervals in intervals_data:
        duration, _ = merge_intervals(intervals)
        individual_durations.append(duration)
    
    bars = ax2.bar(bomb_labels, individual_durations, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Shielding Duration (seconds)", fontsize=12)
    ax2.set_title("Individual Bomb Shielding Performance", fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, duration in zip(bars, individual_durations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{duration:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Add total duration line
    ax2.axhline(y=total_duration, color='red', linestyle='--', linewidth=2, 
               label=f'Merged Total: {total_duration:.3f}s')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to save memory
    print(f"Deployment layout saved to {save_path}")


# -------------------------- 8. Main Program: Optimization and Result Output --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # Step 1: Generate real target sampling points
    print("Generating real target sampling points...")
    target_samples = generate_target_samples(real_target)
    print(f"Number of sampling points: {len(target_samples)}")
    
    # Step 2: Define optimization variable bounds
    # [theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
    bounds = [
        (0.0, 2*np.pi),    # theta: UAV heading angle (0~360°)
        (70.0, 140.0),     # v: UAV speed (70~140m/s)
        (0.0, 60.0),       # t1_1: First bomb drop delay (0~60s)
        (0.0, 20.0),       # t2_1: First bomb detonation delay (0~20s)
        (1.0, 30.0),       # delta_t2: 1st~2nd bomb drop interval (≥1s)
        (0.0, 20.0),       # t2_2: Second bomb detonation delay (0~20s)
        (1.0, 30.0),       # delta_t3: 2nd~3rd bomb drop interval (≥1s)
        (0.0, 20.0)        # t2_3: Third bomb detonation delay (0~20s)
    ]
    
    # Step 3: Initialize PSO and optimize
    print("\nStarting Particle Swarm Optimization...")
    try:
        pso = PSOOptimizer(
            obj_func=lambda p: fitness_function(p, target_samples),
            bounds=bounds,
            num_particles=50,  # Reduced particle count to lower computational load
            max_iter=120
        )
        best_params, best_fitness, gbest_history = pso.update()
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        exit(1)
    
    # Step 4: Parse optimal solution
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
    t1_2_opt = t1_1_opt + delta_t2_opt  # Second bomb drop delay
    t1_3_opt = t1_2_opt + delta_t3_opt  # Third bomb drop delay
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])  # Optimal UAV heading
    
    # Calculate drop points and detonation points for 3 bombs
    bombs_info = []
    intervals_data = []
    
    bomb_times = [(t1_1_opt, t2_1_opt), (t1_2_opt, t2_2_opt), (t1_3_opt, t2_3_opt)]
    for i, (t1, t2) in enumerate(bomb_times):
        # Drop point
        drop_point = fy1_init_pos + v_opt * t1 * uav_dir_opt
        # Detonation point
        det_xy = drop_point[:2] + v_opt * t2 * uav_dir_opt[:2]
        det_z = drop_point[2] - 0.5 * g * t2**2
        det_point = np.array([det_xy[0], det_xy[1], det_z])
        
        # Calculate shielding intervals
        bomb_params = [theta_opt, v_opt, t1, t2]
        intervals = get_smoke_shield_interval(bomb_params, target_samples)
        intervals_data.append(intervals)
        
        bombs_info.append({
            "bomb_id": i + 1,
            "drop_point": drop_point,
            "det_point": det_point,
            "drop_time": t1,
            "det_delay": t2,
            "det_time": t1 + t2,
            "intervals": intervals
        })
    
    # Calculate overlap statistics
    all_intervals = []
    for intervals in intervals_data:
        all_intervals.extend(intervals)
    total_duration, merged_intervals = merge_intervals(all_intervals)
    
    individual_total = sum([sum([end-start for start, end in intervals]) for intervals in intervals_data])
    overlap_rate = 0.0 if individual_total < epsilon else (individual_total - total_duration) / individual_total * 100
    
    # Step 5: Output results
    end_time = time.time()
    optimization_time = end_time - start_time
    
    print("\n" + "="*80)
    print("FY1 THREE SMOKE BOMBS DEPLOYMENT STRATEGY OPTIMIZATION RESULTS")
    print(f"Total Optimization Time: {optimization_time:.2f} s")
    print(f"Total Effective Shielding Duration: {total_duration:.4f} s")
    print(f"Three Bombs Shielding Overlap Rate: {overlap_rate:.2f}%")
    print(f"UAV Fixed Speed: {v_opt:.4f} m/s")
    print(f"UAV Fixed Heading: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f}°)")
    print("="*80)
    
    print("\nDETAILED BOMB PARAMETERS")
    for i, info in enumerate(bombs_info):
        print(f"Bomb {i+1}:")
        print(f"  Drop Point: ({info['drop_point'][0]:.2f}, {info['drop_point'][1]:.2f}, {info['drop_point'][2]:.2f})")
        print(f"  Detonation Point: ({info['det_point'][0]:.2f}, {info['det_point'][1]:.2f}, {info['det_point'][2]:.2f})")
        print(f"  Drop Delay: {info['drop_time']:.2f}s | Detonation Delay: {info['det_delay']:.2f}s")
        print(f"  Number of Shielding Intervals: {len(info['intervals'])}")
        print()
    
    # Step 6: Save results to Excel (standard format)
    try:
        df, excel_filename = save_results_standard_format(
            best_params, total_duration, intervals_data, filename_base="result_optimized"
        )
        print(f"Standard format results saved to {excel_filename}")
    except Exception as e:
        print(f"Failed to save Excel: {str(e)}")
    
    # Step 7: Save detailed results to txt
    try:
        txt_filename = save_detailed_results_to_txt(
            best_params, total_duration, intervals_data, gbest_history, optimization_time
        )
    except Exception as e:
        print(f"Failed to save detailed txt: {str(e)}")
    
    # Step 8: Generate visualizations
    try:
        plot_convergence_curve(gbest_history)
        plot_deployment_layout(best_params, intervals_data)
    except Exception as e:
        print(f"Failed to generate plots: {str(e)}")
    
    print("\nOptimization completed successfully!")
    print(f"Files generated:")
    print(f"  - Excel: result_optimized.xlsx")
    print(f"  - Detailed report: optimization_results_detailed.txt") 
    print(f"  - Convergence plot: convergence_curve_english.png")
    print(f"  - Layout plot: deployment_layout_english.png")
