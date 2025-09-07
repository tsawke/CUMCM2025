#!/usr/bin/env python3
"""
Simplified test version for smoke bomb optimization
Reduced computational complexity for quick validation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ========================== CONSTANTS ==========================
G = 9.81
EPSILON = 1e-10
DT = 0.05  # Larger time step for faster computation

# Target and missile parameters
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
REAL_TARGET = {"center": np.array([0.0, 200.0, 0.0]), "radius": 7.0, "height": 10.0}
FY1_INIT = np.array([17800.0, 0.0, 1800.0])
MISSILE_INIT = np.array([20000.0, 0.0, 2000.0])
MISSILE_SPEED = 300.0
MISSILE_DIR = (FAKE_TARGET - MISSILE_INIT) / np.linalg.norm(FAKE_TARGET - MISSILE_INIT)
MISSILE_ARRIVAL = np.linalg.norm(FAKE_TARGET - MISSILE_INIT) / MISSILE_SPEED

SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0

print(f"Missile arrival time: {MISSILE_ARRIVAL:.2f}s")

# ========================== SIMPLIFIED FUNCTIONS ==========================
def generate_simple_target_samples():
    """Generate simplified target sampling (fewer points for speed)"""
    samples = []
    center = REAL_TARGET["center"]
    radius = REAL_TARGET["radius"]
    height = REAL_TARGET["height"]
    
    # Surface points only (simplified)
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)  # Reduced from 60 to 20
    heights = np.linspace(center[2], center[2] + height, 8)  # Reduced from 20 to 8
    
    # Bottom and top circles
    for theta in angles:
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        samples.append([x, y, center[2]])  # Bottom
        samples.append([x, y, center[2] + height])  # Top
    
    # Side surface
    for z in heights:
        for theta in angles:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            samples.append([x, y, z])
    
    # Center line (simplified interior)
    for z in heights:
        samples.append([center[0], center[1], z])
    
    return np.array(samples)

def line_sphere_intersect(M, P, C, r):
    """Check line-sphere intersection (simplified)"""
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)
    
    if a < EPSILON:
        return np.linalg.norm(MC) <= r
    
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return False
    
    sqrt_d = np.sqrt(max(discriminant, 0))
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    
    return (t1 <= 1.0) and (t2 >= 0.0)

def calculate_shielding_duration(bomb_params, target_points):
    """Calculate total shielding duration for one bomb (simplified)"""
    theta, v, drop_time, det_delay = bomb_params
    
    # UAV direction and positions
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_pos = FY1_INIT + v * drop_time * uav_dir
    det_xy = drop_pos[:2] + v * det_delay * uav_dir[:2]
    det_z = drop_pos[2] - 0.5 * G * det_delay**2
    
    if det_z < 0:
        return 0.0
    
    det_pos = np.array([det_xy[0], det_xy[1], det_z])
    
    # Time window
    det_time = drop_time + det_delay
    start_time = det_time
    end_time = min(det_time + SMOKE_DURATION, MISSILE_ARRIVAL)
    
    if start_time >= end_time:
        return 0.0
    
    # Check shielding at discrete time points (simplified)
    time_points = np.arange(start_time, end_time, DT)
    shielded_duration = 0.0
    
    for t in time_points:
        # Missile position
        missile_pos = MISSILE_INIT + MISSILE_SPEED * t * MISSILE_DIR
        
        # Smoke position (sinking)
        sink_time = t - det_time
        smoke_center = np.array([det_pos[0], det_pos[1], det_pos[2] - SMOKE_SINK_SPEED * sink_time])
        
        if smoke_center[2] < 0:
            continue
        
        # Check if all points are shielded
        all_shielded = True
        for point in target_points:
            if not line_sphere_intersect(missile_pos, point, smoke_center, SMOKE_RADIUS):
                all_shielded = False
                break
        
        if all_shielded:
            shielded_duration += DT
    
    return shielded_duration

def simple_fitness(params, target_points):
    """Simplified fitness function for three bombs"""
    theta, v, t1_1, t2_1, dt2, t2_2, dt3, t2_3 = params
    
    # Constraints
    if not (70.0 <= v <= 140.0):
        return 0.0
    if dt2 < 1.0 or dt3 < 1.0:
        return 0.0
    if any(p < 0 for p in [t1_1, t2_1, t2_2, t2_3]):
        return 0.0
    
    # Calculate three bomb parameters
    t1_2 = t1_1 + dt2
    t1_3 = t1_2 + dt3
    
    # Calculate individual durations
    duration1 = calculate_shielding_duration([theta, v, t1_1, t2_1], target_points)
    duration2 = calculate_shielding_duration([theta, v, t1_2, t2_2], target_points)
    duration3 = calculate_shielding_duration([theta, v, t1_3, t2_3], target_points)
    
    # Simple approximation: assume minimal overlap for speed
    total_duration = duration1 + duration2 + duration3
    return total_duration

# ========================== SIMPLE PSO ==========================
def simple_pso_optimization(target_points):
    """Simplified PSO for quick testing"""
    bounds = [
        (0.0, 2*np.pi),  # theta
        (70.0, 140.0),   # v
        (0.0, 30.0),     # t1_1
        (0.0, 10.0),     # t2_1
        (1.0, 15.0),     # dt2
        (0.0, 10.0),     # t2_2
        (1.0, 15.0),     # dt3
        (0.0, 10.0)      # t2_3
    ]
    
    # Reduced parameters for speed
    num_particles = 20
    max_iter = 30
    
    # Initialize particles
    dim = len(bounds)
    positions = np.zeros((num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    
    for i in range(dim):
        positions[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], num_particles)
        vel_range = bounds[i][1] - bounds[i][0]
        velocities[:, i] = 0.1 * np.random.uniform(-vel_range, vel_range, num_particles)
    
    # Initialize best values
    pbest_pos = positions.copy()
    pbest_fit = np.array([simple_fitness(pos, target_points) for pos in positions])
    
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    history = [gbest_fit]
    
    print(f"Initial best fitness: {gbest_fit:.4f}")
    
    # PSO iterations
    for iteration in range(max_iter):
        w = 0.9 - 0.5 * (iteration / max_iter)  # Decreasing inertia
        
        for i in range(num_particles):
            # Evaluate current position
            current_fit = simple_fitness(positions[i], target_points)
            
            # Update personal best
            if current_fit > pbest_fit[i]:
                pbest_fit[i] = current_fit
                pbest_pos[i] = positions[i].copy()
            
            # Update global best
            if current_fit > gbest_fit:
                gbest_fit = current_fit
                gbest_pos = positions[i].copy()
            
            # Update velocity and position
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (w * velocities[i] + 
                           2.0 * r1 * (pbest_pos[i] - positions[i]) + 
                           2.0 * r2 * (gbest_pos - positions[i]))
            positions[i] += velocities[i]
            
            # Apply bounds
            for j in range(dim):
                positions[i][j] = np.clip(positions[i][j], bounds[j][0], bounds[j][1])
        
        history.append(gbest_fit)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}: Best = {gbest_fit:.4f}s")
    
    return gbest_pos, gbest_fit, history

# ========================== OUTPUT FUNCTIONS ==========================
def save_simple_results(params, fitness, filename="simple_result"):
    """Save results in standard format"""
    theta, v, t1_1, t2_1, dt2, t2_2, dt3, t2_3 = params
    
    # Calculate bomb timings
    t1_2 = t1_1 + dt2
    t1_3 = t1_2 + dt3
    
    # UAV direction
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    # Prepare Excel data
    data = []
    bomb_times = [(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)]
    
    for i, (drop_t, det_delay) in enumerate(bomb_times):
        # Calculate positions
        drop_pos = FY1_INIT + v * drop_t * uav_dir
        det_xy = drop_pos[:2] + v * det_delay * uav_dir[:2]
        det_z = drop_pos[2] - 0.5 * G * det_delay**2
        
        data.append({
            "Smoke_Bomb_Number": f"S{i+1}",
            "UAV_Number": "FY1",
            "Speed_m_per_s": round(v, 2),
            "Direction_degrees": round(np.degrees(theta), 2),
            "Drop_Time_s": round(drop_t, 2),
            "Detonation_Delay_s": round(det_delay, 2),
            "Detonation_Time_s": round(drop_t + det_delay, 2),
            "Detonation_Point_X_m": round(det_xy[0], 2),
            "Detonation_Point_Y_m": round(det_xy[1], 2),
            "Detonation_Point_Z_m": round(det_z, 2),
            "Target_Missile": "M1",
            "Effective_Shielding_Duration_s": round(fitness/3, 3)  # Approximate individual duration
        })
    
    # Save to Excel
    df = pd.DataFrame(data)
    excel_file = f"{filename}.xlsx"
    df.to_excel(excel_file, index=False, engine="openpyxl")
    
    return df, excel_file

def save_simple_txt_report(params, fitness, history, comp_time, filename="simple_report.txt"):
    """Save comprehensive text report"""
    theta, v, t1_1, t2_1, dt2, t2_2, dt3, t2_3 = params
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SMOKE BOMB OPTIMIZATION - SIMPLIFIED TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Computation Time: {comp_time:.2f} seconds\n")
        f.write(f"Total Shielding Duration: {fitness:.6f} seconds\n")
        f.write(f"Algorithm: Simplified Particle Swarm Optimization\n")
        f.write("\n")
        
        f.write("OPTIMAL PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"UAV Speed: {v:.4f} m/s\n")
        f.write(f"UAV Heading: {theta:.4f} rad ({np.degrees(theta):.2f}°)\n")
        f.write(f"Bomb 1 - Drop: {t1_1:.3f}s, Detonation Delay: {t2_1:.3f}s\n")
        f.write(f"Bomb 2 - Drop: {t1_1 + dt2:.3f}s, Detonation Delay: {t2_2:.3f}s\n")
        f.write(f"Bomb 3 - Drop: {t1_1 + dt2 + dt3:.3f}s, Detonation Delay: {t2_3:.3f}s\n")
        f.write("\n")
        
        f.write("CONVERGENCE HISTORY:\n")
        f.write("-" * 40 + "\n")
        for i in range(0, len(history), max(1, len(history)//10)):
            f.write(f"Iteration {i:2d}: {history[i]:.6f}s\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    return filename

def create_simple_plot(history, save_path="simple_convergence.png"):
    """Create simple convergence plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2, label="Best Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Shielding Duration (s)")
    plt.title("PSO Convergence - Simplified Test")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

# ========================== MAIN EXECUTION ==========================
def main():
    """Main execution function"""
    print("=" * 60)
    print("SIMPLIFIED SMOKE BOMB OPTIMIZATION TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate target samples
    print("Generating target samples...")
    target_samples = generate_simple_target_samples()
    print(f"Generated {len(target_samples)} target points")
    
    # Run optimization
    print("Running simplified PSO...")
    optimal_params, best_fitness, fitness_history = simple_pso_optimization(target_samples)
    
    computation_time = time.time() - start_time
    
    # Output results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Computation Time: {computation_time:.2f} seconds")
    print(f"Best Shielding Duration: {best_fitness:.4f} seconds")
    
    theta_opt, v_opt = optimal_params[0], optimal_params[1]
    print(f"Optimal UAV Speed: {v_opt:.2f} m/s")
    print(f"Optimal UAV Heading: {np.degrees(theta_opt):.2f}°")
    
    # Save results
    print("\nSaving results...")
    df, excel_file = save_simple_results(optimal_params, best_fitness)
    txt_file = save_simple_txt_report(optimal_params, best_fitness, fitness_history, computation_time)
    create_simple_plot(fitness_history)
    
    print(f"Files generated:")
    print(f"  - Excel: {excel_file}")
    print(f"  - Report: {txt_file}")
    print(f"  - Plot: simple_convergence.png")
    
    print("\nExcel Data Preview:")
    print(df.to_string(index=False))
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n" + "=" * 60)
            print("SIMPLIFIED TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
