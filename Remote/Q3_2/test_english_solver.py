#!/usr/bin/env python3
"""
Test script for Q3ExSolver_English.py
Quick test with reduced parameters for validation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend for headless operation
import matplotlib.pyplot as plt
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Run a quick test with reduced parameters"""
    print("="*60)
    print("QUICK TEST: English Smoke Bomb Optimization")
    print("="*60)
    
    try:
        # Import the English solver
        from Q3ExSolver_English import (
            generate_target_samples, real_target, PSOOptimizer, 
            fitness_function, save_results_standard_format, 
            save_detailed_results_to_txt, plot_convergence_curve
        )
        
        print("Modules imported successfully")
        
        # Step 1: Generate target samples (reduced for quick test)
        print("Generating target samples...")
        target_samples = generate_target_samples(real_target, num_theta=20, num_height=8)
        print(f"Generated {len(target_samples)} sampling points")
        
        # Step 2: Define bounds for quick test
        bounds = [
            (0.0, 2*np.pi),    # theta: UAV heading angle
            (70.0, 140.0),     # v: UAV speed
            (0.0, 30.0),       # t1_1: First bomb drop delay
            (0.0, 10.0),       # t2_1: First bomb detonation delay
            (1.0, 15.0),       # delta_t2: Drop interval
            (0.0, 10.0),       # t2_2: Second bomb detonation delay
            (1.0, 15.0),       # delta_t3: Drop interval
            (0.0, 10.0)        # t2_3: Third bomb detonation delay
        ]
        
        # Step 3: Quick PSO optimization (reduced parameters)
        print("Starting quick PSO optimization...")
        start_time = time.time()
        
        pso = PSOOptimizer(
            obj_func=lambda p: fitness_function(p, target_samples),
            bounds=bounds,
            num_particles=20,  # Reduced for quick test
            max_iter=30        # Reduced for quick test
        )
        
        best_params, best_fitness, gbest_history = pso.update()
        optimization_time = time.time() - start_time
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.4f} seconds")
        
        # Step 4: Calculate detailed results
        theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
        
        # Calculate intervals for each bomb
        from Q3ExSolver_English import get_smoke_shield_interval, merge_intervals
        
        t1_2_opt = t1_1_opt + delta_t2_opt
        t1_3_opt = t1_2_opt + delta_t3_opt
        
        intervals_data = []
        bomb_params_list = [
            [theta_opt, v_opt, t1_1_opt, t2_1_opt],
            [theta_opt, v_opt, t1_2_opt, t2_2_opt],
            [theta_opt, v_opt, t1_3_opt, t2_3_opt]
        ]
        
        for bomb_params in bomb_params_list:
            intervals = get_smoke_shield_interval(bomb_params, target_samples)
            intervals_data.append(intervals)
        
        # Calculate total duration
        all_intervals = []
        for intervals in intervals_data:
            all_intervals.extend(intervals)
        total_duration, _ = merge_intervals(all_intervals)
        
        # Step 5: Save results
        print("Saving results...")
        
        # Save Excel format
        df, excel_filename = save_results_standard_format(
            best_params, total_duration, intervals_data, filename_base="test_result"
        )
        
        # Save detailed txt
        txt_filename = save_detailed_results_to_txt(
            best_params, total_duration, intervals_data, gbest_history, optimization_time,
            filename="test_optimization_detailed.txt"
        )
        
        # Save convergence plot
        plot_convergence_curve(gbest_history, save_path="test_convergence.png")
        
        print("\nQUICK TEST RESULTS:")
        print("-" * 40)
        print(f"Total Shielding Duration: {total_duration:.4f} seconds")
        print(f"UAV Speed: {v_opt:.2f} m/s")
        print(f"UAV Heading: {np.degrees(theta_opt):.2f}°")
        print(f"Number of Bombs: {len(bombs_data)}")
        print(f"Files generated:")
        print(f"  - {excel_filename}")
        print(f"  - {txt_filename}")
        print(f"  - test_convergence.png")
        
        # Display DataFrame
        print(f"\nGenerated DataFrame:")
        print(df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n" + "="*60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("The English solver is working correctly.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("QUICK TEST FAILED!")
        print("Please check the error messages above.")
        print("="*60)
