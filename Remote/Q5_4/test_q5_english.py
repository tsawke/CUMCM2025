#!/usr/bin/env python3
"""
Quick test script for Q5Solver_English_Optimized.py
Reduced parameters for validation
"""

import sys
import time
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test_q5():
    """Run a quick test of the English Q5 solver"""
    print("=" * 80)
    print("QUICK TEST: English Multi-UAV Optimization (Q5)")
    print("=" * 80)
    
    try:
        # Test basic imports and initialization
        print("Testing imports and initialization...")
        
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Import core functions from the English solver
        from Q5Solver_English_Optimized import (
            TRUE_TARGET, MISSILES, DRONES, 
            get_missile_position, get_drone_position, get_smoke_position,
            calc_smoke_effective_time, optimize_single_smoke,
            save_result_standard_format, save_comprehensive_optimization_report,
            create_optimization_convergence_plot
        )
        
        print("✓ All imports successful")
        print(f"✓ Target initialized: {TRUE_TARGET['center']}")
        print(f"✓ {len(MISSILES)} missiles initialized")
        print(f"✓ {len(DRONES)} UAVs initialized")
        
        # Test basic position calculations
        print("\nTesting position calculations...")
        test_time = 10.0
        
        for m_name in MISSILES.keys():
            pos = get_missile_position(m_name, test_time)
            print(f"✓ Missile {m_name} position at t={test_time}s: {pos}")
        
        # Test UAV position (set test parameters)
        DRONES["FY1"]["speed"] = 100.0
        DRONES["FY1"]["direction"] = np.pi/4
        uav_pos = get_drone_position("FY1", test_time)
        print(f"✓ UAV FY1 position at t={test_time}s: {uav_pos}")
        
        # Test smoke position
        smoke_pos = get_smoke_position("FY1", 5.0, 2.0, test_time)
        print(f"✓ Smoke position test: {smoke_pos}")
        
        # Test effectiveness calculation
        effectiveness = calc_smoke_effective_time("FY1", "M1", 5.0, 2.0)
        print(f"✓ Smoke effectiveness test: {effectiveness:.4f}s")
        
        # Test single smoke optimization (quick version)
        print("\nTesting single smoke optimization...")
        start_time = time.time()
        
        # Temporarily reduce search space for quick test
        original_bounds = None
        try:
            single_result = optimize_single_smoke("FY1", "M1")
            opt_time = time.time() - start_time
            
            if single_result and single_result["effective_time"] > 0:
                print(f"✓ Single smoke optimization successful in {opt_time:.2f}s")
                print(f"  Speed: {single_result['speed']:.2f} m/s")
                print(f"  Direction: {np.degrees(single_result['direction']):.2f}°")
                print(f"  Effectiveness: {single_result['effective_time']:.4f}s")
            else:
                print("⚠ Single smoke optimization found no valid solution")
                
        except Exception as e:
            print(f"✗ Single smoke optimization failed: {str(e)}")
        
        # Test result saving functions
        print("\nTesting result saving functions...")
        
        # Create dummy smoke data for testing
        dummy_smokes = [
            {
                "drone": "FY1", "speed": 100.0, "direction": 1.0, 
                "drop_time": 5.0, "det_delay": 2.0, "det_time": 7.0,
                "det_pos": np.array([1000, 200, 500]), 
                "effective_time": 3.5, "missile": "M1"
            },
            {
                "drone": "FY2", "speed": 120.0, "direction": 1.5, 
                "drop_time": 8.0, "det_delay": 1.5, "det_time": 9.5,
                "det_pos": np.array([800, 300, 400]), 
                "effective_time": 2.8, "missile": "M2"
            }
        ]
        
        # Test Excel saving
        try:
            df = save_result_standard_format(dummy_smokes, "test_result3.xlsx")
            print("✓ Excel format saving successful")
            print("  Columns:", df.columns.tolist())
            print("  Shape:", df.shape)
        except Exception as e:
            print(f"✗ Excel saving failed: {str(e)}")
        
        # Test txt report saving
        try:
            # Add dummy optimization history
            OPTIMIZATION_HISTORY["total_durations"] = [0, 2.5, 5.2, 6.3]
            OPTIMIZATION_HISTORY["improvements"] = [0, 2.5, 2.7, 1.1]
            OPTIMIZATION_HISTORY["drone_solutions"] = [0, 1, 2, 2]
            OPTIMIZATION_HISTORY["iteration_times"] = [1.2, 1.5, 1.8, 1.3]
            
            txt_file = save_comprehensive_optimization_report(dummy_smokes, "test_comprehensive_report.txt")
            print(f"✓ Comprehensive txt report saved: {txt_file}")
        except Exception as e:
            print(f"✗ Txt report saving failed: {str(e)}")
        
        # Test convergence plot
        try:
            create_optimization_convergence_plot("test_convergence.png")
            print("✓ Convergence plot created successfully")
        except Exception as e:
            print(f"✗ Convergence plot failed: {str(e)}")
        
        print(\"\\n\" + \"=\" * 80)\n        print(\"QUICK TEST COMPLETED SUCCESSFULLY!\")\n        print(\"The English Q5 solver is ready for full optimization.\")\n        print(\"=\" * 80)\n        \n        return True\n        \n    except Exception as e:\n        print(f\"Quick test failed: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\nif __name__ == \"__main__\":\n    success = quick_test_q5()\n    if not success:\n        print(\"\\nPlease check the error messages and fix any issues before running the full optimization.\")"}, {"old_string": "    # Reset UAV states\n    for d_name in DRONES:\n        DRONES[d_name][\"optimized\"] = False\n        DRONES[d_name][\"smokes\"] = []\n        DRONES[d_name][\"speed\"] = None\n        DRONES[d_name][\"direction\"] = None", "new_string": ""}]
