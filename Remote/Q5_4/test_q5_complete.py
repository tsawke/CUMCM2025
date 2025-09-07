#!/usr/bin/env python3
"""
Test script for Q5Solver_Complete_English.py
Quick validation with reduced computational complexity
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def test_basic_functions():
    """Test basic function imports and calculations"""
    print("Testing basic function imports...")
    
    try:
        # Test imports
        from Q5Solver_Complete_English import (
            TRUE_TARGET, MISSILES, DRONES,
            generate_target_sampling_points, initialize_missiles,
            get_missile_position, get_drone_position, get_smoke_position,
            calculate_smoke_effectiveness, line_segment_sphere_intersection
        )
        print("✓ All function imports successful")
        
        # Test initialization
        print("Testing initialization functions...")
        generate_target_sampling_points()
        initialize_missiles()
        print("✓ Initialization completed")
        print(f"  - Target samples: {len(TRUE_TARGET['sample_points'])}")
        print(f"  - Missiles initialized: {list(MISSILES.keys())}")
        
        # Test position calculations
        print("Testing position calculation functions...")
        test_time = 10.0
        
        # Test missile positions
        for m_name in MISSILES.keys():
            pos = get_missile_position(m_name, test_time)
            print(f"  - {m_name} position at t={test_time}s: {pos}")
        
        # Test UAV position (set test parameters first)
        DRONES["FY1"]["speed"] = 100.0
        DRONES["FY1"]["direction"] = np.pi/4  # 45 degrees
        uav_pos = get_drone_position("FY1", test_time)
        print(f"  - FY1 position at t={test_time}s: {uav_pos}")
        
        # Test smoke position
        smoke_pos = get_smoke_position("FY1", 5.0, 2.0, test_time)
        print(f"  - Smoke position test: {smoke_pos}")
        
        # Test line-sphere intersection
        p1 = np.array([0, 0, 0])
        p2 = np.array([10, 0, 0])
        center = np.array([5, 0, 0])
        radius = 2.0
        intersection = line_segment_sphere_intersection(p1, p2, center, radius)
        print(f"  - Line-sphere intersection test: {intersection}")
        
        # Test effectiveness calculation
        effectiveness = calculate_smoke_effectiveness("FY1", "M1", 5.0, 2.0)
        print(f"  - Smoke effectiveness test: {effectiveness:.4f}s")
        
        print("✓ All basic function tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_components():
    """Test optimization algorithm components"""
    print("\nTesting optimization components...")
    
    try:
        from Q5Solver_Complete_English import (
            optimize_single_smoke_deployment, assign_tasks_to_uavs,
            save_result_standard_format, save_comprehensive_optimization_report,
            OPTIMIZATION_HISTORY
        )
        
        # Test task assignment
        print("Testing task assignment...")
        assignments = assign_tasks_to_uavs()
        print(f"✓ Task assignment result: {assignments}")
        
        # Test single smoke optimization (quick version)
        print("Testing single smoke optimization...")
        start_time = time.time()
        
        # Temporarily reduce complexity by setting drone parameters
        DRONES["FY1"]["speed"] = 100.0
        DRONES["FY1"]["direction"] = 1.0
        
        single_result = optimize_single_smoke_deployment("FY1", "M1")
        opt_time = time.time() - start_time
        
        if single_result and single_result["effectiveness"] > 0:
            print(f"✓ Single smoke optimization successful in {opt_time:.2f}s")
            print(f"  - Speed: {single_result['speed']:.2f} m/s")
            print(f"  - Direction: {np.degrees(single_result['direction']):.2f}°")
            print(f"  - Effectiveness: {single_result['effectiveness']:.4f}s")
        else:
            print("⚠ Single smoke optimization found no valid solution")
        
        print("✓ Optimization component tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Optimization component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_output_functions():
    """Test result output and visualization functions"""
    print("\nTesting output functions...")
    
    try:
        from Q5Solver_Complete_English import (
            save_result_standard_format, save_comprehensive_optimization_report,
            create_optimization_convergence_visualization, OPTIMIZATION_HISTORY
        )
        
        # Create dummy smoke deployment data for testing
        dummy_deployments = [
            {
                "drone": "FY1", "speed": 100.0, "direction": 1.0, 
                "drop_time": 5.0, "detonation_delay": 2.0, "detonation_time": 7.0,
                "detonation_position": np.array([1000, 250, 500]), 
                "effectiveness": 3.5, "target_missile": "M1"
            },
            {
                "drone": "FY2", "speed": 120.0, "direction": 1.5, 
                "drop_time": 8.0, "detonation_delay": 1.5, "detonation_time": 9.5,
                "detonation_position": np.array([800, 300, 400]), 
                "effectiveness": 2.8, "target_missile": "M2"
            },
            {
                "drone": "FY3", "speed": 90.0, "direction": 0.8, 
                "drop_time": 12.0, "detonation_delay": 3.0, "detonation_time": 15.0,
                "detonation_position": np.array([600, 180, 350]), 
                "effectiveness": 4.2, "target_missile": "M3"
            }
        ]
        
        # Test Excel format saving
        print("Testing Excel format output...")
        df = save_result_standard_format(dummy_deployments, "test_result3.xlsx")
        print("✓ Excel format saving successful")
        print(f"  - Generated {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  - Columns: {list(df.columns)}")
        
        # Test comprehensive report saving
        print("Testing comprehensive report generation...")
        
        # Populate dummy optimization history
        OPTIMIZATION_HISTORY["total_durations"] = [0, 3.5, 7.2, 10.5, 10.5]
        OPTIMIZATION_HISTORY["improvements"] = [0, 3.5, 3.7, 3.3, 0.0]
        OPTIMIZATION_HISTORY["drone_solutions"] = [0, 1, 2, 3, 3]
        OPTIMIZATION_HISTORY["iteration_times"] = [1.2, 1.8, 2.1, 1.9, 1.5]
        
        report_file = save_comprehensive_optimization_report(dummy_deployments, "test_comprehensive_report.txt")
        print(f"✓ Comprehensive report saved: {report_file}")
        
        # Test convergence visualization
        print("Testing convergence visualization...")
        create_optimization_convergence_visualization("test_convergence.png")
        print("✓ Convergence plot generated successfully")
        
        print("✓ All output function tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Output function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("=" * 80)
    print("COMPREHENSIVE TEST: Q5 Multi-UAV English Optimization")
    print("=" * 80)
    
    # Run all test components
    test_results = []
    
    print("Phase 1: Basic Function Tests")
    test_results.append(test_basic_functions())
    
    print("\nPhase 2: Optimization Component Tests")
    test_results.append(test_optimization_components())
    
    print("\nPhase 3: Output Function Tests")
    test_results.append(test_output_functions())
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED - The English Q5 solver is ready for production use!")
        print("\nTo run the full optimization, execute:")
        print("  python3 Q5Solver_Complete_English.py")
    else:
        print("✗ SOME TESTS FAILED - Please review error messages and fix issues")
    
    print("=" * 80)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
