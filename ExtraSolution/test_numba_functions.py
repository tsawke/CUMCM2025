"""
测试Numba优化函数的正确性
"""
import numpy as np
import time
from solver5_numba_parallel import (
    get_missile_pos_numba, get_drone_pos_numba, get_smoke_pos_numba,
    calc_smoke_effective_time_numba, parallel_smoke_evaluation,
    MISSILE_INIT_POSITIONS, MISSILE_DIRECTIONS, MISSILE_FLIGHT_TIMES,
    DRONE_INIT_POSITIONS, DRONE_SPEED_RANGES
)

def test_numba_functions():
    """测试Numba优化函数"""
    print("🧪 测试Numba优化函数...")
    
    # 测试导弹位置计算
    print("\n1. 测试导弹位置计算")
    missile_idx = 0
    t = 10.0
    pos = get_missile_pos_numba(missile_idx, t)
    print(f"导弹0在时刻{t}s的位置: {pos}")
    
    # 测试无人机位置计算  
    print("\n2. 测试无人机位置计算")
    drone_idx = 0
    speed = 100.0
    direction = np.pi/4
    t = 5.0
    drone_pos = get_drone_pos_numba(drone_idx, t, speed, direction)
    print(f"无人机0在时刻{t}s的位置: {drone_pos}")
    
    # 测试烟幕弹位置计算
    print("\n3. 测试烟幕弹位置计算")
    drop_time = 3.0
    det_delay = 2.0
    t = 6.0
    smoke_pos = get_smoke_pos_numba(drone_idx, drop_time, det_delay, t, speed, direction)
    print(f"烟幕弹在时刻{t}s的位置: {smoke_pos}")
    
    # 测试遮蔽时长计算
    print("\n4. 测试遮蔽时长计算")
    existing_times = np.array([1.0, 3.0])
    n_existing = 2
    effective_time = calc_smoke_effective_time_numba(
        drone_idx, missile_idx, drop_time, det_delay, speed, direction, 
        existing_times, n_existing
    )
    print(f"有效遮蔽时长: {effective_time:.2f}s")
    
    # 测试并行评估
    print("\n5. 测试并行参数评估")
    # 创建测试参数数组
    n_params = 100
    param_array = np.random.rand(n_params, 4)
    param_array[:, 0] = param_array[:, 0] * 70 + 70  # 速度 70-140
    param_array[:, 1] = param_array[:, 1] * 2 * np.pi  # 方向 0-2π
    param_array[:, 2] = param_array[:, 2] * 50  # 投放时间 0-50
    param_array[:, 3] = param_array[:, 3] * 10 + 0.1  # 延迟 0.1-10.1
    
    start_time = time.time()
    results = parallel_smoke_evaluation(drone_idx, missile_idx, param_array, 
                                      np.array([0.0]), 1)
    end_time = time.time()
    
    print(f"并行评估{n_params}个参数组合耗时: {end_time - start_time:.4f}s")
    print(f"最佳结果: {np.max(results):.2f}s")
    print(f"有效解数量: {np.sum(results > 0)}")
    
    print("\n✅ 所有Numba函数测试完成")

def performance_benchmark():
    """性能基准测试"""
    print("\n🏃 性能基准测试...")
    
    # 测试参数
    n_tests = 1000
    drone_idx = 0
    missile_idx = 0
    speed = 100.0
    direction = np.pi/4
    
    # 测试导弹位置计算性能
    print(f"\n测试导弹位置计算性能 ({n_tests}次调用)")
    times = np.linspace(0, 50, n_tests)
    
    start_time = time.time()
    for t in times:
        pos = get_missile_pos_numba(missile_idx, t)
    end_time = time.time()
    
    missile_time = end_time - start_time
    print(f"导弹位置计算总耗时: {missile_time:.4f}s")
    print(f"平均每次调用: {missile_time/n_tests*1e6:.2f}μs")
    
    # 测试无人机位置计算性能
    print(f"\n测试无人机位置计算性能 ({n_tests}次调用)")
    start_time = time.time()
    for t in times:
        pos = get_drone_pos_numba(drone_idx, t, speed, direction)
    end_time = time.time()
    
    drone_time = end_time - start_time
    print(f"无人机位置计算总耗时: {drone_time:.4f}s")
    print(f"平均每次调用: {drone_time/n_tests*1e6:.2f}μs")
    
    # 测试烟幕弹位置计算性能
    print(f"\n测试烟幕弹位置计算性能 ({n_tests}次调用)")
    drop_time = 3.0
    det_delay = 2.0
    
    start_time = time.time()
    for t in times:
        pos = get_smoke_pos_numba(drone_idx, drop_time, det_delay, t, speed, direction)
    end_time = time.time()
    
    smoke_time = end_time - start_time
    print(f"烟幕弹位置计算总耗时: {smoke_time:.4f}s")
    print(f"平均每次调用: {smoke_time/n_tests*1e6:.2f}μs")
    
    print("\n📊 性能基准测试完成")

if __name__ == "__main__":
    print("🚀 Numba并行版本功能测试")
    print("="*50)
    
    # 基础功能测试
    test_numba_functions()
    
    # 性能基准测试
    performance_benchmark()
    
    print("\n" + "="*50)
    print("✅ 所有测试完成")
