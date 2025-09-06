"""
性能对比脚本：比较原始版本与Numba并行版本的性能
"""
import time
import subprocess
import sys
import os

def run_solver(solver_file, description):
    """运行求解器并记录时间"""
    print(f"\n{'='*60}")
    print(f"运行 {description}")
    print(f"文件: {solver_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行求解器
        result = subprocess.run([sys.executable, solver_file], 
                              capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"执行时间: {execution_time:.2f} 秒")
        
        if result.returncode == 0:
            print("✅ 执行成功")
            # 提取关键结果信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "总烟幕弹数量" in line or "总遮蔽时长" in line or "优化总耗时" in line:
                    print(f"📊 {line.strip()}")
        else:
            print("❌ 执行失败")
            print("错误输出:", result.stderr)
        
        return execution_time, result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 执行超时 (30分钟)")
        return 1800, False
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return 0, False

def main():
    print("🚀 烟幕弹优化求解器性能对比测试")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查文件是否存在
    original_solver = "solver5.py"
    parallel_solver = "solver5_numba_parallel.py"
    
    if not os.path.exists(original_solver):
        print(f"❌ 找不到原始求解器文件: {original_solver}")
        return
    
    if not os.path.exists(parallel_solver):
        print(f"❌ 找不到并行求解器文件: {parallel_solver}")
        return
    
    results = {}
    
    # 运行原始版本
    print("\n🔄 第一轮测试：原始版本")
    time_orig, success_orig = run_solver(original_solver, "原始版本 (solver5.py)")
    results['original'] = {'time': time_orig, 'success': success_orig}
    
    # 等待一段时间让系统冷却
    print("\n⏳ 等待系统冷却...")
    time.sleep(10)
    
    # 运行Numba并行版本
    print("\n🔄 第二轮测试：Numba并行版本")
    time_parallel, success_parallel = run_solver(parallel_solver, "Numba并行版本 (solver5_numba_parallel.py)")
    results['parallel'] = {'time': time_parallel, 'success': success_parallel}
    
    # 性能对比总结
    print(f"\n{'='*80}")
    print("🏆 性能对比总结")
    print(f"{'='*80}")
    
    print(f"原始版本执行时间:     {results['original']['time']:.2f} 秒")
    print(f"Numba并行版本执行时间: {results['parallel']['time']:.2f} 秒")
    
    if results['original']['success'] and results['parallel']['success']:
        speedup = results['original']['time'] / results['parallel']['time']
        print(f"\n🚀 性能提升倍数: {speedup:.2f}x")
        
        if speedup > 1:
            improvement = ((results['original']['time'] - results['parallel']['time']) / results['original']['time']) * 100
            print(f"📈 性能提升百分比: {improvement:.1f}%")
            print(f"⏱️  节省时间: {results['original']['time'] - results['parallel']['time']:.2f} 秒")
        else:
            print("⚠️  并行版本比原始版本慢，可能需要进一步优化")
    else:
        print("⚠️  无法进行性能对比，因为有版本执行失败")
    
    print(f"\n✅ 原始版本成功: {'是' if results['original']['success'] else '否'}")
    print(f"✅ 并行版本成功: {'是' if results['parallel']['success'] else '否'}")
    
    # 保存结果到文件
    with open("performance_comparison_result.txt", "w", encoding="utf-8") as f:
        f.write(f"烟幕弹优化求解器性能对比结果\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"原始版本执行时间: {results['original']['time']:.2f} 秒\n")
        f.write(f"并行版本执行时间: {results['parallel']['time']:.2f} 秒\n")
        if results['original']['success'] and results['parallel']['success']:
            speedup = results['original']['time'] / results['parallel']['time']
            f.write(f"性能提升倍数: {speedup:.2f}x\n")
        f.write(f"原始版本成功: {'是' if results['original']['success'] else '否'}\n")
        f.write(f"并行版本成功: {'是' if results['parallel']['success'] else '否'}\n")
    
    print(f"\n📁 详细结果已保存到: performance_comparison_result.txt")

if __name__ == "__main__":
    main()
