#!/usr/bin/env python3
"""
烟幕弹投放优化算法对比测试脚本
比较原始算法、优化算法和高级算法的性能
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入三个版本的算法
sys.path.append('/root/q5')

def run_original_algorithm():
    """运行原始算法"""
    print("\n" + "="*60)
    print("运行原始算法 (Q5Solver_origin.py)")
    print("="*60)
    
    try:
        # 动态导入原始算法
        import importlib.util
        spec = importlib.util.spec_from_file_location("original", "/root/q5/Q5Solver_origin.py")
        original_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_module)
        
        start_time = time.time()
        
        # 运行原始算法
        smokes = original_module.iterative_optimization(
            max_iterations=20, 
            improvement_threshold=0.3, 
            max_stall_iter=3
        )
        
        end_time = time.time()
        
        # 计算结果统计
        total_time = sum([s["effective_time"] for s in smokes]) if smokes else 0
        num_smokes = len(smokes) if smokes else 0
        
        return {
            "algorithm": "Original",
            "execution_time": end_time - start_time,
            "total_effective_time": total_time,
            "num_smokes": num_smokes,
            "avg_effectiveness": total_time / num_smokes if num_smokes > 0 else 0,
            "smokes": smokes
        }
        
    except Exception as e:
        print(f"原始算法运行失败: {e}")
        return None

def run_optimized_algorithm():
    """运行优化算法"""
    print("\n" + "="*60)
    print("运行优化算法 (Q5Solver_optimized.py)")
    print("="*60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("optimized", "/root/q5/Q5Solver_optimized.py")
        optimized_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimized_module)
        
        start_time = time.time()
        
        # 运行优化算法
        smokes = optimized_module.iterative_optimization_enhanced(
            max_iterations=25, 
            improvement_threshold=0.2, 
            max_stall_iter=4
        )
        
        end_time = time.time()
        
        # 计算结果统计
        total_time = sum([s["effective_time"] for s in smokes]) if smokes else 0
        num_smokes = len(smokes) if smokes else 0
        
        return {
            "algorithm": "Optimized",
            "execution_time": end_time - start_time,
            "total_effective_time": total_time,
            "num_smokes": num_smokes,
            "avg_effectiveness": total_time / num_smokes if num_smokes > 0 else 0,
            "smokes": smokes
        }
        
    except Exception as e:
        print(f"优化算法运行失败: {e}")
        return None

def run_advanced_algorithm():
    """运行高级算法"""
    print("\n" + "="*60)
    print("运行高级算法 (Q5Solver_advanced.py)")
    print("="*60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("advanced", "/root/q5/Q5Solver_advanced.py")
        advanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(advanced_module)
        
        start_time = time.time()
        
        # 运行高级算法
        smokes = advanced_module.iterative_optimization_advanced(
            max_iterations=30, 
            improvement_threshold=0.15, 
            max_stall_iter=5
        )
        
        end_time = time.time()
        
        # 计算结果统计
        total_time = sum([s["effective_time"] for s in smokes]) if smokes else 0
        num_smokes = len(smokes) if smokes else 0
        
        return {
            "algorithm": "Advanced",
            "execution_time": end_time - start_time,
            "total_effective_time": total_time,
            "num_smokes": num_smokes,
            "avg_effectiveness": total_time / num_smokes if num_smokes > 0 else 0,
            "smokes": smokes
        }
        
    except Exception as e:
        print(f"高级算法运行失败: {e}")
        return None

def compare_results(results):
    """对比分析结果"""
    print("\n" + "="*80)
    print("算法性能对比分析")
    print("="*80)
    
    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("没有有效的算法结果进行对比")
        return
    
    # 创建对比表格
    comparison_data = []
    for result in valid_results:
        comparison_data.append({
            "算法": result["algorithm"],
            "执行时间(s)": f"{result['execution_time']:.2f}",
            "总遮蔽时长(s)": f"{result['total_effective_time']:.3f}",
            "烟幕弹数量": result["num_smokes"],
            "平均效果(s/弹)": f"{result['avg_effectiveness']:.3f}",
            "时间效率(效果/秒)": f"{result['total_effective_time']/result['execution_time']:.3f}" if result['execution_time'] > 0 else "N/A"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 找出最优算法
    best_effectiveness = max(valid_results, key=lambda x: x["total_effective_time"])
    best_efficiency = max(valid_results, key=lambda x: x["total_effective_time"]/x["execution_time"] if x["execution_time"] > 0 else 0)
    
    print(f"\n{'性能排名':=^50}")
    print(f"最佳效果: {best_effectiveness['algorithm']} ({best_effectiveness['total_effective_time']:.3f}s)")
    print(f"最佳效率: {best_efficiency['algorithm']} ({best_efficiency['total_effective_time']/best_efficiency['execution_time']:.3f} 效果/秒)")
    
    # 改进分析
    if len(valid_results) > 1:
        baseline = min(valid_results, key=lambda x: x["total_effective_time"])
        best = max(valid_results, key=lambda x: x["total_effective_time"])
        
        improvement = ((best["total_effective_time"] - baseline["total_effective_time"]) / 
                      baseline["total_effective_time"] * 100)
        
        print(f"\n最大改进幅度: {improvement:.1f}% ({best['algorithm']} vs {baseline['algorithm']})")
    
    # 保存对比结果
    df.to_excel("/root/q5/algorithm_comparison.xlsx", index=False)
    print(f"\n对比结果已保存到: /root/q5/algorithm_comparison.xlsx")
    
    return valid_results

def visualize_comparison(results):
    """可视化对比结果"""
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) < 2:
        print("需要至少2个有效结果才能进行对比可视化")
        return
    
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = [r["algorithm"] for r in valid_results]
    
    # 1. 总遮蔽时长对比
    total_times = [r["total_effective_time"] for r in valid_results]
    bars1 = ax1.bar(algorithms, total_times, color=['skyblue', 'lightgreen', 'lightcoral'][:len(algorithms)])
    ax1.set_title("总遮蔽时长对比", fontsize=14, fontweight='bold')
    ax1.set_ylabel("总遮蔽时长(s)", fontsize=12)
    for bar, time in zip(bars1, total_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 执行时间对比
    exec_times = [r["execution_time"] for r in valid_results]
    bars2 = ax2.bar(algorithms, exec_times, color=['orange', 'gold', 'lightpink'][:len(algorithms)])
    ax2.set_title("执行时间对比", fontsize=14, fontweight='bold')
    ax2.set_ylabel("执行时间(s)", fontsize=12)
    for bar, time in zip(bars2, exec_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 烟幕弹数量对比
    num_smokes = [r["num_smokes"] for r in valid_results]
    bars3 = ax3.bar(algorithms, num_smokes, color=['lightsteelblue', 'lightseagreen', 'lightsalmon'][:len(algorithms)])
    ax3.set_title("烟幕弹数量对比", fontsize=14, fontweight='bold')
    ax3.set_ylabel("烟幕弹数量", fontsize=12)
    for bar, count in zip(bars3, num_smokes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 平均效果对比
    avg_effects = [r["avg_effectiveness"] for r in valid_results]
    bars4 = ax4.bar(algorithms, avg_effects, color=['plum', 'khaki', 'peachpuff'][:len(algorithms)])
    ax4.set_title("平均单弹效果对比", fontsize=14, fontweight='bold')
    ax4.set_ylabel("平均遮蔽时长(s/弹)", fontsize=12)
    for bar, avg in zip(bars4, avg_effects):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{avg:.3f}s', ha='center', va='bottom', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("/root/q5/algorithm_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    """主函数：运行所有算法并对比"""
    print("开始算法对比测试...")
    
    results = []
    
    # 运行三个算法（按复杂度递增）
    algorithms = [
        ("原始算法", run_original_algorithm),
        ("优化算法", run_optimized_algorithm), 
        ("高级算法", run_advanced_algorithm)
    ]
    
    for name, func in algorithms:
        print(f"\n准备运行{name}...")
        try:
            result = func()
            if result:
                results.append(result)
                print(f"{name}完成: 总遮蔽时长 {result['total_effective_time']:.3f}s, 耗时 {result['execution_time']:.2f}s")
            else:
                print(f"{name}未找到有效解")
        except Exception as e:
            print(f"{name}运行出错: {e}")
            continue
    
    # 对比分析
    if results:
        compare_results(results)
        visualize_comparison(results)
        
        print(f"\n{'测试总结':=^60}")
        print(f"成功运行算法数量: {len(results)}")
        
        if len(results) > 1:
            best = max(results, key=lambda x: x["total_effective_time"])
            worst = min(results, key=lambda x: x["total_effective_time"])
            improvement = (best["total_effective_time"] - worst["total_effective_time"]) / worst["total_effective_time"] * 100
            print(f"最大性能提升: {improvement:.1f}% ({best['algorithm']} vs {worst['algorithm']})")
        
        print("="*60)
    else:
        print("所有算法都失败了，请检查代码")

if __name__ == "__main__":
    main()
