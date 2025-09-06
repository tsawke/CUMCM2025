"""
快速演示Numba并行版本的求解过程（缩短迭代次数以快速验证）
"""
import time
from solver5_numba_parallel import iterative_optimization_parallel

def quick_demo():
    """快速演示求解过程"""
    print("🚀 启动Numba并行版本快速演示...")
    print("⚡ 使用较少的迭代次数进行快速验证")
    print("="*60)
    
    start_time = time.time()
    
    # 使用较少的迭代次数进行快速演示
    all_smokes = iterative_optimization_parallel(
        max_iterations=5,      # 减少到5轮迭代
        improvement_threshold=0.1,  # 降低改进阈值
        max_stall_iter=2       # 减少停滞轮数
    )
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    print("\n" + "="*60)
    print("🎯 快速演示结果：")
    print(f"⏱️  总耗时：{optimization_time:.2f}秒")
    
    if all_smokes:
        print(f"💨 总烟幕弹数量：{len(all_smokes)}")
        print(f"🛡️  总遮蔽时长：{sum([s['effective_time'] for s in all_smokes]):.2f}s")
        print(f"⚡ 平均每轮耗时：{optimization_time/5:.2f}秒")
        
        # 显示各无人机结果
        print("\n📊 各无人机投放情况：")
        from solver5_numba_parallel import DRONES
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s["effective_time"] for s in d_data["smokes"]])
                print(f"  {d_name}：{len(d_data['smokes'])}枚弹，遮蔽时长{total:.2f}s")
            else:
                print(f"  {d_name}：未找到有效方案")
        
        print("\n✅ 快速演示成功！Numba并行版本工作正常")
    else:
        print("❌ 未找到有效解，可能需要更多迭代")
    
    print("="*60)

if __name__ == "__main__":
    quick_demo()
