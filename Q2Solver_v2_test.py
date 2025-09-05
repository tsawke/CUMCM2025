# -*- coding: utf-8 -*-
"""
Q2Solver_v2_test.py
测试Q2Solver_v2的基本功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Q2Solver_v2 import EvaluateCombination, FindOptimalParameters
import numpy as np

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 开始测试Q2Solver_v2...")

    # 测试EvaluateCombination函数
    print("\n1. 测试EvaluateCombination函数...")
    try:
        heading = 1.0  # 航向角
        speed = 200.0  # 速度
        drop_time = 10.0  # 投放时间
        fuse_delay = 5.0  # 引信延时

        occlusion, expl_pos, hit_time = EvaluateCombination(heading, speed, drop_time, fuse_delay)
        print(f"   遮蔽时长: {occlusion:.3f}秒")
        print(f"   爆炸位置: ({expl_pos[0]:.1f}, {expl_pos[1]:.1f}, {expl_pos[2]:.1f})")
        print("✅ EvaluateCombination测试通过")
    except Exception as e:
        print(f"❌ EvaluateCombination测试失败: {e}")
        return False

    # 测试FindOptimalParameters函数（小规模测试）
    print("\n2. 测试FindOptimalParameters函数（小规模）...")
    try:
        best_val, best_pos, best_expl, hit_time = FindOptimalParameters(
            pop_size=3,    # 小规模测试
            iterations=2,  # 少量迭代
            workers=2      # 少量线程
        )
        print(f"   最佳遮蔽时长: {best_val:.3f}秒")
        print(f"   最优参数: {best_pos}")
        print("✅ FindOptimalParameters测试通过")
    except Exception as e:
        print(f"❌ FindOptimalParameters测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 所有测试通过！Q2Solver_v2运行正常")
    return True

if __name__ == "__main__":
    test_basic_functionality()
