# -*- coding: utf-8 -*-
"""
Q2Solver_v2_diagnostic.py
诊断Q2Solver_v2的问题，找出为什么找不到有效的遮蔽方案
"""

import sys
import os
import numpy as np
import math

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Q2Solver_v2 import EvaluateCombination, MissileState, UavStateHorizontal
from Q2Solver_v2 import FY1_INIT, MISSILE_SPEED, FAKE_TARGET_ORIGIN

def diagnostic_test():
    """诊断测试"""
    print("🔍 Q2Solver_v2 问题诊断")
    print("=" * 80)

    # 测试1：检查导弹轨迹
    print("\n1. 🧪 测试导弹轨迹计算...")
    try:
        t = 5.0  # 5秒后
        missile_pos, missile_vel = MissileState(t, FY1_INIT)
        target_dist = np.linalg.norm(FAKE_TARGET_ORIGIN - FY1_INIT)
        hit_time = target_dist / MISSILE_SPEED

        print(f"   导弹初始位置: {FY1_INIT}")
        print(f"   目标位置: {FAKE_TARGET_ORIGIN}")
        print(f"   预计命中时间: {hit_time:.2f}s")
        print(f"   {t:.1f}s后的导弹位置: {missile_pos}")
        print("✅ 导弹轨迹计算正常")
    except Exception as e:
        print(f"❌ 导弹轨迹计算失败: {e}")
        return False

    # 测试2：检查无人机轨迹
    print("\n2. 🧪 测试无人机轨迹计算...")
    try:
        heading = math.pi / 4  # 45度
        speed = 200.0  # 200 m/s
        drop_time = 10.0  # 10秒后投放

        uav_pos, uav_vel = UavStateHorizontal(drop_time, FY1_INIT, speed, heading)
        print(f"   无人机初始位置: {FY1_INIT}")
        print(f"   航向角: {heading:.3f}弧度 ({np.degrees(heading):.1f}°)")
        print(f"   飞行速度: {speed:.1f} m/s")
        print(f"   {drop_time:.1f}s后的位置: {uav_pos}")
        print("✅ 无人机轨迹计算正常")
    except Exception as e:
        print(f"❌ 无人机轨迹计算失败: {e}")
        return False

    # 测试3：检查各种参数组合的遮蔽效果
    print("\n3. 🧪 测试遮蔽计算（多种参数组合）...")

    test_cases = [
        {"heading": 0.0, "speed": 200.0, "drop_time": 5.0, "fuse_delay": 2.0, "desc": "正东方向"},
        {"heading": math.pi/2, "speed": 250.0, "drop_time": 8.0, "fuse_delay": 3.0, "desc": "正北方向"},
        {"heading": math.pi, "speed": 150.0, "drop_time": 12.0, "fuse_delay": 1.5, "desc": "正西方向"},
        {"heading": math.pi/4, "speed": 180.0, "drop_time": 6.0, "fuse_delay": 2.5, "desc": "东北方向"},
    ]

    for i, case in enumerate(test_cases, 1):
        try:
            occlusion, expl_pos, hit_time = EvaluateCombination(
                case["heading"], case["speed"], case["drop_time"], case["fuse_delay"]
            )
            print(f"   测试{i} ({case['desc']}): 遮蔽时长={occlusion:.3f}s, 爆炸位置={expl_pos}")
        except Exception as e:
            print(f"   测试{i} ({case['desc']}): 失败 - {e}")

    # 测试4：检查参数空间的随机采样
    print("\n4. 🧪 测试随机参数采样...")
    np.random.seed(42)  # 固定种子以便重现

    for i in range(5):
        try:
            # 随机参数
            heading = np.random.uniform(0, 2*math.pi)
            speed = np.random.uniform(50, 300)
            drop_time = np.random.uniform(1, 30)
            fuse_delay = np.random.uniform(0.5, 10)

            occlusion, expl_pos, hit_time = EvaluateCombination(heading, speed, drop_time, fuse_delay)
            print(f"   随机测试{i+1}: 遮蔽时长={occlusion:.3f}s")
        except Exception as e:
            print(f"   随机测试{i+1}: 失败 - {e}")

    print("\n🔍 诊断完成")
    print("\n💡 建议:")
    print("   • 如果所有测试都显示遮蔽时长为0，可能需要调整参数范围")
    print("   • 检查爆炸点是否在合理的位置（不应该太低或太远）")
    print("   • 考虑增加烟云持续时间或改变烟云参数")
    print("   • 尝试不同的初始位置或轨迹参数")

    return True

if __name__ == "__main__":
    diagnostic_test()
