# -*- coding: utf-8 -*-
"""
Q2Solver_v2_simple_debug.py
简单调试遮蔽检测问题
"""

import sys
import os
import numpy as np
import math

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Q2Solver_v2 import EvaluateCombination, MissileState
from Q2Solver_v2 import FY1_INIT, MISSILE_SPEED, FAKE_TARGET_ORIGIN, SMOG_R

def simple_debug():
    """简单调试"""
    print("🔧 简单遮蔽检测调试")
    print("=" * 50)

    # 使用固定参数
    heading = math.pi / 4  # 45度
    speed = 200.0
    drop_time = 5.0
    fuse_delay = 2.0

    print("测试参数:")
    print(f"  航向角: {heading:.3f}弧度")
    print(f"  速度: {speed:.1f} m/s")
    print(f"  投放时间: {drop_time:.1f} s")
    print(f"  引信延时: {fuse_delay:.1f} s")

    # 计算爆炸时间
    explode_time = drop_time + fuse_delay
    print(f"\n爆炸时间: {explode_time:.2f} s")

    # 计算导弹命中时间
    target_dist = np.linalg.norm(FAKE_TARGET_ORIGIN - FY1_INIT)
    hit_time = target_dist / MISSILE_SPEED
    print(f"导弹命中时间: {hit_time:.2f} s")

    # 检查导弹在爆炸时刻的位置
    missile_pos, _ = MissileState(explode_time, FY1_INIT)
    print(f"\n爆炸时刻导弹位置: {missile_pos}")

    # 估算爆炸点位置（简化计算）
    # 这里我们直接调用EvaluateCombination来获取准确的爆炸点
    occlusion, expl_pos, _ = EvaluateCombination(heading, speed, drop_time, fuse_delay)
    print(f"爆炸点位置: {expl_pos}")

    # 计算距离
    distance = np.linalg.norm(missile_pos - expl_pos)
    print(f"爆炸时刻导弹到爆炸点的距离: {distance:.2f} m")
    print(f"烟云半径: {SMOG_R:.1f} m")
    print(f"是否在烟云范围内: {'是' if distance <= SMOG_R else '否'}")

    print(f"\n最终遮蔽时长: {occlusion:.6f} s")

    if occlusion == 0.0:
        print("\n❌ 问题分析:")
        print("  • 导弹在爆炸时刻没有在烟云范围内")
        print("  • 时间窗口可能有问题")
        print("  • 烟云下沉或消散可能影响结果")
    else:
        print("\n✅ 检测到遮蔽效果")

if __name__ == "__main__":
    simple_debug()
