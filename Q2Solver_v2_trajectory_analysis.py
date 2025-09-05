# -*- coding: utf-8 -*-
"""
Q2Solver_v2_trajectory_analysis.py
分析导弹和无人机轨迹，找出合适的参数范围
"""

import sys
import os
import numpy as np
import math

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Q2Solver_v2 import MissileState, UavStateHorizontal
from Q2Solver_v2 import FY1_INIT, MISSILE_SPEED, FAKE_TARGET_ORIGIN, SMOG_R

def analyze_trajectories():
    """分析导弹和无人机轨迹"""
    print("📊 轨迹分析")
    print("=" * 60)

    # 导弹基本信息
    missile_init_pos = FY1_INIT
    target_pos = FAKE_TARGET_ORIGIN
    missile_speed = MISSILE_SPEED

    print("导弹信息:")
    print(f"  初始位置: {missile_init_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  飞行速度: {missile_speed} m/s")

    # 计算直线距离和飞行时间
    distance = np.linalg.norm(target_pos - missile_init_pos)
    flight_time = distance / missile_speed

    print(f"  直线距离: {distance:.2f} m")
    print(f"  预计飞行时间: {flight_time:.2f} s")

    # 分析导弹轨迹（前20秒）
    print("\n🚀 导弹轨迹分析（前20秒）:")
    print("  时间(s) | 位置(x, y, z) | 距离目标(m)")
    print("  --------|---------------|-------------")

    for t in [0, 5, 10, 15, 20]:
        if t <= flight_time:
            pos, vel = MissileState(t, missile_init_pos)
            dist_to_target = np.linalg.norm(pos - target_pos)
            print("8.1f"
    # 无人机轨迹分析
    print("\n🛩️ 无人机轨迹分析:")
    print("以45度角、200m/s速度飞行")

    heading = math.pi / 4  # 45度
    speed = 200.0

    for t in [0, 5, 10, 15, 20]:
        pos, vel = UavStateHorizontal(t, missile_init_pos, speed, heading)
        dist_to_target = np.linalg.norm(pos - target_pos)
        print("8.1f"    # 分析合理的投放策略
    print("\n🎯 合理的投放策略分析:")    print("要让烟云遮挡导弹，需要：")
    print("1. 投放时机要合适，让烟云出现在导弹路径上")
    print("2. 投放位置要合适，烟云要覆盖导弹轨迹")
    print("3. 引信延时要合适，确保烟云在正确时间爆炸")

    # 建议的参数范围
    print("\n💡 建议参数范围:")    print("投放时间 (drop_time):")
    print("  • 建议范围: 1-15秒")
    print("  • 原因: 太早烟云消散，太晚导弹已接近目标")

    print("引信延时 (fuse_delay):")
    print("  • 建议范围: 0.5-5秒")
    print("  • 原因: 太短烟云位置不好控制，太长延时太久")

    print("航向角 (heading):")
    print("  • 建议范围: 0-π弧度 (0-180°)")
    print("  • 原因: 需要向导弹飞行方向投放")

    print("飞行速度 (speed):")
    print("  • 建议范围: 100-250 m/s")
    print("  • 原因: 太慢反应迟钝，太快控制困难")

    # 计算一个可能的有效参数组合
    print("\n🔍 计算可能的有效参数:")    # 让无人机飞向导弹可能经过的区域
    # 导弹大约20秒后到达目标，我们在10秒时投放
    drop_time = 10.0
    fuse_delay = 3.0
    heading = math.pi / 6  # 30度，更靠近导弹方向
    speed = 180.0

    print("建议参数组合:")
    print(f"  投放时间: {drop_time} s")
    print(f"  引信延时: {fuse_delay} s")
    print(f"  航向角: {heading:.3f}弧度 ({np.degrees(heading):.1f}°)")
    print(f"  飞行速度: {speed} m/s")

if __name__ == "__main__":
    analyze_trajectories()
