# -*- coding: utf-8 -*-
"""
Q2Solver_v2_debug.py
详细调试遮蔽检测过程
"""

import sys
import os
import numpy as np
import math

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Q2Solver_v2 import EvaluateCombination, MissileState, UavStateHorizontal
from Q2Solver_v2 import FY1_INIT, MISSILE_SPEED, FAKE_TARGET_ORIGIN, SMOG_R

def debug_occlusion_calculation():
    """详细调试遮蔽计算过程"""
    print("🔧 详细遮蔽计算调试")
    print("=" * 80)

    # 使用一个固定的参数组合进行详细分析
    heading = math.pi / 4  # 45度
    speed = 200.0  # 200 m/s
    drop_time = 5.0  # 5秒后投放
    fuse_delay = 2.0  # 2秒引信延时

    print(f"测试参数:")
    print(f"  航向角: {heading:.3f}弧度 ({np.degrees(heading):.1f}°)")
    print(f"  飞行速度: {speed:.1f} m/s")
    print(f"  投放时间: {drop_time:.1f} s")
    print(f"  引信延时: {fuse_delay:.1f} s")

    # 手动计算爆炸点
    print("\n📍 爆炸点计算:")
    uav_pos, uav_vel = UavStateHorizontal(drop_time, FY1_INIT, speed, heading)
    print(f"  投放时无人机位置: {uav_pos}")
    print(f"  投放时无人机速度: {uav_vel}")

    # 计算爆炸点
    expl_xy = uav_pos[:2] + uav_vel[:2] * fuse_delay
    expl_z = uav_pos[2] - 0.5 * 9.8 * (fuse_delay ** 2)
    expl_pos = np.array([expl_xy[0], expl_xy[1], expl_z])
    print(f"  爆炸点位置: {expl_pos}")

    # 计算导弹命中时间
    target_dist = np.linalg.norm(FAKE_TARGET_ORIGIN - FY1_INIT)
    hit_time = target_dist / MISSILE_SPEED
    print(f"  导弹命中时间: {hit_time:.2f} s")

    # 烟云有效时间窗口
    explode_time = drop_time + fuse_delay
    t0 = explode_time
    t1 = min(explode_time + 20.0, hit_time)  # SMOG_EFFECT_TIME = 20.0
    print(f"  烟云开始时间: {t0:.2f} s")
    print(f"  烟云结束时间: {t1:.2f} s")
    print(f"  有效时间窗口: {t1-t0:.2f} s")

    if t0 >= t1:
        print("❌ 烟云在导弹命中后才生效，无效")
        return

    # 检查几个关键时间点的导弹位置
    print("\n🚀 导弹位置检查:")    check_times = [t0, t0 + 5.0, t0 + 10.0, min(t0 + 15.0, t1)]

    for t in check_times:
        if t <= t1:
            missile_pos, _ = MissileState(t, FY1_INIT)
            distance_to_explosion = np.linalg.norm(missile_pos - expl_pos)
            print(".1f"
            # 检查是否在烟云范围内
            in_cloud = distance_to_explosion <= SMOG_R
            print(f"      在烟云范围内: {'✅ 是' if in_cloud else '❌ 否'}")

    # 使用原始函数进行计算
    print("\n🧮 原始函数计算结果:")    occlusion, calc_expl_pos, calc_hit_time = EvaluateCombination(heading, speed, drop_time, fuse_delay)
    print(f"  计算的遮蔽时长: {occlusion:.6f} s")
    print(f"  计算的爆炸位置: {calc_expl_pos}")
    print(f"  计算的命中时间: {calc_hit_time:.2f} s")

    # 分析可能的问题
    print("\n🔍 问题分析:")    if occlusion == 0.0:
        print("❌ 未检测到任何遮蔽")
        print("  可能原因:")
        print("  • 烟云位置与导弹轨迹无重叠")
        print("  • 遮蔽检测算法有问题")
        print("  • 时间离散化步长太大")
        print("  • 烟云参数设置不合理")
    else:
        print("✅ 检测到遮蔽效果")

    print("\n💡 建议改进:")    print("  • 检查导弹轨迹是否经过烟云范围")
    print("  • 调整烟云参数（半径、持续时间）")
    print("  • 减小时间步长以提高精度")
    print("  • 检查几何计算是否正确")

if __name__ == "__main__":
    debug_occlusion_calculation()
