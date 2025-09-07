# -*- coding: utf-8 -*-
"""
Q3_verify_result.py - 验证Q3结果与Q3Drawer的一致性

这个脚本重新计算result1.xlsx中的解，使用与Q3Drawer相同的逻辑，
以确保计算结果的一致性。
"""

import numpy as np
import pandas as pd
import math
import Q1Solver_visual as Q1

def read_result_xlsx(path="result1.xlsx"):
    """读取Excel结果文件"""
    df = pd.read_excel(path)
    
    # 提取参数
    heading_text = df.iloc[0]["无人机运动方向"]
    if "rad" in str(heading_text):
        heading = float(str(heading_text).replace("rad", "").strip())
    else:
        heading = math.radians(float(heading_text))
    
    speed = float(df.iloc[0]["无人机运动速度 (m/s)"])
    
    bombs_data = []
    for i in range(3):
        row = df.iloc[i]
        bomb_data = {
            "drop_pos": [row["烟幕干扰弹投放点的x坐标 (m)"], 
                        row["烟幕干扰弹投放点的y坐标 (m)"], 
                        row["烟幕干扰弹投放点的z坐标 (m)"]],
            "expl_pos": [row["烟幕干扰弹起爆点的x坐标 (m)"], 
                        row["烟幕干扰弹起爆点的y坐标 (m)"], 
                        row["烟幕干扰弹起爆点的z坐标 (m)"]],
            "single_time": row["有效干扰时长 (s)"]
        }
        bombs_data.append(bomb_data)
    
    return heading, speed, bombs_data

def recover_timing_params(heading, speed, bombs_data):
    """从位置反推时间参数"""
    times_data = []
    
    for bomb in bombs_data:
        drop_pos = np.array(bomb["drop_pos"])
        expl_pos = np.array(bomb["expl_pos"])
        
        # 计算投放时间 - 修正计算方法
        # 使用无人机轨迹来反推时间
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        
        # 解方程: FY1_INIT + [vx, vy, 0] * t = drop_pos
        if abs(vx) > abs(vy):
            t_drop = (drop_pos[0] - Q1.FY1_INIT[0]) / vx
        else:
            t_drop = (drop_pos[1] - Q1.FY1_INIT[1]) / vy
        
        t_drop = max(0, t_drop)  # 确保时间为正
        
        # 计算引信延时
        flight_distance = math.hypot(expl_pos[0] - drop_pos[0], expl_pos[1] - drop_pos[1])
        horizontal_time = flight_distance / speed
        
        # 垂直运动
        dz = drop_pos[2] - expl_pos[2]
        if dz > 0:
            fuse_delay = math.sqrt(2 * dz / Q1.g)
        else:
            fuse_delay = horizontal_time
        
        t_expl = t_drop + fuse_delay
        
        times_data.append({
            "t_drop": t_drop,
            "fuse_delay": fuse_delay,
            "t_expl": t_expl
        })
    
    return times_data

def segment_sphere_intersect(p1, p2, center, radius):
    """线段与球体相交检测"""
    d = p2 - p1
    f = p1 - center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    
    # 检查交点是否在线段上
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    # 检查端点是否在球内
    if np.linalg.norm(p1 - center) <= radius:
        return True
    if np.linalg.norm(p2 - center) <= radius:
        return True
    
    return False

def calculate_union_coverage(heading, speed, times_data, dt=0.005):
    """计算联合遮蔽时长"""
    # 目标采样点
    cylinder_center = Q1.CYLINDER_BASE_CENTER
    cylinder_r = Q1.CYLINDER_R
    cylinder_h = Q1.CYLINDER_H
    
    # 生成圆柱体采样点 (简化版)
    n_phi = 24  # 减少采样点以匹配Q3Drawer
    n_z = 3
    
    points = []
    # 底面圆
    for i in range(n_phi):
        angle = 2 * math.pi * i / n_phi
        x = cylinder_center[0] + cylinder_r * math.cos(angle)
        y = cylinder_center[1] + cylinder_r * math.sin(angle)
        points.append([x, y, cylinder_center[2]])
    
    # 顶面圆
    for i in range(n_phi):
        angle = 2 * math.pi * i / n_phi
        x = cylinder_center[0] + cylinder_r * math.cos(angle)
        y = cylinder_center[1] + cylinder_r * math.sin(angle)
        points.append([x, y, cylinder_center[2] + cylinder_h])
    
    # 中间层
    for z_level in range(1, n_z-1):
        z = cylinder_center[2] + cylinder_h * z_level / (n_z-1)
        for i in range(n_phi):
            angle = 2 * math.pi * i / n_phi
            x = cylinder_center[0] + cylinder_r * math.cos(angle)
            y = cylinder_center[1] + cylinder_r * math.sin(angle)
            points.append([x, y, z])
    
    points = np.array(points)
    
    # 时间范围
    hit_time = np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN) / Q1.MISSILE_SPEED
    t_expls = [td["t_expl"] for td in times_data]
    
    t_start = min(t_expls)
    t_end = min(max(t_expls) + Q1.SMOG_EFFECT_TIME, hit_time)
    
    if t_end <= t_start:
        return 0.0, [0.0, 0.0, 0.0]
    
    # 时间网格
    time_grid = np.arange(t_start, t_end, dt)
    
    union_mask = np.zeros(len(time_grid), dtype=bool)
    individual_masks = [np.zeros(len(time_grid), dtype=bool) for _ in range(3)]
    
    for i, t in enumerate(time_grid):
        # 导弹位置
        missile_pos = Q1.M1_INIT + (Q1.FAKE_TARGET_ORIGIN - Q1.M1_INIT) * (Q1.MISSILE_SPEED * t) / np.linalg.norm(Q1.M1_INIT - Q1.FAKE_TARGET_ORIGIN)
        
        point_covered_by_any = np.zeros(len(points), dtype=bool)
        
        # 检查每个烟幕弹
        for bomb_idx, time_data in enumerate(times_data):
            if t < time_data["t_expl"]:
                continue
            if t > time_data["t_expl"] + Q1.SMOG_EFFECT_TIME:
                continue
            
            # 烟幕中心位置 (考虑下沉)
            smoke_center = np.array([
                Q1.FY1_INIT[0] + speed * math.cos(heading) * time_data["t_drop"],
                Q1.FY1_INIT[1] + speed * math.sin(heading) * time_data["t_drop"],
                Q1.FY1_INIT[2] - 0.5 * Q1.g * (time_data["fuse_delay"] ** 2)
            ])
            
            # 考虑烟幕下沉
            smoke_center[2] -= Q1.SMOG_SINK_SPEED * (t - time_data["t_expl"])
            
            if smoke_center[2] <= 0:  # 烟幕落地
                continue
            
            # 检查每个采样点
            points_covered_by_this_bomb = np.zeros(len(points), dtype=bool)
            all_points_covered = True
            
            for j, point in enumerate(points):
                covered = segment_sphere_intersect(missile_pos, point, smoke_center, Q1.SMOG_R)
                points_covered_by_this_bomb[j] = covered
                if covered:
                    point_covered_by_any[j] = True
                else:
                    all_points_covered = False
            
            # 单体遮蔽判断
            individual_masks[bomb_idx][i] = all_points_covered
        
        # 联合遮蔽判断：所有点都被至少一个烟幕遮住
        union_mask[i] = np.all(point_covered_by_any)
    
    # 计算时长
    union_time = np.sum(union_mask) * dt
    individual_times = [np.sum(mask) * dt for mask in individual_masks]
    
    return union_time, individual_times

def main():
    print("="*80)
    print("Q3 Result Verification - 验证与Q3Drawer的一致性")
    print("="*80)
    
    try:
        # 读取结果
        heading, speed, bombs_data = read_result_xlsx("result1.xlsx")
        print(f"读取结果: heading={heading:.6f} rad, speed={speed:.1f} m/s")
        
        # 反推时间参数
        times_data = recover_timing_params(heading, speed, bombs_data)
        print("\n时间参数:")
        for i, td in enumerate(times_data):
            print(f"  烟幕弹{i+1}: t_drop={td['t_drop']:.3f}s, fuse={td['fuse_delay']:.3f}s, t_expl={td['t_expl']:.3f}s")
        
        # 重新计算遮蔽时长
        print("\n重新计算遮蔽时长...")
        union_time, individual_times = calculate_union_coverage(heading, speed, times_data)
        
        print("\n计算结果对比:")
        print(f"原Excel联合时长: {sum(bd['single_time'] for bd in bombs_data):.6f}s")
        print(f"重算联合时长:   {union_time:.6f}s")
        print("\n单体时长对比:")
        for i in range(3):
            print(f"  烟幕弹{i+1}: 原={bombs_data[i]['single_time']:.6f}s, 重算={individual_times[i]:.6f}s")
        
        # 分析差异
        original_total = sum(bd['single_time'] for bd in bombs_data)
        diff = abs(union_time - original_total)
        
        print(f"\n差异分析:")
        print(f"总时长差异: {diff:.6f}s ({diff/max(original_total, 1e-9)*100:.2f}%)")
        
        if diff > 0.1:
            print("⚠️  差异较大，可能存在计算逻辑不一致")
            print("建议:")
            print("1. 检查采样点数量和分布")
            print("2. 检查时间步长设置")
            print("3. 检查遮蔽判断算法")
        else:
            print("✅ 差异在合理范围内")
            
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
