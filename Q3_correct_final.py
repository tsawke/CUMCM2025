# -*- coding: utf-8 -*-
"""
Q3_correct_final.py - 基于正确物理理解的问题3解决方案

核心发现：
1. 导弹飞向假目标(0,0,0)，真目标在(0,200,0)
2. 视线角度很小（<20°），真目标相对导弹轨迹是"侧方偏移"
3. 最佳策略：在导弹轨迹附近放置烟幕，遮挡侧向视线
4. 重点时段：导弹接近假目标时（后半程），视线角度增大，更容易遮挡
"""

import math
import numpy as np
import pandas as pd

# 物理常量
g = 9.8
CLOUD_R = 10.0
CLOUD_SINK = 3.0
CLOUD_EFFECT = 20.0
MISSILE_SPEED = 300.0

# 位置
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BASE = np.array([0.0, 200.0, 0.0])
CYL_R = 7.0
CYL_H = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

def missile_pos(t):
    """导弹位置"""
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    return M1_INIT + MISSILE_SPEED * t * direction

def uav_pos(t, speed, heading):
    """无人机位置"""
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return np.array([FY1_INIT[0] + vx * t, FY1_INIT[1] + vy * t, FY1_INIT[2]])

def explosion_pos(t_drop, fuse_delay, speed, heading):
    """爆炸点位置"""
    drop_pos = uav_pos(t_drop, speed, heading)
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    expl_x = drop_pos[0] + vx * fuse_delay
    expl_y = drop_pos[1] + vy * fuse_delay
    expl_z = drop_pos[2] - 0.5 * g * (fuse_delay ** 2)
    
    return np.array([expl_x, expl_y, expl_z])

def point_line_distance(point, line_start, line_end):
    """点到线段的最短距离"""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        return np.linalg.norm(point - line_start)
    
    # 计算投影
    point_vec = point - line_start
    t = np.dot(point_vec, line_vec) / (line_len * line_len)
    t = max(0, min(1, t))  # 限制在线段上
    
    closest_point = line_start + t * line_vec
    return np.linalg.norm(point - closest_point)

def calculate_coverage_time(bomb_params):
    """计算单个烟幕弹的遮蔽时间"""
    t_drop, fuse_delay, speed, heading = bomb_params
    t_expl = t_drop + fuse_delay
    expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
    
    if expl_pos[2] < 0:
        return 0.0
    
    # 关键目标点（简化）
    target_points = [
        TRUE_TARGET_BASE + np.array([CYL_R, 0, 0]),
        TRUE_TARGET_BASE + np.array([-CYL_R, 0, 0]),
        TRUE_TARGET_BASE + np.array([0, 0, CYL_H]),
        TRUE_TARGET_BASE + np.array([0, 0, CYL_H/2]),
    ]
    
    coverage_time = 0.0
    dt = 0.05
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    for t in np.arange(t_expl, min(t_expl + CLOUD_EFFECT, hit_time), dt):
        # 云团位置
        cloud_z = expl_pos[2] - CLOUD_SINK * (t - t_expl)
        if cloud_z < -CLOUD_R:
            break
        
        cloud_pos = np.array([expl_pos[0], expl_pos[1], cloud_z])
        missile_position = missile_pos(t)
        
        # 检查是否所有目标点都被遮蔽
        all_blocked = True
        for target_point in target_points:
            dist = point_line_distance(cloud_pos, missile_position, target_point)
            if dist > CLOUD_R:
                all_blocked = False
                break
        
        if all_blocked:
            coverage_time += dt
    
    return coverage_time

def strategic_search():
    """战略性搜索：基于导弹轨迹的智能布局"""
    print("开始战略性搜索...")
    
    best_params = None
    best_total = 0.0
    
    hit_time = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    
    # 重点关注导弹接近假目标的时段（后半程）
    critical_times = np.arange(40, min(65, hit_time-5), 5)
    
    for critical_t in critical_times:
        missile_at_critical = missile_pos(critical_t)
        
        # 计算到达关键点的航向和速度
        for speed in [90, 110, 130]:
            for offset_y in [-50, 0, 50]:  # Y方向偏移
                # 目标点：导弹轨迹附近，但偏向真目标方向
                target_point = missile_at_critical + np.array([0, offset_y, 0])
                
                # 计算航向
                dx = target_point[0] - FY1_INIT[0]
                dy = target_point[1] - FY1_INIT[1]
                heading = math.atan2(dy, dx)
                
                # 计算到达时间
                dist = math.hypot(dx, dy)
                arrival_time = dist / speed
                
                if arrival_time > critical_t - 10:  # 太晚到达
                    continue
                
                # 设计三枚弹的时序
                t1 = arrival_time + 2
                t2 = t1 + 1.5
                t3 = t2 + 1.5
                
                for f1 in [4, 6]:
                    for f2 in [4, 6]:
                        for f3 in [4, 6]:
                            params = [(t1, f1, speed, heading),
                                     (t2, f2, speed, heading),
                                     (t3, f3, speed, heading)]
                            
                            # 验证物理可行性
                            valid = True
                            for t_drop, fuse, spd, hdg in params:
                                expl_pos = explosion_pos(t_drop, fuse, spd, hdg)
                                if expl_pos[2] < 50:  # 爆炸点太低
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            # 计算总遮蔽时间（简化：各弹独立计算后求和）
                            total_coverage = 0.0
                            individual_coverages = []
                            
                            for param in params:
                                coverage = calculate_coverage_time(param)
                                individual_coverages.append(coverage)
                                total_coverage += coverage
                            
                            if total_coverage > best_total:
                                best_total = total_coverage
                                best_params = params
                                print(f"新最优解: 总时长={total_coverage:.6f}s")
                                print(f"  各弹: {[f'{c:.3f}s' for c in individual_coverages]}")
                                print(f"  参数: critical_t={critical_t:.1f}, speed={speed}, offset_y={offset_y}")
    
    return best_params, best_total

def save_final_result(params, total_time):
    """保存最终结果"""
    if not params:
        print("无有效解")
        return
    
    rows = []
    for i, (t_drop, fuse_delay, speed, heading) in enumerate(params):
        drop_pos = uav_pos(t_drop, speed, heading)
        expl_pos = explosion_pos(t_drop, fuse_delay, speed, heading)
        single_time = calculate_coverage_time((t_drop, fuse_delay, speed, heading))
        
        rows.append({
            "无人机运动方向": f"{heading:.6f} rad",
            "无人机运动速度 (m/s)": speed,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_pos[0], 6),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_pos[1], 6),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_pos[2], 6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(expl_pos[0], 6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(expl_pos[1], 6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(expl_pos[2], 6),
            "有效干扰时长 (s)": round(single_time, 6),
        })
    
    df = pd.DataFrame(rows)
    df.to_excel("result1_corrected.xlsx", index=False)
    
    print("\n最终结果:")
    print(df.to_string(index=False))
    print(f"\n总遮蔽时间: {total_time:.6f}s")
    print("结果已保存到: result1_corrected.xlsx")

def main():
    print("="*80)
    print("问题3正确解决方案 - 基于几何分析的战略布局")
    print("="*80)
    
    # 战略搜索
    params, total_time = strategic_search()
    
    if total_time > 0:
        save_final_result(params, total_time)
        
        # 验证结果
        print(f"\n解决方案验证:")
        print(f"- 找到有效解，总遮蔽时间: {total_time:.6f}s")
        print(f"- 这个结果应该与Q3Drawer验证一致")
    else:
        print("\n❌ 未找到有效解")
        print("可能原因:")
        print("1. 物理约束太严格（爆炸高度、速度范围等）")
        print("2. 遮蔽判定算法需要调整")
        print("3. 搜索范围需要扩大")

if __name__ == "__main__":
    main()
