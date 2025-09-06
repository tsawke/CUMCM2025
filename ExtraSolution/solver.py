import numpy as np
import pandas as pd
import math
from itertools import product
import random

# 全局常量定义
g = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕球半径(m)
SMOKE_EFFECTIVE_TIME = 20  # 烟幕有效时间(s)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
MISSILE_SPEED = 300  # 导弹速度(m/s)
TRUE_TARGET_RADIUS = 7  # 真目标半径(m)
TRUE_TARGET_HEIGHT = 10  # 真目标高度(m)
TRUE_TARGET_CENTER = np.array([0, 200, 0])  # 真目标下底面圆心

# 无人机初始参数（FY1、FY2、FY3）
DRONES_INIT = {
    1: np.array([17800, 0, 1800]),    # FY1: (x0,y0,z0)
    2: np.array([12000, 1400, 1400]), # FY2
    3: np.array([6000, -3000, 700])   # FY3
}

# 导弹初始参数
MISSILE_INIT = np.array([20000, 0, 2000])  # 初始位置
missile_dir = -MISSILE_INIT / np.linalg.norm(MISSILE_INIT)  # 运动方向

# 预生成真目标采样点（覆盖底面、侧面、顶面）
def generate_true_target_points():
    points = []
    cx, cy, cz = TRUE_TARGET_CENTER
    
    # 底面点（z=0）
    angles = np.linspace(0, 2*np.pi, 30)
    for alpha in angles:
        x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
        y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
        points.append([x, y, cz])
    
    # 侧面点（z从0到10）
    angles = np.linspace(0, 2*np.pi, 30)
    heights = np.linspace(0, TRUE_TARGET_HEIGHT, 10)
    for z in heights:
        for alpha in angles:
            x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
            y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
            points.append([x, y, z])
    
    # 顶面点（z=10）
    for alpha in angles:
        x = cx + TRUE_TARGET_RADIUS * math.cos(alpha)
        y = cy + TRUE_TARGET_RADIUS * math.sin(alpha)
        points.append([x, y, cz + TRUE_TARGET_HEIGHT])
    
    return np.array(points)

TRUE_TARGET_POINTS = generate_true_target_points()

# 无人机位置计算
def drone_position(drone_id, t, theta, v):
    x0, y0, z0 = DRONES_INIT[drone_id]
    x = x0 + v * math.cos(theta) * t
    y = y0 + v * math.sin(theta) * t
    return np.array([x, y, z0])

# 烟幕位置计算（修正递归问题）
def calculate_detonate_point(drone_id, theta, v, td, tau):
    xd, yd, zd = drone_position(drone_id, td, theta, v)
    delta_t = tau
    xe = xd + v * math.cos(theta) * delta_t
    ye = yd + v * math.sin(theta) * delta_t
    ze = zd - 0.5 * g * delta_t ** 2
    return np.array([xe, ye, ze])

def smoke_position(drone_id, t, theta, v, td, tau):
    if t < td - 1e-6:  # 未投放
        return drone_position(drone_id, t, theta, v)
    elif td - 1e-6 <= t < td + tau - 1e-6:  # 投放后未起爆
        xd, yd, zd = drone_position(drone_id, td, theta, v)
        delta_t = t - td
        x = xd + v * math.cos(theta) * delta_t
        y = yd + v * math.sin(theta) * delta_t
        z = zd - 0.5 * g * delta_t **2
        return np.array([x, y, z])
    elif td + tau - 1e-6 <= t <= td + tau + SMOKE_EFFECTIVE_TIME + 1e-6:  # 起爆后
        xe, ye, ze = calculate_detonate_point(drone_id, theta, v, td, tau)
        delta_t = t - (td + tau)
        z = ze - SMOKE_SINK_SPEED * delta_t
        return np.array([xe, ye, z])
    else:  # 失效
        return None

# 线段与球相交判定
def segment_sphere_intersection(P1, P2, C, r):
    P1 = np.array(P1)
    P2 = np.array(P2)
    C = np.array(C)
    v = P2 - P1
    w = P1 - C
    
    a = np.dot(v, v)
    if a < 1e-10:  # 两点重合
        return np.linalg.norm(P1 - C) <= r + 1e-10
    
    b = 2 * np.dot(w, v)
    c = np.dot(w, w) - r** 2
    discriminant = b **2 - 4 * a * c
    
    if discriminant < -1e-10:
        return False
    
    discriminant = max(discriminant, 0)
    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    
    return (t1 >= -1e-10 and t1 <= 1 + 1e-10) or (t2 >= -1e-10 and t2 <= 1 + 1e-10)

# 计算单个策略的遮蔽区间
def get_smoke_intervals(drone_id, theta, v, td, tau, time_step=0.2):
    intervals = []
    current_start = None
    t_min = max(0, td - 5)
    t_max = td + tau + SMOKE_EFFECTIVE_TIME + 5
    
    for t in np.arange(t_min, t_max, time_step):
        smoke_center = smoke_position(drone_id, t, theta, v, td, tau)
        if smoke_center is None:
            if current_start is not None:
                intervals.append((current_start, t))
                current_start = None
            continue
        
        # 导弹位置
        missile_pos = MISSILE_INIT + MISSILE_SPEED * t * missile_dir
        
        # 检查是否遮蔽所有目标点
        all_blocked = True
        for point in TRUE_TARGET_POINTS:
            if not segment_sphere_intersection(missile_pos, point, smoke_center, SMOKE_RADIUS):
                all_blocked = False
                break
        
        if all_blocked:
            if current_start is None:
                current_start = t
        else:
            if current_start is not None:
                intervals.append((current_start, t))
                current_start = None
    
    # 过滤短区间并合并
    valid_intervals = []
    for s, e in intervals:
        if e - s > 0.1:  # 过滤无效短区间
            valid_intervals.append((s, e))
    
    return valid_intervals

# 合并区间并计算总时长
def merge_and_calculate(intervals_list):
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)
    
    if not all_intervals:
        return [], 0.0
    
    # 排序并合并
    all_intervals.sort()
    merged = [list(all_intervals[0])]
    for s, e in all_intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 0.1:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    
    total = sum(e - s for s, e in merged if e - s > 0.1)
    return merged, total

# 为单个无人机生成随机策略
def generate_drone_strategies(drone_id, num_strategies=5, min_effective=1.0):
    strategies = []
    x0, y0, z0 = DRONES_INIT[drone_id]
    
    # 参数范围（根据无人机特性调整）
    if drone_id == 1:  # 高度最高，需要更长下落时间
        theta_range = (-0.1, 0.1)  # 方向角范围
        v_range = (90, 110)        # 速度范围
        td_range = (160, 180)      # 投放延迟范围
        tau_range = (18, 22)       # 起爆延迟范围
    elif drone_id == 2:
        theta_range = (-0.2, 0.2)
        v_range = (85, 105)
        td_range = (100, 120)
        tau_range = (16, 20)
    else:  # drone_id == 3，高度最低
        theta_range = (-0.3, 0.3)
        v_range = (80, 100)
        td_range = (50, 70)
        tau_range = (10, 14)
    
    # 生成策略直到满足数量要求
    while len(strategies) < num_strategies:
        theta = random.uniform(*theta_range)
        v = random.uniform(*v_range)
        td = random.uniform(*td_range)
        tau = random.uniform(*tau_range)
        
        # 计算该策略的有效遮蔽时间
        intervals = get_smoke_intervals(drone_id, theta, v, td, tau)
        _, total = merge_and_calculate([intervals])
        
        # 只保留有效时间足够的策略
        if total >= min_effective:
            strategies.append({
                'drone_id': drone_id,
                'theta': theta,
                'v': v,
                'td': td,
                'tau': tau,
                'intervals': intervals,
                'duration': total
            })
            print(f"生成无人机{drone_id}策略{len(strategies)}，有效时间：{total:.2f}s")
    
    return strategies

# 组合策略并找到最优解
def find_best_combination(fy1_strats, fy2_strats, fy3_strats):
    best_total = 0
    best_combination = None
    best_intervals = []
    
    # 遍历所有组合（5×5×5=125种组合）
    total_combinations = len(fy1_strats) * len(fy2_strats) * len(fy3_strats)
    print(f"\n开始评估{total_combinations}种策略组合...")
    
    for i, (s1, s2, s3) in enumerate(product(fy1_strats, fy2_strats, fy3_strats)):
        # 合并三个策略的区间
        all_intervals = [s1['intervals'], s2['intervals'], s3['intervals']]
        merged, total = merge_and_calculate(all_intervals)
        
        # 计算重叠率（越低越好）
        individual_sum = s1['duration'] + s2['duration'] + s3['duration']
        overlap_rate = (individual_sum - total) / individual_sum if individual_sum > 0 else 1.0
        
        # 优先选择总时间长且重叠率低的组合
        if total > best_total or (total == best_total and overlap_rate < 0.5):
            best_total = total
            best_combination = (s1, s2, s3)
            best_intervals = merged
            print(f"找到更优组合（{i+1}/{total_combinations}）：总时间{total:.2f}s，重叠率{overlap_rate:.2f}")
    
    return best_combination, best_intervals, best_total

# 保存结果到Excel
def save_results(best_comb, best_intervals, best_total, save_path="separate_optimization_result.xlsx"):
    s1, s2, s3 = best_comb
    data = []
    
    # 单个无人机策略详情
    for s in [s1, s2, s3]:
        intervals_str = "; ".join([f"[{s:.1f},{e:.1f}]" for s,e in s['intervals'] if e-s>0.1])
        data.append([
            f"FY{s['drone_id']}",
            f"{s['theta']:.4f}",
            f"{s['v']:.1f}",
            f"{s['td']:.1f}",
            f"{s['tau']:.1f}",
            f"{s['duration']:.2f}s",
            intervals_str
        ])
    
    # 总结果
    total_intervals_str = "; ".join([f"[{s:.1f},{e:.1f}]" for s,e in best_intervals if e-s>0.1])
    data.append([
        "总计", "", "", "", "", 
        f"{best_total:.2f}s",
        total_intervals_str
    ])
    
    # 保存
    df = pd.DataFrame(data, columns=[
        "无人机", "方向角(rad)", "速度(m/s)", "投放延迟(s)", 
        "起爆延迟(s)", "有效时间", "遮蔽区间"
    ])
    df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"\n结果已保存至 {save_path}")

# 主程序
if __name__ == "__main__":
    # 步骤1：为每个无人机生成5种有效策略
    print("="*50 + " 生成单个无人机策略 " + "="*50)
    fy1_strategies = generate_drone_strategies(1, num_strategies=5, min_effective=2.0)
    fy2_strategies = generate_drone_strategies(2, num_strategies=5, min_effective=2.0)
    fy3_strategies = generate_drone_strategies(3, num_strategies=5, min_effective=2.0)
    
    # 步骤2：组合策略并找到最优解
    print("\n" + "="*50 + " 寻找最优策略组合 " + "="*50)
    best_comb, best_intervals, best_total = find_best_combination(
        fy1_strategies, fy2_strategies, fy3_strategies
    )
    
    # 步骤3：保存结果
    print("\n" + "="*50 + " 优化结果 " + "="*50)
    if best_total < 10.0:
        print(f"警告：最优组合总遮蔽时间为{best_total:.2f}s，未达到10s目标")
    else:
        print(f"成功：最优组合总遮蔽时间为{best_total:.2f}s，满足要求")
    
    save_results(best_comb, best_intervals, best_total)
    

