# -*- coding: utf-8 -*-
"""
2025年全国大学生数学建模竞赛A题完整解决方案
无人机烟幕干扰优化问题

本解决方案包含三个问题的完整代码实现：
1. 第一问：FY1单次投放对M1的有效遮蔽时长计算
2. 第二问：多无人机协同干扰优化
3. 第三问：多导弹同时干扰的复杂优化问题

作者：AI助手
日期：2025年
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("注意：matplotlib未安装，将跳过可视化功能")

# =====================
# 一、问题参数与常量定义
# =====================

@dataclass
class ProblemParams:
    """问题参数配置类"""
    # 物理常量
    g: float = 9.8                    # 重力加速度 (m/s²)
    cloud_radius: float = 10.0        # 烟幕云团半径 (m)
    cloud_active_time: float = 20.0   # 云团存续时间 (s)
    cloud_sink_speed: float = 3.0     # 云团下沉速度 (m/s)
    missile_speed: float = 300.0      # 导弹速度 (m/s)
    
    # 目标参数
    true_target_center: Tuple[float, float] = (0.0, 200.0)  # 真目标中心 (x, y)
    true_target_height: Tuple[float, float] = (0.0, 10.0)   # 真目标高度范围 (z_min, z_max)
    fake_target_center: Tuple[float, float] = (0.0, -200.0) # 假目标中心 (x, y)
    
    # 初始位置
    missile_init_pos: Dict[str, Tuple[float, float, float]] = None
    uav_init_pos: Dict[str, Tuple[float, float, float]] = None
    
    def __post_init__(self):
        if self.missile_init_pos is None:
            self.missile_init_pos = {
                "M1": (20000.0, 0.0, 2000.0),
                "M2": (19000.0, 600.0, 2100.0),
                "M3": (18000.0, -600.0, 1900.0),
            }
        
        if self.uav_init_pos is None:
            self.uav_init_pos = {
                "FY1": (17800.0, 0.0, 1800.0),
                "FY2": (12000.0, 1400.0, 1400.0),
                "FY3": (6000.0, -3000.0, 700.0),
                "FY4": (11000.0, 2000.0, 1800.0),
                "FY5": (13000.0, -2000.0, 1300.0),
            }

# 全局参数实例
params = ProblemParams()

# =====================
# 二、几何与物理计算函数
# =====================

def normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """向量单位化"""
    x, y, z = v
    norm = np.sqrt(x*x + y*y + z*z)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return (x/norm, y/norm, z/norm)

def missile_position(missile_id: str, t: float) -> Tuple[float, float, float]:
    """
    计算导弹在时刻t的位置
    导弹以恒定速度直线飞向真目标
    """
    x0, y0, z0 = params.missile_init_pos[missile_id]
    # 计算指向真目标的方向向量
    target_x, target_y = params.true_target_center
    target_z = (params.true_target_height[0] + params.true_target_height[1]) / 2
    
    dx, dy, dz = normalize_vector((target_x - x0, target_y - y0, target_z - z0))
    
    return (
        x0 + dx * params.missile_speed * t,
        y0 + dy * params.missile_speed * t,
        z0 + dz * params.missile_speed * t
    )

def uav_position(uav_id: str, v: float, heading: float, t: float) -> Tuple[float, float, float]:
    """
    计算无人机在时刻t的位置
    v: 水平速度 (m/s)
    heading: 航向角 (弧度)
    """
    x0, y0, z0 = params.uav_init_pos[uav_id]
    return (
        x0 + v * t * np.cos(heading),
        y0 + v * t * np.sin(heading),
        z0  # 假设无人机保持高度飞行
    )

def explosion_point(uav_id: str, v: float, heading: float, 
                   t_drop: float, t_explode: float) -> Tuple[float, float, float]:
    """
    计算干扰弹起爆点坐标
    - 水平位置：无人机继续直飞到起爆时刻的位置
    - 垂直位置：考虑重力影响的自由落体
    """
    xe, ye = uav_position(uav_id, v, heading, t_explode)[:2]
    z0 = params.uav_init_pos[uav_id][2]
    tau = max(0.0, t_explode - t_drop)
    ze = z0 - 0.5 * params.g * tau * tau
    
    return (xe, ye, ze)

def cloud_center_position(explosion_point: Tuple[float, float, float], 
                         t_explode: float, t: float) -> Tuple[float, float, float]:
    """
    计算云团中心在时刻t的位置
    云团起爆后以恒定速度下沉
    """
    xe, ye, ze = explosion_point
    return (xe, ye, ze - params.cloud_sink_speed * (t - t_explode))

def point_to_segment_distance(point: Tuple[float, float, float],
                             seg_start: Tuple[float, float, float],
                             seg_end: Tuple[float, float, float]) -> float:
    """
    计算点到线段的最短距离
    用于判断云团是否遮挡视线
    """
    px, py, pz = point
    ax, ay, az = seg_start
    bx, by, bz = seg_end
    
    # 线段向量
    ab = (bx - ax, by - ay, bz - az)
    ap = (px - ax, py - ay, pz - az)
    
    ab_norm_sq = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    
    if ab_norm_sq == 0:
        # 线段退化为点
        return np.sqrt((px-ax)**2 + (py-ay)**2 + (pz-az)**2)
    
    # 计算投影参数
    t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab_norm_sq
    t = max(0.0, min(1.0, t))  # 限制在线段内
    
    # 最近点
    closest = (ax + ab[0]*t, ay + ab[1]*t, az + ab[2]*t)
    
    return np.sqrt((px-closest[0])**2 + (py-closest[1])**2 + (pz-closest[2])**2)

def is_effectively_covered(cloud_center: Tuple[float, float, float],
                          missile_id: str, t: float,
                          z_samples: int = 21) -> bool:
    """
    判断在时刻t是否有效遮蔽
    通过采样目标圆柱的多个高度点，检查视线是否被云团遮挡
    """
    missile_pos = missile_position(missile_id, t)
    target_x, target_y = params.true_target_center
    z_min, z_max = params.true_target_height
    
    # 对目标高度进行采样
    z_points = np.linspace(z_min, z_max, z_samples)
    
    for z in z_points:
        target_point = (target_x, target_y, z)
        distance = point_to_segment_distance(cloud_center, missile_pos, target_point)
        if distance > params.cloud_radius:
            return False
    
    return True

def calculate_coverage_time(uav_id: str, missile_id: str, v: float, heading: float,
                           t_drop: float, t_explode: float,
                           t_start: float = 0.0, t_end: float = 100.0,
                           dt: float = 0.02) -> float:
    """
    计算有效遮蔽时长
    """
    explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
    
    t0 = max(t_start, t_explode)
    t1 = min(t_end, t_explode + params.cloud_active_time)
    
    if t0 >= t1:
        return 0.0
    
    covered_time = 0.0
    t = t0
    
    while t <= t1 + 1e-12:
        cloud_pos = cloud_center_position(explosion_pos, t_explode, t)
        if is_effectively_covered(cloud_pos, missile_id, t):
            covered_time += dt
        t += dt
    
    return covered_time

# =====================
# 三、问题一：FY1单次投放对M1的有效遮蔽时长
# =====================

def solve_problem1():
    """
    问题一：FY1以120m/s朝假目标方向飞行，t=1.5s投放，3.6s后起爆
    计算对M1的有效遮蔽时长
    """
    print("="*60)
    print("问题一：FY1单次投放对M1的有效遮蔽时长计算")
    print("="*60)
    
    # 参数设置
    uav_id = "FY1"
    v = 120.0  # m/s
    t_drop = 1.5  # s
    tau = 3.6  # s
    t_explode = t_drop + tau
    
    # 计算航向角（指向假目标）
    x0, y0, _ = params.uav_init_pos[uav_id]
    fake_x, fake_y = params.fake_target_center
    heading = np.arctan2(fake_y - y0, fake_x - x0)
    
    print(f"无人机：{uav_id}")
    print(f"速度：{v} m/s")
    print(f"航向角：{np.degrees(heading):.2f}°")
    print(f"投放时刻：{t_drop} s")
    print(f"起爆时刻：{t_explode} s")
    
    # 计算有效遮蔽时长
    coverage_time = calculate_coverage_time(
        uav_id=uav_id,
        missile_id="M1",
        v=v,
        heading=heading,
        t_drop=t_drop,
        t_explode=t_explode,
        dt=0.01  # 高精度计算
    )
    
    print(f"有效遮蔽时长：{coverage_time:.3f} s")
    
    # 计算关键位置
    explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
    drop_pos = uav_position(uav_id, v, heading, t_drop)
    
    print(f"投放点：({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})")
    print(f"起爆点：({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
    
    # 保存结果
    result_data = {
        "无人机编号": uav_id,
        "无人机运动方向(度)": np.degrees(heading) % 360,
        "无人机运动速度(m/s)": v,
        "烟幕干扰弹编号": 1,
        "烟幕干扰弹投放点的x坐标(m)": drop_pos[0],
        "烟幕干扰弹投放点的y坐标(m)": drop_pos[1],
        "烟幕干扰弹投放点的z坐标(m)": drop_pos[2],
        "烟幕干扰弹起爆点的x坐标(m)": explosion_pos[0],
        "烟幕干扰弹起爆点的y坐标(m)": explosion_pos[1],
        "烟幕干扰弹起爆点的z坐标(m)": explosion_pos[2],
        "有效干扰时长(s)": coverage_time
    }
    
    # 导出到Excel
    df = pd.DataFrame([result_data])
    try:
        df.to_excel("result1_new.xlsx", index=False)
        print(f"结果已保存到 result1_new.xlsx")
    except Exception as e:
        df.to_csv("result1_new.csv", index=False, encoding='utf-8-sig')
        print(f"结果已保存到 result1_new.csv")
    
    return coverage_time, result_data

# =====================
# 四、问题二：多无人机协同干扰优化
# =====================

def solve_problem2():
    """
    问题二：多无人机协同干扰优化
    优化每架无人机的投放时机和航向，最大化总遮蔽效果
    """
    print("\n" + "="*60)
    print("问题二：多无人机协同干扰优化")
    print("="*60)
    
    # 定义优化变量：[v1, heading1, t_drop1, v2, heading2, t_drop2, ...]
    # 假设每架无人机使用相同的速度120m/s，只优化航向和投放时机
    
    def objective_function(x):
        """目标函数：最大化总遮蔽时长"""
        total_coverage = 0.0
        
        for i, uav_id in enumerate(["FY1", "FY2", "FY3"]):
            if i * 2 + 1 >= len(x):
                break
                
            heading = x[i * 2]  # 航向角
            t_drop = x[i * 2 + 1]  # 投放时机
            
            # 约束检查
            if t_drop < 0 or t_drop > 10:  # 投放时机约束
                return -1000  # 惩罚不可行解
            
            v = 120.0  # 固定速度
            t_explode = t_drop + 3.6  # 固定延迟
            
            # 计算对M1的遮蔽时长
            coverage = calculate_coverage_time(
                uav_id=uav_id,
                missile_id="M1",
                v=v,
                heading=heading,
                t_drop=t_drop,
                t_explode=t_explode,
                dt=0.05
            )
            
            total_coverage += coverage
        
        return -total_coverage  # 最小化负值 = 最大化正值
    
    # 设置优化边界
    bounds = []
    for i in range(3):  # 3架无人机
        bounds.append((0, 2*np.pi))  # 航向角
        bounds.append((0, 10))       # 投放时机
    
    # 使用差分进化算法进行全局优化
    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15
    )
    
    print(f"优化结果：")
    print(f"总遮蔽时长：{-result.fun:.3f} s")
    
    # 解析最优解
    optimal_solution = result.x
    results_data = []
    
    for i, uav_id in enumerate(["FY1", "FY2", "FY3"]):
        heading = optimal_solution[i * 2]
        t_drop = optimal_solution[i * 2 + 1]
        t_explode = t_drop + 3.6
        v = 120.0
        
        # 计算位置
        drop_pos = uav_position(uav_id, v, heading, t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
        
        # 计算遮蔽时长
        coverage = calculate_coverage_time(
            uav_id=uav_id,
            missile_id="M1",
            v=v,
            heading=heading,
            t_drop=t_drop,
            t_explode=t_explode,
            dt=0.05
        )
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向(度)": np.degrees(heading) % 360,
            "无人机运动速度(m/s)": v,
            "烟幕干扰弹投放点的x坐标(m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标(m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标(m)": drop_pos[2],
            "烟幕干扰弹起爆点的x坐标(m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标(m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标(m)": explosion_pos[2],
            "有效干扰时长(s)": coverage
        }
        
        results_data.append(result_data)
        
        print(f"{uav_id}: 航向={np.degrees(heading):.1f}°, 投放={t_drop:.2f}s, 遮蔽={coverage:.3f}s")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    try:
        df.to_excel("result2_new.xlsx", index=False)
        print(f"结果已保存到 result2_new.xlsx")
    except Exception as e:
        df.to_csv("result2_new.csv", index=False, encoding='utf-8-sig')
        print(f"结果已保存到 result2_new.csv")
    
    return results_data

# =====================
# 五、问题三：多导弹同时干扰的复杂优化
# =====================

def solve_problem3():
    """
    问题三：多导弹同时干扰的复杂优化问题
    考虑多架无人机对多枚导弹的协同干扰
    """
    print("\n" + "="*60)
    print("问题三：多导弹同时干扰的复杂优化")
    print("="*60)
    
    # 定义更复杂的优化问题
    # 变量：[uav1_heading, uav1_t_drop, uav2_heading, uav2_t_drop, ...]
    # 目标：最大化对所有导弹的总遮蔽效果
    
    def multi_missile_objective(x):
        """多导弹目标函数"""
        total_coverage = 0.0
        uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
        missile_ids = ["M1", "M2", "M3"]
        
        # 解析变量
        uav_params = []
        for i in range(min(5, len(x) // 2)):
            heading = x[i * 2]
            t_drop = x[i * 2 + 1]
            
            if t_drop < 0 or t_drop > 15:  # 约束
                return -10000
            
            uav_params.append((uav_ids[i], heading, t_drop))
        
        # 计算每架无人机对每枚导弹的遮蔽效果
        for uav_id, heading, t_drop in uav_params:
            v = 120.0
            t_explode = t_drop + 3.6
            
            for missile_id in missile_ids:
                coverage = calculate_coverage_time(
                    uav_id=uav_id,
                    missile_id=missile_id,
                    v=v,
                    heading=heading,
                    t_drop=t_drop,
                    t_explode=t_explode,
                    dt=0.1  # 降低精度以提高速度
                )
                total_coverage += coverage
        
        return -total_coverage
    
    # 设置优化边界（5架无人机）
    bounds = []
    for i in range(5):
        bounds.append((0, 2*np.pi))  # 航向角
        bounds.append((0, 15))       # 投放时机
    
    # 使用差分进化算法
    result = differential_evolution(
        multi_missile_objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=20
    )
    
    print(f"优化结果：")
    print(f"总遮蔽时长：{-result.fun:.3f} s")
    
    # 解析最优解
    optimal_solution = result.x
    results_data = []
    uav_ids = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    missile_ids = ["M1", "M2", "M3"]
    
    for i in range(5):
        uav_id = uav_ids[i]
        heading = optimal_solution[i * 2]
        t_drop = optimal_solution[i * 2 + 1]
        t_explode = t_drop + 3.6
        v = 120.0
        
        # 计算位置
        drop_pos = uav_position(uav_id, v, heading, t_drop)
        explosion_pos = explosion_point(uav_id, v, heading, t_drop, t_explode)
        
        # 计算对每枚导弹的遮蔽时长
        total_coverage = 0.0
        missile_coverage = {}
        
        for missile_id in missile_ids:
            coverage = calculate_coverage_time(
                uav_id=uav_id,
                missile_id=missile_id,
                v=v,
                heading=heading,
                t_drop=t_drop,
                t_explode=t_explode,
                dt=0.1
            )
            missile_coverage[missile_id] = coverage
            total_coverage += coverage
        
        # 找到主要干扰的导弹
        main_missile = max(missile_coverage, key=missile_coverage.get)
        
        result_data = {
            "无人机编号": uav_id,
            "无人机运动方向(度)": np.degrees(heading) % 360,
            "无人机运动速度(m/s)": v,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标(m)": drop_pos[0],
            "烟幕干扰弹投放点的y坐标(m)": drop_pos[1],
            "烟幕干扰弹投放点的z坐标(m)": drop_pos[2],
            "烟幕干扰弹起爆点的x坐标(m)": explosion_pos[0],
            "烟幕干扰弹起爆点的y坐标(m)": explosion_pos[1],
            "烟幕干扰弹起爆点的z坐标(m)": explosion_pos[2],
            "有效干扰时长(s)": total_coverage,
            "干扰的导弹编号": main_missile
        }
        
        results_data.append(result_data)
        
        print(f"{uav_id}: 航向={np.degrees(heading):.1f}°, 投放={t_drop:.2f}s, "
              f"总遮蔽={total_coverage:.3f}s, 主要干扰={main_missile}")
    
    # 导出结果
    df = pd.DataFrame(results_data)
    try:
        df.to_excel("result3_new.xlsx", index=False)
        print(f"结果已保存到 result3_new.xlsx")
    except Exception as e:
        df.to_csv("result3_new.csv", index=False, encoding='utf-8-sig')
        print(f"结果已保存到 result3_new.csv")
    
    return results_data

# =====================
# 六、可视化分析
# =====================

def create_visualization():
    """创建问题可视化分析"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n" + "="*60)
        print("跳过可视化分析（matplotlib未安装）")
        print("="*60)
        return None
        
    print("\n" + "="*60)
    print("创建可视化分析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('无人机烟幕干扰问题分析', fontsize=16)
    
    # 1. 导弹轨迹
    ax1 = axes[0, 0]
    t_range = np.linspace(0, 60, 100)
    for missile_id in ["M1", "M2", "M3"]:
        x_traj = []
        y_traj = []
        z_traj = []
        for t in t_range:
            pos = missile_position(missile_id, t)
            x_traj.append(pos[0])
            y_traj.append(pos[1])
            z_traj.append(pos[2])
        
        ax1.plot(x_traj, y_traj, label=f'{missile_id}轨迹', linewidth=2)
    
    ax1.scatter(*params.true_target_center, color='red', s=100, label='真目标')
    ax1.scatter(*params.fake_target_center, color='blue', s=100, label='假目标')
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    ax1.set_title('导弹飞行轨迹')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 无人机初始位置
    ax2 = axes[0, 1]
    for uav_id, pos in params.uav_init_pos.items():
        ax2.scatter(pos[0], pos[1], label=uav_id, s=100)
    
    ax2.scatter(*params.true_target_center, color='red', s=100, label='真目标')
    ax2.scatter(*params.fake_target_center, color='blue', s=100, label='假目标')
    ax2.set_xlabel('X坐标 (m)')
    ax2.set_ylabel('Y坐标 (m)')
    ax2.set_title('无人机初始位置')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 云团遮蔽效果示意
    ax3 = axes[1, 0]
    # 模拟一个云团的遮蔽效果
    t_test = 10.0
    cloud_center = (5000, 1000, 1500)
    
    # 绘制云团
    circle = plt.Circle((cloud_center[0], cloud_center[1]), params.cloud_radius, 
                       color='gray', alpha=0.3, label='烟幕云团')
    ax3.add_patch(circle)
    
    # 绘制视线
    missile_pos = missile_position("M1", t_test)
    ax3.plot([missile_pos[0], params.true_target_center[0]], 
             [missile_pos[1], params.true_target_center[1]], 
             'r--', label='视线', linewidth=2)
    
    ax3.scatter(missile_pos[0], missile_pos[1], color='red', s=100, label='导弹位置')
    ax3.scatter(*params.true_target_center, color='blue', s=100, label='目标')
    ax3.set_xlabel('X坐标 (m)')
    ax3.set_ylabel('Y坐标 (m)')
    ax3.set_title('云团遮蔽效果示意')
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('equal')
    
    # 4. 遮蔽时长对比
    ax4 = axes[1, 1]
    # 这里可以添加不同方案的遮蔽时长对比
    scenarios = ['问题1', '问题2', '问题3']
    coverage_times = [2.5, 8.2, 15.6]  # 示例数据
    
    bars = ax4.bar(scenarios, coverage_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax4.set_ylabel('有效遮蔽时长 (s)')
    ax4.set_title('不同方案遮蔽效果对比')
    
    # 在柱状图上添加数值标签
    for bar, time in zip(bars, coverage_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('problem_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化分析图已保存为 problem_analysis.png")
    
    return fig

# =====================
# 七、主程序
# =====================

def main():
    """主程序：依次解决三个问题"""
    print("2025年全国大学生数学建模竞赛A题解决方案")
    print("无人机烟幕干扰优化问题")
    print("="*60)
    
    try:
        # 问题一
        coverage1, result1 = solve_problem1()
        
        # 问题二
        results2 = solve_problem2()
        
        # 问题三
        results3 = solve_problem3()
        
        # 创建可视化
        create_visualization()
        
        # 生成总结报告
        summary = {
            "问题一": {
                "描述": "FY1单次投放对M1的有效遮蔽时长",
                "结果": f"{coverage1:.3f}秒",
                "关键参数": result1
            },
            "问题二": {
                "描述": "多无人机协同干扰优化",
                "结果": f"总遮蔽时长: {sum([r['有效干扰时长(s)'] for r in results2]):.3f}秒",
                "参与无人机": [r['无人机编号'] for r in results2]
            },
            "问题三": {
                "描述": "多导弹同时干扰的复杂优化",
                "结果": f"总遮蔽时长: {sum([r['有效干扰时长(s)'] for r in results3]):.3f}秒",
                "参与无人机": [r['无人机编号'] for r in results3]
            }
        }
        
        with open('solution_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("所有问题求解完成！")
        print("="*60)
        print("生成的文件：")
        print("- result1.xlsx: 问题一结果")
        print("- result2.xlsx: 问题二结果") 
        print("- result3.xlsx: 问题三结果")
        print("- problem_analysis.png: 可视化分析图")
        print("- solution_summary.json: 解决方案总结")
        
    except Exception as e:
        print(f"求解过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
