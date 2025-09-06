import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import time

# -------------------------- 1. 常量与基础参数定义 --------------------------
g = 9.81  # 重力加速度 (m/s²)
epsilon = 1e-12  # 数值计算保护阈值
dt_fine = 0.01  # 遮蔽判定时间步长
n_jobs = max(1, multiprocessing.cpu_count() - 2)  # 预留2个核心，避免系统卡顿

# 目标参数
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标（导弹指向）
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # 真目标底面圆心
    "r": 7.0,  # 圆柱半径
    "h": 10.0   # 圆柱高度
}

# 无人机FY1初始参数
fy1_init_pos = np.array([17800.0, 0.0, 1800.0])  # 初始位置

# 烟幕参数
smoke_param = {
    "r": 10.0,  # 有效遮蔽半径(m)
    "sink_speed": 3.0,  # 下沉速度(m/s)
    "valid_time": 20.0  # 单枚有效时长(s)
}

# 导弹M1参数
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
    "speed": 300.0,  # 飞行速度(m/s)
    "dir": (fake_target - np.array([20000.0, 0.0, 2000.0])) / 
           np.linalg.norm(fake_target - np.array([20000.0, 0.0, 2000.0]))  # 飞行方向
}
missile_arrival_time = np.linalg.norm(fake_target - missile_param["init_pos"]) / missile_param["speed"]  # 导弹到达假目标时间


# -------------------------- 2. 真目标高密度采样 --------------------------
def generate_target_samples(target, num_theta=60, num_height=20):
    """生成真目标表面+内部采样点，用于判定完全遮蔽"""
    samples = []
    center_xy = target["center"][:2]
    min_z = target["center"][2]
    max_z = target["center"][2] + target["h"]
    
    # 1. 圆柱外表面（底面、顶面、侧面）
    thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    # 底面（z=min_z）
    for th in thetas:
        x = center_xy[0] + target["r"] * np.cos(th)
        y = center_xy[1] + target["r"] * np.sin(th)
        samples.append([x, y, min_z])
    # 顶面（z=max_z）
    for th in thetas:
        x = center_xy[0] + target["r"] * np.cos(th)
        y = center_xy[1] + target["r"] * np.sin(th)
        samples.append([x, y, max_z])
    # 侧面（均匀高度层）
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    for z in heights:
        for th in thetas:
            x = center_xy[0] + target["r"] * np.cos(th)
            y = center_xy[1] + target["r"] * np.sin(th)
            samples.append([x, y, z])
    
    # 2. 圆柱内部（网格采样）
    inner_radii = np.linspace(0, target["r"], 5, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    for z in inner_heights:
        for rad in inner_radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    return np.unique(np.array(samples), axis=0)  # 去重


# -------------------------- 3. 核心几何计算与遮蔽判定 --------------------------
def segment_sphere_intersect(M, P, C, r):
    """判定线段MP（导弹位置M→目标采样点P）是否与球C(r)（烟幕）相交"""
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)
    
    # 线段长度为0（M=P）
    if a < epsilon:
        return np.linalg.norm(MC) <= r + epsilon
    
    # 一元二次方程求解
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c
    
    # 无实根（不相交）
    if discriminant < -epsilon:
        return False
    # 处理数值误差
    discriminant = max(discriminant, 0)
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)
    
    # 存在交点在[0,1]区间内
    return (s1 <= 1.0 + epsilon) and (s2 >= -epsilon)


def get_smoke_shield_interval(bomb_params, target_samples):
    """计算单枚烟幕弹的有效遮蔽时间段[start, end]"""
    theta, v, t1, t2 = bomb_params
    
    # 1. 计算投放点（无人机固定航向θ、速度v）
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])  # 无人机飞行方向
    drop_point = fy1_init_pos + v * t1 * uav_dir
    
    # 2. 计算起爆点（投放后水平沿无人机方向，竖直自由落体）
    det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t2**2
    # 起爆点高度过低（无效）
    if det_z < 0:
        return []
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    
    # 3. 烟幕有效时间窗口（起爆后20s，且不超过导弹到达假目标时间）
    t_det = t1 + t2  # 起爆时刻
    t_smoke_start = t_det
    t_smoke_end = min(t_det + smoke_param["valid_time"], missile_arrival_time)
    if t_smoke_start >= t_smoke_end:
        return []
    
    # 4. 逐时刻判定遮蔽状态，记录有效时间段
    t_list = np.arange(t_smoke_start, t_smoke_end + dt_fine, dt_fine)
    shield_intervals = []
    in_shield = False
    interval_start = 0
    
    for t in t_list:
        # 导弹当前位置
        missile_pos = missile_param["init_pos"] + missile_param["speed"] * t * missile_param["dir"]
        
        # 烟幕当前中心（xy固定，z下沉）
        sink_time = t - t_det
        smoke_center = np.array([det_point[0], det_point[1], det_point[2] - smoke_param["sink_speed"]*sink_time])
        # 烟幕落地（无效）
        if smoke_center[2] < 0:
            if in_shield:
                shield_intervals.append([interval_start, t])
                in_shield = False
            continue
        
        # 判定是否完全遮蔽（所有采样点均被烟幕阻挡）
        fully_shielded = True
        for p in target_samples:
            if not segment_sphere_intersect(missile_pos, p, smoke_center, smoke_param["r"]):
                fully_shielded = False
                break
        
        # 更新遮蔽状态
        if fully_shielded and not in_shield:
            interval_start = t
            in_shield = True
        elif not fully_shielded and in_shield:
            shield_intervals.append([interval_start, t])
            in_shield = False
    
    # 处理最后一个未结束的区间
    if in_shield:
        shield_intervals.append([interval_start, t_smoke_end])
    
    return shield_intervals


def merge_intervals(intervals):
    """合并重叠/相邻的遮蔽时间段，计算总时长（修复返回值格式）"""
    if not intervals:
        return 0.0, []  # 始终返回元组 (总时长, 合并后的区间)
    
    # 按开始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + epsilon:  # 重叠或相邻
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    # 计算总时长
    total = sum([end - start for start, end in merged])
    return total, merged  # 确保返回元组


# -------------------------- 4. 适应度函数（3枚弹总遮蔽时长） --------------------------
def fitness_function(params, target_samples):
    """
    适应度 = 3枚弹合并后的总遮蔽时长
    params: [theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
    """
    theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3 = params
    
    # 约束1：无人机速度在70~140m/s
    if not (70.0 - epsilon <= v <= 140.0 + epsilon):
        return 0.0
    # 约束2：投放间隔≥1s
    if delta_t2 < 1.0 - epsilon or delta_t3 < 1.0 - epsilon:
        return 0.0
    # 约束3：投放/起爆延迟非负
    if t1_1 < -epsilon or t2_1 < -epsilon or t2_2 < -epsilon or t2_3 < -epsilon:
        return 0.0
    
    # 计算3枚弹的参数（固定theta和v）
    t1_2 = t1_1 + delta_t2  # 第二枚投放延迟
    t1_3 = t1_2 + delta_t3  # 第三枚投放延迟
    bomb1_params = [theta, v, t1_1, t2_1]
    bomb2_params = [theta, v, t1_2, t2_2]
    bomb3_params = [theta, v, t1_3, t2_3]
    
    # 计算3枚弹的遮蔽区间（串行计算避免内存溢出）
    all_intervals = []
    all_intervals.extend(get_smoke_shield_interval(bomb1_params, target_samples))
    all_intervals.extend(get_smoke_shield_interval(bomb2_params, target_samples))
    all_intervals.extend(get_smoke_shield_interval(bomb3_params, target_samples))
    
    # 合并区间并返回总时长
    total_duration, _ = merge_intervals(all_intervals)
    return total_duration


# -------------------------- 5. 粒子群优化（PSO）实现 --------------------------
class PSOOptimizer:
    def __init__(self, obj_func, bounds, num_particles=50, max_iter=120):
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        
        # 初始化粒子位置和速度
        self.pos = np.zeros((num_particles, self.dim))
        self.vel = np.zeros((num_particles, self.dim))
        for i in range(self.dim):
            self.pos[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], num_particles)
            vel_range = bounds[i][1] - bounds[i][0]
            self.vel[:, i] = 0.1 * np.random.uniform(-vel_range, vel_range, num_particles)
        
        # 初始化个体最优（分批计算避免内存峰值）
        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.zeros(num_particles)
        batch_size = 10  # 分批处理，降低内存占用
        for i in range(0, num_particles, batch_size):
            end_idx = min(i + batch_size, num_particles)
            self.pbest_fit[i:end_idx] = Parallel(n_jobs=n_jobs)(
                delayed(self.obj_func)(self.pos[j]) for j in range(i, end_idx)
            )
        
        # 初始化全局最优
        self.gbest_idx = np.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_fit = self.pbest_fit[self.gbest_idx]
        
        # 记录迭代历史
        self.gbest_history = [self.gbest_fit]
    
    def update(self):
        """迭代更新粒子"""
        for iter in range(self.max_iter):
            # 线性减小惯性权重（0.9→0.4）
            w = 0.9 - 0.5 * (iter / self.max_iter)
            c1, c2 = 2.0, 2.0  # 认知/社会因子
            
            # 分批计算适应度，避免内存溢出
            fit_values = np.zeros(self.num_particles)
            batch_size = 10
            for i in range(0, self.num_particles, batch_size):
                end_idx = min(i + batch_size, self.num_particles)
                fit_values[i:end_idx] = Parallel(n_jobs=n_jobs)(
                    delayed(self.obj_func)(self.pos[j]) for j in range(i, end_idx)
                )
            
            # 更新个体最优和全局最优
            for i in range(self.num_particles):
                if fit_values[i] > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit_values[i]
                    self.pbest_pos[i] = self.pos[i].copy()
                if fit_values[i] > self.gbest_fit:
                    self.gbest_fit = fit_values[i]
                    self.gbest_pos = self.pos[i].copy()
            
            # 更新速度和位置
            for i in range(self.num_particles):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                # 速度更新公式
                self.vel[i] = w * self.vel[i] + c1*r1*(self.pbest_pos[i]-self.pos[i]) + c2*r2*(self.gbest_pos-self.pos[i])
                # 位置更新并约束在边界内
                self.pos[i] = self.pos[i] + self.vel[i]
                for j in range(self.dim):
                    self.pos[i][j] = max(self.bounds[j][0], min(self.pos[i][j], self.bounds[j][1]))
            
            # 记录历史并打印信息
            self.gbest_history.append(self.gbest_fit)
            if (iter + 1) % 10 == 0:
                print(f"迭代 {iter+1:3d}/{self.max_iter} | 最优适应度: {self.gbest_fit:.4f} | 全局最优时长: {self.gbest_history[-1]:.4f}s")
        
        return self.gbest_pos, self.gbest_fit, self.gbest_history


# -------------------------- 6. 主程序：优化与结果输出 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 步骤1：生成真目标采样点
    print("生成真目标采样点...")
    target_samples = generate_target_samples(real_target)
    print(f"采样点数量: {len(target_samples)}")
    
    # 步骤2：定义优化变量边界
    # [theta, v, t1_1, t2_1, delta_t2, t2_2, delta_t3, t2_3]
    bounds = [
        (0.0, 2*np.pi),    # theta: 无人机方向角（0~360°）
        (70.0, 140.0),     # v: 无人机速度（70~140m/s）
        (0.0, 60.0),       # t1_1: 第一枚投放延迟（0~60s）
        (0.0, 20.0),       # t2_1: 第一枚起爆延迟（0~20s）
        (1.0, 30.0),       # delta_t2: 1~2枚投放间隔（≥1s）
        (0.0, 20.0),       # t2_2: 第二枚起爆延迟（0~20s）
        (1.0, 30.0),       # delta_t3: 2~3枚投放间隔（≥1s）
        (0.0, 20.0)        # t2_3: 第三枚起爆延迟（0~20s）
    ]
    
    # 步骤3：初始化PSO并优化
    print("\n启动粒子群优化...")
    try:
        pso = PSOOptimizer(
            obj_func=lambda p: fitness_function(p, target_samples),
            bounds=bounds,
            num_particles=50,  # 减少粒子数量，降低计算压力
            max_iter=120
        )
        best_params, best_fitness, gbest_history = pso.update()
    except Exception as e:
        print(f"优化过程出错: {str(e)}")
        exit(1)
    
    # 步骤4：解析最优解
    theta_opt, v_opt, t1_1_opt, t2_1_opt, delta_t2_opt, t2_2_opt, delta_t3_opt, t2_3_opt = best_params
    t1_2_opt = t1_1_opt + delta_t2_opt  # 第二枚投放延迟
    t1_3_opt = t1_2_opt + delta_t3_opt  # 第三枚投放延迟
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])  # 无人机最优航向
    
    # 计算3枚弹的投放点、起爆点
    # 第一枚
    drop1 = fy1_init_pos + v_opt * t1_1_opt * uav_dir_opt
    det1_xy = drop1[:2] + v_opt * t2_1_opt * uav_dir_opt[:2]
    det1_z = drop1[2] - 0.5 * g * t2_1_opt**2
    det1 = np.array([det1_xy[0], det1_xy[1], det1_z])
    # 第二枚
    drop2 = fy1_init_pos + v_opt * t1_2_opt * uav_dir_opt
    det2_xy = drop2[:2] + v_opt * t2_2_opt * uav_dir_opt[:2]
    det2_z = drop2[2] - 0.5 * g * t2_2_opt**2
    det2 = np.array([det2_xy[0], det2_xy[1], det2_z])
    # 第三枚
    drop3 = fy1_init_pos + v_opt * t1_3_opt * uav_dir_opt
    det3_xy = drop3[:2] + v_opt * t2_3_opt * uav_dir_opt[:2]
    det3_z = drop3[2] - 0.5 * g * t2_3_opt**2
    det3 = np.array([det3_xy[0], det3_xy[1], det3_z])
    
    # 计算各枚弹的遮蔽区间和总时长
    bomb1_params = [theta_opt, v_opt, t1_1_opt, t2_1_opt]
    bomb2_params = [theta_opt, v_opt, t1_2_opt, t2_2_opt]
    bomb3_params = [theta_opt, v_opt, t1_3_opt, t2_3_opt]
    intervals1 = get_smoke_shield_interval(bomb1_params, target_samples)
    intervals2 = get_smoke_shield_interval(bomb2_params, target_samples)
    intervals3 = get_smoke_shield_interval(bomb3_params, target_samples)
    total_duration, merged_intervals = merge_intervals(intervals1 + intervals2 + intervals3)
    
    # 计算重合率
    total_single = sum([end-start for start, end in intervals1]) + \
                   sum([end-start for start, end in intervals2]) + \
                   sum([end-start for start, end in intervals3])
    overlap_rate = 0.0 if total_single < epsilon else (total_single - total_duration) / total_single * 100
    
    # 步骤5：输出结果
    end_time = time.time()
    print("\n" + "="*80)
    print("【FY1三枚烟幕弹投放策略优化结果】")
    print(f"优化总耗时: {end_time - start_time:.2f} s")
    print(f"总有效遮蔽时长: {total_duration:.4f} s")
    print(f"三弹遮蔽重合率: {overlap_rate:.2f}%")
    print(f"无人机固定速度: {v_opt:.4f} m/s")
    print(f"无人机固定航向: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f}°)")
    print("="*80)
    
    print("\n【三枚弹详细参数】")
    bombs_info = [
        ["第一枚", drop1[0], drop1[1], drop1[2], det1[0], det1[1], det1[2], t1_1_opt, t2_1_opt, intervals1],
        ["第二枚", drop2[0], drop2[1], drop2[2], det2[0], det2[1], det2[2], t1_2_opt, t2_2_opt, intervals2],
        ["第三枚", drop3[0], drop3[1], drop3[2], det3[0], det3[1], det3[2], t1_3_opt, t2_3_opt, intervals3]
    ]
    for info in bombs_info:
        print(f"{info[0]}:")
        print(f"  投放点: ({info[1]:.2f}, {info[2]:.2f}, {info[3]:.2f})")
        print(f"  起爆点: ({info[4]:.2f}, {info[5]:.2f}, {info[6]:.2f})")
        print(f"  投放延迟: {info[7]:.2f}s | 起爆延迟: {info[8]:.2f}s")
        print(f"  遮蔽区间数量: {len(info[9])}")
        print()
    
    # 步骤6：保存结果到Excel
    try:
        df = pd.DataFrame({
            "弹序号": ["第一枚", "第二枚", "第三枚"],
            "无人机航向(rad)": [theta_opt, theta_opt, theta_opt],
            "无人机速度(m/s)": [v_opt, v_opt, v_opt],
            "投放延迟(s)": [t1_1_opt, t1_2_opt, t1_3_opt],
            "起爆延迟(s)": [t2_1_opt, t2_2_opt, t2_3_opt],
            "投放点X(m)": [drop1[0], drop2[0], drop3[0]],
            "投放点Y(m)": [drop1[1], drop2[1], drop3[1]],
            "投放点Z(m)": [drop1[2], drop2[2], drop3[2]],
            "起爆点X(m)": [det1[0], det2[0], det3[0]],
            "起爆点Y(m)": [det1[1], det2[1], det3[1]],
            "起爆点Z(m)": [det1[2], det2[2], det3[2]],
            "遮蔽区间数量": [len(intervals1), len(intervals2), len(intervals3)]
        })
        df.to_excel("smoke_bomb_result.xlsx", index=False)
        print("结果已保存到 smoke_bomb_result.xlsx")
    except Exception as e:
        print(f"保存Excel失败: {str(e)}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 5))
    plt.plot(gbest_history, label="全局最优遮蔽时长")
    plt.xlabel("迭代次数")
    plt.ylabel("遮蔽时长(s)")
    plt.title("PSO优化收敛曲线")
    plt.grid(True)
    plt.legend()
    plt.savefig("convergence_curve.png")
    plt.show()

