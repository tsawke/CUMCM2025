import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import time

# -------------------------- 1. 常量与参数定义 --------------------------
g = 9.81  # 重力加速度 (m/s²)
epsilon = 1e-15  # 数值保护阈值
dt_coarse = 0.1   # 粗算时间步长
dt_fine = 0.005   # 关键时段精细步长
n_jobs = multiprocessing.cpu_count()  # 并行计算核心数

# 目标定义
fake_target = np.array([0.0, 0.0, 0.0])
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),
    "r": 7.0,
    "h": 10.0
}

# 无人机与武器参数
fy1_init_pos = np.array([17800.0, 0.0, 1800.0])
smoke_param = {
    "r": 10.0,
    "sink_speed": 3.0,
    "valid_time": 20.0
}
missile_param = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),
    "speed": 300.0
}
# 导弹方向向量
missile_dir = (fake_target - missile_param["init_pos"]) / np.linalg.norm(fake_target - missile_param["init_pos"])
# 导弹到达假目标的时间
missile_arrival_time = np.linalg.norm(fake_target - missile_param["init_pos"]) / missile_param["speed"]


# -------------------------- 2. 目标采样点生成 --------------------------
def generate_ultra_dense_samples(target):
    """生成超高密度采样点，包含表面和内部精细网格"""
    samples = []
    center = target["center"]
    r, h = target["r"], target["h"]
    center_xy = center[:2]
    min_z, max_z = center[2], center[2] + h

    # 1. 外表面采样
    theta_dense = np.linspace(0, 2*np.pi, 120, endpoint=False)
    for z in [min_z, max_z]:
        for th in theta_dense:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])

    # 2. 侧面采样
    heights_dense = np.linspace(min_z, max_z, 40, endpoint=True)
    for z in heights_dense:
        for th in theta_dense:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])

    # 3. 内部三维网格
    radii = np.linspace(0, r, 10, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 30, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])

    # 4. 边缘过渡区加密
    edge_radii = np.linspace(r*0.95, r*1.05, 5, endpoint=True)
    for z in np.linspace(min_z, max_z, 10):
        for rad in edge_radii:
            for th in np.linspace(0, 2*np.pi, 60, endpoint=False):
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])

    return np.unique(np.array(samples), axis=0)


# -------------------------- 3. 几何计算与判定函数 --------------------------
def vector_norm(v):
    """高精度向量模长计算"""
    return np.sqrt(np.sum(v**2))


def segment_sphere_intersection(M, P, C, r):
    """高精度线段-球相交判定"""
    MP = P - M
    MC = C - M
    a = np.dot(MP, MP)
    
    # 处理零长度线段
    if a &lt; epsilon:
        dist = vector_norm(MC)
        return 1.0 if dist &lt;= r + epsilon else 0.0
    
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r**2
    discriminant = b**2 - 4*a*c
    
    # 判别式负值处理
    if discriminant &lt; -epsilon:
        return 0.0
    if discriminant &lt; 0:
        discriminant = 0.0
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)
    
    # 计算有效交区间
    s_start = max(0.0, min(s1, s2))
    s_end = min(1.0, max(s1, s2))
    
    return max(0.0, s_end - s_start)


def is_fully_shielded(missile_pos, smoke_center, smoke_r, target_samples):
    """判定是否完全遮蔽"""
    for p in target_samples:
        if segment_sphere_intersection(missile_pos, p, smoke_center, smoke_r) &lt; epsilon:
            return False
    return True


# -------------------------- 4. 自适应时间步长计算 --------------------------
def get_adaptive_time_steps(t_start, t_end, event_time=None):
    """生成自适应时间步长"""
    if event_time is None:
        return np.arange(t_start, t_end + dt_coarse, dt_coarse)
    
    # 事件点前后1秒内使用精细步长
    fine_start = max(t_start, event_time - 1.0)
    fine_end = min(t_end, event_time + 1.0)
    
    # 组合不同步长的时间序列
    times = []
    # 事件前粗步长
    if t_start &lt; fine_start:
        times.extend(np.arange(t_start, fine_start, dt_coarse))
    # 事件附近精细步长
    times.extend(np.arange(fine_start, fine_end + dt_fine, dt_fine))
    # 事件后粗步长
    if fine_end &lt; t_end:
        times.extend(np.arange(fine_end, t_end + dt_coarse, dt_coarse))
    
    return np.unique(times)


# -------------------------- 5. 适应度函数（目标函数） --------------------------
def fitness_function(params, target_samples):
    """计算适应度（遮蔽时长）"""
    theta, v, t1, t2 = params
    
    # 约束检查（快速过滤无效解）
    if not (70.0 &lt;= v &lt;= 140.0):
        return 0.0 + np.random.uniform(-0.1, 0)  # 轻微惩罚
    if t1 &lt; 0 or t2 &lt; 0:
        return 0.0 + np.random.uniform(-0.1, 0)
    
    # 1. 计算投放点
    uav_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = fy1_init_pos + v * t1 * uav_dir
    
    # 2. 计算起爆点（带约束）
    det_xy = drop_point[:2] + v * t2 * uav_dir[:2]
    det_z = drop_point[2] - 0.5 * g * t2**2
    if det_z &lt; 5.0:  # 起爆点高度约束
        return 0.0 + np.random.uniform(-0.5, 0)  # 较大惩罚
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    
    # 3. 时间窗口计算
    t_det = t1 + t2  # 起爆时刻
    t_smoke_end = t_det + smoke_param["valid_time"]
    t_end = min(t_smoke_end, missile_arrival_time)
    if t_det &gt;= t_end:
        return 0.0 + np.random.uniform(-0.1, 0)
    
    # 4. 生成时间序列
    missile_to_target = real_target["center"] - missile_param["init_pos"]
    dist_proj = np.dot(missile_to_target, missile_dir)
    event_time = dist_proj / missile_param["speed"]
    t_list = get_adaptive_time_steps(t_det, t_end, event_time)
    
    # 5. 逐时刻计算遮蔽状态
    valid_duration = 0.0
    prev_t = None
    
    for t in t_list:
        if prev_t is not None:
            dt_current = t - prev_t
            
            # 导弹位置
            missile_pos = missile_param["init_pos"] + missile_param["speed"] * t * missile_dir
            
            # 烟幕位置
            sink_time = t - t_det
            smoke_z = det_point[2] - smoke_param["sink_speed"] * sink_time
            if smoke_z &lt; 2.0:  # 烟幕过低
                prev_t = t
                continue
            smoke_center = np.array([det_point[0], det_point[1], smoke_z])
            
            # 遮蔽判定
            if is_fully_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples):
                valid_duration += dt_current
        
        prev_t = t
    
    # 对边界解给予小幅奖励，鼓励探索边界
    boundary_bonus = 0.0
    if abs(v - 70) &lt; 1 or abs(v - 140) &lt; 1:
        boundary_bonus = 0.1
    if t1 &lt; 1 or t2 &lt; 1:
        boundary_bonus += 0.1
        
    return valid_duration + boundary_bonus


# -------------------------- 6. 粒子群优化算法实现 --------------------------
class ParticleSwarmOptimizer:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100, 
                 c1=2.0, c2=2.0, w_start=0.9, w_end=0.4):
        """
        粒子群优化算法初始化
        :param objective_func: 目标函数
        :param bounds: 变量边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        :param num_particles: 粒子数量
        :param max_iter: 最大迭代次数
        :param c1: 认知系数
        :param c2: 社会系数
        :param w_start: 初始惯性权重
        :param w_end: 结束惯性权重
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1  # 认知系数
        self.c2 = c2  # 社会系数
        self.w_start = w_start  # 初始惯性权重
        self.w_end = w_end      # 结束惯性权重
        
        self.dim = len(bounds)  # 问题维度
        
        # 初始化粒子位置和速度
        self.positions = np.zeros((num_particles, self.dim))
        self.velocities = np.zeros((num_particles, self.dim))
        
        # 初始化粒子的最佳位置和适应度
        self.pbest_positions = np.zeros((num_particles, self.dim))
        self.pbest_fitness = np.zeros(num_particles) - np.inf
        
        # 全局最佳位置和适应度
        self.gbest_position = np.zeros(self.dim)
        self.gbest_fitness = -np.inf
        
        # 记录每代的最优适应度
        self.gbest_history = []
        
        # 初始化粒子
        self._initialize_particles()
    
    def _initialize_particles(self):
        """初始化粒子位置和速度"""
        for i in range(self.num_particles):
            for j in range(self.dim):
                # 在边界内随机初始化位置
                self.positions[i, j] = np.random.uniform(self.bounds[j][0], self.bounds[j][1])
                # 初始化速度（为边界范围的10%）
                vel_range = self.bounds[j][1] - self.bounds[j][0]
                self.velocities[i, j] = np.random.uniform(-0.1*vel_range, 0.1*vel_range)
            
            # 计算初始适应度
            fitness = self.objective_func(self.positions[i])
            self.pbest_positions[i] = self.positions[i].copy()
            self.pbest_fitness[i] = fitness
            
            # 更新全局最优
            if fitness &gt; self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()
    
    def _constrain_position(self, position, dim):
        """约束粒子位置在边界内"""
        min_val, max_val = self.bounds[dim]
        if position &lt; min_val:
            return min_val + 0.01 * (np.random.random() - 0.5)  # 边界附近小幅随机
        elif position &gt; max_val:
            return max_val + 0.01 * (np.random.random() - 0.5)
        return position
    
    def _constrain_velocity(self, velocity, dim):
        """约束粒子速度"""
        min_val, max_val = self.bounds[dim]
        vel_limit = 0.2 * (max_val - min_val)  # 速度限制为边界范围的20%
        return np.clip(velocity, -vel_limit, vel_limit)
    
    def optimize(self):
        """执行粒子群优化"""
        for iter in range(self.max_iter):
            # 线性减小惯性权重
            w = self.w_start - (self.w_start - self.w_end) * (iter / self.max_iter)
            
            # 并行计算所有粒子的适应度
            fitness_values = Parallel(n_jobs=n_jobs)(
                delayed(self.objective_func)(self.positions[i]) 
                for i in range(self.num_particles)
            )
            
            # 更新粒子
            for i in range(self.num_particles):
                fitness = fitness_values[i]
                
                # 更新个体最优
                if fitness &gt; self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if fitness &gt; self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
                
                # 计算新速度
                r1 = np.random.random(self.dim)  # 认知随机因子
                r2 = np.random.random(self.dim)  # 社会随机因子
                
                cognitive_component = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.gbest_position - self.positions[i])
                new_velocity = w * self.velocities[i] + cognitive_component + social_component
                
                # 约束速度
                for j in range(self.dim):
                    new_velocity[j] = self._constrain_velocity(new_velocity[j], j)
                
                # 更新速度
                self.velocities[i] = new_velocity
                
                # 更新位置
                new_position = self.positions[i] + new_velocity
                
                # 约束位置
                for j in range(self.dim):
                    new_position[j] = self._constrain_position(new_position[j], j)
                
                self.positions[i] = new_position
            
            # 记录历史最优
            self.gbest_history.append(self.gbest_fitness)
            
            # 打印迭代信息
            if (iter + 1) % 10 == 0 or iter == 0:
                print(f"迭代 {iter+1}/{self.max_iter}, 最优适应度: {self.gbest_fitness:.6f}")
        
        return self.gbest_position, self.gbest_fitness, self.gbest_history


# -------------------------- 7. 主程序 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 生成目标采样点
    print("生成超高密度目标采样点...")
    target_samples = generate_ultra_dense_samples(real_target)
    print(f"采样点数量: {len(target_samples)}")

    # 定义优化变量边界
    bounds = [
        (0.0, 2 * np.pi),      # theta: 方向角
        (70.0, 140.0),         # v: 无人机速度
        (0.0, 80.0),           # t1: 投放延迟
        (0.0, 25.0)            # t2: 起爆延迟
    ]

    # 定义适应度函数（带参数绑定）
    def objective(params):
        return fitness_function(params, target_samples)

    # 初始化并运行粒子群优化
    print("\n启动粒子群优化...")
    pso = ParticleSwarmOptimizer(
        objective_func=objective,
        bounds=bounds,
        num_particles=50,       # 粒子数量
        max_iter=100,           # 迭代次数
        c1=1.5,                 # 认知系数
        c2=1.5,                 # 社会系数
        w_start=0.9,            # 初始惯性权重
        w_end=0.4               # 结束惯性权重
    )

    # 执行优化
    best_params, best_fitness, history = pso.optimize()

    # 提取最优解
    theta_opt, v_opt, t1_opt, t2_opt = best_params

    # 高精度验证
    print("\n进行最优解高精度验证...")
    verify_fitness = fitness_function(best_params, target_samples)

    # 计算关键位置参数
    uav_dir_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0.0])
    drop_point_opt = fy1_init_pos + v_opt * t1_opt * uav_dir_opt
    det_xy_opt = drop_point_opt[:2] + v_opt * t2_opt * uav_dir_opt[:2]
    det_z_opt = drop_point_opt[2] - 0.5 * g * t2_opt**2
    det_point_opt = np.array([det_xy_opt[0], det_xy_opt[1], det_z_opt])
    t_det_opt = t1_opt + t2_opt

    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 输出结果
    print("\n" + "="*80)
    print("【粒子群优化结果 - 最优烟幕弹投放策略】")
    print(f"优化耗时: {elapsed_time:.2f} 秒")
    print(f"1. 无人机飞行方向角: {theta_opt:.6f} rad ({np.degrees(theta_opt):.2f}°)")
    print(f"2. 无人机飞行速度: {v_opt:.4f} m/s")
    print(f"3. 投放延迟时间: {t1_opt:.4f} s")
    print(f"4. 起爆延迟时间: {t2_opt:.4f} s")
    print(f"\n有效遮蔽总时长:")
    print(f"  优化阶段: {best_fitness:.6f} s")
    print(f"  验证阶段: {verify_fitness:.6f} s")
    print("\n【关键位置信息】")
    print(f"投放点坐标: {drop_point_opt.round(4)}")
    print(f"起爆点坐标: {det_point_opt.round(4)}")
    print(f"烟幕有效窗口: [{t_det_opt:.2f}s, {t_det_opt+20:.2f}s]")
    print("="*80)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('粒子群优化收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('最优遮蔽时长 (s)')
    plt.grid(True)
    plt.savefig('pso_convergence.png')
    plt.show()
    
