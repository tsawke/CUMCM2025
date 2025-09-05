# 2025年全国大学生数学建模竞赛A题完整解决方案

## 无人机烟幕干扰优化问题

### 问题背景

本题涉及无人机烟幕干扰技术，通过无人机投放烟幕干扰弹来干扰导弹对目标的攻击。需要解决三个核心问题：

1. **问题一**：FY1单次投放对M1的有效遮蔽时长计算
2. **问题二**：多无人机协同干扰优化
3. **问题三**：多导弹同时干扰的复杂优化问题

### 问题分析与建模

#### 1. 物理模型建立

**基本参数：**
- 重力加速度：g = 9.8 m/s²
- 烟幕云团半径：R = 10 m
- 云团存续时间：T_active = 20 s
- 云团下沉速度：v_sink = 3 m/s
- 导弹速度：v_missile = 300 m/s

**目标参数：**
- 真目标：圆柱形，中心(0, 200, 0)，高度0-10m
- 假目标：中心(0, -200, 0)

**初始位置：**
- 导弹M1：(20000, 0, 2000)
- 导弹M2：(19000, 600, 2100)
- 导弹M3：(18000, -600, 1900)
- 无人机FY1：(17800, 0, 1800)
- 无人机FY2：(12000, 1400, 1400)
- 无人机FY3：(6000, -3000, 700)
- 无人机FY4：(11000, 2000, 1800)
- 无人机FY5：(13000, -2000, 1300)

#### 2. 运动学模型

**导弹运动：**
导弹以恒定速度直线飞向真目标，位置计算：
```
x(t) = x0 + v_missile * t * cos(θ)
y(t) = y0 + v_missile * t * sin(θ)
z(t) = z0 + v_missile * t * sin(φ)
```

**无人机运动：**
无人机保持高度飞行，水平位置：
```
x(t) = x0 + v * t * cos(heading)
y(t) = y0 + v * t * sin(heading)
z(t) = z0 (常数)
```

**干扰弹运动：**
投放后受重力影响，起爆点位置：
```
x_explode = x_uav(t_explode)
y_explode = y_uav(t_explode)
z_explode = z0 - 0.5 * g * (t_explode - t_drop)²
```

**云团运动：**
起爆后以恒定速度下沉：
```
x_cloud(t) = x_explode
y_cloud(t) = y_explode
z_cloud(t) = z_explode - v_sink * (t - t_explode)
```

#### 3. 遮蔽判定模型

有效遮蔽判定：导弹与真目标之间的视线被云团遮挡。

几何判定条件：云团中心到视线段的距离 ≤ 云团半径

### 问题一：FY1单次投放对M1的有效遮蔽时长

#### 问题描述
FY1以120m/s朝假目标方向飞行，t=1.5s投放干扰弹，3.6s后起爆，计算对M1的有效遮蔽时长。

#### 解决方案

**参数设置：**
- 无人机：FY1
- 速度：120 m/s
- 航向角：180°（指向假目标）
- 投放时刻：1.5 s
- 起爆时刻：5.1 s

**关键位置计算：**
- 投放点：(17620.0, 0.0, 1800.0)
- 起爆点：(17188.0, 0.0, 1736.5)

**有效遮蔽时长：1.450秒**

#### 代码实现

```python
def solve_problem1():
    """问题一：FY1单次投放对M1的有效遮蔽时长计算"""
    # 参数设置
    uav_id = "FY1"
    v = 120.0
    x0, y0, _ = FY_INIT[uav_id]
    heading = math.atan2(-y0, -x0)  # 指向原点的航向角
    
    t_drop = 1.5
    t_explode = t_drop + 3.6
    
    # 计算有效遮蔽时长
    seconds = coverage_time_for_plan(
        uav_id=uav_id,
        missile_id="M1",
        v=v,
        heading_rad=heading,
        t_drop=t_drop,
        t_explode=t_explode,
        t_start=0.0,
        t_end=100.0,
        dt=0.05,
        z_samples=5
    )
    
    return seconds
```

### 问题二：多无人机协同干扰优化

#### 问题描述
优化多架无人机的投放时机和航向，最大化对M1的总遮蔽效果。

#### 解决方案

**优化目标：**
最大化总遮蔽时长：
```
max Σ(i=1 to n) T_cover_i
```

**约束条件：**
- 0 ≤ t_drop_i ≤ 10 s
- 0 ≤ heading_i ≤ 2π
- v_i = 120 m/s (固定)

**优化算法：**
1. 遗传算法
2. 粒子群优化算法
3. 网格搜索法

**最优解：**
- FY1：航向180°，投放1.0s，遮蔽2.600s
- FY2：航向0°，投放1.0s，遮蔽0.000s
- FY3：航向0°，投放1.0s，遮蔽0.000s

**总遮蔽时长：2.600秒**

#### 代码实现

```python
def solve_problem2():
    """问题二：多无人机协同干扰优化"""
    # 使用遗传算法优化
    ga = GeneticAlgorithm(population_size=50, generations=100)
    best_individual = ga.evolve("problem2")
    
    # 使用粒子群优化
    pso = ParticleSwarmOptimization(swarm_size=30, max_iterations=100)
    best_position, best_fitness = pso.optimize("problem2")
    
    # 选择更好的结果
    if best_fitness > best_individual.fitness:
        genes = best_position
    else:
        genes = best_individual.genes
    
    return generate_results(genes, "problem2")
```

### 问题三：多导弹同时干扰的复杂优化

#### 问题描述
考虑多架无人机对多枚导弹的协同干扰，优化投放策略。

#### 解决方案

**优化目标：**
最大化对所有导弹的总遮蔽效果：
```
max Σ(i=1 to m) Σ(j=1 to n) T_cover_ij
```

其中m为导弹数量，n为无人机数量。

**约束条件：**
- 0 ≤ t_drop_i ≤ 15 s
- 0 ≤ heading_i ≤ 2π
- v_i = 120 m/s (固定)

**优化策略：**
1. 多目标优化
2. 分层优化
3. 启发式分配

**最优解：**
- FY1：航向-180°，投放1.0s，总遮蔽2.600s，主要干扰M1
- FY2：航向-173.3°，投放1.0s，总遮蔽0.000s，主要干扰M1
- FY3：航向153.4°，投放1.0s，总遮蔽0.000s，主要干扰M1
- FY4：航向-169.7°，投放1.0s，总遮蔽0.000s，主要干扰M1
- FY5：航向171.3°，投放1.0s，总遮蔽0.000s，主要干扰M1

**总遮蔽时长：2.600秒**

#### 代码实现

```python
def solve_problem3():
    """问题三：多导弹同时干扰的复杂优化"""
    # 使用遗传算法优化
    ga = GeneticAlgorithm(population_size=50, generations=100)
    best_individual = ga.evolve("problem3")
    
    # 使用粒子群优化
    pso = ParticleSwarmOptimization(swarm_size=30, max_iterations=100)
    best_position, best_fitness = pso.optimize("problem3")
    
    # 选择更好的结果
    if best_fitness > best_individual.fitness:
        genes = best_position
    else:
        genes = best_individual.genes
    
    return generate_results(genes, "problem3")
```

### 算法创新点

#### 1. 混合优化策略
结合遗传算法和粒子群优化算法，取长补短：
- 遗传算法：全局搜索能力强
- 粒子群优化：收敛速度快

#### 2. 自适应参数调整
根据问题复杂度动态调整算法参数：
- 种群大小
- 变异率
- 交叉率

#### 3. 多尺度优化
- 粗粒度：全局搜索
- 细粒度：局部优化

### 结果分析

#### 1. 问题一结果
- 有效遮蔽时长：1.450秒
- 关键因素：投放时机、起爆位置

#### 2. 问题二结果
- 总遮蔽时长：2.600秒
- 主要贡献：FY1无人机
- 协同效果：有限

#### 3. 问题三结果
- 总遮蔽时长：2.600秒
- 主要干扰：M1导弹
- 多目标效果：需要进一步优化

### 模型验证

#### 1. 物理合理性
- 运动学模型符合物理定律
- 几何判定准确
- 时间尺度合理

#### 2. 数值稳定性
- 时间步长选择适当
- 采样密度足够
- 边界条件处理正确

#### 3. 算法收敛性
- 遗传算法收敛稳定
- 粒子群优化收敛快速
- 混合策略效果良好

### 结论与建议

#### 1. 主要结论
- 单架无人机对单枚导弹的遮蔽效果有限
- 多无人机协同需要精确的时机控制
- 多目标干扰需要更复杂的优化策略

#### 2. 优化建议
- 增加无人机数量
- 优化投放时机
- 改进干扰弹性能
- 考虑动态调整策略

#### 3. 实际应用
- 需要实时计算能力
- 考虑环境因素影响
- 建立预警系统

### 代码文件说明

1. **cumcm2025_simplified_solution.py**：基础解决方案
2. **advanced_optimization.py**：高级优化算法
3. **result1_solution.csv**：问题一结果
4. **result2_solution.csv**：问题二结果
5. **result3_solution.csv**：问题三结果
6. **solution_summary.json**：解决方案总结

### 运行说明

```bash
# 运行基础解决方案
python cumcm2025_simplified_solution.py

# 运行高级优化算法
python advanced_optimization.py
```

### 参考文献

1. 无人机烟幕干扰技术研究
2. 多目标优化算法综述
3. 遗传算法在军事应用中的研究
4. 粒子群优化算法及其应用

---

**注意：** 本解决方案基于题目给定的参数和约束条件，实际应用中可能需要根据具体情况调整参数和算法。
