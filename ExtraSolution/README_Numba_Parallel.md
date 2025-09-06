# 烟幕弹优化求解器 - Numba并行版本

## 概述

这是基于原始`solver5.py`的高性能并行版本，使用Numba JIT编译和多进程并行计算来显著提升求解速度，同时保持算法逻辑和正确性不变。

## 主要改进

### 1. Numba JIT编译优化
- **核心计算函数加速**: 使用`@jit(nopython=True, cache=True)`装饰器优化计算密集型函数
- **数值计算优化**: 所有核心数值计算函数都经过Numba编译，实现接近C语言的执行速度
- **缓存机制**: 启用Numba缓存，避免重复编译开销

### 2. 并行计算架构
- **多进程并行**: 使用`ProcessPoolExecutor`实现无人机轨迹优化的并行计算
- **Numba并行**: 使用`@jit(parallel=True)`和`prange`实现参数评估的并行计算
- **CPU资源充分利用**: 自动检测并使用所有可用CPU核心

### 3. 算法参数优化
- **更大的搜索空间**: 增加差分进化算法的种群规模(60-80)和迭代次数(80-100)
- **更多候选点**: 速度候选点从5个增加到12个，提高解的质量
- **增强的搜索策略**: 调整变异和交叉参数，平衡探索和开发

### 4. 数据结构优化
- **Numba兼容数据**: 预处理全局数据为NumPy数组，避免Python对象开销
- **高效内存访问**: 优化数据布局，减少内存访问延迟
- **向量化计算**: 使用NumPy向量化操作替代循环

## 性能提升

### 预期性能提升
- **计算速度**: 2-5倍加速（取决于CPU核心数）
- **内存效率**: 减少Python对象创建，降低内存占用
- **扩展性**: 随CPU核心数线性扩展

### 优化的关键函数
1. `get_missile_pos_numba()` - 导弹位置计算
2. `get_drone_pos_numba()` - 无人机位置计算  
3. `get_smoke_pos_numba()` - 烟幕弹位置计算
4. `calc_smoke_effective_time_numba()` - 遮蔽时长计算
5. `parallel_smoke_evaluation()` - 并行参数评估

## 使用方法

### 直接运行
```bash
python solver5_numba_parallel.py
```

### 使用批处理文件
```bash
run_solver5_parallel.bat
```

### 性能对比测试
```bash
python performance_comparison.py
```

## 文件说明

- `solver5_numba_parallel.py` - 主要的并行优化求解器
- `run_solver5_parallel.bat` - Windows批处理运行脚本
- `performance_comparison.py` - 性能对比测试脚本
- `README_Numba_Parallel.md` - 本说明文件

## 依赖要求

### 必需库
```python
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
numba >= 0.56.0
scipy >= 1.7.0
```

### 安装依赖
```bash
pip install numba numpy pandas matplotlib scipy
```

## 系统要求

- **Python版本**: 3.7+
- **内存**: 建议4GB以上
- **CPU**: 多核心处理器（核心数越多性能提升越明显）
- **操作系统**: Windows/Linux/macOS

## 技术特性

### Numba优化特性
- **无Python模式**: 核心函数完全编译为机器码
- **类型推断**: 自动优化数值类型
- **循环优化**: 自动向量化和并行化循环
- **函数内联**: 消除函数调用开销

### 并行计算特性
- **进程级并行**: 无人机优化任务并行执行
- **线程级并行**: 参数评估并行计算
- **负载均衡**: 自动分配计算任务
- **内存共享**: 高效的进程间数据传递

## 结果输出

### 优化结果文件
- `smoke_optimization_result_numba_parallel.xlsx` - 优化结果Excel文件
- `smoke_optimization_visualization_numba_parallel.png` - 可视化图表

### 性能统计
程序运行时会输出：
- CPU核心数检测
- 各轮迭代耗时
- 总优化时间
- 性能提升说明

## 注意事项

1. **首次运行**: Numba首次编译会花费额外时间，后续运行会使用缓存
2. **内存使用**: 并行计算会增加内存使用，建议至少4GB可用内存
3. **CPU使用**: 程序会充分利用所有CPU核心，运行时CPU使用率较高
4. **结果一致性**: 由于随机种子和并行计算的差异，结果可能与原版本略有不同，但质量相当或更好

## 故障排除

### 常见问题
1. **Numba编译错误**: 确保NumPy版本兼容
2. **内存不足**: 减少并行进程数或增加系统内存
3. **导入错误**: 检查所有依赖库是否正确安装

### 调试模式
如需调试，可以临时禁用Numba优化：
```python
# 在文件开头添加
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
```

## 版本信息

- **版本**: 1.0
- **基于**: solver5.py
- **优化日期**: 2024年
- **兼容性**: 保持与原版本完全相同的算法逻辑和输出格式

## 许可证

与原项目保持一致的许可证。
