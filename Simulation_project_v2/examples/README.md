# MFG模拟系统 - 示例代码

本目录包含MFG模拟系统的完整运行示例。

---

## 📁 文件说明

### 主要示例

| 文件 | 说明 | 运行时间 |
|------|------|---------|
| `run_mfg_simulation.py` | **完整MFG均衡求解** | ~5-10分钟 |
| `exact_parameter_search_numba.py` | 参数搜索（Numba加速） | ~1-2小时 |
| `quick_parameter_test.py` | 快速参数测试 | ~5分钟 |
| `verify_params.py` | 参数验证 | ~1分钟 |

---

## 🚀 快速开始

### 1. 环境准备

确保已激活项目环境（**chaos_env**）：

```bash
# Windows PowerShell
C:\Users\21515\miniforge3\Scripts\activate chaos_env

# 或使用项目快捷脚本
cd D:\Python\2025DaChuang\Simulation_project_v2
activate_env.bat
```

### 2. 运行完整MFG模拟

```bash
cd D:\Python\2025DaChuang\Simulation_project_v2
python examples/run_mfg_simulation.py
```

**预期输出**:
```
============================================================
MFG模拟系统 - 完整示例
农村女性就业市场均衡求解
============================================================

步骤1: 加载配置...
✅ 使用内置测试配置

步骤2: 初始化MFG模拟器...
============================================================
初始化MFG模拟器...
============================================================
✅ 状态空间: StateSpace(dimension=4, ...)
✅ 稀疏网格: 137个点 (level=3, 效率=53.52%)
✅ 贝尔曼求解器: BellmanSolver(n_points=137, ...)
✅ KFE求解器: KFESolver(...)
============================================================
✅ MFG模拟器初始化完成！
============================================================

步骤3: 开始求解MFG均衡...
迭代   0: diff_V=1.23e+03, diff_a=2.45e-01, diff_u=1.23e-02 | ...
迭代  10: diff_V=5.67e-02, diff_a=1.23e-03, diff_u=2.34e-04 | ...
...
✅ MFG均衡在 25 次迭代后收敛！

步骤4-8: 保存结果、绘图、查询示例...

============================================================
✅ MFG模拟完成！
============================================================
```

### 3. 查看结果

运行完成后，结果保存在 `results/mfg/` 目录：

```
results/mfg/
├── mfg_equilibrium.npz       # 完整均衡结果（V, a*, m）
├── mfg_history.npz            # 历史演化数据
├── mfg_metadata.json          # 元数据（JSON格式）
├── equilibrium.npz            # 均衡对象
└── mfg_convergence.png        # 收敛曲线图
```

---

## 📊 示例详解

### `run_mfg_simulation.py` - 完整MFG模拟

**功能**：
- 初始化4维稀疏网格（T, S, D, W）
- 交替求解贝尔曼方程和KFE
- 检查三重收敛标准（V, a, u）
- 保存完整均衡结果
- 生成收敛曲线图

**配置说明**：
```python
config = {
    'sparse_grid': {
        'level': 3,  # 精度级别：3=快速测试，5=生产运行
    },
    'bellman': {
        'n_effort_grid': 11,  # 努力水平离散点数
        'max_iterations': 100,  # 贝尔曼最大迭代次数
    },
    'convergence': {
        'epsilon_V': 1e-3,  # 价值函数收敛容差
        'epsilon_a': 1e-3,  # 策略收敛容差
        'epsilon_u': 1e-3,  # 失业率收敛容差
        'max_iterations': 50,  # MFG主循环最大迭代次数
    }
}
```

**调整参数以加快/放慢运行**：

| 参数 | 快速测试 | 生产运行 | 影响 |
|------|---------|---------|------|
| `sparse_grid.level` | 3 (~200点) | 5 (~1500点) | 网格精度 |
| `bellman.n_effort_grid` | 11 | 21 | 策略搜索精度 |
| `convergence.max_iterations` | 30 | 500 | 最大迭代次数 |
| `convergence.epsilon_V` | 1e-3 | 1e-4 | 收敛严格程度 |

---

## 🔧 高级用法

### 1. 使用真实匹配函数参数

如果已经运行了Module 3的匹配函数估计：

```python
import json
import numpy as np
from modules.mfg.mfg_simulator import MFGSimulator

# 加载估计的匹配函数参数
with open('results/estimation/match_function_params.json', 'r') as f:
    match_params_dict = json.load(f)

# 转换为Numba格式（参考match_function.py）
match_func_params = convert_to_numba_params(match_params_dict)

# 初始化模拟器
simulator = MFGSimulator(config, match_func_params=match_func_params)
result = simulator.solve()
```

### 2. 加载保存的均衡结果

```python
from core.data_structures import MFGEquilibriumSparseGrid

# 加载
equilibrium = MFGEquilibriumSparseGrid.load('results/mfg/equilibrium.npz')

# 查询
print(equilibrium.summary())
state = np.array([40.0, 0.5, 0.5, 3000.0])
V_U = equilibrium.get_value_at_state(state, 'unemployed')
a_star = equilibrium.get_optimal_effort(state)
```

### 3. 绘制自定义可视化

```python
import matplotlib.pyplot as plt

# 加载历史数据
history_data = np.load('results/mfg/mfg_history.npz')
u_rates = history_data['unemployment_rate']

# 绘制失业率演化
plt.figure(figsize=(10, 6))
plt.plot(u_rates, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Unemployment Rate')
plt.title('MFG Equilibrium Convergence')
plt.grid(True)
plt.savefig('my_custom_plot.png', dpi=300)
plt.show()
```

---

## ⚙️ 配置文件

完整配置文件位于 `config/default/mfg.yaml`，包含：

1. **状态空间定义** (`state_space`)
   - T: 工作时长 [15, 70]
   - S: 工作能力 [2, 44]
   - D: 数字素养 [0, 20]
   - W: 期望工资 [1400, 8000]

2. **稀疏网格设置** (`sparse_grid`)
   - library: 'chaospy'
   - level: 5
   - dimension: 4

3. **状态转移参数** (`state_transition`)
   - gamma_T, gamma_S, gamma_D, gamma_W
   - T_max, W_min

4. **效用函数参数** (`utility`)
   - 失业: b_0=500, kappa=1.0
   - 就业: alpha_T=10.0

5. **求解器参数** (`bellman`, `kfe`)
   - rho=0.9（贴现因子）
   - mu=0.05（离职率）
   - n_effort_grid=21

6. **收敛标准** (`convergence`)
   - epsilon_V=1e-4
   - epsilon_a=1e-4
   - epsilon_u=1e-3
   - max_iterations=500

---

## 📝 常见问题

### Q1: 运行时间太长怎么办？

**A**: 调整这些参数以加快速度：
```yaml
sparse_grid:
  level: 3  # 降低精度（默认5）

bellman:
  n_effort_grid: 11  # 减少努力网格点（默认21）

convergence:
  max_iterations: 30  # 减少最大迭代（默认500）
  epsilon_V: 1e-3  # 放宽收敛容差（默认1e-4）
```

### Q2: 如何判断是否收敛？

**A**: 查看输出中的三个指标：
- `diff_V < epsilon_V`: 价值函数稳定
- `diff_a < epsilon_a`: 策略稳定
- `diff_u < epsilon_u`: 失业率稳定

全部满足则收敛。

### Q3: 内存不足怎么办？

**A**: 
- 降低 `sparse_grid.level` (每降1级，点数减少约60%)
- 关闭 `output.save_intermediate` (不保存中间结果)
- 使用更强大的机器

### Q4: 如何并行加速？

**A**: Numba并行已自动启用：
```yaml
optimization:
  use_numba: true
  parallel: true  # 自动多核并行
```

---

## 📚 相关文档

- **理论基础**: `docs/原始研究计划/研究计划.md`
- **开发文档**: `docs/developerdocs/modules/Phase4_MFG_Final_Parameters.md`
- **API参考**: 各模块的docstring

---

## 🔍 调试技巧

### 1. 查看详细日志

```bash
# 修改日志级别为DEBUG
export LOG_LEVEL=DEBUG
python examples/run_mfg_simulation.py
```

### 2. 检查中间结果

在 `mfg_simulator.py` 的 `solve()` 方法中添加断点：
```python
if iteration % 10 == 0:
    print(f"当前 V_U 统计: mean={np.mean(V_U_new):.2f}, std={np.std(V_U_new):.2f}")
```

### 3. 验证Numba优化

```python
# 预热Numba（首次运行会编译）
import time
start = time.time()
result = simulator.solve()
print(f"首次运行: {time.time() - start:.2f}s")

# 第二次运行应该更快
start = time.time()
result = simulator.solve()
print(f"第二次运行: {time.time() - start:.2f}s (已编译)")
```

---

**作者**: AI Assistant  
**最后更新**: 2025-10-03  
**版本**: v2.0

