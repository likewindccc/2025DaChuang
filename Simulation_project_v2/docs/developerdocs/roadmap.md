# 开发路线图 (Development Roadmap)

**项目**: Simulation_project_v2  
**版本**: 2.0  
**开始日期**: 2025-09-30  
**预计完成**: 2026-03-31 (6个月)

---

## 📋 目录

- [1. 总体规划](#1-总体规划)
- [2. 详细里程碑](#2-详细里程碑)
- [3. 交付物清单](#3-交付物清单)
- [4. 风险管理](#4-风险管理)

---

## 1. 总体规划

### 1.1 开发阶段概览

```
Phase 1: 基础框架 (2周)       Week 1-2
    ↓
Phase 2: Module 1 (3周)       Week 3-5
    ↓
Phase 3: Module 2+3 (4周)     Week 6-9
    ↓
Phase 4: Module 4 (6周)       Week 10-15
    ↓
Phase 5: Module 5 (3周)       Week 16-18
    ↓
Phase 6: 集成优化 (2周)       Week 19-20
    ↓
Phase 7: 文档完善 (2周)       Week 21-22
```

**总工期**: 22周 (~5.5个月)  
**缓冲时间**: 2周  
**总计**: 24周 (~6个月)

### 1.2 人力安排

- **核心开发**: 1人全职（用户 + AI助手协作）
- **指导教师**: 定期review
- **预计总工时**: 700-900小时

### 1.3 关键里程碑

| 里程碑 | 日期 | 交付物 |
|-------|------|--------|
| M1: 框架搭建完成 | Week 2 | 基础架构代码 |
| M2: 虚拟个体生成 | Week 5 | Module 1完成 |
| M3: 匹配与估计 | Week 9 | Module 2+3完成 |
| M4: MFG求解器 | Week 15 | Module 4完成 |
| M5: 参数校准 | Week 18 | Module 5完成 |
| M6: 系统集成 | Week 20 | 完整系统 |
| M7: 项目交付 | Week 22 | 全部文档 |

---

## 2. 详细里程碑

### Phase 1: 基础框架搭建 (Week 1-2)

**目标**: 建立项目基础设施

#### Week 1: 目录结构与核心类

**任务列表**:
- [x] 创建完整目录结构
- [ ] 实现 `src/core/` 基础类
  - [ ] `base_agent.py` - Agent基类
  - [ ] `base_generator.py` - Generator基类
  - [ ] `base_simulator.py` - Simulator基类
  - [ ] `data_structures.py` - 数据结构
  - [ ] `exceptions.py` - 异常定义
  - [ ] `types.py` - 类型注解
- [ ] 配置系统 (`config_loader.py`)
- [ ] 日志系统 (`utils/logging_config.py`)

**交付物**:
- [x] `README.md`
- [x] 完整目录结构
- [x] `docs/developerdocs/architecture.md`
- [x] `docs/developerdocs/tech_stack.md`
- [x] `docs/developerdocs/roadmap.md` (本文档)
- [ ] `src/core/` 所有基类
- [ ] `config/default/base_config.yaml`

**验收标准**:
- 所有基类可被导入且无语法错误
- 配置加载器可正确解析YAML
- 日志系统可正常输出

#### Week 2: 工具函数与测试框架

**任务列表**:
- [ ] 实现 `src/utils/` 工具模块
  - [ ] `data_validation.py` - 数据验证
  - [ ] `metrics.py` - 评估指标
  - [ ] `visualization.py` - 基础可视化
- [ ] 搭建测试框架
  - [ ] `tests/conftest.py` - pytest配置
  - [ ] 测试数据fixtures
- [ ] 主程序入口 (`src/main.py`)

**交付物**:
- [ ] 完整的工具函数库
- [ ] pytest测试框架
- [ ] 可运行的main.py骨架

**验收标准**:
- `pytest tests/` 可执行（即使测试为空）
- `python -m src.main --help` 显示帮助信息
- 代码符合PEP8规范

---

### Phase 2: Module 1 实现 (Week 3-5)

**目标**: 完成虚拟主体生成器

#### Week 3: Copula引擎重构

**任务列表**:
- [ ] 从旧版提取Copula核心算法
  - [ ] `copula_engine.py` - Copula拟合与采样
  - [ ] `marginal_estimator.py` - 边际分布估计
- [ ] 重构为模块化结构
- [ ] 添加完整docstring
- [ ] 单元测试

**关键代码**:
```python
# src/modules/population/copula_engine.py
class CopulaEngine:
    def __init__(self, config: Dict):
        pass
    
    def fit(self, uniform_data: np.ndarray) -> None:
        """拟合Copula模型"""
        pass
    
    def sample(self, n_samples: int) -> np.ndarray:
        """从Copula采样"""
        pass
```

**交付物**:
- [ ] `copula_engine.py` (300-400行)
- [ ] `marginal_estimator.py` (200-300行)
- [ ] `tests/unit/test_copula_engine.py`

#### Week 4: 劳动力生成器

**任务列表**:
- [ ] 实现 `labor_generator.py`
  - [ ] 继承 `BaseGenerator`
  - [ ] 集成Copula引擎
  - [ ] 数据验证逻辑
- [ ] 配置文件 `config/default/population.yaml`
- [ ] 单元测试与集成测试

**交付物**:
- [ ] `labor_generator.py` (400-500行)
- [ ] 配置文件
- [ ] 完整测试套件

**验收标准**:
- 可生成10,000个虚拟劳动力（<5秒）
- 边际分布保持良好（KS检验p>0.05）
- 相关性误差<0.05

#### Week 5: 企业生成器

**任务列表**:
- [ ] 实现 `enterprise_generator.py`
  - [ ] 四维正态分布
  - [ ] 参数设置接口（用于校准）
  - [ ] 数据验证
- [ ] 文献数据准备 (`data/input/literature_stats.yaml`)
- [ ] 单元测试

**交付物**:
- [ ] `enterprise_generator.py` (300-400行)
- [ ] 文献统计数据
- [ ] 测试套件

**验收标准**:
- 可生成5,000个虚拟企业（<3秒）
- 参数可被外部设置（校准接口）
- 通过Mahalanobis距离检验

**M2里程碑审查**:
- [ ] Module 1全部代码完成
- [ ] 测试覆盖率>85%
- [ ] 文档完整
- [ ] Code Review通过

---

### Phase 3: Module 2 + 3 实现 (Week 6-9)

**目标**: 完成匹配引擎与匹配函数估计

#### Week 6: Gale-Shapley算法

**任务列表**:
- [ ] 实现 `gale_shapley.py`
  - [ ] 经典GS算法
  - [ ] 稳定性检验
- [ ] 实现 `preference.py`
  - [ ] 劳动力偏好计算
  - [ ] 企业偏好计算
- [ ] **Numba优化**: 偏好矩阵计算

**Numba优化目标**:
```python
@njit(parallel=True)
def compute_preference_matrix_batch(
    labor_features: np.ndarray,    # (n_labor, 4)
    enterprise_features: np.ndarray # (n_enterprise, 4)
) -> np.ndarray:                    # (n_labor, n_enterprise)
    """
    并行计算偏好矩阵
    目标: 10,000 × 5,000 矩阵 < 500ms
    """
```

**交付物**:
- [ ] `gale_shapley.py`
- [ ] `preference.py`
- [ ] Numba优化版本
- [ ] 性能基准测试

#### Week 7: 匹配引擎集成

**任务列表**:
- [ ] 实现 `matching_result.py` - 匹配结果数据结构
- [ ] 集成GS算法与偏好计算
- [ ] 批量模拟功能（多轮次、多θ值）
- [ ] 配置文件 `config/default/matching.yaml`

**交付物**:
- [ ] 完整的匹配引擎
- [ ] 配置文件
- [ ] 集成测试

**验收标准**:
- 单轮匹配10K×5K < 30秒
- 生成可用于Logit回归的数据
- 稳定性100%（无不稳定匹配对）

#### Week 8: ABM数据生成

**任务列表**:
- [ ] 实现 `data_generator.py`
  - [ ] 多轮次模拟
  - [ ] θ值扰动策略
  - [ ] 努力水平a扰动
- [ ] 生成训练数据集
- [ ] 数据质量验证

**扰动策略**:
```python
theta_range = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
effort_levels = np.linspace(0, 1, 11)  # 0, 0.1, ..., 1.0
n_rounds_per_combination = 5

total_simulations = 7 × 11 × 5 = 385轮
```

**交付物**:
- [ ] ABM数据生成器
- [ ] 训练数据集（~100K样本）
- [ ] 数据质量报告

#### Week 9: Logit回归与匹配函数

**任务列表**:
- [ ] 实现 `logit_estimator.py`
  - [ ] Statsmodels集成
  - [ ] 参数估计
  - [ ] 模型诊断
- [ ] 实现 `match_function.py`
  - [ ] **Numba优化版匹配函数**
  - [ ] 参数加载
- [ ] 配置文件 `config/default/estimation.yaml`

**核心输出**: 匹配函数λ
```python
@njit(fastmath=True, cache=True)
def match_function(x, sigma, a, theta):
    """
    λ(x, σ, a, θ) = 1 / (1 + exp(-logit))
    
    logit = δ₀ + δₓ'x + δₐa + δ_θ log(θ) + ...
    """
    logit = (
        delta_0 +
        delta_x[0] * x[0] + delta_x[1] * x[1] + 
        delta_x[2] * x[2] + delta_x[3] * x[3] +
        delta_sigma[0] * sigma[0] + delta_sigma[1] * sigma[1] +
        delta_a * a +
        delta_theta * np.log(theta)
    )
    return 1.0 / (1.0 + np.exp(-logit))
```

**交付物**:
- [ ] Logit估计器
- [ ] Numba优化匹配函数
- [ ] 估计报告（参数显著性、伪R²等）

**验收标准**:
- 匹配函数预测准确率>75%
- Numba版本加速比>20x
- 参数统计显著性p<0.05

**M3里程碑审查**:
- [ ] Module 2+3全部完成
- [ ] Numba优化达标
- [ ] 测试覆盖率>85%

---

### Phase 4: Module 4 实现 (Week 10-15)

**目标**: 完成MFG求解器（简化版）

#### Week 10-11: 状态空间设计

**任务列表**:
- [ ] 实现 `state_space.py`
  - [ ] 状态空间离散化（50×50网格）
  - [ ] 状态转移规则
  - [ ] 边界处理
- [ ] 配置文件 `config/default/mfg.yaml`

**状态空间设计**:
```python
# 简化状态空间：只考虑(T, S)两维
T_grid = np.linspace(0, 168, 50)  # 工作时长
S_grid = np.linspace(0, 100, 50)  # 技能评分

# 状态索引: (i, j) → state_value
state_grid = np.meshgrid(T_grid, S_grid)  # (50, 50, 2)
```

**交付物**:
- [ ] 状态空间模块
- [ ] 配置文件
- [ ] 单元测试

#### Week 12-13: 贝尔曼方程求解

**任务列表**:
- [ ] 实现 `bellman_solver.py`
  - [ ] 值迭代算法
  - [ ] **Numba优化内循环**
  - [ ] 收敛判断
- [ ] 实现 `value_iteration.py`
  - [ ] 最优努力搜索
  - [ ] 策略函数

**Numba优化核心**:
```python
@njit
def bellman_iteration(
    V_current: np.ndarray,     # (50, 50) 当前值函数
    match_func_params: np.ndarray,
    theta: float,
    rho: float,               # 贴现因子
    kappa: float              # 努力成本系数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单次贝尔曼迭代
    返回: (新值函数, 最优努力)
    
    目标: 50×50网格 < 10ms
    """
    V_new = np.zeros_like(V_current)
    a_optimal = np.zeros_like(V_current)
    
    for i in range(50):
        for j in range(50):
            # 枚举努力水平找最优
            best_value = -np.inf
            best_effort = 0.0
            
            for a in np.linspace(0, 1, 21):  # 离散化努力
                # 计算贝尔曼方程右侧
                instant_utility = b(state) - 0.5 * kappa * a**2
                continuation = rho * (
                    match_prob(state, a, theta) * V_E[next_state] +
                    (1 - match_prob(...)) * V_U[next_state]
                )
                value = instant_utility + continuation
                
                if value > best_value:
                    best_value = value
                    best_effort = a
            
            V_new[i, j] = best_value
            a_optimal[i, j] = best_effort
    
    return V_new, a_optimal
```

**交付物**:
- [ ] 贝尔曼求解器
- [ ] Numba优化版本
- [ ] 性能基准测试

#### Week 14: KFE演化求解

**任务列表**:
- [ ] 实现 `kfe_solver.py`
  - [ ] KFE离散化
  - [ ] 人口分布演化
  - [ ] **Numba优化**
- [ ] 失业率与θ更新

**KFE演化**:
```python
@njit
def kfe_step(
    m_U_current: np.ndarray,   # (50, 50) 当前失业分布
    m_E_current: np.ndarray,   # (50, 50) 当前就业分布
    a_optimal: np.ndarray,     # (50, 50) 最优努力
    theta: float,
    mu: float                  # 外生离职率
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单步KFE演化
    m_{t+1} = T(m_t, a*)
    """
    m_U_new = np.zeros_like(m_U_current)
    m_E_new = np.zeros_like(m_E_current)
    
    for i in range(50):
        for j in range(50):
            # 未匹配的失业者
            m_U_new[i, j] += (1 - λ(state, a, θ)) * m_U_current[i, j]
            
            # 匹配成功的
            m_E_new[i, j] += λ(state, a, θ) * m_U_current[i, j]
            
            # 外生离职的
            m_U_new[i, j] += μ * m_E_current[i, j]
            m_E_new[i, j] += (1 - μ) * m_E_current[i, j]
    
    return m_U_new, m_E_new
```

**交付物**:
- [ ] KFE求解器
- [ ] Numba优化版本

#### Week 15: MFG主循环集成

**任务列表**:
- [ ] 实现 `mfg_simulator.py`
  - [ ] 主循环: Bellman + KFE
  - [ ] 收敛判断（三个标准）
  - [ ] 结果记录与保存
- [ ] 集成测试
- [ ] 小规模验证（10×10网格）

**主循环伪代码**:
```python
def solve_mfg_equilibrium():
    # 初始化
    V_U, V_E = initialize_value_functions()
    m_U, m_E = initialize_distributions()
    theta = V / sum(m_U)
    
    for iteration in range(max_iterations):
        # Step 1: 贝尔曼方程
        V_U_new, V_E_new, a_optimal = bellman_solver.solve(
            V_U, V_E, theta
        )
        
        # Step 2: KFE演化
        m_U_new, m_E_new = kfe_solver.evolve(
            m_U, m_E, a_optimal, theta
        )
        
        # Step 3: 更新θ
        theta_new = V / sum(m_U_new)
        
        # Step 4: 检查收敛
        if check_convergence(V_U_new, V_U, a_optimal, theta_new, theta):
            break
        
        # 更新
        V_U, V_E = V_U_new, V_E_new
        m_U, m_E = m_U_new, m_E_new
        theta = theta_new
    
    return MFGEquilibrium(V_U, V_E, a_optimal, m_U, m_E, theta)
```

**交付物**:
- [ ] 完整MFG模拟器
- [ ] 小规模测试成功
- [ ] 性能报告

**验收标准**:
- 10×10网格可收敛（<50次迭代）
- 50×50网格可运行（< 10分钟）
- Numba优化达标

**M4里程碑审查**:
- [ ] Module 4全部完成
- [ ] MFG可求解简单案例
- [ ] 性能满足要求

---

### Phase 5: Module 5 实现 (Week 16-18)

**目标**: 完成遗传算法参数校准

#### Week 16: 参数空间定义

**任务列表**:
- [ ] 实现 `parameter_space.py`
  - [ ] 定义15-20个待校准参数
  - [ ] 参数边界约束
  - [ ] 参数编码/解码
- [ ] 文献数据收集与整理

**待校准参数（初步）**:
```python
# 企业端参数（4维正态分布）
μ_enterprise = [μ_T, μ_S, μ_D, μ_W]        # 4个均值参数
Σ_enterprise = [σ²_T, σ²_S, σ²_D, σ²_W,    # 4个方差参数
                ρ_TS, ρ_TD, ρ_TW,           # 6个相关系数
                ρ_SD, ρ_SW, ρ_DW]           # (4×3/2=6)

# MFG参数
rho = 0.95                                  # 贴现因子
kappa = 1.0                                 # 努力成本系数
mu = 0.05                                   # 外生离职率

# 总计: 4+4+6+3 = 17个参数
```

**交付物**:
- [ ] 参数空间定义
- [ ] 文献数据整理
- [ ] 参数约束验证

#### Week 17: 目标函数实现

**任务列表**:
- [ ] 实现 `objective_function.py`
  - [ ] 多指标计算
  - [ ] 加权汇总
  - [ ] **Numba优化模拟核心**
- [ ] 目标数据准备（失业率等）

**目标函数**:
```python
def objective_function(parameters: np.ndarray) -> float:
    """
    目标函数: 最小化模拟指标与真实指标的加权差异
    
    f(θ) = Σᵢ wᵢ |simᵢ(θ) - targetᵢ|²
    
    指标:
    - 失业率 (权重0.4)
    - 平均匹配率 (权重0.3)
    - 平均工资 (权重0.2)
    - 市场紧张度 (权重0.1)
    """
    # 1. 设置企业生成器参数
    enterprise_gen.set_parameters(
        mean=parameters[:4],
        cov=reconstruct_cov_matrix(parameters[4:14])
    )
    
    # 2. 运行完整MFG模拟
    equilibrium = mfg_simulator.solve(
        rho=parameters[14],
        kappa=parameters[15],
        mu=parameters[16]
    )
    
    # 3. 提取模拟指标
    sim_unemployment = equilibrium.unemployment_rate
    sim_match_rate = equilibrium.match_rate
    sim_avg_wage = equilibrium.average_wage
    sim_theta = equilibrium.theta
    
    # 4. 计算加权误差
    error = (
        0.4 * (sim_unemployment - target_unemployment)**2 +
        0.3 * (sim_match_rate - target_match_rate)**2 +
        0.2 * (sim_avg_wage - target_avg_wage)**2 +
        0.1 * (sim_theta - target_theta)**2
    )
    
    return error
```

**交付物**:
- [ ] 目标函数实现
- [ ] 目标数据文件
- [ ] 单次评估测试（确保可运行）

#### Week 18: 遗传算法集成

**任务列表**:
- [ ] 实现 `genetic_algorithm.py`
  - [ ] DEAP工具箱配置
  - [ ] 并行评估
  - [ ] 中间结果保存
- [ ] 配置文件 `config/default/calibration.yaml`
- [ ] 小规模校准测试

**DEAP配置**:
```python
from deap import base, creator, tools, algorithms

# 定义适应度和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 个体生成（17维参数向量）
toolbox.register("individual", tools.initIterate, creator.Individual,
                lambda: random_parameters_in_bounds())

# 种群
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 遗传算子
toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# 并行评估
from multiprocessing import Pool
pool = Pool(processes=4)
toolbox.register("map", pool.map)
```

**交付物**:
- [ ] 遗传算法实现
- [ ] 配置文件
- [ ] 小规模测试（10代，种群20）

**验收标准**:
- 单次评估可完成（< 2分钟）
- 小规模校准可收敛
- 并行评估正常工作

**M5里程碑审查**:
- [ ] Module 5全部完成
- [ ] 校准系统可运行
- [ ] 准备进行大规模校准

---

### Phase 6: 系统集成与优化 (Week 19-20)

**目标**: 端到端集成，性能优化

#### Week 19: 端到端集成

**任务列表**:
- [ ] 主程序完善 (`src/main.py`)
  - [ ] 命令行参数解析
  - [ ] 模式选择（simulation/calibration/policy）
  - [ ] 进度显示
- [ ] 集成测试 (`tests/integration/`)
  - [ ] 完整流程测试
  - [ ] 多配置测试
- [ ] Bug修复

**main.py架构**:
```python
import argparse
from src.config_loader import load_config
from src.modules.population import LaborGenerator, EnterpriseGenerator
from src.modules.matching import MatchingEngine
# ... 其他模块

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['simulation', 'calibration', 'policy'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == 'simulation':
        run_simulation(config)
    elif args.mode == 'calibration':
        run_calibration(config)
    elif args.mode == 'policy':
        run_policy_analysis(config)
```

**交付物**:
- [ ] 完善的main.py
- [ ] 集成测试套件
- [ ] Bug修复清单

#### Week 20: 性能优化与基准测试

**任务列表**:
- [ ] 性能剖析（cProfile）
- [ ] Numba优化调优
- [ ] 内存优化
- [ ] 性能基准测试 (`tests/benchmarks/`)

**基准测试**:
```python
# tests/benchmarks/test_large_scale.py
def test_10k_labor_5k_enterprise(benchmark):
    """10K劳动力 × 5K企业 基准测试"""
    result = benchmark(
        run_full_simulation,
        n_labor=10000,
        n_enterprise=5000
    )
    
    assert result.execution_time < 600  # < 10分钟
    assert result.memory_mb < 4096      # < 4GB
```

**交付物**:
- [ ] 性能优化报告
- [ ] 基准测试结果
- [ ] 优化后的代码

**验收标准**:
- 10K×5K完整模拟 < 10分钟
- 内存使用 < 4GB
- Numba加速比达标

**M6里程碑审查**:
- [ ] 完整系统可运行
- [ ] 性能目标达成
- [ ] 所有测试通过

---

### Phase 7: 文档完善 (Week 21-22)

**目标**: 完善所有文档，准备交付

#### Week 21: API文档与用户手册

**任务列表**:
- [ ] Sphinx API文档生成
- [ ] 用户手册编写
  - [ ] 快速开始
  - [ ] 配置指南
  - [ ] 故障排查
- [ ] FAQ整理

**交付物**:
- [ ] `docs/developerdocs/api_reference.md`
- [ ] `docs/userdocs/user_manual.md`
- [ ] `docs/userdocs/configuration_guide.md`
- [ ] `docs/userdocs/faq.md`

#### Week 22: 学术文档与最终审查

**任务列表**:
- [ ] 学术文档编写
  - [ ] 方法论文档
  - [ ] 算法说明（数学公式）
  - [ ] 参数校准报告
  - [ ] 实验结果分析
- [ ] 代码审查
- [ ] 最终测试
- [ ] 项目打包

**交付物**:
- [ ] `docs/academicdocs/methodology.md`
- [ ] `docs/academicdocs/algorithms.md`
- [ ] `docs/academicdocs/calibration.md`
- [ ] `docs/academicdocs/experiments.md`
- [ ] 完整项目包

**M7里程碑审查**:
- [ ] 所有文档完成
- [ ] 代码质量达标
- [ ] 准备交付

---

## 3. 交付物清单

### 3.1 代码交付物

| 类别 | 文件数量 | 代码行数（估计） |
|------|---------|----------------|
| 核心模块 | 30+ | 8,000-10,000 |
| 工具函数 | 10+ | 1,000-1,500 |
| 测试代码 | 40+ | 3,000-4,000 |
| 配置文件 | 10+ | 500-800 |
| **总计** | **90+** | **12,500-16,300** |

### 3.2 文档交付物

| 类别 | 文档数量 | 页数（估计） |
|------|---------|-------------|
| 用户文档 | 3 | 30-50 |
| 开发者文档 | 5 | 60-100 |
| 学术文档 | 4 | 40-80 |
| **总计** | **12** | **130-230** |

### 3.3 数据交付物

- 虚拟劳动力池 (10,000个体)
- 虚拟企业池 (5,000个体)
- ABM训练数据 (~100K样本)
- 校准后参数
- 实验结果数据

---

## 4. 风险管理

### 4.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| MFG不收敛 | 高 | 高 | 简化状态空间，固定迭代上限 |
| Numba兼容性问题 | 中 | 中 | 预留纯Python备选方案 |
| 校准时间过长 | 高 | 中 | 并行计算，减少代数 |
| 内存溢出 | 中 | 高 | 批处理，结果流式保存 |

### 4.2 时间风险

**缓冲策略**:
- 每个Phase预留20%缓冲时间
- 总体预留2周机动时间
- 如Phase 4超期，考虑进一步简化MFG

### 4.3 质量风险

**质量保证措施**:
- 每周代码审查
- 持续集成测试
- 里程碑验收标准
- 导师定期review

---

## 5. 附录

### 5.1 每周工作时间估算

- **开发时间**: 30-40 小时/周
- **文档时间**: 5-10 小时/周
- **测试时间**: 5-8 小时/周
- **总计**: 40-58 小时/周

### 5.2 关键日期

- **启动**: 2025-09-30 (Week 0)
- **M2审查**: 2025-11-04 (Week 5)
- **M3审查**: 2025-12-02 (Week 9)
- **M4审查**: 2026-01-13 (Week 15)
- **M5审查**: 2026-02-03 (Week 18)
- **M6审查**: 2026-02-17 (Week 20)
- **项目交付**: 2026-03-03 (Week 22)
- **最终截止**: 2026-03-31 (Week 26, 含缓冲)

---

**文档维护**: 每个Phase结束后更新  
**责任人**: 项目负责人  
**审阅频率**: 每2周
