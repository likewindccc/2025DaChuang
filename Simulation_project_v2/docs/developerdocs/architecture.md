# Simulation_project_v2 项目架构设计文档

**版本**: 2.0  
**日期**: 2025-09-30  
**状态**: 设计阶段

---

## 📋 目录

- [1. 项目概述](#1-项目概述)
- [2. 整体架构](#2-整体架构)
- [3. 目录结构详解](#3-目录结构详解)
- [4. 核心模块设计](#4-核心模块设计)
- [5. 数据流转](#5-数据流转)
- [6. 接口设计](#6-接口设计)
- [7. 技术决策](#7-技术决策)

---

## 1. 项目概述

### 1.1 项目定位

**Simulation_project_v2** 是农村女性就业市场MFG模拟系统的第二版，完全重构以满足：
- 代码可读性和可复用性
- 严格的PEP8规范
- 完整的测试覆盖
- Numba强制优化
- 学术级严谨性

### 1.2 核心目标

1. **微观层面**: 模拟个体（农村女性）的努力决策与状态演化
2. **中观层面**: 实现劳动力与企业的双边匹配
3. **宏观层面**: 求解MFG均衡，分析市场动态
4. **应用层面**: 参数校准与政策分析

### 1.3 理论基础

- **平均场博弈(MFG)**: 贝尔曼方程 + KFE演化
- **基于主体建模(ABM)**: Gale-Shapley匹配算法
- **Copula理论**: 解决变量非独立性
- **黑盒优化**: 遗传算法参数校准

---

## 2. 整体架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│              应用层 (Application Layer)              │
│  main.py, 命令行接口, 配置加载                        │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              业务层 (Business Layer)                 │
│  5个核心模块: Population, Matching, Estimation,      │
│             MFG, Calibration                         │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              核心层 (Core Layer)                     │
│  基础类, 数据结构, 接口定义                           │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              工具层 (Utility Layer)                  │
│  数值计算, 数据验证, 日志, 可视化                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
Calibration (Module 5)
    ↓ 依赖
MFG (Module 4)
    ↓ 依赖
Estimation (Module 3)
    ↓ 依赖
Matching (Module 2)
    ↓ 依赖
Population (Module 1)
    ↓ 依赖
Core (基础类) + Utils (工具函数)
```

**依赖原则**: 高层模块依赖低层模块，低层模块不依赖高层模块

---

## 3. 目录结构详解

### 3.1 完整目录树

```
Simulation_project_v2/
│
├── README.md                      # 项目说明
├── setup_directories.py           # 目录初始化脚本
├── requirements.txt               # 依赖清单 (复用旧版)
│
├── src/                           # 源代码根目录
│   ├── __init__.py
│   ├── main.py                    # 主程序入口
│   ├── config_loader.py           # 配置加载器
│   │
│   ├── core/                      # 核心基础模块
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Agent基类
│   │   ├── base_generator.py     # Generator基类
│   │   ├── base_simulator.py     # Simulator基类
│   │   ├── data_structures.py    # 核心数据结构
│   │   ├── exceptions.py          # 自定义异常
│   │   └── types.py               # 类型定义
│   │
│   ├── modules/                   # 5个核心模块
│   │   ├── __init__.py
│   │   │
│   │   ├── population/           # Module 1: 主体生成器
│   │   │   ├── __init__.py
│   │   │   ├── copula_engine.py           # Copula核心引擎
│   │   │   ├── marginal_estimator.py      # 边际分布估计
│   │   │   ├── labor_generator.py         # 劳动力生成器
│   │   │   ├── enterprise_generator.py    # 企业生成器
│   │   │   └── config.py                  # 模块配置
│   │   │
│   │   ├── matching/             # Module 2: 匹配引擎
│   │   │   ├── __init__.py
│   │   │   ├── gale_shapley.py            # GS算法实现
│   │   │   ├── preference.py              # 偏好计算
│   │   │   ├── matching_result.py         # 匹配结果数据结构
│   │   │   └── config.py
│   │   │
│   │   ├── estimation/           # Module 3: 匹配函数估计
│   │   │   ├── __init__.py
│   │   │   ├── logit_estimator.py         # Logit回归估计器
│   │   │   ├── data_generator.py          # ABM数据生成器
│   │   │   ├── match_function.py          # 匹配函数λ(x,σ,a,θ)
│   │   │   └── config.py
│   │   │
│   │   ├── mfg/                  # Module 4: MFG求解器
│   │   │   ├── __init__.py
│   │   │   ├── bellman_solver.py          # 贝尔曼方程求解
│   │   │   ├── kfe_solver.py              # KFE演化求解
│   │   │   ├── mfg_simulator.py           # MFG主模拟器
│   │   │   ├── value_iteration.py         # 值迭代算法
│   │   │   ├── state_space.py             # 状态空间离散化
│   │   │   └── config.py
│   │   │
│   │   └── calibration/          # Module 5: 参数校准
│   │       ├── __init__.py
│   │       ├── genetic_algorithm.py       # 遗传算法（DEAP）
│   │       ├── objective_function.py      # 目标函数
│   │       ├── parameter_space.py         # 参数空间定义
│   │       └── config.py
│   │
│   └── utils/                     # 工具函数库
│       ├── __init__.py
│       ├── numba_acceleration.py  # Numba优化函数
│       ├── data_validation.py     # 数据验证
│       ├── logging_config.py      # 日志配置
│       ├── visualization.py       # 可视化工具
│       └── metrics.py             # 评估指标计算
│
├── config/                        # 配置文件
│   ├── default/
│   │   ├── base_config.yaml       # 基础配置
│   │   ├── population.yaml        # Module 1配置
│   │   ├── matching.yaml          # Module 2配置
│   │   ├── estimation.yaml        # Module 3配置
│   │   ├── mfg.yaml               # Module 4配置
│   │   └── calibration.yaml       # Module 5配置
│   │
│   └── experiments/               # 实验配置
│       ├── baseline.yaml          # 基准实验
│       ├── policy_a.yaml          # 政策A实验
│       └── sensitivity.yaml       # 敏感性分析
│
├── data/                          # 数据文件
│   ├── input/
│   │   ├── labor_survey.csv       # 劳动力调研数据
│   │   └── literature_stats.yaml # 文献统计数据
│   │
│   └── output/
│       ├── virtual_labor_pool.csv        # 虚拟劳动力池
│       ├── virtual_enterprise_pool.csv   # 虚拟企业池
│       └── calibrated_parameters.yaml    # 校准后参数
│
├── results/                       # 结果输出
│   ├── figures/                   # 图表
│   ├── reports/                   # 报告（Markdown/PDF）
│   └── logs/                      # 运行日志
│
├── tests/                         # 测试代码
│   ├── __init__.py
│   ├── conftest.py                # Pytest配置
│   │
│   ├── unit/                      # 单元测试
│   │   ├── test_population.py
│   │   ├── test_matching.py
│   │   ├── test_estimation.py
│   │   ├── test_mfg.py
│   │   └── test_calibration.py
│   │
│   ├── integration/               # 集成测试
│   │   ├── test_full_pipeline.py
│   │   └── test_module_integration.py
│   │
│   └── benchmarks/                # 性能测试
│       ├── test_numba_speedup.py
│       └── test_large_scale.py
│
└── docs/                          # 文档
    ├── userdocs/                  # 用户文档
    │   ├── user_manual.md
    │   ├── configuration_guide.md
    │   └── faq.md
    │
    ├── developerdocs/             # 开发者文档
    │   ├── architecture.md        # 本文档
    │   ├── tech_stack.md
    │   ├── roadmap.md
    │   ├── coding_standards.md
    │   └── api_reference.md
    │
    └── academicdocs/              # 学术文档
        ├── methodology.md
        ├── algorithms.md
        ├── calibration.md
        └── experiments.md
```

### 3.2 关键文件说明

#### 主程序入口

**`src/main.py`**
```python
"""
主程序入口
支持命令行参数：--config, --mode (simulation/calibration/policy)
"""
```

**`src/config_loader.py`**
```python
"""
配置加载器
- 加载YAML配置文件
- 合并默认配置与实验配置
- 环境变量注入
"""
```

#### 核心基类

**`src/core/base_generator.py`**
```python
"""
生成器基类
所有生成器（劳动力、企业）的抽象基类
定义标准接口：fit(), generate(), validate()
"""
```

**`src/core/base_simulator.py`**
```python
"""
模拟器基类
所有模拟器（匹配、MFG）的抽象基类
定义标准接口：setup(), run(), get_results()
"""
```

**`src/core/data_structures.py`**
```python
"""
核心数据结构
- Agent: 个体数据类
- MatchingPair: 匹配对
- SimulationState: 模拟状态
- MFGEquilibrium: MFG均衡
"""
```

---

## 4. 核心模块设计

### 4.1 Module 1: Population Generator

**职责**: 生成虚拟劳动力和企业主体池

#### 类图

```
BaseGenerator (抽象基类)
    ↑
    ├── LaborGenerator
    │       ├── CopulaEngine
    │       └── MarginalEstimator
    │
    └── EnterpriseGenerator
            └── MultivariateNormalEngine
```

#### 核心类设计

**`LaborGenerator`**
```python
class LaborGenerator(BaseGenerator):
    """
    劳动力生成器
    基于Copula理论生成虚拟劳动力个体
    
    Attributes:
        copula_engine: Copula引擎
        marginal_estimator: 边际分布估计器
        fitted_params: 拟合后的参数
    
    Methods:
        fit(data: pd.DataFrame) -> None
        generate(n_agents: int) -> pd.DataFrame
        validate(agents: pd.DataFrame) -> bool
    """
```

**`EnterpriseGenerator`**
```python
class EnterpriseGenerator(BaseGenerator):
    """
    企业生成器
    基于四维正态分布生成虚拟企业
    
    Attributes:
        mean_vector: 均值向量 μ ∈ ℝ⁴
        cov_matrix: 协方差矩阵 Σ ∈ ℝ⁴ˣ⁴
        calibrated: 是否已校准
    
    Methods:
        set_parameters(mean, cov) -> None  # 直接设置参数
        generate(n_enterprises: int) -> pd.DataFrame
        validate(enterprises: pd.DataFrame) -> bool
    """
```

#### 数据流

```
输入数据 (CSV) 
    → fit() → 参数估计
    → generate(N) → 虚拟个体
    → validate() → 质量检查
    → 输出到 data/output/
```

---

### 4.2 Module 2: Matching Engine

**职责**: 实现双边稳定匹配算法

#### 类图

```
BaseSimulator
    ↑
MatchingEngine
    ├── PreferenceCalculator
    └── GaleShapleyAlgorithm
```

#### 核心类设计

**`MatchingEngine`**
```python
class MatchingEngine(BaseSimulator):
    """
    匹配引擎
    实现Gale-Shapley稳定匹配算法
    
    Attributes:
        preference_calculator: 偏好计算器
        matching_algorithm: GS算法实现
    
    Methods:
        setup(laborers, enterprises, theta) -> None
        run() -> MatchingResult
        get_statistics() -> Dict
    """
```

**`PreferenceCalculator`**
```python
class PreferenceCalculator:
    """
    偏好计算器
    计算劳动力和企业的相互偏好分数
    
    Methods:
        compute_laborer_preference(laborer, enterprise) -> float
        compute_enterprise_preference(enterprise, laborer) -> float
    """
```

#### 算法复杂度

- 时间复杂度: O(n²) （n为主体数量）
- 空间复杂度: O(n²) （偏好矩阵）
- **Numba优化目标**: 将偏好矩阵计算加速10倍以上

---

### 4.3 Module 3: Match Function Estimator

**职责**: 估计匹配概率函数λ(x, σ, a, θ)

#### 流程图

```
1. ABM模拟生成训练数据
   ↓
2. 多轮次、多θ值扰动模拟
   ↓
3. Logit回归拟合
   ↓
4. 匹配函数λ输出
```

#### 核心类设计

**`MatchFunctionEstimator`**
```python
class MatchFunctionEstimator:
    """
    匹配函数估计器
    通过ABM模拟+Logit回归构建匹配函数
    
    Methods:
        generate_training_data(
            n_rounds: int,
            theta_range: List[float]
        ) -> pd.DataFrame
        
        fit_logit_model(data: pd.DataFrame) -> None
        
        predict(x, sigma, a, theta) -> float  # 返回匹配概率
    """
```

**`MatchFunction`** (Numba优化)
```python
@njit
def match_function(
    x: np.ndarray,      # 状态变量 [T, S, D, W]
    sigma: np.ndarray,  # 固定特征 [age, edu, ...]
    a: float,           # 努力水平
    theta: float        # 市场紧张度
) -> float:
    """
    匹配概率函数 λ(x, σ, a, θ)
    **强制使用Numba优化**
    
    Returns:
        匹配成功概率 ∈ [0, 1]
    """
```

---

### 4.4 Module 4: MFG Simulator

**职责**: 求解平均场博弈均衡

#### 算法框架

```
初始化: V₀, m₀, θ₀
    ↓
循环直到收敛:
    1. 贝尔曼方程 → 最优努力a*(x,t)
    2. KFE演化 → 人口分布m(x,t+1)
    3. 更新θ(t+1)
    4. 检查收敛条件
    ↓
输出: MFE均衡 (V*, a*, m*, θ*)
```

#### 核心类设计

**`MFGSimulator`**
```python
class MFGSimulator(BaseSimulator):
    """
    MFG模拟器（简化版）
    
    Attributes:
        state_space: 离散状态空间
        bellman_solver: 贝尔曼求解器
        kfe_solver: KFE求解器
        match_function: 匹配函数λ
        
    Config:
        grid_size: (50, 50)  # 状态空间网格
        max_iterations: 500
        tolerance: 1e-6
    
    Methods:
        initialize() -> None
        solve_bellman() -> ValueFunction
        evolve_kfe() -> Distribution
        check_convergence() -> bool
        get_equilibrium() -> MFGEquilibrium
    """
```

**简化策略**:
1. 状态空间离散化（连续 → 50×50网格）
2. 有限期（T=20期）而非无限期
3. 固定迭代次数上限（500次）
4. 简化的收敛判断

---

### 4.5 Module 5: Calibration

**职责**: 参数校准（遗传算法）

#### 校准流程

```
定义参数空间 (15-20维)
    ↓
初始化种群 (pop_size=100)
    ↓
遗传算法迭代:
    1. 评估适应度（运行完整模拟）
    2. 选择
    3. 交叉
    4. 变异
    ↓
输出最优参数组
```

#### 核心类设计

**`GeneticCalibrator`**
```python
class GeneticCalibrator:
    """
    遗传算法校准器
    基于DEAP库实现
    
    Attributes:
        parameter_space: 参数空间定义
        objective_function: 目标函数（加权指标）
        toolbox: DEAP工具箱
    
    Config:
        population_size: 100
        n_generations: 50
        crossover_prob: 0.7
        mutation_prob: 0.2
    
    Methods:
        define_parameter_space() -> Dict
        create_objective() -> Callable
        run_calibration() -> Tuple[parameters, fitness]
    """
```

**`ObjectiveFunction`**
```python
class ObjectiveFunction:
    """
    目标函数
    多指标加权：失业率、匹配率等
    
    Formula:
        f(θ) = Σᵢ wᵢ |sim_metricᵢ(θ) - target_metricᵢ|²
        
    其中θ是参数向量，wᵢ是权重
    """
```

**Numba优化点**: 目标函数中的模拟核心计算

---

## 5. 数据流转

### 5.1 端到端数据流

```
[用户输入] config.yaml
    ↓
[Module 1] → virtual_labor_pool.csv
           → virtual_enterprise_pool.csv
    ↓
[Module 2] → matching_data.csv
    ↓
[Module 3] → match_function_params.json
    ↓
[Module 4] → mfg_equilibrium.pkl
    ↓
[Module 5] → calibrated_parameters.yaml
    ↓
[结果输出] reports/ + figures/
```

### 5.2 核心数据格式

#### 劳动力数据格式
```python
# virtual_labor_pool.csv
columns = [
    'agent_id',        # 个体ID
    'T',               # 每周工作时长
    'S',               # 工作能力评分
    'D',               # 数字素养评分
    'W',               # 每月期望收入
    'age',             # 年龄（控制变量）
    'education',       # 教育年限
    # ... 其他控制变量
]
```

#### 企业数据格式
```python
# virtual_enterprise_pool.csv
columns = [
    'enterprise_id',   # 企业ID
    'T_req',           # 工作时长要求
    'S_req',           # 技能要求
    'D_req',           # 数字化要求
    'W_offer',         # 提供薪资
]
```

#### 匹配结果格式
```python
# matching_data.csv
columns = [
    'laborer_id',
    'enterprise_id',
    'matched',         # 0/1
    'theta',           # 市场紧张度
    'effort_level',    # 努力水平a
    # ... 劳动力和企业的属性
]
```

---

## 6. 接口设计

### 6.1 统一接口规范

所有模块遵循统一的接口模式：

```python
class Module(BaseClass):
    def __init__(self, config: Dict):
        """初始化，加载配置"""
        
    def setup(self, **kwargs) -> None:
        """准备阶段：加载数据、初始化参数"""
        
    def run(self) -> Result:
        """执行主要逻辑"""
        
    def get_results(self) -> Dict:
        """获取结果"""
        
    def save_results(self, path: Path) -> None:
        """保存结果到文件"""
        
    def validate(self) -> bool:
        """验证结果质量"""
```

### 6.2 模块间接口

#### Population → Matching
```python
def get_agents(
    labor_generator: LaborGenerator,
    enterprise_generator: EnterpriseGenerator,
    n_labor: int,
    n_enterprise: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """获取虚拟主体供匹配使用"""
```

#### Matching → Estimation
```python
def get_matching_data(
    matching_engine: MatchingEngine,
    n_rounds: int,
    theta_range: List[float]
) -> pd.DataFrame:
    """获取匹配数据供Logit回归使用"""
```

#### Estimation → MFG
```python
def get_match_function(
    estimator: MatchFunctionEstimator
) -> Callable:
    """获取匹配函数λ供MFG使用"""
```

#### MFG → Calibration
```python
def simulate_market(
    mfg_simulator: MFGSimulator,
    parameters: Dict
) -> Dict[str, float]:
    """运行完整模拟，返回宏观指标"""
```

---

## 7. 技术决策

### 7.1 Numba优化策略

#### 必须优化的函数（热点）

1. **匹配函数λ**: 每次迭代调用数百万次
   ```python
   @njit(fastmath=True, cache=True)
   def match_function(...): ...
   ```

2. **偏好矩阵计算**: O(n²)复杂度
   ```python
   @njit(parallel=True)
   def compute_preference_matrix(...): ...
   ```

3. **值迭代内循环**: MFG核心计算
   ```python
   @njit(fastmath=True)
   def bellman_iteration(...): ...
   ```

4. **KFE演化步骤**: 状态转移概率计算
   ```python
   @njit
   def kfe_step(...): ...
   ```

#### Numba使用原则

- **DO**: 纯数值计算、循环密集、数组操作
- **DON'T**: 复杂对象操作、字符串处理、I/O操作

### 7.2 配置管理

**YAML格式**，层次化结构：

```yaml
# base_config.yaml
project:
  name: "Simulation_project_v2"
  version: "2.0"
  
simulation:
  random_seed: 42
  n_labor: 10000
  n_enterprise: 5000
  
logging:
  level: "INFO"
  save_to_file: true
  log_dir: "results/logs"
```

### 7.3 测试策略

#### 测试金字塔

```
        /\
       /E2E\         ← 少量端到端测试
      /------\
     /集成测试 \       ← 中等数量集成测试
    /----------\
   /  单元测试   \     ← 大量单元测试
  /--------------\
```

#### 测试覆盖目标

- 单元测试: 核心算法 > 90%
- 集成测试: 完整流程覆盖
- 性能测试: Numba加速 > 10x

### 7.4 版本控制策略

遵循用户的**规则六**：

1. Git管理所有代码版本
2. Conventional Commits规范
   - `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
3. Tag标记重要里程碑
   - `v2.0-alpha`, `v2.0-beta`, `v2.0`
4. `Change_Log.md` 详细记录每次修改

### 7.5 文档生成

- **Sphinx**: 自动生成API文档
- **Markdown**: 用户手册和开发者指南
- **Jupyter Notebook**: 学术报告与实验结果

---

## 8. 开发优先级

### Phase 1: 基础框架（2周）
- [ ] 目录结构完成
- [ ] 核心基类实现
- [ ] 配置系统搭建
- [ ] 日志与工具函数

### Phase 2: Module 1（3周）
- [ ] Copula引擎重构
- [ ] 劳动力生成器
- [ ] 企业生成器
- [ ] 单元测试

### Phase 3: Module 2+3（4周）
- [ ] 匹配引擎
- [ ] Logit估计器
- [ ] 匹配函数
- [ ] Numba优化

### Phase 4: Module 4（6周）
- [ ] 状态空间设计
- [ ] 贝尔曼求解器
- [ ] KFE求解器
- [ ] 收敛测试

### Phase 5: Module 5（3周）
- [ ] 遗传算法实现
- [ ] 目标函数设计
- [ ] 参数空间定义
- [ ] 校准实验

### Phase 6: 集成与文档（2周）
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 文档完善
- [ ] 代码审查

---

## 9. 附录

### 9.1 关键术语

- **MFG**: Mean-Field Game，平均场博弈
- **ABM**: Agent-Based Modeling，基于主体建模
- **KFE**: Kolmogorov Forward Equation，柯尔莫哥洛夫前向方程
- **GS算法**: Gale-Shapley稳定匹配算法
- **Copula**: 连接函数，用于建模变量依赖结构

### 9.2 参考资料

- 研究计划文档: `../Simulation_project/研究计划/研究计划.md`
- 旧版代码: `../Simulation_project/`
- 需求文档: `../Simulation_project/docs/需求对齐确认文档.md`

---

**文档维护**: 本文档随项目演进持续更新  
**审阅者**: 开发团队全体成员  
**下次审阅**: 每个Phase结束后
