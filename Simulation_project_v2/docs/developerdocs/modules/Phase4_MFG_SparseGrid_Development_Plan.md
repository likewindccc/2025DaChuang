# Phase 4: MFG求解器开发计划 - 稀疏网格+Numba方法

**创建日期**: 2025-10-03  
**方法选型**: 稀疏网格 + Numba加速  
**目标模块**: `src/modules/mfg/`  
**预计工期**: 6周（可能根据实际情况调整）

---

## 📋 目录

- [1. 设计确认清单](#1-设计确认清单)
- [2. 模块结构设计](#2-模块结构设计)
- [3. 核心组件详细设计](#3-核心组件详细设计)
- [4. 与现有模块的衔接](#4-与现有模块的衔接)
- [5. 开发任务分解](#5-开发任务分解)
- [6. 验收标准](#6-验收标准)

---

## 1. 设计确认清单

**⚠️ 在开始开发前，需要与用户确认以下关键设计问题：**

### 1.1 状态空间设计

#### ✅ 已确认
- **状态空间维度**: 4维 (T, S, D, W)
- **状态变量范围**:
  - T (工作时长): [0, 168] 小时/周
  - S (工作能力): [0, 100] 分
  - D (数字素养): [0, 100] 分
  - W (期望工资): [2000, 12000] 元/月

#### ❓ 需要确认

**Q1. 稀疏网格精度级别**
```
选项A: Level 4 (约3,000个网格点)
  - 优点：计算快（~5-10秒/次MFG求解）
  - 缺点：精度一般（误差~1e-3）
  
选项B: Level 5 (约15,000个网格点)
  - 优点：精度高（误差~5e-4）
  - 缺点：计算较慢（~30-60秒/次MFG求解）
  
选项C: Level 6 (约60,000个网格点)
  - 优点：精度很高（误差~1e-4）
  - 缺点：计算慢（~2-5分钟/次MFG求解）

推荐：选项B（平衡精度和速度）
```

**您的选择**: _______________

---

**Q2. 稀疏网格库的选择**
```
选项A: 使用Tasmanian库（ORNL开发，C++内核）
  - 优点：性能极高，功能完整，文献广泛引用
  - 缺点：需要编译安装，可能在Windows上有兼容性问题
  
选项B: 使用pysgpp库（纯Python，易安装）
  - 优点：安装简单，纯Python接口
  - 缺点：性能较Tasmanian低，文档较少
  
选项C: 自己实现Smolyak稀疏网格
  - 优点：完全可控，可以发论文说明算法细节
  - 缺点：开发时间长（+2周），可能有bug
  
选项D: 初期使用规则网格（简化为2D或粗糙4D），后期再升级
  - 优点：快速验证流程
  - 缺点：需要重构

推荐：选项A（性能最优，如果安装成功）或选项B（保险选择）
```

**您的选择**: _______________

---

### 1.2 时间和动态设定

#### ❓ 需要确认

**Q3. 时间维度设定**
```
根据研究计划，MFG模型需要考虑时间动态。请选择：

选项A: 无限期模型（稳态均衡）
  - 贝尔曼方程：V(x) = max_a [u(x,a) + ρ*E[V(x')]]
  - 优点：简单，只需求解稳态
  - 缺点：无法分析动态路径
  
选项B: 有限期模型（T=20期，每期1个月）
  - 贝尔曼方程：V_t(x) = max_a [u(x,a) + ρ*E[V_{t+1}(x')]]
  - 需要：终值条件V_T(x)
  - 优点：可分析动态演化
  - 缺点：计算量大T倍
  
选项C: 有限期但期限较短（T=5期）
  - 折中方案
  
推荐：选项A（稳态均衡，符合MFG标准做法）
```

**您的选择**: _______________

**如果选择有限期，请指定**:
- 期数 T = _______________
- 终值条件 V_T(x) = _______________ （例如：V_T(x) = 0 或基于外生值）

---

### 1.3 状态转移方程

#### ❓ 需要确认

**Q4. 状态转移方程的具体形式**

根据研究计划，个体努力a会改善自身状态。请确认各维度的状态转移方程：

**方案A: 线性增长模型（上界收敛）**
```python
# 下一期状态（所有变化都是渐进式的，有上界）
T_{t+1} = T_t + γ_T × a × (T_max - T_t)  # 工作时长增加，但不超过T_max
S_{t+1} = S_t + γ_S × a × (100 - S_t)     # 技能提升，但不超过100
D_{t+1} = D_t + γ_D × a × (100 - D_t)     # 数字素养提升，但不超过100
W_{t+1} = max(W_min, W_t - γ_W × a)       # 期望工资降低（更愿意接受），但有下限

# 参数（需要校准）：
γ_T = 0.1  # 工作时长增长率（示例值）
γ_S = 0.05 # 技能增长率
γ_D = 0.08 # 数字素养增长率
γ_W = 100  # 工资期望下降速率
T_max = 168  # 最大工作时长
W_min = 2000 # 最低工资期望
```

**方案B: 非线性增长模型**
```python
T_{t+1} = T_t + γ_T × a^α_T × (T_max - T_t)
S_{t+1} = S_t + γ_S × a^α_S × (100 - S_t)
D_{t+1} = D_t + γ_D × a^α_D × (100 - D_t)
W_{t+1} = max(W_min, W_t - γ_W × a^α_W)

# α_j < 1: 边际报酬递减
# α_j > 1: 边际报酬递增
# α_j = 1: 线性（即方案A）
```

**方案C: 用户自定义**
```
请提供具体的状态转移方程：
T_{t+1} = _______________
S_{t+1} = _______________
D_{t+1} = _______________
W_{t+1} = _______________
```

**您的选择**: _______________

**如果选择方案A或B，请指定参数初值**（可以后续校准调整）:
```python
gamma_T = _______________  # 建议范围：0.05-0.2
gamma_S = _______________  # 建议范围：0.03-0.1
gamma_D = _______________  # 建议范围：0.05-0.15
gamma_W = _______________  # 建议范围：50-200
```

---

### 1.4 效用函数设计

#### ❓ 需要确认

**Q5. 失业者的瞬时效用函数**

失业者在状态x=(T,S,D,W)付出努力a后，获得的瞬时效用：

**方案A: 简单二次成本**
```python
u^U(x, a) = -κ × a^2

# κ: 努力成本系数，κ越大，努力成本越高
# 解释：失业者付出努力没有直接收益，只有成本
```

**方案B: 状态依赖的效用**
```python
u^U(x, a) = b(S) - κ × a^2

# b(S): 失业救济金，可能与技能相关
# 例如：b(S) = b_0 + b_1 × S
```

**方案C: 期望收益导向**
```python
u^U(x, a) = λ(x, σ, a, θ) × W - κ × a^2

# 乐观者效用：考虑匹配成功后的工资收益
# 但这可能导致循环依赖（λ本身依赖于a）
```

**推荐**: 方案A（理论标准，避免循环依赖）

**您的选择**: _______________

**请指定参数**:
```python
kappa = _______________  # 努力成本系数，建议范围：0.5-2.0
```

---

**Q6. 就业者的瞬时效用函数**

就业者在状态x，付出努力a'（用于保持就业或提升技能）：

**方案A: 工资收益 - 努力成本**
```python
u^E(x, a') = W - κ' × (a')^2

# W: 当前工资（即状态中的W）
# κ': 就业者的努力成本系数（可能与失业者不同）
# a': 就业者的努力水平（可能为0，或用于技能提升）
```

**方案B: 简化假设（就业者不主动努力）**
```python
u^E(x) = W

# 假设就业者a'=0，只享受工资
```

**方案C: 包含工作满意度**
```python
u^E(x, a') = W + ω(T, S, D) - κ' × (a')^2

# ω(T, S, D): 工作满意度，可能与工时、技能匹配度相关
```

**推荐**: 方案B（简化，就业者不主动努力）

**您的选择**: _______________

---

### 1.5 控制变量σ的计算

#### ❓ 需要确认

**Q7. 控制变量σ（劳动力与市场的差距）如何计算**

在`MatchFunction`中，σ是劳动力特征与市场平均特征的差距：

**方案A: 与宏观均值的差**
```python
# 市场平均特征（从人口分布m^U计算）
T_bar = ∫ T × m^U(T,S,D,W) dT dS dD dW
S_bar = ∫ S × m^U(T,S,D,W) dT dS dD dW
D_bar = ∫ D × m^U(T,S,D,W) dT dS dD dW
W_bar = ∫ W × m^U(T,S,D,W) dT dS dD dW

# 劳动力i的控制变量
σ_i = [T_i - T_bar, S_i - S_bar, D_i - D_bar, W_i - W_bar]
```

**方案B: 标准化差距**
```python
σ_i = [(T_i - T_bar)/std(T), ..., (W_i - W_bar)/std(W)]
```

**方案C: 固定参考点**
```python
# 使用初始分布或外生参考值
σ_i = [T_i - T_ref, S_i - S_ref, D_i - D_ref, W_i - W_ref]
```

**推荐**: 方案A（符合研究计划，动态反映市场状态）

**您的选择**: _______________

---

### 1.6 初始分布设定

#### ❓ 需要确认

**Q8. 初始人口分布 m^U_0(x) 和 m^E_0(x)**

MFG求解需要初始分布，请选择：

**方案A: 基于真实数据的经验分布**
```python
# 从cleaned_data.csv的300条真实数据构造经验分布
# 使用KDE或直方图映射到稀疏网格点
# 优点：真实
# 缺点：可能不平滑
```

**方案B: 拟合参数分布**
```python
# 假设初始分布为多元正态分布或Copula
# 使用之前的LaborGenerator生成
# 优点：平滑
# 缺点：可能不完全真实
```

**方案C: 均匀分布（测试用）**
```python
# 所有网格点等概率
# 优点：简单
# 缺点：不真实，仅用于算法测试
```

**推荐**: 方案B（使用LaborGenerator）

**您的选择**: _______________

**初始失业率**: u_0 = _______________  （建议：0.1-0.3）

---

### 1.7 市场紧张度θ的更新

#### ❓ 需要确认

**Q9. θ的定义和更新机制**

**θ的定义**（已确认）:
```
θ = V / U
V: 职位数（外生或内生）
U: 失业人数 = ∫ m^U(x) dx
```

**需要确认：V（职位数）如何确定？**

**方案A: 外生固定**
```python
V = V_0  # 例如：V_0 = 1000（固定职位数）
θ_t = V_0 / U_t  # U_t会随m^U变化
```

**方案B: 内生均衡（θ固定）**
```python
θ = θ_bar  # 固定θ，例如θ=1.0
V_t = θ_bar × U_t  # 职位数随失业人数调整
```

**方案C: 外生随机冲击**
```python
θ_t = θ_bar + ε_t  # ε_t ~ N(0, σ_θ^2)
```

**推荐**: 方案B（标准MFG做法，θ是均衡内生变量）

**您的选择**: _______________

**如果选择方案B，请指定初值**: θ_bar = _______________  （建议：0.7-1.3）

---

### 1.8 收敛判断标准

#### ❓ 需要确认

**Q10. 收敛判断的具体标准**

MFG迭代需要判断何时收敛，标准形式：
```python
max|V^{k+1} - V^k| < tol_V  AND
max|a^{k+1} - a^k| < tol_a  AND
|θ^{k+1} - θ^k| < tol_θ
```

**请指定容差**:
```python
tol_V = _______________  # 建议：1e-4
tol_a = _______________  # 建议：1e-4
tol_θ = _______________  # 建议：1e-3
max_iterations = _______  # 建议：500
```

---

### 1.9 努力水平离散化

#### ❓ 需要确认

**Q11. 努力水平a的取值范围和离散化**

贝尔曼方程需要枚举a来找最优值：

**方案A: a ∈ [0, 1]，离散为21个点**
```python
a_grid = np.linspace(0, 1, 21)  # [0, 0.05, 0.1, ..., 0.95, 1.0]
```

**方案B: a ∈ [0, 1]，离散为11个点（粗糙但快）**
```python
a_grid = np.linspace(0, 1, 11)
```

**方案C: a ∈ [0, a_max]，用户自定义**
```
a_min = _______________
a_max = _______________
n_grid = _______________
```

**推荐**: 方案A

**您的选择**: _______________

---

## 2. 模块结构设计

### 2.1 文件组织

```
src/modules/mfg/
├── __init__.py                 # 模块初始化
├── sparse_grid.py              # 稀疏网格核心类
├── state_space.py              # 状态空间管理
├── bellman_solver.py           # 贝尔曼方程求解器
├── kfe_solver.py               # KFE演化求解器
├── mfg_simulator.py            # MFG主循环控制器
├── utility_functions.py        # 效用函数（Numba）
├── state_transition.py         # 状态转移（Numba）
└── interpolation.py            # 插值函数（Numba）
```

### 2.2 类关系图

```
┌─────────────────────────────────────────────────┐
│         MFGSimulator (主控制器)                  │
│  - 协调各组件                                    │
│  - 执行主迭代循环                                │
│  - 收敛判断                                      │
└──────────┬──────────────────────────────────────┘
           │
           ├─────→ SparseGrid (稀疏网格)
           │       - 构造Smolyak网格
           │       - 插值查询
           │
           ├─────→ StateSpace (状态空间管理)
           │       - 状态范围定义
           │       - 初始分布设置
           │
           ├─────→ BellmanSolver (贝尔曼求解器)
           │       - 值迭代
           │       - 最优策略计算
           │       │
           │       └───→ MatchFunction (复用Module 3)
           │             - 计算匹配概率λ
           │
           └─────→ KFESolver (KFE求解器)
                   - 人口分布演化
                   - θ更新
```

### 2.3 数据结构扩展

需要扩展 `src/core/data_structures.py` 中的 `MFGEquilibrium`：

```python
@dataclass
class MFGEquilibrium:
    """
    MFG均衡结果（4维版本）
    
    Attributes:
        value_function_U: 失业状态值函数 (n_grid_points,)
        value_function_E: 就业状态值函数 (n_grid_points,)
        policy_function: 最优努力策略 (n_grid_points,)
        distribution_U: 失业人口分布 (n_grid_points,)
        distribution_E: 就业人口分布 (n_grid_points,)
        grid_points: 稀疏网格点坐标 (n_grid_points, 4)
        theta: 均衡市场紧张度
        converged: 是否收敛
        iterations: 迭代次数
        convergence_error: 收敛误差
        compute_time: 计算耗时（秒）
    """
    # ... (详见下文)
```

---

## 3. 核心组件详细设计

### 3.1 稀疏网格类 (`sparse_grid.py`)

#### 3.1.1 核心功能

```python
class SparseGrid:
    """
    Smolyak稀疏网格
    
    实现4维稀疏网格的构造、插值和查询。
    """
    
    def __init__(self, level: int, bounds: List[Tuple[float, float]]):
        """
        初始化稀疏网格
        
        Args:
            level: 精度级别（3-6）
            bounds: 各维度范围 [(T_min, T_max), (S_min, S_max), ...]
        """
        pass
    
    def get_grid_points(self) -> np.ndarray:
        """
        获取所有网格点坐标
        
        Returns:
            形状 (n_points, 4) 的数组
        """
        pass
    
    def interpolate(self, values: np.ndarray, query_point: np.ndarray) -> float:
        """
        在查询点插值
        
        Args:
            values: 网格点上的函数值 (n_points,)
            query_point: 查询点坐标 (4,)
        
        Returns:
            插值结果
        """
        pass
    
    def interpolate_batch(self, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        """
        批量插值（Numba优化）
        
        Args:
            values: (n_points,)
            query_points: (m, 4)
        
        Returns:
            插值结果 (m,)
        """
        pass
```

#### 3.1.2 实现方案

根据Q2的选择：
- **如果选择方案A/B（使用库）**：包装Tasmanian或pysgpp的接口
- **如果选择方案C（自己实现）**：实现Smolyak构造算法

---

### 3.2 贝尔曼求解器 (`bellman_solver.py`)

```python
class BellmanSolver:
    """
    贝尔曼方程求解器
    
    求解失业者的值函数和最优策略：
    V^U(x) = max_a [u(x,a) + ρ × E[λ×V^E(x') + (1-λ)×V^U(x')]]
    """
    
    def __init__(
        self,
        sparse_grid: SparseGrid,
        match_function: MatchFunction,
        config: Dict
    ):
        """
        初始化
        
        Args:
            sparse_grid: 稀疏网格对象
            match_function: 匹配函数（来自Module 3）
            config: 配置字典（包含ρ, κ, γ等参数）
        """
        self.grid = sparse_grid
        self.match_func = match_function
        self.config = config
        
        # 提取参数
        self.rho = config['rho']
        self.kappa = config['kappa']
        self.gamma_T = config['gamma_T']
        # ...
        
        # 努力水平网格
        self.a_grid = np.linspace(0, 1, config['n_effort_grid'])
    
    def solve(
        self,
        V_U_init: np.ndarray,
        V_E_init: np.ndarray,
        theta: float,
        m_U: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        值迭代法求解贝尔曼方程
        
        Args:
            V_U_init: 初始失业值函数 (n_points,)
            V_E_init: 初始就业值函数 (n_points,)
            theta: 市场紧张度
            m_U: 失业人口分布（用于计算σ）
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        Returns:
            (V_U_new, a_optimal): 值函数和最优策略
        """
        grid_points = self.grid.get_grid_points()
        n_points = len(grid_points)
        
        # 计算市场平均特征（用于σ）
        market_avg = self._compute_market_average(grid_points, m_U)
        
        V_U = V_U_init.copy()
        V_E = V_E_init.copy()
        
        for iteration in range(max_iter):
            V_U_new, a_optimal = self._bellman_iteration(
                V_U, V_E, grid_points, theta, market_avg
            )
            
            # 检查收敛
            error = np.max(np.abs(V_U_new - V_U))
            if error < tol:
                logger.info(f"贝尔曼方程收敛于第{iteration}轮")
                return V_U_new, a_optimal
            
            V_U = V_U_new
        
        logger.warning(f"贝尔曼方程未收敛，最大误差={error:.2e}")
        return V_U, a_optimal
    
    def _bellman_iteration(
        self,
        V_U: np.ndarray,
        V_E: np.ndarray,
        grid_points: np.ndarray,
        theta: float,
        market_avg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单次贝尔曼迭代（Numba加速）
        """
        return bellman_iteration_numba(
            V_U, V_E, grid_points, self.a_grid,
            theta, market_avg,
            self.grid, self.match_func,
            self.rho, self.kappa,
            self.gamma_T, self.gamma_S, self.gamma_D, self.gamma_W
        )
    
    def _compute_market_average(
        self,
        grid_points: np.ndarray,
        m_U: np.ndarray
    ) -> np.ndarray:
        """
        计算市场平均特征 (T_bar, S_bar, D_bar, W_bar)
        
        市场平均 = Σ x_i × m^U(x_i) / Σ m^U(x_i)
        """
        total_mass = np.sum(m_U)
        if total_mass < 1e-10:
            return np.mean(grid_points, axis=0)
        
        weighted_sum = np.sum(grid_points * m_U[:, None], axis=0)
        return weighted_sum / total_mass
```

#### 3.2.1 Numba核心函数

```python
@njit(parallel=True)
def bellman_iteration_numba(
    V_U: np.ndarray,
    V_E: np.ndarray,
    grid_points: np.ndarray,
    a_grid: np.ndarray,
    theta: float,
    market_avg: np.ndarray,
    grid_object,  # 稀疏网格对象（用于插值）
    match_func,   # 匹配函数对象
    rho: float,
    kappa: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    贝尔曼迭代的Numba实现
    
    对每个网格点x_i：
    1. 枚举所有a
    2. 计算x'（状态转移）
    3. 计算λ(x, σ, a, θ)
    4. 插值得到V(x')
    5. 计算u(x,a) + ρ × [λ×V^E(x') + (1-λ)×V^U(x')]
    6. 取最大值的a
    """
    n_points = len(grid_points)
    n_effort = len(a_grid)
    
    V_U_new = np.zeros(n_points)
    a_optimal = np.zeros(n_points)
    
    for i in prange(n_points):  # 并行循环
        x = grid_points[i]
        T, S, D, W = x[0], x[1], x[2], x[3]
        
        # 计算控制变量σ
        sigma = x - market_avg
        
        best_value = -np.inf
        best_a = 0.0
        
        for j in range(n_effort):
            a = a_grid[j]
            
            # 状态转移
            T_next = T + gamma_T * a * (168.0 - T)
            S_next = S + gamma_S * a * (100.0 - S)
            D_next = D + gamma_D * a * (100.0 - D)
            W_next = max(2000.0, W - gamma_W * a)
            x_next = np.array([T_next, S_next, D_next, W_next])
            
            # 匹配概率（调用match_function）
            lambda_val = match_func.compute_match_probability(
                x, sigma, a, theta
            )
            
            # 插值查询V(x')
            V_U_next = grid_object.interpolate(V_U, x_next)
            V_E_next = grid_object.interpolate(V_E, x_next)
            
            # 贝尔曼右侧
            immediate_utility = -kappa * a * a  # u(x,a) = -κa²
            continuation_value = rho * (
                lambda_val * V_E_next + (1 - lambda_val) * V_U_next
            )
            total_value = immediate_utility + continuation_value
            
            if total_value > best_value:
                best_value = total_value
                best_a = a
        
        V_U_new[i] = best_value
        a_optimal[i] = best_a
    
    return V_U_new, a_optimal
```

**⚠️ 注意**：上述代码中的`grid_object`和`match_func`调用需要根据Numba的兼容性调整：
- Numba不支持直接调用Python对象方法
- 需要将插值和匹配概率计算也写成纯Numba函数
- 具体实现见3.5节

---

### 3.3 KFE求解器 (`kfe_solver.py`)

```python
class KFESolver:
    """
    Kolmogorov Forward Equation 求解器
    
    演化人口分布：
    ∂m/∂t = -∇·[m × π(x|a*(x))] + sources - sinks
    
    离散形式：
    m^{t+1}(x) = Σ_{x'} P(x|x', a*(x')) × m^t(x')
    """
    
    def __init__(self, sparse_grid: SparseGrid, config: Dict):
        self.grid = sparse_grid
        self.config = config
        
        # 提取参数
        self.mu = config['mu']  # 外生离职率
    
    def evolve(
        self,
        m_U: np.ndarray,
        m_E: np.ndarray,
        a_optimal: np.ndarray,
        match_function: MatchFunction,
        theta: float,
        market_avg: np.ndarray,
        n_steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        演化人口分布
        
        Args:
            m_U: 当前失业分布 (n_points,)
            m_E: 当前就业分布 (n_points,)
            a_optimal: 最优策略 (n_points,)
            match_function: 匹配函数
            theta: 市场紧张度
            market_avg: 市场平均特征
            n_steps: 演化步数
        
        Returns:
            (m_U_new, m_E_new): 新的分布
        """
        grid_points = self.grid.get_grid_points()
        
        for _ in range(n_steps):
            m_U, m_E = self._kfe_step(
                m_U, m_E, a_optimal, grid_points,
                match_function, theta, market_avg
            )
        
        return m_U, m_E
    
    def _kfe_step(
        self,
        m_U: np.ndarray,
        m_E: np.ndarray,
        a_optimal: np.ndarray,
        grid_points: np.ndarray,
        match_function: MatchFunction,
        theta: float,
        market_avg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步KFE演化（Numba加速）
        """
        return kfe_step_numba(
            m_U, m_E, a_optimal, grid_points,
            match_function, theta, market_avg,
            self.grid, self.mu,
            self.config['gamma_T'], self.config['gamma_S'],
            self.config['gamma_D'], self.config['gamma_W']
        )
```

#### 3.3.1 Numba核心函数

```python
@njit(parallel=True)
def kfe_step_numba(
    m_U: np.ndarray,
    m_E: np.ndarray,
    a_optimal: np.ndarray,
    grid_points: np.ndarray,
    match_function,
    theta: float,
    market_avg: np.ndarray,
    grid_object,
    mu: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KFE演化一步
    
    逻辑：
    1. 对每个网格点x_i（有质量m^U(x_i)的失业者）
    2. 他们按a*(x_i)努力，转移到x'
    3. 以概率λ匹配成功 → 增加m^E(x')
    4. 以概率1-λ失败 → 增加m^U(x')
    5. 就业者以概率μ离职 → m^E减少，m^U增加
    """
    n_points = len(grid_points)
    
    m_U_new = np.zeros(n_points)
    m_E_new = np.zeros(n_points)
    
    # 第一遍：处理失业者的转移
    for i in range(n_points):
        x = grid_points[i]
        a = a_optimal[i]
        sigma = x - market_avg
        
        # 状态转移
        x_next = state_transition(x, a, gamma_T, gamma_S, gamma_D, gamma_W)
        
        # 匹配概率
        lambda_val = match_function.compute_match_probability(
            x, sigma, a, theta
        )
        
        # 找到x_next最近的网格点（或插值权重分配）
        weights, indices = grid_object.get_interpolation_weights(x_next)
        
        # 分配质量
        for j, w in zip(indices, weights):
            # 匹配成功 → 就业
            m_E_new[j] += lambda_val * m_U[i] * w
            # 匹配失败 → 继续失业
            m_U_new[j] += (1 - lambda_val) * m_U[i] * w
    
    # 第二遍：处理就业者的离职
    for i in range(n_points):
        # 外生离职
        m_U_new[i] += mu * m_E[i]
        # 保持就业
        m_E_new[i] += (1 - mu) * m_E[i]
    
    # 归一化（确保总质量守恒）
    total = np.sum(m_U_new) + np.sum(m_E_new)
    m_U_new /= total
    m_E_new /= total
    
    return m_U_new, m_E_new
```

---

### 3.4 MFG主控制器 (`mfg_simulator.py`)

```python
class MFGSimulator:
    """
    MFG模拟器主控制器
    
    协调贝尔曼求解器、KFE求解器，执行主迭代循环。
    """
    
    def __init__(self, config: Dict, match_function: MatchFunction):
        """
        初始化MFG模拟器
        
        Args:
            config: 配置字典（从mfg.yaml加载）
            match_function: 匹配函数（来自Module 3）
        """
        self.config = config
        self.match_func = match_function
        
        # 构造稀疏网格
        self.grid = self._build_sparse_grid()
        
        # 初始化求解器
        self.bellman_solver = BellmanSolver(
            self.grid, self.match_func, config
        )
        self.kfe_solver = KFESolver(self.grid, config)
        
        # 初始化分布
        self.m_U, self.m_E = self._initialize_distribution()
        
        # 初始化值函数
        self.V_U = np.zeros(len(self.grid.get_grid_points()))
        self.V_E = np.zeros(len(self.grid.get_grid_points()))
        
        # 初始化θ
        self.theta = config.get('theta_init', 1.0)
        
        logger.info("MFG模拟器初始化完成")
    
    def solve(
        self,
        max_iterations: int = 500,
        tol_V: float = 1e-4,
        tol_a: float = 1e-4,
        tol_theta: float = 1e-3
    ) -> MFGEquilibrium:
        """
        求解MFG均衡
        
        算法：
        1. 初始化 m^U, m^E, θ, V^U, V^E
        2. 重复：
           a. 固定(m, θ)，解贝尔曼方程 → 得到V^U, a*
           b. 固定a*，演化KFE → 得到新的m^U, m^E
           c. 更新θ = V / U
           d. 检查收敛
        3. 返回均衡结果
        
        Returns:
            MFGEquilibrium对象
        """
        import time
        start_time = time.time()
        
        logger.info("开始MFG求解...")
        
        a_optimal = None
        
        for iteration in range(max_iterations):
            # Step 1: 解贝尔曼方程
            V_U_new, a_optimal = self.bellman_solver.solve(
                self.V_U, self.V_E, self.theta, self.m_U,
                max_iter=100, tol=1e-5
            )
            
            # Step 2: 演化KFE
            market_avg = self.bellman_solver._compute_market_average(
                self.grid.get_grid_points(), self.m_U
            )
            m_U_new, m_E_new = self.kfe_solver.evolve(
                self.m_U, self.m_E, a_optimal,
                self.match_func, self.theta, market_avg,
                n_steps=1
            )
            
            # Step 3: 更新θ
            U_total = np.sum(m_U_new)
            if self.config.get('theta_fixed', True):
                # 方案B：θ固定
                theta_new = self.config.get('theta_bar', 1.0)
            else:
                # 方案A：V固定
                V_fixed = self.config.get('V_fixed', 1000.0)
                theta_new = V_fixed / max(U_total, 1e-10)
            
            # Step 4: 收敛判断
            error_V = np.max(np.abs(V_U_new - self.V_U))
            error_a = np.max(np.abs(a_optimal - (a_optimal if a_optimal is not None else 0)))
            error_theta = abs(theta_new - self.theta)
            
            converged = (
                error_V < tol_V and
                error_a < tol_a and
                error_theta < tol_theta
            )
            
            # 日志输出
            if iteration % 10 == 0 or converged:
                logger.info(
                    f"迭代 {iteration}: "
                    f"err_V={error_V:.2e}, err_a={error_a:.2e}, "
                    f"err_θ={error_theta:.2e}, θ={theta_new:.4f}"
                )
            
            if converged:
                logger.info(f"✅ MFG收敛于第{iteration}轮")
                compute_time = time.time() - start_time
                
                return MFGEquilibrium(
                    value_function_U=V_U_new,
                    value_function_E=self.V_E,
                    policy_function=a_optimal,
                    distribution_U=m_U_new,
                    distribution_E=m_E_new,
                    grid_points=self.grid.get_grid_points(),
                    theta=theta_new,
                    converged=True,
                    iterations=iteration,
                    convergence_error=max(error_V, error_a, error_theta),
                    compute_time=compute_time
                )
            
            # 更新
            self.V_U = V_U_new
            self.m_U = m_U_new
            self.m_E = m_E_new
            self.theta = theta_new
        
        # 未收敛
        logger.warning(f"⚠️ MFG未收敛，达到最大迭代次数{max_iterations}")
        compute_time = time.time() - start_time
        
        return MFGEquilibrium(
            value_function_U=self.V_U,
            value_function_E=self.V_E,
            policy_function=a_optimal,
            distribution_U=self.m_U,
            distribution_E=self.m_E,
            grid_points=self.grid.get_grid_points(),
            theta=self.theta,
            converged=False,
            iterations=max_iterations,
            convergence_error=max(error_V, error_a, error_theta),
            compute_time=compute_time
        )
    
    def _build_sparse_grid(self) -> SparseGrid:
        """构造稀疏网格"""
        bounds = [
            (0, 168),      # T
            (0, 100),      # S
            (0, 100),      # D
            (2000, 12000)  # W
        ]
        level = self.config.get('sparse_grid_level', 5)
        return SparseGrid(level, bounds)
    
    def _initialize_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """初始化人口分布"""
        # 根据Q8的选择实现
        # 方案A: 从真实数据
        # 方案B: 从LaborGenerator
        # 方案C: 均匀分布
        
        n_points = len(self.grid.get_grid_points())
        u_0 = self.config.get('initial_unemployment', 0.2)
        
        # 简化实现：均匀分布（待改进）
        m_U = np.ones(n_points) * u_0 / n_points
        m_E = np.ones(n_points) * (1 - u_0) / n_points
        
        return m_U, m_E
```

---

### 3.5 Numba兼容性处理

**问题**: Numba无法直接调用Python类的方法（如`grid.interpolate()`, `match_func.compute()`）

**解决方案**: 将所有核心计算函数提取为纯Numba函数

```python
# utility_functions.py
@njit
def utility_unemployed(a: float, kappa: float) -> float:
    """失业者效用：-κa²"""
    return -kappa * a * a

@njit
def utility_employed(W: float) -> float:
    """就业者效用：W"""
    return W


# state_transition.py
@njit
def state_transition(
    x: np.ndarray,
    a: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float
) -> np.ndarray:
    """
    状态转移函数（Numba）
    
    Args:
        x: 当前状态 [T, S, D, W]
        a: 努力水平
        gamma_*: 转移参数
    
    Returns:
        下一期状态 [T', S', D', W']
    """
    T, S, D, W = x[0], x[1], x[2], x[3]
    
    T_next = T + gamma_T * a * (168.0 - T)
    S_next = S + gamma_S * a * (100.0 - S)
    D_next = D + gamma_D * a * (100.0 - D)
    W_next = max(2000.0, W - gamma_W * a)
    
    return np.array([T_next, S_next, D_next, W_next])


# interpolation.py
@njit
def hierarchical_interpolate_4d(
    values: np.ndarray,
    grid_points: np.ndarray,
    query_point: np.ndarray,
    grid_structure: ...  # 预计算的网格结构
) -> float:
    """
    4维稀疏网格插值（Numba实现）
    
    使用分层基插值方法
    """
    # 具体实现见稀疏网格库文档
    pass
```

---

## 4. 与现有模块的衔接

### 4.1 复用Module 3的匹配函数

```python
# 在MFG模拟器初始化时
from src.modules.estimation.match_function import MatchFunction

# 加载估计好的参数
match_func = MatchFunction()
match_func.load_params('results/estimation/match_function_params.json')

# 传递给MFG模拟器
mfg_sim = MFGSimulator(config, match_func)
```

### 4.2 使用Module 1生成初始分布

```python
# 在_initialize_distribution()中
from src.modules.population.labor_generator import LaborGenerator

# 生成虚拟劳动力
labor_gen = LaborGenerator(config)
labor_data = labor_gen.generate(n=10000)

# 映射到稀疏网格
m_U_init = self._map_to_grid(labor_data, self.grid.get_grid_points())
```

### 4.3 配置文件统一

更新`config/default/mfg.yaml`：

```yaml
# Module 4: MFG Simulator 配置（4维稀疏网格版本）

sparse_grid:
  level: 5  # 精度级别（3-6）
  bounds:
    T: [0, 168]
    S: [0, 100]
    D: [0, 100]
    W: [2000, 12000]

state_transition:
  gamma_T: 0.1   # 工作时长增长率
  gamma_S: 0.05  # 技能增长率
  gamma_D: 0.08  # 数字素养增长率
  gamma_W: 100   # 工资期望下降速率

utility:
  kappa: 1.0  # 努力成本系数

solver:
  max_iterations: 500
  tolerance:
    value_function: 1.0e-4
    policy: 1.0e-4
    theta: 1.0e-3
  
  # 贝尔曼方程参数
  rho: 0.95          # 贴现因子
  n_effort_grid: 21  # 努力水平离散点数
  
  # KFE演化参数
  mu: 0.05           # 外生离职率

market:
  theta_fixed: true  # θ是否固定
  theta_bar: 1.0     # 固定θ值（如果theta_fixed=true）
  V_fixed: 1000      # 固定职位数（如果theta_fixed=false）

initial_condition:
  unemployment_rate: 0.2  # 初始失业率
  distribution_source: 'labor_generator'  # 'labor_generator' 或 'uniform'

optimization:
  use_numba: true
  parallel: true
```

---

## 5. 开发任务分解

### 5.1 Week 1: 稀疏网格基础

**任务**:
- [ ] 安装并测试稀疏网格库（Tasmanian/pysgpp/自实现）
- [ ] 实现`sparse_grid.py`基础功能
  - [ ] 网格构造
  - [ ] 基础插值
- [ ] 单元测试：插值精度验证
- [ ] 性能测试：不同level的速度对比

**交付物**:
- [ ] `sparse_grid.py` (~300-500行)
- [ ] `tests/unit/test_sparse_grid.py`
- [ ] 性能报告

### 5.2 Week 2: 效用和状态转移

**任务**:
- [ ] 实现`utility_functions.py`（Numba）
- [ ] 实现`state_transition.py`（Numba）
- [ ] 实现`interpolation.py`（Numba优化的插值）
- [ ] 单元测试
- [ ] 与MatchFunction集成测试

**交付物**:
- [ ] `utility_functions.py` (~100行)
- [ ] `state_transition.py` (~100行)
- [ ] `interpolation.py` (~200-300行)
- [ ] 测试套件

### 5.3 Week 3: 贝尔曼求解器

**任务**:
- [ ] 实现`bellman_solver.py`
- [ ] 实现Numba核心迭代函数
- [ ] 单元测试（小网格）
- [ ] 收敛性测试

**交付物**:
- [ ] `bellman_solver.py` (~400-500行)
- [ ] `tests/unit/test_bellman_solver.py`
- [ ] 收敛曲线图

### 5.4 Week 4: KFE求解器

**任务**:
- [ ] 实现`kfe_solver.py`
- [ ] 实现Numba核心演化函数
- [ ] 质量守恒验证
- [ ] 单元测试

**交付物**:
- [ ] `kfe_solver.py` (~300-400行)
- [ ] `tests/unit/test_kfe_solver.py`
- [ ] 分布演化动画

### 5.5 Week 5: MFG主控制器

**任务**:
- [ ] 实现`mfg_simulator.py`
- [ ] 集成所有组件
- [ ] 端到端测试
- [ ] 初始分布生成逻辑

**交付物**:
- [ ] `mfg_simulator.py` (~400-500行)
- [ ] `tests/integration/test_mfg_complete.py`
- [ ] 完整运行示例

### 5.6 Week 6: 优化与文档

**任务**:
- [ ] 性能优化（profiling）
- [ ] 扩展`data_structures.py`（4维MFGEquilibrium）
- [ ] 结果可视化工具
- [ ] 完整文档和示例
- [ ] Code Review

**交付物**:
- [ ] 性能优化报告
- [ ] 可视化工具
- [ ] API文档
- [ ] 使用示例

---

## 6. 验收标准

### 6.1 功能验收

- [ ] ✅ 稀疏网格构造正确（点数符合理论）
- [ ] ✅ 插值精度满足要求（误差<1e-3）
- [ ] ✅ 贝尔曼方程收敛（<200轮，误差<1e-4）
- [ ] ✅ KFE演化守恒（总质量误差<1e-6）
- [ ] ✅ MFG主循环收敛（<500轮）
- [ ] ✅ 结果合理性（θ稳定，策略单调等）

### 6.2 性能验收

- [ ] ✅ 单次MFG求解：<60秒（Level 5）
- [ ] ✅ 单次贝尔曼迭代：<0.5秒
- [ ] ✅ 单次KFE演化：<0.3秒
- [ ] ✅ Numba加速：>10x vs 纯Python

### 6.3 代码质量验收

- [ ] ✅ 测试覆盖率：>85%
- [ ] ✅ PEP8规范：100%通过
- [ ] ✅ Docstring完整：所有公开接口
- [ ] ✅ 类型注解：所有函数签名
- [ ] ✅ 日志完善：关键步骤有INFO级日志

### 6.4 文档验收

- [ ] ✅ 用户文档：如何运行MFG模拟
- [ ] ✅ 开发者文档：架构和扩展指南
- [ ] ✅ 学术文档：算法描述和参数说明
- [ ] ✅ API文档：Sphinx生成

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 稀疏网格库安装失败 | 中 | 高 | 准备备选方案（pysgpp或自实现） |
| MFG不收敛 | 高 | 高 | 调整参数，增加阻尼项，降低精度要求 |
| 计算时间过长 | 中 | 中 | 降低网格精度，优化Numba代码 |
| Numba类型错误 | 中 | 中 | 提前测试，使用`@njit`装饰器调试 |
| 插值精度不足 | 中 | 中 | 提升网格精度或改用更高阶插值 |

---

## 8. 下一步行动

**⚠️ 请用户确认以上所有设计问题（Q1-Q11）后，开始正式开发！**

**确认清单**:
- [ ] Q1: 稀疏网格精度级别
- [ ] Q2: 稀疏网格库选择
- [ ] Q3: 时间维度设定
- [ ] Q4: 状态转移方程
- [ ] Q5: 失业者效用函数
- [ ] Q6: 就业者效用函数
- [ ] Q7: 控制变量σ计算方式
- [ ] Q8: 初始分布设定
- [ ] Q9: 市场紧张度θ更新
- [ ] Q10: 收敛判断标准
- [ ] Q11: 努力水平离散化

**确认后即可启动开发！** 🚀

---

**文档版本**: v1.0  
**创建日期**: 2025-10-03  
**状态**: 待用户审阅确认

