# Phase 3: Matching + Estimation 模块开发计划

**开发阶段**: Phase 3  
**预计时间**: 4周（Week 6-9）  
**当前状态**: 准备开始  
**日期**: 2025-10-01

---

## 📋 目录

- [1. 模块概述](#1-模块概述)
- [2. 技术架构](#2-技术架构)
- [3. 开发任务](#3-开发任务)
- [4. 详细设计](#4-详细设计)
- [5. 性能目标](#5-性能目标)
- [6. 测试计划](#6-测试计划)

---

## 1. 模块概述

### 1.1 模块定位

**Matching模块（Module 2）**：
- 功能：实现劳动力市场双边匹配算法
- 核心：Gale-Shapley稳定匹配算法
- 输入：虚拟劳动力 + 虚拟企业
- 输出：稳定匹配结果

**Estimation模块（Module 3）**：
- 功能：基于匹配数据估计匹配函数
- 核心：Logit回归 + 参数估计
- 输入：ABM模拟数据
- 输出：匹配函数λ(x, σ, a, θ)

### 1.2 模块关系

```
Population模块
    ↓
    ├── LaborGenerator → 虚拟劳动力
    └── EnterpriseGenerator → 虚拟企业
            ↓
        Matching模块
            ├── 偏好计算
            ├── Gale-Shapley算法
            └── 匹配结果
                ↓
            ABM数据生成
                ↓
        Estimation模块
            ├── Logit回归
            └── 匹配函数参数
                ↓
            MFG模块（Phase 4）
```

---

## 2. 技术架构

### 2.1 核心组件

```python
src/modules/matching/
├── __init__.py
├── preference.py              # 偏好计算
├── gale_shapley.py           # GS算法
├── matching_engine.py        # 匹配引擎集成
└── matching_result.py        # 结果数据结构

src/modules/estimation/
├── __init__.py
├── abm_data_generator.py     # ABM数据生成
├── logit_estimator.py        # Logit回归
└── match_function.py         # 匹配函数（Numba优化）
```

### 2.2 依赖关系

**外部依赖**：
- `numba`: 性能优化（核心）
- `statsmodels`: Logit回归
- `scipy`: 统计函数

**内部依赖**：
- `src.core`: 基础类
- `src.modules.population`: 数据生成

---

## 3. 开发任务

### Week 6: Gale-Shapley算法 ⏳

**任务清单**：
- [ ] 实现偏好计算模块（`preference.py`）
  - [ ] 劳动力偏好函数（效用最大化）
  - [ ] 企业偏好函数（生产力最大化）
  - [ ] Numba优化版本
- [ ] 实现GS算法（`gale_shapley.py`）
  - [ ] 经典DA算法
  - [ ] 稳定性验证
- [ ] 单元测试
- [ ] 性能基准测试

**关键指标**：
- 偏好矩阵计算（10K×5K）< 500ms
- GS算法收敛 < 30秒

### Week 7: 匹配引擎集成 ⏳

**任务清单**：
- [ ] 实现匹配结果数据结构（`matching_result.py`）
- [ ] 集成匹配引擎（`matching_engine.py`）
- [ ] 批量模拟功能
- [ ] 配置文件（`config/default/matching.yaml`）
- [ ] 集成测试

**交付物**：
- 完整的匹配引擎
- 可进行批量模拟

### Week 8: ABM数据生成 ⏳

**任务清单**：
- [ ] 实现ABM数据生成器（`abm_data_generator.py`）
  - [ ] 多轮次模拟
  - [ ] θ值扰动策略
  - [ ] 努力水平a扰动
- [ ] 生成训练数据集（~100K样本）
- [ ] 数据质量验证

**扰动策略**：
```python
theta_range = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # 7个值
effort_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]     # 6个值
n_rounds_per_combination = 5

total_simulations = 7 × 6 × 5 = 210轮
```

### Week 9: Logit回归与匹配函数 ⏳

**任务清单**：
- [ ] 实现Logit估计器（`logit_estimator.py`）
  - [ ] Statsmodels集成
  - [ ] 参数估计
  - [ ] 模型诊断
- [ ] 实现匹配函数（`match_function.py`）
  - [ ] Numba优化版本
  - [ ] 参数加载
- [ ] 配置文件（`config/default/estimation.yaml`）
- [ ] 估计报告生成

---

## 4. 详细设计

### 4.1 偏好计算

#### 4.1.1 劳动力偏好

**基于原始研究计划的偏好函数**：

劳动力 $i$ 对企业 $j$ 的偏好函数为：

$$
P_{ij}^{jobseeker} = \gamma_0 - \gamma_1 T_j - \gamma_2 \max(0, S_j - S_i) - \gamma_3 \max(0, D_j - D_i) + \gamma_4 W_j
$$

**经济学含义**：
- $\gamma_0$: 截距项（基准偏好）
- $-\gamma_1 T_j$: **工作时长越长，劳动力偏好越低**（寻求较短工作时间）
- $-\gamma_2 \max(0, S_j - S_i)$: **岗位技能要求超过自身水平时的负面影响**（体现"能力不足"的规避倾向）
- $-\gamma_3 \max(0, D_j - D_i)$: **岗位数字素养要求超过自身时的负面影响**
- $+\gamma_4 W_j$: **工资越高，偏好越高**

**关键特性**：
- 使用 $\max(0, \cdot)$ 体现**不对称性**：只有当岗位要求超过自身能力时才产生负面影响
- 当自身能力高于岗位要求时（$S_i > S_j$），该项为0，不影响偏好
- 这种设计符合现实：劳动力会规避"高攀"的岗位，但不会因为"屈就"而降低偏好

**代码实现**：
```python
@njit(parallel=True, fastmath=True)
def compute_labor_preference_matrix(
    labor_features: np.ndarray,        # (n_labor, 4): [T, S, D, W]
    enterprise_features: np.ndarray,   # (n_enterprise, 4): [T, S, D, W]
    gamma_0: float = 1.0,             # 截距项
    gamma_1: float = 0.01,            # 工作时长系数
    gamma_2: float = 0.5,             # 技能要求超过惩罚系数
    gamma_3: float = 0.5,             # 数字素养要求超过惩罚系数
    gamma_4: float = 0.001            # 工资权重
) -> np.ndarray:                      # (n_labor, n_enterprise)
    """
    计算劳动力对所有企业的偏好矩阵（基于原始研究计划公式）
    
    P_ij^jobseeker = γ₀ - γ₁Tⱼ - γ₂max(0, Sⱼ-Sᵢ) - γ₃max(0, Dⱼ-Dᵢ) + γ₄Wⱼ
    
    返回：偏好分数矩阵（越高越好）
    """
    n_labor = labor_features.shape[0]
    n_enterprise = enterprise_features.shape[0]
    
    preference = np.zeros((n_labor, n_enterprise), dtype=np.float32)
    
    for i in prange(n_labor):
        labor_T, labor_S, labor_D, labor_W = labor_features[i]
        
        for j in range(n_enterprise):
            ent_T, ent_S, ent_D, ent_W = enterprise_features[j]
            
            # 基准偏好
            score = gamma_0
            
            # 工作时长负面影响（时长越长越不喜欢）
            score -= gamma_1 * ent_T
            
            # 技能要求超过自身水平的惩罚
            skill_gap = max(0.0, ent_S - labor_S)
            score -= gamma_2 * skill_gap
            
            # 数字素养要求超过自身的惩罚
            digital_gap = max(0.0, ent_D - labor_D)
            score -= gamma_3 * digital_gap
            
            # 工资正面影响
            score += gamma_4 * ent_W
            
            preference[i, j] = score
    
    return preference
```

#### 4.1.2 企业偏好

**基于原始研究计划的偏好函数**：

企业 $j$ 对劳动力 $i$ 的偏好函数为：

$$
P_{ji}^{employer} = \beta_0 + \beta_1 T_i + \beta_2 S_i + \beta_3 D_i + \beta_4 W_i
$$

**经济学含义**：
- $\beta_0$: 截距项（基准偏好）
- $+\beta_1 T_i$: **可供工作时间越长，企业越偏好**（工作时长充足）
- $+\beta_2 S_i$: **技能水平越高，企业越偏好**（能力强）
- $+\beta_3 D_i$: **数字素养越高，企业越偏好**（适应数字化）
- $+\beta_4 W_i$: **期望工资的影响**（**β₄为负数**：期望工资越高，企业越不喜欢，符合"降本增效"目标）

**关键特性**：
- **简单线性加权模型**，参数意义明确
- 企业偏好"能力强、时间长、要价低"的劳动力（多快好省）
- 不考虑企业与劳动力的匹配度，只看劳动力的绝对水平
- **β₄通常取负值**（如-0.001），体现企业成本控制意识

**代码实现**：
```python
@njit(parallel=True, fastmath=True)
def compute_enterprise_preference_matrix(
    enterprise_features: np.ndarray,   # (n_enterprise, 4): [T, S, D, W]
    labor_features: np.ndarray,        # (n_labor, 4): [T, S, D, W]
    beta_0: float = 0.0,              # 截距项
    beta_1: float = 0.5,              # 工作时间权重（正数）
    beta_2: float = 1.0,              # 技能水平权重（正数）
    beta_3: float = 1.0,              # 数字素养权重（正数）
    beta_4: float = -0.001            # 期望工资权重（负数：降本增效）
) -> np.ndarray:                      # (n_enterprise, n_labor)
    """
    计算企业对所有劳动力的偏好矩阵（基于原始研究计划公式）
    
    P_ji^employer = β₀ + β₁Tᵢ + β₂Sᵢ + β₃Dᵢ + β₄Wᵢ
    
    其中β₄为负数，体现企业的成本控制意识：
    - 期望工资越高 → β₄Wᵢ越负 → 企业偏好越低
    - 期望工资越低 → β₄Wᵢ接近0 → 企业偏好越高
    
    返回：偏好分数矩阵（越高越好）
    """
    n_enterprise = enterprise_features.shape[0]
    n_labor = labor_features.shape[0]
    
    preference = np.zeros((n_enterprise, n_labor), dtype=np.float32)
    
    for j in prange(n_enterprise):
        # 注：企业特征在此公式中不直接使用，但保留参数以保持接口一致
        
        for i in range(n_labor):
            labor_T, labor_S, labor_D, labor_W = labor_features[i]
            
            # 简单线性加权
            score = beta_0
            score += beta_1 * labor_T      # 工作时间越长越好
            score += beta_2 * labor_S      # 技能水平越高越好
            score += beta_3 * labor_D      # 数字素养越高越好
            score += beta_4 * labor_W      # 期望工资（β₄<0，工资越高越不好）
            
            preference[j, i] = score
    
    return preference
```

---

### 4.2 Gale-Shapley算法

#### 4.2.1 算法伪代码

```
输入: labor_pref[n_labor, n_enterprise], enterprise_pref[n_enterprise, n_labor]
输出: matching[n_labor] (每个劳动力的匹配企业ID, -1表示未匹配)

初始化:
    所有劳动力为"自由"状态
    所有企业职位空缺
    
While 存在自由劳动力且其仍有企业未申请:
    选择一个自由劳动力 i
    j = i的下一个偏好企业（按偏好排序）
    
    If 企业j职位空缺:
        匹配(i, j)
    Else if j当前匹配的劳动力k的优先级 < i的优先级（在企业j的偏好中）:
        解除匹配(k, j)
        k变为自由
        匹配(i, j)
    Else:
        i继续为自由，尝试下一个企业
        
返回 matching
```

#### 4.2.2 代码实现

```python
@njit
def gale_shapley(
    labor_pref_order: np.ndarray,      # (n_labor, n_enterprise) 偏好排序索引
    enterprise_pref_order: np.ndarray  # (n_enterprise, n_labor)
) -> np.ndarray:                       # (n_labor,) 匹配结果
    """
    Gale-Shapley延迟接受算法（劳动力提议）
    
    返回：每个劳动力匹配的企业ID（-1表示未匹配）
    """
    n_labor = labor_pref_order.shape[0]
    n_enterprise = enterprise_pref_order.shape[0]
    
    # 初始化
    matching = np.full(n_labor, -1, dtype=np.int32)       # 劳动力→企业
    reverse_matching = np.full(n_enterprise, -1, dtype=np.int32)  # 企业→劳动力
    next_proposal = np.zeros(n_labor, dtype=np.int32)    # 每个劳动力下一个提议的企业索引
    
    # 自由劳动力队列
    free_labor = list(range(n_labor))
    
    while len(free_labor) > 0:
        i = free_labor.pop(0)
        
        # 检查i是否已申请完所有企业
        if next_proposal[i] >= n_enterprise:
            continue  # i无法匹配，保持未匹配状态
        
        # i向其偏好列表中的下一个企业j提议
        j = labor_pref_order[i, next_proposal[i]]
        next_proposal[i] += 1
        
        if reverse_matching[j] == -1:
            # 企业j职位空缺，直接匹配
            matching[i] = j
            reverse_matching[j] = i
        else:
            # 企业j已有匹配的劳动力k
            k = reverse_matching[j]
            
            # 找到i和k在企业j偏好中的排名
            rank_i = np.where(enterprise_pref_order[j] == i)[0][0]
            rank_k = np.where(enterprise_pref_order[j] == k)[0][0]
            
            if rank_i < rank_k:
                # 企业j更偏好i，解除与k的匹配
                matching[k] = -1
                free_labor.append(k)
                
                matching[i] = j
                reverse_matching[j] = i
            else:
                # 企业j拒绝i，i继续为自由
                free_labor.append(i)
    
    return matching


def verify_stability(
    matching: np.ndarray,
    labor_pref_order: np.ndarray,
    enterprise_pref_order: np.ndarray
) -> Tuple[bool, List]:
    """
    验证匹配的稳定性
    
    返回：(是否稳定, 不稳定匹配对列表)
    """
    unstable_pairs = []
    n_labor = len(matching)
    n_enterprise = enterprise_pref_order.shape[0]
    
    # 构建反向匹配
    reverse_matching = {j: None for j in range(n_enterprise)}
    for i, j in enumerate(matching):
        if j != -1:
            reverse_matching[j] = i
    
    # 检查每对(i, j)
    for i in range(n_labor):
        for j in range(n_enterprise):
            current_match_i = matching[i]
            current_match_j = reverse_matching[j]
            
            # i和j是否互相更偏好对方？
            if current_match_i != j:
                # i是否更偏好j而不是当前匹配？
                if current_match_i == -1:
                    i_prefers_j = True
                else:
                    rank_j = np.where(labor_pref_order[i] == j)[0][0]
                    rank_current = np.where(labor_pref_order[i] == current_match_i)[0][0]
                    i_prefers_j = (rank_j < rank_current)
                
                # j是否更偏好i而不是当前匹配？
                if current_match_j is None:
                    j_prefers_i = True
                else:
                    rank_i = np.where(enterprise_pref_order[j] == i)[0][0]
                    rank_current = np.where(enterprise_pref_order[j] == current_match_j)[0][0]
                    j_prefers_i = (rank_i < rank_current)
                
                if i_prefers_j and j_prefers_i:
                    unstable_pairs.append((i, j))
    
    return (len(unstable_pairs) == 0), unstable_pairs
```

---

### 4.3 匹配引擎

```python
class MatchingEngine:
    """
    匹配引擎：集成偏好计算和GS算法
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.preference_params = config.get('preference', {})
    
    def match(
        self,
        labor_agents: pd.DataFrame,
        enterprise_agents: pd.DataFrame
    ) -> MatchingResult:
        """
        执行单轮匹配
        
        Args:
            labor_agents: 劳动力DataFrame
            enterprise_agents: 企业DataFrame
            
        Returns:
            MatchingResult对象
        """
        # Step 1: 提取特征
        labor_features = labor_agents[['T', 'S', 'D', 'W']].values
        enterprise_features = enterprise_agents[['T', 'S', 'D', 'W']].values
        
        # Step 2: 计算偏好矩阵
        labor_pref = compute_labor_preference_matrix(
            labor_features,
            enterprise_features,
            **self.preference_params.get('labor', {})
        )
        
        enterprise_pref = compute_enterprise_preference_matrix(
            enterprise_features,
            labor_features,
            **self.preference_params.get('enterprise', {})
        )
        
        # Step 3: 转换为偏好排序
        labor_pref_order = np.argsort(-labor_pref, axis=1)  # 降序
        enterprise_pref_order = np.argsort(-enterprise_pref, axis=1)
        
        # Step 4: 执行GS算法
        matching = gale_shapley(labor_pref_order, enterprise_pref_order)
        
        # Step 5: 验证稳定性
        is_stable, unstable_pairs = verify_stability(
            matching,
            labor_pref_order,
            enterprise_pref_order
        )
        
        # Step 6: 构造结果对象
        result = MatchingResult(
            labor_agents=labor_agents,
            enterprise_agents=enterprise_agents,
            matching=matching,
            labor_preference=labor_pref,
            enterprise_preference=enterprise_pref,
            is_stable=is_stable,
            unstable_pairs=unstable_pairs
        )
        
        return result
```

---

## 5. 性能目标

### 5.1 Numba优化目标

| 操作 | 规模 | 目标时间 | 加速比 |
|------|------|---------|--------|
| 偏好矩阵计算 | 10K × 5K | < 500ms | > 50x |
| GS算法 | 10K × 5K | < 30s | > 10x |
| 单轮完整匹配 | 10K × 5K | < 60s | - |

### 5.2 优化策略

1. **并行计算**：`@njit(parallel=True)` for偏好矩阵
2. **快速数学**：`fastmath=True` 允许近似计算
3. **数据类型优化**：使用`float32`而非`float64`
4. **缓存编译**：`cache=True` 避免重复编译

---

## 6. 测试计划

### 6.1 单元测试

- [ ] `test_preference.py` - 偏好计算
- [ ] `test_gale_shapley.py` - GS算法正确性
- [ ] `test_matching_engine.py` - 集成测试

### 6.2 性能测试

- [ ] `benchmark_preference.py` - 偏好矩阵性能
- [ ] `benchmark_gs.py` - GS算法性能

### 6.3 验证测试

- [ ] 稳定性验证100%通过
- [ ] 小规模案例（10×10）手工验证

---

## 7. 配置文件示例

```yaml
# config/default/matching.yaml

preference:
  labor:
    # 劳动力偏好函数参数：P_ij = γ₀ - γ₁Tⱼ - γ₂max(0,Sⱼ-Sᵢ) - γ₃max(0,Dⱼ-Dᵢ) + γ₄Wⱼ
    gamma_0: 1.0            # 截距项（基准偏好）
    gamma_1: 0.01           # 工作时长负面系数
    gamma_2: 0.5            # 技能要求超过惩罚系数
    gamma_3: 0.5            # 数字素养要求超过惩罚系数
    gamma_4: 0.001          # 工资正面权重
  
  enterprise:
    # 企业偏好函数参数：P_ji = β₀ + β₁Tᵢ + β₂Sᵢ + β₃Dᵢ + β₄Wᵢ
    beta_0: 0.0             # 截距项（基准偏好）
    beta_1: 0.5             # 工作时间权重
    beta_2: 1.0             # 技能水平权重
    beta_3: 1.0             # 数字素养权重
    beta_4: -0.001          # 期望工资权重（负数：工资越高越不喜欢）

algorithm:
  method: "gale_shapley"
  max_iterations: 10000     # 最大迭代次数（安全限制）
  proposer: "labor"         # 提议方：劳动力提议

performance:
  use_numba: true
  parallel: true
  fastmath: true
  cache: true

output:
  save_preference_matrix: false
  save_matching_details: true
  log_level: "INFO"
```

---

## 8. 时间计划

| Week | 任务 | 工时 | 交付物 |
|------|------|------|--------|
| Week 6 | GS算法 | 40h | preference.py, gale_shapley.py |
| Week 7 | 引擎集成 | 40h | matching_engine.py, 测试通过 |
| Week 8 | ABM生成 | 40h | abm_data_generator.py, 100K数据 |
| Week 9 | Logit估计 | 40h | logit_estimator.py, match_function.py |

---

## 9. 版本历史

### v1.1 (2025-10-01)
- ✅ **重要修正**：偏好函数严格对齐原始研究计划
- ✅ 劳动力偏好改为：$P_{ij}^{jobseeker} = \gamma_0 - \gamma_1 T_j - \gamma_2 \max(0, S_j - S_i) - \gamma_3 \max(0, D_j - D_i) + \gamma_4 W_j$
- ✅ 企业偏好改为：$P_{ji}^{employer} = \beta_0 + \beta_1 T_i + \beta_2 S_i + \beta_3 D_i + \beta_4 W_i$
- ✅ 工作时长处理改为线性负面影响（$-\gamma_1 T_j$）
- ✅ 技能/数字素养使用 $\max(0, \cdot)$ 体现不对称性
- ✅ 更新配置文件参数

### v1.0 (2025-10-01)
- 初始版本创建

---

**文档版本**: v1.1  
**最后修订**: 2025-10-01  
**责任人**: AI Assistant  
**审阅状态**: 已修订，待用户确认

