# Phase 4: MFG求解器 - 最终参数确认

**创建日期**: 2025-10-03  
**状态**: ⏳ 待用户最终确认少数具体数值

---

## ✅ 已确认的参数设定

根据用户反馈（2025-10-03），以下参数已确认：

### 1. 状态空间（原始范围，不扩展）

```yaml
state_space:
  T_range: [15, 70]      # 每周工作小时数（原始数据范围）
  S_range: [2, 44]       # 工作能力评分（原始数据范围）
  D_range: [0, 20]       # 数字素养评分（原始数据范围）
  W_range: [1400, 8000]  # 每月期望收入（原始数据范围）
  
  # 标准化方式：在稀疏网格中使用标准化值
  # S_norm = (S - 2) / (44 - 2) = (S - 2) / 42
  # D_norm = (D - 0) / (20 - 0) = D / 20
  # T和W保持原始尺度
```

### 2. 状态转移参数

```yaml
state_transition:
  gamma_T: 0.1          # 工作时长增长率（待校准）
  gamma_S: 0.05         # 技能增长率（待校准）
  gamma_D: 0.08         # 数字素养增长率（待校准）
  gamma_W: 100          # 工资期望下降速率（待校准）
  
  T_max: 70             # 最大工作时长 = 数据中的最大值
  W_min: 1400           # 最低工资期望 = 数据中的最小值
  
  # 标准化转移公式（S和D在[0,1]尺度）：
  # S_norm_{t+1} = S_norm_t + gamma_S * a * (1 - S_norm_t)
  # D_norm_{t+1} = D_norm_t + gamma_D * a * (1 - D_norm_t)
```

### 3. 效用函数参数

```yaml
utility:
  # 失业者瞬时效用: u^U = b_0 - 0.5 * kappa * a^2
  unemployment:
    b_0: ???            # ⚠️ 需要确认：常数失业补助（建议：500元/月）
    kappa: 1.0          # 努力成本系数（待校准）
  
  # 就业者瞬时效用: u^E = W - alpha_T * T
  employment:
    type: 'wage_minus_disutility'  # 方案B
    alpha_T: ???        # ⚠️ 需要确认：工作负效用系数（建议：10）
    # 经济学解释: 每多工作1小时/周，效用减少alpha_T
```

### 4. 离职率

```yaml
separation:
  type: 'constant'
  mu: 0.05              # 常数离职率：每期5%的外生离职概率
```

### 5. 控制变量σ

```yaml
control_variables:
  type: 'absorbed_in_intercept'  # 方案C：简化处理
  # σ的影响被吸收到匹配函数的截距项δ_0中
  # λ(x, a, θ) = 1 / (1 + exp[-(δ_0' + δ_x'x + δ_a*a + δ_θ*ln(θ))])
```

### 6. 贝尔曼方程参数

```yaml
bellman:
  rho: 0.9              # 贴现因子（用户指定）
  n_effort_grid: 21     # 努力水平离散点数: a ∈ [0, 1]，21个点
```

### 7. 收敛标准

```yaml
convergence:
  epsilon_V: 1.0e-4     # 价值函数收敛容差
  epsilon_a: 1.0e-4     # 努力水平收敛容差
  epsilon_u: 1.0e-3     # 失业率收敛容差
  max_iterations: 500   # 最大迭代次数
```

### 8. 市场紧张度

```yaml
market:
  theta_mechanism: 'fixed_theta'  # 方案A：θ固定
  theta_bar: ???        # ⚠️ 需要确认：固定的θ值（建议：1.0）
  # V_t = theta_bar * U_t （职位数随失业人数调整）
```

### 9. 稀疏网格设置

```yaml
sparse_grid:
  library: 'tasmanian'  # 使用Tasmanian库
  level: 5              # 精度级别5（约15,000个网格点）
  dimension: 4          # 4维状态空间 (T, S_norm, D_norm, W)
```

### 10. 初始分布

```yaml
initial_distribution:
  source: 'copula_from_real_data'  # 基于300条真实数据的Copula
  unemployment_rate: 0.2           # 初始失业率20%
  # 使用Module 1的LaborGenerator生成初始劳动力分布
```

---

## ⚠️ 需要您最终确认的具体数值

### Q1. 失业补助 b_0

失业者的瞬时效用为：`u^U = b_0 - 0.5 * kappa * a^2`

**b_0 的经济学含义**: 失业时的基本生活补助、家庭生产价值、社会救济等

**建议值**: 
- `b_0 = 500` 元/月（约为平均期望工资4520的11%）
- 或 `b_0 = 0`（简化为纯努力成本模型）

**您的选择**: b_0 = _______ 元/月

---

### Q2. 工作负效用系数 α_T

就业者的瞬时效用为：`u^E = W - alpha_T * T`

**α_T 的经济学含义**: 每多工作1小时/周带来的负效用（疲劳、时间机会成本等）

**建议值**:
- `α_T = 10`：工作70小时的负效用 = 10×70 = 700元
  - 净效用 = W - 700（例如：W=4500时，净效用=3800）
- `α_T = 20`：更高的工作负效用
- `α_T = 0`：简化为纯工资收益（就退化为方案A）

**您的选择**: α_T = _______ 元/(小时/周)

---

### Q3. 固定的市场紧张度 θ_bar

您选择了方案A（θ固定），需要指定θ_bar的值。

**θ_bar 的经济学含义**: 
- θ = V/U（职位数/失业人数）
- θ = 1.0：岗位数等于失业人数（均衡市场）
- θ > 1.0：岗位富余（对求职者有利）
- θ < 1.0：岗位紧张（对求职者不利）

**建议值**: `θ_bar = 1.0`（均衡市场）

**您的选择**: θ_bar = _______

---

### Q4. 努力成本系数 κ（可选确认）

您说"先默认确定，后续校准"，我理解为接受建议值。

**当前建议值**: `κ = 1.0`

**如果需要修改**: κ = _______ （留空则使用1.0）

---

## 📊 参数经济学含义示例

假设一个失业劳动力状态为：
- T = 45小时/周
- S_norm = 0.5（标准化后）
- D_norm = 0.4
- W = 4500元/月

### 失业时（a=0.5的努力）

```
瞬时效用 = b_0 - 0.5 * κ * a^2
         = 500 - 0.5 * 1.0 * 0.25
         = 500 - 0.125
         = 499.875 元/月
```

### 就业时

```
瞬时效用 = W - α_T * T
         = 4500 - 10 * 45
         = 4500 - 450
         = 4050 元/月
```

**就业收益增量** = 4050 - 499.875 ≈ 3550 元/月

---

## 📝 完整参数配置文件预览

确认上述4个数值后，我将生成以下配置文件：

```yaml
# config/default/mfg.yaml (最终版本)

# 状态空间定义
state_space:
  dimension: 4
  ranges:
    T: [15, 70]          # 每周工作小时
    S: [2, 44]           # 工作能力（原始分数）
    D: [0, 20]           # 数字素养（原始分数）
    W: [1400, 8000]      # 每月期望收入
  
  # 标准化（仅S和D）
  standardization:
    S: 
      method: 'minmax'
      min: 2
      max: 44
    D:
      method: 'minmax'
      min: 0
      max: 20

# 稀疏网格
sparse_grid:
  library: 'tasmanian'
  level: 5
  dimension: 4

# 状态转移
state_transition:
  gamma_T: 0.1
  gamma_S: 0.05
  gamma_D: 0.08
  gamma_W: 100
  T_max: 70
  W_min: 1400

# 效用函数
utility:
  unemployment:
    b_0: ???          # 待确认
    kappa: 1.0
  employment:
    alpha_T: ???      # 待确认

# 贝尔曼方程
bellman:
  rho: 0.9
  n_effort_grid: 21

# KFE演化
kfe:
  mu: 0.05            # 离职率

# 市场
market:
  theta_bar: ???      # 待确认

# 收敛
convergence:
  epsilon_V: 1.0e-4
  epsilon_a: 1.0e-4
  epsilon_u: 1.0e-3
  max_iterations: 500

# 初始化
initialization:
  unemployment_rate: 0.2
  distribution_source: 'copula'
```

---

## 🎯 总结：只需确认4个数值

请您确认以下4个数值，然后我将立即开始开发：

1. **b_0** (失业补助) = _______ 元/月 （建议：500 或 0）
2. **α_T** (工作负效用) = _______ 元/(小时/周) （建议：10）
3. **θ_bar** (固定市场紧张度) = _______ （建议：1.0）
4. **κ** (努力成本) = _______ （建议：1.0，可留空）

确认后，我将：
1. ✅ 更新`config/default/mfg.yaml`
2. ✅ 安装Tasmanian库
3. ✅ 开始实现8个核心模块
4. ✅ 编写测试代码
5. ✅ 进行功能和性能验收

---

**⏳ 等待您的最终确认...**

