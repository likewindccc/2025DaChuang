# Jacobian敏感性分析方案

**文档版本**：1.2  
**创建日期**：2025-07-01  
**更新日期**：2026-03-03  
**关联文档**：`01_校准方法论总览.md`, `02_目标矩选择与数据来源.md`
**实现状态**：已在 `MODULES/CALIBRATION/smm_calibrator.py` 中落地

---

## 1. 方法论背景

### 1.1 理论基础

Jacobian敏感性分析基于Andrews, Gentzkow & Shapiro (2017) 提出的框架。其核心思想是：通过计算模型模拟矩对结构参数的局部导数（Jacobian矩阵），量化每个参数与每个矩之间的关联强度，从而：

1. 识别哪些参数可以被哪些矩所识别
2. 发现不可识别或弱识别的参数
3. 指导目标矩的选择和参数分类决策

### 1.2 数学定义（理论层）

设模型的模拟矩函数为 $m(\theta): \mathbb{R}^p \to \mathbb{R}^k$，其中 $\theta$ 是 $p$ 维参数向量，$m$ 是 $k$ 维矩向量。

**Jacobian矩阵** $J$ 是 $k \times p$ 的矩阵，其元素为：

$$J_{ij} = \frac{\partial m_i(\theta)}{\partial \theta_j}$$

**敏感性矩阵** $\Lambda$ 是 $p \times k$ 的矩阵，定义为：

$$\Lambda = (J'WJ)^{-1}J'W$$

其中 $W$ 是权重矩阵。$\Lambda_{ji}$ 表示参数 $\theta_j$ 的估计值对矩 $m_i$ 的敏感程度。  
说明：该定义属于方法论目标，当前代码版本尚未直接输出 $\Lambda$。

---

## 2. 数值计算方案

### 2.1 计算流程

```text
输入：基准参数向量 θ₀，微扰比例 relative_step（默认0.03），参数边界，权重矩阵 W

Step 1: 基准运行
    以 θ₀ 运行完整MFG均衡求解
    记录基准矩向量 m(θ₀)

Step 2: 微扰运行（2p次）
    对每个参数 θⱼ（j = 1, ..., p）:
        step = relative_step × (upperⱼ - lowerⱼ)
        θ⁺ = θ₀, θ⁺ⱼ += step     # 正向微扰（并裁剪到边界）
        θ⁻ = θ₀, θ⁻ⱼ -= step     # 负向微扰（并裁剪到边界）
        运行MFG均衡求解，得 m(θ⁺) 和 m(θ⁻)
    
Step 3: 构建Jacobian矩阵
    J[i,j] = (m_i(θ⁺) - m_i(θ⁻)) / (θ⁺ⱼ - θ⁻ⱼ)    # 中心差分

Step 4: 计算弹性矩阵与摘要指标
    E[i,j] = J[i,j] * (θ_j / m_i^*)
    输出 max_abs_elasticity / mean_abs_elasticity

Step 5: 生成诊断报告
    - Jacobian热力图（可选脚本）
    - mean_abs_elasticity elbow图
    - 参数可识别性评估
    - 推荐的参数分类方案

输出：Jacobian矩阵 J、弹性矩阵 E、参数摘要表与诊断图
```

### 2.2 参数与矩的配置（当前代码）

**参数向量**（当前7个待校准参数）：

| 索引 | 参数 | 基准值 | 示例步长（relative_step=0.03） |
| --- | --- | --- | --- |
| 0 | $\rho$ | 0.40 | 0.009 |
| 1 | $\kappa$ | 2000.0 | 90.0 |
| 2 | $\alpha_T$ | 0.30 | 0.015 |
| 3 | $\gamma_T$ | 0.30 | 0.009 |
| 4 | $\gamma_S$ | 0.45 | 0.012 |
| 5 | $\gamma_D$ | 0.45 | 0.012 |
| 6 | $\gamma_W$ | 0.15 | 0.006 |

**矩向量**（可从模型输出中计算的所有矩）：

| 索引 | 矩名称 | 当前口径 |
| --- | --- | --- |
| 0 | unemployment_rate | 失业个体数 / 总个体数 |
| 1 | mean_wage | 就业者工资均值 |
| 2 | std_wage | 就业者工资标准差 |
| 3 | mean_weekly_hours | 就业者平均工时 |
| 4 | job_finding_rate | 失业→就业期望转移率 |
| 5 | separation_rate | 就业→失业期望转移率 |
| 6 | wage_iqr_ratio | 工资P75/P25 |
| 7 | std_weekly_hours | 工时标准差 |

### 2.3 计算成本估计

| 步骤 | MFG求解次数 | 预计单次时间 | 总时间 |
| --- | --- | --- | --- |
| 基准运行 | 1 | ~30秒（N=10000） | 30秒 |
| 微扰运行 | 2 x 7 = 14 | ~30秒/次 | ~7分钟 |
| **合计** | **15次** | - | **~8分钟** |

注：以上估计基于AutoDL服务器（32核CPU），实际时间可能因收敛速度不同而有波动。由于各微扰运行之间相互独立，可以并行化执行，理论上可将时间压缩到约1-2分钟。

---

## 3. 结果解读指南

### 3.1 Jacobian矩阵的解读

Jacobian矩阵 $J$ 的热力图展示每个矩对每个参数的响应强度。

**解读规则**：

- **某列（参数）全为零或接近零**：该参数对所有矩都不敏感 → 不可识别 → 应外部校准
- **某行（矩）全为零或接近零**：该矩对所有参数都不敏感 → 无识别能力 → 不应纳入目标矩
- **某列只有一个大值**：该参数主要由一个矩来识别 → 参数-矩对应关系清晰
- **两列模式相似**：两个参数对矩的影响模式相似 → 两个参数之间存在共线性 → 可能需要额外矩来区分

### 3.2 归一化Jacobian

由于参数和矩的量纲不同，原始Jacobian的绝对值不具可比性。应使用**弹性矩阵**（elasticity matrix）：

$$E_{ij} = \frac{\partial m_i / m_i^*}{\partial \theta_j / \theta_j^0} = J_{ij} \cdot \frac{\theta_j^0}{m_i^*}$$

$E_{ij}$ 表示参数 $\theta_j$ 变化1%时，矩 $m_i$ 变化的百分比。这消除了量纲影响，使不同参数-矩对之间的比较有意义。

### 3.3 参数分类决策标准

基于弹性矩阵 $E$，按以下标准分类：

| 条件 | 分类 | 处理方式 |
| --- | --- | --- |
| $\max_i \|E_{ij}\| < 0.01$ | 不可识别 | 外部校准（取文献值） |
| $0.01 \leq \max_i \|E_{ij}\| < 0.1$ | 弱识别 | 优先外部校准，或增加更敏感的矩 |
| $\max_i \|E_{ij}\| \geq 0.1$ | 可识别 | 纳入SMM内部校准 |

阈值0.01和0.1为经验值，可根据实际结果调整。

---

## 4. 预期结果分析

### 4.1 基于模型结构的理论预期

根据模型的数学结构，可以先验地推断各参数的敏感性模式：

| 参数 | 预期敏感矩 | 预期不敏感矩 | 理由 |
| --- | --- | --- | --- |
| $\rho$ | 弱影响所有矩 | - | 贴现率影响所有前瞻性决策，但边际影响通常较小 |
| $\kappa$ | unemployment_rate, mean_effort | mean_S, mean_D | 努力成本直接决定最优努力 → 就业概率 |
| $\alpha_T$ | mean_T, std_T | mean_S, mean_D | 工时负效用只影响T维度的决策 |
| $\gamma_T$ | mean_T, std_T | mean_wage, mean_S | T更新速率控制工时的动态调整 |
| $\gamma_S$ | mean_wage, std_wage, wage分位数 | mean_T | S影响工资和匹配质量 |
| $\gamma_D$ | mean_D | mean_T | D主要影响数字素养相关的匹配 |
| $\gamma_W$ | mean_wage, std_wage | mean_T, mean_S | W更新速率直接控制工资动态 |

### 4.2 可能出现的问题

1. **$\gamma_S$ 和 $\gamma_W$ 共线性**：两者都影响工资分布，可能难以区分。解决方案：增加工龄-工资梯度矩（S影响工资增长率，W影响工资水平调整速度）
2. **$\gamma_D$ 弱识别**：没有直接的"数字素养"数据矩。解决方案：使用代理变量（如互联网使用率）或固定为文献值
3. **$\rho$ 全局弱识别**：贴现率在静态均衡中通常难以识别。解决方案：外部校准

---

## 5. 与现有敏感性分析代码的关系

### 5.1 当前实现位置

Jacobian流程已集成在校准主流程中：

- 实现函数：`SMMCalibrator._run_jacobian_analysis`
- 触发入口：`SMMCalibrator.calibrate`（Step0）
- 结果文件：`OUTPUT/calibration/jacobian_analysis.csv`、`jacobian_matrix.npy`、`jacobian_elasticity.npy`
- 配套可视化：`plot_jacobian_mean_elbow.py`（输出 elbow 图）

### 5.2 与 `TESTS/sensitivity_analysis.py` 的区别

| 维度 | 现有敏感性分析 | Jacobian敏感性分析 |
| --- | --- | --- |
| 目的 | 探索参数对模型输出的影响 | 系统量化参数-矩的识别关系 |
| 方法 | 单参数大范围扫描 | 基准值附近小幅微扰（中心差分） |
| 输出 | 参数-输出的趋势图 | Jacobian矩阵 $J$、弹性矩阵 $E$、弹性摘要（max/mean） |
| 用途 | 理解模型行为 | 指导参数分类和目标矩选择 |

### 5.3 复用策略

现有敏感性分析的代码框架（参数微扰、模型运行、结果记录）可以复用。需要扩展的部分：

1. 改为小幅双向微扰（中心差分）而非大范围扫描
2. 已支持8个目标矩（M1-M8），后续可按需要扩展至M9+用于稳健性分析
3. 现已实现Jacobian矩阵与弹性矩阵计算
4. 增加热力图/elbow图等可视化输出

### 5.4 当前实现边界（需与规划区分）

1. 当前代码未直接计算并输出敏感性矩阵 $\Lambda=(J'WJ)^{-1}J'W$。
2. 参数分类当前主要依赖 `max_abs_elasticity` 阈值和 override 配置。
3. 若后续需要论文级识别透明度分析，建议补充 $\Lambda$ 及其可视化。

---

## 6. 参考文献

[1] Andrews, I., Gentzkow, M., & Shapiro, J. M. (2017). "Measuring the Sensitivity of Parameter Estimates to Estimation Moments." *Quarterly Journal of Economics*, 132(4), 1553-1592.

[2] Andrews, I., Gentzkow, M., & Shapiro, J. M. (2020). "Transparency in Structural Research." *Journal of Business & Economic Statistics*, 38(4), 711-722.

[3] Kahn, R. J., & Whited, T. M. (2017). "Identification with Models and Exogenous Data Variation." *Working Paper*, University of Michigan.

[4] Kolesar, M. (2018). "Sensitivity Analysis using Approximate Moment Condition Models." *Working Paper*, Princeton University.

---

*本文档说明了Jacobian敏感性分析的方法论与当前实现口径。该模块已可在校准流程中直接运行。*
