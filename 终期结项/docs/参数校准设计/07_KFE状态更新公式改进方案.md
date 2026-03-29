# KFE状态变量更新公式改进方案

**文档版本**：1.0  
**创建日期**：2026-03-05  
**关联代码**：`Simulation_project_v3/MODULES/MFG/kfe_solver.py` → `simulate_population_evolution()`

---

## 1. 问题诊断

当前`kfe_solver.py`中`simulate_population_evolution`函数的状态更新存在三个结构性缺陷，导致均衡处截面方差趋近于零（M3/M7/M8模拟值≈0）：

| 缺陷 | 具体表现 | 后果 |
|------|---------|------|
| 就业者状态冻结 | `T_new=T, S_new=S, D_new=D, W_new=W` | 无在职学习，工资一旦确定永不变化 |
| 失业者确定性收敛 | 所有公式均为不含随机项的收缩映射 | 所有失业者状态趋向同一吸引子 |
| 匹配工资无噪声 | `current_wage = W`（直接赋值） | 入职工资完全由W决定，无岗位异质性 |

**根本原因**：异质性Agent模型在均衡处需要**个体特质冲击**（idiosyncratic shocks）来维持截面分布的非退化性（Achdou et al., 2022），当前模型缺少此机制。

---

## 2. 改进方案

### 2.1 失业者更新 —— 加入随机冲击 + 修正方向

#### T（工时状态）：保持现有方向，加噪声

$$T_{t+1} = T_t + \gamma_T \cdot a \cdot (T_{\max} - T_t) + \sigma_T \cdot \varepsilon_T$$

经济直觉：失业者通过求职努力提高可接受工时，个体间存在差异。

#### S（技能）：失业期间折旧（Ljungqvist & Sargent, 1998）

$$\tilde{S}_{t+1} = \tilde{S}_t - \delta_S \cdot (1 - a) \cdot \tilde{S}_t + \sigma_S \cdot \varepsilon_S$$

> 其中$\tilde{S}$为MinMax标准化后的技能值。现有公式中失业时技能增长，与文献不符——失业者因脱离工作岗位技能衰减，积极求职（如参加培训）可减缓衰减。

#### D（数字素养）：与S类似，失业期间折旧

$$\tilde{D}_{t+1} = \tilde{D}_t - \delta_D \cdot (1 - a) \cdot \tilde{D}_t + \sigma_D \cdot \varepsilon_D$$

数字技能更新换代快，脱离工作环境后折旧率可能高于一般技能。

#### W（期望工资）：改为均值回复过程

$$W_{t+1} = W_t + \gamma_W \cdot (W_{\text{fair}} - W_t) + \sigma_W \cdot \varepsilon_W$$

$$W_{\text{fair}} = \beta_0 + \beta_S \cdot S_t + \beta_D \cdot D_t$$

> 现有公式`W_new = W - γ_W · a`为单调递减，必然收敛到$W_{\min}$。改为向个体人力资本对应的公允工资水平回复，更符合保留工资理论。

### 2.2 就业者更新 —— 引入在职演化

```
T_new = T + γ_T_emp · (T_target - T) + σ_T_emp · ε        # 工时微调
S_new = S + γ_S_emp · (S_max - S)   + σ_S_emp · ε        # 在职学习（Ben-Porath, 1967）
D_new = D + γ_D_emp · (D_max - D)   + σ_D_emp · ε        # 数字技能积累
W_new = W + γ_W_emp · (wage - W)    + σ_W_emp · ε        # 跟随实际工资调整预期
wage_new = wage · (1 + g_wage)       + σ_wage  · ε        # 在职工资增长
```

### 2.3 匹配工资 —— 引入岗位异质性（Burdett & Mortensen, 1998）

$$w_{\text{match}} = W_{\text{new}} \cdot (1 + \sigma_{\text{match}} \cdot \varepsilon)$$

同一个人匹配不同岗位时工资不同，取决于匹配质量。**这是产生工资分散的最直接来源**。

---

## 3. 参数分类与校准策略

> **核心原则**：参数多≠需要估计的多。新增参数按可识别性分类处理，不需要增加目标矩。

### 3.1 纳入SMM估计（新增1-2个内部参数）

| 参数 | 被哪个目标矩识别 | 说明 |
|------|----------------|------|
| `σ_match` | M3（工资标准差）、M7（IQR比） | 匹配工资噪声直接决定工资分散度 |
| `σ_T`（可选） | M8（工时标准差） | 工时噪声决定工时分散度 |

这些参数正好被此前"拟合不上"的M3/M7/M8所识别。

### 3.2 文献赋值（外部固定）

| 参数 | 来源 | 建议初始值 |
|------|------|-----------|
| `δ_S` | Ljungqvist & Sargent (1998) | 0.03/月 |
| `δ_D` | 参照δ_S略高 | 0.05/月 |
| `g_wage` | CLDS面板年工资增长率÷12 | 0.002/月 |
| `γ_S_emp`, `γ_D_emp` | 设为对应γ的10-20% | 0.01-0.02 |
| `γ_T_emp`, `γ_W_emp` | 对应γ的较小比例 | 0.05, 0.10 |

### 3.3 数据标定（按固定比例设置）

| 参数 | 标定方式 | 说明 |
|------|---------|------|
| `σ_S`, `σ_D` | 对应变量截面标准差的5% | 不单独估计 |
| `σ_W`, `σ_wage` | 均值工资的3-5% | 异质性Agent文献常规做法 |
| `σ_T_emp`, `σ_S_emp`, `σ_D_emp`, `σ_W_emp` | 设为失业者对应σ的50% | 就业者噪声通常较小 |

### 3.4 校准结构对比

```
改进前：4内部 + 3外部 = 7参数，8目标矩 → M3/M7/M8无法匹配（浪费）
改进后：5~6内部 + (3+N)外部，8目标矩 → 全部有效利用，仍满足过度识别
```

---

## 4. 实施路径

### 第一步：匹配工资加噪声（最小改动，立即见效）

仅修改`simulate_population_evolution`中匹配成功的工资赋值（1行→2行），验证M3/M7/M8是否出现非零值。

### 第二步：就业者状态演化

引入在职学习和工资增长，解除就业者状态冻结。

### 第三步：失业者方向修正 + 全部噪声项

修正S/D折旧方向，W改为均值回复，加入所有噪声项。需同步修改`bellman_solver.py`中对应的状态转移预期。

### 代码改动范围

- `MODULES/MFG/kfe_solver.py`：修改`simulate_population_evolution`（约60行）
- `MODULES/MFG/kfe_solver.py`：`KFESolver.__init__`读取新参数
- `MODULES/MFG/bellman_solver.py`：状态转移预期需与KFE保持一致（第三步）
- `CONFIG/mfg_config.yaml`：新增噪声参数配置节

---

## 5. 注意事项

1. 加入噪声后MFG均衡求解带有模拟误差，可能需要增大人口规模N或多次运行取平均
2. 第三步修改失业者S/D方向时，需同步更新`bellman_solver.py`中的状态转移方程，否则Bellman期望与KFE实际演化不一致
3. 公允工资函数$W_{\text{fair}}(S,D)$的系数可从CLDS Mincer回归获得
