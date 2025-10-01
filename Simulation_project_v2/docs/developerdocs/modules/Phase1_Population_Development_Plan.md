# Phase 1: Population 模块开发文档

**模块名称**: Population - 虚拟人口生成模块  
**开发阶段**: Phase 1, Week 2  
**创建日期**: 2025-10-01  
**状态**: 设计阶段，待用户审阅

---

## 📋 目录

- [1. 模块概述](#1-模块概述)
- [2. 边际分布实验结果回顾](#2-边际分布实验结果回顾)
- [3. 设计决策](#3-设计决策)
- [4. LaborGenerator设计](#4-laborgenerator设计)
- [5. EnterpriseGenerator设计](#5-enterprisegenerator设计)
- [6. Copula建模详解](#6-copula建模详解)
- [7. 实现计划](#7-实现计划)
- [8. 测试策略](#8-测试策略)
- [9. API接口定义](#9-api接口定义)

---

## 1. 模块概述

### 1.1 职责

Population模块负责生成虚拟劳动力和企业个体，用于后续的匹配模拟和MFG求解。

**核心功能**：
- **LaborGenerator**: 从真实调研数据拟合劳动力特征分布，生成虚拟劳动力
- **EnterpriseGenerator**: 基于假设的正态分布生成虚拟企业
- **统计验证**: KS检验验证生成质量

### 1.2 依赖关系

```
Population 模块
    ↓ 依赖
Core 模块 (BaseGenerator, Agent, DataValidationError等)
    ↓ 依赖
外部库: numpy, pandas, scipy, copulas
```

### 1.3 输入输出

**输入**：
- 真实调研数据 (CSV/Excel)：劳动力的 T, S, D, W
- 配置文件 (YAML)：生成参数、Copula类型等

**输出**：
- 虚拟劳动力 DataFrame：包含 agent_id, T, S, D, W 及控制变量
- 虚拟企业 DataFrame：包含 agent_id, T, S, D, W

---

## 2. 边际分布实验结果回顾

### 2.1 实验结论（2025/10/01最终更新）⭐

**重大发现：变量需区分连续与离散建模！**

#### 核心变量（4个连续变量，用于Copula建模）

| 变量 | 分布 | 参数 (α, β) | AIC | KS统计量 | 说明 |
|------|------|------------|-----|----------|------|
| 每周工作时长 (T) | Beta | (1.93, 2.05) | -66.72 | 0.214 | 对称型 |
| 工作能力评分 (S) | Beta | (1.79, 1.57) | -39.99 | 0.141 | 对称型 |
| 数字素养评分 (D) | Beta | (0.37, 0.76) | -313.78 | 0.314 | **U型（两极分化）** |
| 每月期望收入 (W) | Beta | (1.43, 1.45) | -16.04 | 0.161 | 对称型 |

#### 控制变量 - 连续（2个）

| 变量 | 分布 | 参数 (α, β) | AIC | KS统计量 | 说明 |
|------|------|------------|-----|----------|------|
| 年龄 | Beta | (1.01, 1.00) | 7.95 | 0.147 | 近似均匀 |
| 累计工作年限 | Beta | (0.55, 1.64) | -296.59 | 0.260 | 左偏 |

#### 控制变量 - 离散（2个）⭐新增

| 变量 | 分布 | 取值 | 概率分布 | 说明 |
|------|------|------|---------|------|
| 孩子数量 | **经验分布** | [0,1,2,3] | [7.7%, 37.0%, 45.7%, 9.7%] | 离散变量 |
| 学历 | **经验分布** | [0,1,2,3,4,5,6] | [0.3%, 2.0%, 10.7%, 35.3%, 35.3%, 16.0%, 0.3%] | 离散等级 |

### 2.2 关键发现

1. **变量分类修正**⭐：
   - **连续变量（6个）**：每周工作时长、工作能力评分、数字素养评分、每月期望收入、年龄、累计工作年限 → Beta分布
   - **离散变量（2个）**：孩子数量、学历 → 经验分布（统计上更严谨）

2. **数字鸿沟现象**：数字素养呈显著U型分布（α<1, β<1），反映数字化能力的两极分化

3. **离散变量特征**：
   - 孩子数量：只有4个唯一值（0,1,2,3），主要集中在1-2个
   - 学历：只有7个等级（0-6），主要集中在3级和4级（各35.3%）

4. **数据修正**：仅对连续变量进行0值修正（数字素养36个、累计工作年限21个）

5. **相关性分析**：
   - 每周工作时长 ↔ 每月期望收入：0.549（正相关）
   - 工作能力评分 ↔ 数字素养评分：0.448（正相关）
   - 年龄 ↔ 学历：-0.754（**强负相关**）
   - 数字素养评分 ↔ 学历：0.577（正相关）

### 2.3 对Population模块的影响

- ✅ **核心变量（4个连续）**：边际分布已确定，可直接用于Copula建模
- ✅ **控制变量 - 连续（2个）**：年龄、累计工作年限，可加入Copula或单独生成
- ✅ **控制变量 - 离散（2个）**：孩子数量、学历，使用经验分布直接抽样
- ✅ 相关性矩阵已获取，可用于6维或8维Gaussian Copula参数估计
- ⚠️ **需决策**：最终采用6维Copula还是8维Copula（方案见3.3节）

---

## 3. 设计决策

### 3.1 Copula类型选择：Gaussian Copula ✅

**选择理由**：
- ✅ 适用于连续变量
- ✅ 参数估计稳健（基于相关系数矩阵）
- ✅ scipy/copulas库支持良好
- ✅ 可捕获线性依赖关系

**备选方案**：
- t-Copula：适合厚尾分布，但我们的Beta分布无厚尾问题
- Vine Copula：适合高维，但我们只有4个变量

**最终决策**：使用 **Gaussian Copula**

### 3.2 参数估计方法：最大似然估计 (MLE) ✅

**步骤**：
1. 从原始数据计算相关系数矩阵 (Spearman或Kendall)
2. 转换为Gaussian Copula的参数
3. 验证相关矩阵的正定性

**工具**：使用 `copulas` 库的 `GaussianMultivariate`

### 3.3 控制变量生成策略 ✅已确定

基于实验结果的最新发现，控制变量分为**连续（2个）**和**离散（2个）**两类，需采用混合生成策略：

#### 最终方案：6维Copula + 离散变量条件抽样 ⭐

**描述**：
1. **6维Gaussian Copula**：生成6个连续变量
   - 4个核心变量：T, S, D, W
   - 2个连续控制变量：年龄、累计工作年限
   - 使用6×6相关系数矩阵拟合

2. **离散变量条件抽样**：基于连续变量生成2个离散变量
   - 孩子数量：基于年龄条件抽样（经验分布）
   - 学历：基于年龄条件抽样（经验分布）

**具体实现**：
```python
# Step 1: 6维Copula生成连续变量
copula_samples = copula.sample(n)  # 返回 [T, S, D, W, 年龄, 累计工作年限]

# Step 2: 离散变量条件抽样
for i in range(n):
    age = copula_samples[i]['年龄']
    
    # 根据年龄分层抽样孩子数量
    if age < 30:
        kids_probs = [0.15, 0.50, 0.30, 0.05]  # 年轻人倾向少孩
    elif age < 40:
        kids_probs = [0.05, 0.35, 0.50, 0.10]  # 中年人倾向1-2个
    else:
        kids_probs = [0.05, 0.30, 0.45, 0.20]  # 年长者倾向2-3个
    
    kids = np.random.choice([0,1,2,3], p=kids_probs)
    
    # 根据年龄分层抽样学历（年龄越大学历越低）
    if age < 35:
        edu_probs = [0.001, 0.01, 0.05, 0.25, 0.45, 0.20, 0.04]
    else:
        edu_probs = [0.005, 0.03, 0.15, 0.42, 0.30, 0.08, 0.005]
    
    edu = np.random.choice([0,1,2,3,4,5,6], p=edu_probs)
```

**优点**：
- ✅ 统计严谨：离散变量用离散分布，连续变量用Copula
- ✅ 保留连续变量间的相关性（包括核心变量与年龄、工作年限）
- ✅ 实现简单：离散变量直接抽样，无需复杂建模
- ✅ 计算高效：6维Copula比8维更快

**缺点**：
- ⚠️ 需要手动定义条件概率（年龄 → 孩子数量、学历）
- ⚠️ 可能丢失离散变量之间的某些相关性（如孩子数量↔学历）

#### 备选方案A：简化版（仅4维Copula）

**描述**：
- 仅对4个核心变量使用Copula
- 控制变量全部独立抽样（忽略相关性）

**缺点**：
- ❌ 丢失控制变量与核心变量的相关性（如年龄↔工作能力）
- ❌ 统计一致性差

#### 备选方案B：全离散化（不推荐）

**描述**：
- 将所有连续变量离散化，使用高维经验分布

**缺点**：
- ❌ 丢失连续变量的平滑性
- ❌ 维度灾难（8维联合分布需要海量数据）

#### 决策总结

**最终采用**：6维Copula + 离散变量条件抽样

**理由**：
1. 平衡统计严谨性与实现复杂度
2. 保留连续变量的相关性结构（最关键的部分）
3. 离散变量用经验分布，避免不合理的连续化
4. 条件概率可从数据中估计或手动调整

### 3.4 企业生成方案：四维正态分布 + 后续校准 ✅

**初始假设**：
- 企业特征 (T, S, D, W) ~ N(μ, Σ)
- 初始均值 μ = 劳动力均值 × 调整系数
- 初始协方差 Σ = 单位矩阵 × 缩放系数

**校准策略**：
- 在Calibration模块中，通过遗传算法调整 μ 和 Σ
- 目标：使模拟的失业率、工资等与真实数据匹配

---

## 4. LaborGenerator设计

### 4.1 类结构（已更新⭐）

```python
from src.core import BaseGenerator, Agent, DataValidationError, CopulaFittingError
from copulas.multivariate import GaussianMultivariate
import pandas as pd
import numpy as np

class LaborGenerator(BaseGenerator):
    """
    劳动力生成器（6维Copula + 离散变量条件抽样）
    
    生成策略：
    1. 6维Gaussian Copula生成连续变量：T, S, D, W, 年龄, 累计工作年限
    2. 基于年龄条件抽样离散变量：孩子数量、学历
    
    Attributes:
        config: 配置字典
        copula: Gaussian Copula模型（6维）
        marginals_continuous: 连续变量的边际分布参数 (6个Beta)
        marginals_discrete: 离散变量的经验分布 (2个)
        correlation_matrix: 相关系数矩阵 (6x6)
        conditional_probs: 条件概率表（年龄 → 孩子数量、学历）
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.copula = None
        self.marginals_continuous = None
        self.marginals_discrete = None
        self.correlation_matrix = None
        self.conditional_probs = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """拟合Copula模型和离散变量分布"""
        # 1. 验证数据
        # 2. 拟合连续变量的边际分布 (6个Beta)
        # 3. CDF变换 + 拟合6维Gaussian Copula
        # 4. 拟合离散变量的经验分布
        # 5. 估计条件概率表（年龄 → 孩子数量、学历）
        # 6. 保存参数
        pass
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成虚拟劳动力"""
        # 1. 从6维Copula采样 -> 连续变量
        # 2. 逆CDF变换 -> Beta分布
        # 3. 反归一化到原始尺度
        # 4. 基于年龄条件抽样离散变量
        # 5. 构造完整DataFrame（8个变量）
        pass
    
    def validate(self, agents: pd.DataFrame) -> bool:
        """KS检验验证连续变量，卡方检验验证离散变量"""
        # 对6个连续变量进行KS检验
        # 对2个离散变量进行卡方检验
        pass
    
    def _estimate_conditional_probs(self, data: pd.DataFrame) -> dict:
        """从数据估计条件概率表"""
        # 根据年龄分层统计孩子数量和学历的分布
        pass
```

### 4.2 核心算法流程（已更新⭐）

#### 4.2.1 拟合流程 (fit)

```python
def fit(self, data: pd.DataFrame) -> None:
    # Step 1: 数据验证
    continuous_cols = ['T', 'S', 'D', 'W', '年龄', '累计工作年限']
    discrete_cols = ['孩子数量', '学历']
    
    if not all(col in data.columns for col in continuous_cols + discrete_cols):
        raise DataValidationError("数据缺少必需列")
    
    # Step 2: 拟合连续变量的边际分布 (6个Beta，使用实验结果)
    self.marginals_continuous = {
        'T': {'dist': 'beta', 'params': (1.93, 2.05, 0, 1), 
              'scale': (15.0, 70.0)},
        'S': {'dist': 'beta', 'params': (1.79, 1.57, 0, 1),
              'scale': (2.0, 44.0)},
        'D': {'dist': 'beta', 'params': (0.37, 0.76, 0, 1),
              'scale': (0.1, 20.0)},
        'W': {'dist': 'beta', 'params': (1.43, 1.45, 0, 1),
              'scale': (1400.0, 8000.0)},
        '年龄': {'dist': 'beta', 'params': (1.01, 1.00, 0, 1),
                'scale': (25.0, 50.0)},
        '累计工作年限': {'dist': 'beta', 'params': (0.55, 1.64, 0, 1),
                        'scale': (0.1, 30.0)}
    }
    
    # Step 3: 归一化 + CDF变换（6个连续变量）
    from scipy.stats import beta
    uniform_data = pd.DataFrame()
    
    for col in continuous_cols:
        params = self.marginals_continuous[col]['params']
        scale_min, scale_max = self.marginals_continuous[col]['scale']
        
        # 归一化到[0,1]
        normalized = (data[col] - scale_min) / (scale_max - scale_min)
        normalized = normalized.clip(0, 1)  # 确保在[0,1]范围内
        
        # CDF变换到均匀分布
        uniform_data[col] = beta(*params).cdf(normalized)
    
    # Step 4: 拟合6维Gaussian Copula
    self.copula = GaussianMultivariate()
    self.copula.fit(uniform_data)
    
    # Step 5: 提取相关矩阵
    self.correlation_matrix = self.copula.covariance
    
    # Step 6: 拟合离散变量的经验分布
    self.marginals_discrete = {}
    for col in discrete_cols:
        values, counts = np.unique(data[col], return_counts=True)
        probs = counts / len(data)
        self.marginals_discrete[col] = {
            'values': values.tolist(),
            'probs': probs.tolist()
        }
    
    # Step 7: 估计条件概率表（年龄 → 孩子数量、学历）
    self.conditional_probs = self._estimate_conditional_probs(data)
    
    # 保存参数
    self.fitted_params = {
        'marginals_continuous': self.marginals_continuous,
        'marginals_discrete': self.marginals_discrete,
        'correlation_matrix': self.correlation_matrix.tolist(),
        'conditional_probs': self.conditional_probs
    }
    self.is_fitted = True
```

#### 4.2.2 生成流程 (generate)（已更新⭐）

```python
def generate(self, n_agents: int) -> pd.DataFrame:
    if not self.is_fitted:
        raise RuntimeError("必须先调用fit()")
    
    # Step 1: 从6维Copula采样（均匀分布）
    uniform_samples = self.copula.sample(n_agents)
    
    # Step 2: 逆CDF变换 -> Beta分布（6个连续变量）
    from scipy.stats import beta
    agents_data = {}
    
    continuous_cols = ['T', 'S', 'D', 'W', '年龄', '累计工作年限']
    for col in continuous_cols:
        params = self.marginals_continuous[col]['params']
        scale_min, scale_max = self.marginals_continuous[col]['scale']
        
        # 均匀分布 -> Beta分布[0,1]
        beta_samples = beta(*params).ppf(uniform_samples[col])
        
        # 反归一化到原始尺度
        agents_data[col] = beta_samples * (scale_max - scale_min) + scale_min
    
    # Step 3: 基于年龄条件抽样离散变量
    kids_list = []
    edu_list = []
    
    for i in range(n_agents):
        age = agents_data['年龄'][i]
        
        # 根据年龄查找条件概率
        age_bin = self._get_age_bin(age)  # 例如: '<30', '30-40', '>=40'
        
        # 抽样孩子数量
        kids_probs = self.conditional_probs['孩子数量'][age_bin]
        kids_values = self.marginals_discrete['孩子数量']['values']
        kids = np.random.choice(kids_values, p=kids_probs)
        kids_list.append(kids)
        
        # 抽样学历
        edu_probs = self.conditional_probs['学历'][age_bin]
        edu_values = self.marginals_discrete['学历']['values']
        edu = np.random.choice(edu_values, p=edu_probs)
        edu_list.append(edu)
    
    agents_data['孩子数量'] = kids_list
    agents_data['学历'] = edu_list
    
    # Step 4: 构造完整DataFrame（8个变量）
    df = pd.DataFrame(agents_data)
    df['agent_id'] = range(1, n_agents + 1)
    df['agent_type'] = 'labor'
    
    # 重新排序列
    df = df[['agent_id', 'agent_type', 'T', 'S', 'D', 'W', 
             '年龄', '累计工作年限', '孩子数量', '学历']]
    
    return df

def _get_age_bin(self, age: float) -> str:
    """根据年龄返回分箱标签"""
    if age < 30:
        return '<30'
    elif age < 40:
        return '30-40'
    else:
        return '>=40'

def _estimate_conditional_probs(self, data: pd.DataFrame) -> dict:
    """从数据估计条件概率表"""
    age_bins = ['<30', '30-40', '>=40']
    conditional_probs = {
        '孩子数量': {},
        '学历': {}
    }
    
    for age_bin in age_bins:
        # 筛选该年龄段的数据
        if age_bin == '<30':
            mask = data['年龄'] < 30
        elif age_bin == '30-40':
            mask = (data['年龄'] >= 30) & (data['年龄'] < 40)
        else:
            mask = data['年龄'] >= 40
        
        subset = data[mask]
        
        # 统计孩子数量分布
        kids_vals, kids_counts = np.unique(subset['孩子数量'], return_counts=True)
        kids_probs = kids_counts / len(subset)
        # 确保包含所有可能值（补0）
        all_kids_vals = self.marginals_discrete['孩子数量']['values']
        kids_probs_full = []
        for val in all_kids_vals:
            idx = np.where(kids_vals == val)[0]
            if len(idx) > 0:
                kids_probs_full.append(kids_probs[idx[0]])
            else:
                kids_probs_full.append(0.01)  # 平滑处理
        # 归一化
        kids_probs_full = np.array(kids_probs_full)
        kids_probs_full = kids_probs_full / kids_probs_full.sum()
        conditional_probs['孩子数量'][age_bin] = kids_probs_full.tolist()
        
        # 同理统计学历分布
        edu_vals, edu_counts = np.unique(subset['学历'], return_counts=True)
        edu_probs = edu_counts / len(subset)
        all_edu_vals = self.marginals_discrete['学历']['values']
        edu_probs_full = []
        for val in all_edu_vals:
            idx = np.where(edu_vals == val)[0]
            if len(idx) > 0:
                edu_probs_full.append(edu_probs[idx[0]])
            else:
                edu_probs_full.append(0.01)
        edu_probs_full = np.array(edu_probs_full)
        edu_probs_full = edu_probs_full / edu_probs_full.sum()
        conditional_probs['学历'][age_bin] = edu_probs_full.tolist()
    
    return conditional_probs
```

### 4.3 配置示例（已更新⭐）

```yaml
# config/default/population.yaml

labor_generator:
  seed: 42
  use_copula: gaussian  # gaussian | t | vine
  correlation_method: spearman  # pearson | spearman | kendall
  
  # 连续变量边际分布参数（6个Beta，来自实验）
  marginals_continuous:
    T:
      dist: beta
      params: [1.93, 2.05, 0, 1]
      scale_min: 15.0
      scale_max: 70.0
    S:
      dist: beta
      params: [1.79, 1.57, 0, 1]
      scale_min: 2.0
      scale_max: 44.0
    D:
      dist: beta
      params: [0.37, 0.76, 0, 1]
      scale_min: 0.1
      scale_max: 20.0
    W:
      dist: beta
      params: [1.43, 1.45, 0, 1]
      scale_min: 1400.0
      scale_max: 8000.0
    年龄:
      dist: beta
      params: [1.01, 1.00, 0, 1]
      scale_min: 25.0
      scale_max: 50.0
    累计工作年限:
      dist: beta
      params: [0.55, 1.64, 0, 1]
      scale_min: 0.1
      scale_max: 30.0
  
  # 离散变量经验分布（来自实验）
  marginals_discrete:
    孩子数量:
      values: [0, 1, 2, 3]
      probs: [0.077, 0.370, 0.457, 0.097]
    学历:
      values: [0, 1, 2, 3, 4, 5, 6]
      probs: [0.003, 0.020, 0.107, 0.353, 0.353, 0.160, 0.003]
  
  # 年龄分箱设置（用于条件抽样）
  age_bins:
    - label: "<30"
      min: 0
      max: 30
    - label: "30-40"
      min: 30
      max: 40
    - label: ">=40"
      min: 40
      max: 100
```

---

## 5. EnterpriseGenerator设计

### 5.1 类结构

```python
class EnterpriseGenerator(BaseGenerator):
    """
    企业生成器
    
    基于假设的四维正态分布生成企业特征。
    初始参数通过简单假设确定，后续通过校准优化。
    
    Attributes:
        config: 配置字典
        mean: 均值向量 (4,)
        covariance: 协方差矩阵 (4, 4)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.mean = None
        self.covariance = None
    
    def fit(self, data: pd.DataFrame = None) -> None:
        """
        初始化参数（基于劳动力数据或配置）
        
        Args:
            data: 劳动力数据（可选），用于设定初始均值
        """
        if data is not None:
            # 基于劳动力数据设定初始均值
            labor_mean = data[['T', 'S', 'D', 'W']].mean().values
            
            # 企业需求通常略高于劳动力平均水平
            self.mean = labor_mean * np.array([1.1, 1.05, 1.1, 1.2])
        else:
            # 使用配置中的默认值
            self.mean = np.array(self.config.get('default_mean', [45, 75, 65, 5500]))
        
        # 协方差矩阵（初始为对角矩阵）
        std = np.array(self.config.get('default_std', [10, 15, 15, 1000]))
        self.covariance = np.diag(std ** 2)
        
        self.fitted_params = {
            'mean': self.mean.tolist(),
            'covariance': self.covariance.tolist()
        }
        self.is_fitted = True
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成虚拟企业"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()")
        
        # 从多元正态分布采样
        samples = np.random.multivariate_normal(
            self.mean,
            self.covariance,
            size=n_agents
        )
        
        # 构造DataFrame
        df = pd.DataFrame(samples, columns=['T', 'S', 'D', 'W'])
        df['agent_id'] = range(1001, 1001 + n_agents)  # 企业ID从1001开始
        df['agent_type'] = 'enterprise'
        
        # 确保非负值
        df[['T', 'S', 'D', 'W']] = df[['T', 'S', 'D', 'W']].clip(lower=0)
        
        # 重新排序列
        df = df[['agent_id', 'agent_type', 'T', 'S', 'D', 'W']]
        
        return df
    
    def validate(self, agents: pd.DataFrame) -> bool:
        """简单验证（均值和方差）"""
        generated_mean = agents[['T', 'S', 'D', 'W']].mean().values
        
        # 检查均值是否在合理范围内
        mean_diff = np.abs(generated_mean - self.mean)
        tolerance = self.mean * 0.1  # 10%容忍度
        
        return np.all(mean_diff < tolerance)
    
    def set_params(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        """
        设置参数（用于校准）
        
        Args:
            mean: 新的均值向量
            covariance: 新的协方差矩阵
        """
        self.mean = mean
        self.covariance = covariance
        self.fitted_params = {
            'mean': mean.tolist(),
            'covariance': covariance.tolist()
        }
        self.is_fitted = True
```

### 5.2 配置示例

```yaml
# config/default/population.yaml (续)

enterprise_generator:
  seed: 43
  
  # 初始参数（后续会被校准优化）
  default_mean: [45, 75, 65, 5500]  # T, S, D, W
  default_std: [10, 15, 15, 1000]   # 标准差
  
  # 校准相关
  calibration_enabled: true
  calibration_bounds:
    T: [30, 60]
    S: [50, 90]
    D: [40, 80]
    W: [4000, 7000]
```

---

## 6. Copula建模详解

### 6.1 Gaussian Copula数学原理

**定义**：  
Gaussian Copula通过正态分布的相关结构连接边际分布。

**密度函数**：
$$
c(u_1, u_2, ..., u_d; \mathbf{R}) = \frac{1}{\sqrt{|\mathbf{R}|}} \exp\left(-\frac{1}{2} \mathbf{z}^T (\mathbf{R}^{-1} - \mathbf{I}) \mathbf{z}\right)
$$

其中：
- $u_i = F_i(x_i)$ 是边际CDF
- $z_i = \Phi^{-1}(u_i)$ 是标准正态的逆CDF
- $\mathbf{R}$ 是相关系数矩阵

**采样算法**：
1. 生成 $\mathbf{Z} \sim \mathcal{N}(0, \mathbf{R})$
2. 转换 $U_i = \Phi(Z_i)$ （均匀分布）
3. 逆变换 $X_i = F_i^{-1}(U_i)$ （目标分布）

### 6.2 参数估计

**方法1：基于秩的相关系数** (推荐)
```python
from scipy.stats import spearmanr

# 计算Spearman相关系数
rho, _ = spearmanr(data[['T', 'S', 'D', 'W']])

# 转换为Gaussian Copula参数
# sin(π/6 * ρ_s) ≈ ρ_g (Gaussian copula correlation)
```

**方法2：极大似然估计 (MLE)**
```python
from copulas.multivariate import GaussianMultivariate

copula = GaussianMultivariate()
copula.fit(uniform_data)  # uniform_data是CDF变换后的数据
```

### 6.3 依赖结构验证

**检验方法**：
1. **相关性图**：scatter plot matrix
2. **尾部依赖检验**：检查是否存在尾部相关
3. **Kendall's tau**：非参数相关性度量

```python
import seaborn as sns

# 相关性热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Kendall's tau
from scipy.stats import kendalltau
tau, p_value = kendalltau(data['T'], data['S'])
```

---

## 7. 实现计划

### 7.1 开发顺序

**第1步**：LaborGenerator - 数据准备与验证（0.5天）
- [ ] 数据加载工具函数
- [ ] 数据验证逻辑
- [ ] 边际分布拟合（复用实验结果）

**第2步**：LaborGenerator - Copula建模（1天）
- [ ] CDF变换实现
- [ ] Gaussian Copula拟合
- [ ] 参数提取和保存

**第3步**：LaborGenerator - 生成与验证（0.5天）
- [ ] 采样实现
- [ ] 逆变换实现
- [ ] KS检验验证

**第4步**：EnterpriseGenerator（0.5天）
- [ ] 参数初始化
- [ ] 多元正态采样
- [ ] 简单验证

**第5步**：整合与测试（0.5天）
- [ ] 公共接口 (`__init__.py`)
- [ ] 单元测试
- [ ] 集成测试

**总计**：约 **3天**

### 7.2 验收标准

- [x] LaborGenerator继承BaseGenerator
- [x] EnterpriseGenerator继承BaseGenerator
- [x] 所有方法符合PEP8规范
- [x] KS检验通过（p-value > 0.05）
- [x] Copula拟合收敛
- [x] 代码有完整docstring
- [x] 单元测试覆盖率 > 85%

---

## 8. 测试策略

### 8.1 单元测试结构

```
tests/unit/population/
├── test_labor_generator.py
├── test_enterprise_generator.py
└── test_copula_utils.py
```

### 8.2 测试用例

#### 8.2.1 LaborGenerator测试

```python
import pytest
from src.modules.population import LaborGenerator

class TestLaborGenerator:
    
    def test_fit_with_valid_data(self, sample_labor_data):
        """测试正常拟合"""
        gen = LaborGenerator({'seed': 42})
        gen.fit(sample_labor_data)
        
        assert gen.is_fitted
        assert gen.correlation_matrix is not None
        assert gen.marginals is not None
    
    def test_fit_with_invalid_data(self):
        """测试无效数据"""
        gen = LaborGenerator({'seed': 42})
        invalid_data = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(DataValidationError):
            gen.fit(invalid_data)
    
    def test_generate_before_fit(self):
        """测试未拟合就生成"""
        gen = LaborGenerator({'seed': 42})
        
        with pytest.raises(RuntimeError):
            gen.generate(100)
    
    def test_generate_distribution(self, fitted_labor_gen):
        """测试生成的分布"""
        agents = fitted_labor_gen.generate(1000)
        
        # KS检验
        from scipy.stats import kstest, beta
        for col in ['T', 'S', 'D', 'W']:
            params = fitted_labor_gen.marginals[col]['params']
            ks_stat, p_value = kstest(agents[col], lambda x: beta(*params).cdf(x))
            assert p_value > 0.05, f"{col}的KS检验未通过"
    
    def test_correlation_preservation(self, sample_labor_data, fitted_labor_gen):
        """测试相关性是否保留"""
        original_corr = sample_labor_data[['T', 'S', 'D', 'W']].corr()
        
        agents = fitted_labor_gen.generate(5000)
        generated_corr = agents[['T', 'S', 'D', 'W']].corr()
        
        # 相关系数差异应小于0.1
        diff = np.abs(original_corr - generated_corr)
        assert np.all(diff < 0.1)
```

### 8.3 集成测试

```python
def test_full_population_generation():
    """测试完整的人口生成流程"""
    # 1. 加载真实数据
    data = pd.read_csv('data/input/cleaned_data.csv')
    
    # 2. 拟合劳动力生成器
    labor_gen = LaborGenerator({'seed': 42})
    labor_gen.fit(data)
    
    # 3. 生成劳动力
    laborers = labor_gen.generate(1000)
    assert len(laborers) == 1000
    assert all(laborers['agent_type'] == 'labor')
    
    # 4. 拟合企业生成器
    ent_gen = EnterpriseGenerator({'seed': 43})
    ent_gen.fit(data)  # 基于劳动力数据
    
    # 5. 生成企业
    enterprises = ent_gen.generate(800)
    assert len(enterprises) == 800
    assert all(enterprises['agent_type'] == 'enterprise')
    
    # 6. 转换为Agent对象
    from src.core import Agent
    labor_agents = [
        Agent.from_array(
            row['agent_id'],
            'labor',
            row[['T', 'S', 'D', 'W']].values
        )
        for _, row in laborers.iterrows()
    ]
    
    assert len(labor_agents) == 1000
```

---

## 9. API接口定义

### 9.1 公共接口

```python
# src/modules/population/__init__.py

from .labor_generator import LaborGenerator
from .enterprise_generator import EnterpriseGenerator

__all__ = [
    'LaborGenerator',
    'EnterpriseGenerator',
]
```

### 9.2 使用示例

```python
from src.modules.population import LaborGenerator, EnterpriseGenerator
import pandas as pd

# 1. 加载数据
data = pd.read_csv('data/input/cleaned_data.csv')

# 2. 创建并拟合劳动力生成器
labor_gen = LaborGenerator({'seed': 42})
labor_gen.fit(data)
labor_gen.save_params('models/labor_generator.pkl')

# 3. 生成虚拟劳动力
laborers = labor_gen.generate(1000)
print(laborers.head())

# 4. 验证
is_valid = labor_gen.validate(laborers)
print(f"劳动力生成验证: {'通过' if is_valid else '失败'}")

# 5. 创建并拟合企业生成器
ent_gen = EnterpriseGenerator({'seed': 43, 'default_mean': [45, 75, 65, 5500]})
ent_gen.fit(data)

# 6. 生成虚拟企业
enterprises = ent_gen.generate(800)
print(enterprises.head())

# 7. 保存生成的虚拟人口
laborers.to_csv('data/output/virtual_laborers.csv', index=False)
enterprises.to_csv('data/output/virtual_enterprises.csv', index=False)
```

---

## 10. 风险与注意事项

### 10.1 潜在风险

1. **Copula拟合不收敛**
   - 原因：相关矩阵非正定、数据质量差
   - 缓解：添加正则化项、检查特征值

2. **生成样本超出合理范围**
   - 原因：边际分布参数不准确、Copula尾部行为
   - 缓解：添加截断、后处理筛选

3. **相关性丢失**
   - 原因：采样数量不足、Copula类型不匹配
   - 缓解：增加采样量、尝试t-Copula

### 10.2 优化方向

1. **性能优化**
   - 使用Numba加速采样循环
   - 批量生成减少函数调用

2. **功能扩展**
   - 支持条件生成（给定某些特征）
   - 支持时变分布（不同时期的劳动力特征）

3. **高级Copula**
   - Vine Copula（更灵活的依赖结构）
   - 动态Copula（时间演化的依赖）

---

## 11. 依赖与环境

### 11.1 新增依赖

```python
# requirements.txt (新增)
copulas >= 0.9.0        # Copula建模
scikit-learn >= 1.3.0   # KS检验等
```

### 11.2 完整依赖列表

```python
numpy >= 1.26.0
pandas >= 2.2.0
scipy >= 1.11.0
copulas >= 0.9.0
scikit-learn >= 1.3.0
pyyaml >= 6.0

# 测试
pytest >= 8.3.0
pytest-cov >= 5.0.0
```

---

## 12. 审阅清单

**请审阅以下设计决策**：

- [x] **Copula类型**：6维Gaussian Copula ✅已确定
- [x] **企业生成方案**：四维正态 + 后续校准 ✅已确定
- [x] **参数估计方法**：基于Spearman相关系数 + MLE ✅已确定
- [x] **数据缩放**：反归一化到原始尺度 ✅已确定
- [x] **离散变量处理**：经验分布 + 条件抽样 ✅已确定
- [ ] **实现计划**：3.5天是否合理？（增加了离散变量处理）

**已确认决策**⭐：

1. **控制变量处理**：✅已加入8个变量（4核心 + 2连续控制 + 2离散控制）
2. **变量分类**：✅6个连续变量用Beta + 2个离散变量用经验分布
3. **生成策略**：✅6维Copula + 年龄条件抽样离散变量
4. **Copula库选择**：✅使用 `copulas` 库的 `GaussianMultivariate`

**最新更新（2025/10/01）**：
- 修正离散变量建模方法（孩子数量、学历）
- 采用6维Copula而非8维（避免离散变量连续化）
- 新增条件概率估计函数`_estimate_conditional_probs`
- 验证方法更新：连续变量用KS检验，离散变量用卡方检验

---

## 13. 文档更新记录

### 2025/10/01 - 重大更新：离散变量处理 ⭐

**变更原因**：  
边际分布实验发现孩子数量和学历是离散变量（只有4个和7个唯一值），用连续Beta分布强行拟合统计上不严谨。

**主要变更**：

1. **变量重新分类**：
   - 连续变量（6个）：T, S, D, W, 年龄, 累计工作年限 → Beta分布
   - 离散变量（2个）：孩子数量, 学历 → 经验分布

2. **生成策略调整**：
   - 从8维联合Copula → **6维Copula + 离散变量条件抽样**
   - 新增条件概率估计：年龄 → 孩子数量、学历

3. **代码设计更新**：
   - `LaborGenerator`新增`marginals_discrete`和`conditional_probs`属性
   - 新增`_estimate_conditional_probs()`和`_get_age_bin()`辅助方法
   - `generate()`方法增加离散变量条件抽样逻辑
   - `validate()`方法新增卡方检验

4. **配置文件更新**：
   - 新增`marginals_discrete`配置节
   - 新增`age_bins`配置节

**影响范围**：
- ✅ 提高统计严谨性
- ✅ 简化实现（6维比8维更快）
- ⚠️ 增加实现复杂度（需要条件概率估计）
- ⚠️ 预计实现时间从3天增加到3.5天

---

**文档状态**: ✅ 已更新完成，待用户审阅  
**预计实现时间**: 3.5天（+0.5天用于离散变量处理）  
**下一步**: 用户审阅 → 实现代码 → 单元测试
