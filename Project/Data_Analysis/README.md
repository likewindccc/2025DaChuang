# 农村女性就业市场数据分析模块

**China's Rural Female Employment Market Data Analysis**

> 基于平均场博弈理论与主体建模思想的数据分析系统  
> 对应研究计划第4.2节：市场主体特征的确定

---

## 📋 模块概述

本模块实现了农村女性就业市场数据的完整分析流程，包括概率分布推断、变量依赖性建模、虚拟个体生成等关键功能，为后续的ABM/MFG仿真提供可靠的数据基础。

## 🗂️ 文件结构

### 核心分析脚本
- **`distribution_inference.py`** - 概率分布推断与参数估计
  - 基于MLE的多分布拟合分析
  - Anderson-Darling拟合优度检验
  - AIC/BIC模型选择
  - 支持10种常见概率分布

- **`cleaned_data.csv`** - 清洗后的原始调研数据
  - 300个农村女性就业样本
  - 包含19个调研变量
  - UTF-8编码，支持中文变量名

### Copula建模子模块 (`关于变量不独立问题的研究/`)
- **`copula_agent_generator.py`** - 基于Copula的虚拟个体生成器
  - 解决变量间非独立性问题
  - 支持多种Copula模型（Gaussian, Vine等）
  - 10,000个高质量虚拟个体生成
  - 完整的质量验证体系

- **`virtual_population_gaussian.csv`** - 生成的虚拟个体数据
- **`copula_model_selection_report.md`** - Copula模型选择报告
- **可视化图表**: 分布对比图、相关性热力图

### 分析报告
- **`分布估计结果报告.md`** - 详细的统计分析结果
- **`Copula函数解决变量分布不独立的问题.md`** - 理论说明文档

### 探索性分析
- **`0913a_analysis_v1.ipynb`** - Jupyter notebook探索性分析
  - 数据可视化
  - 描述性统计
  - 初步模式发现

## 🎯 核心功能特性

### 1. 分布推断分析 (`distribution_inference.py`)

```python
from distribution_inference import DistributionFitter

# 创建分布拟合器
fitter = DistributionFitter()

# 对变量进行多分布拟合
results = fitter.fit_variable(data, variable_name)

# 生成比较表
fitter.create_comparison_table()
```

**支持的概率分布**:
- 正态分布 (Normal)
- 贝塔分布 (Beta) ⭐ 主要使用
- 对数正态分布 (LogNormal)
- 伽马分布 (Gamma)
- 威布尔分布 (Weibull)
- 均匀分布 (Uniform)
- 指数分布 (Exponential)
- 帕累托分布 (Pareto)
- 卡方分布 (Chi-squared)
- 广义极值分布 (Generalized Extreme Value)

### 2. Copula虚拟个体生成 (`copula_agent_generator.py`)

```python
from copula_agent_generator import CopulaAgentGenerator

# 创建生成器
generator = CopulaAgentGenerator()

# 完整生成流程
generator.load_data()
generator.setup_marginal_distributions()
generator.fit_and_compare_copulas()
virtual_population = generator.generate_virtual_agents(10000)
```

**核心优势**:
- 🎯 **解决非独立性**: 避免传统独立采样的不现实组合
- 📊 **保持相关性**: 准确复现原始数据的依赖结构
- ✅ **质量验证**: 完整的统计检验与可视化验证
- 🔧 **数值稳定**: 处理边界值和异常情况

## 🎲 核心状态变量定义

根据研究计划，定义个体状态向量 **x = (T, S, D, W)**：

| 变量 | 含义 | 分布类型 | 参数 |
|------|------|----------|------|
| **T** | 每周工作时长 | Beta分布 | α=1.926, β=2.054 |
| **S** | 工作能力评分 | Beta分布 | α=1.790, β=1.568 |
| **D** | 数字素养评分 | Beta分布 | α=0.374, β=0.755 |
| **W** | 每月期望收入 | Beta分布 | α=1.434, β=1.448 |

## 📊 主要分析结果

### 分布拟合结果
- **✅ Beta分布占主导**: 8/11个变量的最佳拟合分布
- **🎯 理论一致性**: 有界变量特征与Beta分布的天然匹配
- **📈 拟合质量**: 平均AIC < -30，伪R² > 0.99

### Copula建模效果
- **🏆 最佳模型**: Gaussian Copula (AIC = 3196.28)
- **✅ 相关性保持**: 相关性误差 < 0.05
- **👥 虚拟个体质量**: 10,000个个体，异常组合 < 10%

## 🔧 环境依赖

### 核心依赖
```txt
numpy==1.26.4
pandas==2.2.3  
scipy==1.14.1
copulas==0.12.3  # Copula建模
statsmodels==0.14.4
scikit-learn==1.5.2
```

### 完整依赖
详见项目根目录 `requirements.txt`

## 🚀 快速开始

### 1. 环境配置
```bash
# 激活虚拟环境
D:\Python\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行分布推断分析
```bash
cd Data_Analysis
python distribution_inference.py
```

### 3. 生成虚拟个体
```bash
cd 关于变量不独立问题的研究
python copula_agent_generator.py
```

## 📝 输出文件说明

| 文件 | 类型 | 描述 |
|------|------|------|
| `分布估计结果报告.md` | 报告 | 详细的分布拟合分析结果 |
| `virtual_population_gaussian.csv` | 数据 | 10,000个虚拟个体数据 |
| `copula_model_selection_report.md` | 报告 | Copula模型比较与选择 |
| `*.png` | 图表 | 分布对比图、相关性热力图 |

## 🎓 学术价值

### 方法论贡献
1. **变量依赖建模**: 首次将Copula理论应用于农村就业市场建模
2. **虚拟个体生成**: 解决传统ABM中个体属性独立性假设的不现实问题
3. **统计严格性**: 基于MLE估计与假设检验的参数化方法

### 应用价值
- 为ABM/MFG仿真提供高质量初始种群
- 支持政策仿真与效果预测
- 建立可重复的量化研究范式

## 📚 相关文档

- [研究计划](../Simulation_project/研究计划/研究计划.md) - 理论背景与整体框架
- [项目需求文档](../requirements.txt) - 完整环境配置
- [Copula理论说明](关于变量不独立问题的研究/Copula函数解决变量分布不独立的问题.md)

---

**Author**: Claude-4 AI Assistant  
**Date**: 2024-09-24  
**Version**: 2.0.0  
**License**: Academic Research Use
