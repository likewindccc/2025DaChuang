# 农村女性就业市场MFG模拟系统 v2.0

**EconLab: Rural Female Employment Market Simulation**

基于平均场博弈(MFG)与主体建模(ABM)的农村女性就业市场动态演化机制研究

---

## 📋 项目概述

本项目实现了融合平均场博弈与基于主体建模思想的农村女性就业市场动态模拟系统，用于分析个体努力决策与宏观市场状态的双向反馈机制，并评估政策干预效果。

### 核心特性

- ✅ 基于Copula理论的虚拟劳动力生成
- ✅ 四维正态分布的企业主体建模  
- ✅ Gale-Shapley稳定匹配算法（用于ABM数据生成）
- ✅ Logit回归匹配函数估计
- ✅ 简化版MFG求解器（离散状态空间）
- ✅ 遗传算法参数校准
- ✅ Numba加速核心计算
- ✅ 完整的测试与文档体系

### 🔑 核心设计理念

本项目采用**两阶段设计**：

#### 阶段1: ABM数据生成 + 参数估计
- **目的**: 估计匹配函数λ(x, σ, a, θ)的参数
- **方法**: 
  1. 生成虚拟劳动力和虚拟企业
  2. 使用**有限轮次匹配**（limited_rounds_matching）模拟真实摩擦
  3. 记录(特征, 努力, 市场紧张度, 匹配结果)
  4. Logit回归估计λ的参数

#### 阶段2: MFG均衡求解
- **目的**: 求解均衡策略和人口分布
- **方法**:
  1. **只生成劳动力个体**（不生成企业）
  2. 使用估计的λ函数计算每个劳动力的匹配概率
  3. **随机抽样判断就业/失业**（生成0-1随机数与概率比较）
  4. 企业被抽象为"匹配概率场"，隐式体现在λ函数中

⚠️ **关键区别**: ABM阶段使用GS匹配算法，MFG阶段使用概率函数

---

## 🏗️ 项目结构

```
Simulation_project_v2/
├── src/                    # 源代码
│   ├── core/              # 核心基础类
│   ├── modules/           # 5个核心模块
│   │   ├── population/   # Module 1: 主体生成
│   │   ├── matching/     # Module 2: 匹配引擎
│   │   ├── estimation/   # Module 3: 匹配函数估计
│   │   ├── mfg/          # Module 4: MFG求解器
│   │   └── calibration/  # Module 5: 参数校准
│   └── utils/             # 工具函数
│
├── config/                 # 配置文件
│   ├── default/           # 默认配置
│   └── experiments/       # 实验配置
│
├── data/                   # 数据文件
│   ├── input/             # 输入数据
│   └── output/            # 输出数据
│
├── results/                # 结果文件
│   ├── figures/           # 图表
│   ├── reports/           # 报告
│   └── logs/              # 日志
│
├── tests/                  # 测试代码
│   ├── unit/              # 单元测试
│   ├── integration/       # 集成测试
│   └── benchmarks/        # 性能测试
│
├── docs/                   # 文档
│   ├── userdocs/          # 用户文档
│   ├── developerdocs/     # 开发者文档
│   └── academicdocs/      # 学术文档
│
└── requirements.txt        # 依赖清单
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# ⚠️ 重要：激活项目专属虚拟环境
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1

# 或使用快捷脚本（推荐）
cd D:\Python\2025DaChuang\Simulation_project_v2
activate_env.bat

# 安装依赖
pip install -r ../requirements.txt
```

### 2. 运行完整流程（开发中）

```bash
# 进入项目目录
cd D:\Python\2025DaChuang\Simulation_project_v2

# 运行主程序（待实现）
python -m src.main --config config/default/base_config.yaml
```

### 3. 运行测试

```bash
# 单元测试
pytest tests/unit/ -v

# 集成测试  
pytest tests/integration/ -v

# 性能基准测试
pytest tests/benchmarks/ -v --benchmark-only
```

---

## 📚 文档导航

### 用户文档
- [用户手册](docs/userdocs/user_manual.md) - 系统使用指南
- [配置说明](docs/userdocs/configuration_guide.md) - 参数配置详解
- [常见问题](docs/userdocs/faq.md) - FAQ

### 开发者文档
- [架构设计](docs/developerdocs/architecture.md) - 系统架构详解 ⭐
- [技术选型](docs/developerdocs/tech_stack.md) - 技术栈说明
- [开发路线图](docs/developerdocs/roadmap.md) - 开发计划
- [代码规范](docs/developerdocs/coding_standards.md) - 编码规范
- [API文档](docs/developerdocs/api_reference.md) - 接口文档

### 学术文档
- [方法论](docs/academicdocs/methodology.md) - 理论基础
- [算法说明](docs/academicdocs/algorithms.md) - 数学公式
- [参数校准](docs/academicdocs/calibration.md) - 校准过程
- [实验结果](docs/academicdocs/experiments.md) - 结果分析

---

## 🎯 开发状态

### 当前版本: v2.0-alpha

| 模块 | 状态 | 进度 | 说明 |
|------|------|------|------|
| Module 1: Population | 🟡 开发中 | 30% | 重构Copula生成器 |
| Module 2: Matching | 🔴 未开始 | 0% | 待实现 |
| Module 3: Estimation | 🔴 未开始 | 0% | 待实现 |
| Module 4: MFG | 🔴 未开始 | 0% | 待实现 |
| Module 5: Calibration | 🔴 未开始 | 0% | 待实现 |

---

## 🔬 技术栈

- **语言**: Python 3.12.5
- **核心库**: numpy, pandas, scipy
- **优化**: numba (强制使用)
- **校准**: DEAP (遗传算法)
- **测试**: pytest, pytest-benchmark
- **文档**: Sphinx
- **代码规范**: PEP8 + 项目特定规范

---

## 📖 研究背景

本项目基于"求是学术"项目申报书：

**项目名称**: "她"应何为：中国农村女性就业市场的动态演化机制探究——基于平均场博弈与主体建模思想

**核心问题**:
1. 农村女性如何动态调整努力水平以提高就业机会？
2. 个体决策与宏观市场状态如何相互作用？
3. 如何设计最优政策组合促进农村女性就业？

**理论创新**:
- 融合MFG与ABM，建立微观-宏观双向反馈机制
- 基于Copula理论解决变量非独立性问题
- 构建微观匹配函数，突破宏观匹配函数局限

---

## 👥 项目团队

**负责人**: 李心泠（经济学院）  
**指导教师**: 李三希、林琳  
**技术开发**: Claude-4 AI Assistant

---

## 📄 许可证

Academic Research Use Only

---

**Last Updated**: 2025-09-30  
**Version**: 2.0-alpha
