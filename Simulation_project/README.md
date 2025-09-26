# 农村女性就业市场MFG模拟系统

## 项目概述

本项目旨在构建融合平均场博弈（MFG）与基于主体建模（ABM）的农村女性就业市场动态模拟系统，用于分析个体努力决策与宏观市场状态的双向反馈机制，并评估政策干预效果。

## 项目架构

```
Simulation_project/
├── modules/              # 核心模块代码
│   ├── population_generator/    # Module 1: 主体生成器
│   ├── matching_engine/         # Module 2: 匹配引擎  
│   ├── match_function_estimator/ # Module 3: 匹配函数估计器
│   ├── mfg_simulator/           # Module 4: MFG模拟器
│   └── experiment_controller/   # Module 5: 实验控制器
├── config/               # 配置文件
│   └── config.yaml             # 全局参数配置
├── data/                 # 数据文件
│   ├── input/                  # 输入数据
│   └── output/                 # 输出数据
├── results/              # 模拟结果
│   ├── reports/                # 分析报告
│   ├── figures/                # 图表
│   └── logs/                   # 运行日志
├── docs/                 # 文档
│   ├── api/                    # API文档
│   ├── user_guide/             # 用户指南
│   └── technical/              # 技术文档
├── tests/                # 测试文件
│   ├── unit_tests/             # 单元测试
│   └── integration_tests/      # 集成测试
├── 模拟系统需求确认文档.md      # 需求分析文档
└── 研究计划/              # 研究计划相关文档
    ├── 模拟部分项目计划书.md
    ├── 研究计划.md
    └── 研究计划.pdf
```

## 核心模块

### Module 1: PopulationGenerator (主体生成器)
- **功能**: 生成虚拟劳动力和企业主体池
- **基础**: 基于现有的Copula建模实现
- **输出**: labor_agents.csv, enterprise_agents.csv

### Module 2: MatchingEngine (匹配引擎)  
- **功能**: 实现Gale-Shapley稳定匹配算法
- **特性**: 支持多θ值场景的批量模拟
- **输出**: matching_data_for_logit.csv

### Module 3: MatchFunctionEstimator (匹配函数估计器)
- **功能**: 通过Logit回归估计匹配概率函数λ
- **方法**: 基于ABM模拟数据训练
- **输出**: match_function_params.json

### Module 4: MFGSimulator (平均场博弈模拟器) ⚠️
- **功能**: 求解MFG均衡状态
- **算法**: 值迭代法 + KFE演化
- **挑战**: 计算复杂度极高，需要优化

### Module 5: ExperimentController (实验控制器)
- **功能**: 参数校准与政策分析
- **模式**: calibration / policy_analysis
- **输出**: 政策效果对比报告

## 开发计划

### 阶段1: 基础框架搭建 (2-3周)
- [ ] 完善PopulationGenerator模块
- [ ] 实现基础MatchingEngine
- [ ] 建立模块间接口标准

### 阶段2: 核心算法实现 (4-6周)
- [ ] 实现MatchFunctionEstimator
- [ ] 简化版MFGSimulator (降维实现)
- [ ] 基础收敛性测试

### 阶段3: 系统集成与优化 (2-4周)  
- [ ] ExperimentController实现
- [ ] 性能优化与并行化
- [ ] 完整系统测试

### 阶段4: 校准与验证 (2-3周)
- [ ] 参数校准与敏感性分析
- [ ] 结果验证与报告生成

## 技术栈

- **核心库**: numpy, scipy, pandas
- **优化库**: numba, cython  
- **并行库**: multiprocessing, joblib
- **可视化**: matplotlib, plotly, seaborn
- **配置**: pyyaml, json

## 使用说明

### 环境准备
```bash
# 激活虚拟环境
cd D:\Python\2025大创
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 运行模拟
```bash
# 进入模拟系统目录
cd Simulation_project

# 运行完整模拟流程
python main.py --config config/config.yaml --mode simulation

# 运行政策分析
python main.py --config config/config.yaml --mode policy_analysis
```

## 注意事项

⚠️ **高计算复杂度**: MFG模块计算复杂度极高，建议使用高性能计算环境  
⚠️ **内存需求**: 大规模模拟需要32GB+内存  
⚠️ **收敛性**: MFG均衡不保证收敛，需要监控数值稳定性

## 相关文档

- [模拟系统需求确认文档](./模拟系统需求确认文档.md) - 详细需求分析与技术方案
- [模拟部分项目计划书](./研究计划/模拟部分项目计划书.md) - 项目架构设计
- [研究计划](./研究计划/研究计划.md) - 完整研究计划

## 联系信息

**项目负责人**: 李心泠  
**技术指导**: AI编程助手  
**最后更新**: 2025年9月26日
