# Change_Log.md

项目修改日志 - 追踪所有重要变更

---

## 格式说明

每条记录包含：
- 北京时间
- Git Commit（短哈希）
- 受影响文件清单
- 变更动机与影响范围

---

# 修改 1 北京时间 2025/09/30 15:45
## Commit: (待提交) - Initial project structure

**新增文件**:
- `Simulation_project_v2/` - 项目根目录
- `README.md` - 项目说明
- `setup_directories.py` - 目录初始化脚本
- `activate_env.bat` - 虚拟环境激活快捷脚本（使用 D:\Python\2025DaChuang\venv）
- `docs/developerdocs/architecture.md` - 架构设计文档
- `docs/developerdocs/tech_stack.md` - 技术选型文档
- `docs/developerdocs/roadmap.md` - 开发路线图
- `docs/developerdocs/coding_standards.md` - 代码规范
- `docs/userdocs/` - 用户文档框架
- `docs/academicdocs/` - 学术文档框架
- `config/default/` - 默认配置文件
- `config/experiments/` - 实验配置文件
- `Change_Log.md` - 本文档

**目的**:  
创建Simulation_project_v2完整项目架构，重构旧版代码，建立清晰的模块化结构和完善的文档体系。

**影响范围**:  
- 全新项目启动
- 不影响旧版 `Simulation_project/`
- 为后续6个月开发奠定基础

**审阅状态**: 待用户审阅

---

# 修改 2 北京时间 2025/09/30 21:25
## Commit: (待提交) - Fix folder path and terminal issues

**修改文件**:
- `SESSION_HANDOVER.md` - 更新所有路径引用
- `README.md` - 更新虚拟环境路径
- `PROJECT_SUMMARY.md` - 更新虚拟环境路径
- `ENVIRONMENT.md` - 更新所有路径示例
- `Change_Log.md` - 更新路径引用
- `activate_env.bat` - 更新虚拟环境路径
- `docs/userdocs/user_manual.md` - 更新路径示例
- `marginal_distribution_experiment.py` - 更新注释中的路径

**变更内容**:
将项目文件夹名从 `2025大创` 改为 `2025DaChuang`（去除中文），解决终端工具无法运行的问题。

所有路径引用已更新：
- `D:\Python\2025大创` → `D:\Python\2025DaChuang`

**变更动机**:  
- 中文路径导致 Cursor 终端工具出现 `ENOENT` 错误
- 更改为英文路径后终端工具恢复正常
- 统一更新所有文档和配置文件中的路径引用

**影响范围**:  
- ✅ 终端工具现已正常工作
- ✅ 所有文档中的路径已同步更新
- ✅ 虚拟环境路径保持一致
- ⚠️ 用户需注意使用新的文件夹名

**测试验证**:
- ✅ `echo "test"` - 终端正常运行
- ✅ `cd D:\Python\2025DaChuang\Simulation_project_v2; dir` - 路径访问正常

**审阅状态**: 已完成

---

# 修改 3 北京时间 2025/09/30 21:30
## Commit: (待提交) - Clean up temporary documents

**删除文件**:
- `SESSION_HANDOVER.md` - 会话交接文档（终端已修复，不需要新会话）
- `PATH_FIX_VERIFICATION.md` - 路径修复验证报告（临时性文档）

**变更动机**:  
- 终端问题已解决，不需要会话交接文档
- 路径修复验证已完成，临时验证报告可删除
- 保持项目目录整洁，只保留核心文档

**保留的核心文档**:
- ✅ README.md - 项目说明
- ✅ PROJECT_SUMMARY.md - 项目架构总结  
- ✅ ENVIRONMENT.md - 环境配置说明
- ✅ Change_Log.md - Git修改日志
- ✅ docs/ - 所有架构、技术、用户、学术文档

**审阅状态**: 已完成

---

# 修改 4 北京时间 2025/09/30 21:35
## Commit: (待提交) - Reorganize experiments and run marginal distribution fitting

**新增文件**:
- `experiments/` - 实验脚本文件夹
- `experiments/README.md` - 实验文件夹说明
- `data/input/cleaned_data.csv` - 清洗后的调研数据（从Project复制）
- `data/output/best_distributions.txt` - 边际分布拟合结果

**移动文件**:
- `marginal_distribution_experiment.py` → `experiments/marginal_distribution_experiment.py`

**修改文件**:
- `experiments/marginal_distribution_experiment.py` - 更新数据路径，去除emoji（解决Windows编码问题）

**实验结果**:
- ✅ 每周工作时长: Beta分布 (α=1.93, β=2.05)
- ✅ 工作能力评分: Beta分布 (α=1.79, β=1.57)
- ⚠️ 数字素养评分: Lognorm分布 (参数异常，需要检查)
- ✅ 每月期望收入: Beta分布 (α=1.43, β=1.45)

**变更动机**:  
- 整理项目结构，将实验脚本独立到 `experiments/` 文件夹
- 复制数据文件到项目内部，便于版本管理
- 完成边际分布拟合实验，为Core模块开发提供参数依据

**影响范围**:  
- ✅ 项目结构更清晰（核心代码 vs 实验脚本）
- ✅ 数据自包含，不依赖外部路径
- ⚠️ 数字素养评分的Lognorm参数需要进一步检查

**审阅状态**: 待审阅（数字素养评分结果异常）

---

# 修改 5 北京时间 2025/09/30 22:00
## Commit: (待提交) - Create Core module development plan

**新增文件**:
- `docs/developerdocs/modules/Phase1_Core_Development_Plan.md` - Core模块详细开发文档

**文档内容**:
- 模块概述（职责、依赖关系）
- 设计决策（dataclass、类型注解、异常体系）
- 核心数据结构（Agent、MatchingPair、SimulationState、MFGEquilibrium）
- 基础类设计（BaseGenerator、BaseSimulator）
- 异常体系（7个细粒度异常）
- 类型系统（类型别名定义）
- API接口定义
- 测试策略
- 实现计划（预计3.5天）

**变更动机**:  
- 遵循开发流程：先文档后代码
- 为Core模块编码提供详细设计依据
- 明确接口设计，便于后续模块开发

**待审阅问题**:
1. 数据结构方案（dataclass）是否合适？
2. 异常粒度（7个异常类）是否合理？
3. Agent的验证逻辑是否足够？
4. 数字素养评分的Lognorm参数异常如何处理？

**审阅状态**: 待用户审阅

---

# 修改 6 北京时间 2025/09/30 22:15
## Commit: (待提交) - Fix digital literacy distribution fitting

**修改文件**:
- `experiments/marginal_distribution_experiment.py` - 修正数据加载逻辑

**修正内容**:
- 识别并处理数字素养评分为0的样本（共36个，占12%）
- 将这些样本的值设为0.1，避免对数正态分布拟合时的log(0)问题

**实验结果更新**:

修正前：
- 数字素养评分 → Lognorm（参数异常：σ=242.62）

修正后：
- 数字素养评分 → **Beta(α=0.37, β=0.76)**, AIC=-313.78 ✅

**关键发现**:
- **所有4个核心变量统一拟合为Beta分布**
- 数字素养评分呈U型分布（α<1, β<1），符合现实（两极分化）
- 为后续Copula建模提供了一致的边际分布

**影响范围**:
- ✅ Core模块开发文档已更新（边际分布结果表格）
- ✅ 数据预处理逻辑已固化（自动修正0值）
- ✅ 为Population模块的Copula建模扫清障碍

**文档更新**:
- `docs/developerdocs/modules/Phase1_Core_Development_Plan.md` - 更新边际分布表格和说明

---

# 修改 7 北京时间 2025/09/30 22:30
## Commit: (待提交) - Fix goodness-of-fit test (AD → KS)

**修改文件**:
- `experiments/marginal_distribution_experiment.py` - 修正拟合优度检验逻辑

**问题诊断**（用户发现）:
- 同一变量的所有候选分布的AD统计量完全相同 ❌
- 原因：所有分布都用`stats.anderson(data, dist='norm')`检验原始数据是否服从正态分布
- 实际应该：检验每个拟合分布与数据的吻合度

**修正方案**:
- Anderson-Darling检验 → **Kolmogorov-Smirnov (KS) 检验**
- 使用拟合后的分布CDF：`stats.kstest(data, fitted_dist.cdf)`
- KS检验支持任意分布，更通用

**修正效果**:

修正前：
```
每周工作时长所有分布的AD = 4.696（相同）❌
```

修正后：
```
每周工作时长：
  beta:   KS=0.2136
  gamma:  KS=0.1740
  lognorm: KS=0.1655
  weibull: KS=0.1437 ✅（各不相同，正确反映拟合质量）
```

**关键洞察**:
- KS统计量越小 → 拟合越好
- Beta分布虽然AIC最优，但KS不总是最小
- AIC考虑模型复杂度，是更科学的准则
- **最终选择仍基于AIC（所有变量均为Beta分布）**

**影响范围**:
- ✅ 拟合优度评估现在更准确
- ✅ 结果输出更科学（KS统计量有区分度）
- ✅ 为后续分布选择提供更可靠的依据

---

---

# 修改 8 北京时间 2025/10/01 00:40
## Commit: (待提交) - Implement Core module (Phase 1, Week 1)

**新增文件**:
- `src/core/data_structures.py` - 核心数据结构（467行）
- `src/core/exceptions.py` - 异常体系（208行）
- `src/core/types.py` - 类型定义（350行）
- `src/core/base_generator.py` - 生成器基类（220行）
- `src/core/base_simulator.py` - 模拟器基类（270行）
- `src/core/__init__.py` - 公共接口（200行）
- `test_core_module.py` - 功能验证脚本

**实现内容**:

1. **数据结构** (`data_structures.py`):
   - `Agent`: 个体基类（劳动力/企业），支持to_array/from_array转换
   - `MatchingPair`: 匹配对，记录匹配结果和质量
   - `SimulationState`: 模拟状态，完整记录某时刻的系统状态
   - `MFGEquilibrium`: MFG均衡结果，包含值函数、策略、分布

2. **异常体系** (`exceptions.py`):
   - `SimulationError`: 基础异常
   - 6个子异常：DataValidationError, CopulaFittingError, MatchingError, 
     ConvergenceError, ConfigurationError, CalibrationError
   - 异常映射工具：`get_exception_class()`

3. **类型系统** (`types.py`):
   - 基础类型：AgentID, TimeStep, ParameterDict
   - NumPy类型：AgentFeatures, PreferenceMatrix, ValueFunction, Distribution等
   - 函数类型：ObjectiveFunction, MatchFunction, UtilityFunction
   - 配置类型：Config, ExperimentConfig
   - 类型检查工具：is_valid_agent_features(), is_valid_distribution()等

4. **基础类** (`base_generator.py`, `base_simulator.py`):
   - `BaseGenerator`: 生成器抽象基类，定义fit→generate→validate流程
   - `BaseSimulator`: 模拟器抽象基类，定义setup→run→get_results流程
   - 支持参数持久化（save/load）
   - 完整的生命周期管理

5. **公共接口** (`__init__.py`):
   - 导出40+个组件
   - 完整性检查
   - 模块信息查询：get_module_info(), print_module_summary()

**代码规范**:
- ✅ 100% PEP8规范
- ✅ 所有公共接口有完整docstring
- ✅ 所有函数有类型注解
- ✅ 所有数据结构有__post_init__验证
- ✅ 无linter错误

**测试结果**:
```
测试1: Agent数据结构           ✅ 通过
测试2: MatchingPair数据结构     ✅ 通过
测试3: SimulationState数据结构  ✅ 通过
测试4: MFGEquilibrium数据结构   ✅ 通过
测试5: 异常体系                ✅ 通过
测试6: 模块信息                ✅ 通过
─────────────────────────────────────
总测试数: 6
通过: 6
失败: 0
```

**关键设计**:
- 使用dataclass实现数据结构（轻量、类型安全）
- 细粒度异常设计（7个异常类）
- 严格类型注解（支持mypy检查）
- Agent的to_array顺序固定为[T, S, D, W]
- additional_attrs保持灵活Dict结构

**影响范围**:
- ✅ 为所有后续模块提供统一基础
- ✅ Population模块可继承BaseGenerator
- ✅ Matching/MFG模块可继承BaseSimulator
- ✅ 统一的异常处理和类型系统
- ✅ 代码总行数：~1715行（含注释）

**代码统计**:
- 数据结构: 4个类，467行
- 异常: 7个类，208行
- 类型: 20+个别名，350行
- 基础类: 2个抽象类，490行
- 公共接口: 200行
- **总计: ~1715行（含文档字符串）**

---

---

# 修改 9 北京时间 2025/10/01 00:50
## Commit: (待提交) - Fix experiments folder location

**问题发现**（用户指出）:
- 实验文件被错误地放在 `docs/developerdocs/experiments/` ❌
- 应该在项目根目录的 `experiments/` ✅

**修正操作**:
- 创建根目录 `experiments/` 文件夹
- 移动 `marginal_distribution_experiment.py` 到正确位置
- 移动 `README.md` 到正确位置
- 删除错误的 `docs/developerdocs/experiments/` 目录

**修正后的正确结构**:
```
Simulation_project_v2/
├── experiments/                          ✅ 正确位置
│   ├── marginal_distribution_experiment.py
│   └── README.md
└── docs/developerdocs/                   
    ├── architecture.md
    ├── modules/
    └── ...                               ✅ 不再有experiments/
```

**影响范围**:
- ✅ 目录结构现在符合项目规范
- ✅ 实验脚本仍可正常运行
- ✅ 文档目录更清晰

---

## 下一步计划

- [ ] 编写Core模块单元测试（使用pytest，覆盖率>90%）
- [ ] 开始Phase 1, Week 2：Population模块开发
- [ ] 创建Population模块开发文档
- [ ] 实现LaborGenerator和EnterpriseGenerator

---

**文档维护规则**: 每次commit后必须更新本文档  
**格式**: 北京时间 + Commit哈希 + 文件清单 + 动机说明
