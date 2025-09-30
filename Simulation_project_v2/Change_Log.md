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

## 下一步计划

- [ ] **用户审阅Core模块开发文档**
- [ ] 根据反馈调整设计
- [ ] 实现Core模块代码（预计3.5天）
- [ ] 编写单元测试（覆盖率>90%）

---

**文档维护规则**: 每次commit后必须更新本文档  
**格式**: 北京时间 + Commit哈希 + 文件清单 + 动机说明
