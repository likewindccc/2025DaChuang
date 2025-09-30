# Simulation_project_v2 项目架构总结

**创建日期**: 2025-09-30  
**状态**: 架构设计完成，待开始开发

---

## 📊 项目概览

**项目名称**: 农村女性就业市场MFG模拟系统 v2.0  
**理论框架**: 平均场博弈(MFG) + 基于主体建模(ABM)  
**开发周期**: 预计6个月（22周 + 2周缓冲）  
**代码规模**: 预计 12,500-16,300 行

---

## 🎯 核心目标

1. **完全重构旧版代码**，消除重复和混乱
2. **严格遵守PEP8**和项目规则
3. **Numba强制优化**热点代码（目标加速>10x）
4. **模块化设计**，清晰的职责划分
5. **完整的测试**和文档体系

---

## 🏗️ 项目结构

```
Simulation_project_v2/
├── src/                    # 源代码
│   ├── core/              # 基础类与数据结构
│   ├── modules/           # 5个核心模块
│   │   ├── population/   # Module 1: 虚拟主体生成
│   │   ├── matching/     # Module 2: 匹配引擎
│   │   ├── estimation/   # Module 3: 匹配函数估计
│   │   ├── mfg/          # Module 4: MFG求解器
│   │   └── calibration/  # Module 5: 参数校准
│   └── utils/             # 工具函数
│
├── config/                 # 配置文件（YAML）
├── data/                   # 数据文件
├── results/                # 结果输出
├── tests/                  # 测试代码
└── docs/                   # 完整文档
    ├── userdocs/          # 用户文档
    ├── developerdocs/     # 开发者文档
    └── academicdocs/      # 学术文档
```

---

## 📚 核心文档清单

### 已完成的架构文档（✅）

| 文档 | 路径 | 说明 |
|------|------|------|
| **架构设计** | `docs/developerdocs/architecture.md` | 完整的系统架构，5个模块详解 |
| **技术选型** | `docs/developerdocs/tech_stack.md` | 依赖库、Numba优化规范、性能目标 |
| **开发路线图** | `docs/developerdocs/roadmap.md` | 7个Phase详细计划，22周时间表 |
| **代码规范** | `docs/developerdocs/coding_standards.md` | PEP8规范、命名约定、测试规范 |
| **用户手册** | `docs/userdocs/user_manual.md` | 安装、使用指南 |
| **学术文档框架** | `docs/academicdocs/` | 方法论、算法、实验（待完善） |

### 配置文件（✅）

| 配置文件 | 说明 |
|---------|------|
| `config/default/base_config.yaml` | 基础配置 |
| `config/default/population.yaml` | Module 1配置 |
| `config/default/matching.yaml` | Module 2配置 |
| `config/default/estimation.yaml` | Module 3配置 |
| `config/default/mfg.yaml` | Module 4配置 |
| `config/default/calibration.yaml` | Module 5配置 |
| `config/experiments/baseline.yaml` | 基准实验配置 |

### 其他关键文件（✅）

- `README.md` - 项目说明
- `Change_Log.md` - 修改日志（符合项目规则）
- `.gitignore` - Git忽略规则
- `setup_directories.py` - 目录初始化脚本

---

## 🔬 核心模块详解

### Module 1: Population Generator (虚拟主体生成)
- **劳动力生成器**: Copula理论 + 边际分布估计
- **企业生成器**: 四维正态分布（参数通过校准确定）
- **数据来源**: 劳动力来自CHARLS数据，企业来自文献

### Module 2: Matching Engine (匹配引擎)
- **算法**: Gale-Shapley稳定匹配
- **偏好计算**: 加权距离法
- **Numba优化**: 偏好矩阵并行计算（目标加速>15x）

### Module 3: Match Function Estimator (匹配函数估计)
- **数据生成**: ABM多轮模拟（多θ、多a扰动）
- **估计方法**: Logit回归
- **输出**: Numba优化的匹配函数 λ(x, σ, a, θ)

### Module 4: MFG Simulator (MFG求解器) 🔥 核心
- **算法**: 贝尔曼方程 + KFE演化
- **简化策略**: 50×50离散网格，有限期，固定迭代上限
- **收敛判断**: 三重标准（值函数、策略、θ）

### Module 5: Calibration (参数校准)
- **方法**: 遗传算法（DEAP库）
- **参数空间**: 17维（企业参数 + MFG参数）
- **目标函数**: 多指标加权（失业率、匹配率等）
- **预计耗时**: 4-8小时（种群100，代数50）

---

## ⚡ 性能优化策略

### Numba强制优化的函数

| 函数 | 加速目标 | 调用频率 |
|------|---------|---------|
| 匹配函数 λ | > 20x | 百万次/迭代 |
| 偏好矩阵计算 | > 15x | 每轮匹配 |
| 贝尔曼迭代 | > 10x | 数百次 |
| KFE演化步骤 | > 10x | 数百次 |

### 大规模计算目标

- **10K劳动力 × 5K企业**
  - 虚拟个体生成: < 5秒
  - 单轮匹配: < 30秒
  - 完整MFG模拟: < 10分钟
  - 校准（50代）: < 8小时

---

## 📅 开发计划

### Phase 1: 基础框架（Week 1-2） ✅ 架构完成
- [x] 目录结构
- [x] 完整架构文档
- [ ] 核心基类实现（下一步）

### Phase 2: Module 1（Week 3-5）
- [ ] Copula引擎重构
- [ ] 劳动力生成器
- [ ] 企业生成器

### Phase 3: Module 2+3（Week 6-9）
- [ ] Gale-Shapley算法
- [ ] ABM数据生成
- [ ] Logit回归估计

### Phase 4: Module 4（Week 10-15） 🔥 最复杂
- [ ] 状态空间设计
- [ ] 贝尔曼求解器
- [ ] KFE求解器
- [ ] MFG主循环

### Phase 5: Module 5（Week 16-18）
- [ ] 遗传算法实现
- [ ] 目标函数设计
- [ ] 参数校准实验

### Phase 6: 集成优化（Week 19-20）
- [ ] 端到端测试
- [ ] 性能优化
- [ ] Bug修复

### Phase 7: 文档完善（Week 21-22）
- [ ] API文档生成
- [ ] 学术文档编写
- [ ] 最终审查

---

## 🛠️ 技术栈

### 核心依赖
- **Python**: 3.12.5
- **NumPy**: 1.26.4（数组计算）
- **Pandas**: 2.2.3（数据处理）
- **Numba**: ≥0.59.0（JIT加速）⭐
- **Copulas**: 0.12.3（Copula建模）
- **DEAP**: ≥1.4.1（遗传算法）
- **Statsmodels**: 0.14.4（Logit回归）

### 测试与文档
- **Pytest**: 8.3.3（单元测试）
- **Pytest-Cov**: 5.0.0（覆盖率）
- **Sphinx**: 8.0.2（API文档生成）

---

## ✅ 设计优势（相比旧版）

1. **消除代码重复**
   - 旧版: `copula_agent_generator.py`(1213行) 与 `labor_generator.py`(748行) 功能重复80%
   - 新版: 统一的Copula引擎，复用性高

2. **职责清晰**
   - 旧版: 数据分析模块包含虚拟个体生成（职责混乱）
   - 新版: 严格的模块边界，单一职责原则

3. **强制性能优化**
   - 旧版: 部分优化，不系统
   - 新版: Numba强制优化 + 性能测试验收

4. **完整的测试体系**
   - 旧版: 测试不完整
   - 新版: 单元测试 + 集成测试 + 性能基准测试

5. **文档齐全**
   - 旧版: 文档分散
   - 新版: 用户/开发者/学术 三类文档完整

---

## ⚠️ 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| MFG不收敛 | 高 | 高 | 简化状态空间，固定迭代上限 |
| 校准时间过长 | 高 | 中 | 并行计算，减少代数 |
| Numba兼容性问题 | 中 | 中 | 保留纯Python备选方案 |

---

## ⚠️ 重要提醒：虚拟环境

本项目使用**项目专属虚拟环境**: `D:\Python\2025大创\venv`

```bash
# ✅ 正确：使用项目虚拟环境
D:\Python\2025大创\venv\Scripts\Activate.ps1
# 或使用快捷脚本
activate_env.bat

# ❌ 错误：不要使用全局环境
D:\Python\.venv\Scripts\Activate.ps1
```

详见：`ENVIRONMENT.md` 文档

---

## 📌 下一步行动

1. **用户审阅本文档和所有架构文档**
2. **确认架构设计符合预期**
3. **开始Phase 1开发**: 实现核心基类
4. **建立Git仓库**并进行首次提交

---

## 📞 文档索引

快速跳转到关键文档：

- 📐 **架构设计**: [`docs/developerdocs/architecture.md`](docs/developerdocs/architecture.md)
- 🔧 **技术选型**: [`docs/developerdocs/tech_stack.md`](docs/developerdocs/tech_stack.md)
- 🗺️ **开发路线**: [`docs/developerdocs/roadmap.md`](docs/developerdocs/roadmap.md)
- 📏 **代码规范**: [`docs/developerdocs/coding_standards.md`](docs/developerdocs/coding_standards.md)
- 📖 **用户手册**: [`docs/userdocs/user_manual.md`](docs/userdocs/user_manual.md)
- 📝 **修改日志**: [`Change_Log.md`](Change_Log.md)

---

**创建者**: AI Agent (Claude Sonnet 4.5)  
**审阅者**: 项目负责人（待审阅）  
**最后更新**: 2025-09-30

---

## ✨ 致谢

感谢您对项目规则的明确定义，这使得架构设计能够严格符合您的期望：
- ✅ 用户第一原则
- ✅ 需求确定原则
- ✅ 模块化与简洁原则
- ✅ 代码易读性原则
- ✅ PEP8规范
- ✅ Git + 日志混合方案
- ✅ 环境激活原则

**期待您的审阅反馈！**
