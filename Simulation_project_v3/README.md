# 农村女性就业市场MFG模拟系统 v3.0

**项目状态**: 核心开发阶段
**版本**: 3.0-beta
**创建日期**: 2025-10-08

---

## 项目简介

基于平均场博弈(MFG)与基于主体建模(ABM)的农村女性就业市场动态模拟系统。

**核心研究问题**:

1. 农村女性如何动态调整努力水平以提高就业机会？
2. 个体决策与宏观市场状态如何相互作用？
3. 如何设计最优政策组合促进农村女性就业？

---

## 项目结构

```
Simulation_project_v3/
├── CONFIG/        # 配置文件
├── MODULES/       # 5个核心模块
│   ├── POPULATION/    # 人口分布模块（已完成）
│   ├── LOGISTIC/      # 匹配与匹配函数模块（已完成）
│   ├── MFG/           # 平均场博弈模块（核心完成）
│   ├── SIMULATOR/     # 模拟器模块（待开发）
│   └── CALIBRATION/   # 参数校准模块（待开发）
├── DATA/          # 数据文件
├── OUTPUT/        # 输出结果
├── DOCS/          # 项目文档
└── TESTS/         # 测试代码（15个测试脚本）
```

---

## 文档导航

- **[用户需求确认文档](DOCS/用户需求确认文档.md)** - 详细需求和架构设计
- **[修改日志](DOCS/Change_Log.md)** - 17次修改的完整记录
- **[KFE实现方法说明](DOCS/KFE实现方法说明.md)** - KFE求解器技术文档
- **[MFG值函数修正方案](DOCS/MFG值函数修正方案.md)** - 值函数计算方法
- **[研究计划](DOCS/原始研究计划/研究计划.md)** - 理论基础和算法流程

---

## 当前状态

### 已完成模块

- **POPULATION（人口分布）** - 100%

  - 劳动力分布建模（Gaussian Copula + 经验分布）
  - 企业分布参数配置
  - 参数保存和加载功能
- **LOGISTIC（匹配与匹配函数）** - 100%

  - 虚拟市场生成器
  - GS匹配算法（Numba加速，max_rounds=32）
  - 匹配函数回归（Logit模型，6变量）
  - 匹配率约50%（theta=1.0时）
- **MFG（平均场博弈）** - 85%

  - Bellman方程求解器（Numba加速值迭代）
  - KFE演化求解器（基于个体的蒙特卡洛）
  - 均衡求解器（交替迭代主控制）
  - 离职率标准化（消除两极分化）
  - 当前问题：收敛性待优化

### 进行中

- MFG模块收敛性优化
  - 已实施：降低贴现因子、添加阻尼更新、使用相对收敛阈值
  - 测试中：验证优化效果

### 待开始

- SIMULATOR模块 - 市场模拟和政策分析
- CALIBRATION模块 - 参数校准（SMM方法）

---

## 快速开始

### 环境要求

- Python 3.10+
- 依赖库见 `requirements.txt`
- Numba JIT编译器（性能加速）

### 安装

```bash
# 1. 激活虚拟环境
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 测试POPULATION模块
python TESTS/test_population.py

# 测试LOGISTIC模块
python TESTS/test_logistic_market.py
python TESTS/test_match_function.py

# 测试MFG模块
python TESTS/test_equilibrium_solver.py
```

### 完整运行

```python
# Python脚本示例
from MODULES.MFG import solve_equilibrium

# 求解MFG均衡
individuals_eq, eq_info = solve_equilibrium()

print(f"是否收敛: {eq_info['converged']}")
print(f"最终失业率: {eq_info['final_unemployment_rate']*100:.2f}%")
```

---

## 技术亮点

### Numba加速

- GS匹配算法：3-5倍加速
- Bellman求解器：10-30倍加速（parallel=True）
- KFE演化：10-30倍加速（parallel=True）

### 核心算法

- Gaussian Copula建模（连续变量联合分布）
- Gale-Shapley稳定匹配
- Logit回归（匹配函数）
- 值迭代（Bellman方程）
- 蒙特卡洛人口演化（KFE）

### 参数优化

- max_rounds调优：10 → 32（匹配率提升到50%）
- 离职率标准化：消除两极分化（中位数0% → 2.94%）
- MinMax标准化：解决偏好集中度问题

---

## 设计理念

### v3版本目标

- **简洁性**: 消除v2的过度复杂设计
- **清晰性**: 明确的模块职责划分
- **严格性**: 严格遵守项目开发规则
- **可维护性**: 易于理解和扩展

### 与v2的主要区别

- 扁平化目录结构
- 简化的状态空间设计（基于个体的蒙特卡洛）
- 清晰的模块依赖关系
- 优先实用性而非完美性

---

## 项目信息

**负责人**: 李心泠
**指导教师**: 李三希、林琳
**研究方向**: 劳动经济学、平均场博弈

---

## 许可证

Academic Research Use Only

---

**最后更新**: 2025-10-17
