# Experiments 实验脚本文件夹

本文件夹用于存放项目开发过程中的各类实验脚本。

## 📁 文件夹用途

- 存放用于参数选择、模型验证、性能测试等实验性脚本
- 与 `tests/` 文件夹区别：
  - `tests/`: 单元测试、集成测试（持续验证代码正确性）
  - `experiments/`: 探索性实验（一次性或间歇性运行）

## 📝 当前实验

### 1. marginal_distribution_experiment.py
**目的**: 为Core模块选择最佳的边际分布类型

**运行方式**:
```powershell
# 1. 激活虚拟环境
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1

# 2. 进入项目目录
cd D:\Python\2025DaChuang\Simulation_project_v2

# 3. 运行实验
python experiments/marginal_distribution_experiment.py
```

**输出**:
- 控制台：详细的拟合结果和AIC/BIC对比
- 文件：`data/output/best_distributions.txt`

**依赖数据**:
- `data/input/cleaned_data.csv` - 清洗后的调研数据

---

## 📊 实验结果管理

实验结果统一保存到：
- 数据输出：`data/output/`
- 图表：`results/figures/`
- 报告：`results/reports/`

---

**创建时间**: 2025-09-30
