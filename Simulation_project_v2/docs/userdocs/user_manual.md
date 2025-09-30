# 用户手册 (User Manual)

**项目**: Simulation_project_v2  
**版本**: 2.0  
**日期**: 2025-09-30

---

## 📋 目录

- [1. 快速开始](#1-快速开始)
- [2. 安装与环境配置](#2-安装与环境配置)
- [3. 基本使用](#3-基本使用)
- [4. 高级功能](#4-高级功能)
- [5. 常见问题](#5-常见问题)

---

## 1. 快速开始

### 1.1 5分钟快速体验

```bash
# 1. 进入项目目录
cd D:\Python\2025大创\Simulation_project_v2

# 2. 激活项目虚拟环境（重要！使用项目专属环境）
D:\Python\2025大创\venv\Scripts\Activate.ps1
# 或直接运行激活脚本
activate_env.bat

# 3. 运行基准模拟
python -m src.main --config config/experiments/baseline.yaml --mode simulation

# 4. 查看结果
# 结果保存在 results/reports/ 和 results/figures/
```

### 1.2 项目功能概览

本系统提供三种运行模式：

1. **simulation**: 运行MFG模拟（使用给定参数）
2. **calibration**: 参数校准（遗传算法）
3. **policy**: 政策分析（对比不同政策效果）

---

## 2. 安装与环境配置

### 2.1 系统要求

- **操作系统**: Windows 10/11, Linux
- **Python版本**: 3.12.5 (推荐) 或 3.11.x
- **内存**: 至少8GB RAM
- **磁盘**: 至少10GB可用空间

### 2.2 环境配置

**方式一: 使用项目虚拟环境**（推荐）

```powershell
# 激活项目专属虚拟环境（重要！）
D:\Python\2025大创\venv\Scripts\Activate.ps1

# 或使用快捷脚本
cd D:\Python\2025大创\Simulation_project_v2
activate_env.bat

# 安装依赖（如果尚未安装）
pip install -r requirements.txt
```

**方式二: 创建新虚拟环境**

```bash
# 创建虚拟环境
python -m venv venv_simulation

# 激活
venv_simulation\Scripts\activate  # Windows
source venv_simulation/bin/activate  # Linux

# 安装依赖
pip install -r requirements.txt
```

### 2.3 验证安装

```bash
# 运行测试
pytest tests/unit/ -v

# 预期输出：所有测试通过
```

---

## 3. 基本使用

### 3.1 运行基准模拟

**步骤**:

1. **准备数据** （可选，系统会使用预设数据）

将劳动力调研数据放在 `data/input/labor_survey.csv`

2. **配置参数**

编辑 `config/experiments/baseline.yaml`：

```yaml
simulation:
  n_labor: 10000
  n_enterprise: 5000
  random_seed: 42

mfg:
  grid_size: [50, 50]
  max_iterations: 500
  tolerance: 1e-6
```

3. **运行模拟**

```bash
python -m src.main \
    --config config/experiments/baseline.yaml \
    --mode simulation
```

4. **查看结果**

- 日志: `results/logs/simulation_20250930_143000.log`
- 报告: `results/reports/simulation_report.md`
- 图表: `results/figures/`

### 3.2 参数校准

**警告**: 校准过程耗时较长（4-8小时）

```bash
python -m src.main \
    --config config/default/calibration.yaml \
    --mode calibration
```

**校准结果**:
- 校准后参数: `data/output/calibrated_parameters.yaml`
- 校准报告: `results/reports/calibration_report.md`
- 收敛曲线: `results/figures/calibration_convergence.png`

### 3.3 政策分析

对比不同政策效果：

```bash
python -m src.main \
    --config config/experiments/policy_a.yaml \
    --mode policy
```

---

## 4. 高级功能

### 4.1 自定义配置

创建自己的配置文件 `config/experiments/my_experiment.yaml`：

```yaml
# 继承默认配置
base: config/default/base_config.yaml

# 覆盖参数
simulation:
  n_labor: 20000      # 增加劳动力数量
  n_enterprise: 10000

mfg:
  grid_size: [100, 100]  # 更精细的网格
  
# 自定义实验参数
experiment:
  name: "大规模模拟"
  description: "测试系统在大规模场景下的表现"
```

### 4.2 批量实验

运行多个配置：

```bash
# 运行所有实验配置
for config in config/experiments/*.yaml; do
    python -m src.main --config $config --mode simulation
done
```

### 4.3 结果可视化

使用内置可视化工具：

```python
from src.utils.visualization import plot_equilibrium, plot_distributions

# 加载结果
equilibrium = load_equilibrium('results/equilibrium.pkl')

# 绘制图表
plot_equilibrium(equilibrium, save_path='results/figures/')
plot_distributions(equilibrium, save_path='results/figures/')
```

---

## 5. 常见问题

### 5.1 安装问题

**Q: pip安装依赖失败**

A: 尝试使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: Numba安装失败**

A: 确保Python版本正确（3.11-3.12），某些版本Numba不兼容

### 5.2 运行问题

**Q: 模拟不收敛**

A: 
1. 检查参数是否合理（MFG配置）
2. 增加 `max_iterations`
3. 放宽 `tolerance`
4. 减小网格规模 `grid_size`

**Q: 内存不足（MemoryError）**

A:
1. 减少 `n_labor` 和 `n_enterprise`
2. 减小 `grid_size`
3. 增加物理内存或使用更大的机器

**Q: 运行很慢**

A:
1. 确认Numba已正确安装（首次运行会编译）
2. 检查CPU占用率（校准时应该接近100%）
3. 使用性能分析工具定位瓶颈

### 5.3 结果问题

**Q: 失业率异常（如>50%）**

A: 
1. 检查企业参数是否合理
2. 检查匹配函数参数
3. 考虑重新校准

**Q: 匹配率为0**

A:
1. 检查劳动力和企业特征分布是否重叠
2. 检查偏好函数计算逻辑

---

## 6. 附录

### 6.1 配置参数完整列表

详见: `docs/userdocs/configuration_guide.md`

### 6.2 输出文件说明

| 文件 | 说明 |
|------|------|
| `virtual_labor_pool.csv` | 虚拟劳动力数据 |
| `virtual_enterprise_pool.csv` | 虚拟企业数据 |
| `matching_data.csv` | 匹配结果数据 |
| `equilibrium.pkl` | MFG均衡（Python对象） |
| `calibrated_parameters.yaml` | 校准后的参数 |

### 6.3 技术支持

- **文档**: `docs/`
- **Issue**: (GitHub链接)
- **Email**: (联系邮箱)

---

**文档维护**: 随功能更新  
**最后更新**: 2025-09-30
