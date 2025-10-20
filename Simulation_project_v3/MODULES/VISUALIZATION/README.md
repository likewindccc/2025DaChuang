# VISUALIZATION 可视化模块

## 模块简介

本模块提供完整的可视化解决方案，支持：
- **静态图表**（Matplotlib/Seaborn）：适合论文、报告
- **交互式图表**（Plotly）：支持缩放、悬停、筛选
- **整合仪表盘**（Plotly Dash）：多图表联动展示

## 模块结构

```
VISUALIZATION/
├── __init__.py                    # 模块初始化
├── style_config.py                # 样式配置（颜色、字体、DPI）
├── data_visualizer.py             # 数据可视化器
├── mfg_visualizer.py              # MFG模块可视化器
├── calibration_visualizer.py      # Calibration模块可视化器
├── simulation_visualizer.py       # Simulation模块可视化器
├── dashboard_builder.py           # 仪表盘构建器
└── README.md                      # 本文件
```

## 快速开始

### 1. 初始化样式

```python
from MODULES.VISUALIZATION import initialize_styles

# 初始化Matplotlib和Plotly样式
initialize_styles()
```

### 2. 数据可视化

```python
from MODULES.VISUALIZATION import DataVisualizer
import pandas as pd

# 创建可视化器
visualizer = DataVisualizer(output_dir='OUTPUT')

# 绘制初始分布（静态 + 交互式）
static_path, interactive_path = visualizer.plot_initial_distribution(individuals)

# 绘制Copula结构
copula_path = visualizer.plot_copula_structure(individuals)

# 劳动者-企业对比
comparison_path = visualizer.plot_laborer_enterprise_comparison(laborers, enterprises)
```

### 3. MFG可视化

```python
from MODULES.VISUALIZATION import MFGVisualizer

visualizer = MFGVisualizer(output_dir='OUTPUT')

# 收敛曲线
convergence_path = visualizer.plot_convergence_curves(convergence_history)

# 价值函数热力图
value_path = visualizer.plot_value_function_heatmap(individuals, value_type='V_U')

# 最优努力分布
effort_path = visualizer.plot_optimal_effort_distribution(individuals)

# 人口演化
evolution_path = visualizer.plot_population_evolution(population_history, variable='T')

# 3D交互式价值函数
value_3d_path = visualizer.create_interactive_value_function_3d(individuals, 'V_U')
```

### 4. Calibration可视化

```python
from MODULES.VISUALIZATION import CalibrationVisualizer

visualizer = CalibrationVisualizer(output_dir='OUTPUT')

# 目标函数历史
objective_path = visualizer.plot_objective_history(objective_history)

# 参数收敛轨迹
param_path = visualizer.plot_parameter_traces(parameter_history, param_names)

# 矩拟合质量
moment_path = visualizer.plot_moment_fit(target_moments, simulated_moments)

# 交互式参数空间
space_path = visualizer.create_interactive_parameter_space(parameter_history)
```

### 5. Simulation可视化

```python
from MODULES.VISUALIZATION import SimulationVisualizer

visualizer = SimulationVisualizer(output_dir='OUTPUT')

# 政策效果对比
comparison_path = visualizer.plot_policy_comparison(policy_results)

# 时间序列演化
time_series_path = visualizer.plot_time_series(time_series_data, metric='unemployment_rate')

# 成本-收益分析
cost_benefit_path = visualizer.plot_cost_benefit(policy_data)

# 交互式政策雷达图
radar_path = visualizer.create_interactive_policy_radar(policy_results)
```

### 6. 仪表盘构建

```python
from MODULES.VISUALIZATION import DashboardBuilder

builder = DashboardBuilder(output_dir='OUTPUT')

# MFG均衡仪表盘
mfg_dashboard = builder.build_mfg_dashboard(individuals, convergence_history)

# 校准过程仪表盘
calibration_dashboard = builder.build_calibration_dashboard(
    objective_history,
    parameter_history,
    moment_fit
)

# 政策仿真仪表盘
policy_dashboard = builder.build_policy_dashboard(policy_results, time_series)
```

## 输出结构

```
OUTPUT/
├── figures/                    # 静态图（PNG，300 DPI）
│   ├── data/
│   ├── mfg/
│   ├── calibration/
│   └── simulation/
├── interactive/                # 交互式图表（HTML）
│   ├── data/
│   ├── mfg/
│   ├── calibration/
│   └── simulation/
└── dashboards/                 # 整合仪表盘（HTML）
    ├── DASHBOARD_mfg_equilibrium.html
    ├── DASHBOARD_calibration.html
    └── DASHBOARD_policy_simulation.html
```

## 样式配置

### 颜色方案

模块使用与网站一致的颜色方案：

- **主色调**：紫色 `#8b5cf6`、蓝色 `#3b82f6`
- **强调色**：粉色 `#ec4899`、绿色 `#10b981`、橙色 `#f59e0b`
- **政策颜色**：6种区分色（基准、政策A-E）

### DPI设置

- 屏幕显示：150 DPI
- 论文打印：300 DPI
- 演示文稿：200 DPI

### 字体配置

- 中文：SimHei（黑体）
- 英文：Arial
- 标题字号：16pt
- 轴标签字号：12pt

## 与网站集成

### 嵌入Plotly图表到网站

```python
# 1. 生成交互式图表
from MODULES.VISUALIZATION import MFGVisualizer

visualizer = MFGVisualizer(output_dir='OUTPUT')
html_path = visualizer.create_interactive_value_function_3d(individuals, 'V_U')

# 2. 在网站中嵌入（复制生成的HTML文件到WEBSITE目录）
# 3. 在对应的页面中使用<iframe>标签引用：
# <iframe src="MFG_value_function_V_U_3D.html" width="100%" height="700px"></iframe>
```

### 更新图表占位符

1. 使用可视化器生成HTML文件
2. 将文件复制到`WEBSITE/charts/`目录
3. 修改对应页面的`.chart-placeholder`为`<iframe>`标签

## 高级功能

### 自定义颜色

```python
from MODULES.VISUALIZATION import COLORS, get_color_palette

# 使用预定义颜色
my_color = COLORS['primary']

# 生成调色板
gradient_colors = get_color_palette(n_colors=5, palette_type='gradient')
policy_colors = get_color_palette(n_colors=6, palette_type='policy')
```

### 自定义图表尺寸

```python
from MODULES.VISUALIZATION import FIGURE_SIZE

# 使用预定义尺寸
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])

# 自定义尺寸
fig, ax = plt.subplots(figsize=(14, 8))
```

## 测试示例

运行完整的测试示例：

```bash
cd 2025DaChuang/Simulation_project_v3
python TESTS/test_visualization_example.py
```

## 依赖包

- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0

## 常见问题

### Q1: 中文显示乱码？
**A**: 确保已安装SimHei字体，或在`style_config.py`中修改字体为系统已安装的中文字体。

### Q2: 图表保存DPI过低？
**A**: 使用`plt.savefig(path, dpi=300)`或调整`style_config.py`中的`DPI_CONFIG`。

### Q3: Plotly图表在浏览器中打不开？
**A**: 确保HTML文件路径正确，且浏览器允许本地文件访问。

### Q4: 如何导出PDF格式？
**A**: 将`plt.savefig(path, dpi=300, bbox_inches='tight')`的后缀改为`.pdf`。

## 贡献指南

如需添加新的可视化功能：

1. 在对应的可视化器类中添加新方法
2. 遵循现有命名规范（`plot_xxx` 或 `create_xxx`）
3. 添加详细的docstring
4. 在测试示例中添加用法演示
5. 更新本README文档

## 版本历史

- v1.0.0 (2025/10/20): 初始版本，支持Data、MFG、Calibration、Simulation四大模块可视化

---

**作者**: 项目组  
**最后更新**: 2025/10/20  
**联系方式**: project@example.com

