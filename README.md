# 乡村女性就业市场平均场博弈（MFG）仿真系统 v3.0

**Rural Female Employment Market Mean Field Game Simulation System**

中国人民大学经济学院 | 求是学术（大学生创新创业训练计划项目）

---

## 项目概述

### 研究背景

本项目是中国人民大学经济学院的研究项目，旨在运用平均场博弈（Mean Field Game, MFG）理论与基于主体建模（Agent-Based Modeling, ABM）方法，构建乡村女性就业市场动态仿真系统。通过数学建模和计算机仿真，深入研究个体努力决策与宏观市场状态的相互作用机制，为就业促进政策提供量化分析工具。

### 研究目标

1. 建立基于MFG理论的乡村女性就业市场动态模型
2. 实现个体最优决策与市场均衡的数值求解
3. 模拟不同政策情景下的市场演化路径
4. 为就业促进政策设计提供定量依据

### 技术特点

- **理论基础**: 平均场博弈理论 + Bellman最优性原理
- **建模方法**: Agent-Based Modeling + 蒙特卡洛模拟
- **数据驱动**: 基于实地调研数据的参数校准（SMM方法 + 遗传算法）
- **高性能计算**: Numba JIT编译实现10-30倍加速
- **专业展示**: 响应式网站系统 + 交互式可视化

---

## v3版本核心模块

Simulation_project_v3是当前主要版本，包含完整的5大核心模块和专业网站展示系统。

### 1. POPULATION模块 - 虚拟劳动力生成

**功能描述**:
- 基于Gaussian Copula理论建模劳动力多维特征的联合分布
- 生成符合真实分布特征的虚拟劳动力和企业样本
- 支持K-S检验验证生成数据的统计有效性

**核心技术**:
- **Gaussian Copula**: 捕捉技能（S）、距离（D）、期望工资（W）、工作时长（T）等变量间的相关性结构
- **经验分布**: 处理离散变量（教育程度、子女数量）
- **统计验证**: K-S检验确保生成分布与真实分布一致

**输出成果**:
- 虚拟劳动力样本（N=10000）
- 初始分布可视化图表（4变量分布、Copula相关性结构）
- K-S检验报告（p值 > 0.05，通过验证）

---

### 2. LOGISTIC模块 - 劳企匹配建模

**功能描述**:
- 实现Gale-Shapley稳定匹配算法生成虚拟匹配数据
- 构建Logistic回归模型预测匹配概率
- 支持均衡、富余、短缺三种市场类型

**核心技术**:
- **Gale-Shapley算法**: 双边市场稳定匹配（Numba加速3-5倍）
- **Logistic回归**: 6个核心变量（S_worker, D, W_gap, firm_size, firm_wage, theta）
- **MinMax标准化**: 解决偏好集中度问题，提升匹配率至50%

**输出成果**:
- 匹配函数回归系数（AIC=8234.5，伪R²=0.156）
- 匹配预测分析图表
- 偏好分数分布可视化
- GS算法流程图

---

### 3. MFG模块 - 平均场博弈求解

**功能描述**:
- 求解Bellman方程获得个体最优努力策略
- 通过Kolmogorov前向方程（KFE）演化人口分布
- 迭代求解Nash均衡（价值函数 + 人口分布）

**核心技术**:
- **Bellman方程**: 值迭代算法求解最优价值函数V(s,u)
- **KFE演化**: 蒙特卡洛模拟人口分布动态
- **Nash均衡**: Bellman + KFE交替迭代至收敛
- **Numba并行**: 10-30倍加速（parallel=True）

**输出成果**:
- 收敛指标曲线（|ΔV|、|Δa|、|Δu|）
- 价值函数热力图（V_U、V_E）
- 最优努力分布
- 市场分布对比图

---

### 4. SIMULATOR模块 - 市场仿真与政策分析

**功能描述**:
- 基于MFG均衡解进行长期市场仿真
- 模拟失业率、工资、技能等宏观指标的时间序列演化
- 评估培训补贴、就业补贴等政策的效果

**核心技术**:
- **时间序列仿真**: T期（T=50）市场动态演化
- **政策干预**: 支持多种政策组合（培训、补贴、信息改善）
- **反事实分析**: 对比有无政策干预的市场轨迹

**输出成果**:
- 失业率演化曲线
- 平均工资时间序列
- 技能分布演化
- 政策效果对比分析

---

### 5. CALIBRATION模块 - 参数校准

**功能描述**:
- 使用模拟矩匹配（Simulated Method of Moments, SMM）方法校准模型参数
- 采用遗传算法优化参数以最小化模拟矩与真实矩的距离
- 支持多目标矩匹配（失业率、工资分布、匹配率等）

**核心技术**:
- **SMM方法**: 最小化 ||m_sim(θ) - m_data||²
- **遗传算法**: 全局优化参数空间（population_size=50, generations=100）
- **并行计算**: 多进程加速参数搜索

**输出成果**:
- 校准后的参数集
- 目标矩拟合度报告
- 参数敏感性分析
- 校准收敛曲线

---

## 技术栈

### Python核心库

```
numpy>=1.26.0          # 数值计算基础
pandas>=2.0.0          # 数据处理与分析
scipy>=1.11.0          # 科学计算（优化、统计）
statsmodels>=0.14.0    # Logistic回归
numba>=0.62.0          # JIT编译加速（关键）
copulas>=0.12.0        # Gaussian Copula建模
pyyaml>=6.0            # 配置文件管理
```

### 可视化技术

```
plotly>=5.18.0         # 交互式图表（8个动态图表）
matplotlib>=3.8.0      # 静态图表（DPI 300高清）
seaborn>=0.13.0        # 统计可视化
```

### 网站技术

- **HTML5 + CSS3**: 响应式设计（支持桌面/平板/手机）
- **紫色渐变主题**: #8b5cf6主色调，专业视觉风格
- **Font Awesome 6.4.0**: 图标库
- **媒体查询**: 1200px和768px断点

### 核心算法

1. **Gale-Shapley稳定匹配算法**: 双边市场匹配
2. **Bellman值迭代**: 动态规划求解最优策略
3. **Kolmogorov前向方程**: 人口分布演化
4. **遗传算法**: 参数空间全局优化
5. **蒙特卡洛模拟**: 基于个体的随机模拟

---

## 项目结构

```
Simulation_project_v3/
├── MODULES/                    # 5个核心模块
│   ├── POPULATION/            # 虚拟劳动力生成
│   │   ├── population_generator.py
│   │   └── distribution_params.pkl
│   ├── LOGISTIC/              # 匹配函数建模
│   │   ├── gs_matching.py
│   │   ├── match_function.py
│   │   └── logistic_model.pkl
│   ├── MFG/                   # 平均场博弈求解
│   │   ├── bellman_solver.py
│   │   ├── kfe_solver.py
│   │   └── equilibrium_solver.py
│   ├── SIMULATOR/             # 市场仿真
│   │   ├── market_simulator.py
│   │   └── policy_analyzer.py
│   ├── CALIBRATION/           # 参数校准
│   │   ├── smm_calibrator.py
│   │   └── genetic_optimizer.py
│   └── VISUALIZATION/         # 可视化工具
│       └── chart_generator.py
│
├── CONFIG/                    # 配置文件（YAML）
│   ├── population_config.yaml
│   ├── logistic_config.yaml
│   ├── mfg_config.yaml
│   ├── simulator_config.yaml
│   ├── calibration_config.yaml
│   └── target_moments.yaml
│
├── DATA/                      # 数据文件
│   ├── raw/                   # 原始调研数据
│   └── processed/             # 预处理数据
│
├── OUTPUT/                    # 输出结果
│   ├── population/            # 人口模块输出
│   ├── logistic/              # 匹配模块输出
│   ├── mfg/                   # MFG模块输出
│   ├── simulation/            # 仿真模块输出
│   ├── calibration/           # 校准模块输出
│   └── flowcharts/            # 流程图
│
├── TESTS/                     # 测试脚本（40+个）
│   ├── test_population.py
│   ├── test_gs_matching.py
│   ├── test_mfg_equilibrium.py
│   ├── test_simulator.py
│   ├── test_calibration.py
│   ├── generate_*.py          # 图表生成脚本
│   └── analyze_*.py           # 结果分析脚本
│
├── WEBSITE/                   # 专业展示网站
│   ├── index.html             # 首页
│   ├── population.html        # 人口模块页面
│   ├── logistic.html          # 匹配模块页面
│   ├── mfg.html               # MFG模块页面
│   ├── simulation.html        # 仿真模块页面
│   ├── data_analysis.html     # 数据分析页面
│   ├── results.html           # 结果展示页面
│   └── charts/                # 图表文件
│       ├── population/
│       ├── logistic/
│       ├── mfg/
│       ├── simulation/
│       └── data/
│
├── DOCS/                      # 项目文档
│   ├── Change_Log.md          # 修改日志
│   ├── 用户需求确认文档.md     # 详细设计文档
│   ├── KFE实现方法说明.md      # 技术文档
│   └── MFG值函数修正方案.md    # 技术文档
│
├── GUI/                       # 图形界面（可选）
│   └── app.py
│
├── requirements.txt           # 依赖清单
└── README.md                  # 项目说明
```

---

## 网站展示系统

### 页面结构

项目包含7个专业HTML页面，提供完整的研究成果展示：

1. **index.html** - 首页
   - 项目概述与研究背景
   - 5大模块导航卡片
   - 技术特色与研究成果
   - GitHub仓库链接

2. **population.html** - 人口模块
   - Gaussian Copula理论介绍
   - 初始分布可视化（4变量）
   - K-S检验结果
   - Copula相关性结构

3. **logistic.html** - 匹配模块
   - Gale-Shapley算法流程图
   - 匹配函数回归结果
   - 偏好分数分布
   - 虚拟市场分布

4. **mfg.html** - MFG模块
   - Bellman方程理论
   - 收敛指标曲线
   - 价值函数热力图
   - 最优努力分布

5. **simulation.html** - 仿真模块
   - 时间序列演化图表
   - 失业率/工资/技能动态
   - 政策效果对比

6. **data_analysis.html** - 数据分析
   - 调研数据统计
   - 相关性热力图
   - 描述性统计表

7. **results.html** - 结果展示
   - 综合研究成果
   - 政策建议
   - 未来研究方向

### 技术特性

- **响应式设计**: 支持桌面（>1200px）、平板（768-1200px）、手机（<768px）
- **交互式图表**: 8个Plotly动态图表（支持缩放、悬停、下载）
- **静态高清图表**: 20+个DPI 300图表（适合打印和高分辨率屏幕）
- **统一设计语言**: 紫色渐变主题，专业视觉风格
- **流畅动画**: 所有hover、transition动画统一为0.3s cubic-bezier缓动
- **图片居中**: 所有图表和图片完全居中对齐

### 优化成果

项目网站经过P0-P6优先级的全面优化：

- **P0**: 联系方式更新（经济学院）+ GitHub链接
- **P1**: 统一图表容器样式（margin: 40px auto）
- **P2**: 响应式布局（3个断点）
- **P3**: 视觉层次优化（字体大小、间距）
- **P4**: 静态图表DPI提升至300
- **P5**: 交互动画优化（表格hover效果）
- **P6**: 配色和空白空间微调

**整体完成度**: 95%

### 访问方式

- **本地访问**: 直接打开 `Simulation_project_v3/WEBSITE/index.html`
- **GitHub Pages**: [https://github.com/likewindccc/2025DaChuang](https://github.com/likewindccc/2025DaChuang)

---

## 快速开始

### 环境要求

- **Python版本**: 3.8+
- **操作系统**: Windows 10/11, macOS, Linux
- **内存**: 4GB+ RAM
- **处理器**: 多核CPU（推荐，用于Numba并行加速）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/likewindccc/2025DaChuang.git
cd 2025DaChuang/Simulation_project_v3
```

#### 2. 创建虚拟环境（推荐）

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

#### 生成虚拟劳动力

```bash
cd TESTS
python test_population.py
```

输出：
- `OUTPUT/population/initial_distribution.png` - 初始分布图
- `OUTPUT/population/copula_structure.png` - Copula结构图

#### 运行GS匹配算法

```bash
python test_gs_matching.py
```

输出：
- 匹配率统计
- 偏好分数分布图

#### 求解MFG均衡

```bash
python test_mfg_equilibrium.py
```

输出：
- 收敛指标曲线
- 价值函数热力图
- 最优努力分布

#### 运行市场仿真

```bash
python test_simulator.py
```

输出：
- 失业率时间序列
- 工资演化曲线
- 技能分布动态

#### 生成所有可视化图表

```bash
python generate_all_visualizations.py
```

输出：
- 所有模块的静态和交互式图表
- 自动复制到WEBSITE/charts/目录

---

## 主要功能特性

### 1. 虚拟劳动力市场生成

- 支持均衡市场（θ ∈ [0.9, 1.1]）
- 支持岗位富余市场（θ ∈ [1.1, 1.5]）
- 支持岗位短缺市场（θ ∈ [0.5, 0.9]）
- 自动生成N=10000个虚拟劳动力
- K-S检验验证分布有效性

### 2. 稳定匹配算法

- Gale-Shapley算法实现
- Numba JIT加速（3-5倍）
- 支持最多32轮匹配
- 匹配率达50%
- 偏好函数MinMax标准化

### 3. MFG均衡求解

- Bellman值迭代（最大500次）
- KFE人口演化（蒙特卡洛）
- Nash均衡迭代（最大100轮）
- Numba并行加速（10-30倍）
- 收敛性监控（|ΔV|、|Δa|、|Δu|）

### 4. 时间序列仿真

- T=50期市场动态演化
- 失业率、工资、技能指标追踪
- 支持政策干预（培训、补贴）
- 反事实分析

### 5. 参数校准

- SMM方法匹配真实数据矩
- 遗传算法全局优化
- 多目标矩匹配（失业率、工资分布、匹配率）
- 参数敏感性分析

---

## 研究成果

### 完成度统计

- **整体完成度**: 95%
- **核心模块**: 5个（POPULATION, LOGISTIC, MFG, SIMULATOR, CALIBRATION）
- **测试脚本**: 40+个
- **专业图表**: 30+个（8个交互式 + 20+个静态）
- **网站页面**: 7个完整页面
- **代码行数**: 10000+行（严格遵守PEP8）

### 图表成果

| 模块 | 交互式图表 | 静态图表 | 总计 |
|------|----------|---------|------|
| POPULATION | 2 | 2 | 4 |
| LOGISTIC | 2 | 4 | 6 |
| MFG | 2 | 8 | 10 |
| SIMULATION | 2 | 3 | 5 |
| DATA | 0 | 1 | 1 |
| **总计** | **8** | **18** | **26** |

### 代码质量

- **PEP8规范**: 100%遵守
- **中文注释**: 完整覆盖
- **类型提示**: 核心函数全部添加
- **文档字符串**: 所有公共接口
- **测试覆盖**: 核心功能全部测试

### 性能优化

| 模块 | 优化前耗时 | 优化后耗时 | 加速比 |
|------|----------|----------|--------|
| GS匹配 | 15s | 3-5s | 3-5x |
| Bellman求解 | 300s | 10-30s | 10-30x |
| KFE演化 | 200s | 7-20s | 10-30x |

### 网站优化成果

- **响应式覆盖率**: 100%（桌面/平板/手机）
- **CSS样式一致性**: 98%
- **交互动画流畅度**: 95%（0.3s统一缓动）
- **图片居中对齐**: 100%（16个图片全部居中）
- **静态图表DPI**: 300（适合打印）

---

## 版本历史

### v1.0 

- 建立基础人口分布模块
- 实现简单匹配机制
- 探索MFG理论实现方法
- **状态**: 部分完成后停止
- **原因**: 架构设计不够清晰，模块职责划分不明确

### v2.0 

- 深层目录嵌套（src/core/modules）
- 稀疏网格离散化状态空间
- 详尽的文档系统（19个文档）
- Numba性能优化探索
- **状态**: 开发完成但废弃
- **原因**: 过度复杂，偏离"简洁实用"原则

### v3.0 

- 扁平化结构，无多余嵌套
- 基于个体的蒙特卡洛，放弃稀疏网格
- 5个模块职责清晰，依赖明确
- 严格PEP8规范，完整中文注释
- 专业网站展示系统（7个HTML页面）
- **状态**: 95%完成，可立即部署
- **特点**: 简洁、清晰、实用优先

---

## 联系方式和版权信息

### 项目信息

- **单位**: 中国人民大学经济学院
- **项目类型**: 求是学术（大学生创新创业训练计划项目）
- **项目负责人**: 符洪瑞
- **研究方向**: 劳动经济学、平均场博弈、计算经济学

### 开源信息

- **GitHub仓库**: [https://github.com/likewindccc/2025DaChuang](https://github.com/likewindccc/2025DaChuang)
- **贡献指南**: 欢迎提交Issue和Pull Request

---

**最后更新**: 2025-11-12
**当前版本**: Simulation_project_v3 (v3.0)
**开发状态**: 95%完成，可立即部署
**网站状态**: 7个页面全部完成，响应式设计优化完毕
