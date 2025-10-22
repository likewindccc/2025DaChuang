# EconLab桌面应用开发文档

**项目名称**: 农村女性就业市场MFG模拟系统 - 桌面版  
**版本**: 1.0  
**开发框架**: PyQt6  
**目标**: 构建专业、美观、易用的展示型桌面应用  
**创建日期**: 2025-10-21

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [设计理念](#2-设计理念)
3. [技术架构](#3-技术架构)
4. [文件结构说明](#4-文件结构说明)
5. [界面设计详解](#5-界面设计详解)
6. [核心功能模块](#6-核心功能模块)
7. [开发路线图](#7-开发路线图)
8. [技术实现细节](#8-技术实现细节)
9. [打包发布](#9-打包发布)
10. [展示要点](#10-展示要点)

---

## 1. 项目概述

### 1.1 定位

**EconLab** - 农村女性就业市场经济学实验室

这是一个将复杂的MFG（平均场博弈）模型封装成易用桌面应用的项目，主要用于：
- 学术展示和演示
- 研究成果可视化
- 政策模拟实验
- 教学辅助工具

### 1.2 核心价值

- **易用性**: 无需编程基础，图形化操作
- **专业性**: 美观的界面设计，专业的经济学术语
- **可视化**: 实时动态图表，直观展示仿真过程
- **可靠性**: 基于成熟的v3后端代码，经过充分测试

### 1.3 目标用户

- **学术展示**: 向导师、评委展示研究成果
- **政策研究**: 为政策制定者提供仿真工具
- **教学演示**: 经济学课程教学辅助
- **研究协作**: 团队成员共同使用

---

## 2. 设计理念

### 2.1 界面设计原则

**简洁而不简单**：
- 主界面清爽，3个主要标签页
- 隐藏复杂参数，提供"专家模式"切换
- 突出核心功能，次要功能收纳在菜单

**专业而不晦涩**：
- 使用专业术语（如"贴现因子"、"市场紧张度"）
- 提供tooltips解释每个参数含义
- 内置帮助文档和示例

**美观而不花哨**：
- 现代化扁平设计风格
- 统一的配色方案（主色：深蓝 #2C3E50，强调色：青绿 #1ABC9C）
- 适当的动画效果（进度条、图表更新）

### 2.2 交互设计原则

**所见即所得**：
- 参数修改立即显示预览效果
- 仿真运行实时更新监控数据
- 结果可视化支持交互探索

**友好的错误处理**：
- 参数范围校验（滑块+输入框组合）
- 运行前检查配置完整性
- 友好的错误提示对话框

**保存用户习惯**：
- 记住上次使用的配置
- 支持配置文件导入导出
- 历史运行记录管理

---

## 3. 技术架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    EconLab Desktop App                  │
├─────────────────────────────────────────────────────────┤
│                   表现层 (Presentation)                  │
│  ┌────────────┬────────────┬────────────┬─────────────┐ │
│  │  主窗口     │  配置页    │  运行页    │  结果页     │ │
│  │ MainWindow │ ConfigPage │ SimulPage  │ ResultPage  │ │
│  └────────────┴────────────┴────────────┴─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   业务层 (Business Logic)                │
│  ┌────────────┬────────────┬────────────┬─────────────┐ │
│  │  配置管理   │  任务调度  │  数据处理  │  图表生成   │ │
│  │ ConfigMgr  │ TaskMgr    │ DataProc   │ ChartGen    │ │
│  └────────────┴────────────┴────────────┴─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  核心引擎 (Core Engine)                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │      现有的MODULES (不修改，直接调用)             │   │
│  │  POPULATION | LOGISTIC | MFG | CALIBRATION      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| **界面** | PyQt6 | 主界面框架 |
| **绘图** | matplotlib | 嵌入式图表 |
| **后台** | QThread | 多线程避免界面卡顿 |
| **配置** | PyYAML | 读写配置文件 |
| **打包** | PyInstaller | 生成exe可执行文件 |
| **图标** | Qt Designer | 设计UI文件 |

### 3.3 依赖关系

```python
# 新增依赖（在现有requirements.txt基础上）
PyQt6>=6.6.0           # 主界面框架
PyQt6-Qt6>=6.6.0       # Qt6核心库
```

所有其他依赖（numpy、pandas、scipy、numba等）已在项目中，无需额外安装。

---

## 4. 文件结构说明

### 4.1 完整文件树

```
Simulation_project_v3/
├── GUI/                              # GUI模块根目录
│   ├── __init__.py                   # 包初始化
│   ├── main_window.py                # 主窗口类（400行）
│   ├── app.py                        # 应用启动入口（50行）
│   │
│   ├── pages/                        # 页面模块
│   │   ├── __init__.py
│   │   ├── config_page.py           # 参数配置页（300行）
│   │   ├── simulation_page.py       # 仿真运行页（350行）
│   │   └── results_page.py          # 结果分析页（400行）
│   │
│   ├── workers/                      # 后台工作线程
│   │   ├── __init__.py
│   │   ├── simulation_worker.py     # MFG仿真线程（150行）
│   │   └── calibration_worker.py    # 校准任务线程（100行）
│   │
│   ├── widgets/                      # 自定义组件
│   │   ├── __init__.py
│   │   ├── parameter_widget.py      # 参数输入组件（100行）
│   │   ├── chart_widget.py          # 图表展示组件（200行）
│   │   └── log_widget.py            # 日志显示组件（80行）
│   │
│   ├── utils/                        # 工具函数
│   │   ├── __init__.py
│   │   ├── config_manager.py        # 配置管理器（150行）
│   │   ├── chart_generator.py       # 图表生成器（200行）
│   │   └── path_helper.py           # 路径处理（50行）
│   │
│   ├── resources/                    # 资源文件
│   │   ├── icons/                   # 图标文件
│   │   │   ├── app.ico              # 应用图标
│   │   │   ├── run.png              # 运行图标
│   │   │   ├── pause.png            # 暂停图标
│   │   │   └── stop.png             # 停止图标
│   │   ├── images/                  # 图片资源
│   │   │   └── splash.png           # 启动画面
│   │   └── styles/                  # 样式表
│   │       ├── main.qss             # 主样式表
│   │       └── dark_theme.qss       # 暗色主题（可选）
│   │
│   └── GUI开发文档.md                # 本文档
│
├── app.py                            # 应用入口（软链接到GUI/app.py）
├── app.spec                          # PyInstaller打包配置
└── (其他现有文件保持不变)
```

### 4.2 文件职责说明

#### 核心文件

**app.py** - 应用启动入口
- 设置Python路径
- 加载配置和资源
- 创建主窗口
- 启动事件循环

**main_window.py** - 主窗口
- 创建菜单栏、工具栏、状态栏
- 管理3个标签页切换
- 处理全局事件（退出、保存等）

#### 页面文件

**config_page.py** - 参数配置页
- 经济参数编辑（rho、kappa等）
- 市场参数配置（个体数、theta等）
- 配置文件导入导出
- 参数验证和默认值重置

**simulation_page.py** - 仿真运行页
- 运行控制按钮（开始/暂停/停止）
- 实时进度监控（进度条、百分比）
- 关键指标实时显示（失业率、收敛状态）
- 运行日志滚动显示

**results_page.py** - 结果分析页
- 多种图表展示（折线图、热力图等）
- 关键指标汇总表格
- 结果导出功能（CSV、图片、报告）

#### 工作线程

**simulation_worker.py** - MFG仿真后台线程
- 调用solve_equilibrium()
- 定期发送进度信号
- 处理暂停/停止请求
- 返回最终结果

**calibration_worker.py** - 参数校准线程（可选）
- 调用SMMCalibrator
- 长时间运行的校准任务

#### 自定义组件

**parameter_widget.py** - 参数输入组件
- 标签 + 滑块 + 输入框组合
- 自动范围验证
- tooltip显示参数说明

**chart_widget.py** - 图表显示组件
- matplotlib嵌入PyQt6
- 支持缩放、保存
- 自动更新数据

**log_widget.py** - 日志显示组件
- 彩色日志（不同级别不同颜色）
- 自动滚动到底部
- 搜索和过滤功能

#### 工具模块

**config_manager.py** - 配置管理
- 读写YAML配置文件
- 合并用户配置和默认配置
- 配置验证

**chart_generator.py** - 图表生成
- 统一的图表样式
- 支持多种图表类型
- 中文字体配置

**path_helper.py** - 路径处理
- 处理打包后的资源路径
- 创建输出目录

---

## 5. 界面设计详解

### 5.1 主窗口布局

```
┌──────────────────────────────────────────────────────────────┐
│ EconLab - 农村女性就业市场仿真平台                 [_][□][×] │
├──────────────────────────────────────────────────────────────┤
│ 文件(F) 编辑(E) 运行(R) 工具(T) 帮助(H)                       │  ← 菜单栏
├──────────────────────────────────────────────────────────────┤
│ [📁新建] [💾保存] [▶运行] [⏹停止] [📊图表] [❓帮助]          │  ← 工具栏
├──────────────────────────────────────────────────────────────┤
│ ┌──────────┬──────────┬──────────┐                           │
│ │ 参数配置 │ 仿真运行 │ 结果分析 │                           │  ← 标签页
│ └──────────┴──────────┴──────────┘                           │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │                                                          │ │
│ │                    标签页内容区域                         │ │
│ │                  (根据当前标签切换)                       │ │
│ │                                                          │ │
│ │                       (600px高)                          │ │
│ │                                                          │ │
│ └──────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ 就绪 | 当前配置: mfg_config.yaml | 上次运行: 2025-10-21      │  ← 状态栏
└──────────────────────────────────────────────────────────────┘
                    窗口尺寸: 1000×700
```

### 5.2 标签页1: 参数配置

```
┌─ 参数配置 ──────────────────────────────────────────────┐
│                                                         │
│  ┌─ 经济参数 ─────────────────────────────────────┐    │
│  │                                                 │    │
│  │  贴现因子 ρ (个体对未来的重视程度)              │    │
│  │  [━━━━●━━━━━━] 0.40                             │    │
│  │  0.30                        0.60               │    │
│  │                                                 │    │
│  │  努力成本系数 κ (努力的边际成本)                │    │
│  │  [━━━━━━●━━━━] 2000                             │    │
│  │  1000                       4000                │    │
│  │                                                 │    │
│  │  T负效用系数 α (工作时间偏离惩罚)               │    │
│  │  [━━━━━●━━━━━] 0.30                             │    │
│  │  0.10                        0.60               │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ 市场参数 ─────────────────────────────────────┐    │
│  │                                                 │    │
│  │  个体数量:        [10000▼]                      │    │
│  │  目标市场紧张度:  [1.5  ]                       │    │
│  │  最大迭代轮数:    [100  ]                       │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ 状态更新参数 (专家模式) ─────────────────────┐    │
│  │  [展开 ▼]                                       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  [📁加载配置文件▼] [💾保存当前配置] [🔄重置为默认值]  │
│                                                         │
│  当前配置: CONFIG/mfg_config.yaml (已修改)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**交互细节**：
- 滑块拖动实时显示数值
- 数值输入框支持手动输入
- 参数旁边有❓图标，鼠标悬停显示详细说明
- "专家模式"折叠/展开高级参数

### 5.3 标签页2: 仿真运行

```
┌─ 仿真运行 ──────────────────────────────────────────────┐
│                                                         │
│  ┌─ 运行控制 ─────────────────────────────────────┐    │
│  │  [▶ 开始仿真] [⏸ 暂停] [⏹ 停止]               │    │
│  │                                                 │    │
│  │  运行状态: 运行中 🟢 | 预计剩余: 12分钟        │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ 实时监控 ─────────────────────────────────────┐    │
│  │                                                 │    │
│  │  当前迭代: 35 / 100                             │    │
│  │  [███████████████████░░░░░░░░] 35%             │    │
│  │                                                 │    │
│  │  ┌─────────────┬─────────────┬─────────────┐  │    │
│  │  │ 失业率      │ 收敛指标    │ 平均工资    │  │    │
│  │  │             │             │             │  │    │
│  │  │   4.2%      │ |ΔV|=0.015  │  4523元     │  │    │
│  │  │   ↓ 0.3%    │ 未收敛      │  ↑ 15元     │  │    │
│  │  └─────────────┴─────────────┴─────────────┘  │    │
│  │                                                 │    │
│  │  [失业率趋势图]                                 │    │
│  │  ┌──────────────────────────────────────┐     │    │
│  │  │   5% ┤                               │     │    │
│  │  │      │  ╱╲                           │     │    │
│  │  │   4% ┤ ╱  ╲___                       │     │    │
│  │  │      │        ───                    │     │    │
│  │  │   3% ┤                               │     │    │
│  │  │      └─────┬─────┬─────┬─────┬─────  │     │    │
│  │  │           20    40    60    80   100 │     │    │
│  │  └──────────────────────────────────────┘     │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ 运行日志 ─────────────────────────────────────┐    │
│  │  [14:23:01] INFO  开始MFG均衡求解...            │    │
│  │  [14:23:02] INFO  初始化10000个个体             │    │
│  │  [14:23:15] INFO  第10轮: 失业率=4.5%           │    │
│  │  [14:23:30] INFO  第20轮: 失业率=4.3%           │    │
│  │  [14:23:45] INFO  第30轮: 失业率=4.2%           │    │
│  │  [14:24:00] WARN  收敛较慢，建议降低rho         │    │
│  │  ▼ (自动滚动到底部)                             │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**交互细节**：
- 开始后按钮状态变化（开始→暂停/停止）
- 实时更新进度条和监控数据
- 失业率趋势图动态绘制
- 日志支持搜索和复制

### 5.4 标签页3: 结果分析

```
┌─ 结果分析 ──────────────────────────────────────────────┐
│                                                         │
│  [失业率趋势] [状态分布] [收敛过程] [市场紧张度]        │  ← 图表切换
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                                                 │   │
│  │         失业率随迭代变化（折线图）              │   │
│  │                                                 │   │
│  │   5% ┤                                         │   │
│  │      │   ╱╲                                    │   │
│  │   4% ┤  ╱  ╲____                               │   │
│  │      │          ────────                       │   │
│  │   3% ┤                                         │   │
│  │      └────┬────┬────┬────┬────┬────           │   │
│  │          20   40   60   80  100               │   │
│  │                                                 │   │
│  │  [🔍放大] [💾保存为图片] [📋复制数据]          │   │
│  │                                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ 关键指标汇总 ─────────────────────────────────┐    │
│  │  最终失业率:    4.04%     收敛轮数:    87      │    │
│  │  平均工资:      4523元    市场紧张度:  1.52    │    │
│  │  平均T值:       45.2小时  平均S值:     0.73    │    │
│  │  平均D值:       0.65      平均W值:     4312元  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─ 结果导出 ─────────────────────────────────────┐    │
│  │  [📄导出CSV数据] [📊导出所有图表] [📝生成报告]  │    │
│  │                                                 │    │
│  │  导出路径: OUTPUT/results_2025-10-21_142530/   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**交互细节**：
- 多个图表快速切换
- 图表支持缩放、平移、保存
- 指标汇总用卡片式布局
- 一键导出所有结果

---

## 6. 核心功能模块

### 6.1 配置管理 (config_manager.py)

**功能**：
- 读取和写入YAML配置文件
- 参数验证（范围、类型）
- 合并用户配置和默认配置
- 配置历史管理

**关键方法**：
```python
class ConfigManager:
    def __init__(self, config_dir="CONFIG"):
        """初始化配置管理器"""
        
    def load_config(self, config_name="mfg_config.yaml"):
        """加载配置文件"""
        
    def save_config(self, config_dict, config_name):
        """保存配置到文件"""
        
    def validate_config(self, config_dict):
        """验证配置参数"""
        
    def get_default_config(self):
        """获取默认配置"""
        
    def list_recent_configs(self):
        """列出最近使用的配置"""
```

### 6.2 仿真工作线程 (simulation_worker.py)

**功能**：
- 在后台线程运行MFG求解
- 定期发送进度更新信号
- 支持暂停和停止
- 异常处理和错误报告

**关键代码**：
```python
class SimulationWorker(QThread):
    # 信号定义
    progress_updated = pyqtSignal(int, dict)  # 进度, 当前状态
    finished = pyqtSignal(dict)               # 最终结果
    error_occurred = pyqtSignal(str)          # 错误信息
    
    def __init__(self, config_dict):
        """初始化工作线程"""
        
    def run(self):
        """运行MFG求解"""
        # 在这里调用solve_equilibrium()
        
    def pause(self):
        """暂停运行"""
        
    def stop(self):
        """停止运行"""
```

### 6.3 图表生成 (chart_generator.py)

**功能**：
- 生成统一风格的matplotlib图表
- 支持多种图表类型
- 处理中文字体
- 图表导出

**支持的图表类型**：
1. 失业率趋势折线图
2. 状态变量分布直方图
3. 收敛过程曲线
4. 市场紧张度变化
5. 参数敏感性热力图

**关键方法**：
```python
class ChartGenerator:
    def __init__(self):
        """初始化图表生成器，配置中文字体"""
        
    def plot_unemployment_trend(self, data):
        """绘制失业率趋势图"""
        
    def plot_state_distribution(self, individuals):
        """绘制状态变量分布"""
        
    def plot_convergence(self, history):
        """绘制收敛曲线"""
        
    def save_chart(self, fig, filename):
        """保存图表为图片"""
```

### 6.4 自定义组件 (widgets/)

**ParameterWidget** - 参数输入组件
```python
class ParameterWidget(QWidget):
    """参数输入组件：标签+滑块+输入框"""
    
    value_changed = pyqtSignal(float)  # 值改变信号
    
    def __init__(self, name, min_val, max_val, 
                 default_val, tooltip=""):
        # 创建标签、滑块、输入框
        # 连接信号槽
        
    def get_value(self):
        """获取当前值"""
        
    def set_value(self, value):
        """设置值"""
```

**ChartWidget** - 图表显示组件
```python
class ChartWidget(QWidget):
    """matplotlib嵌入PyQt6的图表组件"""
    
    def __init__(self):
        # 创建matplotlib画布
        # 添加工具栏
        
    def update_chart(self, data):
        """更新图表数据"""
        
    def clear(self):
        """清空图表"""
```

**LogWidget** - 日志显示组件
```python
class LogWidget(QTextEdit):
    """彩色日志显示组件"""
    
    def append_log(self, level, message):
        """添加一条日志"""
        # level: INFO/WARN/ERROR
        # 不同级别不同颜色
        
    def clear_log(self):
        """清空日志"""
```

---

## 7. 开发路线图

### 阶段1: 基础框架搭建（预计2天）

**目标**：创建主窗口和基本页面结构

**任务清单**：
- [ ] 创建所有`__init__.py`文件
- [ ] 实现`app.py`应用启动入口
- [ ] 实现`main_window.py`主窗口
  - [ ] 菜单栏
  - [ ] 工具栏
  - [ ] 状态栏
  - [ ] 标签页容器
- [ ] 创建3个空白页面类
  - [ ] ConfigPage
  - [ ] SimulationPage
  - [ ] ResultsPage
- [ ] 测试基本导航和切换

**验收标准**：
- 可以启动GUI
- 可以在3个标签页之间切换
- 菜单和工具栏显示正常

### 阶段2: 参数配置页开发（预计1.5天）

**目标**：完成参数配置功能

**任务清单**：
- [ ] 实现`ParameterWidget`自定义组件
- [ ] 实现`ConfigManager`配置管理器
- [ ] 在ConfigPage中布局所有参数
  - [ ] 经济参数区域
  - [ ] 市场参数区域
  - [ ] 高级参数（折叠）
- [ ] 实现配置文件加载/保存
- [ ] 参数验证和错误提示
- [ ] 重置为默认值功能

**验收标准**：
- 可以修改所有参数
- 可以保存配置到YAML
- 可以加载已有配置
- 参数范围验证正常

### 阶段3: 仿真运行页开发（预计2天）

**目标**：实现仿真运行和实时监控

**任务清单**：
- [ ] 实现`SimulationWorker`后台线程
  - [ ] 调用`solve_equilibrium()`
  - [ ] 实现回调机制
  - [ ] 进度信号发送
- [ ] 实现SimulationPage界面
  - [ ] 运行控制按钮
  - [ ] 进度条
  - [ ] 实时监控区域
  - [ ] 动态折线图
  - [ ] 日志显示
- [ ] 连接信号槽
  - [ ] 开始/暂停/停止
  - [ ] 进度更新
  - [ ] 完成通知

**验收标准**：
- 点击开始可以运行MFG求解
- 实时显示进度和监控数据
- 可以暂停和停止
- 运行完成后显示通知

### 阶段4: 结果分析页开发（预计1.5天）

**目标**：实现结果可视化和导出

**任务清单**：
- [ ] 实现`ChartGenerator`图表生成器
- [ ] 实现`ChartWidget`图表组件
- [ ] 在ResultsPage中集成图表
  - [ ] 失业率趋势图
  - [ ] 状态分布图
  - [ ] 收敛曲线图
- [ ] 实现关键指标汇总表格
- [ ] 实现结果导出功能
  - [ ] 导出CSV
  - [ ] 导出图表PNG
  - [ ] 生成PDF报告

**验收标准**：
- 运行完成后自动跳转到结果页
- 所有图表正常显示
- 可以切换不同图表
- 可以导出所有结果

### 阶段5: 美化和优化（预计1天）

**目标**：提升界面美观度和用户体验

**任务清单**：
- [ ] 设计并应用QSS样式表
  - [ ] 统一配色方案
  - [ ] 按钮样式
  - [ ] 输入框样式
  - [ ] 表格样式
- [ ] 添加图标
  - [ ] 应用图标
  - [ ] 工具栏图标
  - [ ] 按钮图标
- [ ] 添加过渡动画
  - [ ] 进度条动画
  - [ ] 图表更新动画
- [ ] 优化布局和间距
- [ ] 中文字体配置

**验收标准**：
- 界面美观专业
- 配色协调统一
- 图标清晰易懂
- 操作流畅自然

### 阶段6: 打包和测试（预计1天）

**目标**：生成可分发的exe文件

**任务清单**：
- [ ] 配置PyInstaller
  - [ ] 编写app.spec
  - [ ] 处理资源文件打包
  - [ ] 处理Numba打包
- [ ] 测试打包后的exe
  - [ ] 功能测试
  - [ ] 性能测试
  - [ ] 异常处理测试
- [ ] 制作安装包（可选）
- [ ] 编写用户手册

**验收标准**：
- 生成的exe可以独立运行
- 文件体积<200MB
- 启动时间<5秒
- 所有功能正常

**总预计时间**：8-9天

---

## 8. 技术实现细节

### 8.1 Numba与PyQt6集成

**问题**：Numba JIT编译在第一次运行时需要时间

**解决方案1**：启动画面
```python
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap

# 显示启动画面
splash = QSplashScreen(QPixmap("resources/images/splash.png"))
splash.show()

# 预热Numba（触发JIT编译）
from MODULES.MFG import solve_equilibrium
# 小规模测试运行...

splash.close()
# 显示主窗口
```

**解决方案2**：AOT编译（可选）
```python
# 使用numba.pycc预编译
# 但会增加开发复杂度，暂不推荐
```

### 8.2 后台线程与界面更新

**问题**：MFG求解耗时，不能阻塞主线程

**解决方案**：使用QThread
```python
class SimulationPage(QWidget):
    def start_simulation(self):
        # 创建工作线程
        self.worker = SimulationWorker(self.config)
        
        # 连接信号
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        
        # 启动线程
        self.worker.start()
        
    def update_progress(self, iteration, stats):
        # 更新界面（在主线程）
        self.progress_bar.setValue(iteration)
        self.unemployment_label.setText(f"{stats['u_rate']:.2%}")
```

### 8.3 配置文件路径处理

**问题**：打包后资源文件路径改变

**解决方案**：
```python
# utils/path_helper.py
import sys
import os

def get_resource_path(relative_path):
    """获取资源文件绝对路径（兼容打包）"""
    if getattr(sys, 'frozen', False):
        # 打包后，资源在_MEIPASS临时目录
        base_path = sys._MEIPASS
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.dirname(__file__))
    
    return os.path.join(base_path, relative_path)

# 使用示例
config_path = get_resource_path("CONFIG/mfg_config.yaml")
icon_path = get_resource_path("GUI/resources/icons/app.ico")
```

### 8.4 matplotlib嵌入PyQt6

**标准方法**：
```python
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

class ChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # 创建matplotlib图表
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y)
        self.ax.set_xlabel('迭代轮数')
        self.ax.set_ylabel('失业率 (%)')
        self.canvas.draw()
```

### 8.5 实时进度回调

**修改equilibrium_solver.py**（小改动，不影响原有逻辑）：
```python
# 在solve()方法中添加callback参数
def solve(self, individuals=None, verbose=True, callback=None):
    for outer_iter in range(self.max_outer_iter):
        # ... 原有求解逻辑 ...
        
        # 新增：调用回调函数
        if callback is not None:
            current_stats = {
                'iteration': outer_iter,
                'u_rate': unemployment_rate,
                'mean_wage': individuals['current_wage'].mean(),
                'converged': converged
            }
            callback(current_stats)
        
        # ... 后续逻辑 ...
```

**在Worker中使用**：
```python
class SimulationWorker(QThread):
    def run(self):
        from MODULES.MFG import solve_equilibrium
        
        def progress_callback(stats):
            # 发送信号到主线程
            self.progress_updated.emit(stats['iteration'], stats)
        
        # 调用求解，传入回调
        individuals, info = solve_equilibrium(callback=progress_callback)
        
        self.finished.emit(info)
```

### 8.6 QSS样式表示例

```css
/* GUI/resources/styles/main.qss */

/* 主窗口背景 */
QMainWindow {
    background-color: #F5F6FA;
}

/* 标签页样式 */
QTabWidget::pane {
    border: 1px solid #D5D8DC;
    background-color: white;
}

QTabBar::tab {
    background-color: #ECF0F1;
    color: #2C3E50;
    padding: 10px 20px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: white;
    color: #1ABC9C;
    border-bottom: 3px solid #1ABC9C;
}

/* 按钮样式 */
QPushButton {
    background-color: #1ABC9C;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 14px;
}

QPushButton:hover {
    background-color: #16A085;
}

QPushButton:pressed {
    background-color: #138D75;
}

QPushButton:disabled {
    background-color: #BDC3C7;
}

/* 进度条样式 */
QProgressBar {
    border: 1px solid #BDC3C7;
    border-radius: 4px;
    text-align: center;
    height: 25px;
}

QProgressBar::chunk {
    background-color: #1ABC9C;
    border-radius: 3px;
}

/* 滑块样式 */
QSlider::groove:horizontal {
    height: 6px;
    background: #E5E8E8;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #1ABC9C;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
```

---

## 9. 打包发布

### 9.1 PyInstaller配置

**app.spec配置文件**：
```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 分析依赖
a = Analysis(
    ['GUI/app.py'],                    # 入口文件
    pathex=[],
    binaries=[],
    datas=[
        # 配置文件
        ('CONFIG', 'CONFIG'),
        # 预处理数据
        ('DATA/processed', 'DATA/processed'),
        # GUI资源
        ('GUI/resources', 'GUI/resources'),
        # 匹配函数模型
        ('OUTPUT/logistic/match_function_model.pkl', 
         'OUTPUT/logistic'),
    ],
    hiddenimports=[
        'numba',
        'scipy._lib.messagestream',
        'scipy.special.cython_special',
        'copulas',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'pytest',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 打包为单个exe
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EconLab',                    # exe文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                          # 使用UPX压缩
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                     # 不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='GUI/resources/icons/app.ico'  # 应用图标
)
```

### 9.2 打包命令

```bash
# 1. 安装PyInstaller
pip install pyinstaller

# 2. 使用spec文件打包
pyinstaller app.spec

# 3. 生成的exe在 dist/ 目录
```

### 9.3 打包优化

**减小体积**：
```bash
# 使用UPX压缩（需要下载UPX）
# 在spec中设置 upx=True

# 排除不必要的库
# 在spec的excludes中添加
```

**加快启动**：
```python
# 在app.py开头添加
import os
os.environ['NUMBA_DISABLE_JIT'] = '0'  # 确保JIT开启
os.environ['NUMBA_CACHE_DIR'] = '.numba_cache'  # 缓存编译结果
```

### 9.4 测试清单

**功能测试**：
- [ ] 启动exe是否正常
- [ ] 加载配置文件
- [ ] 修改参数
- [ ] 运行MFG求解
- [ ] 查看结果图表
- [ ] 导出结果文件

**性能测试**：
- [ ] 启动时间 < 5秒
- [ ] 运行10000个体 < 20分钟
- [ ] 内存占用 < 2GB

**兼容性测试**：
- [ ] Windows 10
- [ ] Windows 11
- [ ] 不同分辨率屏幕

---

## 10. 展示要点

### 10.1 适合展示的场景

**场景1：参数调整演示**
1. 打开应用，进入"参数配置"页
2. 调整贴现因子ρ（如从0.40到0.35）
3. 说明："这个参数控制个体对未来的重视程度"
4. 点击"开始仿真"

**场景2：实时监控演示**
1. 进入"仿真运行"页
2. 展示实时更新的失业率曲线
3. 说明："可以看到失业率在前20轮快速下降，然后趋于稳定"
4. 展示日志输出

**场景3：结果分析演示**
1. 仿真完成后，自动跳转到"结果分析"页
2. 切换不同图表类型
3. 说明各个关键指标
4. 演示导出功能

### 10.2 展示话术建议

**开场白**：
> "这是我们开发的农村女性就业市场仿真平台EconLab，它将复杂的平均场博弈模型封装成了易用的桌面应用。"

**功能介绍**：
> "应用分为三个主要模块：参数配置、仿真运行、结果分析。用户无需编程就能进行政策模拟实验。"

**技术亮点**：
> "后台使用了Numba加速，可以处理10000个个体的复杂交互。前端用PyQt6打造了美观的界面。"

**应用价值**：
> "这个工具可以帮助政策制定者评估不同就业政策的效果，比如提高培训投入、降低匹配摩擦等。"

### 10.3 PPT配图建议

**第1页**：应用启动画面
- 截图：主窗口的欢迎界面

**第2页**：参数配置界面
- 截图：参数配置页，突出滑块和输入框

**第3页**：实时监控界面
- 截图：仿真运行页，进度条50%，失业率曲线动态更新

**第4页**：结果可视化
- 截图：结果分析页的多种图表

**第5页**：技术架构图
- 展示前后端分离、模块化设计

### 10.4 Demo视频脚本（3分钟）

**0:00-0:30** 开场
- 展示应用图标和启动画面
- 介绍项目背景

**0:30-1:00** 参数配置演示
- 打开参数配置页
- 调整几个关键参数
- 加载预设配置文件

**1:00-2:00** 运行仿真演示
- 点击"开始仿真"
- 展示实时更新的监控数据
- 展示失业率下降曲线

**2:00-2:40** 结果分析演示
- 展示多种图表
- 展示关键指标汇总
- 演示导出功能

**2:40-3:00** 总结
- 强调易用性和专业性
- 展示技术亮点

---

## 附录A：快速参考

### 启动开发环境

```bash
# 1. 激活虚拟环境
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1

# 2. 安装PyQt6
pip install PyQt6

# 3. 进入GUI目录
cd Simulation_project_v3/GUI

# 4. 运行应用
python app.py
```

### 常用PyQt6代码片段

**创建窗口**：
```python
from PyQt6.QtWidgets import QMainWindow, QApplication
import sys

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("EconLab")
window.setGeometry(100, 100, 1000, 700)
window.show()
sys.exit(app.exec())
```

**创建布局**：
```python
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout

layout = QVBoxLayout()
layout.addWidget(widget1)
layout.addWidget(widget2)
```

**连接信号槽**：
```python
button.clicked.connect(self.on_button_clicked)

def on_button_clicked(self):
    print("按钮被点击")
```

### 调试技巧

**打印调试信息**：
```python
print(f"[DEBUG] 当前迭代: {iteration}")
```

**异常捕获**：
```python
try:
    # 可能出错的代码
except Exception as e:
    QMessageBox.critical(self, "错误", f"发生异常: {str(e)}")
```

---

## 附录B：开发检查清单

### 编码规范检查
- [ ] PEP8代码风格
- [ ] 完整的中文注释
- [ ] 类型提示
- [ ] 文档字符串

### 功能完整性检查
- [ ] 所有按钮功能正常
- [ ] 所有信号槽连接正确
- [ ] 参数验证完整
- [ ] 错误处理完善

### 用户体验检查
- [ ] 界面美观统一
- [ ] 操作流程顺畅
- [ ] 反馈及时明确
- [ ] 帮助文档完善

### 性能检查
- [ ] 启动时间合理
- [ ] 运行过程流畅
- [ ] 内存占用正常
- [ ] 无内存泄漏

### 打包测试检查
- [ ] exe可以独立运行
- [ ] 资源文件正确加载
- [ ] 配置文件读写正常
- [ ] 异常处理有效

---

**文档版本**: 1.0  
**最后更新**: 2025-10-21  
**作者**: Claude AI Assistant  
**联系**: 如有疑问请在开发过程中随时咨询

---

**祝开发顺利！这将是一个很棒的展示型应用！** 🎉

