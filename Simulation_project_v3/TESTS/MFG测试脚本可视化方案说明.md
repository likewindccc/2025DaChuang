# MFG测试脚本可视化方案说明

## 📋 脚本信息

- **脚本名称**: `test_mfg_equilibrium.py`
- **功能**: 求解MFG均衡并生成详细可视化
- **配置**: 10000个体，最多200轮迭代
- **运行时间**: 约20-40分钟

---

## 🎨 可视化方案

### 设计理念

- **独立图表**：每张图独立输出，便于插入论文或报告
- **统一尺寸**：所有图表均为14×8英寸（宽屏比例）
- **高分辨率**：300 DPI，适合打印和发表
- **中文字体**：Microsoft YaHei（微软雅黑）
- **无标题**：图表不含标题，由使用者自行添加图注
- **大字号**：坐标轴标签18号加粗，刻度14号

---

### 图表清单（共6张）

#### 图1：失业率演化 (`convergence_unemployment_rate.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 迭代轮数 (1-200)，字号18加粗
- **Y轴**: 失业率 (%)，字号18加粗
- **曲线**: 蓝色实线，线宽3
- **功能**: 展示从初始失业率（约47%）到均衡失业率（约6%）的完整演化路径
- **经济学意义**: 反映劳动力市场从失衡到均衡的动态调整过程

#### 图2：市场紧张度演化 (`convergence_theta.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 迭代轮数 (1-200)，字号18加粗
- **Y轴**: 市场紧张度 θ (岗位数/求职者数)，字号18加粗
- **曲线**: 红色实线，线宽3
- **功能**: 展示θ=V/U的变化（当前为外生固定θ=1.5）
- **经济学意义**: θ>1表示岗位供给大于需求（劳动力供不应求）

#### 图3：平均努力水平演化 (`convergence_effort.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 迭代轮数 (1-200)，字号18加粗
- **Y轴**: 平均努力水平 a ∈ [0, 1]，字号18加粗
- **曲线**: 绿色实线，线宽3
- **功能**: 展示失业者求职努力的动态变化
- **经济学意义**: 
  - 初期高努力 (a≈0.12)：失业者积极求职
  - 后期低努力 (a≈0.02)：边际努力收益递减（市场饱和）

#### 图4：收敛指标监控 (`convergence_metrics.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 迭代轮数 (2-200)，字号18加粗
- **Y轴**: 收敛指标（对数尺度），字号18加粗
- **曲线1**: |ΔV| - 价值函数最大变化量（紫色，线宽3）
- **曲线2**: |Δa| - 努力水平平均变化量（橙色，线宽3）
- **参考线**: 收敛阈值 epsilon_V（红色虚线，线宽2）
- **图例**: 字号14，自动最佳位置
- **功能**: 监控算法收敛性
- **收敛条件**: |ΔV| < ε_V 且 |Δa| < ε_a 且 |Δu| < ε_u

#### 图5：失业者最优努力水平分布 (`effort_distribution_histogram.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 最优努力水平 a ∈ [0, 1]，字号18加粗
- **Y轴**: 频数，字号18加粗
- **图表类型**: 直方图，20个bins，钢蓝色，黑色边框，透明度0.7
- **功能**: 展示均衡时失业者努力选择的分布特征
- **典型发现**:
  - 大部分失业者选择低努力（a≈0-0.1）
  - 少数高技能失业者选择中高努力（a≈0.5-1.0）
  - 体现异质性：不同个体面临不同的边际收益

#### 图6：技能水平与最优努力关系 (`effort_vs_skill.png`)
- **尺寸**: 14×8英寸，300 DPI
- **X轴**: 技能水平 S，字号18加粗
- **Y轴**: 最优努力水平 a，字号18加粗
- **图表类型**: 散点图，钢蓝色，透明度0.5，点大小50，深蓝色边框
- **功能**: 揭示个体技能与求职努力的相关关系
- **预期模式**:
  - **正相关**：高技能者往往选择高努力（匹配概率高，努力收益大）
  - **异质性**：同技能水平下努力水平仍有差异（受W, D, T等影响）
  - **边界效应**：极低技能者可能放弃努力（λ≈0，努力无用）

---

## 📊 输出文件清单

### 数据文件（OUTPUT/mfg/）

| 文件名 | 大小 | 内容 | 用途 |
|--------|------|------|------|
| `equilibrium_individuals.csv` | ~1.2MB | 10000行×9列个体状态 | 微观分析、政策模拟 |
| `equilibrium_policy.csv` | ~465KB | 10000行×3列价值函数和策略 | 最优决策分析 |
| `equilibrium_history.csv` | ~23KB | 迭代历史（最多200行） | 收敛性分析 |
| `value_distribution_full.csv` | ~1.5MB | 完整价值函数分布 | 就业溢价研究 |
| `status_comparison_summary.csv` | <1KB | 就业/失业对比统计 | 快速汇总 |
| `equilibrium_summary.pkl` | <1KB | Python对象（汇总信息） | 批量实验 |

### 可视化文件（OUTPUT/mfg/）

| 文件名 | 尺寸 | 分辨率 | 内容 |
|--------|------|--------|------|
| `convergence_unemployment_rate.png` | 14×8英寸 | 300 DPI | 失业率演化曲线 |
| `convergence_theta.png` | 14×8英寸 | 300 DPI | 市场紧张度演化曲线 |
| `convergence_effort.png` | 14×8英寸 | 300 DPI | 平均努力水平演化曲线 |
| `convergence_metrics.png` | 14×8英寸 | 300 DPI | 收敛指标监控图 |
| `effort_distribution_histogram.png` | 14×8英寸 | 300 DPI | 努力分布直方图 |
| `effort_vs_skill.png` | 14×8英寸 | 300 DPI | 技能-努力散点图 |

---

## 🔍 数据解读指南

### 关键指标含义

1. **失业率 (unemployment_rate)**
   - 初始值：~47%（所有人失业后随机匹配一次）
   - 均衡值：~6%（自然失业率）
   - **政策意义**：失业率下降41pp说明市场达到高效率

2. **市场紧张度 (theta)**
   - 定义：θ = V/U （岗位数/失业者数）
   - 当前设置：θ=1.5（外生固定）
   - **经济学含义**：θ>1表示劳动力供不应求

3. **平均努力水平 (mean_effort)**
   - 范围：[0, 1]
   - 趋势：从0.12递减至0.02
   - **原因**：边际努力收益递减（λ对a的敏感性下降）

4. **价值函数 (V_U, V_E)**
   - V_U：失业状态价值（预期终生效用）
   - V_E：就业状态价值
   - Delta_V = V_E - V_U：就业溢价
   - **典型值**：Delta_V ≈ 5249元

5. **收敛指标**
   - |ΔV|：价值函数最大变化量
   - |Δa|：努力水平平均变化量
   - |Δu|：失业率变化量
   - **收敛阈值**：ε_V=0.01, ε_a=0.01, ε_u=0.001

---

## 🚀 快速运行指南

### 方法1：终端命令运行

```powershell
cd "d:\Python\2025DaChuang\Simulation_project_v3"
$env:PYTHONIOENCODING="utf-8"
D:\Python\2025DaChuang\venv\Scripts\python.exe TESTS\test_mfg_equilibrium.py
```

### 方法2：Python脚本调用

```python
import sys
sys.path.insert(0, "d:/Python/2025DaChuang/Simulation_project_v3")

from TESTS.test_mfg_equilibrium import test_mfg_equilibrium_small

# 运行测试（10000个体，200轮迭代）
individuals, info = test_mfg_equilibrium_small()
```

### 方法3：修改参数后运行

如需调整参数，修改脚本第52-54行：

```python
config['population']['n_individuals'] = 10000  # 个体数量
config['equilibrium']['max_outer_iter'] = 200   # 最大迭代轮数
config['market']['vacancy'] = 15000             # 岗位数（θ=V/U）
```

---

## 📈 可视化特性（已应用）

### ✅ 1. 中文字体

**已设置**：Microsoft YaHei（微软雅黑）

```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
```

### ✅ 2. 字体大小

**已优化**：
- 全局字体：14号
- 坐标轴标签：18号加粗
- 刻度标签：14号
- 图例：14号

```python
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.xlabel('迭代轮数', fontsize=18, fontweight='bold')
```

### ✅ 3. 高分辨率输出

**已设置**：300 DPI（论文发表级别）

```python
plt.savefig('OUTPUT/mfg/convergence_unemployment_rate.png', 
            dpi=300, bbox_inches='tight')
```

### ✅ 4. 统一尺寸

**已标准化**：所有图表均为14×8英寸（宽屏比例）

```python
plt.figure(figsize=(14, 8))
```

### ✅ 5. 无标题设计

**已实现**：所有图表不含标题，便于用户自定义图注

### ✅ 6. 增强视觉效果

**已应用**：
- 线宽加粗至3（原2）
- 网格透明度0.3，线宽1.5
- 散点图点大小50（原20）
- 紧凑布局 `bbox_inches='tight'`

---

## 🔧 故障排查

### 问题1：中文乱码

**解决方案**：
```powershell
$env:PYTHONIOENCODING="utf-8"
chcp 65001
```

### 问题2：内存不足

**解决方案**：减少个体数量
```python
config['population']['n_individuals'] = 5000
```

### 问题3：未收敛

**解决方案**：增加迭代轮数或放宽收敛阈值
```python
config['equilibrium']['max_outer_iter'] = 300
config['equilibrium']['convergence']['epsilon_V'] = 0.02
```

---

## 📚 参考文献

1. Lasry, J. M., & Lions, P. L. (2007). Mean field games. *Japanese journal of mathematics*, 2(1), 229-260.
2. Acemoglu, D., & Shimer, R. (1999). Efficient unemployment insurance. *Journal of Political Economy*, 107(5), 893-928.
3. Moen, E. R. (1997). Competitive search equilibrium. *Journal of Political Economy*, 105(2), 385-411.

---

**最后更新**: 2025-11-06
**版本**: v3.0
**维护者**: EconLab团队

