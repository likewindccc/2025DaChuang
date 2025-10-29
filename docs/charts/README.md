# 图表资源目录

本目录包含所有模块的可视化图表，供网页引用。

## 📁 目录结构

```
charts/
├── population/          # POPULATION模块图表
│   ├── static/         # 静态PNG图片（3张）
│   │   ├── initial_distribution.png       # 初始劳动力分布
│   │   └── copula_structure.png           # Copula相关性结构
│   └── interactive/    # 交互式HTML图表（1张）
│       └── initial_distribution.html      # 交互式初始分布
│
├── logistic/           # LOGISTIC模块图表
│   └── static/         # 静态PNG图片（3张）
│       ├── distribution_visualization.png  # 分布可视化
│       ├── prediction_analysis.png        # 预测分析
│       └── preference_distribution.png    # 偏好分数分布
│
├── mfg/                # MFG模块图表
│   ├── static/         # 静态PNG图片（7张）
│   │   ├── convergence_curves.png          # 收敛曲线 ★
│   │   ├── value_function_V_U.png          # 失业价值函数 ★
│   │   ├── value_function_V_E.png          # 就业价值函数 ★
│   │   ├── value_function_delta.png        # 价值函数差 ★
│   │   ├── effort_distribution.png         # 最优努力分布 ★
│   │   ├── market_distribution_comparison.png  # 市场分布对比
│   │   └── separation_rate_distribution.png    # 离职率分布
│   └── interactive/    # 交互式HTML图表（2张）
│       ├── value_function_V_U_3D.html      # 3D失业价值函数 ★
│       └── value_function_V_E_3D.html      # 3D就业价值函数 ★
│
├── calibration/        # CALIBRATION模块（待补充）
│   ├── static/
│   └── interactive/
│
└── simulation/         # SIMULATION模块（待补充）
    ├── static/
    └── interactive/
```

**★ = 重点推荐用于网站展示的高质量图表**

---

## 🎨 在HTML中使用图表

### 方法1：静态图片

```html
<!-- 在 population.html 中 -->
<div class="chart-container">
    <h3>初始劳动力分布</h3>
    <img src="charts/population/static/initial_distribution.png" 
         alt="初始劳动力分布" 
         style="width: 100%; max-width: 1000px;">
</div>
```

### 方法2：交互式图表

```html
<!-- 在 mfg.html 中 -->
<div class="chart-container">
    <h3>3D价值函数可视化</h3>
    <iframe src="charts/mfg/interactive/value_function_V_U_3D.html" 
            width="100%" 
            height="700px" 
            frameborder="0">
    </iframe>
</div>
```

### 方法3：响应式图片

```html
<div class="row">
    <div class="col-md-6">
        <img src="charts/mfg/static/convergence_curves.png" 
             class="img-fluid" 
             alt="收敛曲线">
    </div>
    <div class="col-md-6">
        <img src="charts/mfg/static/effort_distribution.png" 
             class="img-fluid" 
             alt="努力分布">
    </div>
</div>
```

---

## 📊 图表说明

### POPULATION模块

| 文件名 | 类型 | 说明 | 推荐使用页面 |
|--------|------|------|-------------|
| initial_distribution.png | 静态 | 4个子图展示T/S/D/W初始分布 | population.html |
| copula_structure.png | 静态 | 变量相关性热力图 | population.html |
| initial_distribution.html | 交互 | 可交互的分布图 | population.html |

### LOGISTIC模块

| 文件名 | 类型 | 说明 | 推荐使用页面 |
|--------|------|------|-------------|
| distribution_visualization.png | 静态 | 虚拟市场分布 | logistic.html |
| prediction_analysis.png | 静态 | 匹配预测分析 | logistic.html |
| preference_distribution.png | 静态 | 偏好分数分布 | logistic.html |

### MFG模块

| 文件名 | 类型 | 说明 | 推荐使用页面 |
|--------|------|------|-------------|
| convergence_curves.png | 静态 | 3合1收敛曲线 | mfg.html ★ |
| value_function_V_U.png | 静态 | 失业价值函数热力图 | mfg.html ★ |
| value_function_V_E.png | 静态 | 就业价值函数热力图 | mfg.html ★ |
| value_function_delta.png | 静态 | 就业-失业价值差 | mfg.html |
| effort_distribution.png | 静态 | 最优努力分布 | mfg.html ★ |
| market_distribution_comparison.png | 静态 | 劳动力vs企业分布对比 | mfg.html |
| separation_rate_distribution.png | 静态 | 离职率分布 | mfg.html |
| value_function_V_U_3D.html | 交互 | 3D失业价值函数 | mfg.html ★ |
| value_function_V_E_3D.html | 交互 | 3D就业价值函数 | mfg.html ★ |

---

## 🚀 下一步

### 1. 更新HTML文件

修改 `population.html`, `logistic.html`, `mfg.html` 等页面，将占位符替换为实际图表。

### 2. 提交到GitHub

```bash
cd D:\Python\2025DaChuang
git add WEBSITE/charts
git commit -m "添加可视化图表到网站"
git push
```

### 3. 更新docs文件夹（GitHub Pages）

如果网站部署在docs/目录：
```bash
# 复制charts到docs
xcopy WEBSITE\charts docs\charts /E /I /Y
git add docs/charts
git commit -m "更新网站图表"
git push
```

等待1-2分钟，访问网站即可看到新图表！

---

**生成时间**: 2025/10/29  
**图表总数**: 15张（静态12张 + 交互式3张）

