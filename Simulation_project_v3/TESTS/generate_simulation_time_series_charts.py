"""
生成SIMULATION模块的时间序列演化图表
用于网站展示
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "OUTPUT" / "simulation"
WEBSITE_CHARTS_DIR = BASE_DIR / "WEBSITE" / "charts" / "simulation"
WEBSITE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
(WEBSITE_CHARTS_DIR / "static").mkdir(exist_ok=True)
(WEBSITE_CHARTS_DIR / "interactive").mkdir(exist_ok=True)

print("=" * 60)
print("生成SIMULATION模块时间序列演化图表")
print("=" * 60)

# 加载数据
print("\n[1/4] 加载时间序列数据...")
time_series_df = pd.read_csv(OUTPUT_DIR / "all_scenarios_time_series.csv")
scenario_comparison = pd.read_csv(OUTPUT_DIR / "scenario_comparison.csv")

print(f"   ✓ 时间序列数据: {len(time_series_df)} 行")
print(f"   ✓ 场景数量: {time_series_df['scenario_name'].nunique()} 个")

# 生成交互式失业率演化图
print("\n[2/4] 生成失业率动态演化交互式图表...")

fig = go.Figure()

# 为每个场景添加曲线
scenarios = time_series_df['scenario_name'].unique()
colors = {'baseline': '#8b5cf6', 'training_low': '#3b82f6', 'training_high': '#10b981'}
names = {'baseline': '基准场景', 'training_low': '低强度培训', 'training_high': '高强度培训'}

for scenario in scenarios:
    df_scenario = time_series_df[time_series_df['scenario_name'] == scenario]
    
    fig.add_trace(go.Scatter(
        x=df_scenario['iteration'],
        y=df_scenario['unemployment_rate'] * 100,
        mode='lines+markers',
        name=names.get(scenario, scenario),
        line=dict(color=colors.get(scenario, '#666666'), width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{fullData.name}</b><br>迭代: %{x}<br>失业率: %{y:.2f}%<extra></extra>'
    ))

fig.update_layout(
    title=dict(
        text='失业率动态演化路径对比',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    xaxis=dict(title='迭代次数', gridcolor='rgba(139, 92, 246, 0.1)'),
    yaxis=dict(title='失业率 (%)', gridcolor='rgba(139, 92, 246, 0.1)', tickformat='.1f'),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=600,
    width=1200,
    margin=dict(l=80, r=50, b=80, t=100),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.5)'
)

output_path_unemployment = WEBSITE_CHARTS_DIR / "interactive" / "unemployment_evolution.html"
fig.write_html(str(output_path_unemployment))
print(f"   ✓ 保存到: {output_path_unemployment.relative_to(BASE_DIR)}")

# 生成工资演化对比图
print("\n[3/4] 生成工资演化对比交互式图表...")

fig_wage = go.Figure()

for scenario in scenarios:
    df_scenario = time_series_df[time_series_df['scenario_name'] == scenario]
    
    fig_wage.add_trace(go.Scatter(
        x=df_scenario['iteration'],
        y=df_scenario['mean_W'],
        mode='lines+markers',
        name=names.get(scenario, scenario),
        line=dict(color=colors.get(scenario, '#666666'), width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{fullData.name}</b><br>迭代: %{x}<br>平均工资: ¥%{y:.0f}<extra></extra>'
    ))

fig_wage.update_layout(
    title=dict(
        text='平均工资动态演化路径对比',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    xaxis=dict(title='迭代次数', gridcolor='rgba(139, 92, 246, 0.1)'),
    yaxis=dict(title='平均工资 (元)', gridcolor='rgba(139, 92, 246, 0.1)'),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=600,
    width=1200,
    margin=dict(l=80, r=50, b=80, t=100),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.5)'
)

output_path_wage = WEBSITE_CHARTS_DIR / "interactive" / "wage_evolution.html"
fig_wage.write_html(str(output_path_wage))
print(f"   ✓ 保存到: {output_path_wage.relative_to(BASE_DIR)}")

# 生成静态工资分布对比图
print("\n[4/4] 生成工资分布对比静态图表...")

fig_dist, axes = plt.subplots(1, 3, figsize=(18, 5))
fig_dist.suptitle('各场景工资分布对比', fontsize=20, fontweight='bold', color='#8b5cf6')

for idx, scenario in enumerate(scenarios):
    # 加载个体数据
    scenario_dir = OUTPUT_DIR / f"scenario_{scenario}"
    if scenario_dir.exists():
        individuals_df = pd.read_csv(scenario_dir / "equilibrium_individuals.csv")
        
        # 绘制工资分布直方图
        axes[idx].hist(individuals_df['W'], bins=50, color=colors.get(scenario, '#666666'), 
                      alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[idx].set_title(names.get(scenario, scenario), fontsize=16, fontweight='bold')
        axes[idx].set_xlabel('工资 (元)', fontsize=12)
        axes[idx].set_ylabel('频数', fontsize=12)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        
        # 添加均值线
        mean_wage = individuals_df['W'].mean()
        axes[idx].axvline(mean_wage, color='red', linestyle='--', linewidth=2, 
                         label=f'均值: ¥{mean_wage:.0f}')
        axes[idx].legend()

plt.tight_layout()
output_path_dist = WEBSITE_CHARTS_DIR / "static" / "wage_distribution_comparison.png"
plt.savefig(output_path_dist, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✓ 保存到: {output_path_dist.relative_to(BASE_DIR)}")

# 生成技能演化对比图
print("\n[额外] 生成技能演化对比交互式图表...")

fig_skill = go.Figure()

for scenario in scenarios:
    df_scenario = time_series_df[time_series_df['scenario_name'] == scenario]
    
    fig_skill.add_trace(go.Scatter(
        x=df_scenario['iteration'],
        y=df_scenario['mean_S'],
        mode='lines+markers',
        name=names.get(scenario, scenario),
        line=dict(color=colors.get(scenario, '#666666'), width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{fullData.name}</b><br>迭代: %{x}<br>平均技能: %{y:.2f}<extra></extra>'
    ))

fig_skill.update_layout(
    title=dict(
        text='平均技能水平动态演化路径对比',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    xaxis=dict(title='迭代次数', gridcolor='rgba(139, 92, 246, 0.1)'),
    yaxis=dict(title='平均技能水平', gridcolor='rgba(139, 92, 246, 0.1)'),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=600,
    width=1200,
    margin=dict(l=80, r=50, b=80, t=100),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.5)'
)

output_path_skill = WEBSITE_CHARTS_DIR / "interactive" / "skill_evolution.html"
fig_skill.write_html(str(output_path_skill))
print(f"   ✓ 保存到: {output_path_skill.relative_to(BASE_DIR)}")

print("\n" + "=" * 60)
print("✅ SIMULATION模块时间序列图表生成完成!")
print("=" * 60)
print(f"\n生成的文件:")
print(f"  1. {output_path_unemployment.relative_to(BASE_DIR)}")
print(f"  2. {output_path_wage.relative_to(BASE_DIR)}")
print(f"  3. {output_path_dist.relative_to(BASE_DIR)}")
print(f"  4. {output_path_skill.relative_to(BASE_DIR)}")
print(f"\n这些图表已可在网站中使用!")

