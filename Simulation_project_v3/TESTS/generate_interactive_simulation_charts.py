"""
生成SIMULATION模块的交互式政策对比图表
用于网站展示
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# 设置路径
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "OUTPUT" / "simulation"
WEBSITE_CHARTS_DIR = BASE_DIR / "WEBSITE" / "charts" / "simulation"
WEBSITE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
(WEBSITE_CHARTS_DIR / "static").mkdir(exist_ok=True)
(WEBSITE_CHARTS_DIR / "interactive").mkdir(exist_ok=True)

print("=" * 60)
print("生成SIMULATION模块交互式政策对比图表")
print("=" * 60)

# 加载数据
print("\n[1/3] 加载政策仿真数据...")
scenario_comparison = pd.read_csv(OUTPUT_DIR / "scenario_comparison.csv")
policy_effects = pd.read_csv(OUTPUT_DIR / "policy_effects_vs_baseline.csv")

print(f"   ✓ 场景对比数据: {len(scenario_comparison)} 个场景")
print(f"   ✓ 政策效应数据: {len(policy_effects)} 个场景")

# 生成政策对比柱状图
print("\n[2/3] 生成政策对比交互式柱状图...")

# 准备数据
scenarios = scenario_comparison['scenario_display_name'].tolist()
unemployment_rates = scenario_comparison['unemployment_rate'].tolist()
avg_skills = scenario_comparison['mean_S'].tolist()
avg_wages = scenario_comparison['mean_W'].tolist()

# 创建子图
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('失业率对比', '平均技能水平对比', '平均工资对比'),
    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
)

# 失业率对比
fig.add_trace(
    go.Bar(
        x=scenarios,
        y=unemployment_rates,
        name='失业率',
        marker=dict(color='#8b5cf6'),
        hovertemplate='<b>%{x}</b><br>失业率: %{y:.2%}<extra></extra>'
    ),
    row=1, col=1
)

# 平均技能对比
fig.add_trace(
    go.Bar(
        x=scenarios,
        y=avg_skills,
        name='平均技能',
        marker=dict(color='#3b82f6'),
        hovertemplate='<b>%{x}</b><br>平均技能: %{y:.2f}<extra></extra>'
    ),
    row=1, col=2
)

# 平均工资对比
fig.add_trace(
    go.Bar(
        x=scenarios,
        y=avg_wages,
        name='平均工资',
        marker=dict(color='#10b981'),
        hovertemplate='<b>%{x}</b><br>平均工资: ¥%{y:.0f}<extra></extra>'
    ),
    row=1, col=3
)

# 更新布局
fig.update_layout(
    title=dict(
        text='政策场景对比分析',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    showlegend=False,
    height=500,
    width=1400,
    margin=dict(l=50, r=50, b=100, t=100),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# 更新坐标轴
fig.update_yaxes(title_text="失业率", row=1, col=1, tickformat='.1%')
fig.update_yaxes(title_text="技能水平", row=1, col=2)
fig.update_yaxes(title_text="工资(元)", row=1, col=3)

output_path_comparison = WEBSITE_CHARTS_DIR / "interactive" / "policy_comparison.html"
fig.write_html(str(output_path_comparison))
print(f"   ✓ 保存到: {output_path_comparison.relative_to(BASE_DIR)}")

# 生成政策效应雷达图
print("\n[3/3] 生成政策效应雷达图...")

if len(policy_effects) > 0:
    # 准备雷达图数据
    categories = ['失业率改善', '技能提升', '工资增长', '就业率提升', '数字素养提升']
    
    fig_radar = go.Figure()
    
    for idx, row in policy_effects.iterrows():
        scenario_name = row.get('scenario_display_name', row.get('scenario_name', f'场景{idx+1}'))

        # 计算各维度的改善百分比(假设数据中有这些列)
        values = [
            abs(row.get('unemployment_rate_change', row.get('delta_unemployment_rate', 0))) * 100,
            row.get('mean_S_change', row.get('delta_mean_S', 0)) * 10,
            row.get('mean_W_change', row.get('delta_mean_W', 0)) / 100,
            abs(row.get('unemployment_rate_change', row.get('delta_unemployment_rate', 0))) * 100,
            row.get('mean_D_change', row.get('delta_mean_D', 0)) * 10
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=scenario_name,
            hovertemplate='<b>%{theta}</b><br>改善度: %{r:.2f}<extra></extra>'
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max(v) for v in [values]]) * 1.2]
            )
        ),
        title=dict(
            text='政策效应多维度评估',
            font=dict(size=24, color='#8b5cf6', family='Arial Black')
        ),
        showlegend=True,
        height=600,
        width=800,
        margin=dict(l=80, r=80, b=80, t=100),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    output_path_radar = WEBSITE_CHARTS_DIR / "interactive" / "policy_effects_radar.html"
    fig_radar.write_html(str(output_path_radar))
    print(f"   ✓ 保存到: {output_path_radar.relative_to(BASE_DIR)}")

print("\n" + "=" * 60)
print("✅ SIMULATION模块交互式图表生成完成!")
print("=" * 60)
print(f"\n生成的文件:")
print(f"  1. {output_path_comparison.relative_to(BASE_DIR)}")
if len(policy_effects) > 0:
    print(f"  2. {output_path_radar.relative_to(BASE_DIR)}")
print(f"\n这些图表已可在网站中使用!")

