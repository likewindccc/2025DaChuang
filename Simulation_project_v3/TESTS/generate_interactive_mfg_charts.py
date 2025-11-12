"""
生成MFG模块的3D交互式价值函数图表
用于网站展示
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import pickle

# 设置路径
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "OUTPUT" / "mfg"
WEBSITE_CHARTS_DIR = BASE_DIR / "WEBSITE" / "charts" / "mfg" / "interactive"
WEBSITE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("生成MFG模块3D交互式价值函数图表")
print("=" * 60)

# 加载价值函数数据
print("\n[1/3] 加载价值函数数据...")
value_df = pd.read_csv(OUTPUT_DIR / "value_distribution_full.csv")
print(f"   ✓ 加载了 {len(value_df)} 条价值函数记录")

# 准备数据 - 选择中位数的D和W
D_median = value_df['D'].median()
W_median = value_df['W'].median()

# 筛选数据
df_filtered = value_df[
    (value_df['D'].between(D_median - 2, D_median + 2)) &
    (value_df['W'].between(W_median - 500, W_median + 500))
].copy()

print(f"   ✓ 筛选后数据量: {len(df_filtered)} 条")
print(f"   ✓ D中位数: {D_median:.2f}, W中位数: {W_median:.2f}")

# 创建网格数据
T_unique = sorted(df_filtered['T'].unique())
S_unique = sorted(df_filtered['S'].unique())

# 如果数据点太多,进行采样
if len(T_unique) > 30:
    T_unique = T_unique[::len(T_unique)//30]
if len(S_unique) > 30:
    S_unique = S_unique[::len(S_unique)//30]

print(f"   ✓ T网格点数: {len(T_unique)}, S网格点数: {len(S_unique)}")

# 创建网格
T_grid, S_grid = np.meshgrid(T_unique, S_unique)
V_U_grid = np.zeros_like(T_grid, dtype=float)
V_E_grid = np.zeros_like(T_grid, dtype=float)

# 填充网格数据
for i, s in enumerate(S_unique):
    for j, t in enumerate(T_unique):
        # 找到最接近的数据点
        mask = (
            (df_filtered['T'].between(t - 2, t + 2)) &
            (df_filtered['S'].between(s - 2, s + 2))
        )
        if mask.sum() > 0:
            V_U_grid[i, j] = df_filtered.loc[mask, 'V_U'].mean()
            V_E_grid[i, j] = df_filtered.loc[mask, 'V_E'].mean()
        else:
            V_U_grid[i, j] = np.nan
            V_E_grid[i, j] = np.nan

print(f"   ✓ 网格数据填充完成")

# 生成失业价值函数3D图
print("\n[2/3] 生成失业价值函数V_U 3D图...")
fig_V_U = go.Figure(data=[go.Surface(
    x=T_grid,
    y=S_grid,
    z=V_U_grid,
    colorscale='Viridis',
    colorbar=dict(title=dict(text="V_U"), tickmode="linear", tick0=0, dtick=500),
    hovertemplate='<b>T</b>: %{x:.1f}h<br><b>S</b>: %{y:.1f}<br><b>V_U</b>: %{z:.2f}<extra></extra>'
)])

fig_V_U.update_layout(
    title=dict(
        text='失业价值函数 V<sub>U</sub>(T, S)',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    scene=dict(
        xaxis=dict(title='工作时长 T (小时/周)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        yaxis=dict(title='技能水平 S', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        zaxis=dict(title='失业价值 V_U', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    width=1000,
    height=700,
    margin=dict(l=0, r=0, b=0, t=50),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

output_path_V_U = WEBSITE_CHARTS_DIR / "value_function_V_U_3D.html"
fig_V_U.write_html(str(output_path_V_U))
print(f"   ✓ 保存到: {output_path_V_U}")

# 生成就业价值函数3D图
print("\n[3/3] 生成就业价值函数V_E 3D图...")
fig_V_E = go.Figure(data=[go.Surface(
    x=T_grid,
    y=S_grid,
    z=V_E_grid,
    colorscale='Plasma',
    colorbar=dict(title=dict(text="V_E"), tickmode="linear", tick0=0, dtick=500),
    hovertemplate='<b>T</b>: %{x:.1f}h<br><b>S</b>: %{y:.1f}<br><b>V_E</b>: %{z:.2f}<extra></extra>'
)])

fig_V_E.update_layout(
    title=dict(
        text='就业价值函数 V<sub>E</sub>(T, S)',
        font=dict(size=24, color='#8b5cf6', family='Arial Black')
    ),
    scene=dict(
        xaxis=dict(title='工作时长 T (小时/周)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        yaxis=dict(title='技能水平 S', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        zaxis=dict(title='就业价值 V_E', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    width=1000,
    height=700,
    margin=dict(l=0, r=0, b=0, t=50),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

output_path_V_E = WEBSITE_CHARTS_DIR / "value_function_V_E_3D.html"
fig_V_E.write_html(str(output_path_V_E))
print(f"   ✓ 保存到: {output_path_V_E}")

print("\n" + "=" * 60)
print("✅ 所有3D交互式图表生成完成!")
print("=" * 60)
print(f"\n生成的文件:")
print(f"  1. {output_path_V_U.relative_to(BASE_DIR)}")
print(f"  2. {output_path_V_E.relative_to(BASE_DIR)}")
print(f"\n这些图表已可在网站中使用!")

