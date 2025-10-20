"""
数据可视化器

提供初始人口分布、企业需求分布等数据可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Tuple, List
from scipy import stats

from .style_config import COLORS, FIGURE_SIZE, setup_matplotlib_style


class DataVisualizer:
    """数据可视化类"""
    
    def __init__(self, output_dir: Path):
        """
        初始化
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures' / 'data'
        self.interactive_dir = self.output_dir / 'interactive' / 'data'
        
        # 创建输出目录
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        setup_matplotlib_style()
    
    def plot_initial_distribution(
        self,
        individuals: pd.DataFrame,
        save_name: str = 'DATA_initial_distribution'
    ) -> Tuple[str, str]:
        """
        绘制初始人口分布（静态图 + 交互式图）
        
        参数:
            individuals: 个体数据
            save_name: 保存文件名
        
        返回:
            (静态图路径, 交互式图路径)
        """
        # === 静态图（Matplotlib）===
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE['grid_2x2'])
        fig.suptitle('初始人口状态变量分布', fontsize=16, fontweight='bold')
        
        variables = ['T', 'S', 'D', 'W']
        titles = [
            'T - 劳动供给时间（小时/周）',
            'S - 技能水平',
            'D - 数字素养',
            'W - 工资期望（元/月）'
        ]
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            ax = axes[i // 2, i % 2]
            data = individuals[var].values
            
            # 绘制直方图 + KDE
            ax.hist(data, bins=50, density=True, alpha=0.6,
                   color=COLORS['primary'], edgecolor='white', linewidth=0.5)
            
            # KDE曲线
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), color=COLORS['secondary'],
                   linewidth=2.5, label='KDE')
            
            # 统计信息
            mean_val = data.mean()
            median_val = np.median(data)
            ax.axvline(mean_val, color=COLORS['accent_pink'], linestyle='--',
                      linewidth=1.5, label=f'均值={mean_val:.2f}')
            ax.axvline(median_val, color=COLORS['accent_green'], linestyle=':',
                      linewidth=1.5, label=f'中位数={median_val:.2f}')
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(var)
            ax.set_ylabel('密度')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # === 交互式图（Plotly）===
        fig_interactive = make_subplots(
            rows=2, cols=2,
            subplot_titles=titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            row = i // 2 + 1
            col = i % 2 + 1
            data = individuals[var].values
            
            # 添加直方图
            fig_interactive.add_trace(
                go.Histogram(
                    x=data,
                    name=var,
                    nbinsx=50,
                    opacity=0.7,
                    marker_color=COLORS['primary'],
                    histnorm='probability density',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 添加KDE曲线
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            fig_interactive.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode='lines',
                    name=f'{var} KDE',
                    line=dict(color=COLORS['secondary'], width=3),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # 更新坐标轴
            fig_interactive.update_xaxes(title_text=var, row=row, col=col)
            fig_interactive.update_yaxes(title_text='密度', row=row, col=col)
        
        fig_interactive.update_layout(
            title_text='初始人口状态变量分布（交互式）',
            height=800,
            showlegend=False
        )
        
        interactive_path = self.interactive_dir / f'{save_name}.html'
        fig_interactive.write_html(str(interactive_path))
        
        print(f"✓ 初始分布可视化完成:")
        print(f"  静态图: {static_path}")
        print(f"  交互式: {interactive_path}")
        
        return str(static_path), str(interactive_path)
    
    def plot_copula_structure(
        self,
        individuals: pd.DataFrame,
        save_name: str = 'DATA_copula_structure'
    ) -> str:
        """
        绘制Copula结构（状态变量两两相关性）
        
        参数:
            individuals: 个体数据
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        variables = ['T', 'S', 'D', 'W']
        n_vars = len(variables)
        
        fig, axes = plt.subplots(n_vars - 1, n_vars - 1,
                                figsize=(12, 12))
        fig.suptitle('状态变量Copula结构分析', fontsize=16, fontweight='bold')
        
        for i in range(n_vars - 1):
            for j in range(n_vars - 1):
                ax = axes[i, j]
                
                if j > i:
                    # 上三角：散点图 + 密度
                    var_x = variables[j + 1]
                    var_y = variables[i]
                    x = individuals[var_x].values
                    y = individuals[var_y].values
                    
                    # 散点图（带透明度）
                    ax.hexbin(x, y, gridsize=30, cmap='Purples', alpha=0.7)
                    
                    # Spearman相关系数
                    corr, _ = stats.spearmanr(x, y)
                    ax.text(0.05, 0.95, f'ρ={corr:.3f}',
                           transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.8))
                    
                    ax.set_xlabel(var_x, fontsize=9)
                    ax.set_ylabel(var_y, fontsize=9)
                
                elif j == i:
                    # 对角线：密度图
                    var = variables[i + 1]
                    data = individuals[var].values
                    ax.hist(data, bins=30, color=COLORS['primary'],
                           alpha=0.7, edgecolor='white')
                    ax.set_ylabel('频数', fontsize=9)
                    ax.set_title(var, fontweight='bold')
                
                else:
                    # 下三角：隐藏
                    ax.axis('off')
                
                ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Copula结构可视化完成: {static_path}")
        return str(static_path)
    
    def plot_enterprise_demand(
        self,
        enterprises: pd.DataFrame,
        save_name: str = 'DATA_enterprise_demand'
    ) -> str:
        """
        绘制企业需求分布
        
        参数:
            enterprises: 企业数据
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE['grid_2x2'])
        fig.suptitle('企业需求向量分布', fontsize=16, fontweight='bold')
        
        # 假设企业数据列名为 sigma_T, sigma_S, sigma_D, sigma_W
        variables = ['sigma_T', 'sigma_S', 'sigma_D', 'sigma_W']
        titles = [
            'σ_T - 工作时间要求',
            'σ_S - 技能要求',
            'σ_D - 数字素养要求',
            'σ_W - 岗位工资'
        ]
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            ax = axes[i // 2, i % 2]
            
            if var in enterprises.columns:
                data = enterprises[var].values
                
                # 直方图
                ax.hist(data, bins=40, density=True, alpha=0.6,
                       color=COLORS['accent_orange'], edgecolor='white')
                
                # KDE
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_range, kde(x_range), color=COLORS['secondary'],
                       linewidth=2.5)
                
                mean_val = data.mean()
                ax.axvline(mean_val, color='red', linestyle='--',
                          label=f'均值={mean_val:.2f}')
                
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel(var)
                ax.set_ylabel('密度')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{var}\n数据缺失',
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 企业需求分布可视化完成: {static_path}")
        return str(static_path)
    
    def plot_laborer_enterprise_comparison(
        self,
        laborers: pd.DataFrame,
        enterprises: pd.DataFrame,
        save_name: str = 'DATA_laborer_enterprise_comparison'
    ) -> str:
        """
        绘制劳动者-企业对比图
        
        参数:
            laborers: 劳动者数据
            enterprises: 企业数据
            save_name: 保存文件名
        
        返回:
            交互式图路径
        """
        # 创建交互式对比图（小提琴图）
        variables = ['T', 'S', 'D', 'W']
        titles = ['劳动供给/时间要求', '技能水平/要求', '数字素养/要求', '工资期望/岗位工资']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # 劳动者分布
            laborer_data = laborers[var].values
            fig.add_trace(
                go.Violin(
                    y=laborer_data,
                    name='劳动者',
                    marker_color=COLORS['primary'],
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.7,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # 企业需求分布（如果存在）
            enterprise_var = f'sigma_{var}'
            if enterprise_var in enterprises.columns:
                enterprise_data = enterprises[enterprise_var].values
                fig.add_trace(
                    go.Violin(
                        y=enterprise_data,
                        name='企业需求',
                        marker_color=COLORS['accent_orange'],
                        box_visible=True,
                        meanline_visible=True,
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
            
            fig.update_yaxes(title_text=var, row=row, col=col)
        
        fig.update_layout(
            title_text='劳动者特征 vs 企业需求对比',
            height=800,
            showlegend=True,
            violinmode='group'
        )
        
        interactive_path = self.interactive_dir / f'{save_name}.html'
        fig.write_html(str(interactive_path))
        
        print(f"✓ 劳动者-企业对比可视化完成: {interactive_path}")
        return str(interactive_path)

