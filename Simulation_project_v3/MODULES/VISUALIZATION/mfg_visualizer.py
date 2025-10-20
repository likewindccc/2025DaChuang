"""
MFG模块可视化器

提供价值函数、最优策略、人口演化、收敛性等可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy import stats

from .style_config import COLORS, FIGURE_SIZE, setup_matplotlib_style


class MFGVisualizer:
    """MFG可视化类"""
    
    def __init__(self, output_dir: Path):
        """
        初始化
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures' / 'mfg'
        self.interactive_dir = self.output_dir / 'interactive' / 'mfg'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        setup_matplotlib_style()
    
    def plot_convergence_curves(
        self,
        convergence_history: pd.DataFrame,
        save_name: str = 'MFG_convergence_curves'
    ) -> str:
        """
        绘制MFG收敛曲线
        
        参数:
            convergence_history: 收敛历史数据，包含列: iteration, diff_V, diff_a, diff_u
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('MFG均衡收敛过程', fontsize=16, fontweight='bold')
        
        iterations = convergence_history['iteration'].values
        
        # 价值函数相对变化
        ax1 = axes[0]
        diff_V = convergence_history['diff_V'].values
        ax1.semilogy(iterations, diff_V, color=COLORS['primary'],
                    linewidth=2, marker='o', markersize=4)
        ax1.axhline(0.01, color='red', linestyle='--', linewidth=1.5,
                   label='收敛阈值 ε_V=0.01')
        ax1.set_ylabel('|ΔV| / |V|', fontweight='bold')
        ax1.set_title('价值函数相对变化', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 平均努力水平变化
        ax2 = axes[1]
        diff_a = convergence_history['diff_a'].values
        ax2.plot(iterations, diff_a, color=COLORS['secondary'],
                linewidth=2, marker='s', markersize=4)
        ax2.axhline(0.01, color='red', linestyle='--', linewidth=1.5,
                   label='收敛阈值 ε_a=0.01')
        ax2.set_ylabel('|Δmean(a)|', fontweight='bold')
        ax2.set_title('平均努力水平变化', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 失业率变化
        ax3 = axes[2]
        diff_u = convergence_history['diff_u'].values
        ax3.plot(iterations, diff_u, color=COLORS['accent_pink'],
                linewidth=2, marker='^', markersize=4)
        ax3.axhline(0.001, color='red', linestyle='--', linewidth=1.5,
                   label='收敛阈值 ε_u=0.001')
        ax3.set_xlabel('迭代次数', fontweight='bold')
        ax3.set_ylabel('|Δu|', fontweight='bold')
        ax3.set_title('失业率变化', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ MFG收敛曲线可视化完成: {static_path}")
        return str(static_path)
    
    def plot_value_function_heatmap(
        self,
        individuals: pd.DataFrame,
        value_type: str = 'V_U',
        save_name: Optional[str] = None
    ) -> str:
        """
        绘制价值函数热力图（固定D和W，展示T-S平面）
        
        参数:
            individuals: 个体数据，包含T, S, D, W, V_U, V_E列
            value_type: 价值函数类型 ('V_U', 'V_E', 'delta_V')
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        if save_name is None:
            save_name = f'MFG_value_function_{value_type}_heatmap'
        
        # 选择4个固定的(D, W)组合
        D_levels = np.percentile(individuals['D'].values, [25, 50, 75, 90])
        W_levels = np.percentile(individuals['W'].values, [25, 50, 75, 90])
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE['grid_2x2'])
        title_map = {
            'V_U': '失业价值函数',
            'V_E': '就业价值函数',
            'delta_V': '价值函数差异 (V_E - V_U)'
        }
        fig.suptitle(f'{title_map[value_type]} 热力图', fontsize=16, fontweight='bold')
        
        for idx, (D_val, W_val) in enumerate(zip(D_levels, W_levels)):
            ax = axes[idx // 2, idx % 2]
            
            # 筛选接近指定D和W的个体
            mask = (
                (np.abs(individuals['D'] - D_val) < 0.1) &
                (np.abs(individuals['W'] - W_val) < 500)
            )
            subset = individuals[mask]
            
            if len(subset) > 10:
                # 创建网格数据
                T_bins = np.linspace(subset['T'].min(), subset['T'].max(), 30)
                S_bins = np.linspace(subset['S'].min(), subset['S'].max(), 30)
                
                # 计算价值
                if value_type == 'delta_V':
                    subset['value'] = subset['V_E'] - subset['V_U']
                else:
                    subset['value'] = subset[value_type]
                
                # 二维直方图统计
                H, T_edges, S_edges = np.histogram2d(
                    subset['T'].values,
                    subset['S'].values,
                    bins=[T_bins, S_bins],
                    weights=subset['value'].values
                )
                counts, _, _ = np.histogram2d(
                    subset['T'].values,
                    subset['S'].values,
                    bins=[T_bins, S_bins]
                )
                H = np.divide(H, counts, where=counts > 0)
                
                # 绘制热力图
                im = ax.imshow(H.T, origin='lower', aspect='auto',
                             cmap='RdYlBu_r', interpolation='bilinear',
                             extent=[T_edges[0], T_edges[-1], S_edges[0], S_edges[-1]])
                plt.colorbar(im, ax=ax, label=value_type)
                
                ax.set_xlabel('T (劳动供给时间)')
                ax.set_ylabel('S (技能水平)')
                ax.set_title(f'D={D_val:.2f}, W={W_val:.0f}')
            else:
                ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 价值函数热力图可视化完成: {static_path}")
        return str(static_path)
    
    def plot_optimal_effort_distribution(
        self,
        individuals: pd.DataFrame,
        save_name: str = 'MFG_optimal_effort_distribution'
    ) -> str:
        """
        绘制最优努力水平分布
        
        参数:
            individuals: 个体数据，包含a_optimal列
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['single'])
        
        effort = individuals['a_optimal'].values
        
        # 直方图
        ax.hist(effort, bins=50, density=True, alpha=0.6,
               color=COLORS['primary'], edgecolor='white', linewidth=0.5)
        
        # KDE
        kde = stats.gaussian_kde(effort)
        x_range = np.linspace(effort.min(), effort.max(), 200)
        ax.plot(x_range, kde(x_range), color=COLORS['secondary'],
               linewidth=2.5, label='KDE')
        
        # 统计信息
        mean_val = effort.mean()
        median_val = np.median(effort)
        std_val = effort.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                  label=f'均值={mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5,
                  label=f'中位数={median_val:.3f}')
        
        ax.set_xlabel('最优努力水平 a*', fontweight='bold')
        ax.set_ylabel('密度', fontweight='bold')
        ax.set_title(f'最优努力水平分布 (均值={mean_val:.3f}, 标准差={std_val:.3f})',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 最优努力分布可视化完成: {static_path}")
        return str(static_path)
    
    def plot_population_evolution(
        self,
        population_history: List[pd.DataFrame],
        variable: str = 'T',
        save_name: Optional[str] = None
    ) -> str:
        """
        绘制人口分布演化
        
        参数:
            population_history: 人口历史列表（每个元素是一个DataFrame）
            variable: 状态变量 ('T', 'S', 'D', 'W')
            save_name: 保存文件名
        
        返回:
            静态图路径
        """
        if save_name is None:
            save_name = f'MFG_{variable}_distribution_evolution'
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])
        
        # 选择若干迭代展示
        n_history = len(population_history)
        indices = np.linspace(0, n_history - 1, min(6, n_history), dtype=int)
        
        for i, idx in enumerate(indices):
            data = population_history[idx][variable].values
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            
            alpha = 0.3 + 0.7 * (i / len(indices))
            ax.plot(x_range, kde(x_range),
                   label=f'迭代 {idx}',
                   linewidth=2,
                   alpha=alpha)
            
            # 最后一次迭代的均值线
            if idx == indices[-1]:
                mean_val = data.mean()
                ax.axvline(mean_val, color='red', linestyle='--',
                          linewidth=1.5, label=f'最终均值={mean_val:.2f}')
        
        ax.set_xlabel(variable, fontweight='bold', fontsize=12)
        ax.set_ylabel('密度', fontweight='bold', fontsize=12)
        ax.set_title(f'{variable} 分布演化', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 人口演化可视化完成: {static_path}")
        return str(static_path)
    
    def create_interactive_value_function_3d(
        self,
        individuals: pd.DataFrame,
        value_type: str = 'V_U',
        save_name: Optional[str] = None
    ) -> str:
        """
        创建3D交互式价值函数图
        
        参数:
            individuals: 个体数据
            value_type: 价值函数类型
            save_name: 保存文件名
        
        返回:
            HTML文件路径
        """
        if save_name is None:
            save_name = f'MFG_value_function_{value_type}_3D'
        
        # 采样数据（避免过多点）
        sample_size = min(2000, len(individuals))
        sample = individuals.sample(n=sample_size, random_state=42)
        
        if value_type == 'delta_V':
            sample['value'] = sample['V_E'] - sample['V_U']
        else:
            sample['value'] = sample[value_type]
        
        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=sample['T'].values,
            y=sample['S'].values,
            z=sample['value'].values,
            mode='markers',
            marker=dict(
                size=3,
                color=sample['value'].values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=value_type)
            ),
            text=[f'T={t:.1f}<br>S={s:.2f}<br>D={d:.2f}<br>W={w:.0f}<br>{value_type}={v:.2f}'
                  for t, s, d, w, v in zip(sample['T'], sample['S'], sample['D'], sample['W'], sample['value'])],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'{value_type} 3D可视化',
            scene=dict(
                xaxis_title='T (劳动供给时间)',
                yaxis_title='S (技能水平)',
                zaxis_title=value_type
            ),
            height=700
        )
        
        interactive_path = self.interactive_dir / f'{save_name}.html'
        fig.write_html(str(interactive_path))
        
        print(f"✓ 3D价值函数可视化完成: {interactive_path}")
        return str(interactive_path)

