"""
Simulation模块可视化器

提供政策仿真结果对比、时间序列演化、成本收益分析等可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, List

from .style_config import COLORS, FIGURE_SIZE, get_color_palette, setup_matplotlib_style


class SimulationVisualizer:
    """Simulation可视化类"""
    
    def __init__(self, output_dir: Path):
        """初始化"""
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures' / 'simulation'
        self.interactive_dir = self.output_dir / 'interactive' / 'simulation'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        setup_matplotlib_style()
    
    def plot_policy_comparison(
        self,
        policy_results: Dict[str, Dict[str, float]],
        save_name: str = 'SIMULATION_policy_comparison'
    ) -> str:
        """
        绘制政策效果对比
        
        参数:
            policy_results: {policy_name: {metric_name: value}}
            save_name: 保存文件名
        """
        metrics = list(next(iter(policy_results.values())).keys())
        policies = list(policy_results.keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(policies)
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])
        
        colors = get_color_palette(len(policies), 'policy')
        
        for i, policy in enumerate(policies):
            values = [policy_results[policy][m] for m in metrics]
            offset = (i - len(policies) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=policy,
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('评估指标', fontweight='bold')
        ax.set_ylabel('值', fontweight='bold')
        ax.set_title('政策效果对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 政策对比可视化完成: {static_path}")
        return str(static_path)
    
    def plot_time_series(
        self,
        time_series_data: Dict[str, pd.DataFrame],
        metric: str = 'unemployment_rate',
        save_name: Optional[str] = None
    ) -> str:
        """
        绘制时间序列演化
        
        参数:
            time_series_data: {policy_name: DataFrame with columns [time, metric]}
            metric: 指标名称
            save_name: 保存文件名
        """
        if save_name is None:
            save_name = f'SIMULATION_time_series_{metric}'
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])
        
        colors = get_color_palette(len(time_series_data), 'policy')
        
        for i, (policy, data) in enumerate(time_series_data.items()):
            time = data['time'].values
            values = data[metric].values
            
            ax.plot(time, values, linewidth=2.5, label=policy,
                   color=colors[i], marker='o', markersize=4)
        
        ax.set_xlabel('时间（期）', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{metric} 时间序列演化', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 时间序列可视化完成: {static_path}")
        return str(static_path)
    
    def plot_cost_benefit(
        self,
        policy_data: Dict[str, Dict[str, float]],
        save_name: str = 'SIMULATION_cost_benefit'
    ) -> str:
        """
        绘制成本-收益散点图
        
        参数:
            policy_data: {policy_name: {'cost': ..., 'benefit': ..., 'wage_increase': ...}}
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['single'])
        
        policies = list(policy_data.keys())
        costs = [policy_data[p]['cost'] for p in policies]
        benefits = [policy_data[p]['benefit'] for p in policies]
        sizes = [policy_data[p].get('wage_increase', 100) * 10 for p in policies]
        
        colors = get_color_palette(len(policies), 'policy')
        
        for i, policy in enumerate(policies):
            ax.scatter(costs[i], benefits[i], s=sizes[i],
                      c=[colors[i]], alpha=0.7, edgecolors='black',
                      linewidth=1.5, label=policy)
            ax.annotate(policy, (costs[i], benefits[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # 成本-收益平衡线
        max_cost = max(costs) * 1.1
        ax.plot([0, max_cost], [0, max_cost], 'k--', linewidth=1.5,
               alpha=0.5, label='成本=收益线')
        
        ax.set_xlabel('政策成本（归一化）', fontweight='bold')
        ax.set_ylabel('失业率降低幅度（%）', fontweight='bold')
        ax.set_title('政策成本-收益分析', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 成本-收益分析可视化完成: {static_path}")
        return str(static_path)
    
    def create_interactive_policy_radar(
        self,
        policy_results: Dict[str, Dict[str, float]],
        save_name: str = 'SIMULATION_policy_radar_interactive'
    ) -> str:
        """创建交互式政策雷达图"""
        metrics = list(next(iter(policy_results.values())).keys())
        
        fig = go.Figure()
        
        colors = get_color_palette(len(policy_results), 'policy')
        
        for i, (policy, results) in enumerate(policy_results.items()):
            values = [results[m] for m in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=policy,
                line_color=colors[i],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='政策效果雷达图（交互式）',
            height=600
        )
        
        interactive_path = self.interactive_dir / f'{save_name}.html'
        fig.write_html(str(interactive_path))
        
        print(f"✓ 交互式雷达图可视化完成: {interactive_path}")
        return str(interactive_path)

