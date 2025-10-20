"""
仪表盘构建器

整合多个可视化，创建交互式仪表盘
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, List

from .style_config import COLORS


class DashboardBuilder:
    """仪表盘构建类"""
    
    def __init__(self, output_dir: Path):
        """初始化"""
        self.output_dir = Path(output_dir)
        self.dashboard_dir = self.output_dir / 'dashboards'
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    def build_mfg_dashboard(
        self,
        individuals: pd.DataFrame,
        convergence_history: pd.DataFrame,
        save_name: str = 'DASHBOARD_mfg_equilibrium'
    ) -> str:
        """
        构建MFG均衡仪表盘
        
        参数:
            individuals: 个体数据
            convergence_history: 收敛历史
            save_name: 保存文件名
        
        返回:
            HTML文件路径
        """
        # 创建子图布局
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '价值函数分布',
                '最优努力水平分布',
                '收敛曲线',
                '状态变量联合分布'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter3d'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. 价值函数分布
        fig.add_trace(
            go.Histogram(
                x=individuals['V_U'].values,
                name='V_U',
                marker_color=COLORS['primary'],
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # 2. 最优努力分布
        fig.add_trace(
            go.Histogram(
                x=individuals['a_optimal'].values,
                name='a*',
                marker_color=COLORS['secondary'],
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # 3. 收敛曲线
        fig.add_trace(
            go.Scatter(
                x=convergence_history['iteration'].values,
                y=convergence_history['diff_V'].values,
                mode='lines+markers',
                name='|ΔV|/|V|',
                line=dict(color=COLORS['accent_pink'], width=2)
            ),
            row=2, col=1
        )
        
        # 4. 状态变量3D散点图（采样）
        sample = individuals.sample(n=min(1000, len(individuals)), random_state=42)
        fig.add_trace(
            go.Scatter3d(
                x=sample['T'].values,
                y=sample['S'].values,
                z=sample['D'].values,
                mode='markers',
                marker=dict(
                    size=3,
                    color=sample['W'].values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='W', x=1.15)
                ),
                name='个体分布'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text='MFG均衡求解仪表盘',
            showlegend=True,
            height=900
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text='V_U', row=1, col=1)
        fig.update_xaxes(title_text='a*', row=1, col=2)
        fig.update_xaxes(title_text='迭代次数', row=2, col=1)
        
        fig.update_yaxes(title_text='频数', row=1, col=1)
        fig.update_yaxes(title_text='频数', row=1, col=2)
        fig.update_yaxes(title_text='|ΔV|/|V|', row=2, col=1)
        
        dashboard_path = self.dashboard_dir / f'{save_name}.html'
        fig.write_html(str(dashboard_path))
        
        print(f"✓ MFG仪表盘创建完成: {dashboard_path}")
        return str(dashboard_path)
    
    def build_calibration_dashboard(
        self,
        objective_history: pd.DataFrame,
        parameter_history: pd.DataFrame,
        moment_fit: Dict[str, Dict[str, float]],
        save_name: str = 'DASHBOARD_calibration'
    ) -> str:
        """
        构建校准过程仪表盘
        
        参数:
            objective_history: 目标函数历史
            parameter_history: 参数历史
            moment_fit: 矩拟合数据 {'target': {...}, 'simulated': {...}}
            save_name: 保存文件名
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'SMM目标函数收敛',
                '关键参数轨迹',
                '矩拟合质量',
                '参数空间探索'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. SMM目标函数
        fig.add_trace(
            go.Scatter(
                x=objective_history['evaluation'].values,
                y=objective_history['smm_distance'].values,
                mode='lines',
                name='SMM距离',
                line=dict(color=COLORS['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # 2. 关键参数轨迹（选择rho, kappa）
        for param, color in [('rho', COLORS['secondary']), ('kappa', COLORS['accent_pink'])]:
            if param in parameter_history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=parameter_history['evaluation'].values,
                        y=parameter_history[param].values,
                        mode='lines',
                        name=param,
                        line=dict(color=color, width=2)
                    ),
                    row=1, col=2
                )
        
        # 3. 矩拟合对比
        moment_names = list(moment_fit['target'].keys())
        target_values = [moment_fit['target'][m] for m in moment_names]
        simulated_values = [moment_fit['simulated'][m] for m in moment_names]
        
        x_pos = np.arange(len(moment_names))
        fig.add_trace(
            go.Bar(
                x=moment_names,
                y=target_values,
                name='目标矩',
                marker_color=COLORS['primary'],
                opacity=0.7
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=moment_names,
                y=simulated_values,
                name='模拟矩',
                marker_color=COLORS['accent_orange'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. 参数空间（前两个参数的2D散点）
        param_cols = [c for c in parameter_history.columns
                     if c not in ['evaluation', 'smm_distance']]
        if len(param_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=parameter_history[param_cols[0]].values,
                    y=parameter_history[param_cols[1]].values,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=parameter_history['smm_distance'].values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='SMM', x=1.15)
                    ),
                    name='评估点'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text='SMM校准过程监控仪表盘',
            showlegend=True,
            height=900
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text='评估次数', row=1, col=1)
        fig.update_xaxes(title_text='评估次数', row=1, col=2)
        fig.update_xaxes(title_text='统计矩', row=2, col=1)
        if len(param_cols) >= 2:
            fig.update_xaxes(title_text=param_cols[0], row=2, col=2)
            fig.update_yaxes(title_text=param_cols[1], row=2, col=2)
        
        fig.update_yaxes(title_text='SMM距离', row=1, col=1)
        fig.update_yaxes(title_text='参数值', row=1, col=2)
        fig.update_yaxes(title_text='值', row=2, col=1)
        
        dashboard_path = self.dashboard_dir / f'{save_name}.html'
        fig.write_html(str(dashboard_path))
        
        print(f"✓ 校准仪表盘创建完成: {dashboard_path}")
        return str(dashboard_path)
    
    def build_policy_dashboard(
        self,
        policy_results: Dict[str, Dict[str, float]],
        time_series: Dict[str, pd.DataFrame],
        save_name: str = 'DASHBOARD_policy_simulation'
    ) -> str:
        """
        构建政策仿真仪表盘
        
        参数:
            policy_results: 政策结果对比
            time_series: 时间序列数据
            save_name: 保存文件名
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '政策效果雷达图',
                '关键指标对比',
                '失业率时间序列',
                '成本-收益分析'
            ),
            specs=[
                [{'type': 'scatterpolar'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. 政策雷达图
        metrics = list(next(iter(policy_results.values())).keys())
        for i, (policy, results) in enumerate(policy_results.items()):
            values = [results[m] for m in metrics]
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=policy,
                    opacity=0.6
                ),
                row=1, col=1
            )
        
        # 2. 关键指标柱状图
        policies = list(policy_results.keys())
        unemployment_rates = [policy_results[p]['unemployment_rate'] for p in policies]
        
        fig.add_trace(
            go.Bar(
                x=policies,
                y=unemployment_rates,
                name='失业率',
                marker_color=COLORS['accent_pink']
            ),
            row=1, col=2
        )
        
        # 3. 时间序列
        for policy, data in time_series.items():
            fig.add_trace(
                go.Scatter(
                    x=data['time'].values,
                    y=data['unemployment_rate'].values,
                    mode='lines+markers',
                    name=policy
                ),
                row=2, col=1
            )
        
        # 4. 成本-收益（简化版）
        costs = [policy_results[p].get('cost', i) for i, p in enumerate(policies)]
        benefits = [policy_results[p].get('benefit', unemployment_rates[i] * 10)
                   for i, p in enumerate(policies)]
        
        fig.add_trace(
            go.Scatter(
                x=costs,
                y=benefits,
                mode='markers+text',
                text=policies,
                textposition='top center',
                marker=dict(size=15, color=COLORS['secondary']),
                name='政策'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text='政策仿真结果对比仪表盘',
            showlegend=True,
            height=900
        )
        
        fig.update_xaxes(title_text='政策', row=1, col=2)
        fig.update_xaxes(title_text='时间（期）', row=2, col=1)
        fig.update_xaxes(title_text='政策成本', row=2, col=2)
        
        fig.update_yaxes(title_text='失业率', row=1, col=2)
        fig.update_yaxes(title_text='失业率', row=2, col=1)
        fig.update_yaxes(title_text='收益', row=2, col=2)
        
        dashboard_path = self.dashboard_dir / f'{save_name}.html'
        fig.write_html(str(dashboard_path))
        
        print(f"✓ 政策仿真仪表盘创建完成: {dashboard_path}")
        return str(dashboard_path)

