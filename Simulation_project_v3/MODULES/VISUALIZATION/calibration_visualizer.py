"""
Calibration模块可视化器

提供SMM校准过程、参数收敛、矩拟合质量等可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, List

from .style_config import COLORS, FIGURE_SIZE, setup_matplotlib_style


class CalibrationVisualizer:
    """Calibration可视化类"""
    
    def __init__(self, output_dir: Path):
        """初始化"""
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures' / 'calibration'
        self.interactive_dir = self.output_dir / 'interactive' / 'calibration'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        setup_matplotlib_style()
    
    def plot_objective_history(
        self,
        objective_history: pd.DataFrame,
        save_name: str = 'CALIBRATION_objective_history'
    ) -> str:
        """绘制目标函数历史"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])
        
        evaluations = objective_history['evaluation'].values
        smm_distance = objective_history['smm_distance'].values
        
        ax.semilogy(evaluations, smm_distance, color=COLORS['primary'],
                   linewidth=1.5, alpha=0.7, label='SMM距离')
        
        # 最优值线
        best_smm = smm_distance.min()
        ax.axhline(best_smm, color='red', linestyle='--', linewidth=2,
                  label=f'最优SMM={best_smm:.4f}')
        
        ax.set_xlabel('评估次数', fontweight='bold')
        ax.set_ylabel('SMM距离（对数坐标）', fontweight='bold')
        ax.set_title('SMM目标函数收敛历史', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 目标函数历史可视化完成: {static_path}")
        return str(static_path)
    
    def plot_parameter_traces(
        self,
        parameter_history: pd.DataFrame,
        param_names: List[str],
        save_name: str = 'CALIBRATION_parameter_traces'
    ) -> str:
        """绘制参数收敛轨迹"""
        n_params = len(param_names)
        ncols = 4
        nrows = (n_params + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(ncols * 4, nrows * 3))
        fig.suptitle('参数收敛轨迹', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for i, param in enumerate(param_names):
            ax = axes[i]
            
            if param in parameter_history.columns:
                values = parameter_history[param].values
                evaluations = parameter_history['evaluation'].values
                
                ax.plot(evaluations, values, color=COLORS['secondary'],
                       linewidth=1.5, alpha=0.7)
                
                # 最终值
                final_val = values[-1]
                ax.axhline(final_val, color='green', linestyle='--',
                          linewidth=1.5, label=f'最终={final_val:.3f}')
                
                ax.set_title(param, fontweight='bold')
                ax.set_xlabel('评估次数', fontsize=9)
                ax.set_ylabel('参数值', fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余子图
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 参数轨迹可视化完成: {static_path}")
        return str(static_path)
    
    def plot_moment_fit(
        self,
        target_moments: Dict[str, float],
        simulated_moments: Dict[str, float],
        save_name: str = 'CALIBRATION_moment_fit'
    ) -> str:
        """绘制矩拟合对比"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE['wide'])
        
        moment_names = list(target_moments.keys())
        target_values = [target_moments[m] for m in moment_names]
        simulated_values = [simulated_moments[m] for m in moment_names]
        
        x = np.arange(len(moment_names))
        width = 0.35
        
        ax.bar(x - width/2, target_values, width, label='目标矩',
              color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, simulated_values, width, label='模拟矩',
              color=COLORS['accent_orange'], alpha=0.8)
        
        # 误差棒
        errors = [abs(t - s) for t, s in zip(target_values, simulated_values)]
        for i, (t, s, e) in enumerate(zip(target_values, simulated_values, errors)):
            ax.plot([i, i], [min(t, s), max(t, s)], 'k-', linewidth=1.5)
            ax.text(i, max(t, s) + 0.01 * max(target_values),
                   f'误差={e:.2f}', ha='center', fontsize=8)
        
        ax.set_xlabel('统计矩', fontweight='bold')
        ax.set_ylabel('值', fontweight='bold')
        ax.set_title('目标矩 vs 模拟矩拟合质量', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(moment_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        static_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 矩拟合对比可视化完成: {static_path}")
        return str(static_path)
    
    def create_interactive_parameter_space(
        self,
        parameter_history: pd.DataFrame,
        save_name: str = 'CALIBRATION_parameter_space_interactive'
    ) -> str:
        """创建交互式参数空间探索图（平行坐标图）"""
        
        # 标准化参数值
        param_cols = [c for c in parameter_history.columns
                     if c not in ['evaluation', 'smm_distance']]
        
        normalized_data = parameter_history.copy()
        for col in param_cols:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max > col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        
        # 创建平行坐标图
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=parameter_history['smm_distance'].values,
                colorscale='Viridis',
                showscale=True,
                cmin=parameter_history['smm_distance'].min(),
                cmax=parameter_history['smm_distance'].max(),
                colorbar=dict(title='SMM距离')
            ),
            dimensions=[
                dict(label=col,
                     values=normalized_data[col].values)
                for col in param_cols
            ]
        ))
        
        fig.update_layout(
            title='参数空间探索（平行坐标图）',
            height=600
        )
        
        interactive_path = self.interactive_dir / f'{save_name}.html'
        fig.write_html(str(interactive_path))
        
        print(f"✓ 交互式参数空间可视化完成: {interactive_path}")
        return str(interactive_path)

