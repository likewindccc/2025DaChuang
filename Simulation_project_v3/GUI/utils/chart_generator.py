"""
图表生成器

统一风格的matplotlib图表生成
"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import platform


class ChartGenerator:
    """图表生成器类"""
    
    def __init__(self):
        """初始化图表生成器，配置中文字体和样式"""
        self._setup_chinese_font()
        self._setup_style()
    
    def _setup_chinese_font(self):
        """配置中文字体"""
        system = platform.system()
        
        if system == "Windows":
            # Windows系统尝试使用微软雅黑
            fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
        elif system == "Darwin":
            # macOS系统
            fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
        else:
            # Linux系统
            fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
        
        for font in fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
    
    def _setup_style(self):
        """配置图表样式"""
        matplotlib.rcParams['figure.figsize'] = (10, 6)
        matplotlib.rcParams['axes.grid'] = True
        matplotlib.rcParams['grid.alpha'] = 0.3
        matplotlib.rcParams['axes.facecolor'] = '#FAFAFA'
        matplotlib.rcParams['figure.dpi'] = 300
    
    def plot_unemployment_trend(self, iterations, unemployment_rates):
        """
        绘制失业率趋势图
        
        参数:
            iterations: 迭代轮数列表
            unemployment_rates: 失业率列表
        
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(iterations, unemployment_rates, 
               linewidth=2, color='#1ABC9C', marker='o', 
               markersize=4, markevery=5)
        
        ax.set_xlabel('迭代轮数', fontsize=12, fontweight='bold')
        ax.set_ylabel('失业率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('失业率随迭代变化趋势', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        return fig
    
    def plot_state_distribution(self, individuals, state_var='T'):
        """
        绘制状态变量分布直方图
        
        参数:
            individuals: 个体DataFrame
            state_var: 状态变量名，'T'/'S'/'D'/'W'
        
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        values = individuals[state_var]
        
        ax.hist(values, bins=30, color='#3498DB', 
               alpha=0.7, edgecolor='black')
        
        var_names = {
            'T': '每周工作时长 (小时)',
            'S': '工作能力评分',
            'D': '数字素养评分',
            'W': '每月期望收入 (元)'
        }
        
        ax.set_xlabel(var_names.get(state_var, state_var), 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('个体数量', fontsize=12, fontweight='bold')
        ax.set_title(f'{var_names.get(state_var, state_var)}分布', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, iterations, diff_V_list, diff_u_list):
        """
        绘制收敛过程曲线
        
        参数:
            iterations: 迭代轮数列表
            diff_V_list: 价值函数变化列表
            diff_u_list: 失业率变化列表
        
        返回:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 价值函数收敛曲线
        ax1.plot(iterations, diff_V_list, 
                linewidth=2, color='#E74C3C', marker='s', 
                markersize=3, markevery=5)
        ax1.set_ylabel('|ΔV|/|V|', fontsize=12, fontweight='bold')
        ax1.set_title('价值函数收敛过程', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#FAFAFA')
        ax1.axhline(y=0.01, color='red', linestyle='--', 
                   alpha=0.5, label='收敛阈值')
        ax1.legend()
        
        # 失业率收敛曲线
        ax2.plot(iterations, diff_u_list, 
                linewidth=2, color='#9B59B6', marker='D', 
                markersize=3, markevery=5)
        ax2.set_xlabel('迭代轮数', fontsize=12, fontweight='bold')
        ax2.set_ylabel('|Δu|', fontsize=12, fontweight='bold')
        ax2.set_title('失业率收敛过程', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#FAFAFA')
        ax2.axhline(y=0.001, color='red', linestyle='--', 
                   alpha=0.5, label='收敛阈值')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_market_tightness(self, iterations, theta_list):
        """
        绘制市场紧张度变化
        
        参数:
            iterations: 迭代轮数列表
            theta_list: 市场紧张度列表
        
        返回:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(iterations, theta_list, 
               linewidth=2, color='#F39C12', marker='^', 
               markersize=4, markevery=5)
        
        ax.set_xlabel('迭代轮数', fontsize=12, fontweight='bold')
        ax.set_ylabel('市场紧张度 θ', fontsize=12, fontweight='bold')
        ax.set_title('市场紧张度随迭代变化', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        return fig
    
    def save_chart(self, fig, filepath):
        """
        保存图表为图片
        
        参数:
            fig: matplotlib Figure对象
            filepath: 保存路径
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

