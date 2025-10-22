"""
图表显示组件

matplotlib嵌入PyQt6的图表组件
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import matplotlib
import platform


class ChartWidget(QWidget):
    """图表显示组件类"""
    
    def __init__(self):
        """初始化图表组件"""
        super().__init__()
        
        # 配置中文字体
        self._setup_chinese_font()
        
        # 创建matplotlib图表
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        
        # 创建工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.setLayout(layout)
        
        # 初始化空白图表
        self.clear()
    
    def _setup_chinese_font(self):
        """配置matplotlib中文字体"""
        system = platform.system()
        
        if system == "Windows":
            fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
        elif system == "Darwin":
            fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
        else:
            fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
        
        for font in fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
    
    def plot_unemployment_trend(self, iterations, unemployment_rates):
        """
        绘制失业率趋势图
        
        参数:
            iterations: 迭代轮数列表
            unemployment_rates: 失业率列表（百分比）
        """
        self.ax.clear()
        
        self.ax.plot(iterations, unemployment_rates, 
                    linewidth=2, color='#1ABC9C', marker='o', 
                    markersize=4, markevery=max(1, len(iterations)//20))
        
        self.ax.set_xlabel('迭代轮数', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('失业率 (%)', fontsize=11, fontweight='bold')
        self.ax.set_title('失业率随迭代变化趋势', fontsize=12, fontweight='bold')
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#FAFAFA')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_state_distribution(self, values, state_var='T'):
        """
        绘制状态变量分布直方图
        
        参数:
            values: 状态变量值列表
            state_var: 状态变量名
        """
        self.ax.clear()
        
        self.ax.hist(values, bins=30, color='#3498DB', 
                    alpha=0.7, edgecolor='black')
        
        var_names = {
            'T': '每周工作时长 (小时)',
            'S': '工作能力评分',
            'D': '数字素养评分',
            'W': '每月期望收入 (元)'
        }
        
        self.ax.set_xlabel(var_names.get(state_var, state_var), 
                          fontsize=11, fontweight='bold')
        self.ax.set_ylabel('个体数量', fontsize=11, fontweight='bold')
        self.ax.set_title(f'{var_names.get(state_var, state_var)}分布', 
                         fontsize=12, fontweight='bold')
        
        self.ax.grid(True, alpha=0.3, axis='y')
        self.ax.set_facecolor('#FAFAFA')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_convergence(self, iterations, diff_V_list, diff_u_list):
        """
        绘制收敛过程（双子图）
        
        参数:
            iterations: 迭代轮数列表
            diff_V_list: 价值函数变化列表
            diff_u_list: 失业率变化列表
        """
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        
        # 价值函数收敛曲线
        ax1.plot(iterations, diff_V_list, 
                linewidth=2, color='#E74C3C', marker='s', 
                markersize=3, markevery=max(1, len(iterations)//20))
        ax1.set_ylabel('|ΔV|/|V|', fontsize=10, fontweight='bold')
        ax1.set_title('价值函数收敛过程', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#FAFAFA')
        ax1.axhline(y=0.01, color='red', linestyle='--', 
                   alpha=0.5, label='收敛阈值')
        ax1.legend(fontsize=9)
        
        # 失业率收敛曲线
        ax2.plot(iterations, diff_u_list, 
                linewidth=2, color='#9B59B6', marker='D', 
                markersize=3, markevery=max(1, len(iterations)//20))
        ax2.set_xlabel('迭代轮数', fontsize=10, fontweight='bold')
        ax2.set_ylabel('|Δu|', fontsize=10, fontweight='bold')
        ax2.set_title('失业率收敛过程', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#FAFAFA')
        ax2.axhline(y=0.001, color='red', linestyle='--', 
                   alpha=0.5, label='收敛阈值')
        ax2.legend(fontsize=9)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_market_tightness(self, iterations, theta_list):
        """
        绘制市场紧张度变化
        
        参数:
            iterations: 迭代轮数列表
            theta_list: 市场紧张度列表
        """
        self.ax.clear()
        
        self.ax.plot(iterations, theta_list, 
                    linewidth=2, color='#F39C12', marker='^', 
                    markersize=4, markevery=max(1, len(iterations)//20))
        
        self.ax.set_xlabel('迭代轮数', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('市场紧张度 θ', fontsize=11, fontweight='bold')
        self.ax.set_title('市场紧张度随迭代变化', fontsize=12, fontweight='bold')
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#FAFAFA')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def clear(self):
        """清空图表"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, '暂无数据', 
                    ha='center', va='center',
                    fontsize=14, color='#7F8C8D')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()

