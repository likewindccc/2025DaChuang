"""
可视化样式统一配置

提供颜色方案、图表尺寸、字体配置等，确保所有可视化保持一致的风格
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# 颜色方案（与网站设计保持一致）
COLORS = {
    # 主色调
    'primary': '#8b5cf6',      # 紫色
    'secondary': '#3b82f6',    # 蓝色
    'accent_pink': '#ec4899',  # 粉色
    'accent_green': '#10b981', # 绿色
    'accent_orange': '#f59e0b',# 橙色
    
    # 文本颜色
    'text_primary': '#1e293b',
    'text_secondary': '#64748b',
    
    # 政策场景颜色
    'baseline': '#7f7f7f',     # 灰色（基准）
    'policy_A': '#9467bd',     # 技能培训
    'policy_B': '#8c564b',     # 数字素养
    'policy_C': '#e377c2',     # 就业补贴
    'policy_D': '#bcbd22',     # 岗位创造
    'policy_E': '#17becf',     # 匹配效率
    
    # 状态颜色
    'employed': '#2ca02c',     # 绿色（就业）
    'unemployed': '#d62728',   # 红色（失业）
    
    # 渐变色
    'gradient_purple_blue': ['#8b5cf6', '#6366f1', '#3b82f6'],
    'gradient_warm': ['#f59e0b', '#ec4899', '#8b5cf6'],
}

# 图表尺寸配置
FIGURE_SIZE = {
    'single': (8, 6),          # 单图
    'double': (12, 5),         # 双图并排
    'triple': (15, 5),         # 三图并排
    'grid_2x2': (10, 10),      # 2×2网格
    'grid_3x2': (12, 8),       # 3×2网格
    'grid_4x2': (14, 10),      # 4×2网格
    'wide': (14, 5),           # 宽图
    'tall': (8, 12),           # 高图
    'square': (8, 8),          # 方形
}

# 字体配置
FONT_CONFIG = {
    'family': 'SimHei',        # 中文字体（黑体）
    'title_size': 16,          # 标题字号
    'label_size': 12,          # 轴标签字号
    'tick_size': 10,           # 刻度字号
    'legend_size': 10,         # 图例字号
    'annotation_size': 9,      # 注释字号
}

# DPI设置
DPI_CONFIG = {
    'screen': 150,             # 屏幕显示
    'paper': 300,              # 论文打印
    'presentation': 200,       # 演示文稿
}

# Plotly模板配置
PLOTLY_TEMPLATE = {
    'layout': {
        'font': {
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': COLORS['text_primary']
        },
        'title': {
            'font': {'size': 16, 'color': COLORS['text_primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True,
            'gridcolor': 'rgba(139, 92, 246, 0.1)',
            'linecolor': COLORS['text_secondary'],
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': 'rgba(139, 92, 246, 0.1)',
            'linecolor': COLORS['text_secondary'],
        },
        'plot_bgcolor': 'rgba(248, 249, 255, 0.5)',
        'paper_bgcolor': 'white',
        'hovermode': 'closest',
    }
}


def setup_matplotlib_style():
    """配置Matplotlib全局样式"""
    plt.rcParams['font.sans-serif'] = [FONT_CONFIG['family']]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = DPI_CONFIG['screen']
    plt.rcParams['savefig.dpi'] = DPI_CONFIG['paper']
    plt.rcParams['font.size'] = FONT_CONFIG['label_size']
    plt.rcParams['axes.titlesize'] = FONT_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = FONT_CONFIG['label_size']
    plt.rcParams['xtick.labelsize'] = FONT_CONFIG['tick_size']
    plt.rcParams['ytick.labelsize'] = FONT_CONFIG['tick_size']
    plt.rcParams['legend.fontsize'] = FONT_CONFIG['legend_size']
    
    # 设置默认颜色循环
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=[
            COLORS['primary'],
            COLORS['secondary'],
            COLORS['accent_pink'],
            COLORS['accent_green'],
            COLORS['accent_orange']
        ]
    )
    
    # 设置网格样式
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    print("✓ Matplotlib样式配置完成")


def setup_plotly_template():
    """配置Plotly全局模板"""
    pio.templates['custom'] = go.layout.Template(
        layout=PLOTLY_TEMPLATE['layout']
    )
    pio.templates.default = 'custom'
    print("✓ Plotly模板配置完成")


def get_color_palette(n_colors: int, palette_type: str = 'gradient') -> list:
    """
    获取颜色调色板
    
    参数:
        n_colors: 需要的颜色数量
        palette_type: 调色板类型 ('gradient', 'distinct', 'policy')
    
    返回:
        颜色列表
    """
    if palette_type == 'gradient':
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # 创建紫-蓝渐变
        cmap = LinearSegmentedColormap.from_list(
            'custom_gradient',
            COLORS['gradient_purple_blue']
        )
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    
    elif palette_type == 'distinct':
        base_colors = [
            COLORS['primary'],
            COLORS['secondary'],
            COLORS['accent_pink'],
            COLORS['accent_green'],
            COLORS['accent_orange'],
            COLORS['policy_A'],
            COLORS['policy_B'],
            COLORS['policy_C']
        ]
        return base_colors[:n_colors]
    
    elif palette_type == 'policy':
        return [
            COLORS['baseline'],
            COLORS['policy_A'],
            COLORS['policy_B'],
            COLORS['policy_C'],
            COLORS['policy_D'],
            COLORS['policy_E']
        ][:n_colors]
    
    else:
        raise ValueError(f"未知的调色板类型: {palette_type}")


# 初始化配置
def initialize_styles():
    """初始化所有可视化样式"""
    setup_matplotlib_style()
    setup_plotly_template()
    print("✓ 所有可视化样式初始化完成")


if __name__ == "__main__":
    initialize_styles()

