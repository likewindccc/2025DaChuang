import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色
color_title = '#2E86AB'
color_module = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

def draw_box(ax, x, y, width, height, text, color, fontsize=11):
    """绘制模块框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='#333333',
                         facecolor=color,
                         linewidth=2,
                         alpha=0.9)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color='white')

# 标题
ax.text(6, 9.4, '项目架构', 
        ha='center', fontsize=30, fontweight='bold', color=color_title)
ax.text(6, 8.99, '农村女性就业市场仿真平台', 
        ha='center', fontsize=18, fontweight='bold', color='#666666')

# 模块1: POPULATION
y_pos = 7.5
draw_box(ax, 1, y_pos, 10, 1.2, 'POPULATION\n人口分布模块', color_module[0], 17)
ax.text(6, y_pos-0.5, '基于自收集真实数据构建劳动力与企业分布 | Gaussian Copula建模', 
        ha='center', fontsize=14, style='italic', color='#444444')

# 模块2: LOGISTIC
y_pos = 5.5
draw_box(ax, 1, y_pos, 10, 1.2, 'LOGISTIC\n匹配与匹配函数模块', color_module[1], 17)
ax.text(6, y_pos-0.5, '虚拟市场生成 | GS稳定匹配算法 | Logit回归估计匹配函数λ(x,a,θ)', 
        ha='center', fontsize=14, style='italic', color='#444444')

# 模块3: MFG (核心)
y_pos = 3.5
draw_box(ax, 1, y_pos, 10, 1.2, 'MFG\n平均场博弈模块', color_module[2], 17)
ax.text(6, y_pos-0.5, 'Bellman方程求解 | KFE人口演化 | 均衡求解', 
        ha='center', fontsize=14, style='italic', color='#444444')

# 模块4: SIMULATOR
y_pos = 1.5
draw_box(ax, 1, y_pos, 4.8, 1, 'SIMULATOR\n市场模拟器', color_module[3], 16)
ax.text(3.4, y_pos-0.4, '批量场景 | 政策分析', 
        ha='center', fontsize=13, style='italic', color='#444444')

# 模块5: CALIBRATION
draw_box(ax, 6.2, y_pos, 4.8, 1, 'CALIBRATION\n参数校准模块', color_module[4], 16)
ax.text(8.6, y_pos-0.4, 'SMM校准 | 遗传算法优化', 
        ha='center', fontsize=13, style='italic', color='#444444')

# 添加边框
border = mpatches.Rectangle((0.2, 0.3), 11.6, 9.5, 
                            linewidth=2, edgecolor='#CCCCCC', 
                            facecolor='none', linestyle='--')
ax.add_patch(border)

plt.tight_layout()
plt.savefig('OUTPUT/simple_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("简洁架构图已生成: OUTPUT/simple_architecture.png")
print("分辨率: 3600 x 3000 像素 (300 DPI)")

