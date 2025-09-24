import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体（如使用Mac或其他系统，请替换字体）
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 样本量范围
sample_sizes = np.arange(50, 501, 10)

# 固定置信水平为90%，Z值
Z_90 = 1.645

# 计算不同样本量下的误差（允许误差）
errors = np.sqrt((Z_90**2 * 0.25) / sample_sizes)

# 将误差转换为置信区间宽度（误差的两倍）
confidence_intervals = 2 * errors

import numpy as np
import matplotlib.pyplot as plt

# --- 基本设置 ---
# 使用您系统支持的中文matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei'] # 或 'Microsoft YaHei', 'Arial Unicode MS' 等
plt.rcParams['axes.unicode_minus'] = False
# 参考您示例代码中可能使用的字体大小（如果未指定则使用默认或之前设置）
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# --- 参数设定 ---
Z = 1.96      # 95% 置信水平对应的 Z 值
sigma = 11.4  # 基于 "工作技能总评" 的标准差估计值
variable_name_cn = "工作技能总评" # 用于标题说明

# --- 数据计算 ---
sample_sizes = np.arange(50, 501, 10) # 样本量范围
# 计算允许误差 E = Z * sigma / sqrt(n)
errors = Z * sigma / np.sqrt(sample_sizes)
# 计算置信区间宽度 Width = 2 * E
confidence_intervals = 2 * errors

# --- 创建图像 (采用您指定的格式) ---
# 创建图像，指定尺寸
fig, axs = plt.subplots(1, 2, figsize=(14, 5)) # 使用您倾向的 figsize

# 子图1：样本量 vs 置信区间宽度
axs[0].plot(sample_sizes, confidence_intervals, color='darkorange', marker='o', markersize=5, linestyle='-')
axs[0].set_title(f'样本量 vs 置信区间宽度 (95% 置信水平)') # 更新标题
axs[0].set_xlabel('样本量 n')
axs[0].set_ylabel('置信区间宽度 (2E)') # 修改标签以匹配计算
axs[0].grid(True, linestyle='--', alpha=0.7) # 添加网格线样式

# 子图2：样本量 vs 允许误差
axs[1].plot(sample_sizes, errors, color='steelblue', marker='s', markersize=5, linestyle='-')
axs[1].set_title(f'样本量 vs 允许误差 (95% 置信水平)') # 更新标题
axs[1].set_xlabel('样本量 n')
axs[1].set_ylabel('允许误差 E') # 修改标签以匹配计算
axs[1].grid(True, linestyle='--', alpha=0.7) # 添加网格线样式

# 自动调整布局
plt.tight_layout()

# 保存为高分辨率PNG图像（可选，取消注释以保存）
plt.savefig("样本量_误差_置信区间分析图_95CI_WorkAbility_Clean.png", dpi=300)

# 显示图像
plt.show()