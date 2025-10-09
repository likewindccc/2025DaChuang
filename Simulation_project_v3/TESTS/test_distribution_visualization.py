#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
企业与劳动力四维分布可视化

可视化生成的企业和劳动力在T、S、D、W四个变量上的数值分布，
帮助理解为什么某些项的贡献较低。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config


def visualize_distributions():
    """可视化企业和劳动力的四维分布"""
    print("=" * 80)
    print("企业与劳动力四维分布可视化")
    print("=" * 80)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 生成虚拟市场
    print("\n生成虚拟市场...")
    market_gen = VirtualMarket(config)
    laborers, enterprises = market_gen.generate_market(n_laborers=2000, theta=1.0)
    
    print(f"  劳动力: {len(laborers)}")
    print(f"  企业: {len(enterprises)}")
    
    # 提取数据
    T_i = laborers['T'].values
    S_i = laborers['S'].values
    D_i = laborers['D'].values
    W_i = laborers['W'].values
    
    T_req = enterprises['T_req'].values
    S_req = enterprises['S_req'].values
    D_req = enterprises['D_req'].values
    W_offer = enterprises['W_offer'].values
    
    # 创建图形
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('企业与劳动力四维分布对比', fontsize=16, y=0.995)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    variables = [
        ('T', T_i, T_req, '工作时间 T'),
        ('S', S_i, S_req, '专业能力 S'),
        ('D', D_i, D_req, '数字素养 D'),
        ('W', W_i, W_offer, '薪资 W')
    ]
    
    for idx, (var_name, laborer_data, enterprise_data, title) in enumerate(variables):
        # 左侧：直方图对比
        ax_left = axes[idx, 0]
        ax_left.hist(laborer_data, bins=50, alpha=0.6, label=f'劳动力 {var_name}', color='blue', density=True)
        ax_left.hist(enterprise_data, bins=50, alpha=0.6, label=f'企业 {var_name}_req/offer', color='red', density=True)
        ax_left.set_xlabel(title, fontsize=12)
        ax_left.set_ylabel('密度', fontsize=12)
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'劳动力: μ={laborer_data.mean():.2f}, σ={laborer_data.std():.2f}\n'
        stats_text += f'企业: μ={enterprise_data.mean():.2f}, σ={enterprise_data.std():.2f}'
        ax_left.text(0.02, 0.98, stats_text, transform=ax_left.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 右侧：散点图 + 差值分析（仅对S和D）
        ax_right = axes[idx, 1]
        if var_name in ['S', 'D']:
            # 计算差值分布
            # 为了可视化，我们随机抽样1000对劳动力-企业组合
            np.random.seed(42)
            sample_size = 1000
            laborer_samples = np.random.choice(laborer_data, sample_size)
            enterprise_samples = np.random.choice(enterprise_data, sample_size)
            
            diff = enterprise_samples - laborer_samples
            max_diff = np.maximum(0, diff)
            
            ax_right.hist(max_diff, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax_right.set_xlabel(f'max(0, {var_name}_req - {var_name}_i)', fontsize=12)
            ax_right.set_ylabel('频数', fontsize=12)
            ax_right.set_title(f'{title} - 能力差值分布', fontsize=12)
            ax_right.grid(True, alpha=0.3)
            
            # 统计信息
            stats_text = f'max(0, diff)统计:\n'
            stats_text += f'均值: {max_diff.mean():.4f}\n'
            stats_text += f'标准差: {max_diff.std():.4f}\n'
            stats_text += f'非零占比: {(max_diff > 0).mean()*100:.1f}%\n'
            stats_text += f'中位数: {np.median(max_diff):.4f}'
            ax_right.text(0.98, 0.98, stats_text, transform=ax_right.transAxes,
                        verticalalignment='top', horizontalalignment='right', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            # 对于T和W，显示散点图
            sample_size = min(500, len(laborer_data))
            indices = np.random.choice(len(laborer_data), sample_size, replace=False)
            ax_right.scatter(laborer_data[indices], enterprise_data[indices], 
                           alpha=0.3, s=10, color='purple')
            ax_right.set_xlabel(f'劳动力 {var_name}', fontsize=12)
            ax_right.set_ylabel(f'企业 {var_name}_req/offer', fontsize=12)
            ax_right.set_title(f'{title} - 散点图', fontsize=12)
            ax_right.grid(True, alpha=0.3)
            
            # 添加y=x参考线
            min_val = min(laborer_data.min(), enterprise_data.min())
            max_val = max(laborer_data.max(), enterprise_data.max())
            ax_right.plot([min_val, max_val], [min_val, max_val], 
                         'r--', alpha=0.5, label='y=x')
            ax_right.legend()
    
    plt.tight_layout()
    
    # 保存图形
    output_dir = Path("OUTPUT/logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "distribution_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图形已保存至: {output_file}")
    
    # 详细统计分析
    print("\n" + "=" * 80)
    print("详细统计分析")
    print("=" * 80)
    
    for var_name, laborer_data, enterprise_data, title in variables:
        print(f"\n{title} ({var_name}):")
        print(f"  劳动力: 范围=[{laborer_data.min():.2f}, {laborer_data.max():.2f}], "
              f"均值={laborer_data.mean():.2f}, 标准差={laborer_data.std():.2f}")
        print(f"  企业:   范围=[{enterprise_data.min():.2f}, {enterprise_data.max():.2f}], "
              f"均值={enterprise_data.mean():.2f}, 标准差={enterprise_data.std():.2f}")
        
        if var_name in ['S', 'D']:
            # 计算max(0, req - i)的统计
            sample_size = 10000
            laborer_samples = np.random.choice(laborer_data, sample_size)
            enterprise_samples = np.random.choice(enterprise_data, sample_size)
            max_diff = np.maximum(0, enterprise_samples - laborer_samples)
            
            print(f"  max(0, {var_name}_req - {var_name}_i):")
            print(f"    均值: {max_diff.mean():.4f}")
            print(f"    标准差: {max_diff.std():.4f}")
            print(f"    非零占比: {(max_diff > 0).mean()*100:.1f}%")
            print(f"    中位数: {np.median(max_diff):.4f}")
            print(f"    75分位数: {np.percentile(max_diff, 75):.4f}")
            print(f"    95分位数: {np.percentile(max_diff, 95):.4f}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    visualize_distributions()

