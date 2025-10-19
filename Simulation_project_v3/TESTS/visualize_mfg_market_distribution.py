#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG均衡后劳动力市场分布可视化

可视化MFG均衡结果中劳动力和企业在T、S、D、W四个变量上的分布对比
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yaml

matplotlib.use('Agg')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_equilibrium_data():
    """加载MFG均衡结果"""
    individuals_path = project_root / "OUTPUT" / "mfg" / "equilibrium_individuals.csv"
    
    if not individuals_path.exists():
        raise FileNotFoundError(f"未找到均衡结果文件: {individuals_path}")
    
    df = pd.read_csv(individuals_path)
    return df


def generate_enterprise_distribution(n_enterprises):
    """根据配置生成企业分布"""
    config_path = project_root / "CONFIG" / "population_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    ent_dist = config['enterprise_distribution']
    
    mean = ent_dist['mean']
    std = ent_dist['std']
    corr = ent_dist['correlation']
    
    np.random.seed(42)
    
    mean_vec = np.array([
        mean['T_req'],
        mean['S_req'],
        mean['D_req'],
        mean['W_offer']
    ])
    
    cov_matrix = np.array([
        [std['T_req']**2, 
         corr['T_S']*std['T_req']*std['S_req'],
         corr['T_D']*std['T_req']*std['D_req'],
         corr['T_W']*std['T_req']*std['W_offer']],
        [corr['T_S']*std['T_req']*std['S_req'],
         std['S_req']**2,
         corr['S_D']*std['S_req']*std['D_req'],
         corr['S_W']*std['S_req']*std['W_offer']],
        [corr['T_D']*std['T_req']*std['D_req'],
         corr['S_D']*std['S_req']*std['D_req'],
         std['D_req']**2,
         corr['D_W']*std['D_req']*std['W_offer']],
        [corr['T_W']*std['T_req']*std['W_offer'],
         corr['S_W']*std['S_req']*std['W_offer'],
         corr['D_W']*std['D_req']*std['W_offer'],
         std['W_offer']**2]
    ])
    
    samples = np.random.multivariate_normal(mean_vec, cov_matrix, n_enterprises)
    
    enterprises = pd.DataFrame({
        'T_req': np.clip(samples[:, 0], 20, 80),
        'S_req': np.clip(samples[:, 1], 0, 100),
        'D_req': np.clip(samples[:, 2], 0, 100),
        'W_offer': np.clip(samples[:, 3], 1000, 15000)
    })
    
    return enterprises


def visualize_distributions():
    """可视化劳动力和企业的四维分布"""
    print("="*80)
    print("MFG均衡后劳动力市场分布可视化")
    print("="*80)
    
    laborers = load_equilibrium_data()
    print(f"\n加载劳动力数据: {len(laborers)} 人")
    
    n_enterprises = int(len(laborers) / 1.5)
    enterprises = generate_enterprise_distribution(n_enterprises)
    print(f"生成企业数据: {len(enterprises)} 家 (市场紧张度θ≈1.5)")
    
    T_i = laborers['T'].values
    S_i = laborers['S'].values
    D_i = laborers['D'].values
    W_i = laborers['W'].values
    
    T_req = enterprises['T_req'].values
    S_req = enterprises['S_req'].values
    D_req = enterprises['D_req'].values
    W_offer = enterprises['W_offer'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MFG均衡后劳动力市场分布对比 (劳动力 vs 企业需求)', 
                 fontsize=16, fontweight='bold')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    variables = [
        ('T', T_i, T_req, '工作时间 (小时/周)', 'blue', 'red'),
        ('S', S_i, S_req, '专业能力评分', 'green', 'orange'),
        ('D', D_i, D_req, '数字素养评分', 'purple', 'brown'),
        ('W', W_i, W_offer, '薪资期望/提供 (元/月)', 'cyan', 'magenta')
    ]
    
    axes_flat = axes.flatten()
    
    for idx, (var_name, laborer_data, enterprise_data, title, color_labor, color_ent) in enumerate(variables):
        ax = axes_flat[idx]
        
        ax.hist(laborer_data, bins=50, alpha=0.5, label=f'劳动力 {var_name}', 
                color=color_labor, density=True, edgecolor='black', linewidth=0.5)
        ax.hist(enterprise_data, bins=50, alpha=0.5, label=f'企业 {var_name}_req/offer', 
                color=color_ent, density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('密度', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axvline(laborer_data.mean(), color=color_labor, linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'劳动力均值')
        ax.axvline(enterprise_data.mean(), color=color_ent, linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'企业均值')
        
        stats_text = f'【劳动力】\n'
        stats_text += f'  均值: {laborer_data.mean():.2f}\n'
        stats_text += f'  标准差: {laborer_data.std():.2f}\n'
        stats_text += f'  范围: [{laborer_data.min():.1f}, {laborer_data.max():.1f}]\n\n'
        stats_text += f'【企业】\n'
        stats_text += f'  均值: {enterprise_data.mean():.2f}\n'
        stats_text += f'  标准差: {enterprise_data.std():.2f}\n'
        stats_text += f'  范围: [{enterprise_data.min():.1f}, {enterprise_data.max():.1f}]'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        if var_name == 'T':
            ax.axhspan(40, 50, alpha=0.1, color='green', 
                      label='合理范围 (40-50小时)')
            ax.text(0.98, 0.5, '合理范围\n40-50h/周', 
                   transform=ax.transAxes,
                   verticalalignment='center', horizontalalignment='right',
                   fontsize=9, color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = Path("OUTPUT/mfg")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "market_distribution_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n图形已保存至: {output_file}")
    
    print("\n" + "="*80)
    print("详细统计分析")
    print("="*80)
    
    for var_name, laborer_data, enterprise_data, title, _, _ in variables:
        print(f"\n【{title} ({var_name})】")
        print(f"  劳动力:")
        print(f"    范围: [{laborer_data.min():.2f}, {laborer_data.max():.2f}]")
        print(f"    均值: {laborer_data.mean():.2f}")
        print(f"    中位数: {np.median(laborer_data):.2f}")
        print(f"    标准差: {laborer_data.std():.2f}")
        print(f"    25%分位: {np.percentile(laborer_data, 25):.2f}")
        print(f"    75%分位: {np.percentile(laborer_data, 75):.2f}")
        
        print(f"  企业:")
        print(f"    范围: [{enterprise_data.min():.2f}, {enterprise_data.max():.2f}]")
        print(f"    均值: {enterprise_data.mean():.2f}")
        print(f"    中位数: {np.median(enterprise_data):.2f}")
        print(f"    标准差: {enterprise_data.std():.2f}")
        print(f"    25%分位: {np.percentile(enterprise_data, 25):.2f}")
        print(f"    75%分位: {np.percentile(enterprise_data, 75):.2f}")
        
        print(f"  差异分析:")
        print(f"    均值差: {enterprise_data.mean() - laborer_data.mean():+.2f}")
        print(f"    标准差比: {enterprise_data.std() / laborer_data.std():.2f}")
        
        if var_name == 'T':
            合理范围内的劳动力 = ((laborer_data >= 40) & (laborer_data <= 50)).sum()
            合理范围内的企业 = ((enterprise_data >= 40) & (enterprise_data <= 50)).sum()
            print(f"  合理范围(40-50)内占比:")
            print(f"    劳动力: {合理范围内的劳动力/len(laborer_data)*100:.1f}%")
            print(f"    企业: {合理范围内的企业/len(enterprise_data)*100:.1f}%")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    
    print("\n【T值问题诊断】")
    T_mean = laborers['T'].mean()
    if T_mean > 60:
        print(f"  ⚠️  T均值={T_mean:.2f}小时/周，严重偏高！")
        print(f"  ⚠️  超出合理范围(40-50)约 {T_mean-45:.1f} 小时/周")
        print(f"  ⚠️  建议继续调整参数 (降低ρ或提高κ)")
    elif T_mean > 50:
        print(f"  ⚠️  T均值={T_mean:.2f}小时/周，偏高")
        print(f"  ⚠️  超出合理上限约 {T_mean-50:.1f} 小时/周")
    else:
        print(f"  ✓  T均值={T_mean:.2f}小时/周，在合理范围内")


if __name__ == "__main__":
    visualize_distributions()

