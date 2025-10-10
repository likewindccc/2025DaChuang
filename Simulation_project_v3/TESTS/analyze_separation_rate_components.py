#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离职率各项贡献分析

分析为什么会出现两极分化现象
"""

import numpy as np
import pandas as pd
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.POPULATION import LaborDistribution


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def analyze_separation_rate_components():
    """
    分析离职率函数各项的贡献
    """
    print("=== 离职率各项贡献分析 ===\n")
    
    # 当前参数
    eta0 = 20.70
    eta_T = -0.02
    eta_S = -2.0
    eta_D = -1.0
    eta_W = 0.0001
    eta_age = -0.05
    eta_edu = -0.1
    eta_children = 0.1
    
    print("当前系数设置:")
    print(f"  eta0 (截距) = {eta0:.2f}")
    print(f"  eta_T (工作时间) = {eta_T:.4f}")
    print(f"  eta_S (技能) = {eta_S:.2f}")
    print(f"  eta_D (数字素养) = {eta_D:.2f}")
    print(f"  eta_W (期望工资) = {eta_W:.6f}")
    print(f"  eta_age (年龄) = {eta_age:.2f}")
    print(f"  eta_edu (教育年限) = {eta_edu:.2f}")
    print(f"  eta_children (孩子数量) = {eta_children:.2f}")
    print()
    
    # 生成1000个虚拟劳动力
    np.random.seed(42)
    config = load_config("CONFIG/population_config.yaml")
    labor_model = LaborDistribution(config)
    labor_model.fit()
    
    n_samples = 1000
    continuous_samples = labor_model.copula_model.sample(n_samples)
    
    edu_values = list(labor_model.discrete_dist['edu'].keys())
    edu_probs = list(labor_model.discrete_dist['edu'].values())
    edu_samples = np.random.choice(edu_values, size=n_samples, p=edu_probs)
    
    children_values = list(labor_model.discrete_dist['children'].keys())
    children_probs = list(labor_model.discrete_dist['children'].values())
    children_samples = np.random.choice(
        children_values, size=n_samples, p=children_probs
    )
    
    laborers = continuous_samples.copy()
    laborers['education'] = edu_samples
    laborers['children'] = children_samples
    
    # 计算各变量的分布范围
    print("=" * 80)
    print("各变量的分布统计")
    print("=" * 80)
    print(f"{'变量':>12} | {'均值':>8} | {'标准差':>8} | {'最小值':>8} | {'最大值':>8} | {'范围':>8}")
    print("-" * 80)
    
    for col in ['T', 'S', 'D', 'W', 'age', 'education', 'children']:
        data = laborers[col].values
        print(f"{col:>12} | {data.mean():8.2f} | {data.std():8.2f} | "
              f"{data.min():8.2f} | {data.max():8.2f} | {data.max()-data.min():8.2f}")
    
    print()
    
    # 计算每项对z的贡献
    print("=" * 80)
    print("各项对线性组合z的贡献统计")
    print("=" * 80)
    print(f"{'项':>20} | {'均值':>10} | {'标准差':>10} | {'最小值':>10} | {'最大值':>10}")
    print("-" * 80)
    
    contributions = {}
    contributions['截距 (eta0)'] = np.full(n_samples, eta0)
    contributions['T项 (eta_T*T)'] = eta_T * laborers['T'].values
    contributions['S项 (eta_S*S)'] = eta_S * laborers['S'].values
    contributions['D项 (eta_D*D)'] = eta_D * laborers['D'].values
    contributions['W项 (eta_W*W)'] = eta_W * laborers['W'].values
    contributions['age项 (eta_age*age)'] = eta_age * laborers['age'].values
    contributions['edu项 (eta_edu*edu)'] = eta_edu * laborers['education'].values
    contributions['children项 (eta_children*children)'] = eta_children * laborers['children'].values
    
    total_z = np.zeros(n_samples)
    for name, contrib in contributions.items():
        total_z += contrib
        print(f"{name:>20} | {contrib.mean():10.2f} | {contrib.std():10.2f} | "
              f"{contrib.min():10.2f} | {contrib.max():10.2f}")
    
    print("-" * 80)
    print(f"{'总计 z':>20} | {total_z.mean():10.2f} | {total_z.std():10.2f} | "
          f"{total_z.min():10.2f} | {total_z.max():10.2f}")
    print()
    
    # 计算最终离职率
    mu = 1.0 / (1.0 + np.exp(-total_z))
    
    print("=" * 80)
    print("离职率μ的分布统计")
    print("=" * 80)
    print(f"  均值: {mu.mean()*100:.2f}%")
    print(f"  标准差: {mu.std()*100:.2f}%")
    print(f"  最小值: {mu.min()*100:.2f}%")
    print(f"  25分位: {np.percentile(mu, 25)*100:.2f}%")
    print(f"  50分位 (中位数): {np.percentile(mu, 50)*100:.2f}%")
    print(f"  75分位: {np.percentile(mu, 75)*100:.2f}%")
    print(f"  最大值: {mu.max()*100:.2f}%")
    print()
    
    # 分析问题根源
    print("=" * 80)
    print("问题诊断")
    print("=" * 80)
    
    # 检查z的分布
    z_mean = total_z.mean()
    z_std = total_z.std()
    z_min = total_z.min()
    z_max = total_z.max()
    
    print(f"1. 线性组合z的分布:")
    print(f"   z ∈ [{z_min:.2f}, {z_max:.2f}]")
    print(f"   z的范围过大！当z<-5时，μ≈0%; 当z>5时，μ≈100%")
    print()
    
    # 识别主导项
    print(f"2. 各项贡献的绝对值大小 (平均):")
    abs_contributions = {name: abs(contrib.mean()) 
                        for name, contrib in contributions.items()}
    sorted_contrib = sorted(abs_contributions.items(), 
                           key=lambda x: x[1], reverse=True)
    for name, val in sorted_contrib:
        pct = val / sum(abs_contributions.values()) * 100
        print(f"   {name:>20}: {val:8.2f}  ({pct:5.1f}%)")
    print()
    
    print(f"3. 核心问题:")
    print(f"   - 截距eta0={eta0:.2f}是主导项，贡献了{abs_contributions['截距 (eta0)']/sum(abs_contributions.values())*100:.1f}%")
    print(f"   - S项和D项的系数过大（-2.0, -1.0），导致高技能者z变成很负，μ≈0")
    print(f"   - T项和W项的系数过小（-0.02, 0.0001），几乎没有影响")
    print()
    
    print("=" * 80)
    print("建议的解决方案")
    print("=" * 80)
    print("方案1: 调整系数，使各项贡献更平衡")
    print("  - 降低eta0（如从20.70降到2-3）")
    print("  - 调整eta_S和eta_D到较小值（如-0.2到-0.5）")
    print("  - 增大eta_T和eta_W的绝对值（如eta_T=-0.1）")
    print()
    print("方案2: 对变量进行标准化")
    print("  - 将所有变量标准化为均值0、标准差1")
    print("  - 然后重新校准所有eta系数")
    print()
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各项贡献的箱线图
    ax1 = axes[0, 0]
    contrib_data = [contrib for name, contrib in contributions.items()]
    contrib_labels = [name.split('(')[0].strip() for name in contributions.keys()]
    bp = ax1.boxplot(contrib_data, labels=contrib_labels, patch_artist=True)
    ax1.set_ylabel('贡献值', fontsize=11)
    ax1.set_title('各项对z的贡献分布', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. z的分布直方图
    ax2 = axes[0, 1]
    ax2.hist(total_z, bins=50, color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='z=0')
    ax2.axvline(-5, color='green', linestyle=':', linewidth=1.5, label='z=-5 (μ≈0.7%)')
    ax2.axvline(5, color='green', linestyle=':', linewidth=1.5, label='z=5 (μ≈99.3%)')
    ax2.set_xlabel('线性组合 z', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.set_title('线性组合z的分布', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 离职率μ的分布
    ax3 = axes[1, 0]
    ax3.hist(mu * 100, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(mu.mean() * 100, color='red', linestyle='--', linewidth=2,
                label=f'均值={mu.mean()*100:.2f}%')
    ax3.axvline(5, color='green', linestyle='--', linewidth=2, label='目标=5%')
    ax3.set_xlabel('离职率 μ (%)', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.set_title('离职率μ的分布（两极分化）', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 各项贡献的绝对值占比饼图
    ax4 = axes[1, 1]
    sorted_names = [name.split('(')[0].strip() for name, _ in sorted_contrib[:6]]
    sorted_vals = [val for _, val in sorted_contrib[:6]]
    colors = plt.cm.Set3(range(len(sorted_names)))
    ax4.pie(sorted_vals, labels=sorted_names, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('各项贡献的绝对值占比', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = "OUTPUT/mfg/separation_rate_component_analysis.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存至: {output_path}")
    
    return laborers, contributions, total_z, mu


if __name__ == "__main__":
    analyze_separation_rate_components()

