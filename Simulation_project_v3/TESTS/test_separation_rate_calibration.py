#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离职率系数校准测试

目标：找到合适的eta系数，使得平均离职率约为5%
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

# 生成虚拟数据（使用POPULATION模块的分布）
from MODULES.POPULATION import LaborDistribution


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_separation_rate(
    T, S, D, W, age, education, children,
    eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
):
    """
    计算离职率
    μ(x, σ_i) = 1 / (1 + exp(-η'Z))
    """
    z = (eta0 +
         eta_T * T +
         eta_S * S +
         eta_D * D +
         eta_W * W +
         eta_age * age +
         eta_edu * education +
         eta_children * children)
    
    mu = 1.0 / (1.0 + np.exp(-z))
    return mu


def test_separation_rate(
    eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children,
    return_full_array=False
):
    """
    测试给定系数下的平均离职率
    
    参数:
        return_full_array: 如果为True，返回完整的离职率数组
    """
    # 生成1000个虚拟劳动力
    np.random.seed(42)
    
    # 加载劳动力分布
    config = load_config("CONFIG/population_config.yaml")
    labor_model = LaborDistribution(config)
    labor_model.fit()
    
    # 采样1000个个体（直接使用Copula模型）
    n_samples = 1000
    
    # 采样连续变量
    continuous_samples = labor_model.copula_model.sample(n_samples)
    
    # 采样离散变量
    edu_values = list(labor_model.discrete_dist['edu'].keys())
    edu_probs = list(labor_model.discrete_dist['edu'].values())
    edu_samples = np.random.choice(edu_values, size=n_samples, p=edu_probs)
    
    children_values = list(labor_model.discrete_dist['children'].keys())
    children_probs = list(labor_model.discrete_dist['children'].values())
    children_samples = np.random.choice(
        children_values, size=n_samples, p=children_probs
    )
    
    # 组合为DataFrame
    laborers = continuous_samples.copy()
    laborers['education'] = edu_samples
    laborers['children'] = children_samples
    
    # 计算每个人的离职率
    mu_list = []
    for idx, row in laborers.iterrows():
        mu = compute_separation_rate(
            row['T'], row['S'], row['D'], row['W'],
            row['age'], row['education'], row['children'],
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        mu_list.append(mu)
    
    mu_array = np.array(mu_list)
    
    if return_full_array:
        return mu_array
    
    mean_mu = mu_array.mean()
    std_mu = mu_array.std()
    min_mu = mu_array.min()
    max_mu = mu_array.max()
    
    return mean_mu, std_mu, min_mu, max_mu


if __name__ == "__main__":
    print("=== 离职率系数校准 ===\n")
    print("目标：平均离职率 ≈ 5%\n")
    
    # 固定的系数（从mfg_config.yaml）
    eta_T = -0.02
    eta_S = -2.0
    eta_D = -1.0
    eta_W = 0.0001
    eta_age = -0.05
    eta_edu = -0.1
    eta_children = 0.1
    
    # 第一步：粗调，找到5%的大致范围
    print("第一步：粗调搜索")
    print("-" * 80)
    print(f"{'eta0':>8} | {'平均离职率':>10} | {'标准差':>8} | {'最小值':>8} | {'最大值':>8}")
    print("-" * 80)
    
    coarse_results = []
    for eta0 in np.linspace(10.0, 20.0, 11):
        mean_mu, std_mu, min_mu, max_mu = test_separation_rate(
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        coarse_results.append((eta0, mean_mu))
        print(f"{eta0:8.2f} | {mean_mu*100:9.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {max_mu*100:7.2f}%")
    
    print("-" * 80)
    
    # 找到最接近5%的两个点，确定精细搜索范围
    target = 0.05
    coarse_diffs = [(eta0, abs(mean_mu - target)) 
                    for eta0, mean_mu in coarse_results]
    coarse_diffs.sort(key=lambda x: x[1])
    
    best_coarse_eta0 = coarse_diffs[0][0]
    fine_search_center = best_coarse_eta0
    fine_search_width = 1.0  # 在最佳值±1.0范围内精细搜索
    fine_search_min = fine_search_center - fine_search_width
    fine_search_max = fine_search_center + fine_search_width
    
    print(f"\n粗调最佳: eta0 ≈ {best_coarse_eta0:.2f}")
    print(f"精细搜索范围: [{fine_search_min:.2f}, {fine_search_max:.2f}]\n")
    
    # 第二步：精细调整
    print("第二步：精细搜索")
    print("-" * 80)
    print(f"{'eta0':>8} | {'平均离职率':>10} | {'标准差':>8} | {'最小值':>8} | {'最大值':>8}")
    print("-" * 80)
    
    best_eta0 = None
    best_mean_mu = None
    best_diff = float('inf')
    
    for eta0 in np.linspace(fine_search_min, fine_search_max, 21):
        mean_mu, std_mu, min_mu, max_mu = test_separation_rate(
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        
        print(f"{eta0:8.2f} | {mean_mu*100:9.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {max_mu*100:7.2f}%")
        
        diff = abs(mean_mu - target)
        if diff < best_diff:
            best_diff = diff
            best_eta0 = eta0
            best_mean_mu = mean_mu
    
    print("-" * 80)
    print(f"\n>>> 最终推荐: eta0 = {best_eta0:.2f}")
    print(f"    平均离职率 = {best_mean_mu*100:.2f}%")
    print(f"    与目标差距 = {abs(best_mean_mu - target)*100:.2f}%")
    
    # 可视化：离职率分布直方图
    print("\n生成离职率分布可视化...")
    mu_array = test_separation_rate(
        best_eta0, eta_T, eta_S, eta_D, eta_W,
        eta_age, eta_edu, eta_children,
        return_full_array=True
    )
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    ax1 = axes[0]
    ax1.hist(mu_array * 100, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(best_mean_mu * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均值={best_mean_mu*100:.2f}%')
    ax1.axvline(target * 100, color='green', linestyle='--', linewidth=2,
                label=f'目标={target*100:.0f}%')
    ax1.set_xlabel('离职率 (%)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title(f'离职率分布直方图 (eta0={best_eta0:.2f})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：累积分布函数
    ax2 = axes[1]
    sorted_mu = np.sort(mu_array * 100)
    cumulative = np.arange(1, len(sorted_mu) + 1) / len(sorted_mu) * 100
    ax2.plot(sorted_mu, cumulative, color='navy', linewidth=2)
    ax2.axvline(best_mean_mu * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均值={best_mean_mu*100:.2f}%')
    ax2.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('离职率 (%)', fontsize=12)
    ax2.set_ylabel('累积百分比 (%)', fontsize=12)
    ax2.set_title('离职率累积分布函数 (CDF)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = "OUTPUT/mfg/separation_rate_distribution.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存至: {output_path}")
    
    # 输出详细统计信息
    print(f"\n离职率分布统计:")
    print(f"  平均值: {mu_array.mean()*100:.2f}%")
    print(f"  中位数: {np.median(mu_array)*100:.2f}%")
    print(f"  标准差: {mu_array.std()*100:.2f}%")
    print(f"  最小值: {mu_array.min()*100:.2f}%")
    print(f"  25分位: {np.percentile(mu_array, 25)*100:.2f}%")
    print(f"  75分位: {np.percentile(mu_array, 75)*100:.2f}%")
    print(f"  最大值: {mu_array.max()*100:.2f}%")

