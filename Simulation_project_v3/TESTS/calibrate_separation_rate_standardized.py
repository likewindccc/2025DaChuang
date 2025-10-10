#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于标准化变量的离职率系数校准

核心改进：
1. 对所有变量进行标准化：z-score = (x - mean) / std
2. 标准化后所有变量都在同一尺度，系数的大小直接反映影响力
3. 重新校准所有eta系数
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


def compute_separation_rate_standardized(
    T_std, S_std, D_std, W_std, age_std, edu_std, children_std,
    eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
):
    """
    计算离职率（基于标准化变量）
    μ(x, σ_i) = 1 / (1 + exp(-η'Z))
    """
    z = (eta0 +
         eta_T * T_std +
         eta_S * S_std +
         eta_D * D_std +
         eta_W * W_std +
         eta_age * age_std +
         eta_edu * edu_std +
         eta_children * children_std)
    
    mu = 1.0 / (1.0 + np.exp(-z))
    return mu


def test_separation_rate_standardized(
    eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children,
    return_full_data=False
):
    """
    测试给定系数下的平均离职率（标准化版本）
    """
    # 生成1000个虚拟劳动力
    np.random.seed(42)
    
    config = load_config("CONFIG/population_config.yaml")
    labor_model = LaborDistribution(config)
    labor_model.fit()
    
    n_samples = 1000
    
    # 采样
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
    
    # 标准化所有变量
    T_std = (laborers['T'] - laborers['T'].mean()) / laborers['T'].std()
    S_std = (laborers['S'] - laborers['S'].mean()) / laborers['S'].std()
    D_std = (laborers['D'] - laborers['D'].mean()) / laborers['D'].std()
    W_std = (laborers['W'] - laborers['W'].mean()) / laborers['W'].std()
    age_std = (laborers['age'] - laborers['age'].mean()) / laborers['age'].std()
    edu_std = (laborers['education'] - laborers['education'].mean()) / laborers['education'].std()
    children_std = (laborers['children'] - laborers['children'].mean()) / laborers['children'].std()
    
    # 计算每个人的离职率
    mu_list = []
    for i in range(n_samples):
        mu = compute_separation_rate_standardized(
            T_std.iloc[i], S_std.iloc[i], D_std.iloc[i], W_std.iloc[i],
            age_std.iloc[i], edu_std.iloc[i], children_std.iloc[i],
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        mu_list.append(mu)
    
    mu_array = np.array(mu_list)
    
    if return_full_data:
        return mu_array, laborers
    
    mean_mu = mu_array.mean()
    std_mu = mu_array.std()
    min_mu = mu_array.min()
    max_mu = mu_array.max()
    median_mu = np.median(mu_array)
    
    return mean_mu, std_mu, min_mu, max_mu, median_mu


if __name__ == "__main__":
    print("=== 基于标准化变量的离职率系数校准 ===\n")
    print("目标：平均离职率 ≈ 5%，且分布合理\n")
    
    # 初始猜测：所有系数都设为相近的值
    # 预期：负向影响的系数为负，正向影响的系数为正
    # T（工作时间长）→ 稳定性高 → 负系数
    # S（技能高）→ 稳定性高 → 负系数
    # D（数字素养高）→ 稳定性高 → 负系数
    # W（期望工资高）→ 稳定性略低（更容易不满） → 小的正系数
    # age（年龄大）→ 稳定性高 → 负系数
    # education（教育高）→ 稳定性高 → 负系数
    # children（孩子多）→ 稳定性低（家庭负担） → 正系数
    
    # 先进行粗调搜索eta0
    print("第一步：粗调eta0（其他系数暂设为合理初值）")
    print("-" * 90)
    
    # 初始系数设定（基于标准化后的合理性）
    eta_T = -0.3  # 工作时间长 → 稳定
    eta_S = -0.5  # 技能高 → 稳定
    eta_D = -0.3  # 数字素养高 → 稳定
    eta_W = 0.1   # 期望工资高 → 略不稳定
    eta_age = -0.4  # 年龄大 → 稳定
    eta_edu = -0.2  # 教育高 → 稳定
    eta_children = 0.2  # 孩子多 → 不稳定
    
    print(f"初始系数设定（标准化变量）:")
    print(f"  eta_T = {eta_T:.2f}")
    print(f"  eta_S = {eta_S:.2f}")
    print(f"  eta_D = {eta_D:.2f}")
    print(f"  eta_W = {eta_W:.2f}")
    print(f"  eta_age = {eta_age:.2f}")
    print(f"  eta_edu = {eta_edu:.2f}")
    print(f"  eta_children = {eta_children:.2f}")
    print()
    
    print(f"{'eta0':>8} | {'均值':>8} | {'标准差':>8} | {'最小值':>8} | {'中位数':>8} | {'最大值':>8}")
    print("-" * 90)
    
    coarse_results = []
    for eta0 in np.linspace(-2.0, 2.0, 21):
        mean_mu, std_mu, min_mu, max_mu, median_mu = test_separation_rate_standardized(
            eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
        )
        coarse_results.append((eta0, mean_mu, median_mu))
        print(f"{eta0:8.2f} | {mean_mu*100:7.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {median_mu*100:7.2f}% | {max_mu*100:7.2f}%")
    
    print("-" * 90)
    
    # 找到最接近5%的eta0
    target = 0.05
    coarse_diffs = [(eta0, abs(mean_mu - target), median_mu) 
                    for eta0, mean_mu, median_mu in coarse_results]
    coarse_diffs.sort(key=lambda x: x[1])
    
    best_eta0 = coarse_diffs[0][0]
    best_mean_mu = coarse_results[[x[0] for x in coarse_results].index(best_eta0)][1]
    best_median_mu = coarse_diffs[0][2]
    
    print(f"\n粗调最佳: eta0 = {best_eta0:.2f}")
    print(f"  平均离职率 = {best_mean_mu*100:.2f}%")
    print(f"  中位数离职率 = {best_median_mu*100:.2f}%")
    
    # 精细搜索eta0
    print(f"\n第二步：精细搜索eta0")
    fine_search_min = best_eta0 - 0.2
    fine_search_max = best_eta0 + 0.2
    print(f"搜索范围: [{fine_search_min:.2f}, {fine_search_max:.2f}]")
    print("-" * 90)
    print(f"{'eta0':>8} | {'均值':>8} | {'标准差':>8} | {'最小值':>8} | {'中位数':>8} | {'最大值':>8}")
    print("-" * 90)
    
    best_eta0_fine = None
    best_mean_mu_fine = None
    best_diff_fine = float('inf')
    best_median_mu_fine = None
    
    for eta0 in np.linspace(fine_search_min, fine_search_max, 21):
        mean_mu, std_mu, min_mu, max_mu, median_mu = test_separation_rate_standardized(
            eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
        )
        
        print(f"{eta0:8.2f} | {mean_mu*100:7.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {median_mu*100:7.2f}% | {max_mu*100:7.2f}%")
        
        diff = abs(mean_mu - target)
        if diff < best_diff_fine:
            best_diff_fine = diff
            best_eta0_fine = eta0
            best_mean_mu_fine = mean_mu
            best_median_mu_fine = median_mu
    
    print("-" * 90)
    print(f"\n>>> 最终推荐（标准化变量版本）:")
    print(f"    eta0 = {best_eta0_fine:.4f}")
    print(f"    eta_T = {eta_T:.2f}")
    print(f"    eta_S = {eta_S:.2f}")
    print(f"    eta_D = {eta_D:.2f}")
    print(f"    eta_W = {eta_W:.2f}")
    print(f"    eta_age = {eta_age:.2f}")
    print(f"    eta_edu = {eta_edu:.2f}")
    print(f"    eta_children = {eta_children:.2f}")
    print()
    print(f"    平均离职率 = {best_mean_mu_fine*100:.2f}%")
    print(f"    中位数离职率 = {best_median_mu_fine*100:.2f}%")
    print(f"    与目标差距 = {abs(best_mean_mu_fine - target)*100:.2f}%")
    
    # 可视化
    print("\n生成离职率分布可视化...")
    mu_array, laborers = test_separation_rate_standardized(
        best_eta0_fine, eta_T, eta_S, eta_D, eta_W, 
        eta_age, eta_edu, eta_children,
        return_full_data=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    ax1 = axes[0]
    ax1.hist(mu_array * 100, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(best_mean_mu_fine * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均值={best_mean_mu_fine*100:.2f}%')
    ax1.axvline(best_median_mu_fine * 100, color='orange', linestyle='--', linewidth=2,
                label=f'中位数={best_median_mu_fine*100:.2f}%')
    ax1.axvline(target * 100, color='green', linestyle='--', linewidth=2,
                label=f'目标={target*100:.0f}%')
    ax1.set_xlabel('离职率 (%)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title(f'离职率分布（标准化版本，eta0={best_eta0_fine:.3f}）', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：CDF
    ax2 = axes[1]
    sorted_mu = np.sort(mu_array * 100)
    cumulative = np.arange(1, len(sorted_mu) + 1) / len(sorted_mu) * 100
    ax2.plot(sorted_mu, cumulative, color='navy', linewidth=2)
    ax2.axvline(best_mean_mu_fine * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均值={best_mean_mu_fine*100:.2f}%')
    ax2.axvline(best_median_mu_fine * 100, color='orange', linestyle='--', linewidth=2,
                label=f'中位数={best_median_mu_fine*100:.2f}%')
    ax2.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('离职率 (%)', fontsize=12)
    ax2.set_ylabel('累积百分比 (%)', fontsize=12)
    ax2.set_title('离职率累积分布函数 (CDF)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "OUTPUT/mfg/separation_rate_distribution_standardized.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存至: {output_path}")
    
    # 详细统计
    print(f"\n离职率分布统计:")
    print(f"  平均值: {mu_array.mean()*100:.2f}%")
    print(f"  中位数: {np.median(mu_array)*100:.2f}%")
    print(f"  标准差: {mu_array.std()*100:.2f}%")
    print(f"  最小值: {mu_array.min()*100:.2f}%")
    print(f"  25分位: {np.percentile(mu_array, 25)*100:.2f}%")
    print(f"  75分位: {np.percentile(mu_array, 75)*100:.2f}%")
    print(f"  最大值: {mu_array.max()*100:.2f}%")
    print()
    print("【重要】：这些系数适用于标准化后的变量！")
    print("在实际代码中使用时，需要先对变量进行标准化：")
    print("  x_std = (x - x_mean) / x_std")

