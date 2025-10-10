#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精细调整标准化离职率系数

目标：平均离职率 ≈ 5%，且分布合理
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
    """计算离职率（基于标准化变量）"""
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
    """测试给定系数下的平均离职率（标准化版本）"""
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
    
    # 标准化
    T_std = (laborers['T'] - laborers['T'].mean()) / laborers['T'].std()
    S_std = (laborers['S'] - laborers['S'].mean()) / laborers['S'].std()
    D_std = (laborers['D'] - laborers['D'].mean()) / laborers['D'].std()
    W_std = (laborers['W'] - laborers['W'].mean()) / laborers['W'].std()
    age_std = (laborers['age'] - laborers['age'].mean()) / laborers['age'].std()
    edu_std = (laborers['education'] - laborers['education'].mean()) / laborers['education'].std()
    children_std = (laborers['children'] - laborers['children'].mean()) / laborers['children'].std()
    
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
    
    return (mu_array.mean(), mu_array.std(), mu_array.min(), 
            mu_array.max(), np.median(mu_array))


if __name__ == "__main__":
    print("=== 精细调整标准化离职率系数 ===\n")
    print("策略：增大负向系数的绝对值，使平均离职率降到5%\n")
    
    # 从初步结果来看，eta0=-2.2时平均值约11.58%
    # 需要进一步降低，可以通过两种方式：
    # 1. 进一步降低eta0
    # 2. 增大负向系数的绝对值
    
    # 尝试方案：增大负向系数
    eta_T = -0.5  # 从-0.3增加到-0.5
    eta_S = -0.8  # 从-0.5增加到-0.8
    eta_D = -0.5  # 从-0.3增加到-0.5
    eta_W = 0.05  # 从0.1降到0.05
    eta_age = -0.6  # 从-0.4增加到-0.6
    eta_edu = -0.3  # 从-0.2增加到-0.3
    eta_children = 0.15  # 从0.2降到0.15
    
    print(f"调整后的系数设定:")
    print(f"  eta_T = {eta_T:.2f} (工作时间长→稳定)")
    print(f"  eta_S = {eta_S:.2f} (技能高→稳定)")
    print(f"  eta_D = {eta_D:.2f} (数字素养高→稳定)")
    print(f"  eta_W = {eta_W:.2f} (期望工资高→略不稳定)")
    print(f"  eta_age = {eta_age:.2f} (年龄大→稳定)")
    print(f"  eta_edu = {eta_edu:.2f} (教育高→稳定)")
    print(f"  eta_children = {eta_children:.2f} (孩子多→不稳定)")
    print()
    
    # 粗调eta0
    print("第一步：粗调eta0")
    print("-" * 90)
    print(f"{'eta0':>8} | {'均值':>8} | {'标准差':>8} | {'最小值':>8} | {'中位数':>8} | {'最大值':>8}")
    print("-" * 90)
    
    target = 0.05
    coarse_results = []
    for eta0 in np.linspace(-4.0, -2.0, 21):
        mean_mu, std_mu, min_mu, max_mu, median_mu = test_separation_rate_standardized(
            eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
        )
        coarse_results.append((eta0, mean_mu, median_mu))
        print(f"{eta0:8.2f} | {mean_mu*100:7.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {median_mu*100:7.2f}% | {max_mu*100:7.2f}%")
    
    print("-" * 90)
    
    # 找最接近5%的
    coarse_diffs = [(eta0, abs(mean_mu - target), median_mu) 
                    for eta0, mean_mu, median_mu in coarse_results]
    coarse_diffs.sort(key=lambda x: x[1])
    
    best_eta0_coarse = coarse_diffs[0][0]
    best_mean_coarse = [x[1] for x in coarse_results if x[0] == best_eta0_coarse][0]
    best_median_coarse = coarse_diffs[0][2]
    
    print(f"\n粗调最佳: eta0 = {best_eta0_coarse:.2f}")
    print(f"  平均离职率 = {best_mean_coarse*100:.2f}%")
    print(f"  中位数离职率 = {best_median_coarse*100:.2f}%")
    
    # 精细调整
    print(f"\n第二步：精细搜索eta0")
    fine_min = best_eta0_coarse - 0.2
    fine_max = best_eta0_coarse + 0.2
    print(f"搜索范围: [{fine_min:.2f}, {fine_max:.2f}]")
    print("-" * 90)
    print(f"{'eta0':>8} | {'均值':>8} | {'标准差':>8} | {'最小值':>8} | {'中位数':>8} | {'最大值':>8}")
    print("-" * 90)
    
    best_eta0 = None
    best_mean = None
    best_median = None
    best_diff = float('inf')
    
    for eta0 in np.linspace(fine_min, fine_max, 21):
        mean_mu, std_mu, min_mu, max_mu, median_mu = test_separation_rate_standardized(
            eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
        )
        
        print(f"{eta0:8.2f} | {mean_mu*100:7.2f}% | {std_mu*100:7.2f}% | "
              f"{min_mu*100:7.2f}% | {median_mu*100:7.2f}% | {max_mu*100:7.2f}%")
        
        diff = abs(mean_mu - target)
        if diff < best_diff:
            best_diff = diff
            best_eta0 = eta0
            best_mean = mean_mu
            best_median = median_mu
    
    print("-" * 90)
    print(f"\n>>> 最终推荐（标准化变量版本）:")
    print(f"    eta0 = {best_eta0:.4f}")
    print(f"    eta_T = {eta_T:.2f}")
    print(f"    eta_S = {eta_S:.2f}")
    print(f"    eta_D = {eta_D:.2f}")
    print(f"    eta_W = {eta_W:.2f}")
    print(f"    eta_age = {eta_age:.2f}")
    print(f"    eta_edu = {eta_edu:.2f}")
    print(f"    eta_children = {eta_children:.2f}")
    print()
    print(f"    平均离职率 = {best_mean*100:.2f}%")
    print(f"    中位数离职率 = {best_median*100:.2f}%")
    print(f"    与目标5%差距 = {abs(best_mean - target)*100:.2f}%")
    
    # 可视化
    print("\n生成最终离职率分布可视化...")
    mu_array, laborers = test_separation_rate_standardized(
        best_eta0, eta_T, eta_S, eta_D, eta_W, 
        eta_age, eta_edu, eta_children,
        return_full_data=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图
    ax1 = axes[0]
    ax1.hist(mu_array * 100, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax1.axvline(best_mean * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均={best_mean*100:.2f}%')
    ax1.axvline(best_median * 100, color='orange', linestyle='--', linewidth=2,
                label=f'中位数={best_median*100:.2f}%')
    ax1.axvline(target * 100, color='green', linestyle='--', linewidth=2.5,
                label=f'目标={target*100:.0f}%')
    ax1.set_xlabel('离职率 (%)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title(f'最终离职率分布（标准化，eta0={best_eta0:.3f}）', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：CDF
    ax2 = axes[1]
    sorted_mu = np.sort(mu_array * 100)
    cumulative = np.arange(1, len(sorted_mu) + 1) / len(sorted_mu) * 100
    ax2.plot(sorted_mu, cumulative, color='darkgreen', linewidth=2)
    ax2.axvline(best_mean * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均={best_mean*100:.2f}%')
    ax2.axvline(best_median * 100, color='orange', linestyle='--', linewidth=2,
                label=f'中位数={best_median*100:.2f}%')
    ax2.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('离职率 (%)', fontsize=12)
    ax2.set_ylabel('累积百分比 (%)', fontsize=12)
    ax2.set_title('离职率累积分布函数 (CDF)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "OUTPUT/mfg/separation_rate_final_standardized.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存至: {output_path}")
    
    # 详细统计
    print(f"\n最终离职率分布统计:")
    print(f"  平均值: {mu_array.mean()*100:.2f}%")
    print(f"  中位数: {np.median(mu_array)*100:.2f}%")
    print(f"  标准差: {mu_array.std()*100:.2f}%")
    print(f"  最小值: {mu_array.min()*100:.2f}%")
    print(f"  10分位: {np.percentile(mu_array, 10)*100:.2f}%")
    print(f"  25分位: {np.percentile(mu_array, 25)*100:.2f}%")
    print(f"  75分位: {np.percentile(mu_array, 75)*100:.2f}%")
    print(f"  90分位: {np.percentile(mu_array, 90)*100:.2f}%")
    print(f"  最大值: {mu_array.max()*100:.2f}%")
    print()
    print("=" * 90)
    print("【重要提示】")
    print("=" * 90)
    print("这些系数必须与标准化后的变量一起使用！")
    print()
    print("在bellman_solver.py和kfe_solver.py中实现时：")
    print("1. 先计算所有个体的均值和标准差（群体层面）")
    print("2. 对每个个体进行标准化：x_std = (x - mean) / std")
    print("3. 然后用标准化后的变量和这些系数计算离职率")

