#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
劳动力对企业偏好得分分布分析

分析所有劳动力对企业的偏好得分的统计分布，
帮助理解匹配算法中的偏好集中度问题。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config
from MODULES.LOGISTIC.gs_matching import compute_laborer_preferences_core


def analyze_preference_score_distribution():
    """分析劳动力对企业的偏好得分分布"""
    print("=" * 80)
    print("劳动力对企业偏好得分分布分析（使用完整模拟配置）")
    print("=" * 80)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 从配置读取参数
    n_laborers = config['market_size']['n_laborers']
    # 使用均衡市场的theta均值
    theta_min = config['data_generation']['theta_scenarios']['balanced']['min']
    theta_max = config['data_generation']['theta_scenarios']['balanced']['max']
    theta = (theta_min + theta_max) / 2
    
    print(f"\n配置参数:")
    print(f"  劳动力数量: {n_laborers}")
    print(f"  市场紧张度: {theta:.2f} (均衡市场均值)")
    
    # 生成虚拟市场
    print("\n生成虚拟市场...")
    market_gen = VirtualMarket(config)
    laborers, enterprises = market_gen.generate_market(n_laborers=n_laborers, theta=theta)
    
    print(f"  实际劳动力: {len(laborers)}")
    print(f"  实际企业: {len(enterprises)}")
    
    # 提取原始数据
    S_i_raw = laborers['S'].values
    D_i_raw = laborers['D'].values
    T_i_raw = laborers['T'].values
    W_i_raw = laborers['W'].values
    
    T_req_raw = enterprises['T_req'].values
    S_req_raw = enterprises['S_req'].values
    D_req_raw = enterprises['D_req'].values
    W_offer_raw = enterprises['W_offer'].values
    
    # 读取偏好函数参数
    laborer_params = config['gs_matching']['laborer_preference']
    gamma_0 = laborer_params['gamma_0']
    gamma_1 = laborer_params['gamma_1']
    gamma_2 = laborer_params['gamma_2']
    gamma_3 = laborer_params['gamma_3']
    gamma_4 = laborer_params['gamma_4']
    
    print(f"\n偏好函数参数:")
    print(f"  γ_0={gamma_0}, γ_1={gamma_1}, γ_2={gamma_2}, γ_3={gamma_3}, γ_4={gamma_4}")
    
    # 使用实际GS匹配中的偏好计算函数（numba加速版本）
    print("\n计算偏好得分矩阵（使用GS匹配算法）...")
    preference_scores = compute_laborer_preferences_core(
        S_i_raw, D_i_raw,
        T_req_raw, S_req_raw, D_req_raw, W_offer_raw,
        gamma_0, gamma_1, gamma_2, gamma_3, gamma_4
    )
    
    print("  [完成]")
    
    # 统计分析
    print("\n" + "=" * 80)
    print("偏好得分统计")
    print("=" * 80)
    
    # 所有偏好得分
    all_scores = preference_scores.flatten()
    print(f"\n所有偏好得分（{len(all_scores)}个）:")
    print(f"  均值: {all_scores.mean():.4f}")
    print(f"  标准差: {all_scores.std():.4f}")
    print(f"  最小值: {all_scores.min():.4f}")
    print(f"  最大值: {all_scores.max():.4f}")
    print(f"  中位数: {np.median(all_scores):.4f}")
    print(f"  25分位: {np.percentile(all_scores, 25):.4f}")
    print(f"  75分位: {np.percentile(all_scores, 75):.4f}")
    
    # 每个劳动力的最高偏好得分
    max_scores_per_laborer = preference_scores.max(axis=1)
    print(f"\n每个劳动力的最高偏好得分:")
    print(f"  均值: {max_scores_per_laborer.mean():.4f}")
    print(f"  标准差: {max_scores_per_laborer.std():.4f}")
    print(f"  最小值: {max_scores_per_laborer.min():.4f}")
    print(f"  最大值: {max_scores_per_laborer.max():.4f}")
    
    # 每个企业收到的平均偏好得分
    mean_scores_per_enterprise = preference_scores.mean(axis=0)
    print(f"\n每个企业收到的平均偏好得分:")
    print(f"  均值: {mean_scores_per_enterprise.mean():.4f}")
    print(f"  标准差: {mean_scores_per_enterprise.std():.4f}")
    print(f"  最小值: {mean_scores_per_enterprise.min():.4f}")
    print(f"  最大值: {mean_scores_per_enterprise.max():.4f}")
    
    # 企业吸引力排名（按平均偏好得分）
    top_k = 10
    top_enterprises = np.argsort(-mean_scores_per_enterprise)[:top_k]
    print(f"\n最受欢迎的前{top_k}个企业:")
    for rank, eid in enumerate(top_enterprises, 1):
        print(f"  {rank}. 企业{eid}: 平均得分={mean_scores_per_enterprise[eid]:.4f}")
    
    # 可视化
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('劳动力对企业偏好得分分布分析', fontsize=16)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 所有偏好得分分布
    ax1 = axes[0, 0]
    ax1.hist(all_scores, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('偏好得分', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('所有偏好得分分布', fontsize=12)
    ax1.axvline(all_scores.mean(), color='red', linestyle='--', label=f'均值={all_scores.mean():.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 每个劳动力的最高偏好得分分布
    ax2 = axes[0, 1]
    ax2.hist(max_scores_per_laborer, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('最高偏好得分', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('每个劳动力的最高偏好得分分布', fontsize=12)
    ax2.axvline(max_scores_per_laborer.mean(), color='red', linestyle='--', 
                label=f'均值={max_scores_per_laborer.mean():.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 每个企业收到的平均偏好得分分布
    ax3 = axes[1, 0]
    ax3.hist(mean_scores_per_enterprise, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('平均偏好得分', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('每个企业收到的平均偏好得分分布', fontsize=12)
    ax3.axvline(mean_scores_per_enterprise.mean(), color='red', linestyle='--',
                label=f'均值={mean_scores_per_enterprise.mean():.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 企业吸引力排名（前20）
    ax4 = axes[1, 1]
    top_20 = np.argsort(-mean_scores_per_enterprise)[:20]
    ax4.bar(range(20), mean_scores_per_enterprise[top_20], color='purple', alpha=0.7)
    ax4.set_xlabel('企业排名', fontsize=12)
    ax4.set_ylabel('平均偏好得分', fontsize=12)
    ax4.set_title('最受欢迎的前20个企业', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图形
    output_dir = Path("OUTPUT/logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "preference_score_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图形已保存至: {output_file}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    analyze_preference_score_distribution()

