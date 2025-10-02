"""
快速参数测试 - 找到合适的参数范围

策略：
1. 测试少量代表性配置（~10个）
2. 快速评估匹配率
3. 根据结果调整参数范围
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.population.labor_generator import LaborGenerator
from src.modules.population.enterprise_generator import EnterpriseGenerator
from src.modules.matching.gale_shapley import limited_rounds_matching
from src.modules.matching.preference import (
    compute_labor_preference_matrix,
    compute_enterprise_preference_matrix,
    compute_preference_rankings
)


def test_quick_config(
    labor_gen,
    enterprise_gen,
    gamma_params,
    beta_params,
    noise_std=0.3,
    n_labor=200,  # 减少样本数
    theta_list=[0.7, 1.0, 1.3],
    n_repeats=3,  # 减少重复次数
    max_rounds=1  # 单轮匹配
):
    """快速测试单个配置"""
    results = {theta: [] for theta in theta_list}
    
    for repeat in range(n_repeats):
        seed = repeat
        np.random.seed(seed)
        
        for theta in theta_list:
            n_enterprise = int(n_labor * theta)
            
            # 生成数据
            labor_df = labor_gen.generate(n_agents=n_labor)
            enterprise_df = enterprise_gen.generate(n_agents=n_enterprise)
            
            labor_features = labor_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
            enterprise_features = enterprise_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
            
            # 计算偏好
            labor_pref = compute_labor_preference_matrix(
                labor_features, enterprise_features, **gamma_params
            )
            
            enterprise_pref = compute_enterprise_preference_matrix(
                enterprise_features, labor_features, **beta_params
            )
            
            # 添加扰动
            np.random.seed(seed * 1000 + int(theta * 10))
            labor_pref += np.random.normal(0, noise_std, labor_pref.shape)
            enterprise_pref += np.random.normal(0, noise_std, enterprise_pref.shape)
            
            # 转换为排序
            labor_pref_order = compute_preference_rankings(labor_pref)
            enterprise_pref_order = compute_preference_rankings(enterprise_pref)
            
            # 执行单轮匹配
            matching = limited_rounds_matching(
                labor_pref_order, enterprise_pref_order, max_rounds=max_rounds
            )
            
            # 计算匹配率
            match_rate = np.sum(matching != -1) / n_labor
            results[theta].append(match_rate)
    
    # 计算平均匹配率
    avg_results = {theta: np.mean(results[theta]) for theta in theta_list}
    return avg_results


def analyze_preference_scale(labor_gen, enterprise_gen):
    """分析偏好函数各项的数量级"""
    print("\n分析偏好函数各项的数量级...")
    print("-" * 80)
    
    # 生成测试数据
    labor_df = labor_gen.generate(n_agents=100)
    enterprise_df = enterprise_gen.generate(n_agents=100)
    
    # 统计特征范围
    print("\n特征统计:")
    for feat in ['T', 'S', 'D', 'W']:
        labor_vals = labor_df[feat].values
        ent_vals = enterprise_df[feat].values
        print(f"  {feat}:")
        print(f"    劳动力: [{labor_vals.min():.2f}, {labor_vals.max():.2f}], 均值={labor_vals.mean():.2f}")
        print(f"    企业:   [{ent_vals.min():.2f}, {ent_vals.max():.2f}], 均值={ent_vals.mean():.2f}")
    
    # 估算合理的参数范围
    T_range = labor_df['T'].max() - labor_df['T'].min()
    S_range = labor_df['S'].max() - labor_df['S'].min()
    D_range = labor_df['D'].max() - labor_df['D'].min()
    W_range = labor_df['W'].max() - labor_df['W'].min()
    
    print("\n建议参数范围（目标：各项贡献在10-50之间）:")
    print(f"  gamma_1 (T权重): {10/T_range:.6f} ~ {50/T_range:.6f}")
    print(f"  gamma_2 (S权重): {10/S_range:.6f} ~ {50/S_range:.6f}")
    print(f"  gamma_3 (D权重): {10/D_range:.6f} ~ {50/D_range:.6f}")
    print(f"  gamma_4 (W权重): {10/W_range:.6f} ~ {50/W_range:.6f}")
    print(f"  beta_1-3: 同上")
    print(f"  beta_4: {-50/W_range:.6f} ~ {-10/W_range:.6f} (负值)")


def main():
    print("=" * 80)
    print("快速参数测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载数据并拟合生成器
    print("加载数据并拟合生成器...")
    real_data = pd.read_csv('data/input/cleaned_data.csv')
    real_data['T'] = real_data['每周期望工作天数'] * real_data['每天期望工作时数']
    real_data['S'] = real_data['工作能力评分']
    real_data['D'] = real_data['数字素养评分']
    real_data['W'] = real_data['每月期望收入']
    
    labor_gen = LaborGenerator()
    enterprise_gen = EnterpriseGenerator()
    
    labor_gen.fit(real_data)
    enterprise_gen.fit(real_data)
    print("[OK] 生成器拟合完成\n")
    
    # 分析特征范围
    analyze_preference_scale(labor_gen, enterprise_gen)
    
    # 定义测试配置（基于分析结果手动选择）
    print("\n" + "=" * 80)
    print("测试配置")
    print("=" * 80)
    
    test_configs = [
        # 配置1: 较小权重
        {
            'name': '配置1 - 小权重',
            'gamma': {'gamma_0': 1.0, 'gamma_1': 0.01, 'gamma_2': 0.02, 'gamma_3': 0.05, 'gamma_4': 0.0001},
            'beta': {'beta_0': 0.0, 'beta_1': 0.01, 'beta_2': 0.02, 'beta_3': 0.05, 'beta_4': -0.0001}
        },
        # 配置2: 中等权重
        {
            'name': '配置2 - 中权重',
            'gamma': {'gamma_0': 1.0, 'gamma_1': 0.02, 'gamma_2': 0.04, 'gamma_3': 0.08, 'gamma_4': 0.0002},
            'beta': {'beta_0': 0.0, 'beta_1': 0.02, 'beta_2': 0.04, 'beta_3': 0.08, 'beta_4': -0.0002}
        },
        # 配置3: 较大权重
        {
            'name': '配置3 - 大权重',
            'gamma': {'gamma_0': 1.0, 'gamma_1': 0.04, 'gamma_2': 0.08, 'gamma_3': 0.15, 'gamma_4': 0.0004},
            'beta': {'beta_0': 0.0, 'beta_1': 0.04, 'beta_2': 0.08, 'beta_3': 0.15, 'beta_4': -0.0004}
        },
        # 配置4: 强调工资
        {
            'name': '配置4 - 强调工资',
            'gamma': {'gamma_0': 1.0, 'gamma_1': 0.02, 'gamma_2': 0.03, 'gamma_3': 0.05, 'gamma_4': 0.0005},
            'beta': {'beta_0': 0.0, 'beta_1': 0.02, 'beta_2': 0.03, 'beta_3': 0.05, 'beta_4': -0.0005}
        },
        # 配置5: 弱化工资
        {
            'name': '配置5 - 弱化工资',
            'gamma': {'gamma_0': 1.0, 'gamma_1': 0.03, 'gamma_2': 0.05, 'gamma_3': 0.10, 'gamma_4': 0.00005},
            'beta': {'beta_0': 0.0, 'beta_1': 0.03, 'beta_2': 0.05, 'beta_3': 0.10, 'beta_4': -0.00005}
        },
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n测试 {i}/{len(test_configs)}: {config['name']}")
        print("-" * 80)
        print(f"  劳动力参数: {config['gamma']}")
        print(f"  企业参数:   {config['beta']}")
        
        start_time = time.time()
        match_rates = test_quick_config(
            labor_gen, enterprise_gen,
            config['gamma'], config['beta']
        )
        elapsed = time.time() - start_time
        
        print(f"\n  结果 (耗时 {elapsed:.1f}秒):")
        print(f"    theta=0.7 -> {match_rates[0.7]:.2%}")
        print(f"    theta=1.0 -> {match_rates[1.0]:.2%}")
        print(f"    theta=1.3 -> {match_rates[1.3]:.2%}")
        
        # 评估
        monotonic = (match_rates[1.0] > match_rates[0.7]) and (match_rates[1.3] > match_rates[1.0])
        in_range = all(0.3 <= rate <= 0.8 for rate in match_rates.values())
        
        status = []
        if monotonic:
            status.append("单调递增")
        else:
            status.append("非单调")
        if in_range:
            status.append("范围合理")
        else:
            status.append("范围不佳")
        
        print(f"  评估: {', '.join(status)}")
        
        results.append({
            'config': config['name'],
            'theta_0.7': match_rates[0.7],
            'theta_1.0': match_rates[1.0],
            'theta_1.3': match_rates[1.3],
            'monotonic': monotonic,
            'in_range': in_range,
            **config['gamma'],
            **config['beta']
        })
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 保存结果
    output_path = Path('results/quick_test_results.csv')
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] 结果已保存: {output_path}")
    
    # 建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    good_configs = results_df[results_df['monotonic'] & results_df['in_range']]
    if len(good_configs) > 0:
        print("\n表现良好的配置:")
        for idx, row in good_configs.iterrows():
            print(f"\n  {row['config']}:")
            print(f"    匹配率: {row['theta_0.7']:.2%} -> {row['theta_1.0']:.2%} -> {row['theta_1.3']:.2%}")
            print(f"    gamma: [{row['gamma_1']:.6f}, {row['gamma_2']:.6f}, {row['gamma_3']:.6f}, {row['gamma_4']:.6f}]")
            print(f"    beta:  [{row['beta_1']:.6f}, {row['beta_2']:.6f}, {row['beta_3']:.6f}, {row['beta_4']:.6f}]")
    else:
        print("\n没有找到完全符合要求的配置，需要继续调整参数范围。")
        print("建议:")
        print("  1. 如果匹配率过低：增大gamma_4（工资权重）")
        print("  2. 如果匹配率过高：减小gamma_4，或增大其他惩罚项")
        print("  3. 如果不单调：检查参数配置，增加随机扰动")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

