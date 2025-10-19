#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
价值函数各项贡献分析

分析贝尔曼方程中各项的绝对值量级，诊断为什么负效用函数不起作用
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path.cwd()))

print("="*80)
print("价值函数各项贡献分析")
print("="*80)
print()

config_path = "CONFIG/mfg_config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

individuals = pd.read_csv("OUTPUT/mfg/equilibrium_individuals.csv")
policy = pd.read_csv("OUTPUT/mfg/equilibrium_policy.csv")

individuals = pd.concat([individuals, policy[['a_optimal', 'V_U', 'V_E']]], axis=1)

unemployed = individuals[individuals['employment_status'] == 'unemployed'].copy()

print(f"分析 {len(unemployed)} 个失业者的价值函数各项贡献")
print()

rho = config['economics']['rho']
kappa = config['economics']['kappa']
b0 = config['economics']['unemployment_benefit']['b0']
alpha_T = config['economics']['disutility_T']['alpha']

print("【参数设置】")
print(f"  rho (贴现因子): {rho}")
print(f"  kappa (努力成本系数): {kappa}")
print(f"  b0 (失业救济金): {b0}")
print(f"  alpha_T (T负效用系数): {alpha_T}")
print()

unemployed['effort_cost'] = 0.5 * kappa * unemployed['a_optimal'] ** 2

from MODULES.POPULATION import LaborDistribution
pop_config_path = "CONFIG/population_config.yaml"
with open(pop_config_path, 'r', encoding='utf-8') as f:
    pop_config = yaml.safe_load(f)

labor_model = LaborDistribution(pop_config)
labor_model.fit()
continuous_samples = labor_model.copula_model.sample(len(unemployed))
initial_T_sampled = continuous_samples['T'].values[:len(unemployed)]

unemployed['T_deviation'] = unemployed['T'] - initial_T_sampled[:len(unemployed)]
unemployed['disutility_T'] = alpha_T * unemployed['T_deviation'] ** 2

unemployed['instant_utility'] = b0 - unemployed['effort_cost'] - unemployed['disutility_T']

unemployed['future_value'] = rho * (
    unemployed['V_E'] * 0.5 + unemployed['V_U'] * 0.5
)

print("="*80)
print("价值函数各项统计（失业者）")
print("="*80)
print()

components = {
    'b0 (失业救济金)': b0,
    'effort_cost (努力成本)': unemployed['effort_cost'],
    'disutility_T (T负效用)': unemployed['disutility_T'],
    'instant_utility (即时效用)': unemployed['instant_utility'],
    'future_value (未来期望价值)': unemployed['future_value'],
    'V_U (总价值函数)': unemployed['V_U']
}

for name, values in components.items():
    if isinstance(values, (int, float)):
        print(f"{name:30s}: 固定值 = {values:.2f}")
    else:
        print(f"{name:30s}: 均值={values.mean():8.2f}, 标准差={values.std():7.2f}, "
              f"范围=[{values.min():7.2f}, {values.max():7.2f}]")

print()
print("="*80)
print("关键比例分析")
print("="*80)
print()

avg_effort_cost = unemployed['effort_cost'].mean()
avg_disutility_T = unemployed['disutility_T'].mean()
avg_future_value = unemployed['future_value'].mean()

print(f"1. 努力成本 / 失业救济金 = {avg_effort_cost / b0 * 100:.2f}%")
print(f"2. T负效用 / 失业救济金 = {avg_disutility_T / b0 * 100:.2f}%")
print(f"3. (努力成本 + T负效用) / 失业救济金 = {(avg_effort_cost + avg_disutility_T) / b0 * 100:.2f}%")
print(f"4. 未来期望价值 / 失业救济金 = {avg_future_value / b0 * 100:.2f}%")
print()
print(f"5. T负效用 / 未来期望价值 = {avg_disutility_T / avg_future_value * 100:.2f}%")
print(f"6. 努力成本 / 未来期望价值 = {avg_effort_cost / avg_future_value * 100:.2f}%")
print(f"7. (努力成本 + T负效用) / 未来期望价值 = {(avg_effort_cost + avg_disutility_T) / avg_future_value * 100:.2f}%")
print()

print("="*80)
print("T偏离分析")
print("="*80)
print()

print(f"【T偏离统计】")
print(f"  平均偏离: {unemployed['T_deviation'].mean():.2f} 小时/周")
print(f"  标准差: {unemployed['T_deviation'].std():.2f}")
print(f"  范围: [{unemployed['T_deviation'].min():.2f}, {unemployed['T_deviation'].max():.2f}]")
print()

print("【T偏离分布】")
bins = [0, 5, 10, 15, 20, 25, 100]
labels = ['0-5h', '5-10h', '10-15h', '15-20h', '20-25h', '>25h']
unemployed['T_deviation_abs'] = unemployed['T_deviation'].abs()
unemployed['deviation_bin'] = pd.cut(unemployed['T_deviation_abs'], bins=bins, labels=labels)

for label in labels:
    count = (unemployed['deviation_bin'] == label).sum()
    pct = count / len(unemployed) * 100
    if count > 0:
        avg_disutil = unemployed[unemployed['deviation_bin'] == label]['disutility_T'].mean()
        print(f"  {label:8s}: {count:5d} 人 ({pct:5.1f}%), 平均负效用={avg_disutil:7.2f}")

print()
print("="*80)
print("边际分析：T增加1小时的影响")
print("="*80)
print()

sample_T = unemployed['T'].median()
sample_deviation = unemployed['T_deviation'].median()

marginal_disutility = alpha_T * ((sample_deviation + 1)**2 - sample_deviation**2)
marginal_disutility_approx = alpha_T * 2 * sample_deviation

print(f"假设当前 T 偏离中位数: {sample_deviation:.2f} 小时")
print(f"T 增加 1 小时的边际负效用:")
print(f"  精确计算: {marginal_disutility:.4f}")
print(f"  近似计算 (2*alpha_T*deviation): {marginal_disutility_approx:.4f}")
print()
print(f"对比:")
print(f"  边际负效用 vs 失业救济金: {marginal_disutility / b0 * 100:.3f}%")
print(f"  边际负效用 vs 未来期望价值: {marginal_disutility / avg_future_value * 100:.3f}%")
print()

print("【问题诊断】")
if avg_disutility_T / avg_future_value < 0.05:
    print(f"  ❌ T负效用仅占未来价值的 {avg_disutility_T / avg_future_value * 100:.2f}%，太小！")
    print(f"  建议: alpha_T 需要增大至少 {0.05 / (avg_disutility_T / avg_future_value):.0f} 倍")
elif avg_disutility_T / avg_future_value < 0.10:
    print(f"  ⚠️  T负效用占未来价值的 {avg_disutility_T / avg_future_value * 100:.2f}%，仍偏小")
    print(f"  建议: alpha_T 可继续增大 2-3 倍")
else:
    print(f"  ✓  T负效用占未来价值的 {avg_disutility_T / avg_future_value * 100:.2f}%，量级合理")

print()
print("="*80)
print("推荐alpha_T值")
print("="*80)
print()

target_ratio = 0.10
recommended_alpha = alpha_T * (target_ratio * avg_future_value / avg_disutility_T)
print(f"目标: T负效用 = 未来价值的10%")
print(f"当前alpha_T: {alpha_T}")
print(f"推荐alpha_T: {recommended_alpha:.4f} (约为当前的 {recommended_alpha/alpha_T:.1f} 倍)")
print()

print("="*80)

