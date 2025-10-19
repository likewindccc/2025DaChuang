import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from MODULES.MFG import solve_equilibrium

print("="*80)
print("测试T负效用函数 - 第4次调整（同时提高kappa和alpha_T）")
print("="*80)
print("参数设置:")
print("  kappa: 2000.0 (努力成本，从10.0提高200倍！)")
print("  rho: 0.40 (贴现因子)")
print("  alpha_T: 0.3 (T的负效用系数)")
print("  disutility_T: enabled (每个个体的T_ideal = 初始T值)")
print("="*80)
print("诊断结果（kappa=10.0时）:")
print("  努力成本均值 = 0.24 (太小，仅占b0的0.05%！)")
print("  T负效用均值 = 58.23 (占b0的11.65%)")
print("  未来价值均值 = 1806.59 (主导价值函数)")
print("="*80)
print("调整策略:")
print("  kappa: 10.0 → 2000.0 (增大200倍)")
print("  预期努力成本(a=0.2): 0.5*2000*0.04 = 40 (占b0的8%)")
print("  预期：努力成本与T负效用达到同一数量级，显著抑制努力")
print("="*80)
print()

individuals, info = solve_equilibrium('CONFIG/mfg_config.yaml')

print("\n" + "="*80)
print("稳态结果分析")
print("="*80)

print("\n【T的分布】")
print(f"  平均值: {individuals['T'].mean():.2f} 小时/周")
print(f"  中位数: {individuals['T'].median():.2f} 小时/周")
print(f"  标准差: {individuals['T'].std():.2f}")
print(f"  范围: [{individuals['T'].min():.2f}, {individuals['T'].max():.2f}]")

print("\n【对比修改前】")
print("  修改前(无T负效用): T均值 = 65.53")
print(f"  修改后(引入T负效用): T均值 = {individuals['T'].mean():.2f}")
print(f"  变化: {individuals['T'].mean() - 65.53:+.2f} 小时/周")
print(f"  变化百分比: {(individuals['T'].mean() - 65.53)/65.53*100:+.1f}%")
print(f"\n【合理性检查】")
print(f"  目标范围: 40-50 小时/周")
in_range = ((individuals['T'] >= 40) & (individuals['T'] <= 50)).sum()
print(f"  在目标范围内的个体: {in_range}/{len(individuals)} ({in_range/len(individuals)*100:.1f}%)")

print("\n【其他指标】")
print(f"  失业率: {info['final_statistics']['unemployment_rate']*100:.2f}%")
print(f"  平均S: {individuals['S'].mean():.2f}")
print(f"  平均D: {individuals['D'].mean():.2f}")
print(f"  平均W: {individuals['W'].mean():.2f}")

# 读取努力值
import pandas as pd
policy_df = pd.read_csv('OUTPUT/mfg/equilibrium_policy.csv')
print(f"  平均努力: {policy_df['a_optimal'].mean():.4f}")

print("\n【收敛状态】")
print(f"  是否收敛: {'是' if info.get('converged', False) else '否'}")
print(f"  迭代次数: {info.get('iterations', 0)}")

print("\n【结论】")
T_mean = individuals['T'].mean()
if 40 <= T_mean <= 50:
    print(f"  ✓ T值进入合理范围 ({T_mean:.2f} 小时/周)！")
    print("  ✓ T的负效用函数成功抑制了过度劳动供给")
elif T_mean > 50 and T_mean < 55:
    print(f"  ~ T值略偏高 ({T_mean:.2f} 小时/周)")
    print("  建议：增大alpha_T (0.001 -> 0.002)")
elif T_mean >= 55:
    print(f"  ! T值仍然偏高 ({T_mean:.2f} 小时/周)")
    print("  建议：大幅增大alpha_T (0.001 -> 0.005 或更高)")
else:
    print(f"  ! T值偏低 ({T_mean:.2f} 小时/周)")
    print("  建议：降低alpha_T (0.001 -> 0.0005)")

print("="*80)

