import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from MODULES.MFG import solve_equilibrium

print("="*80)
print("测试参数调整后的MFG稳态 - 方案B（激进调整）")
print("="*80)
print("参数设置:")
print("  kappa: 1.0 -> 4.0 (努力成本提高4倍)")
print("  rho: 0.75 -> 0.40 (放大系数从4倍降到1.67倍)")
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

print("\n【对比原参数】")
print("  原参数(rho=0.75, kappa=1.0): T均值 = 70.37")
print(f"  新参数(rho=0.40, kappa=4.0): T均值 = {individuals['T'].mean():.2f}")
print(f"  变化: {individuals['T'].mean() - 70.37:+.2f} 小时/周")
print(f"  变化百分比: {(individuals['T'].mean() - 70.37)/70.37*100:+.1f}%")

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
if 40 <= individuals['T'].mean() <= 50:
    print("  ✓ T值在合理范围内 (40-50小时/周)")
elif individuals['T'].mean() < 40:
    print("  ! T值偏低，可能需要微调参数")
else:
    print("  ! T值仍偏高，需要进一步调整参数")

print("="*80)

