import sys
from pathlib import Path
import pickle
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

print("="*80)
print("诊断T值过高的问题")
print("="*80)

# 1. 检查原始数据
print("\n[步骤1] 原始数据中的T分布:")
print("-"*80)
df_raw = pd.read_csv('DATA/processed/cleaned_data.csv')
T_raw = df_raw['每周期望工作天数'] * df_raw['每天期望工作时数']
print(f"均值: {T_raw.mean():.2f} 小时/周")
print(f"中位数: {T_raw.median():.2f} 小时/周")
print(f"标准差: {T_raw.std():.2f}")
print(f"范围: [{T_raw.min():.2f}, {T_raw.max():.2f}]")

# 2. 检查Copula模型采样
print("\n[步骤2] Copula模型采样的T分布:")
print("-"*80)
with open('OUTPUT/population/labor_distribution_params.pkl', 'rb') as f:
    params = pickle.load(f)

copula_model = params['copula_model']
samples = copula_model.sample(10000)
T_copula = samples['T']

print(f"均值: {T_copula.mean():.2f} 小时/周")
print(f"中位数: {T_copula.median():.2f} 小时/周")
print(f"标准差: {T_copula.std():.2f}")
print(f"范围: [{T_copula.min():.2f}, {T_copula.max():.2f}]")

# 3. 检查MFG均衡后的T分布
print("\n[步骤3] MFG均衡后的T分布:")
print("-"*80)
df_eq = pd.read_csv('OUTPUT/mfg/equilibrium_individuals.csv')
T_eq = df_eq['T']

print(f"均值: {T_eq.mean():.2f} 小时/周")
print(f"中位数: {T_eq.median():.2f} 小时/周")
print(f"标准差: {T_eq.std():.2f}")
print(f"范围: [{T_eq.min():.2f}, {T_eq.max():.2f}]")

# 4. 分析变化
print("\n" + "="*80)
print("诊断结果:")
print("="*80)

delta_copula = T_copula.mean() - T_raw.mean()
delta_eq = T_eq.mean() - T_copula.mean()

print(f"原始数据 → Copula采样: {delta_copula:+.2f} 小时/周 ({delta_copula/T_raw.mean()*100:+.1f}%)")
print(f"Copula采样 → MFG均衡: {delta_eq:+.2f} 小时/周 ({delta_eq/T_copula.mean()*100:+.1f}%)")
print(f"总变化: {T_eq.mean() - T_raw.mean():+.2f} 小时/周 ({(T_eq.mean() - T_raw.mean())/T_raw.mean()*100:+.1f}%)")

print("\n问题定位:")
if abs(delta_copula) > 10:
    print("  ⚠️ 主要问题在Copula拟合/采样阶段")
    print("     可能原因：Copula模型拟合不准确，或者数据标准化有问题")
elif abs(delta_eq) > 10:
    print("  ⚠️ 主要问题在MFG均衡求解阶段")
    print("     可能原因：状态更新机制导致T被系统性高估")
else:
    print("  ✓ T值基本保持稳定")

print("\n合理范围参考:")
print("  标准全职工作: 40小时/周")
print("  中国标准工时: 44小时/周")
print("  包含加班: 48-50小时/周")
print(f"  当前均衡: {T_eq.mean():.2f}小时/周 ← {'✗ 严重过高' if T_eq.mean() > 60 else '✓ 合理'}")

