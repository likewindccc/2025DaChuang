import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path.cwd()))

from MODULES.POPULATION import LaborDistribution

# 加载配置
with open('CONFIG/population_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化分布
ld = LaborDistribution(config)

# 采样
df = ld.sample(10000)

print("="*80)
print("人口初始化后的T分布（MFG求解之前）")
print("="*80)
print(df['T'].describe())

print("\n对比:")
print(f"原始数据T均值: 42.24 小时/周")
print(f"初始采样T均值: {df['T'].mean():.2f} 小时/周")
print(f"MFG均衡T均值: 70.37 小时/周")

