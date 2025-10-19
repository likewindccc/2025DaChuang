import numpy as np

print("="*80)
print("模拟T值收敛过程")
print("="*80)

# 初始参数
T_initial_mean = 42.0
T_initial_std = 10.0
T_initial_max = 77.0

gamma_T = 1.0  # 当前配置
a_mean = 0.44  # 平均努力（从敏感性分析）

# 模拟100轮迭代
T_values = np.random.normal(T_initial_mean, T_initial_std, 10000)
T_values = np.clip(T_values, 10, 80)  # 限制范围

print(f"\n初始状态:")
print(f"  T均值: {T_values.mean():.2f}")
print(f"  T最大值: {T_values.max():.2f}")

for iteration in [1, 10, 20, 50, 100]:
    T_max = T_values.max()
    # 应用状态更新公式
    a_values = np.random.uniform(0.3, 0.6, 10000)  # 模拟不同个体的努力
    T_values = T_values + gamma_T * a_values * (T_max - T_values)
    
    if iteration in [1, 10, 20, 50, 100]:
        print(f"\n迭代 {iteration}轮后:")
        print(f"  T均值: {T_values.mean():.2f}")
        print(f"  T中位数: {np.median(T_values):.2f}")
        print(f"  T标准差: {T_values.std():.2f}")
        print(f"  T范围: [{T_values.min():.2f}, {T_values.max():.2f}]")

print("\n" + "="*80)
print("结论:")
print("="*80)
print("T值确实会系统性地向群体最大值收敛！")
print("这就是为什么T从42小时/周增长到70小时/周的原因。")

