import pandas as pd

baseline = pd.read_csv('OUTPUT/simulation/scenario_baseline/equilibrium_individuals.csv')
low = pd.read_csv('OUTPUT/simulation/scenario_training_low/equilibrium_individuals.csv')
high = pd.read_csv('OUTPUT/simulation/scenario_training_high/equilibrium_individuals.csv')

print('='*80)
print('S Range Analysis')
print('='*80)
print(f'Baseline S: [{baseline["S"].min():.2f}, {baseline["S"].max():.2f}], mean={baseline["S"].mean():.2f}')
print(f'Low Train S: [{low["S"].min():.2f}, {low["S"].max():.2f}], mean={low["S"].mean():.2f}')
print(f'High Train S: [{high["S"].min():.2f}, {high["S"].max():.2f}], mean={high["S"].mean():.2f}')
print()

print('='*80)
print('D Range Analysis')
print('='*80)
print(f'Baseline D: [{baseline["D"].min():.2f}, {baseline["D"].max():.2f}], mean={baseline["D"].mean():.2f}')
print(f'Low Train D: [{low["D"].min():.2f}, {low["D"].max():.2f}], mean={low["D"].mean():.2f}')
print(f'High Train D: [{high["D"].min():.2f}, {high["D"].max():.2f}], mean={high["D"].mean():.2f}')
print()

print('='*80)
print('T and Effort Analysis')
print('='*80)
print(f'Baseline T: mean={baseline["T"].mean():.2f}')
print(f'Low Train T: mean={low["T"].mean():.2f}')
print(f'High Train T: mean={high["T"].mean():.2f}')
print()

policy_baseline = pd.read_csv('OUTPUT/simulation/scenario_baseline/equilibrium_policy.csv')
policy_low = pd.read_csv('OUTPUT/simulation/scenario_training_low/equilibrium_policy.csv')
policy_high = pd.read_csv('OUTPUT/simulation/scenario_training_high/equilibrium_policy.csv')

print(f'Baseline effort: mean={policy_baseline["a_optimal"].mean():.4f}, min={policy_baseline["a_optimal"].min():.4f}, max={policy_baseline["a_optimal"].max():.4f}')
print(f'Low Train effort: mean={policy_low["a_optimal"].mean():.4f}, min={policy_low["a_optimal"].min():.4f}, max={policy_low["a_optimal"].max():.4f}')
print(f'High Train effort: mean={policy_high["a_optimal"].mean():.4f}, min={policy_high["a_optimal"].min():.4f}, max={policy_high["a_optimal"].max():.4f}')
print()

print('='*80)
print('MinMax Standardization Impact Analysis')
print('='*80)
baseline_S_range = baseline["S"].max() - baseline["S"].min()
low_S_range = low["S"].max() - low["S"].min()
high_S_range = high["S"].max() - high["S"].min()

print(f'Baseline S range width: {baseline_S_range:.2f}')
print(f'Low Train S range width: {low_S_range:.2f}')
print(f'High Train S range width: {high_S_range:.2f}')
print()

baseline_mean_S_norm = (baseline["S"].mean() - baseline["S"].min()) / baseline_S_range
low_mean_S_norm = (low["S"].mean() - low["S"].min()) / low_S_range
high_mean_S_norm = (high["S"].mean() - high["S"].min()) / high_S_range

print(f'Baseline mean S_norm: {baseline_mean_S_norm:.4f}, (1-S_norm)={1-baseline_mean_S_norm:.4f}')
print(f'Low Train mean S_norm: {low_mean_S_norm:.4f}, (1-S_norm)={1-low_mean_S_norm:.4f}')
print(f'High Train mean S_norm: {high_mean_S_norm:.4f}, (1-S_norm)={1-high_mean_S_norm:.4f}')

