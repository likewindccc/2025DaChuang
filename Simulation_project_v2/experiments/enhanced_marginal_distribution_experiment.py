#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版边际分布拟合实验
为Core模块和Population模块开发选择最佳的边际分布类型

新增功能：
1. 包含4个核心变量 + 4个控制变量的分布拟合
2. 相关性分析（相关系数矩阵热图）
3. 可视化对比图（理论分布 vs 原始数据）

运行方式：
1. 激活虚拟环境: D:\Python\2025DaChuang\venv\Scripts\Activate.ps1
2. 进入项目目录: cd D:\Python\2025DaChuang\Simulation_project_v2
3. 运行: python experiments/enhanced_marginal_distribution_experiment.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 核心状态变量（用于匹配模型）
CORE_VARIABLES = [
    '每周工作时长',    # T
    '工作能力评分',    # S
    '数字素养评分',    # D
    '每月期望收入'     # W
]

# 控制变量（用于匹配函数和离职率）
CONTROL_VARIABLES_CONTINUOUS = [
    '年龄',
    '累计工作年限'
]

CONTROL_VARIABLES_DISCRETE = [
    '孩子数量',
    '学历'           # 受教育年限
]

# 所有变量分类
CONTINUOUS_VARIABLES = CORE_VARIABLES + CONTROL_VARIABLES_CONTINUOUS
DISCRETE_VARIABLES = CONTROL_VARIABLES_DISCRETE
ALL_VARIABLES = CONTINUOUS_VARIABLES + DISCRETE_VARIABLES

# 候选分布
CANDIDATE_DISTRIBUTIONS = {
    'beta': stats.beta,
    'gamma': stats.gamma,
    'lognorm': stats.lognorm,
    'norm': stats.norm,
    'weibull_min': stats.weibull_min,
    'expon': stats.expon,
    'uniform': stats.uniform,
}


def load_data(data_path: str) -> pd.DataFrame:
    """加载清洗后的数据"""
    print("=" * 70)
    print("[数据加载]")
    print("=" * 70)
    
    # 尝试多种文件格式
    try:
        df = pd.read_excel(data_path)
        print(f"[OK] 成功加载Excel文件")
    except:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        print(f"[OK] 成功加载CSV文件")
    
    # 构造复合变量：每周工作时长
    if '每周期望工作天数' in df.columns and '每天期望工作时数' in df.columns:
        df['每周工作时长'] = df['每周期望工作天数'] * df['每天期望工作时数']
        print(f"[OK] 构造复合变量: 每周工作时长 = 每周天数 x 每天小时")
    
    # 重命名学历列（如果需要）
    if '受教育年限' in df.columns and '学历' not in df.columns:
        df['学历'] = df['受教育年限']
        print(f"[OK] 重命名: 受教育年限 -> 学历")
    
    # 提取所有变量
    data = df[ALL_VARIABLES].copy()
    
    # 数据清洗：去除缺失值
    original_size = len(data)
    data = data.dropna()
    print(f"[清洗] 原始样本: {original_size}个, 去除缺失后: {len(data)}个")
    
    # 修正连续变量中的0值：为0的值加上0.1偏移（避免对数正态分布拟合失败）
    # 离散变量保持原值（0是有意义的）
    print(f"\n[0值检测与修正]（仅针对连续变量）")
    for var in CONTINUOUS_VARIABLES:
        zero_count = (data[var] == 0).sum()
        if zero_count > 0:
            print(f"  [警告] {var}: 发现 {zero_count} 个0值样本")
            data.loc[data[var] == 0, var] = 0.1
            print(f"  [修正] {var}: 已将0值样本设为0.1（避免log(0)问题）")
    
    print(f"[OK] 加载完成: {len(data)} 个有效样本")
    print(f"[OK] 连续变量（6个）: {CONTINUOUS_VARIABLES}")
    print(f"[OK] 离散变量（2个）: {DISCRETE_VARIABLES}")
    print()
    
    return data


def fit_distribution(data: np.ndarray, dist_name: str, dist_obj) -> Dict:
    """拟合单个分布并返回统计信息"""
    result = {
        'dist_name': dist_name,
        'success': False,
        'params': None,
        'aic': np.inf,
        'bic': np.inf,
        'ks_stat': np.inf
    }
    
    try:
        # 特殊处理Beta分布（需要标准化到[0,1]）
        if dist_name == 'beta':
            data_min, data_max = data.min(), data.max()
            data_range = data_max - data_min
            if data_range == 0:
                return result
            data_scaled = (data - data_min) / data_range
            data_scaled = np.clip(data_scaled, 1e-6, 1 - 1e-6)
            params = dist_obj.fit(data_scaled, floc=0, fscale=1)
            fitted_dist = dist_obj(*params)
            
            # Kolmogorov-Smirnov检验（通用且准确）
            ks_stat, ks_pvalue = stats.kstest(data_scaled, fitted_dist.cdf)
            
        else:
            # 标准MLE估计
            params = dist_obj.fit(data)
            fitted_dist = dist_obj(*params)
            
            # Kolmogorov-Smirnov检验（通用且准确）
            ks_stat, ks_pvalue = stats.kstest(data, fitted_dist.cdf)
        
        # 计算AIC和BIC
        n = len(data)
        k = len(params)  # 参数个数
        
        # 对数似然
        if dist_name == 'beta':
            log_likelihood = np.sum(fitted_dist.logpdf(data_scaled))
        else:
            log_likelihood = np.sum(fitted_dist.logpdf(data))
        
        # AIC = 2k - 2ln(L)
        aic = 2 * k - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = k * np.log(n) - 2 * log_likelihood
        
        result['success'] = True
        result['params'] = params
        result['aic'] = aic
        result['bic'] = bic
        result['ks_stat'] = ks_stat
        result['log_likelihood'] = log_likelihood
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def fit_all_distributions(data: np.ndarray, var_name: str) -> List[Dict]:
    """对单个变量拟合所有候选分布"""
    print(f"[拟合] 变量: {var_name}")
    print(f"   样本量: {len(data)}, 范围: [{data.min():.2f}, {data.max():.2f}]")
    print()
    
    results = []
    
    for dist_name, dist_obj in CANDIDATE_DISTRIBUTIONS.items():
        result = fit_distribution(data, dist_name, dist_obj)
        
        if result['success']:
            results.append(result)
            print(f"   [OK] {dist_name:12s} | "
                  f"AIC={result['aic']:8.2f} | "
                  f"BIC={result['bic']:8.2f} | "
                  f"KS={result['ks_stat']:6.4f}")
        else:
            print(f"   [失败] {dist_name:12s} | 原因: {result.get('error', '未知')}")
    
    # 按AIC排序
    results.sort(key=lambda x: x['aic'])
    
    if results:
        best = results[0]
        print()
        print(f"   [最佳] 分布: {best['dist_name'].upper()}")
        print(f"      参数: {best['params']}")
        print(f"      AIC: {best['aic']:.2f}")
        print()
    
    return results


def fit_discrete_distribution(data: np.ndarray, var_name: str) -> Dict:
    """使用经验分布拟合离散变量"""
    print(f"[拟合离散变量] {var_name}")
    
    # 统计频率
    unique_vals, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    
    print(f"   样本量: {len(data)}")
    print(f"   唯一值: {unique_vals.tolist()}")
    print(f"   分布:")
    for val, prob, count in zip(unique_vals, probs, counts):
        print(f"      {int(val):2d}: {count:3d}个 ({prob*100:5.1f}%)")
    print()
    
    return {
        'variable': var_name,
        'type': 'discrete',
        'dist_name': 'empirical',
        'values': unique_vals,
        'probabilities': probs,
        'counts': counts,
        'n_samples': len(data)
    }


def plot_distribution_fit(data: np.ndarray, var_name: str, 
                          best_result: Dict, output_dir: Path) -> None:
    """绘制分布拟合对比图（理论分布曲线 vs 原始数据）"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图 + 理论PDF曲线
    ax1 = axes[0]
    
    # 绘制原始数据直方图
    ax1.hist(data, bins=30, density=True, alpha=0.6, 
             color='skyblue', edgecolor='black', label='原始数据')
    
    # 绘制理论分布曲线
    dist_name = best_result['dist_name']
    params = best_result['params']
    dist_obj = CANDIDATE_DISTRIBUTIONS[dist_name]
    
    if dist_name == 'beta':
        # Beta分布需要归一化
        data_min, data_max = data.min(), data.max()
        data_scaled = (data - data_min) / (data_max - data_min)
        x = np.linspace(0, 1, 200)
        y = dist_obj(*params).pdf(x)
        # 转换回原始尺度
        x_original = x * (data_max - data_min) + data_min
        y_original = y / (data_max - data_min)
        ax1.plot(x_original, y_original, 'r-', linewidth=2, 
                label=f'理论分布 ({dist_name.upper()})')
    else:
        x = np.linspace(data.min(), data.max(), 200)
        y = dist_obj(*params).pdf(x)
        ax1.plot(x, y, 'r-', linewidth=2, 
                label=f'理论分布 ({dist_name.upper()})')
    
    ax1.set_xlabel(var_name, fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12)
    ax1.set_title(f'{var_name} - 分布拟合对比（PDF）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 右图：Q-Q图（分位数-分位数图）
    ax2 = axes[1]
    
    if dist_name == 'beta':
        data_sorted = np.sort(data_scaled)
        theoretical_quantiles = dist_obj(*params).ppf(np.linspace(0.01, 0.99, len(data_sorted)))
        # 转换回原始尺度
        data_sorted_original = data_sorted * (data_max - data_min) + data_min
        theoretical_quantiles_original = theoretical_quantiles * (data_max - data_min) + data_min
    else:
        data_sorted = np.sort(data)
        theoretical_quantiles = dist_obj(*params).ppf(np.linspace(0.01, 0.99, len(data_sorted)))
        data_sorted_original = data_sorted
        theoretical_quantiles_original = theoretical_quantiles
    
    ax2.scatter(theoretical_quantiles_original, data_sorted_original, 
               alpha=0.5, s=20, color='blue')
    
    # 绘制45度参考线
    min_val = min(theoretical_quantiles_original.min(), data_sorted_original.min())
    max_val = max(theoretical_quantiles_original.max(), data_sorted_original.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合线')
    
    ax2.set_xlabel('理论分位数', fontsize=12)
    ax2.set_ylabel('样本分位数', fontsize=12)
    ax2.set_title(f'{var_name} - Q-Q图', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 添加统计信息
    info_text = f"AIC: {best_result['aic']:.2f}\nBIC: {best_result['bic']:.2f}\nKS统计量: {best_result['ks_stat']:.4f}"
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    safe_name = var_name.replace('/', '_').replace('\\', '_')
    fig_path = output_dir / f'{safe_name}_分布拟合.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[保存] 拟合对比图: {fig_path}")
    
    plt.close()


def plot_discrete_distribution(data: np.ndarray, var_name: str,
                               result: Dict, output_dir: Path) -> None:
    """绘制离散变量分布图"""
    values = result['values']
    probs = result['probabilities']
    counts = result['counts']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：条形图（频数）
    axes[0].bar(values, counts, alpha=0.7, color='steelblue', edgecolor='black', width=0.6)
    axes[0].set_xlabel(var_name, fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title(f'{var_name} - 频数分布', fontsize=14, fontweight='bold')
    axes[0].set_xticks(values)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for val, count in zip(values, counts):
        axes[0].text(val, count + max(counts)*0.02, str(count), 
                    ha='center', va='bottom', fontsize=10)
    
    # 右图：条形图（概率）
    axes[1].bar(values, probs, alpha=0.7, color='coral', edgecolor='black', width=0.6)
    axes[1].set_xlabel(var_name, fontsize=12)
    axes[1].set_ylabel('概率', fontsize=12)
    axes[1].set_title(f'{var_name} - 概率分布（经验分布）', fontsize=14, fontweight='bold')
    axes[1].set_xticks(values)
    axes[1].set_ylim(0, max(probs) * 1.15)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 添加百分比标签
    for val, prob in zip(values, probs):
        axes[1].text(val, prob + max(probs)*0.03, f'{prob*100:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    safe_name = var_name.replace('/', '_').replace('\\', '_')
    fig_path = output_dir / f'{safe_name}_分布拟合.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[保存] 分布图: {fig_path}")
    plt.close()


def analyze_correlations(data: pd.DataFrame, output_dir: Path) -> None:
    """分析并可视化变量之间的相关性"""
    print("\n" + "=" * 70)
    print("[相关性分析]")
    print("=" * 70)
    
    # 计算相关系数矩阵（Pearson、Spearman、Kendall）
    corr_pearson = data[ALL_VARIABLES].corr(method='pearson')
    corr_spearman = data[ALL_VARIABLES].corr(method='spearman')
    corr_kendall = data[ALL_VARIABLES].corr(method='kendall')
    
    # 保存相关系数矩阵
    corr_pearson.to_csv(output_dir / 'correlation_pearson.csv')
    corr_spearman.to_csv(output_dir / 'correlation_spearman.csv')
    corr_kendall.to_csv(output_dir / 'correlation_kendall.csv')
    
    print(f"[保存] Pearson相关系数矩阵 -> {output_dir / 'correlation_pearson.csv'}")
    print(f"[保存] Spearman相关系数矩阵 -> {output_dir / 'correlation_spearman.csv'}")
    print(f"[保存] Kendall相关系数矩阵 -> {output_dir / 'correlation_kendall.csv'}")
    
    # 绘制相关性热图（使用Spearman，对非线性关系更敏感）
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：核心变量之间的相关性
    ax1 = axes[0]
    core_corr = corr_spearman.loc[CORE_VARIABLES, CORE_VARIABLES]
    sns.heatmap(core_corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'label': 'Spearman相关系数'},
                vmin=-1, vmax=1)
    ax1.set_title('核心变量相关性热图', fontsize=16, fontweight='bold')
    
    # 右图：所有变量之间的相关性
    ax2 = axes[1]
    sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax2, cbar_kws={'label': 'Spearman相关系数'},
                vmin=-1, vmax=1)
    ax2.set_title('所有变量相关性热图（核心变量 + 控制变量）', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    heatmap_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"[保存] 相关性热图: {heatmap_path}")
    plt.close()
    
    # 打印关键相关性
    print("\n[关键相关性]（Spearman系数 > 0.3 或 < -0.3）:")
    for i, var1 in enumerate(ALL_VARIABLES):
        for j, var2 in enumerate(ALL_VARIABLES):
            if i < j:  # 避免重复
                corr_val = corr_spearman.loc[var1, var2]
                if abs(corr_val) > 0.3:
                    print(f"  - {var1} <-> {var2}: {corr_val:.3f}")


def save_summary(results_dict: Dict[str, List[Dict]], output_path: str) -> None:
    """保存实验总结"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("边际分布拟合实验结果（增强版）\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("核心变量（用于匹配模型）:\n")
        f.write("-" * 70 + "\n")
        for var_name in CORE_VARIABLES:
            results = results_dict.get(var_name, [])
            if results:
                best = results[0]
                f.write(f"\n变量: {var_name}\n")
                f.write(f"  推荐分布: {best['dist_name']}\n")
                f.write(f"  参数:     {best['params']}\n")
                f.write(f"  AIC:      {best['aic']:.2f}\n")
                f.write(f"  BIC:      {best['bic']:.2f}\n")
                f.write(f"  KS统计量: {best['ks_stat']:.4f}\n")
        
        f.write("\n\n控制变量（连续）:\n")
        f.write("-" * 70 + "\n")
        for var_name in CONTROL_VARIABLES_CONTINUOUS:
            results = results_dict.get(var_name, [])
            if results:
                best = results[0]
                f.write(f"\n变量: {var_name}\n")
                f.write(f"  推荐分布: {best['dist_name']}\n")
                f.write(f"  参数:     {best['params']}\n")
                f.write(f"  AIC:      {best['aic']:.2f}\n")
                f.write(f"  BIC:      {best['bic']:.2f}\n")
                f.write(f"  KS统计量: {best['ks_stat']:.4f}\n")
        
        f.write("\n\n控制变量（离散，使用经验分布）:\n")
        f.write("-" * 70 + "\n")
        for var_name in DISCRETE_VARIABLES:
            results = results_dict.get(var_name, [])
            if results:
                result = results[0]
                f.write(f"\n变量: {var_name}\n")
                f.write(f"  类型: 离散变量\n")
                f.write(f"  取值: {result['values'].tolist()}\n")
                f.write(f"  概率: {[f'{p:.3f}' for p in result['probabilities']]}\n")
                f.write(f"  频数: {result['counts'].tolist()}\n")
    
    print(f"\n[OK] 结果已保存到: {output_path}")


def main():
    """主函数"""
    print("\n" + "*" * 70)
    print(" " * 15 + "增强版边际分布拟合实验 - Simulation_project_v2")
    print("*" * 70 + "\n")
    
    # 创建输出目录
    output_dir = Path('data/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = Path('results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data_path = 'data/input/cleaned_data.csv'
    data = load_data(data_path)
    
    # 相关性分析（先做，了解变量关系）
    analyze_correlations(data, output_dir)
    
    # 拟合所有变量的边际分布
    print("\n" + "=" * 70)
    print("[边际分布拟合]")
    print("=" * 70 + "\n")
    
    results_dict = {}
    
    # 拟合连续变量
    for var_name in CONTINUOUS_VARIABLES:
        var_data = data[var_name].values
        results = fit_all_distributions(var_data, var_name)
        results_dict[var_name] = results
        
        # 绘制拟合对比图
        if results:
            plot_distribution_fit(var_data, var_name, results[0], figures_dir)
    
    # 拟合离散变量
    for var_name in DISCRETE_VARIABLES:
        var_data = data[var_name].values
        result = fit_discrete_distribution(var_data, var_name)
        results_dict[var_name] = [result]  # 保持格式一致
        
        # 绘制分布图
        plot_discrete_distribution(var_data, var_name, result, figures_dir)
    
    # 保存总结
    print("\n" + "=" * 70)
    print("[实验总结]")
    print("=" * 70 + "\n")
    
    print("连续变量（6个）:")
    for var_name in CONTINUOUS_VARIABLES:
        results = results_dict.get(var_name, [])
        if results:
            best = results[0]
            var_type = "核心" if var_name in CORE_VARIABLES else "控制"
            print(f"  [{var_type}] {var_name:12s} → {best['dist_name']:10s} "
                  f"(AIC={best['aic']:7.2f}, KS={best['ks_stat']:.4f})")
    
    print("\n离散变量（2个）- 使用经验分布:")
    for var_name in DISCRETE_VARIABLES:
        results = results_dict.get(var_name, [])
        if results:
            result = results[0]
            print(f"  [控制] {var_name:12s} → 经验分布 "
                  f"({len(result['values'])}个唯一值)")
    print()
    
    save_summary(results_dict, output_dir / 'enhanced_distribution_summary.txt')
    
    print("=" * 70)
    print(" " * 25 + "实验完成！")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - 分布拟合图: {figures_dir}/*.png")
    print(f"  - 相关性热图: {output_dir}/correlation_heatmap.png")
    print(f"  - 相关系数矩阵: {output_dir}/correlation_*.csv")
    print(f"  - 实验总结: {output_dir}/enhanced_distribution_summary.txt")


if __name__ == '__main__':
    main()

