#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
边际分布拟合实验
为Core模块开发选择最佳的边际分布类型

运行方式：
1. 激活虚拟环境: D:\Python\2025DaChuang\venv\Scripts\Activate.ps1
2. 进入项目目录: cd D:\Python\2025DaChuang\Simulation_project_v2
3. 运行: python experiments/marginal_distribution_experiment.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 核心状态变量
CORE_VARIABLES = [
    '每周工作时长',    # T
    '工作能力评分',    # S
    '数字素养评分',    # D
    '每月期望收入'     # W
]

# 候选分布
CANDIDATE_DISTRIBUTIONS = {
    'beta': stats.beta,
    'gamma': stats.gamma,
    'lognorm': stats.lognorm,
    'norm': stats.norm,
    'weibull_min': stats.weibull_min
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
    
    # 提取核心变量
    data = df[CORE_VARIABLES].copy()
    
    # 数据清洗：去除缺失值和异常值
    data = data.dropna()
    
    # 修正数字素养评分：为0的值加上0.1偏移（避免对数正态分布拟合失败）
    zero_count = (data['数字素养评分'] == 0).sum()
    if zero_count > 0:
        print(f"[警告] 发现 {zero_count} 个数字素养评分为0的样本")
        data.loc[data['数字素养评分'] == 0, '数字素养评分'] = 0.1
        print(f"[修正] 已将这些样本的数字素养评分设为0.1（避免log(0)问题）")
    
    print(f"[OK] 加载完成: {len(data)} 个有效样本")
    print(f"[OK] 核心变量: {CORE_VARIABLES}")
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
            print(f"   [OK] {dist_name:12s} | AIC={result['aic']:8.2f} | "
                  f"BIC={result['bic']:8.2f} | KS={result['ks_stat']:6.4f}")
            results.append(result)
        else:
            print(f"   [FAIL] {dist_name:12s} | 拟合失败")
    
    print()
    return results


def select_best_distribution(results: List[Dict]) -> Dict:
    """根据AIC选择最佳分布"""
    if not results:
        return None
    
    # 按AIC排序（越小越好）
    sorted_results = sorted(results, key=lambda x: x['aic'])
    best = sorted_results[0]
    
    return best


def main():
    """主函数"""
    print()
    print("*" * 70)
    print(" 边际分布拟合实验 - Simulation_project_v2".center(70))
    print("*" * 70)
    print()
    
    # 1. 加载数据
    data_path = "data/input/cleaned_data.csv"
    df = load_data(data_path)
    
    # 2. 对每个核心变量进行分布拟合
    best_distributions = {}
    
    for var in CORE_VARIABLES:
        var_data = df[var].values
        results = fit_all_distributions(var_data, var)
        
        best = select_best_distribution(results)
        if best:
            print(f"   [最佳] 分布: {best['dist_name'].upper()}")
            print(f"      参数: {best['params']}")
            print(f"      AIC: {best['aic']:.2f}")
            print()
            best_distributions[var] = best
    
    # 3. 生成总结报告
    print("=" * 70)
    print("[实验总结]")
    print("=" * 70)
    print()
    
    for var, result in best_distributions.items():
        print(f"变量: {var}")
        print(f"  推荐分布: {result['dist_name']}")
        print(f"  参数:     {result['params']}")
        print(f"  AIC:      {result['aic']:.2f}")
        print(f"  BIC:      {result['bic']:.2f}")
        print(f"  KS统计量: {result['ks_stat']:.4f}")
        print()
    
    # 4. 保存结果到文件
    output_path = "data/output/best_distributions.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("边际分布拟合实验结果\n")
        f.write("=" * 70 + "\n\n")
        
        for var, result in best_distributions.items():
            f.write(f"变量: {var}\n")
            f.write(f"  分布类型: {result['dist_name']}\n")
            f.write(f"  参数: {result['params']}\n")
            f.write(f"  AIC: {result['aic']:.2f}\n")
            f.write(f"  BIC: {result['bic']:.2f}\n")
            f.write(f"  KS统计量: {result['ks_stat']:.4f}\n")
            f.write("\n")
    
    print(f"[OK] 结果已保存到: {output_path}")
    print()
    print("=" * 70)
    print(" 实验完成！".center(70))
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
