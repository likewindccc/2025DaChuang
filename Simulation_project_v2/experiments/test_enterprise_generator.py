#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EnterpriseGenerator测试脚本

测试四维多元正态分布的企业生成器

作者：AI Assistant
日期：2025-10-01
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modules.population import EnterpriseGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("EnterpriseGenerator 测试脚本")
    print("=" * 80)
    
    # 测试1：使用默认配置
    test_with_default_config()
    
    # 测试2：基于劳动力数据
    test_with_labor_data()
    
    # 测试3：参数更新（校准）
    test_parameter_update()
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80 + "\n")


def test_with_default_config():
    """测试1：使用默认配置"""
    print("\n" + "=" * 80)
    print("[测试1] 使用默认配置")
    print("=" * 80)
    
    # 创建生成器
    config = {
        'seed': 43,
        'default_mean': [45.0, 75.0, 65.0, 5500.0],
        'default_std': [11.0, 15.0, 15.0, 1100.0]
    }
    
    gen = EnterpriseGenerator(config)
    gen.fit()
    
    # 生成企业
    n_enterprises = 800
    enterprises = gen.generate(n_enterprises)
    
    print(f"\n生成的虚拟企业样本（前5行）：")
    print(enterprises.head())
    
    print(f"\n统计摘要：")
    print(enterprises[['T', 'S', 'D', 'W']].describe())
    
    # 验证
    is_valid = gen.validate(enterprises)
    
    # 保存
    output_path = 'data/output/virtual_enterprises_default.csv'
    enterprises.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 企业数据已保存到: {output_path}")
    
    # 可视化
    visualize_enterprise_distribution(enterprises, "默认配置")


def test_with_labor_data():
    """测试2：基于劳动力数据"""
    print("\n" + "=" * 80)
    print("[测试2] 基于劳动力数据初始化")
    print("=" * 80)
    
    # 加载劳动力数据
    data_path = 'data/input/cleaned_data.csv'
    
    if not os.path.exists(data_path):
        print(f"警告：劳动力数据不存在: {data_path}")
        print("跳过此测试")
        return
    
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 构造复合变量
    if '每周期望工作天数' in data.columns and '每天期望工作时数' in data.columns:
        data['每周工作时长'] = data['每周期望工作天数'] * data['每天期望工作时数']
    
    # 重命名
    name_mapping = {
        '每周工作时长': 'T',
        '工作能力评分': 'S',
        '数字素养评分': 'D',
        '每月期望收入': 'W'
    }
    data = data.rename(columns=name_mapping)
    
    # 创建生成器
    config = {
        'seed': 43,
        'labor_multiplier': np.array([1.1, 1.05, 1.1, 1.2]),  # 企业需求略高
        'default_std': [12.0, 16.0, 16.0, 1200.0]
    }
    
    gen = EnterpriseGenerator(config)
    gen.fit(data)
    
    # 生成企业
    n_enterprises = 800
    enterprises = gen.generate(n_enterprises)
    
    print(f"\n统计摘要：")
    print(enterprises[['T', 'S', 'D', 'W']].describe())
    
    # 验证
    is_valid = gen.validate(enterprises)
    
    # 对比
    print("\n[劳动力 vs 企业对比]")
    labor_mean = data[['T', 'S', 'D', 'W']].mean()
    enterprise_mean = enterprises[['T', 'S', 'D', 'W']].mean()
    
    comparison = pd.DataFrame({
        '劳动力均值': labor_mean,
        '企业均值': enterprise_mean,
        '差异%': ((enterprise_mean - labor_mean) / labor_mean * 100).round(2)
    })
    print(comparison)
    
    # 保存
    output_path = 'data/output/virtual_enterprises_labor_based.csv'
    enterprises.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 企业数据已保存到: {output_path}")
    
    # 可视化对比
    visualize_labor_enterprise_comparison(data, enterprises)


def test_parameter_update():
    """测试3：参数更新（模拟校准）"""
    print("\n" + "=" * 80)
    print("[测试3] 参数更新（模拟校准）")
    print("=" * 80)
    
    # 创建生成器
    gen = EnterpriseGenerator({'seed': 43})
    gen.fit()
    
    # 初始参数
    print("\n[初始参数]")
    print(f"均值: {gen.mean}")
    print(f"标准差: {np.sqrt(np.diag(gen.covariance))}")
    
    # 生成初始企业
    enterprises_before = gen.generate(500)
    mean_before = enterprises_before[['T', 'S', 'D', 'W']].mean().values
    
    # 模拟校准：更新参数
    new_mean = np.array([50.0, 80.0, 70.0, 6000.0])
    new_std = np.array([13.0, 17.0, 17.0, 1300.0])
    new_cov = np.diag(new_std ** 2)
    
    gen.set_params(new_mean, new_cov)
    
    print("\n[更新后参数]")
    print(f"均值: {gen.mean}")
    print(f"标准差: {np.sqrt(np.diag(gen.covariance))}")
    
    # 生成更新后企业
    enterprises_after = gen.generate(500)
    mean_after = enterprises_after[['T', 'S', 'D', 'W']].mean().values
    
    # 验证
    is_valid = gen.validate(enterprises_after)
    
    # 对比
    print("\n[参数更新前后对比]")
    comparison = pd.DataFrame({
        '更新前均值': mean_before,
        '目标均值': new_mean,
        '更新后均值': mean_after,
        '偏差%': ((mean_after - new_mean) / new_mean * 100).round(2)
    }, index=['T', 'S', 'D', 'W'])
    print(comparison)


def visualize_enterprise_distribution(enterprises, title):
    """可视化企业分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    cols = ['T', 'S', 'D', 'W']
    col_names = {
        'T': '每周工作时长（小时）',
        'S': '工作能力要求评分',
        'D': '数字素养要求评分',
        'W': '每月提供工资（元）'
    }
    
    for i, col in enumerate(cols):
        ax = axes[i]
        
        # 直方图 + KDE
        ax.hist(enterprises[col], bins=30, alpha=0.6, density=True, 
                color='skyblue', edgecolor='black')
        
        # 理论正态分布曲线
        mean = enterprises[col].mean()
        std = enterprises[col].std()
        x = np.linspace(enterprises[col].min(), enterprises[col].max(), 100)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mean, std), 'r-', linewidth=2, 
                label=f'N({mean:.1f}, {std:.1f}²)')
        
        ax.set_xlabel(col_names[col])
        ax.set_ylabel('密度')
        ax.set_title(f'{col_names[col]} 分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'企业特征分布 - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f'results/figures/enterprise_distribution_{title.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 分布图已保存: {output_path}")
    plt.close()


def visualize_labor_enterprise_comparison(labor_data, enterprise_data):
    """可视化劳动力与企业对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    cols = ['T', 'S', 'D', 'W']
    col_names = {
        'T': '每周工作时长（小时）',
        'S': '工作能力/要求评分',
        'D': '数字素养/要求评分',
        'W': '期望收入/提供工资（元）'
    }
    
    for i, col in enumerate(cols):
        ax = axes[i]
        
        # 劳动力
        ax.hist(labor_data[col], bins=25, alpha=0.5, density=True, 
                color='blue', label='劳动力', edgecolor='black')
        
        # 企业
        ax.hist(enterprise_data[col], bins=25, alpha=0.5, density=True, 
                color='red', label='企业', edgecolor='black')
        
        ax.set_xlabel(col_names[col])
        ax.set_ylabel('密度')
        ax.set_title(col_names[col])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加均值线
        labor_mean = labor_data[col].mean()
        enterprise_mean = enterprise_data[col].mean()
        ax.axvline(labor_mean, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(enterprise_mean, color='red', linestyle='--', alpha=0.7)
    
    plt.suptitle('劳动力 vs 企业特征对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'results/figures/labor_enterprise_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 对比图已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()

