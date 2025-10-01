#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaborGenerator测试脚本

测试6维Copula + 离散变量条件抽样的劳动力生成器

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

from src.modules.population import LaborGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("LaborGenerator 测试脚本")
    print("=" * 80)
    
    # 1. 加载真实数据
    print("\n[步骤1] 加载真实数据...")
    data_path = 'data/input/cleaned_data.csv'
    
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return
    
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 构造复合变量：每周工作时长
    if '每周期望工作天数' in data.columns and '每天期望工作时数' in data.columns:
        data['每周工作时长'] = data['每周期望工作天数'] * data['每天期望工作时数']
        print(f"[OK] 构造复合变量: 每周工作时长 = 每周天数 x 每天小时")
    
    # 定义所需的列名（使用中文全名）
    required_cols = ['每周工作时长', '工作能力评分', '数字素养评分', '每月期望收入', 
                     '年龄', '累计工作年限', '孩子数量', '学历']
    
    # 简化变量名映射（用于显示和处理）
    name_mapping = {
        '每周工作时长': 'T',
        '工作能力评分': 'S',
        '数字素养评分': 'D',
        '每月期望收入': 'W',
        '年龄': '年龄',
        '累计工作年限': '累计工作年限',
        '孩子数量': '孩子数量',
        '学历': '学历'
    }
    
    # 检查列是否存在
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"错误：数据缺少列: {missing_cols}")
        print(f"实际列名: {data.columns.tolist()}")
        return
    
    # 重命名为简化名称
    data = data.rename(columns=name_mapping)
    required_cols = list(name_mapping.values())
    
    print(f"[OK] 数据加载成功，样本量: {len(data)}")
    print(f"列名: {data.columns.tolist()}")
    
    # 2. 创建并拟合生成器
    print("\n[步骤2] 创建并拟合LaborGenerator...")
    
    config = {
        'seed': 42,
        'use_copula': 'gaussian',
        'correlation_method': 'spearman'
    }
    
    gen = LaborGenerator(config)
    gen.fit(data)
    
    # 3. 生成虚拟劳动力
    print("\n[步骤3] 生成虚拟劳动力...")
    n_agents = 1000
    virtual_laborers = gen.generate(n_agents)
    
    print(f"\n生成的虚拟劳动力样本（前5行）：")
    print(virtual_laborers.head())
    
    print(f"\n统计摘要：")
    print(virtual_laborers[required_cols].describe())
    
    # 4. 验证生成质量
    print("\n[步骤4] 验证生成质量...")
    is_valid = gen.validate(virtual_laborers)
    
    # 5. 对比分析
    print("\n[步骤5] 真实数据 vs 生成数据对比...")
    
    comparison = pd.DataFrame({
        '真实均值': data[required_cols].mean(),
        '生成均值': virtual_laborers[required_cols].mean(),
        '真实标准差': data[required_cols].std(),
        '生成标准差': virtual_laborers[required_cols].std()
    })
    
    comparison['均值偏差%'] = (
        (comparison['生成均值'] - comparison['真实均值']) / 
        comparison['真实均值'] * 100
    ).round(2)
    
    print("\n对比统计：")
    print(comparison)
    
    # 6. 可视化
    print("\n[步骤6] 生成可视化图表...")
    visualize_comparison(data, virtual_laborers, required_cols)
    
    # 7. 保存结果
    output_path = 'data/output/virtual_laborers_test.csv'
    virtual_laborers.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 虚拟劳动力数据已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80 + "\n")


def visualize_comparison(real_data, virtual_data, cols):
    """可视化对比图"""
    
    # 创建对比图（连续变量）
    continuous_cols = ['T', 'S', 'D', 'W', '年龄', '累计工作年限']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        
        # 绘制直方图
        ax.hist(real_data[col], bins=30, alpha=0.5, label='真实数据', 
                density=True, color='blue')
        ax.hist(virtual_data[col], bins=30, alpha=0.5, label='生成数据', 
                density=True, color='red')
        
        ax.set_xlabel(col)
        ax.set_ylabel('密度')
        ax.set_title(f'{col} 分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/labor_generator_continuous_comparison.png', 
                dpi=150, bbox_inches='tight')
    print(f"[OK] 连续变量对比图已保存")
    
    # 离散变量对比
    discrete_cols = ['孩子数量', '学历']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, col in enumerate(discrete_cols):
        ax = axes[i]
        
        # 计算频率
        real_freq = real_data[col].value_counts(normalize=True).sort_index()
        virtual_freq = virtual_data[col].value_counts(normalize=True).sort_index()
        
        # 绘制条形图
        x = np.arange(len(real_freq))
        width = 0.35
        
        ax.bar(x - width/2, real_freq.values, width, label='真实数据', 
               alpha=0.7, color='blue')
        ax.bar(x + width/2, virtual_freq.values, width, label='生成数据', 
               alpha=0.7, color='red')
        
        ax.set_xlabel(col)
        ax.set_ylabel('频率')
        ax.set_title(f'{col} 分布对比')
        ax.set_xticks(x)
        ax.set_xticklabels(real_freq.index)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/labor_generator_discrete_comparison.png', 
                dpi=150, bbox_inches='tight')
    print(f"[OK] 离散变量对比图已保存")
    
    plt.close('all')


if __name__ == '__main__':
    main()

