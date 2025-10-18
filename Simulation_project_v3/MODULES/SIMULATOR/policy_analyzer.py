#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
政策分析工具

提供政策效果分析和对比报告生成功能。

核心功能：
1. 分析各场景与基准场景的差异
2. 生成对比分析报告（直接输出到终端）
3. 可视化结果对比（可选，初版未实现）
"""

import pandas as pd
import numpy as np
from typing import Dict


def analyze_policy_effects(comparison_df: pd.DataFrame) -> Dict:
    """
    分析政策效果
    
    对比各政策场景与基准场景的差异，包括：
    - 失业率变化（百分点）
    - 平均工资变化（元）
    - 技能水平提升（%）
    - 数字素养提升（%）
    - 努力水平变化
    
    参数:
        comparison_df: 场景对比汇总表
    
    返回:
        政策效果分析字典
    """
    # 获取基准场景数据
    baseline = comparison_df[comparison_df['scenario_name'] == 'baseline'].iloc[0]
    
    effects = {}
    
    # 分析每个非基准场景
    for _, row in comparison_df.iterrows():
        if row['scenario_name'] == 'baseline':
            continue
        
        scenario_name = row['scenario_name']
        
        # 计算各项指标变化
        effects[scenario_name] = {
            # 失业率变化（百分点）
            'unemployment_rate_change': 
                (row['unemployment_rate'] - baseline['unemployment_rate']) * 100,
            
            # 失业率变化率（%）
            'unemployment_rate_change_pct': 
                (row['unemployment_rate'] - baseline['unemployment_rate']) / 
                baseline['unemployment_rate'] * 100,
            
            # 平均工资变化（元）
            'mean_wage_change': 
                row['mean_wage_employed'] - baseline['mean_wage_employed'],
            
            # 平均工资变化率（%）
            'mean_wage_change_pct': 
                (row['mean_wage_employed'] - baseline['mean_wage_employed']) / 
                baseline['mean_wage_employed'] * 100,
            
            # 技能水平变化率（%）
            'mean_S_change_pct': 
                (row['mean_S'] - baseline['mean_S']) / 
                baseline['mean_S'] * 100,
            
            # 数字素养变化率（%）
            'mean_D_change_pct': 
                (row['mean_D'] - baseline['mean_D']) / 
                baseline['mean_D'] * 100,
            
            # 努力水平变化
            'mean_effort_change': 
                row['mean_effort'] - baseline['mean_effort'],
            
            # 努力水平变化率（%）
            'mean_effort_change_pct': 
                (row['mean_effort'] - baseline['mean_effort']) / 
                baseline['mean_effort'] * 100,
        }
    
    return effects


def generate_comparison_report(comparison_df: pd.DataFrame) -> None:
    """
    生成对比分析报告
    
    直接输出到终端，不生成文件。
    
    报告内容：
    1. 场景对比表格
    2. 政策效果分析
    3. 关键指标变化百分比
    
    参数:
        comparison_df: 场景对比汇总表
    """
    print("\n" + "="*80)
    print("政策效果分析报告")
    print("="*80)
    
    # 1. 基准场景信息
    baseline = comparison_df[comparison_df['scenario_name'] == 'baseline'].iloc[0]
    print(f"\n【基准场景】")
    print(f"  失业率: {baseline['unemployment_rate']*100:.2f}%")
    print(f"  平均技能水平S: {baseline['mean_S']:.2f}")
    print(f"  平均数字素养D: {baseline['mean_D']:.2f}")
    print(f"  平均工资(就业者): {baseline['mean_wage_employed']:.2f} 元")
    print(f"  平均努力水平: {baseline['mean_effort']:.4f}")
    
    # 2. 政策效果分析
    effects = analyze_policy_effects(comparison_df)
    
    print(f"\n{'='*80}")
    print("政策场景对比分析")
    print("="*80)
    
    for scenario_name, effect in effects.items():
        # 获取场景显示名称
        scenario_row = comparison_df[
            comparison_df['scenario_name'] == scenario_name
        ].iloc[0]
        display_name = scenario_row['scenario_display_name']
        
        print(f"\n【{display_name}】")
        print(f"  失业率变化:")
        print(f"    绝对变化: {effect['unemployment_rate_change']:+.2f} 百分点")
        print(f"    相对变化: {effect['unemployment_rate_change_pct']:+.1f}%")
        
        print(f"  平均工资变化:")
        print(f"    绝对变化: {effect['mean_wage_change']:+.2f} 元")
        print(f"    相对变化: {effect['mean_wage_change_pct']:+.1f}%")
        
        print(f"  技能水平提升: {effect['mean_S_change_pct']:+.1f}%")
        print(f"  数字素养提升: {effect['mean_D_change_pct']:+.1f}%")
        
        print(f"  努力水平变化:")
        print(f"    绝对变化: {effect['mean_effort_change']:+.4f}")
        print(f"    相对变化: {effect['mean_effort_change_pct']:+.1f}%")
    
    # 3. 政策效果排名
    print(f"\n{'='*80}")
    print("政策效果排名")
    print("="*80)
    
    # 按失业率降低幅度排名
    ranked = sorted(
        effects.items(), 
        key=lambda x: x[1]['unemployment_rate_change']
    )
    
    print(f"\n【按失业率降低效果排名】")
    for i, (scenario_name, effect) in enumerate(ranked, 1):
        scenario_row = comparison_df[
            comparison_df['scenario_name'] == scenario_name
        ].iloc[0]
        display_name = scenario_row['scenario_display_name']
        print(f"  {i}. {display_name}: "
              f"失业率降低 {-effect['unemployment_rate_change']:.2f} 百分点 "
              f"({effect['unemployment_rate_change_pct']:+.1f}%)")
    
    # 按工资提升幅度排名
    ranked_wage = sorted(
        effects.items(), 
        key=lambda x: x[1]['mean_wage_change'],
        reverse=True
    )
    
    print(f"\n【按工资提升效果排名】")
    for i, (scenario_name, effect) in enumerate(ranked_wage, 1):
        scenario_row = comparison_df[
            comparison_df['scenario_name'] == scenario_name
        ].iloc[0]
        display_name = scenario_row['scenario_display_name']
        print(f"  {i}. {display_name}: "
              f"工资提升 {effect['mean_wage_change']:.2f} 元 "
              f"({effect['mean_wage_change_pct']:+.1f}%)")
    
    print(f"\n{'='*80}")
    print("报告生成完成")
    print("="*80)


def visualize_results(comparison_df: pd.DataFrame) -> None:
    """
    可视化结果对比
    
    可选功能，初版暂不实现。
    
    计划生成的图表：
    - 失业率对比柱状图
    - 平均工资对比柱状图
    - 技能水平对比柱状图
    - 数字素养对比柱状图
    
    参数:
        comparison_df: 场景对比汇总表
    """
    raise NotImplementedError("可视化功能将在后续版本中实现")

