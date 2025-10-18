#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试SIMULATOR模块

测试批量场景运行功能，包括：
1. 场景运行
2. 结果汇总
3. 政策效果分析
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.SIMULATOR import MarketSimulator


def test_simulator():
    """测试批量场景运行"""
    
    print("="*80)
    print("测试SIMULATOR模块")
    print("="*80)
    
    # 1. 初始化模拟器
    print("\n步骤1: 初始化市场模拟器")
    simulator = MarketSimulator('CONFIG/simulator_config.yaml')
    print("  模拟器初始化成功")
    print(f"  场景数量: {len(simulator.config['scenarios'])}")
    
    # 2. 批量运行所有场景
    print("\n步骤2: 批量运行所有场景")
    results_df = simulator.run_batch()
    
    # 3. 验证结果
    print("\n步骤3: 验证结果")
    assert len(results_df) >= 3, "场景数量应至少为3个"
    assert 'baseline' in results_df['scenario_name'].values, "应包含基准场景"
    print("  结果验证通过")
    
    # 4. 显示汇总结果
    print("\n步骤4: 场景对比汇总")
    print("\n" + "="*80)
    print("场景对比表")
    print("="*80)
    print(results_df[['scenario_display_name', 'unemployment_rate', 
                      'mean_S', 'mean_D', 'mean_wage_employed', 
                      'mean_effort']].to_string(index=False))
    
    # 5. 政策效果分析
    print("\n步骤5: 政策效果分析")
    from MODULES.SIMULATOR.policy_analyzer import (
        analyze_policy_effects,
        generate_comparison_report
    )
    
    effects = analyze_policy_effects(results_df)
    generate_comparison_report(results_df)
    
    print("\n" + "="*80)
    print("测试通过！")
    print("="*80)


if __name__ == '__main__':
    test_simulator()

