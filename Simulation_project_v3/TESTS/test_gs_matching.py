#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOGISTIC模块 - GS匹配算法测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config, perform_matching


def test_gs_matching():
    """测试GS匹配算法"""
    print("=" * 70)
    print("GS匹配算法测试")
    print("=" * 70)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 初始化市场生成器
    print("\n1. 初始化虚拟市场生成器...")
    market = VirtualMarket(config)
    print("   [完成] 分布参数加载成功")
    
    # 生成虚拟市场
    print("\n2. 生成虚拟市场（n=100, theta=0.8）...")
    laborers, enterprises = market.generate_market(
        n_laborers=100,
        theta=0.8
    )
    print(f"   [完成] 劳动力: {len(laborers)}, 企业: {len(enterprises)}")
    
    # 执行GS匹配
    print("\n3. 执行GS匹配算法...")
    match_result = perform_matching(laborers, enterprises, config)
    print(f"   [完成] 匹配完成")
    
    # 统计匹配结果
    print("\n4. 匹配结果统计...")
    n_matched = match_result['matched'].sum()
    match_rate = n_matched / len(match_result)
    print(f"   匹配成功数: {n_matched}")
    print(f"   匹配率: {match_rate:.2%}")
    print(f"   未匹配数: {len(match_result) - n_matched}")
    
    # 查看匹配结果示例
    print("\n5. 匹配成功的劳动力示例（前5个）:")
    matched_laborers = match_result[match_result['matched'] == 1].head()
    print(matched_laborers[['id', 'T', 'S', 'D', 'W', 'enterprise_id', 'matched']])
    
    print("\n6. 未匹配的劳动力示例（前5个）:")
    unmatched_laborers = match_result[match_result['matched'] == 0].head()
    if len(unmatched_laborers) > 0:
        print(unmatched_laborers[['id', 'T', 'S', 'D', 'W', 'enterprise_id', 'matched']])
    else:
        print("   所有劳动力都已匹配")
    
    print("\n[通过] GS匹配算法测试通过！")


def test_different_scenarios():
    """测试不同市场场景下的匹配"""
    print("\n" + "=" * 70)
    print("不同市场场景测试")
    print("=" * 70)
    
    config = load_config("CONFIG/logistic_config.yaml")
    market = VirtualMarket(config)
    
    scenarios = [
        (0.7, "岗位紧张型市场（theta=0.7）"),
        (1.0, "均衡市场（theta=1.0）"),
        (1.3, "岗位富余型市场（theta=1.3）")
    ]
    
    for theta, desc in scenarios:
        print(f"\n场景: {desc}")
        
        # 生成市场
        laborers, enterprises = market.generate_market(
            n_laborers=200,
            theta=theta
        )
        
        # 执行匹配
        match_result = perform_matching(laborers, enterprises, config)
        
        # 统计结果
        n_matched = match_result['matched'].sum()
        match_rate = n_matched / len(match_result)
        
        print(f"   劳动力数: {len(laborers)}")
        print(f"   企业数: {len(enterprises)}")
        print(f"   匹配成功: {n_matched}")
        print(f"   匹配率: {match_rate:.2%}")
    
    print("\n[通过] 不同场景测试通过！")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOGISTIC模块 - GS匹配算法完整测试")
    print("=" * 70 + "\n")
    
    test_gs_matching()
    test_different_scenarios()
    
    print("\n" + "=" * 70)
    print("[完成] 所有测试通过！")
    print("=" * 70)

