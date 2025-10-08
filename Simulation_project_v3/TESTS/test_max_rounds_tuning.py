#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
max_rounds参数调优测试

测试不同max_rounds对匹配率的影响，找到使匹配率接近50%的最优值。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config, perform_matching
import copy


def test_max_rounds_impact():
    """测试不同max_rounds对匹配率的影响"""
    print("=" * 80)
    print("max_rounds参数调优测试")
    print("=" * 80)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    market = VirtualMarket(config)
    
    # 测试不同的max_rounds值
    max_rounds_values = [5, 10, 15, 20, 25, 30, 40, 50]
    
    # 三种市场场景
    scenarios = [
        (0.7, "岗位紧张型（theta=0.7）"),
        (1.0, "均衡市场（theta=1.0）"),
        (1.3, "岗位富余型（theta=1.3）")
    ]
    
    print(f"\n{'max_rounds':>12} | {'theta=0.7':>12} | {'theta=1.0':>12} | {'theta=1.3':>12}")
    print("-" * 80)
    
    for max_rounds in max_rounds_values:
        match_rates = []
        
        for theta, _ in scenarios:
            # 生成市场
            laborers, enterprises = market.generate_market(
                n_laborers=200,
                theta=theta
            )
            
            # 修改配置中的max_rounds
            test_config = copy.deepcopy(config)
            test_config['gs_matching']['max_rounds'] = max_rounds
            
            # 执行匹配
            result = perform_matching(laborers, enterprises, test_config)
            match_rate = result['matched'].sum() / len(result) * 100
            match_rates.append(match_rate)
        
        print(f"{max_rounds:12d} | {match_rates[0]:11.1f}% | {match_rates[1]:11.1f}% | {match_rates[2]:11.1f}%")
    
    print("\n" + "=" * 80)
    print("建议：选择使均衡市场（theta=1.0）匹配率接近50%的max_rounds值")
    print("=" * 80)


if __name__ == "__main__":
    test_max_rounds_impact()

