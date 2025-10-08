#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
max_rounds精细调优
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config, perform_matching
import copy


def test_fine_tune():
    """精细测试30-45之间的max_rounds"""
    print("=" * 80)
    print("max_rounds精细调优（30-45区间）")
    print("=" * 80)
    
    config = load_config("CONFIG/logistic_config.yaml")
    market = VirtualMarket(config)
    
    # 精细测试
    max_rounds_values = [30, 32, 35, 38, 40, 42, 45]
    
    print(f"\n{'max_rounds':>12} | {'theta=0.7':>12} | {'theta=1.0':>12} | {'theta=1.3':>12}")
    print("-" * 80)
    
    for max_rounds in max_rounds_values:
        # 生成市场
        laborers, enterprises = market.generate_market(n_laborers=200, theta=1.0)
        
        # 修改配置
        test_config = copy.deepcopy(config)
        test_config['gs_matching']['max_rounds'] = max_rounds
        
        # 执行匹配
        result = perform_matching(laborers, enterprises, test_config)
        match_rate = result['matched'].sum() / len(result) * 100
        
        # 三种场景
        rates = []
        for theta in [0.7, 1.0, 1.3]:
            l, e = market.generate_market(n_laborers=200, theta=theta)
            r = perform_matching(l, e, test_config)
            rates.append(r['matched'].sum() / len(r) * 100)
        
        print(f"{max_rounds:12d} | {rates[0]:11.1f}% | {rates[1]:11.1f}% | {rates[2]:11.1f}%")


if __name__ == "__main__":
    test_fine_tune()

