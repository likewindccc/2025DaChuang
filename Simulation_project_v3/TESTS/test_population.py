#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POPULATION模块测试

测试劳动力分布功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.POPULATION import LaborDistribution, load_config


def test_labor_distribution():
    """测试劳动力分布"""
    print("=" * 70)
    print("测试劳动力分布")
    print("=" * 70)
    
    # 加载配置
    config = load_config("CONFIG/population_config.yaml")
    
    # 初始化并拟合
    print("\n1. 拟合分布...")
    labor_dist = LaborDistribution(config)
    labor_dist.fit()
    print("   [完成] 拟合完成")
    
    # 保存参数
    print("\n2. 保存参数...")
    labor_dist.save_params()
    print("   [完成] 参数已保存到: OUTPUT/population/labor_distribution_params.pkl")
    
    print("\n[通过] 劳动力分布测试通过！\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("POPULATION模块测试")
    print("=" * 70 + "\n")
    
    test_labor_distribution()
    
    print("=" * 70)
    print("[完成] 测试通过！")
    print("=" * 70)
