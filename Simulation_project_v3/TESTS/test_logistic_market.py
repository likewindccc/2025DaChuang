#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOGISTIC模块 - 虚拟市场生成测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config


def test_virtual_market():
    """测试虚拟市场生成"""
    print("=" * 70)
    print("虚拟市场生成测试")
    print("=" * 70)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 初始化市场生成器
    print("\n1. 初始化虚拟市场生成器...")
    market = VirtualMarket(config)
    print("   [完成] 分布参数加载成功")
    
    # 测试生成劳动力
    print("\n2. 生成虚拟劳动力（n=100, theta=0.8）...")
    laborers = market.generate_laborers(n_laborers=100, theta=0.8)
    print(f"   [完成] 生成 {len(laborers)} 个劳动力")
    print(f"   列名: {laborers.columns.tolist()}")
    print(f"   前5行:\n{laborers.head()}")
    
    # 测试生成企业
    print("\n3. 生成虚拟企业（n=50, theta=0.8）...")
    enterprises = market.generate_enterprises(n_enterprises=50, theta=0.8)
    print(f"   [完成] 生成 {len(enterprises)} 个企业")
    print(f"   列名: {enterprises.columns.tolist()}")
    print(f"   前5行:\n{enterprises.head()}")
    
    # 测试生成完整市场
    print("\n4. 生成完整虚拟市场（n_laborers=200, theta=1.2）...")
    laborers, enterprises = market.generate_market(
        n_laborers=200,
        theta=1.2
    )
    print(f"   [完成] 劳动力: {len(laborers)}, 企业: {len(enterprises)}")
    print(f"   市场紧张度 theta = {len(enterprises) / len(laborers):.2f}")
    
    # 数据统计
    print("\n5. 数据统计...")
    print("   劳动力统计:")
    print(f"     T 均值: {laborers['T'].mean():.2f}, 标准差: {laborers['T'].std():.2f}")
    print(f"     S 均值: {laborers['S'].mean():.2f}, 标准差: {laborers['S'].std():.2f}")
    print(f"     D 均值: {laborers['D'].mean():.2f}, 标准差: {laborers['D'].std():.2f}")
    print(f"     W 均值: {laborers['W'].mean():.2f}, 标准差: {laborers['W'].std():.2f}")
    
    print("\n   企业统计:")
    print(f"     T_req 均值: {enterprises['T_req'].mean():.2f}, 标准差: {enterprises['T_req'].std():.2f}")
    print(f"     S_req 均值: {enterprises['S_req'].mean():.2f}, 标准差: {enterprises['S_req'].std():.2f}")
    print(f"     D_req 均值: {enterprises['D_req'].mean():.2f}, 标准差: {enterprises['D_req'].std():.2f}")
    print(f"     W_offer 均值: {enterprises['W_offer'].mean():.2f}, 标准差: {enterprises['W_offer'].std():.2f}")
    
    print("\n[通过] 虚拟市场生成测试通过！")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOGISTIC模块 - 虚拟市场生成测试")
    print("=" * 70 + "\n")
    
    test_virtual_market()
    
    print("\n" + "=" * 70)
    print("[完成] 测试通过！")
    print("=" * 70)

