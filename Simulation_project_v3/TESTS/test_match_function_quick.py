#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匹配函数快速测试（小样本）

用于验证修改后的代码是否正常工作，使用较小的样本量以加快测试速度。
"""

import sys
from pathlib import Path
import copy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import MatchFunction, load_config


def test_quick():
    """快速测试（小样本）"""
    print("=" * 80)
    print("匹配函数快速测试（10轮，1000劳动力/轮）")
    print("=" * 80)
    
    # 加载并修改配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 减小样本量以加快测试
    config['data_generation']['n_rounds'] = 10
    config['market_size']['n_laborers'] = 1000
    
    print(f"\n配置: {config['data_generation']['n_rounds']}轮 × {config['market_size']['n_laborers']}劳动力")
    
    # 初始化
    match_func = MatchFunction(config)
    
    # 生成数据
    print("\n生成训练数据...")
    data = match_func.generate_training_data()
    print(f"总样本数: {len(data)}")
    print(f"匹配率: {data['matched'].mean() * 100:.2f}%")
    
    # 检查sigma
    print(f"\nsigma统计:")
    print(f"  最小值: {data['sigma'].min():.4f}")
    print(f"  最大值: {data['sigma'].max():.4f}")
    print(f"  均值: {data['sigma'].mean():.4f}")
    
    # 拟合回归
    print("\n拟合Logit回归...")
    match_func.fit()
    print(f"伪R²: {match_func.model.prsquared:.4f}")
    
    # 显示参数
    print("\n回归参数:")
    print(match_func.params)
    
    # 测试预测
    print("\n测试预测（前3个样本）:")
    test_X = data.iloc[:3][['T', 'S', 'D', 'W', 'sigma', 'theta']]
    pred_prob = match_func.predict(test_X)
    for i, prob in enumerate(pred_prob):
        print(f"  样本{i}: 预测={prob:.4f}, 实际={data.iloc[i]['matched']}")
    
    print("\n" + "=" * 80)
    print("[完成] 快速测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_quick()

