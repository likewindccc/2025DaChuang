#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匹配函数回归模块测试

测试Logit回归拟合匹配函数λ(x,σ,θ)的完整流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import MatchFunction, load_config


def test_match_function():
    """测试匹配函数回归"""
    print("=" * 80)
    print("匹配函数Logit回归测试")
    print("=" * 80)
    
    # 加载配置
    print("\n1. 加载配置...")
    config = load_config("CONFIG/logistic_config.yaml")
    print(f"   数据生成轮数: {config['data_generation']['n_rounds']}")
    print(f"   每轮劳动力数: {config['market_size']['n_laborers']}")
    
    # 初始化匹配函数
    print("\n2. 初始化匹配函数...")
    match_func = MatchFunction(config)
    print("   [完成]")
    
    # 生成训练数据
    print("\n3. 生成训练数据...")
    data = match_func.generate_training_data()
    print(f"   总样本数: {len(data)}")
    print(f"   匹配成功数: {data['matched'].sum()}")
    print(f"   匹配率: {data['matched'].mean() * 100:.2f}%")
    print(f"\n   数据前5行:")
    print(data.head())
    
    # 拟合Logit回归
    print("\n4. 拟合Logit回归...")
    match_func.fit()
    print("   [完成]")
    
    # 显示回归结果
    print("\n5. 回归结果摘要:")
    print(f"   伪R²: {match_func.model.prsquared:.4f}")
    print(f"   AIC: {match_func.model.aic:.2f}")
    print(f"   BIC: {match_func.model.bic:.2f}")
    
    print("\n   回归参数:")
    print(match_func.params)
    
    # 保存结果
    print("\n6. 保存回归结果...")
    match_func.save_results()
    print("   保存路径: OUTPUT/logistic/match_function_params.pkl")
    print("   保存路径: OUTPUT/logistic/match_function_model.pkl")
    
    # 测试预测
    print("\n7. 测试预测功能...")
    test_X = data.iloc[:5][['T', 'S', 'D', 'W', 'sigma', 'theta']]
    pred_prob = match_func.predict(test_X)
    print(f"   前5个样本的预测匹配概率:")
    for i, prob in enumerate(pred_prob):
        actual = data.iloc[i]['matched']
        print(f"     样本{i}: 预测概率={prob:.4f}, 实际结果={actual}")
    
    print("\n" + "=" * 80)
    print("[完成] 匹配函数回归测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_match_function()

