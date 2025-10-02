"""
验证偏好参数更新

检查：
1. 配置文件中的参数是否正确
2. MatchingEngine是否正确读取
3. 简单测试验证参数效果
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.matching.matching_engine import MatchingEngine
from src.modules.matching.abm_data_generator import ABMDataGenerator
from src.modules.population.labor_generator import LaborGenerator
from src.modules.population.enterprise_generator import EnterpriseGenerator


def verify_config():
    """验证配置参数"""
    print("=" * 80)
    print("验证偏好参数配置")
    print("=" * 80)
    
    # 1. 检查配置文件
    print("\n1. 配置文件参数:")
    print("-" * 80)
    config_path = project_root / "config" / "default" / "matching.yaml"
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    labor_pref = config['preference']['labor']
    enterprise_pref = config['preference']['enterprise']
    
    print("劳动力偏好参数:")
    for k, v in labor_pref.items():
        print(f"  {k}: {v}")
    
    print("\n企业偏好参数:")
    for k, v in enterprise_pref.items():
        print(f"  {k}: {v}")
    
    # 2. 检查MatchingEngine读取
    print("\n2. MatchingEngine读取的参数:")
    print("-" * 80)
    engine = MatchingEngine()
    
    print("劳动力偏好参数:")
    for k, v in engine.labor_pref_params.items():
        print(f"  {k}: {v}")
    
    print("\n企业偏好参数:")
    for k, v in engine.enterprise_pref_params.items():
        print(f"  {k}: {v}")
    
    # 3. 验证目标参数（快速测试中表现最好的配置1）
    print("\n3. 参数验证:")
    print("-" * 80)
    
    target_params = {
        'gamma_1': 0.01,
        'gamma_2': 0.02,
        'gamma_3': 0.05,
        'gamma_4': 0.0001,
        'beta_1': 0.01,
        'beta_2': 0.02,
        'beta_3': 0.05,
        'beta_4': -0.0001
    }
    
    all_correct = True
    for param_name, target_value in target_params.items():
        if param_name.startswith('gamma'):
            actual_value = engine.labor_pref_params[param_name]
        else:
            actual_value = engine.enterprise_pref_params[param_name]
        
        is_correct = abs(actual_value - target_value) < 1e-10
        status = "[OK]" if is_correct else "[FAIL]"
        
        print(f"  {status} {param_name}: 目标={target_value}, 实际={actual_value}")
        
        if not is_correct:
            all_correct = False
    
    if all_correct:
        print("\n" + "=" * 80)
        print("[成功] 所有参数配置正确！")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[失败] 部分参数配置不正确，请检查！")
        print("=" * 80)
    
    return all_correct


def quick_test():
    """快速测试新参数效果"""
    print("\n4. 快速匹配测试（使用新参数）:")
    print("-" * 80)
    
    # 加载真实数据
    real_data = pd.read_csv('data/input/cleaned_data.csv')
    real_data['T'] = real_data['每周期望工作天数'] * real_data['每天期望工作时数']
    real_data['S'] = real_data['工作能力评分']
    real_data['D'] = real_data['数字素养评分']
    real_data['W'] = real_data['每月期望收入']
    
    # 初始化生成器
    labor_gen = LaborGenerator()
    enterprise_gen = EnterpriseGenerator()
    
    labor_gen.fit(real_data)
    enterprise_gen.fit(real_data)
    
    # 创建匹配引擎（会自动使用新参数）
    engine = MatchingEngine()
    
    # 生成测试数据
    labor_df = labor_gen.generate(n_agents=100)
    enterprise_df = enterprise_gen.generate(n_agents=100)
    
    # 执行匹配
    result = engine.match(labor_df, enterprise_df)
    
    match_rate = result.statistics['match_rate']
    print(f"\n匹配率: {match_rate:.2%}")
    print(f"匹配数: {result.statistics['n_matched']} / {result.statistics['n_labor']}")
    print(f"是否稳定: {result.is_stable}")
    
    print("\n[提示] 预期匹配率约在 30-40% 之间（theta=1.0, 单轮模拟）")
    
    return match_rate


if __name__ == "__main__":
    # 验证配置
    config_ok = verify_config()
    
    if config_ok:
        # 快速测试
        try:
            match_rate = quick_test()
            print("\n" + "=" * 80)
            print("[完成] 参数验证和测试完成！")
            print("=" * 80)
        except Exception as e:
            print(f"\n[错误] 测试失败: {e}")
            import traceback
            traceback.print_exc()

