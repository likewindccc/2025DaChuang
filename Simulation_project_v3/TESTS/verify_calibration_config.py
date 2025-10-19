import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from MODULES.CALIBRATION import OptimizationUtils


def verify_calibration_config():
    """
    验证校准配置文件的正确性
    
    检查项：
    1. 参数范围是否包含当前最优值
    2. alpha_T参数是否正确添加
    3. 配置路径是否正确
    4. 能否正确更新MFG配置
    """
    print("\n" + "="*80)
    print("CALIBRATION配置验证")
    print("="*80)
    
    # 加载校准配置
    calibration_config_path = "CONFIG/calibration_config.yaml"
    with open(calibration_config_path, 'r', encoding='utf-8') as f:
        calib_config = yaml.safe_load(f)
    
    # 加载MFG配置
    mfg_config_path = "CONFIG/mfg_config.yaml"
    with open(mfg_config_path, 'r', encoding='utf-8') as f:
        mfg_config = yaml.safe_load(f)
    
    print("\n【检查1：参数范围是否包含当前最优值】")
    print("-"*80)
    
    # 当前最优值
    current_optimal = {
        'rho': 0.40,
        'kappa': 2000.0,
        'alpha_T': 0.30,
        'gamma_T': 0.30,
        'gamma_S': 0.45,
        'gamma_D': 0.45,
        'gamma_W': 0.15
    }
    
    all_in_bounds = True
    for param in calib_config['parameters']:
        name = param['name']
        bounds = param['bounds']
        
        if name in current_optimal:
            optimal_value = current_optimal[name]
            in_bounds = bounds[0] <= optimal_value <= bounds[1]
            
            status = "✓" if in_bounds else "✗"
            print(f"{status} {name:10s}: [{bounds[0]:8.2f}, {bounds[1]:8.2f}]  "
                  f"当前值={optimal_value:8.2f}")
            
            if not in_bounds:
                all_in_bounds = False
                print(f"  ⚠️ 警告：当前最优值不在范围内！")
    
    if all_in_bounds:
        print("\n结果：所有参数范围正确 ✓")
    else:
        print("\n结果：存在参数范围问题 ✗")
    
    print("\n【检查2：alpha_T参数是否正确添加】")
    print("-"*80)
    
    param_names = [p['name'] for p in calib_config['parameters']]
    has_alpha_T = 'alpha_T' in param_names
    
    if has_alpha_T:
        alpha_T_param = next(p for p in calib_config['parameters'] 
                            if p['name'] == 'alpha_T')
        print(f"✓ alpha_T参数已添加")
        print(f"  配置路径: {alpha_T_param['config_path']}")
        print(f"  参数范围: {alpha_T_param['bounds']}")
        print(f"  初始值: {alpha_T_param['initial_value']}")
    else:
        print(f"✗ alpha_T参数未找到")
    
    print("\n【检查3：配置路径是否与MFG配置匹配】")
    print("-"*80)
    
    all_paths_valid = True
    for param in calib_config['parameters']:
        name = param['name']
        config_path = param['config_path']
        keys = config_path.split('.')
        
        # 导航到MFG配置中的目标位置
        try:
            target = mfg_config
            for key in keys[:-1]:
                target = target[key]
            
            # 检查最后一个键是否存在
            if keys[-1] in target:
                print(f"✓ {name:10s}: {config_path:40s} → {target[keys[-1]]}")
            else:
                print(f"✗ {name:10s}: {config_path:40s} → 键不存在！")
                all_paths_valid = False
        except (KeyError, TypeError) as e:
            print(f"✗ {name:10s}: {config_path:40s} → 路径错误: {e}")
            all_paths_valid = False
    
    if all_paths_valid:
        print("\n结果：所有配置路径正确 ✓")
    else:
        print("\n结果：存在配置路径问题 ✗")
    
    print("\n【检查4：OptimizationUtils能否正常加载】")
    print("-"*80)
    
    try:
        param_utils = OptimizationUtils(calib_config)
        
        print(f"✓ OptimizationUtils初始化成功")
        print(f"  参数数量: {param_utils.n_params}")
        print(f"  参数名称: {param_utils.get_param_names()}")
        
        # 测试获取初始值
        initial_values = param_utils.get_initial_values('baseline')
        print(f"\n初始值向量:")
        for name, value in zip(param_utils.get_param_names(), initial_values):
            print(f"  {name:10s}: {value:8.2f}")
        
        # 测试向量到字典的转换
        params_dict = param_utils.vector_to_dict(initial_values)
        print(f"\n参数字典:")
        for name, value in params_dict.items():
            print(f"  {name:10s}: {value:8.2f}")
        
        print("\n结果：参数工具类正常工作 ✓")
    except Exception as e:
        print(f"✗ OptimizationUtils初始化失败: {e}")
        print("\n结果：参数工具类存在问题 ✗")
    
    print("\n【检查5：测试参数更新MFG配置】")
    print("-"*80)
    
    try:
        from MODULES.CALIBRATION.optimization_utils import (
            update_mfg_config_with_params
        )
        
        # 创建测试参数
        test_params = {
            'rho': 0.45,
            'kappa': 2500.0,
            'alpha_T': 0.35,
            'gamma_T': 0.32,
            'gamma_S': 0.50,
            'gamma_D': 0.50,
            'gamma_W': 0.18
        }
        
        # 更新配置（不保存）
        updated_config = update_mfg_config_with_params(
            Path(mfg_config_path),
            test_params,
            param_utils,
            save_path=None
        )
        
        print("✓ 参数更新测试成功")
        print("\n更新后的参数值:")
        print(f"  rho:     {updated_config['economics']['rho']}")
        print(f"  kappa:   {updated_config['economics']['kappa']}")
        print(f"  alpha_T: {updated_config['economics']['disutility_T']['alpha']}")
        print(f"  gamma_T: {updated_config['economics']['state_update']['gamma_T']}")
        print(f"  gamma_S: {updated_config['economics']['state_update']['gamma_S']}")
        print(f"  gamma_D: {updated_config['economics']['state_update']['gamma_D']}")
        print(f"  gamma_W: {updated_config['economics']['state_update']['gamma_W']}")
        
        print("\n结果：配置更新功能正常 ✓")
    except Exception as e:
        print(f"✗ 配置更新测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n结果：配置更新功能存在问题 ✗")
    
    print("\n" + "="*80)
    print("验证完成")
    print("="*80)
    
    if all_in_bounds and has_alpha_T and all_paths_valid:
        print("\n✅ 所有检查通过，校准模块配置正确！")
    else:
        print("\n⚠️ 部分检查未通过，请检查上述问题")
    
    print("="*80)


if __name__ == '__main__':
    verify_calibration_config()

