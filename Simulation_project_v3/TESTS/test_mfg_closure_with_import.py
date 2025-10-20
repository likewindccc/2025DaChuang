#!/usr/bin/env python3
# 最终测试：包含闭包内部import的真实MFG场景

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_closure_with_internal_import():
    """
    测试包含内部import的闭包（完全模拟真实MFG场景）
    """
    print("="*80)
    print("最终测试：包含内部import的MFG闭包序列化")
    print("="*80)
    
    # 模拟外部函数（需要能被序列化）
    def mock_validate_params(vector, utils):
        return True, ""
    
    def mock_update_config(src, params, utils, dest):
        return True
    
    # 模拟被捕获的变量
    mfg_config_path = Path("CONFIG/mfg_config.yaml")
    output_dir = Path("OUTPUT/calibration")
    
    class MockParamUtils:
        def vector_to_dict(self, vector):
            return {'rho': vector[0], 'kappa': vector[1]}
    
    param_utils = MockParamUtils()
    
    # 创建包含内部import的闭包
    def create_solver():
        config_path = mfg_config_path
        utils = param_utils
        out_dir = output_dir
        validate = mock_validate_params
        update = mock_update_config
        
        def solver(params_vector):
            """
            真实MFG闭包的完整结构：
            1. 捕获外部变量
            2. 内部有import语句  ← 关键！
            3. 调用外部函数
            4. 文件操作
            """
            # 模拟闭包内部的import（这是关键测试点）
            from pathlib import Path as P  # 模拟内部import
            
            # 使用捕获的变量和函数
            is_valid, _ = validate(params_vector, utils)
            if not is_valid:
                return None
            
            params_dict = utils.vector_to_dict(params_vector)
            
            # 模拟文件操作
            temp_path = out_dir / 'test.yaml'
            result = update(config_path, params_dict, utils, temp_path)
            
            # 模拟返回MFG结果
            return {
                'params': params_dict,
                'sum': float(np.sum(params_vector)),
                'import_test': P('test').name  # 使用内部导入的模块
            }
        
        return solver
    
    mfg_solver = create_solver()
    
    # 测试1：单进程
    print("\n[1] 单进程测试...")
    result = mfg_solver(np.array([0.4, 2000.0]))
    print(f"  ✓ 单进程结果: {result}")
    
    # 测试2：multiprocess并行
    print("\n[2] multiprocess并行测试（关键）...")
    try:
        import multiprocess as mp
        
        test_params = [
            np.array([0.4, 2000.0]),
            np.array([0.5, 1500.0]),
            np.array([0.3, 2500.0])
        ]
        
        with mp.Pool(2) as pool:
            results = pool.map(mfg_solver, test_params)
        
        print(f"  ✓ 并行执行成功！")
        for i, res in enumerate(results):
            print(f"    参数{i+1}: {res}")
        
        # 验证内部import是否正常工作
        if all(r['import_test'] == 'test' for r in results):
            print(f"  ✓ 内部import功能正常！")
        
        print("\n" + "="*80)
        print("✅ 最终结论：multiprocess完全支持真实MFG闭包！")
        print("   包括：")
        print("   - 捕获外部变量 ✓")
        print("   - 闭包内部import ✓")
        print("   - 调用外部函数 ✓")
        print("   - 并行执行 ✓")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {type(e).__name__}: {e}")
        print("\n" + "="*80)
        print("❌ 最终结论：存在序列化问题")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_closure_with_internal_import()
    
    if success:
        print("\n🎉 可以放心运行完整校准！")
    else:
        print("\n⚠️  需要重构代码结构！")
    
    sys.exit(0 if success else 1)

