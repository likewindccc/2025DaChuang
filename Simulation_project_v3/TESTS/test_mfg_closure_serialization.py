#!/usr/bin/env python3
# 测试multiprocess是否能序列化真实的MFG闭包

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_real_mfg_closure():
    """
    模拟真实的_create_mfg_solver闭包结构
    """
    print("="*80)
    print("测试真实MFG闭包的序列化")
    print("="*80)
    
    # 模拟捕获的外部变量
    mfg_config_path = Path("CONFIG/mfg_config.yaml")
    output_dir = Path("OUTPUT/calibration")
    
    # 模拟OptimizationUtils类
    class MockOptimizationUtils:
        def __init__(self):
            self.param_names = ['rho', 'kappa', 'alpha_T']
        
        def vector_to_dict(self, vector):
            return dict(zip(self.param_names, vector))
    
    param_utils = MockOptimizationUtils()
    
    # 创建闭包（模拟_create_mfg_solver）
    def create_solver():
        # 这些变量被闭包捕获
        config_path = mfg_config_path
        utils = param_utils
        out_dir = output_dir
        
        def solver(params_vector):
            """模拟mfg_solver闭包"""
            # 使用捕获的变量
            params_dict = utils.vector_to_dict(params_vector)
            
            # 模拟文件操作
            temp_file = out_dir / 'test_temp.txt'
            
            # 模拟返回值
            return {
                'config': str(config_path),
                'params': params_dict,
                'sum': np.sum(params_vector)
            }
        
        return solver
    
    # 创建闭包
    mfg_solver = create_solver()
    
    # 测试单进程执行
    print("\n[1] 单进程测试...")
    result_single = mfg_solver(np.array([0.4, 2000.0, 0.3]))
    print(f"  ✓ 单进程结果: {result_single}")
    
    # 测试multiprocess序列化
    print("\n[2] multiprocess并行测试...")
    try:
        import multiprocess as mp
        
        # 准备测试数据（3组参数）
        test_params = [
            np.array([0.4, 2000.0, 0.3]),
            np.array([0.5, 1500.0, 0.2]),
            np.array([0.3, 2500.0, 0.4])
        ]
        
        with mp.Pool(2) as pool:
            results = pool.map(mfg_solver, test_params)
        
        print(f"  ✓ 并行执行成功！")
        for i, res in enumerate(results):
            print(f"    参数{i+1}: sum={res['sum']:.1f}, params={res['params']}")
        
        print("\n" + "="*80)
        print("✅ 结论：multiprocess可以序列化真实的MFG闭包！")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"  ✗ 并行执行失败: {type(e).__name__}: {e}")
        print("\n" + "="*80)
        print("❌ 结论：multiprocess无法序列化MFG闭包")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_real_mfg_closure()
    sys.exit(0 if success else 1)

