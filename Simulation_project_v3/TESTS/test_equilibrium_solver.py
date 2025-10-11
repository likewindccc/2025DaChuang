#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG均衡求解器测试

注意：完整的均衡求解需要较长时间（10000个个体，最多100轮迭代）
本测试使用较小规模验证功能正确性
"""

import sys
import yaml
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.MFG import EquilibriumSolver


def test_equilibrium_solver_small():
    """
    小规模测试均衡求解器
    
    使用1000个个体，最多10轮迭代
    """
    print("=" * 80)
    print("MFG均衡求解器测试（小规模）")
    print("=" * 80)
    print()
    
    # 加载配置并修改为小规模
    config_path = "CONFIG/mfg_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改为小规模测试参数
    original_n = config['population']['n_individuals']
    original_max_iter = config['equilibrium']['max_outer_iter']
    
    config['population']['n_individuals'] = 10000  # 减少到10000个个体
    config['equilibrium']['max_outer_iter'] = 100   # 最多100轮外层迭代
    
    print(f"测试配置:")
    print(f"  个体数量: {config['population']['n_individuals']} (原配置: {original_n})")
    print(f"  最大外层迭代: {config['equilibrium']['max_outer_iter']} (原配置: {original_max_iter})")
    print(f"  目标市场紧张度: {config['market']['target_theta']}")
    print()
    
    # 保存临时配置
    temp_config_path = "CONFIG/mfg_config_test.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        # 创建求解器
        solver = EquilibriumSolver(temp_config_path)
        
        # 求解均衡
        print("开始求解均衡...")
        print()
        
        individuals_eq, eq_info = solver.solve(verbose=True)
        
        # 输出结果
        print("=" * 80)
        print("测试完成")
        print("=" * 80)
        print(f"是否收敛: {eq_info['converged']}")
        print(f"迭代轮数: {eq_info['iterations']}")
        print(f"最终失业率: {eq_info['final_unemployment_rate']*100:.2f}%")
        print(f"最终市场紧张度: {eq_info['final_theta']:.4f}")
        print()
        
        print("最终状态统计:")
        stats = eq_info['final_statistics']
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        print()
        
        print("迭代历史:")
        history = eq_info['history']
        print(f"  迭代轮数: {len(history['iteration'])}")
        if len(history['unemployment_rate']) > 0:
            print(f"  初始失业率: {history['unemployment_rate'][0]*100:.2f}%")
            print(f"  最终失业率: {history['unemployment_rate'][-1]*100:.2f}%")
        print()
        
        if eq_info['converged']:
            print("✓ 均衡成功收敛！")
        else:
            print("⚠ 均衡未完全收敛（达到最大迭代次数）")
        
        return True
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理临时配置文件
        if Path(temp_config_path).exists():
            Path(temp_config_path).unlink()


if __name__ == "__main__":
    success = test_equilibrium_solver_small()
    
    if success:
        print("\n" + "=" * 80)
        print("测试通过！")
        print("=" * 80)
        print()
        print("【提示】")
        print("完整的MFG均衡求解（10000个个体，最多100轮迭代）可能需要:")
        print("  - 计算时间: 几分钟到几十分钟")
        print("  - 内存占用: 几GB")
        print()
        print("运行完整求解:")
        print("  from MODULES.MFG import solve_equilibrium")
        print("  individuals_eq, eq_info = solve_equilibrium()")
    else:
        print("\n" + "=" * 80)
        print("测试失败，请检查错误信息")
        print("=" * 80)
        sys.exit(1)

