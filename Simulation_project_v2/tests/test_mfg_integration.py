"""
MFG模块集成测试

测试MFG求解器的端到端功能。

作者：AI Assistant
日期：2025-10-03
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml

def test_mfg_components_import():
    """测试1: 验证所有MFG组件可以正确导入"""
    print("=" * 70)
    print("测试1: 导入所有MFG组件")
    print("=" * 70)
    
    try:
        from src.modules.mfg.state_space import StateSpace
        from src.modules.mfg.sparse_grid import SparseGrid
        from src.modules.mfg.interpolation import linear_interpolate, batch_linear_interpolate
        from src.modules.mfg.state_transition import state_transition_full
        from src.modules.mfg.utility_functions import utility_unemployment, utility_employment
        from src.modules.mfg.bellman_solver import BellmanSolver
        from src.modules.mfg.kfe_solver import KFESolver
        from src.modules.mfg.mfg_simulator import MFGSimulator
        from src.core.data_structures import MFGEquilibriumSparseGrid
        
        print("✅ 所有组件导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_sparse_grid_generation():
    """测试2: 验证稀疏网格生成"""
    print("\n" + "=" * 70)
    print("测试2: 稀疏网格生成")
    print("=" * 70)
    
    try:
        from src.modules.mfg.sparse_grid import SparseGrid
        
        bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        grid = SparseGrid(bounds, level=3)
        
        print(f"✅ 稀疏网格生成成功")
        print(f"   维度: {grid.dimension}D")
        print(f"   Level: {grid.level}")
        print(f"   网格点数: {grid.n_points}")
        print(f"   效率: {grid.efficiency:.2%}")
        
        return grid.n_points > 0
    except Exception as e:
        print(f"❌ 稀疏网格生成失败: {e}")
        return False


def test_state_transition():
    """测试3: 验证状态转移函数"""
    print("\n" + "=" * 70)
    print("测试3: 状态转移函数")
    print("=" * 70)
    
    try:
        from src.modules.mfg.state_transition import state_transition_full
        
        x_current = np.array([30.0, 0.5, 0.4, 4000.0])
        x_next = state_transition_full(
            x_current,
            a=0.5,
            gamma_T=0.1,
            gamma_S=0.05,
            gamma_D=0.08,
            gamma_W=100.0,
            T_max=70.0,
            W_min=1400.0
        )
        
        print(f"✅ 状态转移计算成功")
        print(f"   当前状态: T={x_current[0]:.1f}, S={x_current[1]:.2f}, D={x_current[2]:.2f}, W={x_current[3]:.0f}")
        print(f"   下期状态: T={x_next[0]:.1f}, S={x_next[1]:.2f}, D={x_next[2]:.2f}, W={x_next[3]:.0f}")
        
        # 验证T增加，W减少
        assert x_next[0] > x_current[0], "T应该增加"
        assert x_next[3] < x_current[3], "W应该减少"
        
        return True
    except Exception as e:
        print(f"❌ 状态转移测试失败: {e}")
        return False


def test_utility_functions():
    """测试4: 验证效用函数"""
    print("\n" + "=" * 70)
    print("测试4: 效用函数")
    print("=" * 70)
    
    try:
        from src.modules.mfg.utility_functions import utility_unemployment, utility_employment
        
        u_unemp = utility_unemployment(a=0.5, b_0=500.0, kappa=1.0)
        u_emp = utility_employment(W=4000.0, T=40.0, alpha_T=10.0)
        
        print(f"✅ 效用函数计算成功")
        print(f"   失业效用 (a=0.5): {u_unemp:.2f}")
        print(f"   就业效用 (W=4000, T=40): {u_emp:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ 效用函数测试失败: {e}")
        return False


def test_config_loading():
    """测试5: 验证配置文件加载"""
    print("\n" + "=" * 70)
    print("测试5: 配置文件加载")
    print("=" * 70)
    
    try:
        config_path = project_root / "config" / "default" / "mfg.yaml"
        
        if not config_path.exists():
            print(f"⚠️  配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功")
        if 'state_space' in config:
            print(f"   网格Level: {config['state_space'].get('grid_level', 'N/A')}")
        if 'solver' in config:
            print(f"   最大迭代: {config['solver'].get('max_iterations', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_bellman_solver_init():
    """测试6: 验证Bellman求解器初始化"""
    print("\n" + "=" * 70)
    print("测试6: Bellman求解器初始化")
    print("=" * 70)
    
    try:
        from src.modules.mfg.sparse_grid import SparseGrid
        from src.modules.mfg.bellman_solver import BellmanSolver
        
        # 加载配置
        config_path = project_root / "config" / "default" / "mfg.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建小规模网格
        bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        grid = SparseGrid(bounds, level=2)  # 小规模用于测试
        
        # 创建Bellman求解器
        match_func_params = np.array([-2.0, 0.1, 0.05, 0.03, 0.02, 0.5])
        solver = BellmanSolver(grid.nodes, config, match_func_params)
        
        print(f"✅ Bellman求解器初始化成功")
        print(f"   网格点数: {solver.n_points}")
        print(f"   努力水平数: {len(solver.a_grid)}")
        
        return True
    except Exception as e:
        print(f"❌ Bellman求解器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kfe_solver_init():
    """测试7: 验证KFE求解器初始化"""
    print("\n" + "=" * 70)
    print("测试7: KFE求解器初始化")
    print("=" * 70)
    
    try:
        from src.modules.mfg.sparse_grid import SparseGrid
        from src.modules.mfg.kfe_solver import KFESolver
        
        # 加载配置
        config_path = project_root / "config" / "default" / "mfg.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建小规模网格
        bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        grid = SparseGrid(bounds, level=2)
        
        # 创建KFE求解器
        match_func_params = np.array([-2.0, 0.1, 0.05, 0.03, 0.02, 0.5])
        solver = KFESolver(grid.nodes, config, match_func_params)
        
        # 测试人口分布（KFE求解器在__init__时自动初始化）
        total = np.sum(solver.m_U) + np.sum(solver.m_E)
        
        print(f"✅ KFE求解器初始化成功")
        print(f"   网格点数: {solver.n_points}")
        print(f"   初始失业率: {solver.get_unemployment_rate():.2%}")
        print(f"   人口总和: {total:.6f}")
        
        assert abs(total - 1.0) < 1e-6, "人口总和应该为1"
        
        return True
    except Exception as e:
        print(f"❌ KFE求解器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("=" * 70)
    print("MFG模块集成测试套件")
    print("=" * 70)
    print()
    
    tests = [
        ("组件导入", test_mfg_components_import),
        ("稀疏网格生成", test_sparse_grid_generation),
        ("状态转移", test_state_transition),
        ("效用函数", test_utility_functions),
        ("配置加载", test_config_loading),
        ("Bellman求解器", test_bellman_solver_init),
        ("KFE求解器", test_kfe_solver_init),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("=" * 70)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

