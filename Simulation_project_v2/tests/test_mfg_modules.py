"""
MFG模块单元测试

测试所有核心MFG组件的功能正确性。

测试覆盖：
1. StateSpace - 状态空间管理
2. SparseGrid - 稀疏网格生成
3. Interpolation - 插值算法
4. StateTransition - 状态转移函数
5. UtilityFunctions - 效用函数
6. BellmanSolver - 贝尔曼方程求解
7. KFESolver - KFE演化

作者：AI Assistant
日期：2025-10-03
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.mfg.state_space import StateSpace
from src.modules.mfg.sparse_grid import SparseGrid
from src.modules.mfg.interpolation import (
    batch_linear_interpolate,
    linear_interpolate
)
from src.modules.mfg.state_transition import (
    state_transition_T,
    state_transition_S,
    state_transition_D,
    state_transition_W,
    state_transition_full
)
from src.modules.mfg.utility_functions import (
    utility_unemployment,
    utility_employment
)
from src.modules.mfg.bellman_solver import BellmanSolver
from src.modules.mfg.kfe_solver import KFESolver


class TestStateSpace(unittest.TestCase):
    """测试StateSpace类"""

    def setUp(self):
        """初始化测试环境"""
        self.config = {
            'T_range': [15, 70],
            'S_range': [2, 44],
            'D_range': [0, 20],
            'W_range': [1400, 8000]
        }
        self.state_space = StateSpace(self.config)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.state_space.dims, 4)
        np.testing.assert_array_equal(
            self.state_space.T_range,
            np.array([15, 70], dtype=np.float64)
        )

    def test_normalization(self):
        """测试状态标准化"""
        # 测试边界值
        state_min = np.array([15, 2, 0, 1400], dtype=np.float64)
        state_max = np.array([70, 44, 20, 8000], dtype=np.float64)

        norm_min = self.state_space.normalize_state(state_min)
        norm_max = self.state_space.normalize_state(state_max)

        np.testing.assert_array_almost_equal(norm_min, np.zeros(4))
        np.testing.assert_array_almost_equal(norm_max, np.ones(4))

    def test_denormalization(self):
        """测试状态反标准化"""
        norm_state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        original_state = self.state_space.denormalize_state(norm_state)

        expected = np.array([
            (15 + 70) / 2,
            (2 + 44) / 2,
            (0 + 20) / 2,
            (1400 + 8000) / 2
        ], dtype=np.float64)

        np.testing.assert_array_almost_equal(original_state, expected)

    def test_roundtrip(self):
        """测试标准化-反标准化循环"""
        original = np.array([40, 20, 10, 5000], dtype=np.float64)
        normalized = self.state_space.normalize_state(original)
        recovered = self.state_space.denormalize_state(normalized)

        np.testing.assert_array_almost_equal(original, recovered)


class TestSparseGrid(unittest.TestCase):
    """测试SparseGrid类"""

    def test_grid_generation_2d(self):
        """测试2维稀疏网格生成"""
        bounds = [(0, 1), (0, 1)]
        grid = SparseGrid(bounds, level=2)

        self.assertGreater(grid.n_points, 0)
        self.assertEqual(grid.nodes.shape[0], 2)
        self.assertEqual(grid.nodes.shape[1], grid.n_points)

    def test_grid_generation_4d(self):
        """测试4维稀疏网格生成（实际使用场景）"""
        bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        grid = SparseGrid(bounds, level=3)

        self.assertGreater(grid.n_points, 0)
        self.assertEqual(grid.nodes.shape[0], 4)
        self.assertEqual(grid.dimension, 4)

        # 检查节点范围
        for i, (low, high) in enumerate(bounds):
            self.assertTrue(np.all(grid.nodes[i, :] >= low))
            self.assertTrue(np.all(grid.nodes[i, :] <= high))

    def test_efficiency(self):
        """测试稀疏网格效率"""
        bounds = [(0, 1)] * 4
        level = 3
        grid = SparseGrid(bounds, level=level)

        full_tensor_points = (level + 1) ** 4
        self.assertLess(grid.n_points, full_tensor_points)
        self.assertLess(grid.efficiency, 1.0)


class TestInterpolation(unittest.TestCase):
    """测试插值算法"""

    def setUp(self):
        """初始化测试数据"""
        # 创建简单的2D网格用于测试
        np.random.seed(42)
        self.grid_nodes = np.random.rand(2, 50)  # 2D, 50个点
        self.grid_values = np.sum(self.grid_nodes, axis=0)  # 简单的和函数

    def test_single_interpolation(self):
        """测试单点插值"""
        query_point = np.array([0.5, 0.5])
        result = linear_interpolate(
            query_point,
            self.grid_nodes,
            self.grid_values,
            k=5
        )

        # 结果应该接近1.0 (0.5 + 0.5)
        self.assertIsInstance(result, (float, np.floating))
        self.assertGreater(result, 0.0)

    def test_batch_interpolation(self):
        """测试批量插值"""
        query_points = np.random.rand(2, 10)  # 10个查询点
        results = batch_linear_interpolate(
            query_points,
            self.grid_nodes,
            self.grid_values,
            k=5
        )

        self.assertEqual(results.shape, (10,))
        self.assertTrue(np.all(results > 0))


class TestStateTransition(unittest.TestCase):
    """测试状态转移函数"""

    def test_update_T(self):
        """测试工作时间更新"""
        T_current = 20.0
        a = 0.5
        gamma_T = 0.1
        T_max = 70.0

        T_next = state_transition_T(T_current, a, gamma_T, T_max)

        # T应该增加，且不超过T_max
        self.assertGreater(T_next, T_current)
        self.assertLessEqual(T_next, T_max)

    def test_update_W(self):
        """测试期望工资更新"""
        W_current = 5000.0
        a = 0.5
        gamma_W = 100.0
        W_min = 1400.0

        W_next = state_transition_W(W_current, a, gamma_W, W_min)

        # W应该减少，但不低于W_min
        self.assertLess(W_next, W_current)
        self.assertGreaterEqual(W_next, W_min)

    def test_update_S(self):
        """测试工作能力更新（标准化）"""
        S_current = 0.5
        a = 0.3
        gamma_S = 0.05

        S_next = state_transition_S(S_current, a, gamma_S)

        # S应该增加，且在[0, 1]范围内
        self.assertGreater(S_next, S_current)
        self.assertLessEqual(S_next, 1.0)

    def test_update_D(self):
        """测试数字素养更新（标准化）"""
        D_current = 0.3
        a = 0.6
        gamma_D = 0.08

        D_next = state_transition_D(D_current, a, gamma_D)

        # D应该增加，且在[0, 1]范围内
        self.assertGreater(D_next, D_current)
        self.assertLessEqual(D_next, 1.0)

    def test_state_transition_full(self):
        """测试完整状态转移"""
        x_current = np.array([30.0, 0.5, 0.4, 4000.0])
        a = 0.5

        x_next = state_transition_full(
            x_current,
            a=a,
            gamma_T=0.1,
            gamma_S=0.05,
            gamma_D=0.08,
            gamma_W=100.0,
            T_max=70.0,
            W_min=1400.0
        )

        # 检查维度
        self.assertEqual(x_next.shape, (4,))

        # 检查T增加
        self.assertGreater(x_next[0], x_current[0])

        # 检查W减少
        self.assertLess(x_next[3], x_current[3])


class TestUtilityFunctions(unittest.TestCase):
    """测试效用函数"""

    def test_unemployment_utility(self):
        """测试失业效用"""
        a = 0.5
        b_0 = 500.0
        kappa = 1.0

        utility = utility_unemployment(a, b_0, kappa)

        # 应该等于 b_0 - 0.5 * kappa * a^2
        expected = b_0 - 0.5 * kappa * a * a
        self.assertAlmostEqual(utility, expected, places=10)

    def test_employment_utility(self):
        """测试就业效用"""
        W = 4000.0
        T = 30.0
        alpha_T = 10.0

        utility = utility_employment(W, T, alpha_T)

        # 应该等于 W - alpha_T * T
        expected = W - alpha_T * T
        self.assertAlmostEqual(utility, expected, places=10)

    def test_effort_cost(self):
        """测试努力成本随努力水平变化"""
        b_0 = 500.0
        kappa = 1.0

        u_low = utility_unemployment(0.1, b_0, kappa)
        u_high = utility_unemployment(0.9, b_0, kappa)

        # 高努力应该导致更低的效用（更高的成本）
        self.assertLess(u_high, u_low)


class TestBellmanSolver(unittest.TestCase):
    """测试贝尔曼方程求解器"""

    def setUp(self):
        """初始化求解器"""
        # 创建小规模网格用于快速测试
        self.bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        self.grid = SparseGrid(self.bounds, level=2)

        # 简化的匹配函数参数（假设）
        self.match_func_params = np.array([
            -2.0, 0.1, 0.05, 0.03, 0.02, 0.5
        ])

        self.solver = BellmanSolver(
            grid_nodes=self.grid.nodes,
            rho=0.9,
            kappa=1.0,
            mu=0.05,
            b_0=500.0,
            alpha_T=10.0,
            theta=1.0,
            gamma_T=0.1,
            gamma_S=0.05,
            gamma_D=0.08,
            gamma_W=100.0,
            T_max=70.0,
            W_min=1400.0,
            effort_levels=11,
            match_func_params=self.match_func_params,
            k_neighbors=8
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.solver.n_points, self.grid.n_points)
        self.assertEqual(len(self.solver.a_grid), 11)

    def test_solve_bellman_iteration(self):
        """测试贝尔曼方程迭代"""
        # 运行少量迭代
        V_U, V_E, a_star, convergence = self.solver.solve_bellman(
            max_iterations=10,
            tol_value=1e-4,
            tol_policy=1e-4
        )

        # 检查输出形状
        self.assertEqual(V_U.shape, (self.grid.n_points,))
        self.assertEqual(V_E.shape, (self.grid.n_points,))
        self.assertEqual(a_star.shape, (self.grid.n_points,))

        # 检查策略在合理范围内
        self.assertTrue(np.all(a_star >= 0))
        self.assertTrue(np.all(a_star <= 1))


class TestKFESolver(unittest.TestCase):
    """测试KFE求解器"""

    def setUp(self):
        """初始化KFE求解器"""
        self.bounds = [(15, 70), (2, 44), (0, 20), (1400, 8000)]
        self.grid = SparseGrid(self.bounds, level=2)

        self.match_func_params = np.array([
            -2.0, 0.1, 0.05, 0.03, 0.02, 0.5
        ])

        self.solver = KFESolver(
            grid_nodes=self.grid.nodes,
            mu=0.05,
            theta=1.0,
            match_func_params=self.match_func_params,
            gamma_T=0.1,
            gamma_S=0.05,
            gamma_D=0.08,
            gamma_W=100.0,
            T_max=70.0,
            W_min=1400.0,
            k_neighbors=8
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.solver.n_points, self.grid.n_points)

    def test_uniform_initialization(self):
        """测试均匀初始化"""
        m_U, m_E = self.solver.initialize_uniform_distribution()

        # 检查归一化
        total = np.sum(m_U) + np.sum(m_E)
        self.assertAlmostEqual(total, 1.0, places=10)

        # 检查非负
        self.assertTrue(np.all(m_U >= 0))
        self.assertTrue(np.all(m_E >= 0))

    def test_evolve_population(self):
        """测试人口演化"""
        # 初始化
        m_U, m_E = self.solver.initialize_uniform_distribution()

        # 创建简单的策略（所有人努力0.5）
        a_star = np.full(self.grid.n_points, 0.5)

        # 演化一步
        m_U_next, m_E_next = self.solver.evolve_population(
            m_U, m_E, a_star
        )

        # 检查归一化
        total = np.sum(m_U_next) + np.sum(m_E_next)
        self.assertAlmostEqual(total, 1.0, places=8)

        # 检查非负
        self.assertTrue(np.all(m_U_next >= 0))
        self.assertTrue(np.all(m_E_next >= 0))

    def test_unemployment_rate_calculation(self):
        """测试失业率计算"""
        m_U = np.array([0.3, 0.2])
        m_E = np.array([0.4, 0.1])

        u_rate = self.solver.calculate_unemployment_rate(m_U, m_E)

        expected = 0.5 / 1.0  # (0.3 + 0.2) / (0.3 + 0.2 + 0.4 + 0.1)
        self.assertAlmostEqual(u_rate, expected, places=10)


def run_tests(verbosity=2):
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestStateSpace))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseGrid))
    suite.addTests(loader.loadTestsFromTestCase(TestInterpolation))
    suite.addTests(loader.loadTestsFromTestCase(TestStateTransition))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestBellmanSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestKFESolver))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("MFG模块单元测试")
    print("=" * 70)
    print()

    result = run_tests(verbosity=2)

    # 输出测试摘要
    print("\n" + "=" * 70)
    print("测试摘要:")
    print(f"  运行测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 70)

    # 退出码
    sys.exit(0 if result.wasSuccessful() else 1)

