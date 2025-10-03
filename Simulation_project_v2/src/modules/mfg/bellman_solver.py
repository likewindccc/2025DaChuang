"""
贝尔曼方程求解器

使用值迭代法求解MFG中的贝尔曼方程，得到最优价值函数和努力策略。

核心功能：
1. 失业者价值函数 V^U(x)
2. 就业者价值函数 V^E(x)
3. 最优努力策略 a*(x)
4. 值迭代收敛判断

理论基础（研究计划4.1.1节）：

失业者贝尔曼方程：
V^U(x) = max_a {[b_0 - 0.5*κ*a²] + ρ[λ(x,a,θ)*V^E(x') + (1-λ)*V^U(x')]}

就业者贝尔曼方程：
V^E(x) = (W - α_T*T) + ρ[μ*V^U(x') + (1-μ)*V^E(x')]

其中:
- x' = 状态转移函数(x, a)
- λ = 匹配概率函数
- μ = 外生离职率（常数）
- ρ = 贴现因子

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, Optional, Callable
import logging

# 导入真实的状态转移函数
from .state_transition import state_transition_full
# 导入真实的匹配函数（Numba版本）
from ..estimation.match_function import compute_match_probability_numba

logger = logging.getLogger(__name__)


@njit
def compute_expected_continuation_value_unemployment(
    x: np.ndarray,
    a: float,
    match_prob: float,
    V_E_next: np.ndarray,
    V_U_next: np.ndarray,
    grid_nodes: np.ndarray,
    k_neighbors: int = 16
) -> float:
    """
    计算失业者的期望延续价值（Numba优化）
    
    E[V_{t+1} | x, a] = λ(x,a,θ)*V^E(x') + (1-λ)*V^U(x')
    
    Args:
        x: 当前状态（标准化） (4,)
        a: 努力水平
        match_prob: 匹配概率 λ(x,a,θ)
        V_E_next: 下一期就业价值函数（在网格点上）
        V_U_next: 下一期失业价值函数（在网格点上）
        grid_nodes: 稀疏网格节点 (4, n_points)
        k_neighbors: 插值近邻数
    
    Returns:
        期望延续价值
    
    Notes:
        - 需要先计算x'（状态转移），然后插值查询V(x')
    """
    # 这里假设x'已经通过状态转移计算好
    # 实际使用时需要从外部传入x'或在此处调用状态转移
    
    # 使用IDW插值查询V^E(x')和V^U(x')
    V_E_next_interp = linear_interpolate_single(x, grid_nodes, V_E_next, k_neighbors)
    V_U_next_interp = linear_interpolate_single(x, grid_nodes, V_U_next, k_neighbors)
    
    # 加权期望
    expected_value = match_prob * V_E_next_interp + (1.0 - match_prob) * V_U_next_interp
    
    return expected_value


@njit
def linear_interpolate_single(
    query_point: np.ndarray,
    grid_nodes: np.ndarray,
    grid_values: np.ndarray,
    k_neighbors: int = 16
) -> float:
    """
    单点线性插值（Numba优化，简化版本）
    
    使用反距离加权（IDW）方法。
    
    Args:
        query_point: 查询点 (dimension,)
        grid_nodes: 网格节点 (dimension, n_points)
        grid_values: 网格点值 (n_points,)
        k_neighbors: 近邻数
    
    Returns:
        插值结果
    """
    n_points = grid_nodes.shape[1]
    dimension = query_point.shape[0]
    
    # 计算所有距离
    distances = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        dist_sq = 0.0
        for d in range(dimension):
            diff = query_point[d] - grid_nodes[d, i]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)
    
    # 找k个最近邻
    k = min(k_neighbors, n_points)
    k_indices = np.argsort(distances)[:k]
    
    # IDW插值
    epsilon = 1e-10
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for idx in k_indices:
        dist = distances[idx]
        
        # 如果距离极小，直接返回该点值
        if dist < epsilon:
            return grid_values[idx]
        
        weight = 1.0 / (dist * dist + epsilon)
        weighted_sum += weight * grid_values[idx]
        weight_sum += weight
    
    return weighted_sum / weight_sum


@njit
def bellman_update_unemployment(
    x: np.ndarray,
    a_grid: np.ndarray,
    grid_nodes: np.ndarray,
    V_E_next: np.ndarray,
    V_U_next: np.ndarray,
    match_prob_func: Callable,
    state_transition_func: Callable,
    b_0: float,
    kappa: float,
    rho: float,
    theta: float,
    state_params: Dict,
    k_neighbors: int = 16
) -> Tuple[float, float]:
    """
    失业者贝尔曼更新（无法用Numba，因为需要Callable）
    
    V^U(x) = max_a {[b_0 - 0.5*κ*a²] + ρ[λ(x,a,θ)*V^E(x') + (1-λ)*V^U(x')]}
    
    Args:
        x: 当前状态
        a_grid: 努力水平离散网格
        grid_nodes: 稀疏网格节点
        V_E_next: 下一期就业价值函数
        V_U_next: 下一期失业价值函数
        match_prob_func: 匹配概率函数
        state_transition_func: 状态转移函数
        b_0: 失业补助
        kappa: 努力成本系数
        rho: 贴现因子
        theta: 市场紧张度
        state_params: 状态转移参数
        k_neighbors: 插值近邻数
    
    Returns:
        (最优价值, 最优努力水平)
    """
    n_efforts = len(a_grid)
    values = np.empty(n_efforts, dtype=np.float64)
    
    for i in range(n_efforts):
        a = a_grid[i]
        
        # 1. 瞬时效用
        instant_utility = b_0 - 0.5 * kappa * a * a
        
        # 2. 计算下一期状态 x'
        # 注意：这里x是标准化状态，需要传递给状态转移函数
        # x_next = state_transition_func(x, a, state_params)
        
        # 简化处理：假设x'已经计算（实际需要调用state_transition模块）
        # 这里先用当前x代替，后续修改
        x_next = x  # TODO: 调用state_transition
        
        # 3. 计算匹配概率
        # lambda_val = match_prob_func(x, a, theta)
        lambda_val = 0.5  # TODO: 调用match_function
        
        # 4. 插值查询V^E(x')和V^U(x')
        V_E_next_val = linear_interpolate_single(x_next, grid_nodes, V_E_next, k_neighbors)
        V_U_next_val = linear_interpolate_single(x_next, grid_nodes, V_U_next, k_neighbors)
        
        # 5. 期望延续价值
        continuation = lambda_val * V_E_next_val + (1.0 - lambda_val) * V_U_next_val
        
        # 6. 总价值
        values[i] = instant_utility + rho * continuation
    
    # 找到最优努力
    best_idx = np.argmax(values)
    best_value = values[best_idx]
    best_effort = a_grid[best_idx]
    
    return best_value, best_effort


class BellmanSolver:
    """
    贝尔曼方程求解器
    
    使用值迭代法求解失业者和就业者的价值函数。
    
    Attributes:
        grid_nodes: 稀疏网格节点 (dimension, n_points)
        n_points: 网格点数
        a_grid: 努力水平离散网格 (n_efforts,)
        config: 配置参数
    """
    
    def __init__(
        self,
        grid_nodes: np.ndarray,
        config: Dict,
        match_func_params: Optional[np.ndarray] = None
    ):
        """
        初始化贝尔曼求解器
        
        Args:
            grid_nodes: 稀疏网格节点矩阵 (dimension, n_points)
            config: 配置字典，包含所有参数
            match_func_params: 匹配函数参数数组（Numba格式）
        """
        self.grid_nodes = grid_nodes
        self.dimension = grid_nodes.shape[0]
        self.n_points = grid_nodes.shape[1]
        self.config = config
        
        # 努力水平离散化
        n_efforts = config['bellman']['n_effort_grid']
        self.a_grid = np.linspace(0.0, 1.0, n_efforts)
        self.n_efforts = n_efforts
        
        # 效用函数参数
        self.rho = config['bellman']['rho']
        self.b_0 = config['utility']['unemployment']['b_0']
        self.kappa = config['utility']['unemployment']['kappa']
        self.alpha_T = config['utility']['employment']['alpha_T']
        self.mu = config['kfe']['mu']
        self.theta = config['market']['theta_bar']
        
        # 状态转移参数
        self.gamma_T = config['state_transition']['gamma_T']
        self.gamma_S = config['state_transition']['gamma_S']
        self.gamma_D = config['state_transition']['gamma_D']
        self.gamma_W = config['state_transition']['gamma_W']
        self.T_max = config['state_transition']['T_max']
        self.W_min = config['state_transition']['W_min']
        
        # 匹配函数参数
        self.match_func_params = match_func_params
        if match_func_params is None:
            # 默认参数（全零，相当于lambda=0.5）
            self.match_func_params = np.zeros(11, dtype=np.float64)
            logger.warning("未提供匹配函数参数，使用默认值（全零）")
        
        # sigma控制变量（方案C：吸收到截距，传递零向量）
        self.sigma_zero = np.zeros(4, dtype=np.float64)
        
        # 收敛参数
        self.max_iterations = config['bellman']['max_iterations']
        self.tolerance = config['bellman']['tolerance']
        
        # 插值参数
        self.k_neighbors = 16  # 默认16个近邻
        
        # 初始化价值函数（在网格点上）
        self.V_U = np.zeros(self.n_points, dtype=np.float64)
        self.V_E = np.zeros(self.n_points, dtype=np.float64)
        
        # 最优努力策略（在网格点上）
        self.a_star = np.zeros(self.n_points, dtype=np.float64)
        
        logger.info(
            f"初始化贝尔曼求解器：{self.n_points}个网格点，"
            f"{self.n_efforts}个努力水平"
        )
    
    def solve(
        self,
        match_prob_func: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict:
        """
        求解贝尔曼方程（值迭代法）
        
        Args:
            match_prob_func: 匹配概率函数 λ(x, a, θ)
            verbose: 是否输出详细信息
        
        Returns:
            包含求解结果的字典：
            - 'V_U': 失业价值函数
            - 'V_E': 就业价值函数
            - 'a_star': 最优努力策略
            - 'n_iterations': 迭代次数
            - 'converged': 是否收敛
        """
        logger.info("开始值迭代求解贝尔曼方程...")
        
        # 初始化
        V_U_old = self.V_U.copy()
        V_E_old = self.V_E.copy()
        
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # 更新就业者价值函数（较简单，先更新）
            V_E_new = self._update_employment_value(V_U_old, V_E_old)
            
            # 更新失业者价值函数和最优策略
            V_U_new, a_star_new = self._update_unemployment_value(
                V_E_new, V_U_old, match_prob_func
            )
            
            # 检查收敛
            diff_V_U = np.max(np.abs(V_U_new - V_U_old))
            diff_V_E = np.max(np.abs(V_E_new - V_E_old))
            max_diff = max(diff_V_U, diff_V_E)
            
            if verbose and (iteration % 10 == 0):
                logger.info(
                    f"  迭代 {iteration}: "
                    f"max_diff={max_diff:.6e}, "
                    f"V_U_mean={np.mean(V_U_new):.2f}, "
                    f"V_E_mean={np.mean(V_E_new):.2f}"
                )
            
            # 判断收敛
            if max_diff < self.tolerance:
                converged = True
                break
            
            # 更新
            V_U_old = V_U_new
            V_E_old = V_E_new
            self.a_star = a_star_new
        
        # 保存结果
        self.V_U = V_U_new
        self.V_E = V_E_new
        
        if converged:
            logger.info(
                f"✅ 贝尔曼方程在{iteration}次迭代后收敛！"
                f"（容差={self.tolerance:.2e}）"
            )
        else:
            logger.warning(
                f"⚠️ 贝尔曼方程未在{self.max_iterations}次迭代内收敛！"
                f"（当前差异={max_diff:.2e}）"
            )
        
        return {
            'V_U': self.V_U,
            'V_E': self.V_E,
            'a_star': self.a_star,
            'n_iterations': iteration + 1,
            'converged': converged,
            'final_diff': max_diff
        }
    
    def _update_employment_value(
        self,
        V_U_current: np.ndarray,
        V_E_current: np.ndarray
    ) -> np.ndarray:
        """
        更新就业者价值函数
        
        V^E(x) = (W - α_T*T) + ρ[μ*V^U(x') + (1-μ)*V^E(x')]
        
        Args:
            V_U_current: 当前失业价值函数
            V_E_current: 当前就业价值函数
        
        Returns:
            更新后的就业价值函数
        """
        V_E_new = np.empty(self.n_points, dtype=np.float64)
        
        for i in range(self.n_points):
            # 获取网格点状态
            x = self.grid_nodes[:, i]
            
            # 提取T和W（标准化状态中：x = [T, S_norm, D_norm, W]）
            T = x[0]
            W = x[3]
            
            # 瞬时效用
            instant_utility = W - self.alpha_T * T
            
            # 下一期状态（就业者没有努力决策，a=0）
            x_next = state_transition_full(
                x, a=0.0,
                gamma_T=self.gamma_T,
                gamma_S=self.gamma_S,
                gamma_D=self.gamma_D,
                gamma_W=self.gamma_W,
                T_max=self.T_max,
                W_min=self.W_min
            )
            
            # 插值查询下一期价值
            V_U_next = linear_interpolate_single(
                x_next, self.grid_nodes, V_U_current, self.k_neighbors
            )
            V_E_next = linear_interpolate_single(
                x_next, self.grid_nodes, V_E_current, self.k_neighbors
            )
            
            # 期望延续价值
            continuation = self.mu * V_U_next + (1.0 - self.mu) * V_E_next
            
            # 总价值
            V_E_new[i] = instant_utility + self.rho * continuation
        
        return V_E_new
    
    def _update_unemployment_value(
        self,
        V_E_current: np.ndarray,
        V_U_current: np.ndarray,
        match_prob_func: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新失业者价值函数和最优努力策略
        
        V^U(x) = max_a {[b_0 - 0.5*κ*a²] + ρ[λ(x,a,θ)*V^E(x') + (1-λ)*V^U(x')]}
        
        Args:
            V_E_current: 当前就业价值函数
            V_U_current: 当前失业价值函数
            match_prob_func: 匹配概率函数（可选，未提供则使用内置）
        
        Returns:
            (更新后的失业价值函数, 最优努力策略)
        """
        V_U_new = np.empty(self.n_points, dtype=np.float64)
        a_star_new = np.empty(self.n_points, dtype=np.float64)
        
        for i in range(self.n_points):
            # 获取网格点状态
            x = self.grid_nodes[:, i]
            
            # 对所有努力水平进行搜索
            best_value = -np.inf
            best_effort = 0.0
            
            for a in self.a_grid:
                # 1. 瞬时效用
                instant_utility = self.b_0 - 0.5 * self.kappa * a * a
                
                # 2. 状态转移（使用真实的状态转移函数）
                x_next = state_transition_full(
                    x, a=a,
                    gamma_T=self.gamma_T,
                    gamma_S=self.gamma_S,
                    gamma_D=self.gamma_D,
                    gamma_W=self.gamma_W,
                    T_max=self.T_max,
                    W_min=self.W_min
                )
                
                # 3. 计算匹配概率（使用真实的匹配函数）
                lambda_val = compute_match_probability_numba(
                    x, self.sigma_zero, a, self.theta, self.match_func_params
                )
                
                # 4. 插值查询下一期价值
                V_E_next = linear_interpolate_single(
                    x_next, self.grid_nodes, V_E_current, self.k_neighbors
                )
                V_U_next = linear_interpolate_single(
                    x_next, self.grid_nodes, V_U_current, self.k_neighbors
                )
                
                # 5. 期望延续价值
                continuation = lambda_val * V_E_next + (1.0 - lambda_val) * V_U_next
                
                # 6. 总价值
                total_value = instant_utility + self.rho * continuation
                
                # 更新最优
                if total_value > best_value:
                    best_value = total_value
                    best_effort = a
            
            V_U_new[i] = best_value
            a_star_new[i] = best_effort
        
        return V_U_new, a_star_new
    
    def get_value_at_point(
        self,
        x: np.ndarray,
        employment_status: str = 'unemployed'
    ) -> float:
        """
        查询任意状态点的价值函数
        
        Args:
            x: 状态向量 (4,)
            employment_status: 'unemployed' 或 'employed'
        
        Returns:
            插值后的价值
        """
        if employment_status == 'unemployed':
            return linear_interpolate_single(
                x, self.grid_nodes, self.V_U, self.k_neighbors
            )
        else:
            return linear_interpolate_single(
                x, self.grid_nodes, self.V_E, self.k_neighbors
            )
    
    def get_optimal_effort(self, x: np.ndarray) -> float:
        """
        查询任意状态点的最优努力水平
        
        Args:
            x: 状态向量 (4,)
        
        Returns:
            最优努力水平
        """
        return linear_interpolate_single(
            x, self.grid_nodes, self.a_star, self.k_neighbors
        )
    
    def __repr__(self) -> str:
        return (
            f"BellmanSolver(n_points={self.n_points}, "
            f"n_efforts={self.n_efforts}, "
            f"rho={self.rho}, tolerance={self.tolerance:.2e})"
        )


# 简单测试
if __name__ == "__main__":
    # 当作为脚本运行时，需要添加路径
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # 重新导入（使用绝对导入）
    from modules.mfg.state_transition import state_transition_full
    from modules.estimation.match_function import compute_match_probability_numba
    
    print("=" * 60)
    print("测试贝尔曼求解器模块")
    print("=" * 60)
    
    # 创建测试配置
    config = {
        'bellman': {
            'n_effort_grid': 21,
            'rho': 0.9,
            'max_iterations': 100,
            'tolerance': 1e-4
        },
        'utility': {
            'unemployment': {'b_0': 500, 'kappa': 1.0},
            'employment': {'alpha_T': 10.0}
        },
        'kfe': {'mu': 0.05},
        'market': {'theta_bar': 1.0},
        'state_transition': {
            'gamma_T': 0.1,
            'gamma_S': 0.05,
            'gamma_D': 0.08,
            'gamma_W': 100.0,
            'T_max': 70.0,
            'W_min': 1400.0
        }
    }
    
    # 创建小规模测试网格（缩放到实际状态空间）
    np.random.seed(42)
    n_test_points = 50
    
    # 生成合理范围的网格点: [T, S_norm, D_norm, W]
    T_samples = np.random.uniform(15, 70, n_test_points)
    S_norm_samples = np.random.uniform(0, 1, n_test_points)
    D_norm_samples = np.random.uniform(0, 1, n_test_points)
    W_samples = np.random.uniform(1400, 8000, n_test_points)
    
    grid_nodes = np.vstack([T_samples, S_norm_samples, D_norm_samples, W_samples])
    
    # 初始化求解器（使用默认匹配函数参数）
    solver = BellmanSolver(grid_nodes, config, match_func_params=None)
    print(f"\n初始化: {solver}")
    
    # 求解（使用真实的状态转移和匹配函数）
    print("\n开始求解（使用真实函数）...")
    result = solver.solve(verbose=True)
    
    print(f"\n求解完成！")
    print(f"  迭代次数: {result['n_iterations']}")
    print(f"  是否收敛: {result['converged']}")
    print(f"  最终差异: {result['final_diff']:.2e}")
    print(f"  V_U 均值: {np.mean(result['V_U']):.2f}")
    print(f"  V_E 均值: {np.mean(result['V_E']):.2f}")
    print(f"  a* 均值: {np.mean(result['a_star']):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 贝尔曼求解器模块测试完成！")
    print("=" * 60)

