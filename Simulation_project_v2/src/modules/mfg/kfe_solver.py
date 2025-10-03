"""
Kolmogorov前向方程(KFE)求解器

描述人口分布在状态空间中的动态演化。

核心功能：
1. 失业者人口分布 m^U(x, t)
2. 就业者人口分布 m^E(x, t)
3. 失业率计算
4. 人口流动模拟

理论基础（研究计划4.1.2节）：

失业者人口演化：
m^U(x_{t+1}) = Σ[(1-λ(x_t, a*, θ))*I(x_{t+1}|x_t, a*)*m^U(x_t)] + μ*m^E(x_t)

就业者人口演化：
m^E(x_{t+1}) = Σ[λ(x_t, a*, θ)*I(x_{t+1}|x_t, a*)*m^U(x_t)] + (1-μ)*m^E(x_t)

其中:
- m^U, m^E: 失业/就业人口密度
- λ: 匹配概率
- a*: 最优努力策略（来自贝尔曼求解）
- μ: 外生离职率
- I(x'|x, a): 状态转移指示函数

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Optional
import logging

# 导入状态转移和匹配函数
from .state_transition import state_transition_full
from ..estimation.match_function import compute_match_probability_numba

logger = logging.getLogger(__name__)


@njit
def evolve_unemployment_distribution(
    m_U_current: np.ndarray,
    m_E_current: np.ndarray,
    a_star: np.ndarray,
    grid_nodes: np.ndarray,
    match_func_params: np.ndarray,
    sigma_zero: np.ndarray,
    theta: float,
    mu: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float,
    T_max: float,
    W_min: float,
    k_neighbors: int = 16
) -> np.ndarray:
    """
    更新失业者人口分布（Numba优化）
    
    m^U(x_{t+1}) = Σ[(1-λ)*I*m^U] + μ*m^E
    
    Args:
        m_U_current: 当前失业者分布 (n_points,)
        m_E_current: 当前就业者分布 (n_points,)
        a_star: 最优努力策略 (n_points,)
        grid_nodes: 稀疏网格节点 (4, n_points)
        match_func_params: 匹配函数参数
        sigma_zero: 控制变量零向量
        theta: 市场紧张度
        mu: 离职率
        gamma_*: 状态转移参数
        k_neighbors: 插值近邻数
    
    Returns:
        更新后的失业者分布 (n_points,)
    """
    n_points = len(m_U_current)
    m_U_new = np.zeros(n_points, dtype=np.float64)
    
    # 第1部分：失业者保持失业（未匹配）
    for i in range(n_points):
        if m_U_current[i] < 1e-12:  # 跳过空网格点
            continue
        
        x = grid_nodes[:, i]
        a = a_star[i]
        
        # 计算匹配概率
        lambda_val = compute_match_probability_numba(
            x, sigma_zero, a, theta, match_func_params
        )
        
        # 状态转移
        x_next = state_transition_full(
            x, a, gamma_T, gamma_S, gamma_D, gamma_W, T_max, W_min
        )
        
        # 未匹配的人口流向x_next附近的网格点
        # 使用反距离加权分配
        stay_prob = 1.0 - lambda_val
        flow_mass = stay_prob * m_U_current[i]
        
        # 找到最近的网格点（简化：只分配到最近点）
        min_dist_sq = np.inf
        nearest_idx = i
        
        for j in range(n_points):
            dist_sq = 0.0
            for d in range(4):
                diff = x_next[d] - grid_nodes[d, j]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_idx = j
        
        m_U_new[nearest_idx] += flow_mass
    
    # 第2部分：就业者离职流入失业
    for i in range(n_points):
        if m_E_current[i] < 1e-12:
            continue
        
        # 离职人口（状态保持不变，简化处理）
        m_U_new[i] += mu * m_E_current[i]
    
    return m_U_new


@njit
def evolve_employment_distribution(
    m_U_current: np.ndarray,
    m_E_current: np.ndarray,
    a_star: np.ndarray,
    grid_nodes: np.ndarray,
    match_func_params: np.ndarray,
    sigma_zero: np.ndarray,
    theta: float,
    mu: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float,
    T_max: float,
    W_min: float,
    k_neighbors: int = 16
) -> np.ndarray:
    """
    更新就业者人口分布（Numba优化）
    
    m^E(x_{t+1}) = Σ[λ*I*m^U] + (1-μ)*m^E
    
    Args:
        同 evolve_unemployment_distribution
    
    Returns:
        更新后的就业者分布 (n_points,)
    """
    n_points = len(m_U_current)
    m_E_new = np.zeros(n_points, dtype=np.float64)
    
    # 第1部分：失业者匹配成功
    for i in range(n_points):
        if m_U_current[i] < 1e-12:
            continue
        
        x = grid_nodes[:, i]
        a = a_star[i]
        
        # 计算匹配概率
        lambda_val = compute_match_probability_numba(
            x, sigma_zero, a, theta, match_func_params
        )
        
        # 状态转移
        x_next = state_transition_full(
            x, a, gamma_T, gamma_S, gamma_D, gamma_W, T_max, W_min
        )
        
        # 匹配成功的人口流向x_next
        match_prob = lambda_val
        flow_mass = match_prob * m_U_current[i]
        
        # 分配到最近网格点
        min_dist_sq = np.inf
        nearest_idx = i
        
        for j in range(n_points):
            dist_sq = 0.0
            for d in range(4):
                diff = x_next[d] - grid_nodes[d, j]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_idx = j
        
        m_E_new[nearest_idx] += flow_mass
    
    # 第2部分：就业者保持就业
    for i in range(n_points):
        if m_E_current[i] < 1e-12:
            continue
        
        # 状态转移（就业者a=0）
        x = grid_nodes[:, i]
        x_next = state_transition_full(
            x, 0.0, gamma_T, gamma_S, gamma_D, gamma_W, T_max, W_min
        )
        
        # 保持就业的人口
        stay_mass = (1.0 - mu) * m_E_current[i]
        
        # 分配到最近网格点
        min_dist_sq = np.inf
        nearest_idx = i
        
        for j in range(n_points):
            dist_sq = 0.0
            for d in range(4):
                diff = x_next[d] - grid_nodes[d, j]
                dist_sq += diff * diff
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_idx = j
        
        m_E_new[nearest_idx] += stay_mass
    
    return m_E_new


class KFESolver:
    """
    Kolmogorov前向方程求解器
    
    管理人口分布的演化，与贝尔曼求解器耦合。
    
    Attributes:
        grid_nodes: 稀疏网格节点 (4, n_points)
        n_points: 网格点数
        m_U: 失业者分布 (n_points,)
        m_E: 就业者分布 (n_points,)
        config: 配置参数
    """
    
    def __init__(
        self,
        grid_nodes: np.ndarray,
        config: Dict,
        match_func_params: Optional[np.ndarray] = None,
        initial_unemployment_rate: float = 0.2
    ):
        """
        初始化KFE求解器
        
        Args:
            grid_nodes: 稀疏网格节点 (4, n_points)
            config: 配置字典
            match_func_params: 匹配函数参数
            initial_unemployment_rate: 初始失业率
        """
        self.grid_nodes = grid_nodes
        self.dimension = grid_nodes.shape[0]
        self.n_points = grid_nodes.shape[1]
        self.config = config
        
        # 参数
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
            self.match_func_params = np.zeros(11, dtype=np.float64)
            logger.warning("未提供匹配函数参数，使用默认值")
        
        self.sigma_zero = np.zeros(4, dtype=np.float64)
        
        # 插值参数
        self.k_neighbors = 16
        
        # 初始化人口分布（均匀分布在网格点上）
        total_population = 1.0  # 归一化
        self.m_U = np.ones(self.n_points, dtype=np.float64) * (
            initial_unemployment_rate / self.n_points
        )
        self.m_E = np.ones(self.n_points, dtype=np.float64) * (
            (1.0 - initial_unemployment_rate) / self.n_points
        )
        
        logger.info(
            f"初始化KFE求解器：{self.n_points}个网格点，"
            f"初始失业率={initial_unemployment_rate:.1%}"
        )
    
    def step(
        self,
        a_star: np.ndarray,
        n_steps: int = 1
    ) -> Dict:
        """
        执行KFE演化步骤
        
        Args:
            a_star: 最优努力策略（在网格点上） (n_points,)
            n_steps: 演化步数
        
        Returns:
            包含演化结果的字典
        """
        for _ in range(n_steps):
            # 更新失业者分布
            m_U_new = evolve_unemployment_distribution(
                self.m_U, self.m_E, a_star,
                self.grid_nodes,
                self.match_func_params,
                self.sigma_zero,
                self.theta,
                self.mu,
                self.gamma_T, self.gamma_S, self.gamma_D, self.gamma_W,
                self.T_max, self.W_min,
                self.k_neighbors
            )
            
            # 更新就业者分布
            m_E_new = evolve_employment_distribution(
                self.m_U, self.m_E, a_star,
                self.grid_nodes,
                self.match_func_params,
                self.sigma_zero,
                self.theta,
                self.mu,
                self.gamma_T, self.gamma_S, self.gamma_D, self.gamma_W,
                self.T_max, self.W_min,
                self.k_neighbors
            )
            
            # 归一化（保证总人口=1）
            total = np.sum(m_U_new) + np.sum(m_E_new)
            if total > 1e-12:
                m_U_new = m_U_new / total
                m_E_new = m_E_new / total
            
            # 更新
            self.m_U = m_U_new
            self.m_E = m_E_new
        
        # 计算失业率
        unemployment_rate = np.sum(self.m_U)
        employment_rate = np.sum(self.m_E)
        
        return {
            'm_U': self.m_U.copy(),
            'm_E': self.m_E.copy(),
            'unemployment_rate': unemployment_rate,
            'employment_rate': employment_rate
        }
    
    def get_unemployment_rate(self) -> float:
        """获取当前失业率"""
        return np.sum(self.m_U)
    
    def get_employment_rate(self) -> float:
        """获取当前就业率"""
        return np.sum(self.m_E)
    
    def reset(self, initial_unemployment_rate: float = 0.2):
        """重置人口分布"""
        total_population = 1.0
        self.m_U = np.ones(self.n_points, dtype=np.float64) * (
            initial_unemployment_rate / self.n_points
        )
        self.m_E = np.ones(self.n_points, dtype=np.float64) * (
            (1.0 - initial_unemployment_rate) / self.n_points
        )
        logger.info(f"KFE求解器已重置，失业率={initial_unemployment_rate:.1%}")
    
    def __repr__(self) -> str:
        u_rate = self.get_unemployment_rate()
        e_rate = self.get_employment_rate()
        return (
            f"KFESolver(n_points={self.n_points}, "
            f"unemployment_rate={u_rate:.2%}, "
            f"employment_rate={e_rate:.2%})"
        )


# 测试
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from modules.mfg.state_transition import state_transition_full
    from modules.estimation.match_function import compute_match_probability_numba
    
    print("=" * 60)
    print("测试KFE求解器模块")
    print("=" * 60)
    
    # 创建测试配置
    config = {
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
    
    # 创建测试网格
    np.random.seed(42)
    n_test_points = 50
    
    T_samples = np.random.uniform(15, 70, n_test_points)
    S_norm_samples = np.random.uniform(0, 1, n_test_points)
    D_norm_samples = np.random.uniform(0, 1, n_test_points)
    W_samples = np.random.uniform(1400, 8000, n_test_points)
    
    grid_nodes = np.vstack([T_samples, S_norm_samples, D_norm_samples, W_samples])
    
    # 初始化求解器
    kfe_solver = KFESolver(
        grid_nodes, config,
        match_func_params=None,
        initial_unemployment_rate=0.2
    )
    print(f"\n初始化: {kfe_solver}")
    
    # 模拟最优策略（随机）
    a_star = np.random.uniform(0, 0.5, n_test_points)
    
    # 执行演化
    print("\n执行KFE演化...")
    for t in range(10):
        result = kfe_solver.step(a_star, n_steps=1)
        if t % 2 == 0:
            print(
                f"  t={t}: "
                f"失业率={result['unemployment_rate']:.4f}, "
                f"就业率={result['employment_rate']:.4f}"
            )
    
    print(f"\n最终状态: {kfe_solver}")
    
    print("\n" + "=" * 60)
    print("✅ KFE求解器测试完成！")
    print("=" * 60)

