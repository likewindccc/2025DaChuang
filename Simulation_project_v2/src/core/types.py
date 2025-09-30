#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core类型定义模块

定义项目中通用的类型别名，提供更清晰的类型提示和代码可读性。

类型分类：
- 基础类型：ID、时间步、参数字典
- NumPy数组类型：特征矩阵、偏好矩阵、值函数、分布等
- 函数类型：目标函数、匹配函数等
- 配置类型：配置字典

使用示例：
    >>> from src.core.types import AgentFeatures, PreferenceMatrix
    >>> 
    >>> def compute_preference(
    ...     labor_features: AgentFeatures,
    ...     enterprise_features: AgentFeatures
    ... ) -> PreferenceMatrix:
    ...     # 实现...
    ...     pass
"""

from typing import Dict, List, Tuple, Callable, Union, Any
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# 基础类型别名
# ============================================================================

AgentID = int
"""个体唯一标识符（整数）"""

TimeStep = int
"""时间步（整数）"""

ParameterDict = Dict[str, float]
"""参数字典，键为参数名，值为参数值"""


# ============================================================================
# NumPy数组类型别名
# ============================================================================

AgentFeatures = NDArray[np.float64]
"""
个体特征数组

形状:
    - 单个个体: (n_features,)，通常为 (4,) 即 [T, S, D, W]
    - 多个个体: (n_agents, n_features)

示例:
    >>> labor_features: AgentFeatures = np.array([[40, 75, 60, 4500]])
    >>> enterprise_features: AgentFeatures = np.array([[48, 70, 65, 5000]])
"""

PreferenceMatrix = NDArray[np.float64]
"""
偏好矩阵

形状: (n_labor, n_enterprise)

元素 [i, j] 表示劳动力 i 对企业 j 的偏好值（或企业 j 对劳动力 i 的偏好值）

示例:
    >>> pref_matrix: PreferenceMatrix = np.random.rand(100, 80)
    >>> # 100个劳动力对80个企业的偏好
"""

ValueFunction = NDArray[np.float64]
"""
值函数

形状: (grid_size_T, grid_size_S)

V(T, S) 表示在状态 (T, S) 下的值函数值

示例:
    >>> V_U: ValueFunction = np.zeros((50, 50))  # 失业值函数
    >>> V_E: ValueFunction = np.ones((50, 50)) * 10  # 就业值函数
"""

PolicyFunction = NDArray[np.float64]
"""
策略函数

形状: (grid_size_T, grid_size_S)

a*(T, S) 表示在状态 (T, S) 下的最优策略（如努力水平）

示例:
    >>> policy: PolicyFunction = np.ones((50, 50)) * 0.5
    >>> # 所有状态下的努力水平为0.5
"""

Distribution = NDArray[np.float64]
"""
人口分布

形状: (grid_size_T, grid_size_S)

m(T, S) 表示在状态 (T, S) 的人口密度

应满足归一化条件: sum(m) ≈ 1

示例:
    >>> grid_size = (50, 50)
    >>> m: Distribution = np.ones(grid_size) / (grid_size[0] * grid_size[1])
    >>> # 均匀分布
"""

CovarianceMatrix = NDArray[np.float64]
"""
协方差矩阵

形状: (n_features, n_features)

用于Copula建模中的依赖结构

示例:
    >>> cov: CovarianceMatrix = np.eye(4)  # 4x4单位矩阵
"""

CorrelationMatrix = NDArray[np.float64]
"""
相关系数矩阵

形状: (n_features, n_features)

元素 [i, j] 表示特征 i 和特征 j 之间的相关系数

对角元素为1，非对角元素在[-1, 1]之间

示例:
    >>> corr: CorrelationMatrix = np.array([
    ...     [1.0, 0.5, 0.3, 0.2],
    ...     [0.5, 1.0, 0.4, 0.1],
    ...     [0.3, 0.4, 1.0, 0.6],
    ...     [0.2, 0.1, 0.6, 1.0]
    ... ])
"""


# ============================================================================
# 函数类型别名
# ============================================================================

ObjectiveFunction = Callable[[np.ndarray], float]
"""
目标函数

输入: 参数数组（形状任意）
输出: 目标函数值（标量）

用于优化算法（如遗传算法）

示例:
    >>> def my_objective(params: np.ndarray) -> float:
    ...     return np.sum(params ** 2)
    >>> 
    >>> obj_func: ObjectiveFunction = my_objective
"""

MatchFunction = Callable[
    [np.ndarray, np.ndarray, float, float],
    float
]
"""
匹配函数

输入:
    - x: 个体特征差异（形状任意）
    - sigma: 匹配效率参数
    - a: 努力水平
    - theta: 市场紧张度
输出:
    - 匹配概率（0-1之间的浮点数）

示例:
    >>> def cobb_douglas_matching(
    ...     x: np.ndarray,
    ...     sigma: float,
    ...     a: float,
    ...     theta: float
    ... ) -> float:
    ...     distance = np.linalg.norm(x)
    ...     return sigma * (a ** 0.5) * (theta ** 0.3) * np.exp(-distance)
    >>> 
    >>> match_func: MatchFunction = cobb_douglas_matching
"""

UtilityFunction = Callable[[float, float, float], float]
"""
效用函数

输入:
    - consumption: 消费水平
    - effort: 努力水平
    - leisure: 闲暇时间
输出:
    - 效用值

示例:
    >>> def cobb_douglas_utility(c: float, a: float, l: float) -> float:
    ...     return (c ** 0.5) * (l ** 0.3) - (a ** 2) / 2
    >>> 
    >>> utility_func: UtilityFunction = cobb_douglas_utility
"""


# ============================================================================
# 配置类型别名
# ============================================================================

Config = Dict[str, Union[int, float, str, bool, List, Dict]]
"""
配置字典

键: 配置参数名（字符串）
值: 参数值（支持多种类型）

示例:
    >>> config: Config = {
    ...     'grid_size': 50,
    ...     'max_iterations': 1000,
    ...     'tolerance': 1e-6,
    ...     'use_numba': True,
    ...     'output_dir': 'results/',
    ...     'distributions': ['beta', 'beta', 'beta', 'beta']
    ... }
"""

ExperimentConfig = Dict[str, Config]
"""
实验配置字典

键: 实验名称
值: 该实验的配置字典

示例:
    >>> exp_config: ExperimentConfig = {
    ...     'baseline': {
    ...         'grid_size': 50,
    ...         'max_iterations': 1000
    ...     },
    ...     'sensitivity_analysis': {
    ...         'grid_size': 100,
    ...         'max_iterations': 2000
    ...     }
    ... }
"""


# ============================================================================
# 复合类型别名
# ============================================================================

AgentPool = List[Dict[str, Any]]
"""
个体池

元素: 个体属性字典

示例:
    >>> agent_pool: AgentPool = [
    ...     {'agent_id': 1, 'T': 40, 'S': 75, 'D': 60, 'W': 4500},
    ...     {'agent_id': 2, 'T': 45, 'S': 80, 'D': 70, 'W': 5000}
    ... ]
"""

MatchingResult = Tuple[List[Tuple[int, int]], float]
"""
匹配结果

元素:
    - 匹配对列表: [(labor_id, enterprise_id), ...]
    - 总匹配质量: 浮点数

示例:
    >>> result: MatchingResult = (
    ...     [(1, 101), (2, 102), (3, 103)],
    ...     0.85
    ... )
"""

SimulationMetrics = Dict[str, Union[float, List[float], np.ndarray]]
"""
模拟指标字典

键: 指标名称
值: 指标值（标量、列表或数组）

示例:
    >>> metrics: SimulationMetrics = {
    ...     'unemployment_rate': 0.1,
    ...     'avg_wage': 4800.0,
    ...     'theta_history': [0.8, 0.9, 1.0, 1.1],
    ...     'wage_distribution': np.array([4000, 4500, 5000, 5500])
    ... }
"""


# ============================================================================
# 类型检查辅助函数
# ============================================================================

def is_valid_agent_features(arr: np.ndarray) -> bool:
    """
    检查是否为有效的AgentFeatures
    
    Args:
        arr: 待检查的数组
    
    Returns:
        是否有效
    
    Examples:
        >>> arr = np.array([40, 75, 60, 4500])
        >>> is_valid_agent_features(arr)
        True
        >>> 
        >>> arr2 = np.array([40, 75])  # 维度不对
        >>> is_valid_agent_features(arr2)
        False
    """
    if arr.ndim == 1:
        return arr.shape[0] == 4
    elif arr.ndim == 2:
        return arr.shape[1] == 4
    return False


def is_valid_distribution(arr: np.ndarray, tolerance: float = 1e-3) -> bool:
    """
    检查是否为有效的Distribution（归一化）
    
    Args:
        arr: 待检查的分布数组
        tolerance: 归一化误差容忍度
    
    Returns:
        是否有效
    
    Examples:
        >>> dist = np.ones((50, 50)) / 2500
        >>> is_valid_distribution(dist)
        True
        >>> 
        >>> invalid_dist = np.ones((50, 50))  # 和不为1
        >>> is_valid_distribution(invalid_dist)
        False
    """
    total = np.sum(arr)
    return abs(total - 1.0) < tolerance


def is_symmetric_matrix(arr: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    检查是否为对称矩阵
    
    Args:
        arr: 待检查的矩阵
        tolerance: 对称性误差容忍度
    
    Returns:
        是否对称
    
    Examples:
        >>> corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> is_symmetric_matrix(corr)
        True
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    return np.allclose(arr, arr.T, atol=tolerance)
