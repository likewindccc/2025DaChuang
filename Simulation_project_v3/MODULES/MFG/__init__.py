"""
MFG模块 - 平均场博弈均衡求解

包含：
- bellman_solver: 贝尔曼方程求解器（值迭代）
- kfe_solver: KFE演化求解器（人口分布演化）
- equilibrium_solver: 均衡求解器（主控制，交替迭代）
"""

from .bellman_solver import BellmanSolver, load_match_function_model
from .kfe_solver import KFESolver
from .equilibrium_solver import EquilibriumSolver, solve_equilibrium

__all__ = [
    'BellmanSolver',
    'KFESolver',
    'EquilibriumSolver',
    'load_match_function_model',
    'solve_equilibrium'
]

