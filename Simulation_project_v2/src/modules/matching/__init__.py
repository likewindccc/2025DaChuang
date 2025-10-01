"""
Matching模块

实现劳动力市场的双边匹配算法，基于Gale-Shapley稳定匹配理论。
"""

from .preference import (
    compute_labor_preference_matrix,
    compute_enterprise_preference_matrix
)
from .gale_shapley import gale_shapley, verify_stability
from .matching_result import MatchingResult

__all__ = [
    'compute_labor_preference_matrix',
    'compute_enterprise_preference_matrix',
    'gale_shapley',
    'verify_stability',
    'MatchingResult'
]
