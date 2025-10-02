"""
Matching模块

实现劳动力市场的双边匹配算法，基于Gale-Shapley稳定匹配理论。
"""

from .preference import (
    compute_labor_preference_matrix,
    compute_enterprise_preference_matrix
)
from .gale_shapley import gale_shapley, verify_stability, limited_rounds_matching
from .matching_result import MatchingResult
from .matching_engine import MatchingEngine
from .abm_data_generator import ABMDataGenerator

__all__ = [
    'compute_labor_preference_matrix',
    'compute_enterprise_preference_matrix',
    'gale_shapley',
    'limited_rounds_matching',
    'verify_stability',
    'MatchingResult',
    'MatchingEngine',
    'ABMDataGenerator'
]
