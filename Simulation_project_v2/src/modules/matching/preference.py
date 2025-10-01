"""
偏好计算模块

基于原始研究计划的偏好函数：
- 劳动力偏好：P_ij = γ₀ - γ₁Tⱼ - γ₂max(0, Sⱼ-Sᵢ) - γ₃max(0, Dⱼ-Dᵢ) + γ₄Wⱼ
- 企业偏好：P_ji = β₀ + β₁Tᵢ + β₂Sᵢ + β₃Dᵢ + β₄Wᵢ（β₄<0）
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(parallel=True, fastmath=True, cache=True)
def compute_labor_preference_matrix(
    labor_features: np.ndarray,
    enterprise_features: np.ndarray,
    gamma_0: float = 1.0,
    gamma_1: float = 0.01,
    gamma_2: float = 0.5,
    gamma_3: float = 0.5,
    gamma_4: float = 0.001
) -> np.ndarray:
    """
    计算劳动力对所有企业的偏好矩阵（基于原始研究计划公式）
    
    P_ij^jobseeker = γ₀ - γ₁Tⱼ - γ₂max(0, Sⱼ-Sᵢ) - γ₃max(0, Dⱼ-Dᵢ) + γ₄Wⱼ
    
    经济学含义：
    - 工作时长越长，劳动力偏好越低（-γ₁Tⱼ）
    - 岗位技能/数字素养要求超过自身时有负面影响（max(0,·)不对称）
    - 工资越高，偏好越高（+γ₄Wⱼ）
    
    Args:
        labor_features: (n_labor, 4) 劳动力特征 [T, S, D, W]
        enterprise_features: (n_enterprise, 4) 企业特征 [T, S, D, W]
        gamma_0: 截距项（基准偏好）
        gamma_1: 工作时长负面系数
        gamma_2: 技能要求超过惩罚系数
        gamma_3: 数字素养要求超过惩罚系数
        gamma_4: 工资正面权重
    
    Returns:
        preference: (n_labor, n_enterprise) 偏好分数矩阵（越高越好）
    """
    n_labor = labor_features.shape[0]
    n_enterprise = enterprise_features.shape[0]
    
    preference = np.zeros((n_labor, n_enterprise), dtype=np.float32)
    
    for i in prange(n_labor):
        labor_T, labor_S, labor_D, labor_W = labor_features[i]
        
        for j in range(n_enterprise):
            ent_T, ent_S, ent_D, ent_W = enterprise_features[j]
            
            # 基准偏好
            score = gamma_0
            
            # 工作时长负面影响（时长越长越不喜欢）
            score -= gamma_1 * ent_T
            
            # 技能要求超过自身水平的惩罚（不对称）
            skill_gap = max(0.0, ent_S - labor_S)
            score -= gamma_2 * skill_gap
            
            # 数字素养要求超过自身的惩罚（不对称）
            digital_gap = max(0.0, ent_D - labor_D)
            score -= gamma_3 * digital_gap
            
            # 工资正面影响
            score += gamma_4 * ent_W
            
            preference[i, j] = score
    
    return preference


@njit(parallel=True, fastmath=True, cache=True)
def compute_enterprise_preference_matrix(
    enterprise_features: np.ndarray,
    labor_features: np.ndarray,
    beta_0: float = 0.0,
    beta_1: float = 0.5,
    beta_2: float = 1.0,
    beta_3: float = 1.0,
    beta_4: float = -0.001
) -> np.ndarray:
    """
    计算企业对所有劳动力的偏好矩阵（基于原始研究计划公式）
    
    P_ji^employer = β₀ + β₁Tᵢ + β₂Sᵢ + β₃Dᵢ + β₄Wᵢ
    
    其中β₄为负数，体现企业的成本控制意识：
    - 期望工资越高 → β₄Wᵢ越负 → 企业偏好越低
    - 期望工资越低 → β₄Wᵢ接近0 → 企业偏好越高
    
    经济学含义：
    - 可供工作时间越长，企业越偏好（+β₁Tᵢ）
    - 技能水平越高，企业越偏好（+β₂Sᵢ）
    - 数字素养越高，企业越偏好（+β₃Dᵢ）
    - 期望工资越高，企业越不喜欢（β₄<0，降本增效）
    
    Args:
        enterprise_features: (n_enterprise, 4) 企业特征 [T, S, D, W]
                            （注：在此公式中不直接使用，但保留以保持接口一致）
        labor_features: (n_labor, 4) 劳动力特征 [T, S, D, W]
        beta_0: 截距项（基准偏好）
        beta_1: 工作时间权重（正数）
        beta_2: 技能水平权重（正数）
        beta_3: 数字素养权重（正数）
        beta_4: 期望工资权重（负数：降本增效）
    
    Returns:
        preference: (n_enterprise, n_labor) 偏好分数矩阵（越高越好）
    """
    n_enterprise = enterprise_features.shape[0]
    n_labor = labor_features.shape[0]
    
    preference = np.zeros((n_enterprise, n_labor), dtype=np.float32)
    
    for j in prange(n_enterprise):
        # 注：企业特征在此公式中不直接使用，但保留参数以保持接口一致
        
        for i in range(n_labor):
            labor_T, labor_S, labor_D, labor_W = labor_features[i]
            
            # 简单线性加权
            score = beta_0
            score += beta_1 * labor_T      # 工作时间越长越好
            score += beta_2 * labor_S      # 技能水平越高越好
            score += beta_3 * labor_D      # 数字素养越高越好
            score += beta_4 * labor_W      # 期望工资（β₄<0，工资越高越不好）
            
            preference[j, i] = score
    
    return preference


def compute_preference_rankings(
    preference_matrix: np.ndarray
) -> np.ndarray:
    """
    将偏好分数矩阵转换为偏好排序矩阵
    
    Args:
        preference_matrix: (n, m) 偏好分数矩阵
    
    Returns:
        ranking: (n, m) 偏好排序索引（降序，0表示最偏好）
    """
    # 使用argsort降序排序（-preference_matrix）
    # 返回的是每一行中，按偏好从高到低的列索引
    return np.argsort(-preference_matrix, axis=1).astype(np.int32)

