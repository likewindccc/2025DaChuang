"""
效用函数模块

实现失业者和就业者的瞬时效用函数，使用Numba进行JIT编译加速。

根据研究计划第4.1.1节：
- 失业者效用: u^U(x, a) = b_0 - 0.5 * κ * a^2
- 就业者效用: u^E(x) = W - α_T * T

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@njit
def utility_unemployment(
    a: float,
    b_0: float = 500.0,
    kappa: float = 1.0
) -> float:
    """
    失业者瞬时效用函数（Numba优化）
    
    u^U(a) = b_0 - 0.5 * κ * a^2
    
    Args:
        a: 努力水平，a ∈ [0, 1]
        b_0: 失业补助（元/月），默认500
        kappa: 努力成本系数，默认1.0
    
    Returns:
        失业者的瞬时效用
    
    Examples:
        >>> utility_unemployment(0.0, 500.0, 1.0)
        500.0
        >>> utility_unemployment(0.5, 500.0, 1.0)
        499.875
        >>> utility_unemployment(1.0, 500.0, 1.0)
        499.5
    
    Notes:
        - 努力成本项为 0.5 * κ * a^2，体现边际成本递增
        - b_0 > 0 表示有失业救济或家庭生产价值
        - κ 越大，努力的成本越高
    """
    return b_0 - 0.5 * kappa * a * a


@njit
def utility_employment(
    W: float,
    T: float,
    alpha_T: float = 10.0
) -> float:
    """
    就业者瞬时效用函数（Numba优化）
    
    u^E(W, T) = W - α_T * T
    
    Args:
        W: 期望工资（元/月）
        T: 每周工作小时数
        alpha_T: 工作负效用系数（元/(小时/周)），默认10.0
    
    Returns:
        就业者的瞬时效用
    
    Examples:
        >>> utility_employment(4500.0, 45.0, 10.0)
        4050.0
        >>> utility_employment(4500.0, 70.0, 10.0)
        3800.0
    
    Notes:
        - W 是工资收益
        - α_T * T 是工作负效用（疲劳、时间机会成本等）
        - α_T 越大，长时间工作的负效用越高
    """
    return W - alpha_T * T


@njit
def utility_unemployment_batch(
    a: np.ndarray,
    b_0: float = 500.0,
    kappa: float = 1.0
) -> np.ndarray:
    """
    批量计算失业者效用（Numba优化）
    
    Args:
        a: 努力水平数组 (n,)
        b_0: 失业补助
        kappa: 努力成本系数
    
    Returns:
        失业者效用数组 (n,)
    """
    n = len(a)
    utilities = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        utilities[i] = utility_unemployment(a[i], b_0, kappa)
    
    return utilities


@njit
def utility_employment_batch(
    W: np.ndarray,
    T: np.ndarray,
    alpha_T: float = 10.0
) -> np.ndarray:
    """
    批量计算就业者效用（Numba优化）
    
    Args:
        W: 工资数组 (n,)
        T: 工作时长数组 (n,)
        alpha_T: 工作负效用系数
    
    Returns:
        就业者效用数组 (n,)
    """
    n = len(W)
    utilities = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        utilities[i] = utility_employment(W[i], T[i], alpha_T)
    
    return utilities


@njit
def effort_cost(a: float, kappa: float = 1.0) -> float:
    """
    努力成本函数（Numba优化）
    
    Cost(a) = 0.5 * κ * a^2
    
    Args:
        a: 努力水平
        kappa: 努力成本系数
    
    Returns:
        努力成本
    
    Examples:
        >>> effort_cost(0.5, 1.0)
        0.125
        >>> effort_cost(1.0, 1.0)
        0.5
    """
    return 0.5 * kappa * a * a


@njit
def employment_value_difference(
    W: float,
    T: float,
    a: float,
    b_0: float = 500.0,
    kappa: float = 1.0,
    alpha_T: float = 10.0
) -> float:
    """
    就业与失业的瞬时效用差（Numba优化）
    
    Δu = u^E - u^U = (W - α_T*T) - (b_0 - 0.5*κ*a^2)
    
    Args:
        W: 工资
        T: 工作时长
        a: 失业时的努力水平
        b_0: 失业补助
        kappa: 努力成本系数
        alpha_T: 工作负效用系数
    
    Returns:
        就业与失业的效用差
    
    Notes:
        - Δu > 0: 就业带来更高的瞬时效用
        - Δu < 0: 失业（含失业补助）效用更高（通常不应出现）
        - 用于判断就业的吸引力
    """
    u_E = utility_employment(W, T, alpha_T)
    u_U = utility_unemployment(a, b_0, kappa)
    return u_E - u_U


class UtilityConfig:
    """
    效用函数配置类
    
    封装所有效用函数相关的参数，便于配置管理和传递。
    
    Attributes:
        b_0: 失业补助（元/月）
        kappa: 努力成本系数
        alpha_T: 工作负效用系数（元/(小时/周)）
    """
    
    def __init__(
        self,
        b_0: float = 500.0,
        kappa: float = 1.0,
        alpha_T: float = 10.0
    ):
        """
        初始化效用函数配置
        
        Args:
            b_0: 失业补助，默认500元/月
            kappa: 努力成本系数，默认1.0
            alpha_T: 工作负效用系数，默认10.0
        """
        self.b_0 = b_0
        self.kappa = kappa
        self.alpha_T = alpha_T
        
        logger.info(
            f"效用函数配置: b_0={b_0}, κ={kappa}, α_T={alpha_T}"
        )
    
    def get_unemployment_utility(self, a: float) -> float:
        """
        计算失业者效用
        
        Args:
            a: 努力水平
        
        Returns:
            失业者效用
        """
        return utility_unemployment(a, self.b_0, self.kappa)
    
    def get_employment_utility(self, W: float, T: float) -> float:
        """
        计算就业者效用
        
        Args:
            W: 工资
            T: 工作时长
        
        Returns:
            就业者效用
        """
        return utility_employment(W, T, self.alpha_T)
    
    def get_parameters(self) -> Tuple[float, float, float]:
        """
        获取所有参数
        
        Returns:
            (b_0, kappa, alpha_T)
        """
        return self.b_0, self.kappa, self.alpha_T
    
    def __repr__(self) -> str:
        return (
            f"UtilityConfig(b_0={self.b_0}, "
            f"kappa={self.kappa}, alpha_T={self.alpha_T})"
        )


def validate_utility_parameters(
    b_0: float,
    kappa: float,
    alpha_T: float
) -> None:
    """
    验证效用函数参数的合理性
    
    Args:
        b_0: 失业补助
        kappa: 努力成本系数
        alpha_T: 工作负效用系数
    
    Raises:
        ValueError: 参数不合理
    """
    if kappa <= 0:
        raise ValueError(f"努力成本系数κ必须为正，得到：{kappa}")
    
    if alpha_T < 0:
        raise ValueError(f"工作负效用系数α_T不能为负，得到：{alpha_T}")
    
    if b_0 < 0:
        logger.warning(f"失业补助b_0为负值：{b_0}，这在经济学上不常见")
    
    logger.info("效用函数参数验证通过")


# 预编译Numba函数（首次导入时触发编译）
if __name__ == "__main__":
    # 触发Numba编译
    _ = utility_unemployment(0.5, 500.0, 1.0)
    _ = utility_employment(4500.0, 45.0, 10.0)
    
    print("效用函数模块Numba编译完成")
    print(f"失业者效用（a=0.5）: {utility_unemployment(0.5, 500.0, 1.0):.4f}")
    print(f"就业者效用（W=4500, T=45）: {utility_employment(4500.0, 45.0, 10.0):.4f}")
    print(f"效用差: {employment_value_difference(4500.0, 45.0, 0.5, 500.0, 1.0, 10.0):.4f}")

