"""
状态转移模块

实现4个状态变量的转移公式，基于研究计划第4.3节的设计。

状态转移方程：
1. T_{t+1} = T_t + γ_T * a * (T_max - T_t)
2. S_norm_{t+1} = S_norm_t + γ_S * a * (1 - S_norm_t)
3. D_norm_{t+1} = D_norm_t + γ_D * a * (1 - D_norm_t)
4. W_{t+1} = max(W_min, W_t - γ_W * a)

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@njit
def state_transition_T(
    T: float,
    a: float,
    gamma_T: float,
    T_max: float
) -> float:
    """
    工作时间投入增加公式（Numba优化）
    
    T_{t+1} = T_t + γ_T * a * (T_max - T_t)
    
    Args:
        T: 当前工作时长（小时/周）
        a: 努力水平，a ∈ [0, 1]
        gamma_T: 工作时长增长率
        T_max: 最大工作时长上限
    
    Returns:
        下一期工作时长
    
    Examples:
        >>> state_transition_T(45.0, 0.5, 0.1, 70.0)
        46.25  # 45 + 0.1 * 0.5 * (70 - 45) = 45 + 1.25
    
    Notes:
        - 越靠近T_max，增长空间越小（边际递减）
        - a=0时，T不变
        - a=1时，T增长最快
    """
    return T + gamma_T * a * (T_max - T)


@njit
def state_transition_S(
    S_norm: float,
    a: float,
    gamma_S: float
) -> float:
    """
    工作能力提升公式（Numba优化，标准化尺度）
    
    S_norm_{t+1} = S_norm_t + γ_S * a * (1 - S_norm_t)
    
    Args:
        S_norm: 当前工作能力（标准化，∈ [0, 1]）
        a: 努力水平
        gamma_S: 技能增长率
    
    Returns:
        下一期工作能力（标准化）
    
    Examples:
        >>> state_transition_S(0.5, 0.5, 0.05)
        0.5125  # 0.5 + 0.05 * 0.5 * (1 - 0.5) = 0.5 + 0.0125
    
    Notes:
        - S_norm ∈ [0, 1]，1表示能力达到最大值
        - 越接近1，提升空间越小（饱和效应）
        - 能力提升存在边际递减
    """
    return S_norm + gamma_S * a * (1.0 - S_norm)


@njit
def state_transition_D(
    D_norm: float,
    a: float,
    gamma_D: float
) -> float:
    """
    数字素养提升公式（Numba优化，标准化尺度）
    
    D_norm_{t+1} = D_norm_t + γ_D * a * (1 - D_norm_t)
    
    Args:
        D_norm: 当前数字素养（标准化，∈ [0, 1]）
        a: 努力水平
        gamma_D: 数字素养增长率
    
    Returns:
        下一期数字素养（标准化）
    
    Examples:
        >>> state_transition_D(0.4, 0.5, 0.08)
        0.424  # 0.4 + 0.08 * 0.5 * (1 - 0.4) = 0.4 + 0.024
    
    Notes:
        - D_norm ∈ [0, 1]，1表示数字素养达到最大值
        - 与工作能力类似，存在饱和效应
    """
    return D_norm + gamma_D * a * (1.0 - D_norm)


@njit
def state_transition_W(
    W: float,
    a: float,
    gamma_W: float,
    W_min: float
) -> float:
    """
    期望工作待遇调整公式（Numba优化）
    
    W_{t+1} = max(W_min, W_t - γ_W * a)
    
    Args:
        W: 当前期望工资（元/月）
        a: 努力水平
        gamma_W: 工资期望下降速率
        W_min: 最低工资期望（保障最低生活需求）
    
    Returns:
        下一期期望工资
    
    Examples:
        >>> state_transition_W(4500.0, 0.5, 100.0, 1400.0)
        4450.0  # max(1400, 4500 - 100*0.5) = max(1400, 4450) = 4450
        >>> state_transition_W(1500.0, 1.0, 100.0, 1400.0)
        1400.0  # max(1400, 1500 - 100) = max(1400, 1400) = 1400
    
    Notes:
        - 努力增加会主动降低工资期望，以适应市场
        - W_min 保障最低生活需求，不会无限降低
        - γ_W 越大，期望下降越快
    """
    W_new = W - gamma_W * a
    return max(W_min, W_new)


@njit
def state_transition_full(
    x: np.ndarray,
    a: float,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float,
    T_max: float,
    W_min: float
) -> np.ndarray:
    """
    完整的4维状态转移（Numba优化）
    
    Args:
        x: 当前状态向量 (4,): [T, S_norm, D_norm, W]
        a: 努力水平
        gamma_T, gamma_S, gamma_D, gamma_W: 转移速率参数
        T_max: 最大工作时长
        W_min: 最低工资期望
    
    Returns:
        下一期状态向量 (4,): [T', S_norm', D_norm', W']
    
    Examples:
        >>> x = np.array([45.0, 0.5, 0.4, 4500.0])
        >>> x_next = state_transition_full(x, 0.5, 0.1, 0.05, 0.08, 100.0, 70.0, 1400.0)
        >>> # x_next ≈ [46.25, 0.5125, 0.424, 4450.0]
    """
    T = x[0]
    S_norm = x[1]
    D_norm = x[2]
    W = x[3]
    
    T_next = state_transition_T(T, a, gamma_T, T_max)
    S_norm_next = state_transition_S(S_norm, a, gamma_S)
    D_norm_next = state_transition_D(D_norm, a, gamma_D)
    W_next = state_transition_W(W, a, gamma_W, W_min)
    
    return np.array([T_next, S_norm_next, D_norm_next, W_next])


@njit
def state_transition_batch(
    X: np.ndarray,
    a: np.ndarray,
    gamma_T: float,
    gamma_S: float,
    gamma_D: float,
    gamma_W: float,
    T_max: float,
    W_min: float
) -> np.ndarray:
    """
    批量状态转移（Numba并行优化）
    
    Args:
        X: 状态矩阵 (n, 4): [[T, S_norm, D_norm, W], ...]
        a: 努力水平数组 (n,)
        gamma_*: 转移速率参数
        T_max: 最大工作时长
        W_min: 最低工资期望
    
    Returns:
        下一期状态矩阵 (n, 4)
    """
    n = X.shape[0]
    X_next = np.empty((n, 4), dtype=np.float64)
    
    for i in range(n):
        X_next[i] = state_transition_full(
            X[i], a[i],
            gamma_T, gamma_S, gamma_D, gamma_W,
            T_max, W_min
        )
    
    return X_next


class StateTransitionConfig:
    """
    状态转移配置类
    
    封装所有状态转移相关的参数。
    
    Attributes:
        gamma_T: 工作时长增长率
        gamma_S: 技能增长率
        gamma_D: 数字素养增长率
        gamma_W: 工资期望下降速率
        T_max: 最大工作时长
        W_min: 最低工资期望
    """
    
    def __init__(
        self,
        gamma_T: float = 0.1,
        gamma_S: float = 0.05,
        gamma_D: float = 0.08,
        gamma_W: float = 100.0,
        T_max: float = 70.0,
        W_min: float = 1400.0
    ):
        """
        初始化状态转移配置
        
        Args:
            gamma_T: 工作时长增长率，默认0.1
            gamma_S: 技能增长率，默认0.05
            gamma_D: 数字素养增长率，默认0.08
            gamma_W: 工资期望下降速率，默认100.0
            T_max: 最大工作时长，默认70.0小时/周
            W_min: 最低工资期望，默认1400.0元/月
        """
        self.gamma_T = gamma_T
        self.gamma_S = gamma_S
        self.gamma_D = gamma_D
        self.gamma_W = gamma_W
        self.T_max = T_max
        self.W_min = W_min
        
        logger.info(
            f"状态转移配置: γ_T={gamma_T}, γ_S={gamma_S}, "
            f"γ_D={gamma_D}, γ_W={gamma_W}, "
            f"T_max={T_max}, W_min={W_min}"
        )
        
        self._validate()
    
    def _validate(self):
        """验证参数合理性"""
        if self.gamma_T < 0 or self.gamma_T > 1:
            raise ValueError(f"γ_T应在[0,1]之间，得到：{self.gamma_T}")
        
        if self.gamma_S < 0 or self.gamma_S > 1:
            raise ValueError(f"γ_S应在[0,1]之间，得到：{self.gamma_S}")
        
        if self.gamma_D < 0 or self.gamma_D > 1:
            raise ValueError(f"γ_D应在[0,1]之间，得到：{self.gamma_D}")
        
        if self.gamma_W < 0:
            raise ValueError(f"γ_W不能为负，得到：{self.gamma_W}")
        
        if self.T_max <= 0:
            raise ValueError(f"T_max必须为正，得到：{self.T_max}")
        
        if self.W_min < 0:
            raise ValueError(f"W_min不能为负，得到：{self.W_min}")
        
        logger.info("状态转移参数验证通过")
    
    def transition(self, x: np.ndarray, a: float) -> np.ndarray:
        """
        执行状态转移
        
        Args:
            x: 当前状态 (4,)
            a: 努力水平
        
        Returns:
            下一期状态 (4,)
        """
        return state_transition_full(
            x, a,
            self.gamma_T, self.gamma_S, self.gamma_D, self.gamma_W,
            self.T_max, self.W_min
        )
    
    def get_parameters(self) -> Tuple[float, float, float, float, float, float]:
        """
        获取所有参数
        
        Returns:
            (gamma_T, gamma_S, gamma_D, gamma_W, T_max, W_min)
        """
        return (
            self.gamma_T, self.gamma_S, self.gamma_D,
            self.gamma_W, self.T_max, self.W_min
        )
    
    def __repr__(self) -> str:
        return (
            f"StateTransitionConfig("
            f"γ_T={self.gamma_T}, γ_S={self.gamma_S}, "
            f"γ_D={self.gamma_D}, γ_W={self.gamma_W}, "
            f"T_max={self.T_max}, W_min={self.W_min})"
        )


# 预编译Numba函数
if __name__ == "__main__":
    # 触发Numba编译
    x_test = np.array([45.0, 0.5, 0.4, 4500.0])
    x_next = state_transition_full(
        x_test, 0.5,
        0.1, 0.05, 0.08, 100.0,
        70.0, 1400.0
    )
    
    print("状态转移模块Numba编译完成")
    print(f"当前状态: {x_test}")
    print(f"努力水平: 0.5")
    print(f"下一期状态: {x_next}")
    print(f"各维度变化:")
    print(f"  T: {x_test[0]:.2f} → {x_next[0]:.2f} (Δ={x_next[0]-x_test[0]:.2f})")
    print(f"  S_norm: {x_test[1]:.4f} → {x_next[1]:.4f} (Δ={x_next[1]-x_test[1]:.4f})")
    print(f"  D_norm: {x_test[2]:.4f} → {x_next[2]:.4f} (Δ={x_next[2]-x_test[2]:.4f})")
    print(f"  W: {x_test[3]:.2f} → {x_next[3]:.2f} (Δ={x_next[3]-x_test[3]:.2f})")

