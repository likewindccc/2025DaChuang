"""
状态空间管理模块

管理4维状态空间的定义、标准化和反标准化。

状态空间定义：
- T: 每周工作小时数，原始范围 [15, 70]
- S: 工作能力评分，原始范围 [2, 44]，标准化到 [0, 1]
- D: 数字素养评分，原始范围 [0, 20]，标准化到 [0, 1]
- W: 每月期望收入，原始范围 [1400, 8000]

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


@njit
def standardize_S(S: float, S_min: float = 2.0, S_max: float = 44.0) -> float:
    """
    工作能力标准化（Numba优化）
    
    S_norm = (S - S_min) / (S_max - S_min)
    
    Args:
        S: 工作能力评分（原始分数）
        S_min: 最小值，默认2
        S_max: 最大值，默认44
    
    Returns:
        标准化后的工作能力，∈ [0, 1]
    """
    return (S - S_min) / (S_max - S_min)


@njit
def destandardize_S(S_norm: float, S_min: float = 2.0, S_max: float = 44.0) -> float:
    """
    工作能力反标准化（Numba优化）
    
    S = S_norm * (S_max - S_min) + S_min
    
    Args:
        S_norm: 标准化后的工作能力，∈ [0, 1]
        S_min: 最小值
        S_max: 最大值
    
    Returns:
        原始分数
    """
    return S_norm * (S_max - S_min) + S_min


@njit
def standardize_D(D: float, D_min: float = 0.0, D_max: float = 20.0) -> float:
    """
    数字素养标准化（Numba优化）
    
    D_norm = (D - D_min) / (D_max - D_min)
    
    Args:
        D: 数字素养评分（原始分数）
        D_min: 最小值，默认0
        D_max: 最大值，默认20
    
    Returns:
        标准化后的数字素养，∈ [0, 1]
    """
    return (D - D_min) / (D_max - D_min)


@njit
def destandardize_D(D_norm: float, D_min: float = 0.0, D_max: float = 20.0) -> float:
    """
    数字素养反标准化（Numba优化）
    
    D = D_norm * (D_max - D_min) + D_min
    
    Args:
        D_norm: 标准化后的数字素养，∈ [0, 1]
        D_min: 最小值
        D_max: 最大值
    
    Returns:
        原始分数
    """
    return D_norm * (D_max - D_min) + D_min


@njit
def standardize_state(
    x_raw: np.ndarray,
    S_min: float = 2.0,
    S_max: float = 44.0,
    D_min: float = 0.0,
    D_max: float = 20.0
) -> np.ndarray:
    """
    完整状态向量标准化（Numba优化）
    
    Args:
        x_raw: 原始状态向量 (4,): [T, S, D, W]
        S_min, S_max: S的范围
        D_min, D_max: D的范围
    
    Returns:
        标准化状态向量 (4,): [T, S_norm, D_norm, W]
    
    Notes:
        - T和W保持原始尺度
        - 仅S和D进行标准化
    """
    T = x_raw[0]
    S = x_raw[1]
    D = x_raw[2]
    W = x_raw[3]
    
    S_norm = standardize_S(S, S_min, S_max)
    D_norm = standardize_D(D, D_min, D_max)
    
    return np.array([T, S_norm, D_norm, W])


@njit
def destandardize_state(
    x_norm: np.ndarray,
    S_min: float = 2.0,
    S_max: float = 44.0,
    D_min: float = 0.0,
    D_max: float = 20.0
) -> np.ndarray:
    """
    完整状态向量反标准化（Numba优化）
    
    Args:
        x_norm: 标准化状态向量 (4,): [T, S_norm, D_norm, W]
        S_min, S_max: S的范围
        D_min, D_max: D的范围
    
    Returns:
        原始状态向量 (4,): [T, S, D, W]
    """
    T = x_norm[0]
    S_norm = x_norm[1]
    D_norm = x_norm[2]
    W = x_norm[3]
    
    S = destandardize_S(S_norm, S_min, S_max)
    D = destandardize_D(D_norm, D_min, D_max)
    
    return np.array([T, S, D, W])


@njit
def standardize_batch(
    X_raw: np.ndarray,
    S_min: float = 2.0,
    S_max: float = 44.0,
    D_min: float = 0.0,
    D_max: float = 20.0
) -> np.ndarray:
    """
    批量标准化（Numba优化）
    
    Args:
        X_raw: 原始状态矩阵 (n, 4)
        S_min, S_max, D_min, D_max: 范围参数
    
    Returns:
        标准化状态矩阵 (n, 4)
    """
    n = X_raw.shape[0]
    X_norm = np.empty((n, 4), dtype=np.float64)
    
    for i in range(n):
        X_norm[i] = standardize_state(X_raw[i], S_min, S_max, D_min, D_max)
    
    return X_norm


@njit
def destandardize_batch(
    X_norm: np.ndarray,
    S_min: float = 2.0,
    S_max: float = 44.0,
    D_min: float = 0.0,
    D_max: float = 20.0
) -> np.ndarray:
    """
    批量反标准化（Numba优化）
    
    Args:
        X_norm: 标准化状态矩阵 (n, 4)
        S_min, S_max, D_min, D_max: 范围参数
    
    Returns:
        原始状态矩阵 (n, 4)
    """
    n = X_norm.shape[0]
    X_raw = np.empty((n, 4), dtype=np.float64)
    
    for i in range(n):
        X_raw[i] = destandardize_state(X_norm[i], S_min, S_max, D_min, D_max)
    
    return X_raw


class StateSpace:
    """
    状态空间管理类
    
    管理4维状态空间的定义、范围、标准化和验证。
    
    Attributes:
        T_range: T的范围 (min, max)
        S_range: S的范围（原始分数）
        D_range: D的范围（原始分数）
        W_range: W的范围
        dimension: 状态空间维度（固定为4）
    """
    
    def __init__(
        self,
        T_range: Tuple[float, float] = (15.0, 70.0),
        S_range: Tuple[float, float] = (2.0, 44.0),
        D_range: Tuple[float, float] = (0.0, 20.0),
        W_range: Tuple[float, float] = (1400.0, 8000.0)
    ):
        """
        初始化状态空间
        
        Args:
            T_range: 每周工作小时数范围，默认(15, 70)
            S_range: 工作能力评分范围，默认(2, 44)
            D_range: 数字素养评分范围，默认(0, 20)
            W_range: 每月期望收入范围，默认(1400, 8000)
        """
        self.T_range = T_range
        self.S_range = S_range
        self.D_range = D_range
        self.W_range = W_range
        self.dimension = 4
        
        logger.info(
            f"状态空间初始化: T∈{T_range}, S∈{S_range}, "
            f"D∈{D_range}, W∈{W_range}"
        )
        
        self._validate_ranges()
    
    def _validate_ranges(self):
        """验证范围的合理性"""
        for name, range_tuple in [
            ('T', self.T_range),
            ('S', self.S_range),
            ('D', self.D_range),
            ('W', self.W_range)
        ]:
            if range_tuple[0] >= range_tuple[1]:
                raise ValueError(
                    f"{name}范围不合理：min={range_tuple[0]} >= max={range_tuple[1]}"
                )
        
        logger.info("状态空间范围验证通过")
    
    def standardize(self, x_raw: np.ndarray) -> np.ndarray:
        """
        标准化单个状态向量
        
        Args:
            x_raw: 原始状态 (4,): [T, S, D, W]
        
        Returns:
            标准化状态 (4,): [T, S_norm, D_norm, W]
        """
        return standardize_state(
            x_raw,
            self.S_range[0], self.S_range[1],
            self.D_range[0], self.D_range[1]
        )
    
    def destandardize(self, x_norm: np.ndarray) -> np.ndarray:
        """
        反标准化单个状态向量
        
        Args:
            x_norm: 标准化状态 (4,)
        
        Returns:
            原始状态 (4,)
        """
        return destandardize_state(
            x_norm,
            self.S_range[0], self.S_range[1],
            self.D_range[0], self.D_range[1]
        )
    
    def standardize_batch(self, X_raw: np.ndarray) -> np.ndarray:
        """
        批量标准化
        
        Args:
            X_raw: 原始状态矩阵 (n, 4)
        
        Returns:
            标准化状态矩阵 (n, 4)
        """
        return standardize_batch(
            X_raw,
            self.S_range[0], self.S_range[1],
            self.D_range[0], self.D_range[1]
        )
    
    def destandardize_batch(self, X_norm: np.ndarray) -> np.ndarray:
        """
        批量反标准化
        
        Args:
            X_norm: 标准化状态矩阵 (n, 4)
        
        Returns:
            原始状态矩阵 (n, 4)
        """
        return destandardize_batch(
            X_norm,
            self.S_range[0], self.S_range[1],
            self.D_range[0], self.D_range[1]
        )
    
    def validate_state(self, x: np.ndarray, normalized: bool = True) -> bool:
        """
        验证状态向量是否在合理范围内
        
        Args:
            x: 状态向量 (4,)
            normalized: 是否为标准化后的状态
        
        Returns:
            是否合理
        """
        if normalized:
            # 标准化状态：[T, S_norm, D_norm, W]
            T_valid = self.T_range[0] <= x[0] <= self.T_range[1]
            S_norm_valid = 0.0 <= x[1] <= 1.0
            D_norm_valid = 0.0 <= x[2] <= 1.0
            W_valid = self.W_range[0] <= x[3] <= self.W_range[1]
            return T_valid and S_norm_valid and D_norm_valid and W_valid
        else:
            # 原始状态：[T, S, D, W]
            T_valid = self.T_range[0] <= x[0] <= self.T_range[1]
            S_valid = self.S_range[0] <= x[1] <= self.S_range[1]
            D_valid = self.D_range[0] <= x[2] <= self.D_range[1]
            W_valid = self.W_range[0] <= x[3] <= self.W_range[1]
            return T_valid and S_valid and D_valid and W_valid
    
    def get_bounds_for_grid(self) -> List[Tuple[float, float]]:
        """
        获取稀疏网格的边界
        
        Returns:
            4个维度的边界列表: [(T_min, T_max), (0, 1), (0, 1), (W_min, W_max)]
        
        Notes:
            - S和D在稀疏网格中使用标准化值[0, 1]
            - T和W使用原始范围
        """
        return [
            self.T_range,
            (0.0, 1.0),  # S标准化后
            (0.0, 1.0),  # D标准化后
            self.W_range
        ]
    
    def get_state_names(self) -> List[str]:
        """
        获取状态变量名称
        
        Returns:
            ['T', 'S_norm', 'D_norm', 'W']
        """
        return ['T', 'S_norm', 'D_norm', 'W']
    
    def __repr__(self) -> str:
        return (
            f"StateSpace(dimension={self.dimension}, "
            f"T∈{self.T_range}, S∈{self.S_range}, "
            f"D∈{self.D_range}, W∈{self.W_range})"
        )


# 预编译Numba函数
if __name__ == "__main__":
    # 测试标准化和反标准化
    x_raw = np.array([45.0, 25.0, 10.0, 4500.0])
    print("原始状态:", x_raw)
    
    x_norm = standardize_state(x_raw)
    print("标准化后:", x_norm)
    
    x_recovered = destandardize_state(x_norm)
    print("反标准化:", x_recovered)
    
    print(f"恢复误差: {np.max(np.abs(x_raw - x_recovered)):.10f}")
    
    # 测试StateSpace类
    state_space = StateSpace()
    print(f"\n{state_space}")
    print(f"稀疏网格边界: {state_space.get_bounds_for_grid()}")
    print(f"状态验证（标准化）: {state_space.validate_state(x_norm, normalized=True)}")
    print(f"状态验证（原始）: {state_space.validate_state(x_raw, normalized=False)}")

