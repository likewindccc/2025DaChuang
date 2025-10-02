"""
匹配函数

实现匹配概率函数λ(x, σ_i, a, θ)，基于Logit回归估计的参数。

函数形式（原始研究计划）：
λ(x, σ_i, a, θ) = 1 / (1 + exp[-(δ_0 + δ_x'x + δ_σ'σ_i + δ_a·a + δ_θ·ln(θ))])
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Optional
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class MatchFunction:
    """
    匹配函数
    
    实现匹配概率函数λ，用于MFG求解中判断个体匹配成功概率。
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化匹配函数
        
        Args:
            params: 估计参数字典（如果为None，需要后续load）
        """
        self.params: Optional[Dict] = params
        self.param_array: Optional[np.ndarray] = None
        
        if params is not None:
            self._parse_params(params)
        
        logger.info("匹配函数初始化完成")
    
    def _parse_params(self, params: Dict):
        """
        解析参数为NumPy数组（便于Numba优化）
        
        Args:
            params: 参数字典
        """
        # 提取参数
        self.delta_0 = params.get('const', 0.0)
        
        # 状态变量系数 (T, S, D, W)
        self.delta_x = np.array([
            params.get('delta_labor_T', 0.0),
            params.get('delta_labor_S', 0.0),
            params.get('delta_labor_D', 0.0),
            params.get('delta_labor_W', 0.0)
        ], dtype=np.float64)
        
        # 控制变量系数（劳动力与市场差距）
        self.delta_sigma = np.array([
            params.get('delta_sigma_labor_market_gap_T', 0.0),
            params.get('delta_sigma_labor_market_gap_S', 0.0),
            params.get('delta_sigma_labor_market_gap_D', 0.0),
            params.get('delta_sigma_labor_market_gap_W', 0.0)
        ], dtype=np.float64)
        
        # 努力水平系数
        self.delta_a = params.get('delta_a', 0.0)
        
        # 市场松紧度系数
        self.delta_theta = params.get('delta_theta', 0.0)
        
        # 合并为单个数组（用于Numba函数）
        self.param_array = np.concatenate([
            [self.delta_0],
            self.delta_x,
            self.delta_sigma,
            [self.delta_a],
            [self.delta_theta]
        ])
        
        logger.info("参数解析完成")
    
    def load_params(self, param_path: str):
        """
        从文件加载参数
        
        Args:
            param_path: 参数文件路径（JSON）
        """
        import json
        
        with open(param_path, 'r', encoding='utf-8') as f:
            param_dict = json.load(f)
        
        self.params = param_dict['params']
        self._parse_params(self.params)
        
        logger.info(f"参数已从{param_path}加载")
    
    def compute_match_probability(
        self,
        x: np.ndarray,
        sigma: np.ndarray,
        a: float,
        theta: float
    ) -> float:
        """
        计算单个个体的匹配概率
        
        Args:
            x: 状态变量 (T, S, D, W)
            sigma: 控制变量（劳动力与市场差距）
            a: 努力水平
            theta: 市场松紧度
        
        Returns:
            匹配概率 λ ∈ [0, 1]
        """
        if self.param_array is None:
            raise RuntimeError("参数尚未设置，请先load_params或提供params")
        
        return compute_match_probability_numba(
            x, sigma, a, theta, self.param_array
        )
    
    def compute_match_probability_batch(
        self,
        X: np.ndarray,
        Sigma: np.ndarray,
        a: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        批量计算匹配概率（Numba优化）
        
        Args:
            X: 状态变量矩阵 (n, 4): [T, S, D, W]
            Sigma: 控制变量矩阵 (n, 4)
            a: 努力水平数组 (n,)
            theta: 市场松紧度数组 (n,)
        
        Returns:
            匹配概率数组 (n,)
        """
        if self.param_array is None:
            raise RuntimeError("参数尚未设置")
        
        return compute_match_probability_batch_numba(
            X, Sigma, a, theta, self.param_array
        )
    
    def sample_match_outcome(
        self,
        x: np.ndarray,
        sigma: np.ndarray,
        a: float,
        theta: float,
        random_seed: Optional[int] = None
    ) -> bool:
        """
        根据匹配概率抽样匹配结果
        
        用于MFG求解中的随机匹配判定：
        若 p ~ Uniform(0,1) ≤ λ，则匹配成功；否则失败
        
        Args:
            x: 状态变量
            sigma: 控制变量
            a: 努力水平
            theta: 市场松紧度
            random_seed: 随机种子
        
        Returns:
            是否匹配成功
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        match_prob = self.compute_match_probability(x, sigma, a, theta)
        random_draw = np.random.uniform(0, 1)
        
        return random_draw <= match_prob
    
    def __repr__(self) -> str:
        status = "已加载参数" if self.param_array is not None else "未加载参数"
        return f"MatchFunction(status={status})"


@njit
def compute_match_probability_numba(
    x: np.ndarray,
    sigma: np.ndarray,
    a: float,
    theta: float,
    params: np.ndarray
) -> float:
    """
    计算匹配概率（Numba优化版本）
    
    λ(x, σ, a, θ) = 1 / (1 + exp[-(δ_0 + δ_x'x + δ_σ'σ + δ_a·a + δ_θ·ln(θ))])
    
    Args:
        x: 状态变量 (4,): [T, S, D, W]
        sigma: 控制变量 (4,)
        a: 努力水平
        theta: 市场松紧度
        params: 参数数组 (10,): [δ_0, δ_x (4个), δ_σ (4个), δ_a, δ_θ]
    
    Returns:
        匹配概率
    """
    # 提取参数
    delta_0 = params[0]
    delta_x = params[1:5]
    delta_sigma = params[5:9]
    delta_a = params[9]
    delta_theta = params[10]
    
    # 计算线性组合
    linear_term = delta_0
    
    # δ_x'x
    for i in range(4):
        linear_term += delta_x[i] * x[i]
    
    # δ_σ'σ
    for i in range(4):
        linear_term += delta_sigma[i] * sigma[i]
    
    # δ_a·a
    linear_term += delta_a * a
    
    # δ_θ·ln(θ)
    linear_term += delta_theta * np.log(theta)
    
    # Logit函数
    match_prob = 1.0 / (1.0 + np.exp(-linear_term))
    
    return match_prob


@njit(parallel=True)
def compute_match_probability_batch_numba(
    X: np.ndarray,
    Sigma: np.ndarray,
    a: np.ndarray,
    theta: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """
    批量计算匹配概率（Numba并行优化）
    
    Args:
        X: 状态变量矩阵 (n, 4)
        Sigma: 控制变量矩阵 (n, 4)
        a: 努力水平数组 (n,)
        theta: 市场松紧度数组 (n,)
        params: 参数数组 (10,)
    
    Returns:
        匹配概率数组 (n,)
    """
    n = X.shape[0]
    match_probs = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        match_probs[i] = compute_match_probability_numba(
            X[i], Sigma[i], a[i], theta[i], params
        )
    
    return match_probs

