#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG 匹配与离职率辅助工具。

本模块负责统一 BellmanSolver 与 KFESolver 在以下三类逻辑上的实现：
1. 劳动力人口统计控制变量 sigma 的构造；
2. T/S/D/W 在给定努力水平下的状态投影；
3. 基于目标均值反解 Logit 截距/平移项。

这样可以避免不同求解器在量纲、标准化方式和概率口径上出现分叉。
"""

from typing import Tuple

import numpy as np


def stable_sigmoid(linear_term: np.ndarray) -> np.ndarray:
    """
    对线性指标执行数值稳定的 sigmoid 变换。

    参数：
        linear_term: Logit 模型的线性项。

    返回：
        与输入同形状的概率数组。
    """
    clipped = np.clip(np.asarray(linear_term, dtype=float), -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_sigma_from_demographics(
    age: np.ndarray,
    education: np.ndarray,
    children: np.ndarray,
) -> np.ndarray:
    """
    根据年龄、学历和子女数构造控制变量 sigma。

    该实现与 Logistic 匹配函数训练阶段保持一致：先分别 Min-Max 标准化，
    再求和，最后对总和再次做 Min-Max 标准化。
    """
    age = np.asarray(age, dtype=float)
    education = np.asarray(education, dtype=float)
    children = np.asarray(children, dtype=float)

    age_norm = (age - age.min()) / (age.max() - age.min() + 1.0e-10)
    education_norm = (
        (education - education.min()) /
        (education.max() - education.min() + 1.0e-10)
    )
    children_norm = (
        (children - children.min()) /
        (children.max() - children.min() + 1.0e-10)
    )

    sigma_sum = age_norm + education_norm + children_norm
    sigma = (
        (sigma_sum - sigma_sum.min()) /
        (sigma_sum.max() - sigma_sum.min() + 1.0e-10)
    )
    return sigma


def project_states_with_effort_grid(
    T: np.ndarray,
    S: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    effort_grid: np.ndarray,
    gamma_T: float,
    gamma_W: float,
    gamma_S: float,
    gamma_D: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将单期努力网格映射到下一期状态。

    返回形状均为 `(N, n_effort)`，用于 Bellman 方程中的批量匹配概率计算。
    S/D 的更新采用“先 Min-Max 标准化到 [0,1]，更新后再反标准化”的统一口径。
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=float)
    W = np.asarray(W, dtype=float)
    effort_grid = np.asarray(effort_grid, dtype=float)

    t_max = float(T.max())
    w_min = float(W.min())
    s_min = float(S.min())
    s_max = float(S.max())
    d_min = float(D.min())
    d_max = float(D.max())

    T_new = T[:, None] + gamma_T * effort_grid[None, :] * (
        t_max - T[:, None]
    )
    W_new = np.maximum(
        w_min,
        W[:, None] - gamma_W * effort_grid[None, :]
    )

    s_range = s_max - s_min
    if s_range > 1.0e-10:
        s_norm = (S - s_min) / s_range
        s_norm_new = s_norm[:, None] + gamma_S * effort_grid[None, :] * (
            1.0 - s_norm[:, None]
        )
        S_new = s_norm_new * s_range + s_min
    else:
        S_new = np.broadcast_to(S[:, None], (len(S), len(effort_grid)))

    d_range = d_max - d_min
    if d_range > 1.0e-10:
        d_norm = (D - d_min) / d_range
        d_norm_new = d_norm[:, None] + gamma_D * effort_grid[None, :] * (
            1.0 - d_norm[:, None]
        )
        D_new = d_norm_new * d_range + d_min
    else:
        D_new = np.broadcast_to(D[:, None], (len(D), len(effort_grid)))

    return T_new, S_new, D_new, W_new


def project_states_with_effort_vector(
    T: np.ndarray,
    S: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    effort: np.ndarray,
    gamma_T: float,
    gamma_W: float,
    gamma_S: float,
    gamma_D: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将逐个体努力向量映射到下一期状态。

    返回形状均为 `(N,)`，用于 KFE 演化和结构参数的参考样本校准。
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=float)
    W = np.asarray(W, dtype=float)
    effort = np.asarray(effort, dtype=float)

    if effort.shape != T.shape:
        raise ValueError("逐个体努力向量的形状必须与人口状态向量一致。")

    t_max = float(T.max())
    w_min = float(W.min())
    s_min = float(S.min())
    s_max = float(S.max())
    d_min = float(D.min())
    d_max = float(D.max())

    T_new = T + gamma_T * effort * (t_max - T)
    W_new = np.maximum(w_min, W - gamma_W * effort)

    s_range = s_max - s_min
    if s_range > 1.0e-10:
        s_norm = (S - s_min) / s_range
        s_norm_new = s_norm + gamma_S * effort * (1.0 - s_norm)
        S_new = s_norm_new * s_range + s_min
    else:
        S_new = S.copy()

    d_range = d_max - d_min
    if d_range > 1.0e-10:
        d_norm = (D - d_min) / d_range
        d_norm_new = d_norm + gamma_D * effort * (1.0 - d_norm)
        D_new = d_norm_new * d_range + d_min
    else:
        D_new = D.copy()

    return T_new, S_new, D_new, W_new


def solve_logit_shift_for_target(
    base_linear_term: np.ndarray,
    target_rate: float,
    lower: float = -20.0,
    upper: float = 5.0,
    tol: float = 1.0e-10,
    max_iter: int = 100,
) -> float:
    """
    反解 Logit 平移项，使平均概率逼近给定目标。

    该函数用于两类场景：
    1. 对匹配概率添加统一截距平移，压回可解释的 hazard 量级；
    2. 对离职率反解 eta0，使平均 separation rate 对齐目标矩。
    """
    base_linear_term = np.asarray(base_linear_term, dtype=float).reshape(-1)
    if base_linear_term.size == 0:
        raise ValueError("反解 Logit 平移项时，输入样本不能为空。")

    if not 0.0 < float(target_rate) < 1.0:
        raise ValueError("target_rate 必须位于 (0, 1) 区间内。")

    left = float(lower)
    right = float(upper)

    def mean_prob(shift: float) -> float:
        return float(stable_sigmoid(base_linear_term + shift).mean())

    left_value = mean_prob(left) - target_rate
    right_value = mean_prob(right) - target_rate

    expansion_step = 5.0
    expansion_count = 0
    while left_value > 0.0 and expansion_count < 20:
        right = left
        right_value = left_value
        left -= expansion_step
        left_value = mean_prob(left) - target_rate
        expansion_count += 1

    expansion_count = 0
    while right_value < 0.0 and expansion_count < 20:
        left = right
        left_value = right_value
        right += expansion_step
        right_value = mean_prob(right) - target_rate
        expansion_count += 1

    if left_value > 0.0:
        return left
    if right_value < 0.0:
        return right

    for _ in range(max_iter):
        midpoint = 0.5 * (left + right)
        midpoint_value = mean_prob(midpoint) - target_rate

        if abs(midpoint_value) < tol:
            return midpoint

        if midpoint_value > 0.0:
            right = midpoint
        else:
            left = midpoint

    return 0.5 * (left + right)
