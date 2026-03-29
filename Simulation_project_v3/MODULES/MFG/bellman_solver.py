#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝尔曼方程求解器 V2（统一处理版本，Numba 加速）。

【重构】核心改进：
- 所有个体同时计算 V_U 和 V_E（无论当前处于失业还是就业）。
- V_U[i]：个体 i 处于失业状态的价值。
- V_E[i]：个体 i 处于就业状态的价值。
- 解决了旧版本中状态切换时价值函数不连续的问题。

设计结构：
1. Numba 加速的核心计算函数（纯 NumPy 数组 + njit 装饰）。
2. Python 封装类（数据准备、模型调用、结果整理）。
"""

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from numba import njit, prange
import yaml

from .matching_utils import (
    compute_sigma_from_demographics,
    project_states_with_effort_grid,
    project_states_with_effort_vector,
    solve_logit_shift_for_target,
    stable_sigmoid,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Numba 加速的核心计算函数
# =============================================================================

@njit
def update_state_numba(
    T: float, S: float, D: float, W: float,
    a: float,
    T_max_population: float,
    W_min_population: float,
    S_min_population: float, S_max_population: float,
    D_min_population: float, D_max_population: float,
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float
) -> Tuple[float, float, float, float]:
    """
    状态更新函数（Numba 加速）。

    对 S 和 D 进行 MinMax 标准化到 [0, 1]，应用更新公式后再反标准化。
    """
    # T 和 W 直接更新
    T_new = T + gamma_T * a * (T_max_population - T)
    W_new = max(W_min_population, W - gamma_W * a)
    
    # S: MinMax 标准化 -> 更新 -> 反标准化
    S_range = S_max_population - S_min_population
    if S_range > 1e-10:
        S_norm = (S - S_min_population) / S_range  # 标准化到 [0, 1]
        S_norm_new = S_norm + gamma_S * a * (1.0 - S_norm)  # 更新
        S_new = S_norm_new * S_range + S_min_population  # 反标准化
    else:
        S_new = S  # 如果所有人 S 相同，保持不变
    # D: MinMax 标准化 -> 更新 -> 反标准化
    D_range = D_max_population - D_min_population
    if D_range > 1e-10:
        D_norm = (D - D_min_population) / D_range  # 标准化到 [0, 1]
        D_norm_new = D_norm + gamma_D * a * (1.0 - D_norm)  # 更新
        D_new = D_norm_new * D_range + D_min_population  # 反标准化
    else:
        D_new = D  # 如果所有人 D 相同，保持不变
    return T_new, S_new, D_new, W_new


@njit
def compute_separation_rate_numba(
    T: float, S: float, D: float, W: float,
    age: float, education: float, children: float,
    T_mean: float, T_std: float,
    S_mean: float, S_std: float,
    D_mean: float, D_std: float,
    W_mean: float, W_std: float,
    age_mean: float, age_std: float,
    edu_mean: float, edu_std: float,
    children_mean: float, children_std: float,
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float
) -> float:
    """计算外生离职率。"""
    T_std_val = (T - T_mean) / (T_std + 1.0e-10)
    S_std_val = (S - S_mean) / (S_std + 1.0e-10)
    D_std_val = (D - D_mean) / (D_std + 1.0e-10)
    W_std_val = (W - W_mean) / (W_std + 1.0e-10)
    age_std_val = (age - age_mean) / (age_std + 1.0e-10)
    edu_std_val = (education - edu_mean) / (edu_std + 1.0e-10)
    children_std_val = (children - children_mean) / (
        children_std + 1.0e-10
    )

    z = (
        eta0 +
        eta_T * T_std_val +
        eta_S * S_std_val +
        eta_D * D_std_val +
        eta_W * W_std_val +
        eta_age * age_std_val +
        eta_edu * edu_std_val +
        eta_children * children_std_val
    )

    mu = 1.0 / (1.0 + np.exp(-z))
    mu = max(1.0e-6, min(1.0 - 1.0e-6, mu))
    return mu


@njit(parallel=True)
def solve_bellman_unified_numba(
    T: np.ndarray,
    S: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    age: np.ndarray,
    edu: np.ndarray,
    children: np.ndarray,
    is_unemployed: np.ndarray,
    current_wage: np.ndarray,
    lambda_probs: np.ndarray,
    V_U_next: np.ndarray,
    V_E_next: np.ndarray,
    T_mean: float,
    T_std: float,
    S_mean: float,
    S_std: float,
    D_mean: float,
    D_std: float,
    W_mean: float,
    W_std: float,
    age_mean: float,
    age_std: float,
    edu_mean: float,
    edu_std: float,
    children_mean: float,
    children_std: float,
    a_grid: np.ndarray,
    rho: float,
    kappa: float,
    b0: float,
    initial_T: np.ndarray,
    disutility_T_enabled: bool,
    alpha_T: float,
    eta0: float,
    eta_T: float,
    eta_S: float,
    eta_D: float,
    eta_W: float,
    eta_age: float,
    eta_edu: float,
    eta_children: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一求解所有个体的失业价值、就业价值和最优努力。
    """
    N = len(T)
    n_effort = len(a_grid)

    V_U = np.zeros(N)
    V_E = np.zeros(N)
    a_optimal = np.zeros(N)
    b = b0

    for i in prange(N):
        max_value_u = -np.inf
        best_a = 0.0

        for j in range(n_effort):
            a = a_grid[j]
            effort_cost = 0.5 * kappa * a * a

            if disutility_T_enabled:
                disutility_t = alpha_T * (T[i] - initial_T[i]) ** 2
            else:
                disutility_t = 0.0

            instant_utility = b - effort_cost - disutility_t
            lambda_prob = lambda_probs[i, j]
            v_next_expected = (
                lambda_prob * V_E_next[i] +
                (1.0 - lambda_prob) * V_U_next[i]
            )
            total_value = instant_utility + rho * v_next_expected

            if total_value > max_value_u:
                max_value_u = total_value
                best_a = a

        V_U[i] = max_value_u
        a_optimal[i] = best_a

        if is_unemployed[i]:
            omega = W[i]
        else:
            omega = current_wage[i]

        mu = compute_separation_rate_numba(
            T[i], S[i], D[i], W[i],
            age[i], edu[i], children[i],
            T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
            age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        v_next_expected_e = mu * V_U_next[i] + (1.0 - mu) * V_E_next[i]
        V_E[i] = omega + rho * v_next_expected_e

    return V_U, V_E, a_optimal


@njit
def value_iteration_unified_numba(
    T: np.ndarray,
    S: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    age: np.ndarray,
    edu: np.ndarray,
    children: np.ndarray,
    is_unemployed: np.ndarray,
    current_wage: np.ndarray,
    lambda_probs: np.ndarray,
    a_grid: np.ndarray,
    rho: float,
    kappa: float,
    b0: float,
    initial_T: np.ndarray,
    disutility_T_enabled: bool,
    alpha_T: float,
    eta0: float,
    eta_T: float,
    eta_S: float,
    eta_D: float,
    eta_W: float,
    eta_age: float,
    eta_edu: float,
    eta_children: float,
    initial_V_U: np.ndarray,
    initial_V_E: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    值迭代主循环。

    使用传入的初始值函数作为 warm start；若未提供，则由调用方传入零数组。
    """
    N = len(T)

    V_U_all = initial_V_U.copy()
    V_E_all = initial_V_E.copy()
    a_optimal = np.zeros(N)

    T_mean, T_std = T.mean(), T.std()
    S_mean, S_std = S.mean(), S.std()
    D_mean, D_std = D.mean(), D.std()
    W_mean, W_std = W.mean(), W.std()
    age_mean, age_std = age.mean(), age.std()
    edu_mean, edu_std = edu.mean(), edu.std()
    children_mean, children_std = children.mean(), children.std()

    max_diff = 0.0

    for iteration in range(max_iter):
        V_U_old = V_U_all.copy()
        V_E_old = V_E_all.copy()

        V_U_all, V_E_all, a_optimal = solve_bellman_unified_numba(
            T, S, D, W, age, edu, children,
            is_unemployed, current_wage,
            lambda_probs, V_U_old, V_E_old,
            T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
            age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
            a_grid, rho, kappa, b0,
            initial_T, disutility_T_enabled, alpha_T,
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )

        V_U_all = np.clip(V_U_all, -1.0e6, 1.0e6)
        V_E_all = np.clip(V_E_all, -1.0e6, 1.0e6)

        diff_U = np.max(np.abs(V_U_all - V_U_old))
        diff_E = np.max(np.abs(V_E_all - V_E_old))
        max_diff = max(diff_U, diff_E)

        if max_diff < tol:
            return V_U_all, V_E_all, a_optimal, iteration + 1, max_diff

    return V_U_all, V_E_all, a_optimal, max_iter, max_diff


# =============================================================================
# Python 封装类
# =============================================================================

class BellmanSolver:
    """Bellman 方程求解器。"""

    def __init__(self, config, match_function_model):
        """
        初始化 Bellman 求解器。

        参数：
            config: 已加载的 MFG 配置字典。
            match_function_model: 已训练好的匹配函数模型。
        """
        effort_cfg = config['effort']
        self.a_grid = np.linspace(
            effort_cfg['a_min'],
            effort_cfg['a_max'],
            effort_cfg['a_points']
        )

        econ = config['economics']
        self.rho = econ['rho']
        self.kappa = econ['kappa']
        self.b0 = econ['unemployment_benefit']['b0']

        disutility_T_cfg = econ.get(
            'disutility_T',
            {'enabled': False, 'alpha': 0.0}
        )
        self.disutility_T_enabled = disutility_T_cfg.get('enabled', False)
        self.alpha_T = disutility_T_cfg.get('alpha', 0.0)

        state_update = econ['state_update']
        self.gamma_T = state_update['gamma_T']
        self.gamma_W = state_update['gamma_W']
        self.gamma_S = state_update['gamma_S']
        self.gamma_D = state_update['gamma_D']

        sep_rate = econ['separation_rate']
        self.eta0 = sep_rate['eta0']
        self.eta_T = sep_rate['eta_T']
        self.eta_S = sep_rate['eta_S']
        self.eta_D = sep_rate['eta_D']
        self.eta_W = sep_rate['eta_W']
        self.eta_age = sep_rate['eta_age']
        self.eta_edu = sep_rate['eta_edu']
        self.eta_children = sep_rate['eta_children']

        val_iter = config['value_iteration']
        self.max_iter = val_iter['max_iter']
        self.tol = val_iter['tol']

        self.match_function_model = match_function_model
        match_cfg = config.get('market', {}).get('match_probability', {})
        self.lambda_intercept_shift = float(
            match_cfg.get('intercept_shift', 0.0)
        )
        self.auto_calibrate_match_intercept = bool(
            match_cfg.get('auto_calibrate_intercept', False)
        )
        self.target_job_finding_rate = float(
            match_cfg.get('target_rate', 0.0)
        )
        lambda_bounds = match_cfg.get('intercept_bounds', [-20.0, 5.0])
        self.lambda_intercept_bounds = (
            float(lambda_bounds[0]),
            float(lambda_bounds[1]),
        )

        params = getattr(self.match_function_model, 'params', None)
        if params is None:
            raise ValueError("匹配函数模型缺少 params，无法执行矩阵化预测。")

        if hasattr(params, 'index'):
            self.match_model_columns = list(params.index)
            self.match_model_coef = params.to_numpy(dtype=float)
        else:
            self.match_model_columns = list(
                getattr(self.match_function_model.model, 'exog_names', [])
            )
            self.match_model_coef = np.asarray(params, dtype=float)

        if not self.match_model_columns:
            raise ValueError(
                "匹配函数模型缺少 exog_names，无法对齐预测列顺序。"
            )

    def _predict_match_linear_term(
        self,
        T: np.ndarray,
        S: np.ndarray,
        D: np.ndarray,
        W: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        使用缓存系数执行矩阵化预测，并保持与 statsmodels 列顺序一致。
        """
        target_shape = np.broadcast_shapes(
            np.shape(T),
            np.shape(S),
            np.shape(D),
            np.shape(W),
            np.shape(sigma),
            np.shape(theta)
        )

        columns = {
            'const': np.ones(target_shape, dtype=float),
            'T': np.broadcast_to(np.asarray(T, dtype=float), target_shape),
            'S': np.broadcast_to(np.asarray(S, dtype=float), target_shape),
            'D': np.broadcast_to(np.asarray(D, dtype=float), target_shape),
            'W': np.broadcast_to(np.asarray(W, dtype=float), target_shape),
            'sigma': np.broadcast_to(np.asarray(sigma, dtype=float), target_shape),
            'theta': np.broadcast_to(np.asarray(theta, dtype=float), target_shape),
        }

        try:
            design_matrix = np.column_stack([
                columns[name].reshape(-1)
                for name in self.match_model_columns
            ])
        except KeyError as exc:
            raise KeyError(
                f"匹配函数模型包含未支持的列: {exc.args[0]}"
            ) from exc

        linear_term = design_matrix @ self.match_model_coef
        return linear_term.reshape(target_shape)

    def _predict_match_probabilities(
        self,
        T: np.ndarray,
        S: np.ndarray,
        D: np.ndarray,
        W: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        基于线性项和统一截距平移预测匹配概率。
        """
        linear_term = self._predict_match_linear_term(
            T=T,
            S=S,
            D=D,
            W=W,
            sigma=sigma,
            theta=theta,
        )
        return stable_sigmoid(linear_term + self.lambda_intercept_shift)

    def set_lambda_intercept_shift(self, intercept_shift: float) -> None:
        """
        设置匹配概率的统一截距平移项。
        """
        self.lambda_intercept_shift = float(intercept_shift)

    def set_eta0(self, eta0: float) -> None:
        """
        设置离职率函数的截距项 eta0。
        """
        self.eta0 = float(eta0)

    def calibrate_lambda_intercept(
        self,
        individuals: pd.DataFrame,
        theta: float,
        effort: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> float:
        """
        根据参考样本反解匹配概率截距平移项。

        该平移项会在保留个体排序与相对比较静态的前提下，将总体 job-finding
        hazard 压回目标量级。
        """
        if not self.auto_calibrate_match_intercept:
            return self.lambda_intercept_shift

        if not 0.0 < self.target_job_finding_rate < 1.0:
            logger.warning("目标 job-finding rate 无效，跳过匹配截距校准。")
            return self.lambda_intercept_shift

        unemployed_mask = (
            individuals['employment_status'].to_numpy() == 'unemployed'
        )
        if not unemployed_mask.any():
            if verbose:
                logger.warning("参考样本中没有失业者，跳过匹配截距校准。")
            return self.lambda_intercept_shift

        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        education = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)

        if effort is None:
            effort_array = np.zeros(len(individuals), dtype=float)
        else:
            effort_array = np.asarray(effort, dtype=float)

        sigma = compute_sigma_from_demographics(age, education, children)
        T_new, S_new, D_new, W_new = project_states_with_effort_vector(
            T,
            S,
            D,
            W,
            effort_array,
            self.gamma_T,
            self.gamma_W,
            self.gamma_S,
            self.gamma_D,
        )
        linear_term = self._predict_match_linear_term(
            T=T_new,
            S=S_new,
            D=D_new,
            W=W_new,
            sigma=sigma,
            theta=np.full(len(T), float(theta), dtype=float),
        )
        intercept_shift = solve_logit_shift_for_target(
            linear_term[unemployed_mask],
            self.target_job_finding_rate,
            lower=self.lambda_intercept_bounds[0],
            upper=self.lambda_intercept_bounds[1],
        )

        if verbose:
            raw_mean = float(stable_sigmoid(linear_term[unemployed_mask]).mean())
            adjusted_mean = float(
                stable_sigmoid(
                    linear_term[unemployed_mask] + intercept_shift
                ).mean()
            )
            logger.info(
                "匹配概率截距校准完成：raw=%.4f, target=%.4f, adjusted=%.4f, shift=%.4f",
                raw_mean,
                self.target_job_finding_rate,
                adjusted_mean,
                intercept_shift,
            )

        return float(intercept_shift)
    
    def compute_match_probabilities_batch(
        self,
        individuals: pd.DataFrame,
        a_grid: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """
        批量计算每个个体在所有努力水平下的匹配概率矩阵。
        """
        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        education = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)

        sigma = compute_sigma_from_demographics(age, education, children)
        a_grid_arr = np.asarray(a_grid, dtype=float)
        T_new, S_new, D_new, W_new = project_states_with_effort_grid(
            T,
            S,
            D,
            W,
            a_grid_arr,
            self.gamma_T,
            self.gamma_W,
            self.gamma_S,
            self.gamma_D,
        )

        sigma_matrix = np.broadcast_to(sigma[:, None], T_new.shape)
        theta_matrix = np.full(T_new.shape, float(theta), dtype=float)

        return self._predict_match_probabilities(
            T=T_new,
            S=S_new,
            D=D_new,
            W=W_new,
            sigma=sigma_matrix,
            theta=theta_matrix
        )

    def solve(
        self,
        individuals: pd.DataFrame,
        theta: float,
        initial_T: np.ndarray,
        initial_V_U: Optional[np.ndarray] = None,
        initial_V_E: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        求解贝尔曼方程，返回失业价值、就业价值和最优努力。
        """
        N = len(individuals)

        if verbose:
            logger.info("求解贝尔曼方程（N=%s）...", N)
            logger.info("批量计算匹配概率...")

        lambda_probs = self.compute_match_probabilities_batch(
            individuals, self.a_grid, theta
        )

        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        edu = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)
        is_unemployed = (
            individuals['employment_status'].to_numpy() == 'unemployed'
        )
        current_wage = individuals['current_wage'].fillna(0.0).to_numpy(
            dtype=float
        )

        if verbose:
            logger.info("开始值迭代（Numba加速）...")

        initial_v_u_arr = (
            np.zeros(N, dtype=float)
            if initial_V_U is None
            else np.asarray(initial_V_U, dtype=float).copy()
        )
        initial_v_e_arr = (
            np.zeros(N, dtype=float)
            if initial_V_E is None
            else np.asarray(initial_V_E, dtype=float).copy()
        )

        (
            V_U_array,
            V_E_array,
            a_optimal_array,
            iterations,
            max_diff
        ) = value_iteration_unified_numba(
            T, S, D, W,
            age, edu, children,
            is_unemployed,
            current_wage,
            lambda_probs,
            self.a_grid,
            self.rho, self.kappa, self.b0,
            initial_T, self.disutility_T_enabled, self.alpha_T,
            self.eta0, self.eta_T, self.eta_S, self.eta_D, self.eta_W,
            self.eta_age, self.eta_edu, self.eta_children,
            initial_v_u_arr, initial_v_e_arr,
            self.max_iter, self.tol
        )

        # `a_optimal` 的经济含义是“处于失业状态时的最优搜索努力”。
        # 对当前已就业个体，将该反事实策略在导出层面统一记为 0，
        # 避免后续统计把它误解为就业者真实投入的搜索努力。
        a_optimal_array = np.asarray(a_optimal_array, dtype=float).copy()
        a_optimal_array[~is_unemployed] = 0.0

        if verbose:
            logger.info(
                "值迭代完成：迭代%s轮，最大差异%.6f",
                iterations,
                max_diff
            )
            logger.info(
                "  V_U统计: min=%.2f, max=%.2f, mean=%.2f",
                V_U_array.min(),
                V_U_array.max(),
                V_U_array.mean()
            )
            logger.info(
                "  V_E统计: min=%.2f, max=%.2f, mean=%.2f",
                V_E_array.min(),
                V_E_array.max(),
                V_E_array.mean()
            )
            logger.info(
                "  a统计: min=%.4f, max=%.4f, mean=%.4f",
                a_optimal_array.min(),
                a_optimal_array.max(),
                a_optimal_array.mean()
            )

        V_U = pd.Series(V_U_array, index=individuals.index)
        V_E = pd.Series(V_E_array, index=individuals.index)
        a_optimal = pd.Series(a_optimal_array, index=individuals.index)
        return V_U, V_E, a_optimal


def load_match_function_model(model_path: str):
    """
    加载已训练的匹配函数模型。

    匹配函数在 LOGISTIC 模块中训练为 Logit 回归模型：
    位(x, sigma, theta) = P(matched=1 | T, S, D, W, sigma, theta)

    参数:
        model_path: 模型文件路径（pkl 格式）。

    返回:
        已训练的 statsmodels Logit 模型对象。
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
