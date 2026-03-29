#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KFE 演化求解器。

基于个体层面的蒙特卡洛模拟更新人口状态分布，并计算失业率、
岗位找到率和离职率等宏观统计量。

修改记录：
  - 方案B：匹配工资改为从企业出价分布 N(mean_wage_firm, std_wage_firm) 采样，
    不再使用劳动力期望工资 W × (1 + 噪声)。
    这使工资均值由企业侧配置决定（对齐 CLDS 目标矩 3578 元），
    工资分散度来自 std_wage_firm（1500 元），可更好复现 M3/M7/M8 分散度矩。
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange

from .matching_utils import (
    compute_sigma_from_demographics,
    project_states_with_effort_vector,
    solve_logit_shift_for_target,
    stable_sigmoid,
)


logger = logging.getLogger(__name__)


@njit
def simulate_employment_transition(
    is_unemployed: bool,
    lambda_prob: float,
    mu_prob: float
) -> bool:
    """模拟单个个体的就业状态转移。"""
    if is_unemployed:
        if np.random.random() < lambda_prob:
            return False
        return True

    if np.random.random() < mu_prob:
        return True
    return False


@njit(parallel=True)
def simulate_population_evolution(
    employment_status: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    current_wage: np.ndarray,
    age: np.ndarray,
    edu: np.ndarray,
    children: np.ndarray,
    optimal_effort: np.ndarray,
    lambda_probs: np.ndarray,
    mu_probs: np.ndarray,
    T_max_population: float,
    W_min_population: float,
    S_min_population: float,
    S_max_population: float,
    D_min_population: float,
    D_max_population: float,
    gamma_T: float,
    gamma_W: float,
    gamma_S: float,
    gamma_D: float,
    sigma_match: float,
    mean_wage_firm: float,   # 企业出价工资均值（来自 employment_utility.mean_wage）
    std_wage_firm: float,    # 残差工资噪声标准差（来自 employment_utility.std_wage）
    S_mean: float,           # 当前人口 S 均值，用于计算技能溢价
    D_mean: float,           # 当前人口 D 均值，用于计算数字素养溢价
    beta_wage_S: float,      # S 技能工资溢价系数（元/单位S偏差）
    beta_wage_D: float,      # D 数字素养工资溢价系数（元/单位D偏差）
    min_wage: float          # 最低入职工资下界（元），防止对数工资截断极端值
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """并行模拟整个人口的一期演化。"""
    N = len(employment_status)

    employment_status_new = np.empty(N, dtype=np.bool_)
    T_new = np.zeros(N)
    S_new = np.zeros(N)
    D_new = np.zeros(N)
    W_new = np.zeros(N)
    current_wage_new = np.zeros(N)

    for i in prange(N):
        is_unemployed = employment_status[i]

        if is_unemployed:
            lambda_i = lambda_probs[i]
            employment_status_new[i] = simulate_employment_transition(
                True, lambda_i, 0.0
            )

            a_opt = optimal_effort[i]
            T_new[i] = T[i] + gamma_T * a_opt * (T_max_population - T[i])
            W_new[i] = max(W_min_population, W[i] - gamma_W * a_opt)

            S_range = S_max_population - S_min_population
            if S_range > 1.0e-10:
                S_norm = (S[i] - S_min_population) / S_range
                S_norm_new = S_norm + gamma_S * a_opt * (1.0 - S_norm)
                S_new[i] = S_norm_new * S_range + S_min_population
            else:
                S_new[i] = S[i]

            D_range = D_max_population - D_min_population
            if D_range > 1.0e-10:
                D_norm = (D[i] - D_min_population) / D_range
                D_norm_new = D_norm + gamma_D * a_opt * (1.0 - D_norm)
                D_new[i] = D_norm_new * D_range + D_min_population
            else:
                D_new[i] = D[i]

            if not employment_status_new[i]:
                # 技能溢价工资模型：工资 = 基础工资 + 技能溢价 + 残差噪声
                # 技能溢价部分：高 S/D 个体享有更高工资，增强工资截面分散度，
                # 有助于复现 M7（IQR比）和 M3（对数工资标准差）目标矩。
                # 最低工资下界防止 max(0, ·) 截断后取对数产生极端负值。
                skill_premium = (
                    beta_wage_S * (S[i] - S_mean)
                    + beta_wage_D * (D[i] - D_mean)
                )
                residual_noise = std_wage_firm * np.random.normal(0.0, 1.0)
                matched_wage = mean_wage_firm + skill_premium + residual_noise
                current_wage_new[i] = max(min_wage, matched_wage)
            else:
                current_wage_new[i] = 0.0
        else:
            mu_i = mu_probs[i]
            employment_status_new[i] = simulate_employment_transition(
                False, 0.0, mu_i
            )

            T_new[i] = T[i]
            S_new[i] = S[i]
            D_new[i] = D[i]
            W_new[i] = W[i]

            if employment_status_new[i]:
                current_wage_new[i] = 0.0
            else:
                current_wage_new[i] = current_wage[i]

    return (
        employment_status_new,
        T_new,
        S_new,
        D_new,
        W_new,
        current_wage_new
    )


class KFESolver:
    """KFE 演化求解器。"""

    def __init__(self, config: Dict, match_function_model):
        """初始化 KFE 求解器。"""
        self.config = config
        self.match_model = match_function_model

        self.gamma_T = config['economics']['state_update']['gamma_T']
        self.gamma_W = config['economics']['state_update']['gamma_W']
        self.gamma_S = config['economics']['state_update']['gamma_S']
        self.gamma_D = config['economics']['state_update']['gamma_D']
        # 保留 sigma_match 字段以兼容旧配置，方案B不再使用该噪声
        self.sigma_match = config['economics']['state_update'].get(
            'sigma_match', 0.10
        )
        # 从配置读取企业工资分布参数（均值、标准差、技能溢价、最低工资）
        eu = config['economics'].get('employment_utility', {})
        self.mean_wage_firm = float(eu.get('mean_wage', 3500.0))
        self.std_wage_firm = float(eu.get('std_wage', 1500.0))
        self.beta_wage_S = float(eu.get('beta_wage_S', 0.0))
        self.beta_wage_D = float(eu.get('beta_wage_D', 0.0))
        self.min_wage = float(eu.get('min_wage', 0.0))

        sep = config['economics']['separation_rate']
        self.eta0 = sep['eta0']
        self.eta_T = sep['eta_T']
        self.eta_S = sep['eta_S']
        self.eta_D = sep['eta_D']
        self.eta_W = sep['eta_W']
        self.eta_age = sep['eta_age']
        self.eta_edu = sep['eta_edu']
        self.eta_children = sep['eta_children']
        self.auto_calibrate_eta0 = bool(
            sep.get('auto_calibrate_eta0', False)
        )
        self.target_separation_rate = float(sep.get('target_rate', 0.0))
        eta0_bounds = sep.get('eta0_bounds', [-20.0, 5.0])
        self.eta0_bounds = (float(eta0_bounds[0]), float(eta0_bounds[1]))

        self.target_theta = config['market']['target_theta']
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

        params = getattr(self.match_model, 'params', None)
        if params is None:
            raise ValueError("匹配函数模型缺少 params，无法执行矩阵化预测。")

        if hasattr(params, 'index'):
            self.match_model_columns = list(params.index)
            self.match_model_coef = params.to_numpy(dtype=float)
        else:
            self.match_model_columns = list(
                getattr(self.match_model.model, 'exog_names', [])
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
        """使用缓存系数执行矩阵化预测。"""
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
        effort: pd.Series,
        verbose: bool = True,
    ) -> float:
        """
        对匹配概率增加统一截距平移，使平均 job-finding rate 对齐目标值。
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

    def calibrate_eta0(
        self,
        individuals: pd.DataFrame,
        verbose: bool = True,
    ) -> float:
        """
        基于当前参考样本反解 eta0，使平均 separation rate 对齐目标矩。
        """
        if not self.auto_calibrate_eta0:
            return self.eta0

        if not 0.0 < self.target_separation_rate < 1.0:
            logger.warning("目标 separation rate 无效，跳过 eta0 校准。")
            return self.eta0

        employed_mask = (
            individuals['employment_status'].to_numpy() == 'employed'
        )
        if not employed_mask.any():
            if verbose:
                logger.warning("参考样本中没有就业者，跳过 eta0 校准。")
            return self.eta0

        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        education = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)

        T_std_val = (T - T.mean()) / (T.std() + 1.0e-10)
        S_std_val = (S - S.mean()) / (S.std() + 1.0e-10)
        D_std_val = (D - D.mean()) / (D.std() + 1.0e-10)
        W_std_val = (W - W.mean()) / (W.std() + 1.0e-10)
        age_std_val = (age - age.mean()) / (age.std() + 1.0e-10)
        edu_std_val = (
            (education - education.mean()) /
            (education.std() + 1.0e-10)
        )
        children_std_val = (
            (children - children.mean()) /
            (children.std() + 1.0e-10)
        )

        base_linear = (
            self.eta_T * T_std_val +
            self.eta_S * S_std_val +
            self.eta_D * D_std_val +
            self.eta_W * W_std_val +
            self.eta_age * age_std_val +
            self.eta_edu * edu_std_val +
            self.eta_children * children_std_val
        )
        eta0 = solve_logit_shift_for_target(
            base_linear[employed_mask],
            self.target_separation_rate,
            lower=self.eta0_bounds[0],
            upper=self.eta0_bounds[1],
        )

        if verbose:
            raw_mean = float(stable_sigmoid(base_linear[employed_mask]).mean())
            adjusted_mean = float(
                stable_sigmoid(base_linear[employed_mask] + eta0).mean()
            )
            logger.info(
                "离职率截距校准完成：raw=%.4f, target=%.4f, adjusted=%.4f, eta0=%.4f",
                raw_mean,
                self.target_separation_rate,
                adjusted_mean,
                eta0,
            )

        return float(eta0)

    def compute_separation_rates(
        self,
        individuals: pd.DataFrame
    ) -> np.ndarray:
        """计算就业者离职率，对失业者返回 0。"""
        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        education = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)
        employed_mask = (
            individuals['employment_status'].to_numpy() == 'employed'
        )

        z = (
            self.eta0 +
            self.eta_T * ((T - T.mean()) / (T.std() + 1.0e-10)) +
            self.eta_S * ((S - S.mean()) / (S.std() + 1.0e-10)) +
            self.eta_D * ((D - D.mean()) / (D.std() + 1.0e-10)) +
            self.eta_W * ((W - W.mean()) / (W.std() + 1.0e-10)) +
            self.eta_age * ((age - age.mean()) / (age.std() + 1.0e-10)) +
            self.eta_edu * (
                (education - education.mean()) /
                (education.std() + 1.0e-10)
            ) +
            self.eta_children * (
                (children - children.mean()) /
                (children.std() + 1.0e-10)
            )
        )

        mu_probs = np.zeros(len(individuals), dtype=float)
        mu_probs[employed_mask] = stable_sigmoid(z[employed_mask])
        return mu_probs

    def compute_match_probabilities(
        self,
        individuals: pd.DataFrame,
        optimal_effort: pd.Series,
        theta: float
    ) -> np.ndarray:
        """计算失业者在最优努力下的匹配概率。"""
        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        unemployed_mask = (
            individuals['employment_status'].to_numpy() == 'unemployed'
        )
        effort_array = np.asarray(optimal_effort, dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        education = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)

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

        probs = self._predict_match_probabilities(
            T=T_new,
            S=S_new,
            D=D_new,
            W=W_new,
            sigma=sigma,
            theta=np.full(len(T), float(theta), dtype=float)
        )
        lambda_probs = np.zeros(len(individuals), dtype=float)
        lambda_probs[unemployed_mask] = probs[unemployed_mask]
        return lambda_probs

    def evolve(
        self,
        individuals: pd.DataFrame,
        optimal_effort: pd.Series,
        theta: float,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """执行一期人口演化。"""
        N = len(individuals)

        if verbose:
            logger.info("计算匹配概率和离职率...")
        lambda_probs = self.compute_match_probabilities(
            individuals, optimal_effort, theta
        )
        mu_probs = self.compute_separation_rates(individuals)

        employment_status = (
            individuals['employment_status'].to_numpy() == 'unemployed'
        )
        employed_status = ~employment_status

        T = individuals['T'].to_numpy(dtype=float)
        S = individuals['S'].to_numpy(dtype=float)
        D = individuals['D'].to_numpy(dtype=float)
        W = individuals['W'].to_numpy(dtype=float)
        current_wage = individuals['current_wage'].to_numpy(dtype=float)
        age = individuals['age'].to_numpy(dtype=float)
        edu = individuals['education'].to_numpy(dtype=float)
        children = individuals['children'].to_numpy(dtype=float)

        T_max_population = T.max()
        W_min_population = W.min()
        S_min = S.min()
        S_max = S.max()
        D_min = D.min()
        D_max = D.max()

        # 计算当前人口的 S/D 均值，用于工资技能溢价计算
        S_mean = float(S.mean())
        D_mean = float(D.mean())

        if verbose:
            logger.info("开始人口演化（Numba加速）...")
        (
            employment_status_new,
            T_new,
            S_new,
            D_new,
            W_new,
            current_wage_new
        ) = simulate_population_evolution(
            employment_status, T, S, D, W, current_wage,
            age, edu, children,
            np.asarray(optimal_effort, dtype=float),
            lambda_probs, mu_probs,
            T_max_population, W_min_population,
            S_min, S_max, D_min, D_max,
            self.gamma_T,
            self.gamma_W,
            self.gamma_S,
            self.gamma_D,
            self.sigma_match,
            self.mean_wage_firm,
            self.std_wage_firm,
            S_mean,
            D_mean,
            self.beta_wage_S,
            self.beta_wage_D,
            self.min_wage,
        )

        individuals_next = individuals.copy()
        individuals_next['employment_status'] = np.where(
            employment_status_new, 'unemployed', 'employed'
        )
        individuals_next['T'] = T_new
        individuals_next['S'] = S_new
        individuals_next['D'] = D_new
        individuals_next['W'] = W_new
        individuals_next['current_wage'] = current_wage_new

        n_unemployed = int(employment_status_new.sum())
        n_employed = N - n_unemployed
        unemployment_rate = n_unemployed / N
        theta_next = self.target_theta

        n_unemployed_prev = int(employment_status.sum())
        n_employed_prev = int(employed_status.sum())
        job_finding_rate_expected = (
            float(lambda_probs[employment_status].mean())
            if n_unemployed_prev > 0
            else 0.0
        )
        separation_rate_expected = (
            float(mu_probs[employed_status].mean())
            if n_employed_prev > 0
            else 0.0
        )

        job_found_count = int(
            np.logical_and(employment_status, ~employment_status_new).sum()
        )
        separated_count = int(
            np.logical_and(employed_status, employment_status_new).sum()
        )
        job_finding_rate_realized = (
            job_found_count / n_unemployed_prev
            if n_unemployed_prev > 0
            else 0.0
        )
        separation_rate_realized = (
            separated_count / n_employed_prev
            if n_employed_prev > 0
            else 0.0
        )

        statistics = {
            'n_unemployed': n_unemployed,
            'n_employed': n_employed,
            'unemployment_rate': unemployment_rate,
            'theta': theta_next,
            'lambda_intercept_shift': self.lambda_intercept_shift,
            'eta0': self.eta0,
            'mean_T': T_new.mean(),
            'mean_S': S_new.mean(),
            'mean_D': D_new.mean(),
            'mean_W': W_new.mean(),
            'job_finding_rate': job_finding_rate_expected,
            'separation_rate': separation_rate_expected,
            'job_finding_rate_expected': job_finding_rate_expected,
            'separation_rate_expected': separation_rate_expected,
            'job_finding_rate_realized': job_finding_rate_realized,
            'separation_rate_realized': separation_rate_realized,
            'n_unemployed_prev': n_unemployed_prev,
            'n_employed_prev': n_employed_prev,
            'mean_wage_employed': (
                current_wage_new[~employment_status_new].mean()
                if n_employed > 0
                else 0.0
            )
        }

        if verbose:
            logger.info(
                "演化完成：失业率=%.2f%%, 市场紧张度θ=%.3f",
                unemployment_rate * 100.0,
                theta_next
            )

        return individuals_next, statistics
