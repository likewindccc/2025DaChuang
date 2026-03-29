#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG 均衡求解器。

该模块负责组织 Bellman 方程与 KFE 的交替求解，并在需要时将均衡结果
落盘保存。当前实现也支持在参数校准阶段复用基础人口样本，以减少重复采样
带来的固定开销。
"""

import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .bellman_solver import BellmanSolver, load_match_function_model
from .kfe_solver import KFESolver


logger = logging.getLogger(__name__)


class EquilibriumSolver:
    """
    MFG 均衡求解器。

    主要职责：
    1. 生成或复用基础人口样本。
    2. 协调 BellmanSolver 与 KFESolver 的交替迭代。
    3. 判断均衡是否收敛，并记录历史轨迹。
    4. 在需要时保存均衡结果文件。
    """

    def __init__(
        self,
        config_path: str,
        population_adjustment: Optional[Dict] = None,
        save_results: bool = True,
    ):
        """
        初始化均衡求解器。

        参数：
            config_path: MFG 配置文件路径。
            population_adjustment: 人口分布调整参数，例如培训政策冲击。
            save_results: 是否将求解结果写入输出目录。
        """
        self.save_results = save_results

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        model_path = self.config["paths"]["match_function_model"]
        self.match_model = load_match_function_model(model_path)

        self.bellman_solver = BellmanSolver(self.config, self.match_model)
        self.kfe_solver = KFESolver(self.config, self.match_model)

        self.n_individuals = self.config["population"]["n_individuals"]
        self.target_theta = self.config["market"]["target_theta"]
        self.max_outer_iter = self.config["equilibrium"]["max_outer_iter"]
        self.damping_factor = self.config["equilibrium"]["damping_factor"]
        self.epsilon_V = self.config["equilibrium"]["convergence"]["epsilon_V"]
        self.epsilon_a = self.config["equilibrium"]["convergence"]["epsilon_a"]
        self.epsilon_u = self.config["equilibrium"]["convergence"]["epsilon_u"]
        self.use_relative_tol = self.config["equilibrium"]["convergence"][
            "use_relative_tol"
        ]
        self.population_adjustment = population_adjustment

        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.initial_T: Optional[np.ndarray] = None
        self.history = self._create_empty_history()
        self.lambda_intercept_shift = float(
            getattr(self.bellman_solver, "lambda_intercept_shift", 0.0)
        )
        self.calibrated_eta0 = float(getattr(self.bellman_solver, "eta0", 0.0))

    def _sync_transition_parameters(
        self,
        lambda_intercept_shift: Optional[float] = None,
        eta0: Optional[float] = None,
    ) -> None:
        """
        将结构参数同步到 BellmanSolver 与 KFESolver。

        由于两个求解器都会独立使用匹配概率和离职率函数，因此任何结构
        截距项更新都必须同时写入二者，避免价值函数与人口演化口径不一致。
        """
        if lambda_intercept_shift is not None:
            self.lambda_intercept_shift = float(lambda_intercept_shift)
            self.bellman_solver.set_lambda_intercept_shift(
                self.lambda_intercept_shift
            )
            self.kfe_solver.set_lambda_intercept_shift(
                self.lambda_intercept_shift
            )

        if eta0 is not None:
            self.calibrated_eta0 = float(eta0)
            self.bellman_solver.set_eta0(self.calibrated_eta0)
            self.kfe_solver.set_eta0(self.calibrated_eta0)

    def _calibrate_transition_intercepts(
        self,
        individuals: pd.DataFrame,
        theta: float,
        effort: Optional[pd.Series] = None,
        calibrate_lambda: bool = True,
        calibrate_eta0: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        基于参考样本校准匹配概率平移项和离职率截距。

        参数：
            individuals: 参考样本。
            theta: 参考市场紧张度。
            effort: 参考努力水平；若为空则默认使用全 0 努力。
            calibrate_lambda: 是否校准匹配概率截距平移项。
            calibrate_eta0: 是否同时反解 eta0。
            verbose: 是否输出日志。
        """
        if effort is None:
            effort_series = pd.Series(
                np.zeros(len(individuals), dtype=float),
                index=individuals.index,
            )
        else:
            effort_series = pd.Series(
                np.asarray(effort, dtype=float),
                index=individuals.index,
            )

        if calibrate_lambda:
            lambda_intercept_shift = self.kfe_solver.calibrate_lambda_intercept(
                individuals,
                theta,
                effort_series,
                verbose=verbose,
            )
            self._sync_transition_parameters(
                lambda_intercept_shift=lambda_intercept_shift
            )

        if calibrate_eta0:
            eta0 = self.kfe_solver.calibrate_eta0(
                individuals,
                verbose=verbose,
            )
            self._sync_transition_parameters(eta0=eta0)

    @staticmethod
    def _create_empty_history() -> Dict[str, list]:
        """
        创建一份空的迭代历史记录容器。
        """
        return {
            "iteration": [],
            "theta": [],
            "unemployment_rate": [],
            "mean_T": [],
            "mean_S": [],
            "mean_D": [],
            "mean_W": [],
            "mean_wage_employed": [],
            "mean_value_U": [],
            "mean_value_E": [],
            "mean_effort": [],
            "convergence_V": [],
            "convergence_a": [],
            "convergence_u": [],
        }

    def create_base_population_sample(
        self,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        创建可复用的基础人口样本。

        该样本只包含状态变量与静态特征，不包含会在求解过程中被重写的
        `employment_status` 和 `current_wage`，以便在不同参数点评估时安全复用。

        参数：
            verbose: 是否输出过程日志。

        返回：
            (base_population, initial_T): 基础人口样本与其初始 T 值副本。
        """
        if verbose:
            logger.info("%s", "=" * 80)
            logger.info("生成基础人口样本")
            logger.info("%s", "=" * 80)

        from MODULES.POPULATION import LaborDistribution

        pop_config_path = "CONFIG/population_config.yaml"
        with open(pop_config_path, "r", encoding="utf-8") as file:
            pop_config = yaml.safe_load(file)

        labor_model = LaborDistribution(pop_config)
        labor_model.fit()

        if verbose:
            logger.info("从人口分布中采样 %s 个个体...", self.n_individuals)

        continuous_samples = labor_model.copula_model.sample(self.n_individuals)

        edu_values = list(labor_model.discrete_dist["edu"].keys())
        edu_probs = list(labor_model.discrete_dist["edu"].values())
        edu_samples = np.random.choice(
            edu_values,
            size=self.n_individuals,
            p=edu_probs,
        )

        children_values = list(labor_model.discrete_dist["children"].keys())
        children_probs = list(labor_model.discrete_dist["children"].values())
        children_samples = np.random.choice(
            children_values,
            size=self.n_individuals,
            p=children_probs,
        )

        individuals = continuous_samples.copy()
        individuals["education"] = edu_samples
        individuals["children"] = children_samples

        individuals = self._apply_population_adjustment(
            individuals,
            verbose=verbose,
        )

        initial_t_array = individuals["T"].to_numpy(dtype=float).copy()
        if verbose:
            logger.info(
                "记录初始 T 值：均值 = %.2f 小时/周",
                initial_t_array.mean(),
            )
            logger.info("")

        return individuals, initial_t_array

    def _apply_population_adjustment(
        self,
        individuals: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        对基础人口样本施加政策引起的状态分布冲击。

        该方法既服务于“即时采样后立刻调整”，也服务于“复用同一基础样本时
        再按场景复制并调整”，从而确保不同政策场景共享同一底层人口抽样，
        同时保留政策冲击本身。
        """
        adjusted = individuals.copy(deep=True)

        if self.population_adjustment is None:
            return adjusted

        if verbose:
            logger.info("应用人口分布调整（培训政策）...")

        if "mean_S_multiplier" in self.population_adjustment:
            multiplier = self.population_adjustment["mean_S_multiplier"]
            adjusted["S"] = adjusted["S"] * multiplier
            if verbose:
                logger.info("  技能水平 S × %s", multiplier)

        if "mean_D_multiplier" in self.population_adjustment:
            multiplier = self.population_adjustment["mean_D_multiplier"]
            adjusted["D"] = adjusted["D"] * multiplier
            if verbose:
                logger.info("  数字素养 D × %s", multiplier)

        if verbose:
            logger.info("")

        return adjusted

    def initialize_population(
        self,
        base_population: Optional[pd.DataFrame] = None,
        initial_T: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        初始化人口状态。

        如果外部未提供基础人口样本，则即时生成；否则基于传入样本深拷贝，
        再补齐就业状态和当前工资，并执行一次初始随机匹配。

        参数：
            base_population: 仅包含基础特征的人口样本。
            initial_T: 与基础样本对应的初始 T 数组。
            verbose: 是否输出过程日志。

        返回：
            初始化后的个体状态表。
        """
        if verbose:
            logger.info("%s", "=" * 80)
            logger.info("初始化人口")
            logger.info("%s", "=" * 80)

        generated_from_internal_sampler = base_population is None
        if base_population is None:
            base_population, initial_T = self.create_base_population_sample(
                verbose=verbose,
            )

        if initial_T is None:
            raise ValueError("初始化人口时缺少 initial_T。")

        individuals = base_population.copy(deep=True)
        if self.population_adjustment is not None and not generated_from_internal_sampler:
            individuals = self._apply_population_adjustment(
                individuals,
                verbose=verbose,
            )
        self.initial_T = np.asarray(initial_T, dtype=float).copy()

        individuals["employment_status"] = "unemployed"
        individuals["current_wage"] = 0.0

        if verbose:
            logger.info("初始化完成：%s 个个体，全部失业", self.n_individuals)
            logger.info("运行初始随机匹配...")

        initial_effort = pd.Series(
            np.zeros(self.n_individuals, dtype=float),
            index=individuals.index,
        )
        theta_initial = self.target_theta

        if verbose:
            logger.info("初始市场紧张度 θ = %.4f（外生参数）", theta_initial)

        # 当外部传入 base_population（政策场景复用共享样本）时，
        # 不对匹配截距重校准，保留基准均衡校准好的截距，
        # 使政策冲击（S/D/θ变化）能真实传导到匹配概率，而非被截距抵消。
        # 只有首次基准均衡（base_population 为空，内部生成人口）时才重校准截距。
        calibrate_lambda_on_init = generated_from_internal_sampler
        self._calibrate_transition_intercepts(
            individuals,
            theta=theta_initial,
            effort=initial_effort,
            calibrate_lambda=calibrate_lambda_on_init,
            calibrate_eta0=False,
            verbose=verbose,
        )

        lambda_probs = self.kfe_solver.compute_match_probabilities(
            individuals,
            initial_effort,
            theta_initial,
        )
        matched_mask = np.random.random(self.n_individuals) < lambda_probs
        n_matched = int(matched_mask.sum())

        individuals.loc[matched_mask, "employment_status"] = "employed"
        individuals.loc[matched_mask, "current_wage"] = individuals.loc[
            matched_mask,
            "W",
        ]

        n_employed = int(
            (individuals["employment_status"] == "employed").sum()
        )
        initial_u_rate = 1.0 - n_employed / self.n_individuals

        if verbose:
            logger.info("初始匹配完成：%s 人匹配成功", n_matched)
            logger.info("初始失业率 = %.2f%%", initial_u_rate * 100)
        self._calibrate_transition_intercepts(
            individuals,
            theta=theta_initial,
            effort=initial_effort,
            calibrate_lambda=False,
            calibrate_eta0=True,
            verbose=verbose,
        )
        if verbose:
            logger.info("")

        return individuals

    def solve(
        self,
        individuals: Optional[pd.DataFrame] = None,
        verbose: bool = True,
        callback: Optional[Callable[[int, Dict], None]] = None,
        base_population: Optional[pd.DataFrame] = None,
        initial_T: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        求解 MFG 稳态均衡。

        参数：
            individuals: 初始人口状态；若为空则自动初始化。
            verbose: 是否输出详细日志。
            callback: 可选回调函数，签名为 `callback(iteration, stats)`。
            base_population: 可复用的基础人口样本。
            initial_T: 与基础样本配套的初始 T 值数组。

        返回：
            (individuals_equilibrium, equilibrium_info)。
        """
        self.history = self._create_empty_history()
        created_internal_population = individuals is None

        if individuals is None:
            individuals = self.initialize_population(
                base_population=base_population,
                initial_T=initial_T,
                verbose=verbose,
            )
        elif initial_T is not None:
            self.initial_T = np.asarray(initial_T, dtype=float).copy()

        if self.initial_T is None:
            raise ValueError("求解均衡前缺少 initial_T，无法计算 Bellman 方程。")

        if (
            not created_internal_population
            and individuals is not None
            and "employment_status" in individuals.columns
        ):
            self._calibrate_transition_intercepts(
                individuals,
                theta=self.target_theta,
                effort=pd.Series(
                    np.zeros(len(individuals), dtype=float),
                    index=individuals.index,
                ),
                calibrate_lambda=True,
                calibrate_eta0=True,
                verbose=verbose,
            )

        if verbose:
            logger.info("%s", "=" * 80)
            logger.info("开始求解 MFG 均衡")
            logger.info("%s", "=" * 80)
            logger.info("最大外层迭代轮数: %s", self.max_outer_iter)
            logger.info(
                "阻尼因子: %.4f (V_new = %.4f * V_computed + %.4f * V_old)",
                self.damping_factor,
                self.damping_factor,
                1 - self.damping_factor,
            )
            if self.use_relative_tol:
                logger.info(
                    "收敛阈值: ΔV=%.6f（相对）, Δa=%.6f, Δu=%.6f",
                    self.epsilon_V,
                    self.epsilon_a,
                    self.epsilon_u,
                )
            else:
                logger.info(
                    "收敛阈值: ΔV=%.6f, Δa=%.6f, Δu=%.6f",
                    self.epsilon_V,
                    self.epsilon_a,
                    self.epsilon_u,
                )

        prev_V_U = None
        prev_V_E = None
        prev_a_optimal = None
        prev_u_rate = None

        last_V_U = None
        last_V_E = None
        last_a_optimal = None
        last_theta = self.target_theta
        last_u_rate = np.nan
        last_stats: Dict = {}

        for outer_iter in range(self.max_outer_iter):
            if verbose:
                logger.info("%s", "=" * 80)
                logger.info(
                    "外层迭代 %s/%s",
                    outer_iter + 1,
                    self.max_outer_iter,
                )
                logger.info("%s", "=" * 80)

            unemployed_mask = individuals["employment_status"] == "unemployed"
            employed_mask = individuals["employment_status"] == "employed"
            n_unemployed = int(unemployed_mask.sum())
            n_employed = int(employed_mask.sum())
            u_rate = n_unemployed / self.n_individuals
            theta = self.target_theta

            if verbose:
                logger.info("失业人数: %s, 就业人数: %s", n_unemployed, n_employed)
                logger.info("失业率: %.2f%%", u_rate * 100)
                logger.info("市场紧张度 θ = %.4f", theta)
                logger.info("步骤 1: 求解 Bellman 方程...")

            V_U_computed, V_E_computed, a_optimal = self.bellman_solver.solve(
                individuals,
                theta,
                self.initial_T,
                initial_V_U=prev_V_U,
                initial_V_E=prev_V_E,
                verbose=verbose,
            )

            if outer_iter > 0 and prev_V_U is not None and prev_V_E is not None:
                V_U = (
                    self.damping_factor * V_U_computed
                    + (1 - self.damping_factor) * prev_V_U
                )
                V_E = (
                    self.damping_factor * V_E_computed
                    + (1 - self.damping_factor) * prev_V_E
                )
                if verbose:
                    logger.info("应用阻尼更新（权重 = %.4f）", self.damping_factor)
            else:
                V_U = np.asarray(V_U_computed, dtype=float).copy()
                V_E = np.asarray(V_E_computed, dtype=float).copy()

            mean_V_U = float(np.mean(V_U[unemployed_mask])) if n_unemployed > 0 else 0.0
            mean_V_E = float(np.mean(V_E[employed_mask])) if n_employed > 0 else 0.0
            mean_a = float(np.mean(a_optimal[unemployed_mask])) if n_unemployed > 0 else 0.0

            if verbose:
                logger.info("平均失业价值函数: %.4f", mean_V_U)
                logger.info("平均就业价值函数: %.4f", mean_V_E)
                logger.info("平均最优努力: %.4f", mean_a)
                logger.info("步骤 2: 求解 KFE（人口演化）...")

            individuals_next, stats = self.kfe_solver.evolve(
                individuals,
                a_optimal,
                theta,
                verbose=verbose,
            )
            u_rate_next = float(stats["unemployment_rate"])

            if verbose:
                logger.info(
                    "演化后失业率: %.2f%%",
                    u_rate_next * 100,
                )
                logger.info("平均 T: %.4f", stats["mean_T"])
                logger.info("平均 S: %.4f", stats["mean_S"])
                logger.info("平均 D: %.4f", stats["mean_D"])
                logger.info("平均 W: %.4f", stats["mean_W"])

            if outer_iter > 0 and prev_V_U is not None and prev_V_E is not None:
                diff_V_U_abs = float(np.max(np.abs(V_U - prev_V_U)))
                diff_V_E_abs = float(np.max(np.abs(V_E - prev_V_E)))

                if self.use_relative_tol:
                    V_U_magnitude = float(np.mean(np.abs(V_U))) + 1e-10
                    V_E_magnitude = float(np.mean(np.abs(V_E))) + 1e-10
                    diff_V = max(
                        diff_V_U_abs / V_U_magnitude,
                        diff_V_E_abs / V_E_magnitude,
                    )
                else:
                    diff_V = max(diff_V_U_abs, diff_V_E_abs)

                diff_a = float(
                    abs(np.mean(a_optimal) - np.mean(prev_a_optimal))
                )
                diff_u = float(abs(u_rate_next - prev_u_rate))
            else:
                diff_V = np.nan
                diff_a = np.nan
                diff_u = np.nan

            self.history["iteration"].append(outer_iter + 1)
            self.history["theta"].append(theta)
            # 历史轨迹统一记录“演化后”的总体统计口径，与 final_statistics 对齐。
            self.history["unemployment_rate"].append(u_rate_next)
            self.history["mean_T"].append(stats["mean_T"])
            self.history["mean_S"].append(stats["mean_S"])
            self.history["mean_D"].append(stats["mean_D"])
            self.history["mean_W"].append(stats["mean_W"])
            self.history["mean_wage_employed"].append(
                stats.get("mean_wage_employed", 0.0)
            )
            self.history["mean_value_U"].append(mean_V_U)
            self.history["mean_value_E"].append(mean_V_E)
            self.history["mean_effort"].append(mean_a)
            self.history["convergence_V"].append(diff_V)
            self.history["convergence_a"].append(diff_a)
            self.history["convergence_u"].append(diff_u)

            if callback is not None:
                callback_stats = {
                    "unemployment_rate": u_rate_next,
                    "theta": theta,
                    "mean_wage": float(individuals_next["current_wage"].mean()),
                    "mean_T": stats["mean_T"],
                    "mean_S": stats["mean_S"],
                    "diff_V": 0.0 if np.isnan(diff_V) else diff_V,
                    "diff_u": 0.0 if np.isnan(diff_u) else diff_u,
                }
                callback(outer_iter + 1, callback_stats)

            last_V_U = V_U
            last_V_E = V_E
            last_a_optimal = np.asarray(a_optimal, dtype=float).copy()
            last_theta = theta
            last_u_rate = u_rate_next
            last_stats = stats

            if outer_iter > 0:
                if verbose:
                    if self.use_relative_tol:
                        logger.info(
                            "收敛检查: |ΔV|/|V|=%.6f, |Δmean(a)|=%.6f, |Δu|=%.6f",
                            diff_V,
                            diff_a,
                            diff_u,
                        )
                    else:
                        logger.info(
                            "收敛检查: |ΔV|=%.6f, |Δmean(a)|=%.6f, |Δu|=%.6f",
                            diff_V,
                            diff_a,
                            diff_u,
                        )

                if (
                    diff_V < self.epsilon_V
                    and diff_a < self.epsilon_a
                    and diff_u < self.epsilon_u
                ):
                    if verbose:
                        logger.info("%s", "=" * 80)
                        logger.info("均衡已收敛，共迭代 %s 轮", outer_iter + 1)
                        logger.info("%s", "=" * 80)

                    self._save_equilibrium(
                        individuals_next,
                        last_V_U,
                        last_V_E,
                        last_a_optimal,
                        outer_iter + 1,
                        converged=True,
                    )
                    return individuals_next, {
                        "converged": True,
                        "iterations": outer_iter + 1,
                        "final_unemployment_rate": last_stats.get(
                            "unemployment_rate",
                            last_u_rate,
                        ),
                        "final_theta": last_theta,
                        "lambda_intercept_shift": self.lambda_intercept_shift,
                        "eta0": self.calibrated_eta0,
                        "final_statistics": last_stats,
                        "history": self.history,
                    }

            individuals = individuals_next.copy(deep=True)
            prev_V_U = np.asarray(V_U, dtype=float).copy()
            prev_V_E = np.asarray(V_E, dtype=float).copy()
            prev_a_optimal = np.asarray(a_optimal, dtype=float).copy()
            prev_u_rate = u_rate_next

        if verbose:
            logger.warning(
                "达到最大外层迭代次数 %s，均衡仍未收敛。",
                self.max_outer_iter,
            )

        if last_V_U is None or last_V_E is None or last_a_optimal is None:
            raise RuntimeError("均衡求解未执行任何有效迭代。")

        self._save_equilibrium(
            individuals,
            last_V_U,
            last_V_E,
            last_a_optimal,
            self.max_outer_iter,
            converged=False,
        )
        return individuals, {
            "converged": False,
            "iterations": self.max_outer_iter,
            "final_unemployment_rate": last_stats.get(
                "unemployment_rate",
                last_u_rate,
            ),
            "final_theta": last_theta,
            "lambda_intercept_shift": self.lambda_intercept_shift,
            "eta0": self.calibrated_eta0,
            "final_statistics": last_stats,
            "history": self.history,
        }

    def _save_equilibrium(
        self,
        individuals: pd.DataFrame,
        V_U: np.ndarray,
        V_E: np.ndarray,
        a_optimal: np.ndarray,
        iterations: int,
        converged: bool,
    ) -> None:
        """
        保存均衡求解结果。

        参数：
            individuals: 均衡时刻的个体状态。
            V_U: 失业状态价值函数。
            V_E: 就业状态价值函数。
            a_optimal: 最优努力策略。
            iterations: 迭代轮数。
            converged: 是否已经收敛。
        """
        if not self.save_results:
            return

        logger.info("保存均衡结果到 %s", self.output_dir)

        v_u_series = pd.Series(np.asarray(V_U, dtype=float), index=individuals.index)
        v_e_series = pd.Series(np.asarray(V_E, dtype=float), index=individuals.index)
        a_series = pd.Series(
            np.asarray(a_optimal, dtype=float),
            index=individuals.index,
        )
        employed_mask = individuals["employment_status"] == "employed"
        a_series.loc[employed_mask] = 0.0

        individuals_path = self.output_dir / "equilibrium_individuals.csv"
        individuals.to_csv(individuals_path, index=False, encoding="utf-8-sig")

        policy_df = pd.DataFrame(
            {
                "V_U": v_u_series,
                "V_E": v_e_series,
                "a_optimal": a_series,
            }
        )
        policy_path = self.output_dir / "equilibrium_policy.csv"
        policy_df.to_csv(policy_path, index=True, encoding="utf-8-sig")

        history_df = pd.DataFrame(self.history)
        history_path = self.output_dir / "equilibrium_history.csv"
        history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

        summary = {
            "converged": converged,
            "iterations": iterations,
            "n_individuals": self.n_individuals,
            "target_theta": self.target_theta,
            "lambda_intercept_shift": self.lambda_intercept_shift,
            "eta0": self.calibrated_eta0,
            "final_unemployment_rate": float(
                (individuals["employment_status"] == "unemployed").mean()
            ),
            "final_theta": self.history["theta"][-1],
        }
        summary_path = self.output_dir / "equilibrium_summary.pkl"
        with open(summary_path, "wb") as file:
            pickle.dump(summary, file)

        value_distribution = pd.DataFrame(
            {
                "individual_id": individuals.index,
                "V_U": v_u_series,
                "V_E": v_e_series,
                "delta_V": v_e_series - v_u_series,
                "a_optimal": a_series,
                "employment_status": individuals["employment_status"],
                "T": individuals["T"],
                "S": individuals["S"],
                "D": individuals["D"],
                "W": individuals["W"],
            }
        )
        value_dist_path = self.output_dir / "value_distribution_full.csv"
        value_distribution.to_csv(
            value_dist_path,
            index=False,
            encoding="utf-8-sig",
        )

        status_summary = individuals.groupby("employment_status").agg(
            {
                "T": ["mean", "std", "min", "max"],
                "S": ["mean", "std", "min", "max"],
                "D": ["mean", "std", "min", "max"],
                "W": ["mean", "std", "min", "max"],
                "current_wage": ["mean", "std", "count"],
            }
        ).round(2)
        status_summary_path = self.output_dir / "status_comparison_summary.csv"
        status_summary.to_csv(status_summary_path, encoding="utf-8-sig")

    def solve_equilibrium(
        self,
        verbose: bool = True,
        base_population: Optional[pd.DataFrame] = None,
        initial_T: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        类方法形式的均衡求解入口。

        该方法仅对 `solve` 做一层轻量包装，便于外部按面向对象方式调用。
        """
        return self.solve(
            verbose=verbose,
            base_population=base_population,
            initial_T=initial_T,
        )


def solve_equilibrium(
    config_path: str = "CONFIG/mfg_config.yaml",
    population_adjustment: Optional[Dict] = None,
    save_results: bool = True,
    base_population: Optional[pd.DataFrame] = None,
    initial_T: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    求解 MFG 均衡的便捷函数。

    参数：
        config_path: MFG 配置文件路径。
        population_adjustment: 可选的人口分布调整参数。
        save_results: 是否将结果写入磁盘。
        base_population: 可复用的基础人口样本。
        initial_T: 与基础样本对应的初始 T 数组。
        verbose: 是否输出详细日志。

    返回：
        (individuals_equilibrium, equilibrium_info)。
    """
    solver = EquilibriumSolver(config_path, population_adjustment, save_results)
    return solver.solve(
        verbose=verbose,
        base_population=base_population,
        initial_T=initial_T,
    )
