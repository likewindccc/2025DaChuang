#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场模拟器。

负责在不同政策场景下调用 MFG 求解器，进行批量模拟和结果对比。

本版本重点修复三类论文复现实验问题：
1. 所有场景可复用同一份基础人口样本，避免把抽样噪声误写成政策效果。
2. 每个场景使用独立临时 MFG 配置文件与独立原始输出目录，避免相互覆盖。
3. 允许论文脚本复用阶段一基准结果，避免 baseline 被重复随机求解后口径漂移。
"""

import copy
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from MODULES.MFG import solve_equilibrium


logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    市场模拟器。

    管理多场景模拟，包括：
    - 加载场景配置
    - 调整人口分布与 MFG 参数
    - 批量运行 MFG 均衡求解
    - 汇总并保存对比结果
    """

    def __init__(
        self,
        config_path: str,
        mfg_config_path: str = "CONFIG/mfg_config.yaml",
    ):
        """
        初始化市场模拟器。

        参数：
            config_path: SIMULATOR 配置文件路径。
            mfg_config_path: 基准 MFG 配置文件路径。
        """
        self.root_dir = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve_path(config_path)
        self.mfg_config_path = self._resolve_path(mfg_config_path)

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        with open(self.mfg_config_path, "r", encoding="utf-8") as file:
            self.base_mfg_config = yaml.safe_load(file)

        output_dir = Path(self.config["output"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = self.root_dir / output_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 每个场景单独保存原始 MFG 结果，避免批量运行时互相覆盖。
        self.raw_output_root = self.output_dir / "_raw_mfg"
        self.raw_output_root.mkdir(parents=True, exist_ok=True)

        self.temp_config_dir = self.output_dir / "_temp_configs"
        self.temp_config_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path_str: str) -> Path:
        """
        将相对路径解析到项目根目录下，避免依赖当前工作目录。
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _adjust_population_params(self, adjustments: Optional[Dict]) -> Optional[Dict]:
        """
        准备人口分布调整参数。

        这些参数不直接写入配置文件，而是交给 MFG 求解器在初始化人口时
        施加到基础样本副本上。
        """
        return adjustments

    def _get_raw_output_dir(self, scenario_name: str) -> Path:
        """
        获取当前场景对应的 MFG 原始输出目录。
        """
        scenario_output_dir = self.raw_output_root / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        return scenario_output_dir

    def _build_mfg_config(
        self,
        scenario_name: str,
        adjustments: Optional[Dict],
    ) -> Dict:
        """
        基于基准配置构建当前场景的临时 MFG 配置。

        支持的调整项（均为乘法因子）：
        - gamma_S_multiplier: 技能状态更新系数
        - gamma_D_multiplier: 数字素养状态更新系数
        - target_theta_multiplier: 市场紧张度
        - alpha_T_multiplier: 工时偏离负效用系数
        - b0_multiplier: 失业即时收益
        """
        mfg_config = copy.deepcopy(self.base_mfg_config)
        mfg_config.setdefault("paths", {})
        mfg_config["paths"]["output_dir"] = str(
            self._get_raw_output_dir(scenario_name)
        )

        if adjustments is None:
            return mfg_config

        if "gamma_S_multiplier" in adjustments:
            original = mfg_config["economics"]["state_update"]["gamma_S"]
            mfg_config["economics"]["state_update"]["gamma_S"] = (
                original * adjustments["gamma_S_multiplier"]
            )

        if "gamma_D_multiplier" in adjustments:
            original = mfg_config["economics"]["state_update"]["gamma_D"]
            mfg_config["economics"]["state_update"]["gamma_D"] = (
                original * adjustments["gamma_D_multiplier"]
            )

        if "target_theta_multiplier" in adjustments:
            original = mfg_config["market"]["target_theta"]
            mfg_config["market"]["target_theta"] = (
                original * adjustments["target_theta_multiplier"]
            )

        if "alpha_T_multiplier" in adjustments:
            original = mfg_config["economics"]["disutility_T"]["alpha"]
            mfg_config["economics"]["disutility_T"]["alpha"] = (
                original * adjustments["alpha_T_multiplier"]
            )

        if "b0_multiplier" in adjustments:
            original = mfg_config["economics"]["unemployment_benefit"]["b0"]
            mfg_config["economics"]["unemployment_benefit"]["b0"] = (
                original * adjustments["b0_multiplier"]
            )

        return mfg_config

    def _write_temp_mfg_config(
        self,
        scenario_name: str,
        mfg_config: Dict,
    ) -> Path:
        """
        为当前场景写入独立的临时 MFG 配置文件。
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix=f"{scenario_name}_",
            dir=self.temp_config_dir,
            delete=False,
        ) as file:
            yaml.dump(
                mfg_config,
                file,
                allow_unicode=True,
                default_flow_style=False,
            )
            return Path(file.name)

    def run_scenario(
        self,
        scenario_name: str,
        scenario_config: Dict,
        base_population: Optional[pd.DataFrame] = None,
        initial_T: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        运行单个场景。

        参数：
            scenario_name: 场景名称。
            scenario_config: 场景配置字典。
            base_population: 可复用的基础人口样本。
            initial_T: 与基础样本对应的初始 T 数组。

        返回：
            场景结果字典，包含所有关键指标。
        """
        logger.info("%s", "=" * 80)
        logger.info("运行场景: %s", scenario_config["name"])
        logger.info("描述: %s", scenario_config["description"])
        logger.info("%s", "=" * 80)

        population_adjustment = self._adjust_population_params(
            scenario_config.get("population_adjustment")
        )
        mfg_config = self._build_mfg_config(
            scenario_name,
            scenario_config.get("state_update_adjustment")
        )
        temp_config_path = self._write_temp_mfg_config(
            scenario_name,
            mfg_config,
        )

        try:
            individuals_eq, eq_info = solve_equilibrium(
                config_path=str(temp_config_path),
                population_adjustment=population_adjustment,
                base_population=base_population,
                initial_T=initial_T,
            )

            policy_path = (
                self._get_raw_output_dir(scenario_name) / "equilibrium_policy.csv"
            )
            policy_df = pd.read_csv(policy_path, index_col=0)
            policy_df = policy_df.reindex(individuals_eq.index)

            unemployed_mask = (
                individuals_eq["employment_status"] == "unemployed"
            )
            if unemployed_mask.any():
                mean_effort = float(
                    policy_df.loc[unemployed_mask, "a_optimal"].mean()
                )
            else:
                mean_effort = 0.0

            result = {
                "scenario_name": scenario_name,
                "scenario_display_name": scenario_config["name"],
                "policy_type": scenario_config.get("policy_type", "none"),
                "converged": eq_info["converged"],
                "iterations": eq_info["iterations"],
                "unemployment_rate": eq_info["final_statistics"]["unemployment_rate"],
                "mean_T": eq_info["final_statistics"]["mean_T"],
                "mean_S": eq_info["final_statistics"]["mean_S"],
                "mean_D": eq_info["final_statistics"]["mean_D"],
                "mean_W": eq_info["final_statistics"]["mean_W"],
                "mean_wage_employed": (
                    eq_info["final_statistics"]["mean_wage_employed"]
                ),
                "mean_effort": mean_effort,
                "mean_value_U": float(policy_df["V_U"].mean()),
                "mean_value_E": float(policy_df["V_E"].mean()),
            }

            if self.config["output"]["save_detailed_results"]:
                self._save_scenario_results(
                    scenario_name,
                    individuals_eq,
                    eq_info,
                )

            logger.info("场景 '%s' 运行完成", scenario_config["name"])
            logger.info("  收敛状态: %s", result["converged"])
            logger.info("  失业率: %.2f%%", result["unemployment_rate"] * 100)
            logger.info("  平均工资: %.2f", result["mean_wage_employed"])
            logger.info("  平均努力: %.4f", result["mean_effort"])
            if not result["converged"]:
                logger.warning(
                    "场景 '%s' 尚未收敛，结果更适合诊断用途，不建议直接写入主结论。",
                    scenario_config["name"],
                )

            return result
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()

    def _save_scenario_results(
        self,
        scenario_name: str,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> None:
        """
        保存单个场景的详细结果。
        """
        scenario_dir = self.output_dir / f"scenario_{scenario_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        individuals.to_csv(
            scenario_dir / "equilibrium_individuals.csv",
            index=False,
        )

        if "history" in eq_info:
            history_df = pd.DataFrame(eq_info["history"])
            history_df.to_csv(
                scenario_dir / "equilibrium_history.csv",
                index=False,
            )

        summary = {
            "scenario_name": scenario_name,
            "converged": eq_info["converged"],
            "iterations": eq_info["iterations"],
            "final_statistics": eq_info["final_statistics"],
        }

        import pickle

        with open(scenario_dir / "equilibrium_summary.pkl", "wb") as file:
            pickle.dump(summary, file)

        state_vars = ["T", "S", "D", "W", "current_wage"]
        distribution_stats = {}
        for var in state_vars:
            if var in individuals.columns:
                distribution_stats[var] = {
                    "mean": individuals[var].mean(),
                    "std": individuals[var].std(),
                    "min": individuals[var].min(),
                    "q25": individuals[var].quantile(0.25),
                    "median": individuals[var].median(),
                    "q75": individuals[var].quantile(0.75),
                    "max": individuals[var].max(),
                }

        dist_stats_df = pd.DataFrame(distribution_stats).T
        dist_stats_df.to_csv(scenario_dir / "distribution_statistics.csv")

        if "employment_status" in individuals.columns:
            status_comparison = individuals.groupby("employment_status")[
                state_vars
            ].describe()
            status_comparison.to_csv(scenario_dir / "status_comparison.csv")

        if "history" in eq_info:
            time_series = pd.DataFrame(eq_info["history"])
            time_series["scenario_name"] = scenario_name
            time_series.to_csv(
                scenario_dir / "time_series_full.csv",
                index=False,
            )

    def run_batch(
        self,
        base_population: Optional[pd.DataFrame] = None,
        initial_T: Optional[np.ndarray] = None,
        precomputed_baseline_result: Optional[Dict] = None,
        baseline_individuals: Optional[pd.DataFrame] = None,
        baseline_eq_info: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        批量运行所有场景。

        参数：
            base_population: 可复用的基础人口样本。
            initial_T: 与基础样本对应的初始 T 数组。

        返回：
            场景对比汇总表 DataFrame。
        """
        logger.info("%s", "=" * 80)
        logger.info("开始批量场景模拟")
        logger.info("%s", "=" * 80)

        results = []
        for scenario_name, scenario_config in self.config["scenarios"].items():
            if (
                scenario_name == "baseline"
                and precomputed_baseline_result is not None
            ):
                logger.info(
                    "场景 '%s' 直接复用阶段一基准均衡结果，不再重复随机求解。",
                    scenario_config["name"],
                )
                result = dict(precomputed_baseline_result)
                if (
                    self.config["output"]["save_detailed_results"]
                    and baseline_individuals is not None
                    and baseline_eq_info is not None
                ):
                    self._save_scenario_results(
                        scenario_name,
                        baseline_individuals,
                        baseline_eq_info,
                    )
            else:
                result = self.run_scenario(
                    scenario_name,
                    scenario_config,
                    base_population=base_population,
                    initial_T=initial_T,
                )
            results.append(result)

        results_df = pd.DataFrame(results)

        if self.config["output"]["save_comparison_table"]:
            comparison_path = self.output_dir / "scenario_comparison.csv"
            results_df.to_csv(comparison_path, index=False)
            logger.info("场景对比汇总表已保存至: %s", comparison_path)

        all_time_series = []
        for scenario_name in self.config["scenarios"].keys():
            scenario_dir = self.output_dir / f"scenario_{scenario_name}"
            ts_file = scenario_dir / "time_series_full.csv"
            if ts_file.exists():
                ts_data = pd.read_csv(ts_file)
                all_time_series.append(ts_data)

        if all_time_series:
            combined_ts = pd.concat(all_time_series, ignore_index=True)
            combined_ts_path = self.output_dir / "all_scenarios_time_series.csv"
            combined_ts.to_csv(combined_ts_path, index=False)
            logger.info("所有场景时间序列数据已合并保存至: %s", combined_ts_path)

        if "baseline" in self.config["scenarios"] and not results_df.empty:
            baseline_result = results_df[
                results_df["scenario_name"] == "baseline"
            ].iloc[0]
            policy_effects = pd.DataFrame()

            for _, row in results_df.iterrows():
                if row["scenario_name"] == "baseline":
                    continue

                effect = {
                    "scenario_name": row["scenario_name"],
                    "scenario_display_name": row["scenario_display_name"],
                    "converged": row.get("converged", True),
                    "paper_eligible": True,
                    "delta_unemployment_rate": (
                        row["unemployment_rate"]
                        - baseline_result["unemployment_rate"]
                    ) * 100,
                    "delta_mean_wage": (
                        row["mean_wage_employed"]
                        - baseline_result["mean_wage_employed"]
                    ),
                    "delta_mean_T": row["mean_T"] - baseline_result["mean_T"],
                    "delta_mean_S": row["mean_S"] - baseline_result["mean_S"],
                    "delta_mean_D": row["mean_D"] - baseline_result["mean_D"],
                    "pct_change_unemployment": (
                        (row["unemployment_rate"] - baseline_result["unemployment_rate"])
                        / baseline_result["unemployment_rate"]
                    ) * 100,
                    "pct_change_wage": (
                        (row["mean_wage_employed"] - baseline_result["mean_wage_employed"])
                        / baseline_result["mean_wage_employed"]
                    ) * 100,
                }
                policy_effects = pd.concat(
                    [policy_effects, pd.DataFrame([effect])],
                    ignore_index=True,
                )

            if not policy_effects.empty:
                effects_path = self.output_dir / "policy_effects_vs_baseline.csv"
                policy_effects.to_csv(effects_path, index=False)
                logger.info("政策效果（相对基准）已保存至: %s", effects_path)

        logger.info("%s", "=" * 80)
        logger.info("批量场景模拟完成")
        logger.info("%s", "=" * 80)
        return results_df
