#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文数据生成脚本（服务器端运行）

功能：
1. 运行MFG基准均衡求解（10,000个体，200轮迭代）
2. 批量运行11个政策场景模拟（baseline 复用阶段一结果）
3. 运行参数灵敏度分析（ρ/κ/α 各±10%/±20%）
4. 汇总导出论文所需的全部结构化数据表

输出目录：
  OUTPUT/paper_tables/  ── 论文所需的结构化数据表
  OUTPUT/mfg/           ── MFG基准均衡原始数据
  OUTPUT/simulation/    ── 政策模拟原始数据

使用方式（服务器端）：
  python run_paper_simulation.py                 # 完整运行
  python run_paper_simulation.py --skip-sensitivity  # 跳过灵敏度分析

注意：
  - 完整运行预计耗时12-16小时（含灵敏度分析）
  - 不包含绘图，绘图请在本地运行 plot_paper_figures.py
"""

import sys
import os
import argparse
import logging
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "OUTPUT" / "paper_simulation.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("PaperSimulation")


# 论文结果导出统一使用的目标矩集合。
# 当前口径仅使用 M1/M2/M3/M4/M7/M8，明确排除 M5/M6。
# 这里显式列出活跃矩，避免后续只看 target_moments.yaml 或
# calibration_config.yaml 时，对“论文表4.4到底认哪些矩”产生歧义。
PAPER_TARGET_MOMENT_NAMES = [
    "unemployment_rate",
    "mean_wage",
    "log_std_wage",
    "mean_weekly_hours",
    "wage_iqr_ratio",
    "std_weekly_hours",
]


# ============================================================
# 工具函数
# ============================================================

def timer(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("  → %s 耗时 %.1f 分钟", func.__name__, elapsed / 60)
        return result
    return wrapper


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_latest_calibrated_parameters(config: dict) -> dict:
    """
    将最新校准参数覆盖到论文运行配置中。

    说明：
        `run_paper_simulation.py` 的求解链路实际使用临时 MFG 配置。
        这里在创建临时配置时，显式读取
        `OUTPUT/calibration/calibrated_parameters.yaml`，避免论文脚本继续
        使用过期的 `CONFIG/mfg_config.yaml` 参数快照。
    """
    cal_path = project_root / "OUTPUT" / "calibration" / "calibrated_parameters.yaml"
    if not cal_path.exists():
        logger.warning("未找到校准结果文件: %s，论文脚本将继续使用基准配置参数。", cal_path)
        return config

    with open(cal_path, "r", encoding="utf-8") as file:
        cal_data = yaml.safe_load(file)

    calibration_info = cal_data.get("calibration_info", {})
    if not calibration_info.get("success", True):
        logger.warning(
            "当前校准结果 success=false，论文脚本仍将同步这份最新参数快照，请注意结果解释口径。"
        )

    params = cal_data.get("parameters", {})
    if not params:
        logger.warning("校准结果文件中缺少 parameters 字段，跳过参数同步。")
        return config

    path_mapping = {
        "rho": ("economics", "rho"),
        "kappa": ("economics", "kappa"),
        "alpha_T": ("economics", "disutility_T", "alpha"),
        "gamma_T": ("economics", "state_update", "gamma_T"),
        "gamma_S": ("economics", "state_update", "gamma_S"),
        "gamma_D": ("economics", "state_update", "gamma_D"),
        "gamma_W": ("economics", "state_update", "gamma_W"),
    }

    for param_name, config_path in path_mapping.items():
        if param_name not in params:
            continue

        target = config
        for key in config_path[:-1]:
            target = target[key]
        target[config_path[-1]] = float(params[param_name])

    logger.info(
        "论文临时配置已同步最新校准参数：timestamp=%s",
        calibration_info.get("timestamp", "unknown"),
    )
    return config


def load_paper_target_moments():
    """
    加载论文导表使用的统一目标矩配置。

    当前明确使用 M1/M2/M3/M4/M7/M8 六个活跃矩，并从
    CONFIG/target_moments.yaml 读取对应的目标值与元数据。
    """
    from MODULES.CALIBRATION.target_moments import TargetMoments

    return TargetMoments(
        "CONFIG/target_moments.yaml",
        selected_moments=PAPER_TARGET_MOMENT_NAMES,
    )


def create_paper_mfg_config() -> Path:
    """
    创建论文正式运行使用的临时 MFG 配置文件。

    该配置统一指定：
    - 样本规模：10,000 个体
    - 最大外层迭代：200 轮
    """
    base_config_path = project_root / "CONFIG" / "mfg_config.yaml"
    with open(base_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config = apply_latest_calibrated_parameters(config)
    config["population"]["n_individuals"] = 10000
    config["equilibrium"]["max_outer_iter"] = 200

    temp_dir = ensure_dir(project_root / "OUTPUT" / "_temp_configs")
    temp_config_path = temp_dir / "mfg_config_paper.yaml"

    with open(temp_config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True, default_flow_style=False)

    logger.info("已创建论文运行临时配置: %s", temp_config_path)
    return temp_config_path


def create_shared_population_sample(config_path: Path) -> tuple:
    """
    创建供基准场景和全部政策场景共用的基础人口样本。

    这样可以把政策效果与人口重采样噪声分离开来。
    """
    from MODULES.MFG.equilibrium_solver import EquilibriumSolver

    solver = EquilibriumSolver(str(config_path), save_results=False)
    base_population, initial_T = solver.create_base_population_sample(
        verbose=False
    )
    logger.info(
        "已生成共享基础人口样本：%d 个体",
        len(base_population),
    )
    return base_population, initial_T


# ============================================================
# 阶段一：MFG 基准均衡求解
# ============================================================

@timer
def run_baseline_equilibrium(
    config_path: Path,
    base_population: pd.DataFrame,
    initial_T: np.ndarray,
) -> tuple:
    """
    运行MFG基准均衡求解

    使用校准后的参数，10,000个体，200轮迭代。
    结果自动保存到 OUTPUT/mfg/ 目录。

    返回:
        (individuals_eq, eq_info, baseline_policy_df):
        均衡个体数据、求解信息和基准策略快照
    """
    logger.info("=" * 80)
    logger.info("阶段一：MFG 基准均衡求解")
    logger.info("=" * 80)

    from MODULES.MFG import solve_equilibrium

    individuals_eq, eq_info = solve_equilibrium(
        config_path=str(config_path),
        base_population=base_population,
        initial_T=initial_T,
    )

    baseline_policy_path = Path("OUTPUT/mfg/equilibrium_policy.csv")
    baseline_policy_df = pd.read_csv(
        baseline_policy_path,
        index_col=0,
    ).reindex(individuals_eq.index)

    baseline_policy_snapshot_path = (
        ensure_dir(Path("OUTPUT/paper_tables"))
        / "baseline_equilibrium_policy.csv"
    )
    baseline_policy_df.to_csv(
        baseline_policy_snapshot_path,
        index=True,
        encoding="utf-8-sig",
    )

    logger.info("基准均衡求解完成")
    logger.info("  收敛状态: %s", eq_info["converged"])
    logger.info("  迭代轮数: %s", eq_info["iterations"])
    logger.info("  最终失业率: %.2f%%", eq_info["final_unemployment_rate"] * 100)
    logger.info("  已保存基准策略快照: %s", baseline_policy_snapshot_path)
    return individuals_eq, eq_info, baseline_policy_df


# ============================================================
# 阶段二：政策场景批量模拟
# ============================================================

@timer
def run_policy_simulations(
    config_path: Path,
    base_population: pd.DataFrame,
    initial_T: np.ndarray,
    baseline_result: dict,
    baseline_individuals_eq: pd.DataFrame,
    baseline_eq_info: dict,
) -> pd.DataFrame:
    """
    批量运行全部11个政策场景模拟

    调用 MarketSimulator.run_batch() 运行所有场景，并复用与基准场景
    相同的基础人口样本。baseline 场景直接复用阶段一结果，避免同口径
    下重复随机求解后出现第二个 baseline 失业率。

    返回:
        场景对比汇总表 DataFrame
    """
    logger.info("=" * 80)
    logger.info("阶段二：政策场景批量模拟")
    logger.info("=" * 80)

    from MODULES.SIMULATOR import MarketSimulator

    simulator = MarketSimulator(
        "CONFIG/simulator_config.yaml",
        mfg_config_path=str(config_path),
    )
    results_df = simulator.run_batch(
        base_population=base_population,
        initial_T=initial_T,
        precomputed_baseline_result=baseline_result,
        baseline_individuals=baseline_individuals_eq,
        baseline_eq_info=baseline_eq_info,
    )

    logger.info("政策模拟完成，共 %d 个场景", len(results_df))
    return results_df


# ============================================================
# 阶段三：参数灵敏度分析
# ============================================================

@timer
def run_sensitivity_analysis() -> pd.DataFrame:
    """
    参数灵敏度分析

    对关键参数 ρ、κ、α_T 在校准值附近做 ±10%、±20% 扰动，
    每次运行一个精简版均衡（5,000个体、100轮迭代）以节省时间。

    注意：
    当前灵敏度分析有意不复用阶段一/二的共享基础人口样本，而是让每个
    扰动点独立重新采样。这样做的目的，是把“参数扰动后的完整均衡求解”
    当作独立实验来跑；如果后续要做更严格的局部比较，可再补固定样本版。

    返回:
        灵敏度分析结果 DataFrame
    """
    logger.info("=" * 80)
    logger.info("阶段三：参数灵敏度分析")
    logger.info("=" * 80)
    logger.info(
        "说明：灵敏度分析当前故意不复用共享样本，每个扰动点独立重新采样。"
    )

    from MODULES.MFG import solve_equilibrium

    # 读取基准配置
    config_path = "CONFIG/mfg_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 基准参数值
    base_params = {
        "rho": base_config["economics"]["rho"],
        "kappa": base_config["economics"]["kappa"],
        "alpha_T": base_config["economics"]["disutility_T"]["alpha"],
    }

    # 扰动比例
    perturbations = [-0.20, -0.10, 0.10, 0.20]

    results = []

    # 对每个参数进行扰动
    for param_name, base_value in base_params.items():
        for pct in perturbations:
            perturbed_value = base_value * (1 + pct)
            label = f"{param_name}_{pct:+.0%}"
            logger.info("运行灵敏度分析: %s = %.4f (基准 %.4f, 扰动 %+.0f%%)",
                        param_name, perturbed_value, base_value, pct * 100)

            # 创建临时配置
            temp_config = yaml.safe_load(yaml.dump(base_config))
            temp_config["population"]["n_individuals"] = 5000
            temp_config["equilibrium"]["max_outer_iter"] = 100

            # 应用扰动，并对 rho 做经济约束检查（rho 须在 (0, 1) 开区间内）
            if param_name == "rho":
                # ρ ≥ 1 在经济上无意义（未来比现在更有价值），模型会崩溃到角点解。
                # 超出上界的扰动点直接跳过，不记录入结果表，论文表7.1不显示这些行。
                if perturbed_value >= 1.0:
                    logger.warning(
                        "灵敏度分析跳过 %s：扰动后 ρ=%.4f ≥ 1.0，超出经济有效区间 (0,1)。",
                        label, perturbed_value,
                    )
                    continue
                temp_config["economics"]["rho"] = perturbed_value
            elif param_name == "kappa":
                temp_config["economics"]["kappa"] = perturbed_value
            elif param_name == "alpha_T":
                temp_config["economics"]["disutility_T"]["alpha"] = perturbed_value

            # 保存临时配置
            temp_path = f"CONFIG/mfg_config_sensitivity_{param_name}_{pct:+.0f}.yaml"
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(temp_config, f, allow_unicode=True, default_flow_style=False)

            try:
                individuals_eq, eq_info = solve_equilibrium(
                    temp_path, save_results=False, verbose=False
                )

                # 读取最终状态
                history = eq_info["history"]
                result = {
                    "parameter": param_name,
                    "perturbation_pct": pct * 100,
                    "base_value": base_value,
                    "perturbed_value": perturbed_value,
                    "unemployment_rate": eq_info["final_unemployment_rate"],
                    "mean_S": history["mean_S"][-1],
                    "mean_D": history["mean_D"][-1],
                    "mean_T": history["mean_T"][-1],
                    "mean_wage": history["mean_wage_employed"][-1],
                    "mean_effort": history["mean_effort"][-1],
                    "converged": eq_info["converged"],
                    "iterations": eq_info["iterations"],
                }
                results.append(result)
                logger.info("  → 失业率 = %.2f%%, 平均工资 = %.2f",
                            result["unemployment_rate"] * 100, result["mean_wage"])

            except Exception as e:
                logger.warning("灵敏度分析 %s 失败: %s", label, e)
                results.append({
                    "parameter": param_name,
                    "perturbation_pct": pct * 100,
                    "base_value": base_value,
                    "perturbed_value": perturbed_value,
                    "unemployment_rate": np.nan,
                    "mean_S": np.nan,
                    "mean_D": np.nan,
                    "mean_T": np.nan,
                    "mean_wage": np.nan,
                    "mean_effort": np.nan,
                    "converged": False,
                    "iterations": 0,
                })
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    sensitivity_df = pd.DataFrame(results)
    logger.info("灵敏度分析完成，共 %d 组扰动", len(sensitivity_df))
    return sensitivity_df


# ============================================================
# 阶段四：论文数据汇总导出
# ============================================================

def export_paper_tables(
    eq_info: dict,
    individuals_eq: pd.DataFrame,
    baseline_policy_df: pd.DataFrame,
    simulation_results: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    """
    汇总导出论文所需的全部结构化数据表

    参数:
        eq_info: MFG基准均衡求解信息
        individuals_eq: 基准均衡个体数据
        baseline_policy_df: 基准均衡对应的价值函数和策略快照
        simulation_results: 政策模拟结果汇总表
        sensitivity_df: 灵敏度分析结果
    """
    logger.info("=" * 80)
    logger.info("阶段四：论文数据汇总导出")
    logger.info("=" * 80)

    output_dir = ensure_dir(Path("OUTPUT/paper_tables"))
    paper_baseline = _build_paper_baseline_summary(
        eq_info,
        individuals_eq,
        baseline_policy_df,
    )
    _save_baseline_policy_snapshot(output_dir, baseline_policy_df)

    # -----------------------------------------------------------
    # 表4.2：校准参数汇总表
    # -----------------------------------------------------------
    logger.info("导出 表4.2 校准参数汇总表...")
    _export_calibration_params_table(output_dir)

    # -----------------------------------------------------------
    # 表4.4：模拟矩 vs 数据矩对比表
    # -----------------------------------------------------------
    logger.info("导出 表4.4 模拟矩vs数据矩对比表...")
    _export_moment_comparison_table(output_dir, eq_info, individuals_eq)

    # -----------------------------------------------------------
    # 表5.2：就业者/失业者多维对比表
    # -----------------------------------------------------------
    logger.info("导出 表5.2 就业/失业者对比表...")
    _export_status_comparison_table(
        output_dir,
        individuals_eq,
        baseline_policy_df=baseline_policy_df,
    )

    # -----------------------------------------------------------
    # 表5.4：匹配概率回归系数表
    # -----------------------------------------------------------
    logger.info("导出 表5.4 匹配概率回归系数表...")
    _export_matching_coefficients_table(output_dir)

    # -----------------------------------------------------------
    # 表6.2：政策效果汇总表
    # -----------------------------------------------------------
    logger.info("导出 表6.2 政策效果汇总表...")
    _export_policy_effects_table(
        output_dir,
        simulation_results,
        paper_baseline,
    )

    # -----------------------------------------------------------
    # 表6.3：分群体差异化政策效果
    # -----------------------------------------------------------
    logger.info("导出 表6.3 分群体差异化政策效果...")
    _export_subgroup_policy_effects(
        output_dir,
        simulation_results,
        individuals_eq,
    )

    # -----------------------------------------------------------
    # 表7.1：参数灵敏度分析表
    # -----------------------------------------------------------
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        logger.info("导出 表7.1 参数灵敏度分析表...")
        _export_sensitivity_table(output_dir, sensitivity_df, paper_baseline)

    logger.info("全部论文数据表已导出到 %s", output_dir)


def _build_paper_baseline_summary(
    eq_info: dict,
    individuals_eq: pd.DataFrame,
    baseline_policy_df: pd.DataFrame | None,
) -> dict:
    """
    构造论文导表统一使用的基准场景摘要。

    说明：
        表4.4/表5.2使用的是阶段一 MFG 基准均衡，而表6.* 来自阶段二政策
        模拟。这里显式把阶段一结果整理成标准化基准行，供表6.2、表6.3、
        表7.1 复用，避免不同表格落到不同 baseline 上。
    """
    stats = eq_info.get("final_statistics", {})
    aligned_policy_df = None
    unemployed_mask = individuals_eq["employment_status"].eq("unemployed")

    if baseline_policy_df is not None:
        aligned_policy_df = baseline_policy_df.reindex(individuals_eq.index)

    employed = individuals_eq[individuals_eq["employment_status"] == "employed"]
    mean_effort = 0.0
    mean_value_u = np.nan
    mean_value_e = np.nan

    if aligned_policy_df is not None:
        if unemployed_mask.any():
            mean_effort = float(
                aligned_policy_df.loc[unemployed_mask, "a_optimal"].mean()
            )
        mean_value_u = float(aligned_policy_df["V_U"].mean())
        mean_value_e = float(aligned_policy_df["V_E"].mean())

    return {
        "scenario_name": "baseline",
        "scenario_display_name": "基准场景",
        "policy_type": "",
        "converged": bool(eq_info.get("converged", False)),
        "iterations": int(eq_info.get("iterations", 0)),
        "unemployment_rate": float(
            stats.get(
                "unemployment_rate",
                eq_info.get("final_unemployment_rate", np.nan),
            )
        ),
        "mean_T": float(stats.get("mean_T", individuals_eq["T"].mean())),
        "mean_S": float(stats.get("mean_S", individuals_eq["S"].mean())),
        "mean_D": float(stats.get("mean_D", individuals_eq["D"].mean())),
        "mean_W": float(stats.get("mean_W", individuals_eq["W"].mean())),
        "mean_wage_employed": float(
            stats.get(
                "mean_wage_employed",
                employed["current_wage"].mean() if len(employed) > 0 else 0.0,
            )
        ),
        "mean_effort": mean_effort,
        "mean_value_U": mean_value_u,
        "mean_value_E": mean_value_e,
    }


def _save_baseline_policy_snapshot(
    output_dir: Path,
    baseline_policy_df: pd.DataFrame | None,
) -> None:
    """
    将当前导表实际使用的基准策略快照重新写入论文输出目录。

    这样磁盘上的 `baseline_equilibrium_policy.csv` 会和同一轮表5.2/表6.*
    使用的内存数据保持一致，便于复核。
    """
    if baseline_policy_df is None:
        logger.warning("未提供基准策略快照，跳过 baseline_equilibrium_policy.csv 刷新。")
        return

    snapshot_path = output_dir / "baseline_equilibrium_policy.csv"
    baseline_policy_df.to_csv(
        snapshot_path,
        index=True,
        encoding="utf-8-sig",
    )


def _extract_scenario_order(simulation_results: pd.DataFrame | None) -> list[str]:
    """
    提取当前这轮政策模拟结果中的场景顺序。

    该顺序同时用于表6.2与表6.3，避免扫描目录时把历史残留场景混入论文表。
    """
    if simulation_results is None or len(simulation_results) == 0:
        return []

    scenario_names = simulation_results["scenario_name"].astype(str).tolist()
    return list(dict.fromkeys(scenario_names))


def _sort_rows_by_scenario_order(
    df: pd.DataFrame,
    scenario_order: list[str],
    scenario_column: str,
) -> pd.DataFrame:
    """
    按当前运行的场景顺序排序表格。
    """
    if len(df) == 0 or not scenario_order:
        return df.reset_index(drop=True)

    order_mapping = {
        scenario_name: order
        for order, scenario_name in enumerate(scenario_order)
    }
    sorted_df = df.copy()
    sorted_df["_scenario_order"] = sorted_df[scenario_column].map(order_mapping)
    sorted_df["_scenario_order"] = sorted_df["_scenario_order"].fillna(
        len(order_mapping)
    )
    sorted_df = sorted_df.sort_values(
        by=["_scenario_order", scenario_column]
    ).drop(columns="_scenario_order")
    return sorted_df.reset_index(drop=True)


def _export_calibration_params_table(output_dir: Path) -> None:
    """导出表4.2：校准参数汇总表"""
    # 读取校准结果
    cal_path = Path("OUTPUT/calibration/calibrated_parameters.yaml")
    if not cal_path.exists():
        logger.warning("校准结果文件不存在: %s，跳过", cal_path)
        return

    with open(cal_path, "r", encoding="utf-8") as f:
        cal_data = yaml.safe_load(f)

    if not cal_data.get("success", True):
        logger.warning(
            "检测到校准结果 success=false，表4.2 将继续导出当前参数快照，但不应表述为已成功收敛的最终校准结果。"
        )

    params = cal_data["parameters"]
    partition = cal_data.get("parameter_partition", {})

    # 参数中文名映射
    param_names = {
        "rho": ("ρ", "贴现因子"),
        "kappa": ("κ", "努力成本系数"),
        "alpha_T": ("α_T", "T负效用系数"),
        "gamma_T": ("γ_T", "T状态更新系数"),
        "gamma_S": ("γ_S", "S状态更新系数"),
        "gamma_D": ("γ_D", "D状态更新系数"),
        "gamma_W": ("γ_W", "W状态更新系数"),
    }

    rows = []
    external = partition.get("external", [])
    internal = partition.get("internal", [])

    for key, value in params.items():
        symbol, desc = param_names.get(key, (key, key))
        if key in external:
            calibration_type = "external"
        elif key in internal:
            calibration_type = "internal"
        else:
            calibration_type = "unknown"
        rows.append({
            "parameter": key,
            "symbol": symbol,
            "description": desc,
            "calibrated_value": value,
            "calibration_type": calibration_type,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_4_2_calibration_params.csv",
              index=False, encoding="utf-8-sig")


def _export_moment_comparison_table(
    output_dir: Path,
    eq_info: dict,
    individuals_eq: pd.DataFrame,
) -> None:
    """导出表4.4：模拟矩与数据矩对比表"""
    target_moments = load_paper_target_moments()
    comparison_df = target_moments.get_moment_comparison(individuals_eq, eq_info)

    rows = []
    for _, row in comparison_df.iterrows():
        moment_name = row["moment_name"]
        metadata = target_moments.get_moment_metadata(moment_name)
        rows.append({
            "moment": moment_name,
            "tag": metadata.get("tag", ""),
            "target_value": row["target_value"],
            "simulated_value": row["simulated_value"],
            "absolute_diff": row["difference"],
            "pct_diff": row["relative_error"],
            "unit": metadata.get("unit", ""),
            "confidence": metadata.get("confidence", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_4_4_moment_comparison.csv",
              index=False, encoding="utf-8-sig")


def _export_status_comparison_table(
    output_dir: Path,
    individuals_eq: pd.DataFrame,
    baseline_policy_df: pd.DataFrame | None = None,
) -> None:
    """导出表5.2：就业者/失业者多维对比表"""
    if baseline_policy_df is not None:
        policy_df = baseline_policy_df.reindex(individuals_eq.index)
        individuals_eq = individuals_eq.copy()
        individuals_eq["V_U"] = policy_df["V_U"].to_numpy(dtype=float)
        individuals_eq["V_E"] = policy_df["V_E"].to_numpy(dtype=float)
        individuals_eq["a_optimal"] = policy_df["a_optimal"].to_numpy(dtype=float)
    else:
        logger.warning(
            "表5.2 未收到基准策略快照，将只使用个体状态数据导出，不附加 V_U/V_E/a_optimal。"
        )

    state_vars = ["T", "S", "D", "W", "current_wage"]
    if "V_U" in individuals_eq.columns:
        state_vars.extend(["V_U", "V_E", "a_optimal"])

    rows = []
    for status in ["employed", "unemployed"]:
        subset = individuals_eq[individuals_eq["employment_status"] == status]
        row = {"employment_status": status, "count": len(subset)}
        for var in state_vars:
            if var in subset.columns:
                row[f"{var}_mean"] = float(subset[var].mean())
                row[f"{var}_std"] = float(subset[var].std())
                row[f"{var}_median"] = float(subset[var].median())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_5_2_status_comparison.csv",
              index=False, encoding="utf-8-sig")


def _export_matching_coefficients_table(output_dir: Path) -> None:
    """导出表5.4：匹配概率Logistic回归系数表"""
    model_path = Path("OUTPUT/logistic/match_function_model.pkl")
    diagnostics_path = Path("OUTPUT/logistic/regression_diagnostics.pkl")

    if not model_path.exists():
        logger.warning("匹配函数模型文件不存在: %s，跳过", model_path)
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    # 尝试提取回归系数
    rows = []
    if hasattr(model_data, "coef_") and hasattr(model_data, "intercept_"):
        # sklearn LogisticRegression 模型
        feature_names = getattr(model_data, "feature_names_in_", None)
        coefs = model_data.coef_.flatten()
        intercept = model_data.intercept_.flatten()

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(coefs))]

        # 截距项
        rows.append({
            "feature": "intercept",
            "coefficient": float(intercept[0]),
            "odds_ratio": float(np.exp(intercept[0])),
        })

        # 各特征系数
        for name, coef in zip(feature_names, coefs):
            rows.append({
                "feature": str(name),
                "coefficient": float(coef),
                "odds_ratio": float(np.exp(coef)),
            })
    elif isinstance(model_data, dict):
        # 字典格式的模型
        if "coefficients" in model_data:
            for name, coef in model_data["coefficients"].items():
                rows.append({
                    "feature": str(name),
                    "coefficient": float(coef),
                    "odds_ratio": float(np.exp(coef)),
                })
        if "intercept" in model_data:
            rows.insert(0, {
                "feature": "intercept",
                "coefficient": float(model_data["intercept"]),
                "odds_ratio": float(np.exp(model_data["intercept"])),
            })
    elif hasattr(model_data, "params"):
        # statsmodels BinaryResultsWrapper / LogitResults
        params = model_data.params
        bse = getattr(model_data, "bse", None)
        pvalues = getattr(model_data, "pvalues", None)
        z_scores = getattr(model_data, "tvalues", None)
        conf_int = model_data.conf_int() if hasattr(model_data, "conf_int") else None

        if hasattr(params, "items"):
            param_items = list(params.items())
        else:
            param_items = list(enumerate(np.asarray(params, dtype=float)))

        for name, coef in param_items:
            feature_name = "intercept" if str(name) == "const" else str(name)
            row = {
                "feature": feature_name,
                "coefficient": float(coef),
                "odds_ratio": float(np.exp(coef)),
            }

            if bse is not None:
                std_error = bse[name] if hasattr(bse, "__getitem__") else None
                if std_error is not None:
                    row["std_error"] = float(std_error)

            if pvalues is not None:
                p_value = pvalues[name] if hasattr(pvalues, "__getitem__") else None
                if p_value is not None:
                    row["p_value"] = float(p_value)

            if z_scores is not None:
                z_value = z_scores[name] if hasattr(z_scores, "__getitem__") else None
                if z_value is not None:
                    row["z_score"] = float(z_value)

            if conf_int is not None and hasattr(conf_int, "loc"):
                ci_row = conf_int.loc[name]
                row["ci_lower"] = float(ci_row.iloc[0])
                row["ci_upper"] = float(ci_row.iloc[1])

            rows.append(row)
    else:
        logger.warning("无法识别匹配函数模型格式: %s", type(model_data))
        return

    # 尝试从诊断信息中获取标准误和p值
    if diagnostics_path.exists():
        try:
            with open(diagnostics_path, "rb") as f:
                diag = pickle.load(f)
            if isinstance(diag, dict):
                for row in rows:
                    feat = row["feature"]
                    if "std_errors" in diag and feat in diag["std_errors"]:
                        row["std_error"] = float(diag["std_errors"][feat])
                    if "p_values" in diag and feat in diag["p_values"]:
                        row["p_value"] = float(diag["p_values"][feat])
                    if "z_scores" in diag and feat in diag["z_scores"]:
                        row["z_score"] = float(diag["z_scores"][feat])
                # 提取模型拟合指标
                if "pseudo_r2" in diag:
                    logger.info("  Pseudo R² = %.4f", diag["pseudo_r2"])
                if "accuracy" in diag:
                    logger.info("  Accuracy = %.4f", diag["accuracy"])
        except Exception as e:
            logger.warning("读取回归诊断信息失败: %s", e)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_5_4_matching_coefficients.csv",
              index=False, encoding="utf-8-sig")


def _export_policy_effects_table(
    output_dir: Path,
    simulation_results: pd.DataFrame,
    paper_baseline: dict | None = None,
) -> None:
    """导出表6.2：政策效果汇总表"""
    if simulation_results is None or len(simulation_results) == 0:
        logger.warning("政策模拟结果为空，跳过")
        return

    results_df = simulation_results.copy()
    scenario_order = _extract_scenario_order(results_df)

    if paper_baseline is not None:
        results_df = results_df[results_df["scenario_name"] != "baseline"].copy()
        results_df = pd.concat(
            [pd.DataFrame([paper_baseline]), results_df],
            ignore_index=True,
        )

    if "converged" not in results_df.columns:
        results_df["converged"] = True

    diagnostic_path = output_dir / "table_6_2_policy_effects_all_runs.csv"
    diagnostic_df = _sort_rows_by_scenario_order(
        results_df,
        scenario_order,
        "scenario_name",
    )
    diagnostic_df.to_csv(diagnostic_path, index=False, encoding="utf-8-sig")

    baseline_candidates = results_df[
        results_df["scenario_name"] == "baseline"
    ]

    baseline = None
    baseline_available = len(baseline_candidates) > 0
    if baseline_available:
        baseline = baseline_candidates.iloc[0]
    else:
        logger.warning(
            "找不到基准场景，主表将保留诊断信息但不计算政策增量。"
        )

    source_df = results_df

    rows = []
    for _, row in source_df.iterrows():
        entry = {
            "scenario_name": row["scenario_name"],
            "scenario_display_name": row["scenario_display_name"],
            "policy_type": row.get("policy_type", ""),
            "converged": row.get("converged", True),
            "paper_eligible": baseline_available,
            "unemployment_rate": row["unemployment_rate"],
            "mean_wage_employed": row["mean_wage_employed"],
            "mean_S": row["mean_S"],
            "mean_D": row["mean_D"],
            "mean_T": row["mean_T"],
            "mean_effort": row["mean_effort"],
            "mean_value_U": row.get("mean_value_U", np.nan),
            "mean_value_E": row.get("mean_value_E", np.nan),
        }

        if baseline_available and row["scenario_name"] != "baseline":
            entry["delta_u_pp"] = (
                row["unemployment_rate"] - baseline["unemployment_rate"]
            ) * 100
            entry["delta_u_pct"] = (
                (row["unemployment_rate"] - baseline["unemployment_rate"])
                / baseline["unemployment_rate"] * 100
            )
            entry["delta_wage"] = (
                row["mean_wage_employed"] - baseline["mean_wage_employed"]
            )
            entry["delta_S_pct"] = (
                (row["mean_S"] - baseline["mean_S"]) / baseline["mean_S"] * 100
            )
            entry["delta_D_pct"] = (
                (row["mean_D"] - baseline["mean_D"]) / baseline["mean_D"] * 100
            )
            entry["delta_welfare_U"] = (
                row.get("mean_value_U", 0) - baseline.get("mean_value_U", 0)
            )
            entry["delta_welfare_E"] = (
                row.get("mean_value_E", 0) - baseline.get("mean_value_E", 0)
            )
        else:
            for k in ["delta_u_pp", "delta_u_pct", "delta_wage",
                       "delta_S_pct", "delta_D_pct",
                       "delta_welfare_U", "delta_welfare_E"]:
                entry[k] = 0.0 if baseline_available else np.nan

        rows.append(entry)

    df = pd.DataFrame(rows)
    df = _sort_rows_by_scenario_order(df, scenario_order, "scenario_name")
    df.to_csv(output_dir / "table_6_2_policy_effects.csv",
              index=False, encoding="utf-8-sig")


def _export_subgroup_policy_effects(
    output_dir: Path,
    simulation_results: pd.DataFrame,
    baseline_individuals_eq: pd.DataFrame,
) -> None:
    """导出表6.3：分群体（高/低技能）差异化政策效果"""
    simulation_dir = Path("OUTPUT/simulation")
    scenario_order = _extract_scenario_order(simulation_results)

    if not scenario_order:
        logger.warning("当前运行缺少场景结果，跳过分群体分析")
        return

    rows = []
    for scenario_name in scenario_order:
        if scenario_name == "baseline":
            df = baseline_individuals_eq.copy()
        else:
            scenario_dir = simulation_dir / f"scenario_{scenario_name}"
            individuals_path = scenario_dir / "equilibrium_individuals.csv"
            if not individuals_path.exists():
                logger.warning(
                    "场景 %s 缺少个体结果文件，跳过分群体导出。",
                    scenario_name,
                )
                continue
            df = pd.read_csv(individuals_path)

        # 按技能水平分为高/低技能群体（中位数分割）
        median_s = df["S"].median()
        high_skill = df[df["S"] >= median_s]
        low_skill = df[df["S"] < median_s]

        for group_name, group_df in [("high_skill", high_skill),
                                      ("low_skill", low_skill)]:
            n_total = len(group_df)
            n_unemployed = len(
                group_df[group_df["employment_status"] == "unemployed"]
            )
            employed = group_df[group_df["employment_status"] == "employed"]

            rows.append({
                "scenario": scenario_name,
                "subgroup": group_name,
                "n_individuals": n_total,
                "unemployment_rate": n_unemployed / n_total if n_total > 0 else np.nan,
                "mean_wage": float(employed["current_wage"].mean()) if len(employed) > 0 else 0.0,
                "mean_S": float(group_df["S"].mean()),
                "mean_D": float(group_df["D"].mean()),
                "mean_T": float(group_df["T"].mean()),
            })

    if rows:
        subgroup_df = pd.DataFrame(rows)
        subgroup_df = _sort_rows_by_scenario_order(
            subgroup_df,
            scenario_order,
            "scenario",
        )
        subgroup_df.to_csv(
            output_dir / "table_6_3_subgroup_effects.csv",
            index=False, encoding="utf-8-sig",
        )


def _export_sensitivity_table(
    output_dir: Path,
    sensitivity_df: pd.DataFrame,
    paper_baseline: dict | None = None,
) -> None:
    """导出表7.1：参数灵敏度分析表"""
    # 直接复用阶段一基准均衡的摘要，避免 OUTPUT/mfg 在后续步骤中被覆盖。
    if paper_baseline is not None:
        baseline_vals = {
            "unemployment_rate": paper_baseline.get("unemployment_rate", np.nan),
            "mean_wage": paper_baseline.get("mean_wage_employed", np.nan),
            "mean_S": paper_baseline.get("mean_S", np.nan),
            "mean_D": paper_baseline.get("mean_D", np.nan),
        }
    else:
        baseline_vals = {
            "unemployment_rate": np.nan,
            "mean_wage": np.nan,
            "mean_S": np.nan,
            "mean_D": np.nan,
        }

    # 添加相对于基准的变化率
    result_df = sensitivity_df.copy()
    for metric in ["unemployment_rate", "mean_wage", "mean_S", "mean_D"]:
        base = baseline_vals.get(metric, np.nan)
        if not np.isnan(base) and base != 0:
            result_df[f"{metric}_pct_change"] = (
                (result_df[metric] - base) / abs(base) * 100
            )
        else:
            result_df[f"{metric}_pct_change"] = np.nan

    result_df.to_csv(
        output_dir / "table_7_1_sensitivity_analysis.csv",
        index=False, encoding="utf-8-sig",
    )


# ============================================================
# 主入口
# ============================================================

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="论文数据生成脚本（服务器端运行）"
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="跳过灵敏度分析（节省约8小时运行时间）",
    )
    args = parser.parse_args()

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 80)
    logger.info("论文数据生成脚本启动")
    logger.info("时间: %s", timestamp)
    logger.info("参数: skip_sensitivity=%s", args.skip_sensitivity)
    logger.info("=" * 80)

    # 确保输出目录存在
    ensure_dir(Path("OUTPUT/paper_tables"))

    paper_config_path = create_paper_mfg_config()

    try:
        base_population, initial_T = create_shared_population_sample(
            paper_config_path
        )

        # 阶段一：MFG基准均衡求解
        individuals_eq, eq_info, baseline_policy_df = run_baseline_equilibrium(
            paper_config_path,
            base_population,
            initial_T,
        )
        baseline_result = _build_paper_baseline_summary(
            eq_info,
            individuals_eq,
            baseline_policy_df,
        )

        # 阶段二：政策场景批量模拟
        simulation_results = run_policy_simulations(
            paper_config_path,
            base_population,
            initial_T,
            baseline_result,
            individuals_eq,
            eq_info,
        )

        # 阶段三：参数灵敏度分析（可选跳过）
        sensitivity_df = None
        if not args.skip_sensitivity:
            sensitivity_df = run_sensitivity_analysis()
        else:
            logger.info("跳过灵敏度分析（--skip-sensitivity 已启用）")
            existing_path = Path(
                "OUTPUT/paper_tables/table_7_1_sensitivity_analysis.csv"
            )
            if existing_path.exists():
                sensitivity_df = pd.read_csv(existing_path)
                logger.info("已加载已有灵敏度分析结果: %s", existing_path)

        # 阶段四：论文数据汇总导出
        export_paper_tables(
            eq_info,
            individuals_eq,
            baseline_policy_df,
            simulation_results,
            sensitivity_df,
        )
    finally:
        if paper_config_path.exists():
            paper_config_path.unlink()
            logger.info("已清理论文运行临时配置: %s", paper_config_path)

    # 完成
    total_time = (time.time() - start_time) / 3600
    logger.info("=" * 80)
    logger.info("全部任务完成！总耗时 %.2f 小时", total_time)
    logger.info("输出目录:")
    logger.info("  数据表: OUTPUT/paper_tables/")
    logger.info("  MFG均衡: OUTPUT/mfg/")
    logger.info("  政策模拟: OUTPUT/simulation/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
