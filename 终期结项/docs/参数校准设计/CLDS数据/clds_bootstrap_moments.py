"""
clds_bootstrap_moments.py
=========================
CLDS 目标矩 Bootstrap 标准误估计脚本。

功能：
    1. 读取 CLDS 2018/2016 数据并筛选“农村女性，15-64岁”样本；
    2. 对 M1-M8 目标矩进行 B 次 Bootstrap（默认 500 次）；
    3. 将 bootstrap_se 写回 YAML，供 SMM 逆方差加权使用。

输出文件：
    - target_moments_clds_bootstrap500.yaml（默认）

说明：
    - 脚本默认只为 M1-M8 写入 bootstrap_se；
    - 若传入 --include-m9，则同时估计 M9 的 bootstrap_se。
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from clds_compute_moments import (
    compute_digital_ratio,
    compute_hours_moments,
    compute_m1_unemployment,
    compute_transition_rates,
    compute_wage_moments,
    filter_rural_female,
    load_2016,
    load_2018,
)


logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_YAML = SCRIPT_DIR / "target_moments_clds.yaml"
DEFAULT_OUTPUT_YAML = SCRIPT_DIR / "target_moments_clds_bootstrap500.yaml"

# 与 clds_compute_moments.py 保持一致
DTA_2018 = SCRIPT_DIR / "2018" / "CLDS2018Stata（转码后）" / "2018个体问卷 （191111）.dta"
DTA_2016 = SCRIPT_DIR / "CLDS2016 适用STATA14及以上" / "individual2016.dta"

CORE_MOMENTS = [
    "unemployment_rate",     # M1
    "mean_wage",             # M2
    "std_wage",              # M3
    "mean_weekly_hours",     # M4
    "job_finding_rate",      # M5
    "separation_rate",       # M6
    "wage_iqr_ratio",        # M7
    "std_weekly_hours",      # M8
]


def setup_logging(verbose: bool) -> None:
    """初始化日志。"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_compute_cross_section_moments(rf18_boot: pd.DataFrame) -> Dict[str, float]:
    """
    计算截面矩（M1/M2/M3/M4/M7/M8）。

    若某次自助样本异常导致某个矩不可计算，返回 np.nan，
    后续标准误计算使用 nanstd 忽略该次抽样。
    """
    output = {}

    try:
        m1 = compute_m1_unemployment(rf18_boot)
        output["unemployment_rate"] = float(m1["value"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("M1 本次抽样失败: %s", exc)
        output["unemployment_rate"] = np.nan

    try:
        wage = compute_wage_moments(rf18_boot)
        output["mean_wage"] = float(wage["mean_wage"]["value"])
        output["std_wage"] = float(wage["std_wage"]["value"])
        output["wage_iqr_ratio"] = float(wage["wage_iqr_ratio"]["value"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("M2/M3/M7 本次抽样失败: %s", exc)
        output["mean_wage"] = np.nan
        output["std_wage"] = np.nan
        output["wage_iqr_ratio"] = np.nan

    try:
        hours = compute_hours_moments(rf18_boot)
        output["mean_weekly_hours"] = float(hours["mean_weekly_hours"]["value"])
        output["std_weekly_hours"] = float(hours["std_weekly_hours"]["value"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("M4/M8 本次抽样失败: %s", exc)
        output["mean_weekly_hours"] = np.nan
        output["std_weekly_hours"] = np.nan

    return output


def _safe_compute_transition_moments(
    rf16_boot: pd.DataFrame,
    rf18_boot: pd.DataFrame,
) -> Dict[str, float]:
    """
    计算面板转移率矩（M5/M6）。

    若本次抽样因样本基数不足等原因失败，返回 np.nan。
    """
    output = {}
    try:
        trans = compute_transition_rates(rf16_boot, rf18_boot)
        output["job_finding_rate"] = float(trans["job_finding_rate"]["value"])
        output["separation_rate"] = float(trans["separation_rate"]["value"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("M5/M6 本次抽样失败: %s", exc)
        output["job_finding_rate"] = np.nan
        output["separation_rate"] = np.nan
    return output


def _safe_compute_m9(rf18_boot: pd.DataFrame) -> float:
    """计算 M9（可选）。"""
    try:
        m9 = compute_digital_ratio(rf18_boot)
        return float(m9["digital_job_ratio"]["value"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("M9 本次抽样失败: %s", exc)
        return np.nan


def _validate_required_files() -> None:
    """校验输入文件存在性。"""
    missing_files = [path for path in [DTA_2018, DTA_2016] if not path.exists()]
    if missing_files:
        missing_str = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"缺少输入数据文件: {missing_str}")


def run_bootstrap(
    n_bootstrap: int,
    seed: int,
    include_m9: bool,
) -> Dict[str, List[float]]:
    """
    执行 Bootstrap 主循环。

    返回：
        每个矩对应一列抽样值列表。
    """
    if n_bootstrap < 2:
        raise ValueError("Bootstrap次数必须 >= 2")

    _validate_required_files()
    rng = np.random.default_rng(seed)

    logger.info("加载原始数据并筛样本...")
    data_2018 = load_2018(str(DTA_2018))
    data_2016 = load_2016(str(DTA_2016))
    rf18 = filter_rural_female(data_2018, gender_col="Igender")
    rf16 = filter_rural_female(data_2016, gender_col="gender")

    n18 = len(rf18)
    n16 = len(rf16)
    logger.info("样本规模：rf18=%d, rf16=%d", n18, n16)

    all_moments = CORE_MOMENTS.copy()
    if include_m9:
        all_moments.append("digital_job_ratio")

    draws = {moment_name: [] for moment_name in all_moments}

    logger.info("开始Bootstrap: B=%d, seed=%d", n_bootstrap, seed)
    for idx in range(n_bootstrap):
        # 行抽样（有放回）
        idx18 = rng.integers(0, n18, n18)
        idx16 = rng.integers(0, n16, n16)

        rf18_boot = rf18.iloc[idx18].reset_index(drop=True)
        rf16_boot = rf16.iloc[idx16].reset_index(drop=True)

        cross = _safe_compute_cross_section_moments(rf18_boot)
        trans = _safe_compute_transition_moments(rf16_boot, rf18_boot)

        for name in CORE_MOMENTS:
            if name in cross:
                draws[name].append(cross[name])
            elif name in trans:
                draws[name].append(trans[name])

        if include_m9:
            draws["digital_job_ratio"].append(_safe_compute_m9(rf18_boot))

        if (idx + 1) % max(1, n_bootstrap // 10) == 0:
            logger.info("Bootstrap进度: %d/%d", idx + 1, n_bootstrap)

    return draws


def update_yaml_with_bootstrap_se(
    base_yaml_path: Path,
    output_yaml_path: Path,
    draws: Dict[str, List[float]],
    n_bootstrap: int,
    seed: int,
) -> None:
    """把 bootstrap_se 写入输出 YAML。"""
    with open(base_yaml_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    if "moments" not in yaml_data:
        raise KeyError("基础YAML缺少 moments 节点")

    for moment_name, values in draws.items():
        if moment_name not in yaml_data["moments"]:
            logger.warning("YAML中不存在矩 %s，跳过写入", moment_name)
            continue

        arr = np.asarray(values, dtype=float)
        valid_count = int(np.sum(~np.isnan(arr)))
        if valid_count < 2:
            raise ValueError(
                f"{moment_name} 有效抽样次数不足（valid={valid_count}），"
                "无法计算bootstrap_se，请检查样本与口径定义。"
            )

        bootstrap_se = float(np.nanstd(arr, ddof=1))
        yaml_data["moments"][moment_name]["bootstrap_se"] = round(bootstrap_se, 6)
        yaml_data["moments"][moment_name]["bootstrap_valid_draws"] = valid_count

    yaml_data.setdefault("meta", {})
    yaml_data["meta"]["bootstrap"] = {
        "n_replications": int(n_bootstrap),
        "seed": int(seed),
    }

    with open(output_yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(
            yaml_data,
            file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info("已写出Bootstrap结果: %s", output_yaml_path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对 CLDS 目标矩执行 Bootstrap 并写入 bootstrap_se"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap重复次数（默认500）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260302,
        help="随机种子（默认20260302）",
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=DEFAULT_BASE_YAML,
        help=f"基础目标矩YAML路径（默认: {DEFAULT_BASE_YAML}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_YAML,
        help=f"输出YAML路径（默认: {DEFAULT_OUTPUT_YAML}）",
    )
    parser.add_argument(
        "--include-m9",
        action="store_true",
        help="是否同时估计 M9(digital_job_ratio) 的 bootstrap_se",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，仅输出警告和错误",
    )
    return parser.parse_args()


def main() -> None:
    """主入口。"""
    args = parse_args()
    setup_logging(verbose=not args.quiet)

    draws = run_bootstrap(
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        include_m9=args.include_m9,
    )
    update_yaml_with_bootstrap_se(
        base_yaml_path=args.base_yaml,
        output_yaml_path=args.output,
        draws=draws,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
