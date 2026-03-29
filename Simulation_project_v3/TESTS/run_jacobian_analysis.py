#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jacobian敏感性分析独立入口脚本

功能：
    在正式SMM校准前，独立运行Step 0（Jacobian敏感性预分析）和Step 1（参数分类），
    输出各参数对各目标矩的弹性矩阵，辅助判断哪些参数可识别、哪些需外部校准。

用法：
    # 基本运行（使用默认配置）
    python TESTS/run_jacobian_analysis.py

    # 指定配置文件
    python TESTS/run_jacobian_analysis.py --config CONFIG/calibration_config.yaml

    # 快速测试模式（减少MFG迭代轮数）
    python TESTS/run_jacobian_analysis.py --quick

    # 覆盖扰动步长（默认3%）
    python TESTS/run_jacobian_analysis.py --relative-step 0.05

    # 仅做Jacobian，跳过参数分类（Step 1）
    python TESTS/run_jacobian_analysis.py --no-classify

输出文件（均在 OUTPUT/calibration/ 下）：
    jacobian_analysis.csv       参数弹性摘要表（含max_abs_elasticity排序）
    jacobian_matrix.npy         原始数值Jacobian矩阵（n_moments × n_params）
    jacobian_elasticity.npy     弹性矩阵
    jacobian_heatmap.png        弹性热力图（需要matplotlib）
    parameter_partition.yaml    Step 1参数分类结果（内部/外部）

注意：
    本脚本会调用完整MFG均衡求解器，对每个参数做 ±relative_step 扰动，
    共需 2 × n_params 次完整MFG求解。默认参数数为7，共14次求解，
    建议在 AutoDL 服务器上运行以节省时间。

    若需快速验证脚本可用性，使用 --quick 选项可将MFG迭代轮数降至5轮。
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 将项目根目录加入sys.path，确保MODULES可以被正确导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODULES.CALIBRATION.smm_calibrator import SMMCalibrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Jacobian敏感性分析独立入口（SMM校准 Step 0 + Step 1）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="CONFIG/calibration_config.yaml",
        help="校准配置文件路径（默认：CONFIG/calibration_config.yaml）",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试模式：临时将MFG最大迭代轮数降至5轮",
    )
    parser.add_argument(
        "--relative-step",
        type=float,
        default=None,
        help="参数扰动步长（相对于参数范围的比例，默认读取配置文件中的值）",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="跳过Step 1参数分类，仅输出Jacobian矩阵",
    )
    return parser.parse_args()


def _patch_quick_mode(config_path: str) -> Path:
    """
    快速测试模式：读取mfg_config，将max_outer_iter临时设为5，写入临时文件。

    参数：
        config_path: 原始MFG配置文件路径

    返回：
        临时配置文件路径（使用完毕后需手动删除）
    """
    with open(config_path, "r", encoding="utf-8") as f:
        mfg_cfg = yaml.safe_load(f)

    # 将MFG最大迭代轮数设为5以快速跑完
    original_iter = mfg_cfg.get("equilibrium", {}).get("max_outer_iter", 150)
    mfg_cfg["equilibrium"]["max_outer_iter"] = 5
    logger.info("快速模式：MFG最大迭代轮数从 %d 临时降至 5", original_iter)

    quick_path = Path(config_path).parent / "_mfg_config_quick_jacobian.yaml"
    with open(quick_path, "w", encoding="utf-8") as f:
        yaml.dump(mfg_cfg, f, allow_unicode=True)

    return quick_path


def _try_plot_heatmap(
    jacobian_summary: pd.DataFrame,
    moment_names: list,
    output_dir: Path,
) -> None:
    """
    尝试绘制弹性热力图，若matplotlib不可用则跳过。

    参数：
        jacobian_summary: Jacobian摘要DataFrame（来自_run_jacobian_analysis）
        moment_names:     目标矩名称列表
        output_dir:       输出目录
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互后端，适合服务器
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("未安装matplotlib/seaborn，跳过热力图绘制。")
        return

    # 提取弹性矩阵（elastic_*列）
    elastic_cols = [c for c in jacobian_summary.columns if c.startswith("elastic_")]
    if not elastic_cols:
        logger.warning("Jacobian摘要中未找到 elastic_* 列，跳过热力图。")
        return

    param_names = jacobian_summary["param_name"].tolist()
    # 行=参数，列=矩
    elasticity_matrix = jacobian_summary[elastic_cols].values  # shape: (n_params, n_moments)

    # 转置为 (n_moments, n_params) 更直观
    elasticity_T = elasticity_matrix.T
    col_labels = [c.replace("elastic_", "") for c in elastic_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(param_names) * 1.2), max(5, len(moment_names))))
    sns.heatmap(
        np.abs(elasticity_T),
        xticklabels=param_names,
        yticklabels=col_labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Jacobian弹性矩阵（绝对值）\n行=目标矩，列=参数，颜色越深=敏感性越强")
    ax.set_xlabel("参数")
    ax.set_ylabel("目标矩")
    plt.tight_layout()

    heatmap_path = output_dir / "jacobian_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("热力图已保存至: %s", heatmap_path)


def main() -> None:
    """主流程：运行Step 0 Jacobian分析 + Step 1参数分类。"""
    args = parse_args()

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        logger.error("配置文件不存在: %s", base_config_path)
        sys.exit(1)

    config_path = base_config_path
    logger.info("=" * 70)
    logger.info("Jacobian敏感性分析独立入口")
    logger.info("配置文件: %s", config_path)
    logger.info("=" * 70)

    # 所有“运行时覆盖”都写入临时配置，避免修改原始yaml
    temp_files = []
    if args.quick or args.relative_step is not None:
        with open(base_config_path, "r", encoding="utf-8") as f:
            runtime_cfg = yaml.safe_load(f)

        if args.quick:
            logger.info("启用快速模式")
            mfg_config_path = runtime_cfg["mfg_solver"]["config_path"]
            quick_mfg_path = _patch_quick_mode(mfg_config_path)
            runtime_cfg["mfg_solver"]["config_path"] = str(quick_mfg_path)
            temp_files.append(quick_mfg_path)

        if args.relative_step is not None:
            runtime_cfg.setdefault("calibration_strategy", {}).setdefault(
                "step0_jacobian", {}
            )["relative_step"] = float(args.relative_step)
            logger.info("参数扰动步长仅本次生效: %.4f", args.relative_step)

        runtime_config_path = (
            base_config_path.parent / "_calibration_config_runtime_jacobian.yaml"
        )
        with open(runtime_config_path, "w", encoding="utf-8") as f:
            yaml.dump(runtime_cfg, f, allow_unicode=True)
        config_path = runtime_config_path
        temp_files.append(runtime_config_path)
        logger.info("运行时临时配置文件: %s", config_path)

    try:
        # 初始化校准器（不会自动运行校准）
        calibrator = SMMCalibrator(str(config_path))

        # 获取初始参数向量
        initial_values = calibrator.param_utils.get_initial_values("baseline")
        full_initial = calibrator.param_utils.clip_to_bounds(
            np.asarray(initial_values, dtype=float)
        )
        logger.info("基准参数: %s", dict(zip(calibrator.param_names, full_initial)))

        # 创建MFG求解器
        logger.info("\nStep 0: 开始Jacobian敏感性分析...")
        logger.info(
            "共需 %d 次完整MFG求解（每参数两次 ±扰动）",
            2 * len(calibrator.param_names),
        )
        full_solver = calibrator._create_mfg_solver()

        # 运行Jacobian分析（核心步骤，计算量大）
        jacobian_summary = calibrator._run_jacobian_analysis(full_solver, full_initial)

        # 打印摘要表
        logger.info("\nJacobian弹性摘要（按最大弹性升序排列，弹性越小越难识别）：")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(jacobian_summary[["param_name", "max_abs_elasticity", "mean_abs_elasticity"]].to_string(index=False))

        # 尝试绘制热力图
        _try_plot_heatmap(
            jacobian_summary,
            calibrator.target_moments.get_moment_names(),
            calibrator.output_dir,
        )

        # Step 1参数分类（可选）
        if not args.no_classify:
            logger.info("\nStep 1: 参数分类（基于弹性阈值）...")
            partition = calibrator._classify_parameters(jacobian_summary)
            logger.info("内部参数（纳入SMM）: %s", partition["internal"])
            logger.info("外部参数（外部校准）: %s", partition["external"])
        else:
            logger.info("\n已跳过Step 1参数分类（--no-classify）。")

        logger.info("\n分析完成！")
        logger.info("结果目录: %s", calibrator.output_dir.resolve())

    finally:
        # 清理临时配置文件
        for tmp_path in temp_files:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()
                logger.debug("已删除临时配置: %s", tmp_path)


if __name__ == "__main__":
    main()
