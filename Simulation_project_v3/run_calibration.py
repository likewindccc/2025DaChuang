#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准模块一键运行脚本。

设计目标：
1. 提供统一的命令行入口，避免每次手写 `python -c ...`；
2. 与当前 SMMCalibrator 实现保持一致（method/initial_values/auto_resume）；
3. 运行结束后打印关键结果与落盘文件检查，便于快速验收。
"""

from __future__ import annotations

import os

# ── 双层并行保护 ────────────────────────────────────────────────────────────────
# DE 优化器（loky）会启动若干子进程，内层 Numba @njit(parallel=True) 默认会
# 再次占满所有核，导致"进程数 × Numba线程数"远超核心数，CPU全满。
# 此处在所有 import 之前设置 NUMBA_NUM_THREADS，限制每个进程的 Numba 线程数。
#
# 计算规则：numba_threads = max_cores // de_workers
#   CALIBRATION_WORKERS  : DE 并行进程数，需与 calibration_config.yaml 一致（默认16）
#   CALIBRATION_MAX_CORES: 允许程序整体占用的最大核数（默认 = workers，即每worker 1线程）
#
# 示例：workers=16, max_cores=16 → numba_threads=1 → 总占用 16 核
#       workers=16, max_cores=32 → numba_threads=2 → 总占用 32 核（全核）
_de_workers = int(os.environ.get("CALIBRATION_WORKERS", "16"))
_max_cores = int(os.environ.get("CALIBRATION_MAX_CORES", str(_de_workers)))
_numba_threads = max(1, _max_cores // _de_workers)
os.environ.setdefault("NUMBA_NUM_THREADS", str(_numba_threads))
# ───────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# 保证从项目根目录运行时可以导入 MODULES 包
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MODULES.CALIBRATION import SMMCalibrator


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="运行参数校准（SMM）的一键入口脚本。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="CONFIG/calibration_config.yaml",
        help="校准配置文件路径（默认：CONFIG/calibration_config.yaml）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help=(
            "优化方法覆盖值（例如 differential_evolution / Powell）。"
            "不传则使用配置文件中的 optimization.method。"
        ),
    )
    parser.add_argument(
        "--initial-strategy",
        type=str,
        choices=["baseline", "random", "midpoint", "sensitivity"],
        default="baseline",
        help="初始值策略（默认：baseline）。",
    )
    parser.add_argument(
        "--initial-values",
        type=str,
        default=None,
        help=(
            "手工指定初始参数向量，逗号分隔。"
            "例如：0.4,1500,0.25,0.3,0.35,0.45,0.15"
        ),
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="禁用自动断点恢复（对应 allow_auto_resume=False）。",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认：INFO）。",
    )
    return parser.parse_args()


def parse_initial_values(raw_values: str) -> np.ndarray:
    """
    解析逗号分隔的初始参数字符串。

    参数:
        raw_values: 例如 "0.4,1500,0.25,0.3,0.35,0.45,0.15"

    返回:
        numpy 参数向量
    """
    parts = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not parts:
        raise ValueError("`--initial-values` 不能为空。")
    return np.array([float(item) for item in parts], dtype=float)


def resolve_path(path_text: str) -> Path:
    """
    将相对路径解析为项目根目录下的绝对路径。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_initial_values(
    calibrator: SMMCalibrator,
    args: argparse.Namespace
) -> Optional[np.ndarray]:
    """
    根据命令行参数构建初始值向量。

    规则：
    1. 若指定 `--initial-values`，优先使用手工值；
    2. 否则按 `--initial-strategy` 获取；
    3. baseline 策略直接返回 None，让 calibrate 使用默认逻辑。
    """
    if args.initial_values is not None:
        initial_values = parse_initial_values(args.initial_values)
        n_params = calibrator.param_utils.get_n_params()
        if len(initial_values) != n_params:
            raise ValueError(
                f"`--initial-values` 参数数量不匹配：期望 {n_params}，实际 {len(initial_values)}"
            )
        return initial_values

    if args.initial_strategy == "baseline":
        return None

    return calibrator.param_utils.get_initial_values(args.initial_strategy)


def check_output_files(output_dir: Path) -> None:
    """
    校准结束后检查关键落盘文件是否存在并打印状态。
    """
    required_files = [
        "calibration_stage_summary.yaml",
        "calibrated_parameters.yaml",
        "optimization_result.pkl",
    ]

    logging.info("关键输出文件检查（目录：%s）", output_dir)
    for name in required_files:
        file_path = output_dir / name
        status = "FOUND" if file_path.exists() else "MISSING"
        logging.info("  [%s] %s", status, file_path)


def main() -> int:
    """
    主流程：
    1. 初始化校准器
    2. 解析运行参数
    3. 调用 calibrate 执行校准
    4. 输出关键结果摘要
    """
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = resolve_path(args.config)
    if not config_path.exists():
        logging.error("配置文件不存在：%s", config_path)
        return 1

    logging.info("=" * 72)
    logging.info("启动参数校准")
    logging.info("项目根目录: %s", PROJECT_ROOT)
    logging.info("配置文件: %s", config_path)
    logging.info("优化方法覆盖: %s", args.method if args.method else "使用配置值")
    logging.info("初始值策略: %s", args.initial_strategy)
    logging.info("自动续跑: %s", "禁用" if args.no_auto_resume else "启用")
    logging.info("=" * 72)

    try:
        calibrator = SMMCalibrator(str(config_path))
        initial_values = build_initial_values(calibrator, args)

        result = calibrator.calibrate(
            method=args.method,
            initial_values=initial_values,
            allow_auto_resume=(not args.no_auto_resume),
        )

        logging.info("=" * 72)
        logging.info("校准完成：success=%s, fun=%.6f, nfev=%s",
                     bool(result.success), float(result.fun), int(result.nfev))
        logging.info("最优参数向量: %s", np.asarray(result.x, dtype=float))
        check_output_files(calibrator.output_dir)
        logging.info("=" * 72)

        return 0 if bool(result.success) else 2
    except Exception:
        logging.exception("校准运行失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
