#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jacobian弹性排序的Elbow可视化脚本。

功能：
1. 读取 jacobian_analysis.csv；
2. 按指定指标降序绘制折线图；
3. 使用“端点连线最大垂距法”自动标注拐点；
4. 输出PNG图，辅助判断外部/内部参数分组阈值。
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

# 使用非交互后端，适配服务器与无桌面环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="绘制Jacobian弹性Elbow折线图（自动标注拐点）"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="OUTPUT/calibration/jacobian_analysis.csv",
        help="Jacobian摘要CSV路径（默认：OUTPUT/calibration/jacobian_analysis.csv）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="OUTPUT/calibration",
        help="图片输出目录（默认：OUTPUT/calibration）",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="max_abs_elasticity,mean_abs_elasticity",
        help="要绘制的列名，逗号分隔（默认：max_abs_elasticity,mean_abs_elasticity）",
    )
    return parser.parse_args()


def find_elbow_index(values: np.ndarray) -> int:
    """
    使用“端点连线最大垂距法”寻找拐点索引（0-based）。

    参数:
        values: 已按降序排列的一维数组

    返回:
        拐点索引（0-based）
    """
    n = len(values)
    if n <= 2:
        return 0

    x = np.arange(n, dtype=float)
    y = values.astype(float)

    # 归一化，避免横纵尺度差异影响距离
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1.0)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1.0e-12)

    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]

    denom = math.hypot(y2 - y1, x2 - x1)
    if denom <= 1.0e-12:
        return 0

    distances = []
    for idx in range(n):
        x0, y0 = x_norm[idx], y_norm[idx]
        # 点到直线距离公式
        distance = abs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
        ) / denom
        distances.append(distance)

    return int(np.argmax(distances))


def build_sorted_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    生成按 metric 降序排序的数据框。

    参数:
        df: 原始DataFrame
        metric: 指标列名

    返回:
        排序后的DataFrame（重置索引）
    """
    required_cols = {"param_name", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"CSV缺少必要列: {sorted(missing)}")

    out = df[["param_name", metric]].copy()
    out = out.sort_values(metric, ascending=False).reset_index(drop=True)
    return out


def plot_elbow(sorted_df: pd.DataFrame, metric: str, out_dir: Path) -> Tuple[Path, int]:
    """
    绘制单个指标的Elbow图并保存。

    参数:
        sorted_df: 已按指标降序排序的数据
        metric: 指标名
        out_dir: 输出目录

    返回:
        (输出文件路径, elbow索引)
    """
    values = sorted_df[metric].to_numpy(dtype=float)
    elbow_idx = find_elbow_index(values)

    x = np.arange(1, len(values) + 1)
    elbow_x = elbow_idx + 1
    elbow_param = sorted_df.loc[elbow_idx, "param_name"]
    elbow_value = float(sorted_df.loc[elbow_idx, metric])

    plt.figure(figsize=(10, 6))
    plt.plot(x, values, marker="o", linewidth=2, label=metric)
    plt.scatter(
        [elbow_x], [elbow_value], color="red", s=90, zorder=5, label="elbow point"
    )
    plt.axvline(elbow_x, color="red", linestyle="--", alpha=0.7)
    plt.title(f"Jacobian Elbow Plot ({metric})")
    plt.xlabel("Rank (descending sensitivity)")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()

    text = f"elbow: rank={elbow_x}\nparam={elbow_param}\nvalue={elbow_value:.4f}"
    plt.annotate(
        text,
        xy=(elbow_x, elbow_value),
        xytext=(elbow_x + 0.5, elbow_value),
        textcoords="data",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    out_file = out_dir / f"jacobian_elbow_{metric}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    return out_file, elbow_idx


def run() -> None:
    """主流程。"""
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到CSV文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    metrics: List[str] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("metrics不能为空")

    print(f"读取文件: {csv_path}")
    print(f"参数数量: {len(df)}")

    for metric in metrics:
        sorted_df = build_sorted_frame(df, metric)
        out_file, elbow_idx = plot_elbow(sorted_df, metric, out_dir)
        elbow_row = sorted_df.iloc[elbow_idx]

        print("\n" + "=" * 72)
        print(f"指标: {metric}")
        print("降序排序:")
        print(sorted_df.to_string(index=False))
        print(
            f"Elbow点: rank={elbow_idx + 1}, "
            f"param={elbow_row['param_name']}, "
            f"value={float(elbow_row[metric]):.6f}"
        )
        print(f"图片输出: {out_file}")


if __name__ == "__main__":
    run()
