#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jacobian 平均弹性（mean_abs_elasticity）Elbow 可视化脚本。

设计约束：
1. 仅使用 mean_abs_elasticity 指标；
2. 按升序排列（小弹性在左）；
3. 使用 SciencePlots 风格绘图；
4. 放置在正式 CALIBRATION 模块中，便于长期维护。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib

# 非交互环境下也可保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

# 仅用于注册 SciencePlots 样式，必须保留导入
import scienceplots  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "OUTPUT" / "calibration" / "jacobian_analysis.csv"
DEFAULT_OUT_PNG = (
    PROJECT_ROOT / "OUTPUT" / "calibration" / "jacobian_elbow_mean_asc_science.png"
)
DEFAULT_OUT_SORTED = (
    PROJECT_ROOT / "OUTPUT" / "calibration" / "jacobian_mean_sorted_asc.csv"
)


def param_to_latex_symbol(param_name: str) -> str:
    """
    将参数名映射为 LaTeX 符号片段（不包含 $ 包裹）。

    参数:
        param_name: 参数名称（来自 jacobian_analysis.csv）

    返回:
        LaTeX 符号字符串（不含 $...$）
    """
    mapping = {
        "rho": r"\rho",
        "kappa": r"\kappa",
        "alpha_T": r"\alpha_T",
        "gamma_T": r"\gamma_T",
        "gamma_S": r"\gamma_S",
        "gamma_D": r"\gamma_D",
        "gamma_W": r"\gamma_W",
    }
    return mapping.get(param_name, rf"\mathrm{{{param_name}}}")


def param_to_latex(param_name: str) -> str:
    """
    将参数名映射为带 $ 包裹的 LaTeX 标签，适用于普通注释文本。
    """
    return rf"${param_to_latex_symbol(param_name)}$"


def setup_latex_style() -> None:
    """
    配置 SciencePlots + LaTeX 渲染，并放大字号。
    """
    # science 不再使用 no-latex 回退，强制启用 LaTeX 排版
    plt.style.use(["science", "grid"])
    plt.rcParams.update(
        {
            "text.usetex": True,
            # 数学公式走 LaTeX；中文文本单独用 usetex=False 渲染
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )


def get_songti_font() -> fm.FontProperties:
    """
    获取可用的宋体类字体。

    返回:
        FontProperties对象（若未找到宋体则回退到serif）
    """
    preferred_fonts = [
        "SimSun",              # Windows 宋体
        "NSimSun",             # Windows 新宋体
        "Songti SC",           # macOS 宋体
        "STSong",              # macOS/部分Linux
        "Noto Serif CJK SC",   # Linux 常见中文衬线
        "Source Han Serif SC", # 思源宋体
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            return fm.FontProperties(family=font_name)
    return fm.FontProperties(family="serif")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="绘制 Jacobian mean_abs_elasticity 的升序 Elbow 折线图（SciencePlots）"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"Jacobian摘要CSV路径（默认：{DEFAULT_CSV}）",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(DEFAULT_OUT_PNG),
        help=f"输出图片路径（默认：{DEFAULT_OUT_PNG}）",
    )
    parser.add_argument(
        "--out-sorted-csv",
        type=str,
        default=str(DEFAULT_OUT_SORTED),
        help=f"输出升序排序CSV路径（默认：{DEFAULT_OUT_SORTED}）",
    )
    return parser.parse_args()


def find_elbow_index(values: np.ndarray) -> int:
    """
    用端点连线最大垂距法寻找拐点索引（0-based）。

    参数:
        values: 升序排列的一维数值数组

    返回:
        elbow 索引（0-based）
    """
    n_points = len(values)
    if n_points <= 2:
        return 0

    x_axis = np.arange(n_points, dtype=float)
    y_axis = values.astype(float)

    # 归一化后再计算距离，避免尺度偏置
    x_norm = (x_axis - x_axis.min()) / max(x_axis.max() - x_axis.min(), 1.0)
    y_norm = (y_axis - y_axis.min()) / max(y_axis.max() - y_axis.min(), 1.0e-12)

    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    denom = math.hypot(y2 - y1, x2 - x1)
    if denom <= 1.0e-12:
        return 0

    distances = np.abs(
        (y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1
    ) / denom
    return int(np.argmax(distances))


def build_sorted_mean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建按 mean_abs_elasticity 升序排序的表。

    参数:
        df: 原始 jacobian_analysis DataFrame

    返回:
        排序后的 DataFrame（含 rank_asc）
    """
    required_cols = {"param_name", "mean_abs_elasticity"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"CSV缺少必要列: {sorted(missing_cols)}")

    sorted_df = (
        df[["param_name", "mean_abs_elasticity"]]
        .copy()
        .sort_values("mean_abs_elasticity", ascending=True)
        .reset_index(drop=True)
    )
    sorted_df["rank_asc"] = np.arange(1, len(sorted_df) + 1)
    return sorted_df


def plot_mean_elbow(sorted_df: pd.DataFrame, out_png: Path) -> Tuple[int, str, float]:
    """
    绘制升序 mean 弹性折线图并标注 elbow。

    参数:
        sorted_df: 升序排序 DataFrame
        out_png: 输出图片路径

    返回:
        (elbow_rank, elbow_param, elbow_value)
    """
    # SciencePlots 样式：启用真正 LaTeX 渲染
    setup_latex_style()
    cn_font = get_songti_font()

    values = sorted_df["mean_abs_elasticity"].to_numpy(dtype=float)
    elbow_idx = find_elbow_index(values)

    elbow_rank = int(sorted_df.loc[elbow_idx, "rank_asc"])
    elbow_param = str(sorted_df.loc[elbow_idx, "param_name"])
    elbow_value = float(sorted_df.loc[elbow_idx, "mean_abs_elasticity"])

    ranks = sorted_df["rank_asc"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot(
        ranks,
        values,
        marker="o",
        linewidth=2.8,
        markersize=7.2,
        color="#1f77b4",
        label="平均弹性（升序）",
    )
    ax.scatter(
        [elbow_rank],
        [elbow_value],
        s=130,
        color="#d62728",
        zorder=6,
        label="拐点",
    )
    ax.axvline(elbow_rank, linestyle="--", color="#d62728", alpha=0.75)
    # 左侧为低敏感区，做浅色高亮
    ax.axvspan(
        1,
        elbow_rank,
        color="#2ca02c",
        alpha=0.12,
        label="低敏感区",
    )

    ax.set_title(
        "Jacobian拐点图",
        fontproperties=cn_font,
        usetex=False,
    )
    ax.set_xlabel(
        "参数排序",
        fontproperties=cn_font,
        usetex=False,
    )
    ax.set_ylabel(r"$\bar{\epsilon}_j = \frac{1}{K}\sum_{i=1}^{K}\left|E_{ij}\right|$")
    legend = ax.legend(loc="upper left", prop=cn_font)
    for legend_text in legend.get_texts():
        legend_text.set_usetex(False)
        legend_text.set_fontproperties(cn_font)

    for _, row in sorted_df.iterrows():
        latex_label = param_to_latex(str(row["param_name"]))
        ax.annotate(
            latex_label,
            (row["rank_asc"], row["mean_abs_elasticity"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=14,
        )

    elbow_param_symbol = param_to_latex_symbol(elbow_param)
    text = (
        rf"$j^\star={elbow_rank}$" "\n"
        rf"$\theta_{{j^\star}}={elbow_param_symbol}$" "\n"
        rf"$\bar{{\epsilon}}_{{j^\star}}={elbow_value:.4f}$"
    )
    ax.annotate(
        text,
        xy=(elbow_rank, elbow_value),
        xytext=(elbow_rank + 0.5, elbow_value),
        textcoords="data",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray", "alpha": 0.9},
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg = out_png.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    # 同步输出矢量图，便于论文与排版场景放大不失真
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    return elbow_rank, elbow_param, elbow_value


def main() -> None:
    """主流程。"""
    args = parse_args()
    csv_path = Path(args.csv)
    out_png = Path(args.out_png)
    out_sorted_csv = Path(args.out_sorted_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    sorted_df = build_sorted_mean_frame(df)
    out_sorted_csv.parent.mkdir(parents=True, exist_ok=True)
    sorted_df.to_csv(out_sorted_csv, index=False, encoding="utf-8-sig")

    elbow_rank, elbow_param, elbow_value = plot_mean_elbow(sorted_df, out_png)

    print(f"输入文件: {csv_path}")
    print(f"排序CSV: {out_sorted_csv}")
    print(f"输出图片(PNG): {out_png}")
    print(f"输出图片(SVG): {out_png.with_suffix('.svg')}")
    print("\n按 mean_abs_elasticity 升序排序：")
    print(sorted_df.to_string(index=False))
    print(
        f"\nElbow结果: rank={elbow_rank}, "
        f"param={elbow_param}, mean_abs_elasticity={elbow_value:.6f}"
    )


if __name__ == "__main__":
    main()
