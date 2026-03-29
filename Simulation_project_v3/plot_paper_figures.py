#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文图表生成脚本（本地端运行）

功能：
读取服务器端产出的CSV数据，使用中文字体生成论文级图表。

前提条件：
  - 已在服务器端运行 run_paper_simulation.py 并将结果下载到本地
  - 本地安装了微软雅黑字体（Windows默认自带）

输出目录：
  OUTPUT/paper_figures/  ── 论文用图表（300 DPI PNG）

使用方式（本地端）：
  python plot_paper_figures.py                   # 生成全部图表
  python plot_paper_figures.py --only fig1 fig7  # 只生成指定图表

生成的图表：
  fig1  - §5.1 收敛指标三合一图
  fig2  - §5.1 失业率与人力资本演化
  fig3  - §5.2 就业/失业者四维分布对比
  fig4  - §5.3 均衡努力水平分布
  fig5  - §5.3 技能-努力关系散点图
  fig6  - §5.4 匹配概率回归系数图
  fig7  - §6.2 政策效果对比柱状图（失业率）
  fig8  - §6.2 政策效果对比柱状图（工资+福利）
  fig9  - §6.2 多场景失业率时序演化
  fig10 - §6.3 分群体差异化政策效果热力图
  fig11 - §7.1 参数灵敏度龙卷风图
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import scienceplots  # noqa: F401  必须先 import 才能注册 'science' 样式

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("PaperFigures")

# ============================================================
# 全局样式设置（SciencePlots 期刊风格 + 中文补丁）
# ============================================================

# SciencePlots 样式上下文：["science", "no-latex"] 提供完整轴框、内向刻度、紧凑布局
# no-latex 避免 TeX 依赖错误；实际使用时在函数内 with plt.style.context() 应用
STYLE_CTX = ["science", "no-latex"]

# 公共 savefig 配置（不在 context 内，要永久生效）
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def _apply_cn(fig=None):
    """
    中文字体补丁函数。
    SciencePlots 会将 font.family 定为 serif/sans-serif，
    需要在 context 内部再次强制覆盖字体以支持中文。
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


# 色盲友好的 Wong (2011) 调色板
# 参考: https://www.nature.com/articles/nmeth.1617
COLORS = {
    "primary":   "#0077BB",   # 蓝
    "secondary": "#009988",   # 青维
    "accent":    "#CC3311",   # 砖红
    "orange":    "#EE7733",   # 橙
    "gray":      "#BBBBBB",   # 浅灰
    "dark":      "#33bbee",   # 淡蓝
}

# 政策颜色映射：色盲友好，组合政策用深色紫红突出
POLICY_COLORS = {
    "baseline":     "#000000",   # 黑，基准
    "digital_high": "#0077BB",   # 蓝，A
    "skill_low":    "#009988",   # 青维，B
    "info_low":     "#EE7733",   # 橙，C
    "flexwork_low": "#CC3311",   # 砖红，D
    "combined":     "#AA3377",   # 紫红，F（组合最强，颜色最特殊）
}

# 统一标注框样式
ANNO_BBOX = {"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999999", "alpha": 0.85}

# 输出目录
FIGURES_DIR = PROJECT_ROOT / "OUTPUT" / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 数据目录
MFG_DIR = PROJECT_ROOT / "OUTPUT" / "mfg"
SIM_DIR = PROJECT_ROOT / "OUTPUT" / "simulation"
TABLES_DIR = PROJECT_ROOT / "OUTPUT" / "paper_tables"


# ============================================================
# 数据加载工具
# ============================================================

def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    """安全加载CSV文件"""
    if not path.exists():
        if required:
            logger.error("文件不存在: %s", path)
            raise FileNotFoundError(f"文件不存在: {path}")
        else:
            logger.warning("文件不存在（可选）: %s", path)
            return pd.DataFrame()
    return pd.read_csv(path)


# ============================================================
# 图1：收敛指标三合一图
# ============================================================

def plot_fig1_convergence():
    """§5.1 收敛指标三合一图（ΔV/Δa/Δu）"""
    logger.info("生成 Fig.1 收敛指标三合一图...")

    history = load_csv(MFG_DIR / "equilibrium_history.csv")
    valid = history.dropna(subset=["convergence_V"])

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
        fig.suptitle("MFG均衡收敛过程", fontsize=13)

        # ΔV
        ax = axes[0]
        ax.semilogy(valid["iteration"], valid["convergence_V"],
                    color=COLORS["primary"], linewidth=1.2)
        ax.set_ylabel("$|\\Delta V| / |V|$")
        ax.set_title("价值函数相对变化", fontsize=9)

        # Δa
        ax = axes[1]
        ax.semilogy(valid["iteration"], valid["convergence_a"],
                    color=COLORS["secondary"], linewidth=1.2)
        ax.set_ylabel("$|\\Delta \\bar{a}|$")
        ax.set_title("平均努力水平变化", fontsize=9)

        # Δu
        ax = axes[2]
        ax.plot(valid["iteration"], valid["convergence_u"],
                color=COLORS["orange"], linewidth=1.2)
        ax.set_xlabel("迭代轮次")
        ax.set_ylabel("$|\\Delta u|$")
        ax.set_title("失业率变化", fontsize=9)

        plt.tight_layout()
        path = FIGURES_DIR / "fig1_convergence.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图2：失业率与人力资本演化
# ============================================================

def plot_fig2_evolution():
    """§5.1 失业率与人力资本演化四又子图"""
    logger.info("生成 Fig.2 失业率与人力资本演化...")

    history = load_csv(MFG_DIR / "equilibrium_history.csv")

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
        fig.suptitle("基准均衡演化过程", fontsize=13)

        # 失业率：用蓝色实线，无marker（数据密时线条更清晰）
        ax = axes[0, 0]
        ax.plot(history["iteration"],
                history["unemployment_rate"] * 100,
                color=COLORS["primary"], linewidth=1.2)
        ax.set_ylabel("失业率 (%)")
        ax.set_title("失业率演化", fontsize=9)

        # 平均技能S
        ax = axes[0, 1]
        ax.plot(history["iteration"], history["mean_S"],
                color=COLORS["secondary"], linewidth=1.2)
        ax.set_ylabel("平均技能水平 $S$")
        ax.set_title("技能水平演化", fontsize=9)

        # 平均数字素养D
        ax = axes[1, 0]
        ax.plot(history["iteration"], history["mean_D"],
                color=COLORS["orange"], linewidth=1.2)
        ax.set_xlabel("迭代轮次")
        ax.set_ylabel("平均数字素养 $D$")
        ax.set_title("数字素养演化", fontsize=9)

        # 平均努力水平
        ax = axes[1, 1]
        ax.plot(history["iteration"], history["mean_effort"],
                color=COLORS["accent"], linewidth=1.2)
        ax.set_xlabel("迭代轮次")
        ax.set_ylabel("平均努力水平 $a^*$")
        ax.set_title("最优努力水平演化", fontsize=9)

        plt.tight_layout()
        path = FIGURES_DIR / "fig2_evolution.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图3：就业/失业者四维分布对比
# ============================================================

def plot_fig3_status_comparison():
    """§5.2 就业/失业者四维分布对比箱线图"""
    logger.info("生成 Fig.3 就业/失业者对比箱线图...")

    individuals = load_csv(MFG_DIR / "equilibrium_individuals.csv")

    # 就/失业者分组
    emp = individuals[individuals["employment_status"] == "employed"]
    unemp = individuals[individuals["employment_status"] == "unemployed"]

    variables = [
        ("T", "劳动供给时间 $T$ (小时/周)"),
        ("S", "工作技能水平 $S$"),
        ("D", "数字素养水平 $D$"),
        ("W", "期望工资 $W$ (元/月)"),
    ]

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        fig.suptitle("均衡状态下就业者与失业者分布对比", fontsize=11)

        for idx, (var, label) in enumerate(variables):
            ax = axes[idx // 2, idx % 2]

            # 用 matplotlib 原生 violinplot 替代 seaborn boxplot
            parts = ax.violinplot(
                [emp[var].dropna().values, unemp[var].dropna().values],
                positions=[0, 1],
                showmedians=True,
                showextrema=False,
                widths=0.6,
            )
            # 就业者蓝色，失业者砖红色
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(COLORS["primary"] if i == 0 else COLORS["accent"])
                pc.set_edgecolor("white")
                pc.set_alpha(0.75)
            parts["cmedians"].set_colors("black")
            parts["cmedians"].set_linewidth(1.5)

            # 均值散点+标注
            for i, (status, col) in enumerate([("employed", COLORS["primary"]),
                                                ("unemployed", COLORS["accent"])]):
                subset = individuals[individuals["employment_status"] == status]
                mean_val = subset[var].mean()
                ax.scatter(i, mean_val, s=30, color="white",
                           edgecolors=col, linewidths=1.5, zorder=5)
                ax.annotate(f"{mean_val:.1f}",
                            xy=(i, mean_val), xytext=(10, 0),
                            textcoords="offset points",
                            fontsize=8, va="center", ha="left",
                            bbox=ANNO_BBOX)

            ax.set_ylabel(label, fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["就业者", "失业者"])

        plt.tight_layout()
        path = FIGURES_DIR / "fig3_status_comparison.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图4：均衡努力水平分布
# ============================================================

def plot_fig4_effort_distribution():
    """§5.3 均衡努力水平分布"""
    logger.info("生成 Fig.4 努力水平分布...")

    policy = load_csv(MFG_DIR / "equilibrium_policy.csv")
    individuals = load_csv(MFG_DIR / "equilibrium_individuals.csv")

    # 仅失业者的努力水平
    unemployed_mask = individuals["employment_status"] == "unemployed"
    effort_u = policy.loc[unemployed_mask, "a_optimal"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(effort_u, bins=30, density=True, alpha=0.6,
            color=COLORS["primary"], edgecolor="white", linewidth=0.5,
            label="直方图")

    # KDE曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(effort_u.dropna())
    x = np.linspace(effort_u.min(), effort_u.max(), 200)
    ax.plot(x, kde(x), color=COLORS["accent"], linewidth=2, label="核密度估计")

    # 均值和中位数
    mean_a = effort_u.mean()
    median_a = effort_u.median()
    ax.axvline(mean_a, color=COLORS["orange"], linestyle="--", linewidth=2,
               label=f"均值 = {mean_a:.3f}")
    ax.axvline(median_a, color=COLORS["secondary"], linestyle=":", linewidth=2,
               label=f"中位数 = {median_a:.3f}")

    ax.set_xlabel("最优努力水平 a*")
    ax.set_ylabel("密度")
    ax.set_title(
        f"均衡状态下失业者最优努力水平分布 "
        f"(N={len(effort_u)}, 标准差={effort_u.std():.3f})",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_effort_distribution.png"
    plt.savefig(path)
    plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图5：技能-努力散点图
# ============================================================

def plot_fig5_skill_effort():
    """§5.3 低努力均衡展示：左图人数比例，右图各努力层技能箱线图"""
    logger.info("生成 Fig.5 努力层次分布与技能对比...")

    import re as _re
    policy = load_csv(MFG_DIR / "equilibrium_policy.csv")
    individuals = load_csv(MFG_DIR / "equilibrium_individuals.csv")

    unemployed_mask = individuals["employment_status"] == "unemployed"
    S_u = individuals.loc[unemployed_mask, "S"].values
    a_u = policy.loc[unemployed_mask, "a_optimal"].values
    D_u = individuals.loc[unemployed_mask, "D"].values

    # 将连续努力值归档到离散层次（基于实际数据分布）
    # 0 → 放弃努力；0.1/0.2/0.3 等离散正值 → 积极投入
    def classify_effort(a):
        """将努力值量化为4个层次"""
        if a < 0.05:
            return "放弃努力\n(a=0)"
        elif a < 0.15:
            return "低努力\n(a=0.1)"
        elif a < 0.25:
            return "中努力\n(a=0.2)"
        else:
            return "高努力\n(a≥0.3)"

    effort_labels = np.array([classify_effort(a) for a in a_u])
    level_order = ["放弃努力\n(a=0)", "低努力\n(a=0.1)", "中努力\n(a=0.2)", "高努力\n(a≥0.3)"]
    # 各层次占比
    counts = {lv: np.sum(effort_labels == lv) for lv in level_order}
    total = len(effort_labels)
    ratios = [counts[lv] / total * 100 for lv in level_order]

    # 各层次技能S数据
    S_by_level = [S_u[effort_labels == lv] for lv in level_order]

    # 各层次颜色：放弃=砖红，低/中/高=蓝色渐变
    level_colors = [
        COLORS["accent"],    # 放弃: 砖红
        "#6baed6",           # 低努力: 浅蓝
        COLORS["secondary"], # 中努力: 青维
        COLORS["primary"],   # 高努力: 深蓝
    ]

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("失业者努力水平分布与技能差异", fontsize=13)

        # ── 左图：各努力层次人数占比（水平条形） ──
        ax = axes[0]
        bars = ax.barh(range(4), ratios, color=level_colors, alpha=0.8,
                       edgecolor="white", height=0.55)
        # 标注：放弃努力(大柱)内部白字，其他小柱仅当比例>0.5%才标注在外部
        for bar, pct, lv in zip(bars, ratios, level_order):
            n = counts[lv]
            if pct > 50:
                # 宽柱：标注放在条内部，白色文字，不会超出
                ax.annotate(
                    f"{pct:.1f}%  (n={n})",
                    xy=(pct / 2, bar.get_y() + bar.get_height() / 2),
                    fontsize=8, va="center", ha="center", color="white",
                )
            elif pct > 0.1:
                # 小柱：标注放在外部（小柱本身很短，外部空间充裕）
                ax.annotate(
                    f"{pct:.1f}%  (n={n})",
                    xy=(pct, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, va="center", ha="left",
                )
        # xlim：以最大比例为右边界（大柱内部标注，不需要右侧留白）
        ax.set_xlim(0, max(ratios) * 1.05)
        ax.set_yticks(range(4))
        ax.set_yticklabels(level_order, fontsize=9)
        ax.set_xlabel("占失业者比例 (%)")
        ax.set_title("努力选择分布", fontsize=9)

        # ── 右图：各努力层次的技能S箱线图 ──
        ax = axes[1]
        # 过滤掉空层次
        valid_levels = [(i, lv, data) for i, (lv, data)
                        in enumerate(zip(level_order, S_by_level)) if len(data) > 0]
        positions = [item[0] for item in valid_levels]
        plot_data_box = [item[2] for item in valid_levels]
        box_colors = [level_colors[item[0]] for item in valid_levels]

        bp = ax.boxplot(
            plot_data_box,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        # 标注各组均值
        for i, (pos, data) in enumerate(zip(positions, plot_data_box)):
            ax.annotate(
                f"{np.mean(data):.1f}",
                xy=(pos, np.mean(data)), xytext=(8, 0),
                textcoords="offset points",
                fontsize=8, va="center", bbox=ANNO_BBOX,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([item[1] for item in valid_levels], fontsize=8)
        ax.set_ylabel("工作技能水平 $S$")
        ax.set_title("各努力层次的技能分布", fontsize=9)

        plt.tight_layout()
        path = FIGURES_DIR / "fig5_skill_effort.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图6：匹配概率回归系数图
# ============================================================

def plot_fig6_matching_coefficients():
    """§5.4 匹配概率回归系数森林图"""
    logger.info("生成 Fig.6 匹配概率回归系数图...")

    coef_path = TABLES_DIR / "table_5_4_matching_coefficients.csv"
    if not coef_path.exists():
        logger.warning("回归系数表不存在，跳过 Fig.6")
        return

    coef_df = load_csv(coef_path)

    # 排除截距项，只看特征系数
    features = coef_df[coef_df["feature"] != "intercept"].copy()
    features = features.sort_values("coefficient", ascending=True)

    # 特征名称中文映射
    feature_labels = {
        "T": "劳动供给时间 T",
        "S": "工作技能 S",
        "D": "数字素养 D",
        "W": "期望工资 W",
        "sigma": "个体特征 σ",
        "theta": "市场紧张度 θ",
        "T_normalized": "T (标准化)",
        "S_normalized": "S (标准化)",
        "D_normalized": "D (标准化)",
        "W_normalized": "W (标准化)",
    }
    features["label"] = features["feature"].map(
        lambda x: feature_labels.get(x, x)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.8)))

    colors = [COLORS["primary"] if v > 0 else COLORS["accent"]
              for v in features["coefficient"]]

    y_pos = range(len(features))
    ax.barh(y_pos, features["coefficient"], color=colors, height=0.6, alpha=0.8)

    # 如果有标准误，绘制误差棒
    if "std_error" in features.columns and not features["std_error"].isna().all():
        ax.errorbar(
            features["coefficient"], y_pos,
            xerr=features["std_error"] * 1.96,
            fmt="none", ecolor="black", elinewidth=1.5, capsize=3,
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features["label"])
    ax.set_xlabel("Logistic回归系数")
    ax.set_title("匹配概率决定因素（Logistic回归系数）")
    ax.grid(alpha=0.3)

    # 添加显著性标注（对齐 jacobian elbow 标注框样式）
    if "p_value" in features.columns:
        for i, (_, row) in enumerate(features.iterrows()):
            p = row.get("p_value", 1)
            if not np.isnan(p):
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                if stars:
                    ax.annotate(stars, xy=(row["coefficient"], i),
                                fontsize=12,
                                ha="left" if row["coefficient"] > 0 else "right",
                                va="center",
                                bbox=ANNO_BBOX)

    plt.tight_layout()
    path = FIGURES_DIR / "fig6_matching_coefficients.png"
    plt.savefig(path)
    plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图7：政策效果对比柱状图（失业率）
# ============================================================

def plot_fig7_policy_unemployment():
    """§6.2 政策效果对比竖向柱状图（失业率变化）"""
    logger.info("生成 Fig.7 政策效果对比（失业率）...")

    import re as _re
    effects_path = TABLES_DIR / "table_6_2_policy_effects.csv"
    if not effects_path.exists():
        effects_path = SIM_DIR / "policy_effects_vs_baseline.csv"

    effects = load_csv(effects_path)

    # 仅保留核心场景，排除基准
    reported_scenarios = list(POLICY_COLORS.keys())
    effects = effects[effects["scenario_name"].isin(reported_scenarios)].copy()

    y_col = "delta_u_pct" if "delta_u_pct" in effects.columns else "pct_change_unemployment"
    label_col = "scenario_display_name"

    # 按降幅升序（效果最好排最后，对应竖图从左到右降幅增大）
    plot_data = effects[effects["scenario_name"] != "baseline"].copy()
    plot_data = plot_data.sort_values(y_col, ascending=False)

    # 去掉括号标注(高)/(低)，再去掉前缀编号如"A2-"/"B1-"
    clean_labels = [_re.sub(r"^[A-Za-z]\d*-", "",
                    _re.sub(r"[（(][^)）]*[)）]", "", lbl)).strip()
                    for lbl in plot_data[label_col]]

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, ax = plt.subplots(figsize=(8, 5))

        # 竖向柱状图，失业率下降为负值
        bar_colors = [POLICY_COLORS[n] for n in plot_data["scenario_name"]]
        bars = ax.bar(range(len(plot_data)), plot_data[y_col],
                      color=bar_colors, alpha=0.75,
                      edgecolor="white", width=0.6)

        # 竖向标注：降值在条下方，正值在上方
        for bar, val in zip(bars, plot_data[y_col]):
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, -12 if val < 0 else 4),
                textcoords="offset points",
                fontsize=8, ha="center", va="top" if val < 0 else "bottom",
            )

        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(clean_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("失业率变化 (%)")
        ax.set_title("各政策场景的失业率变化效果（相对基准）")
        ax.axhline(0, color="black", linewidth=0.8)
        # 扩展 ylim 下限，为柱内标注文字留出空间
        min_val = plot_data[y_col].min()
        ax.set_ylim(min_val * 1.18, 0.5)

        plt.tight_layout()
        path = FIGURES_DIR / "fig7_policy_unemployment.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图8：政策效果对比柱状图（工资+福利）
# ============================================================

def plot_fig8_policy_wage_welfare():
    """§6.2 政策效果对比（工资和福利变化），去掉括号内高低标注"""
    logger.info("生成 Fig.8 政策效果对比（工资+福利）...")

    import re as _re
    effects_path = TABLES_DIR / "table_6_2_policy_effects.csv"
    if not effects_path.exists():
        logger.warning("政策效果表不存在，跳过 Fig.8")
        return

    effects = load_csv(effects_path)
    reported_scenarios = list(POLICY_COLORS.keys())
    effects = effects[effects["scenario_name"].isin(reported_scenarios)].copy()
    effects = effects[effects["scenario_name"] != "baseline"].copy()

    def clean_label(s):
        """去掉括号标注并去掉前缀编号如\"A2-\"/\"B1-\""""
        s = _re.sub(r"[（(][^)）]*[)）]", "", s)  # 去括号
        s = _re.sub(r"^[A-Za-z]\d*-", "", s)       # 去前缀编号
        return s.strip()

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        effects_sorted = effects.sort_values("delta_wage", ascending=True)
        clean_wage_labels = [clean_label(s) for s in effects_sorted["scenario_display_name"]]

        # 工资变化：正值用青维，负值用砖红
        ax = axes[0]
        colors = [COLORS["secondary"] if v >= 0 else COLORS["accent"]
                  for v in effects_sorted["delta_wage"]]
        ax.barh(range(len(effects_sorted)), effects_sorted["delta_wage"],
                color=colors, alpha=0.75, height=0.55)
        ax.set_yticks(range(len(effects_sorted)))
        ax.set_yticklabels(clean_wage_labels, fontsize=9)
        ax.set_xlabel("平均工资变化 (元/月)")
        ax.set_title("就业者平均工资变化")
        ax.axvline(0, color="black", linewidth=0.8)

        # 福利变化（V_U）
        if "delta_welfare_U" in effects.columns:
            effects_sorted2 = effects.sort_values("delta_welfare_U", ascending=True)
            clean_welfare_labels = [clean_label(s)
                                    for s in effects_sorted2["scenario_display_name"]]
            ax = axes[1]
            colors2 = [COLORS["primary"] if v >= 0 else COLORS["accent"]
                       for v in effects_sorted2["delta_welfare_U"]]
            ax.barh(range(len(effects_sorted2)), effects_sorted2["delta_welfare_U"],
                    color=colors2, alpha=0.75, height=0.55)
            ax.set_yticks(range(len(effects_sorted2)))
            ax.set_yticklabels(clean_welfare_labels, fontsize=9)
            ax.set_xlabel("失业者期望价值变化 $\\Delta V_U$")
            ax.set_title("失业者福利变化")
            ax.axvline(0, color="black", linewidth=0.8)

        plt.tight_layout()
        path = FIGURES_DIR / "fig8_policy_wage_welfare.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图9：多场景失业率时序演化
# ============================================================

def plot_fig9_policy_timeseries():
    """§6.2 多场景失业率时序演化：大图全局，右下角嵌入最后30轮局部放大图"""
    logger.info("生成 Fig.9 多场景时序对比...")

    ts_path = SIM_DIR / "all_scenarios_time_series.csv"
    if not ts_path.exists():
        logger.warning("时间序列数据不存在，跳过 Fig.9")
        return

    ts_data = load_csv(ts_path)

    scenario_labels = {
        "baseline": "基准",
        "digital_high": "A-数字素养提升",
        "skill_low": "B-职业技能提升",
        "info_low": "C-就业信息优化",
        "flexwork_low": "D-灵活工时",
        "combined": "F-组合政策",
    }

    with plt.style.context(STYLE_CTX):
        _apply_cn()
        fig, ax = plt.subplots(figsize=(10, 5))

        # ── 大图：全局演化路径，无终值标注 ──
        all_iters = {}   # 存储各场景数据供 inset 使用
        all_uvals = {}

        for scenario_name, label in scenario_labels.items():
            scenario_data = ts_data[ts_data["scenario_name"] == scenario_name]
            if len(scenario_data) == 0:
                continue

            color = POLICY_COLORS.get(scenario_name, COLORS["gray"])
            if scenario_name == "baseline":
                lw, ls = 1.5, "-"
            elif scenario_name == "combined":
                lw, ls = 1.5, "--"
            else:
                lw, ls = 1.0, "--"

            u_vals = scenario_data["unemployment_rate"].values * 100
            iters = scenario_data["iteration"].values
            all_iters[scenario_name] = iters
            all_uvals[scenario_name] = u_vals

            ax.plot(iters, u_vals, linewidth=lw, label=label,
                    color=color, linestyle=ls)

        ax.set_xlabel("迭代轮次")
        ax.set_ylabel("失业率 (%)")
        ax.set_title("不同政策场景下失业率的演化路径")
        ax.legend(loc="upper right", ncol=2, fontsize=10, handlelength=2)

        # ── 嵌入小图：最后30轮局部放大，在此标注各场景终值 ──
        # 小图位置：[left, bottom, width, height]，高度拉高到0.52
        ax_inset = ax.inset_axes([0.18, 0.40, 0.44, 0.52])

        for scenario_name, label in scenario_labels.items():
            if scenario_name not in all_iters:
                continue
            iters = all_iters[scenario_name]
            u_vals = all_uvals[scenario_name]
            color = POLICY_COLORS.get(scenario_name, COLORS["gray"])

            # 取最后30轮数据
            mask = iters >= (iters[-1] - 30)
            iters_tail = iters[mask]
            u_tail = u_vals[mask]

            if scenario_name == "baseline":
                lw, ls = 1.5, "-"
            elif scenario_name == "combined":
                lw, ls = 1.5, "--"
            else:
                lw, ls = 1.0, "--"

            ax_inset.plot(iters_tail, u_tail, linewidth=lw,
                          color=color, linestyle=ls)

            # 在 inset 小图中标注终值
            ax_inset.annotate(
                f"{u_tail[-1]:.2f}%",
                xy=(iters_tail[-1], u_tail[-1]),
                xytext=(3, 0), textcoords="offset points",
                fontsize=6.5, va="center", color=color,
            )

        ax_inset.set_title("末端30轮演化", fontsize=8)
        ax_inset.tick_params(labelsize=7)
        # 纵轴山兣间距设置为0.3，避免刻度过密
        import matplotlib.ticker as _ticker
        ax_inset.yaxis.set_major_locator(_ticker.MultipleLocator(0.3))
        # 添加边框让小图更突出
        for spine in ax_inset.spines.values():
            spine.set_linewidth(0.8)

        plt.tight_layout()
        path = FIGURES_DIR / "fig9_policy_timeseries.png"
        plt.savefig(path)
        plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图10：分群体差异化政策效果热力图
# ============================================================

def plot_fig10_subgroup_heatmap():
    """§6.3 分群体差异化政策效果热力图"""
    logger.info("生成 Fig.10 分群体差异化效果...")

    subgroup_path = TABLES_DIR / "table_6_3_subgroup_effects.csv"
    if not subgroup_path.exists():
        logger.warning("分群体数据不存在，跳过 Fig.10")
        return

    subgroup = load_csv(subgroup_path)

    # 计算基准场景各群体的失业率
    baseline_high = subgroup[
        (subgroup["scenario"] == "baseline") & (subgroup["subgroup"] == "high_skill")
    ]
    baseline_low = subgroup[
        (subgroup["scenario"] == "baseline") & (subgroup["subgroup"] == "low_skill")
    ]

    if len(baseline_high) == 0 or len(baseline_low) == 0:
        logger.warning("基准场景分群体数据不完整，跳过 Fig.10")
        return

    base_u_high = float(baseline_high["unemployment_rate"].iloc[0])
    base_u_low = float(baseline_low["unemployment_rate"].iloc[0])

    # 计算各场景相对基准的失业率变化
    scenarios = subgroup["scenario"].unique()
    scenarios = [s for s in scenarios if s != "baseline"]

    rows = []
    for scenario in sorted(scenarios):
        for group, base_u in [("high_skill", base_u_high),
                               ("low_skill", base_u_low)]:
            row = subgroup[
                (subgroup["scenario"] == scenario) & (subgroup["subgroup"] == group)
            ]
            if len(row) > 0:
                u_rate = float(row["unemployment_rate"].iloc[0])
                delta = (u_rate - base_u) * 100
                rows.append({
                    "scenario": scenario,
                    "subgroup": "高技能" if group == "high_skill" else "低技能",
                    "delta_u_pp": delta,
                })

    if not rows:
        logger.warning("无有效的分群体数据，跳过 Fig.10")
        return

    pivot_df = pd.DataFrame(rows).pivot(
        index="scenario", columns="subgroup", values="delta_u_pp"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        pivot_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
        center=0, ax=ax, linewidths=0.5,
        cbar_kws={"label": "失业率变化 (百分点)"},
        annot_kws={"fontsize": 11},
    )
    ax.set_xlabel("群体")
    ax.set_ylabel("政策场景")
    ax.set_title("分群体差异化政策效果（失业率变化，百分点）")

    plt.tight_layout()
    path = FIGURES_DIR / "fig10_subgroup_heatmap.png"
    plt.savefig(path)
    plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 图11：参数灵敏度龙卷风图
# ============================================================

def plot_fig11_sensitivity():
    """§7.1 参数灵敏度龙卷风图"""
    logger.info("生成 Fig.11 参数灵敏度龙卷风图...")

    sens_path = TABLES_DIR / "table_7_1_sensitivity_analysis.csv"
    if not sens_path.exists():
        logger.warning("灵敏度分析数据不存在，跳过 Fig.11")
        return

    sens = load_csv(sens_path)

    # 参数名中文映射
    param_labels = {
        "rho": "ρ (贴现因子)",
        "kappa": "κ (努力成本)",
        "alpha_T": "α_T (工时负效用)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (metric, metric_label) in enumerate([
        ("unemployment_rate", "失业率"),
        ("mean_wage", "平均工资 (元/月)"),
    ]):
        ax = axes[ax_idx]

        params = sens["parameter"].unique()
        y_positions = np.arange(len(params))

        for i, param in enumerate(params):
            param_data = sens[sens["parameter"] == param].sort_values("perturbation_pct")

            if len(param_data) < 4:
                continue

            # ±20% 的范围
            val_low_20 = param_data[param_data["perturbation_pct"] == -20][metric]
            val_high_20 = param_data[param_data["perturbation_pct"] == 20][metric]
            # ±10% 的范围
            val_low_10 = param_data[param_data["perturbation_pct"] == -10][metric]
            val_high_10 = param_data[param_data["perturbation_pct"] == 10][metric]

            if len(val_low_20) > 0 and len(val_high_20) > 0:
                low_20, high_20 = float(val_low_20.iloc[0]), float(val_high_20.iloc[0])
                ax.barh(i, high_20 - low_20, left=low_20, height=0.5,
                        color=COLORS["primary"], alpha=0.4, label="±20%" if i == 0 else "")

            if len(val_low_10) > 0 and len(val_high_10) > 0:
                low_10, high_10 = float(val_low_10.iloc[0]), float(val_high_10.iloc[0])
                ax.barh(i, high_10 - low_10, left=low_10, height=0.3,
                        color=COLORS["primary"], alpha=0.8, label="±10%" if i == 0 else "")

        ax.set_yticks(y_positions)
        ax.set_yticklabels([param_labels.get(p, p) for p in params])
        ax.set_xlabel(metric_label)
        ax.set_title(f"{metric_label}的参数灵敏度")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("参数灵敏度分析（龙卷风图）", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "fig11_sensitivity.png"
    plt.savefig(path)
    plt.close()
    logger.info("  → 已保存: %s", path)


# ============================================================
# 主入口
# ============================================================

# 图表注册表
FIGURE_REGISTRY = {
    "fig1": ("§5.1 收敛指标三合一图", plot_fig1_convergence),
    "fig2": ("§5.1 失业率与人力资本演化", plot_fig2_evolution),
    "fig3": ("§5.2 就业/失业者四维分布对比", plot_fig3_status_comparison),
    "fig4": ("§5.3 均衡努力水平分布", plot_fig4_effort_distribution),
    "fig5": ("§5.3 技能-努力散点图", plot_fig5_skill_effort),
    "fig6": ("§5.4 匹配概率回归系数图", plot_fig6_matching_coefficients),
    "fig7": ("§6.2 政策效果对比（失业率）", plot_fig7_policy_unemployment),
    "fig8": ("§6.2 政策效果对比（工资+福利）", plot_fig8_policy_wage_welfare),
    "fig9": ("§6.2 多场景失业率时序对比", plot_fig9_policy_timeseries),
    "fig10": ("§6.3 分群体差异化效果热力图", plot_fig10_subgroup_heatmap),
}


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="论文图表生成脚本（本地端运行）")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="只生成指定图表，如 --only fig1 fig7",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用图表",
    )
    args = parser.parse_args()

    if args.list:
        print("\n可用图表列表：")
        print("-" * 50)
        for key, (desc, _) in FIGURE_REGISTRY.items():
            print(f"  {key:6s}  {desc}")
        print()
        return

    logger.info("=" * 80)
    logger.info("论文图表生成脚本（本地端）")
    logger.info("输出目录: %s", FIGURES_DIR)
    logger.info("=" * 80)

    # 确定要生成的图表
    if args.only:
        figures_to_plot = args.only
    else:
        figures_to_plot = list(FIGURE_REGISTRY.keys())

    success_count = 0
    fail_count = 0

    for fig_name in figures_to_plot:
        if fig_name not in FIGURE_REGISTRY:
            logger.warning("未知图表: %s，跳过", fig_name)
            fail_count += 1
            continue

        desc, func = FIGURE_REGISTRY[fig_name]
        try:
            func()
            success_count += 1
        except FileNotFoundError as e:
            logger.error("  × %s 失败（缺少数据文件）: %s", fig_name, e)
            fail_count += 1
        except Exception as e:
            logger.error("  × %s 失败: %s", fig_name, e, exc_info=True)
            fail_count += 1

    logger.info("=" * 80)
    logger.info("图表生成完成：成功 %d，失败 %d", success_count, fail_count)
    logger.info("图表保存位置: %s", FIGURES_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
