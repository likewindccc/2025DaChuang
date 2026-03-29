"""
clds_compute_moments.py
=======================
CLDS数据目标矩计算脚本（整合版）

功能：
    从 CLDS 2016年和2018年个体问卷数据中，计算农村女性劳动力市场的8个核心目标矩
    （M1-M8）和1个辅助矩（M9），并输出 target_moments_clds.yaml 配置文件。

数据来源：
    - CLDS 2018年个体问卷（主截面数据）
    - CLDS 2016年个体问卷（面板追踪数据，用于M5/M6转移率计算）

样本定义：
    性别=女，户口类型 I1_3_2 ∈ {2, 4}（农业/失地农业），年龄15-64岁

目标矩：
    M1 unemployment_rate    失业率（ILO定义）
    M2 mean_wage            平均月工资
    M3 std_wage             工资标准差
    M4 mean_weekly_hours    平均周工时
    M5 job_finding_rate     月度就业转移率（面板计算）
    M6 separation_rate      月度分离率（面板计算）
    M7 wage_iqr_ratio       工资P75/P25比值
    M8 std_weekly_hours     工时标准差
    M9 digital_job_ratio    数字技能岗位比例（辅助矩）

输出：
    target_moments_clds.yaml（与本脚本同目录）

用法：
    python clds_compute_moments.py
    python clds_compute_moments.py --output /path/to/target_moments.yaml

版本历史：
    2026-03-02  v1.0  初始版本，整合探索期所有脚本
    2026-03-02  v1.1  修复M5/M6缺失值处理，取消静默兜底，改为crash early
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────

# 当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CLDS 2018年个体问卷（主截面数据）
DTA_2018 = os.path.join(
    SCRIPT_DIR,
    r"2018\CLDS2018Stata（转码后）\2018个体问卷 （191111）.dta"
)

# CLDS 2016年个体问卷（面板追踪数据）
DTA_2016 = os.path.join(
    SCRIPT_DIR,
    r"CLDS2016 适用STATA14及以上\individual2016.dta"
)

# 输出YAML路径（默认与脚本同目录）
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "target_moments_clds.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载函数
# ─────────────────────────────────────────────────────────────────────────────

def load_2018(dta_path: str) -> pd.DataFrame:
    """
    加载2018年CLDS个体问卷关键列。

    参数:
        dta_path: .dta文件路径

    返回:
        DataFrame，包含筛选所需的关键变量
    """
    # 需要加载的关键变量
    cols_needed = [
        "IID2018",       # 个体唯一编号（16位）
        "Igender",       # 性别（1男，2女）
        "I1_3_2",        # 户口类型（2=农业，4=失地农业）
        "birthyear",     # 出生年
        "I2_1",          # 最高学历
        "I3_1",          # 2017年以来是否工作过（1=是，2=否）→ 就业/失业主变量
        "I3a_7_0",       # 目前工作状态（1=有工作，2=无工作）→ 面板口径诊断
        "I3c_5",         # 最近3个月是否找工作（1=是）→ ILO失业判断条件
        "I3a_6",         # 2017年总收入（元/年）→ 月工资来源
        "I3a_1",         # 一周工作时长（小时/周）→ 主要工时变量
        "I3a_2",         # 过去一周工作时长（备用工时变量）
        "I3a1_19_4",     # 工作中使用互联网（1=经常，2=有时，3=较少，4=否）
    ]

    logger.info("加载2018年数据: %s", dta_path)
    it = pd.read_stata(dta_path, iterator=True)
    vl = it.variable_labels()
    load_cols = [c for c in cols_needed if c in vl]
    logger.info("  实际加载 %d 个关键列（共 %d 个变量）", len(load_cols), len(vl))

    df = pd.read_stata(dta_path, columns=load_cols, convert_categoricals=False)
    df["age"] = 2018 - df["birthyear"]
    # 构建面板匹配用的字符串ID（去除.0后缀）
    df["IID_key"] = df["IID2018"].astype(str).str.replace(r"\.0$", "", regex=True)
    logger.info("  2018年数据加载完成：%d 行", len(df))
    return df


def load_2016(dta_path: str) -> pd.DataFrame:
    """
    加载2016年CLDS个体问卷关键列。

    注意：
        2016年性别列名为 `gender`（不是 `Igender`）。
        就业状态变量为 `I3a_7_0`（1=有工作，2=无工作），
        而不是2018年的 `I3_1`。

    参数:
        dta_path: .dta文件路径

    返回:
        DataFrame
    """
    cols_needed = [
        "IID2016",    # 个体唯一编号（格式与IID2018相同，追访时保持不变）
        "gender",     # 性别（注意：2016年列名为gender，不是Igender）
        "I1_3_2",     # 户口类型
        "birthyear",  # 出生年
        "rtype",      # 样本类型（0=新增，1=追访）
        "I3a_7_0",    # 目前工作状态（1=有工作，2=无工作）→ 2016年就业主变量
    ]

    logger.info("加载2016年数据: %s", dta_path)
    it = pd.read_stata(dta_path, iterator=True)
    vl = it.variable_labels()
    load_cols = [c for c in cols_needed if c in vl]
    logger.info("  实际加载 %d 个关键列（共 %d 个变量）", len(load_cols), len(vl))

    df = pd.read_stata(dta_path, columns=load_cols, convert_categoricals=False)
    df["age"] = 2016 - df["birthyear"]
    df["IID_key"] = df["IID2016"].astype(str).str.replace(r"\.0$", "", regex=True)
    logger.info(
        "  2016年数据加载完成：%d 行（新增:%d，追访:%d）",
        len(df),
        (df.get("rtype", pd.Series()) == 0).sum(),
        (df.get("rtype", pd.Series()) == 1).sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 子样本筛选函数
# ─────────────────────────────────────────────────────────────────────────────

def filter_rural_female(df: pd.DataFrame, gender_col: str = "Igender") -> pd.DataFrame:
    """
    筛选农村女性劳动年龄人口（15-64岁）。

    筛选条件：
        - 性别 = 女（{gender_col} == 2）
        - 户口 = 农业或失地农业（I1_3_2 ∈ {2, 4}）
        - 年龄 ∈ [15, 64]

    参数:
        df:         原始数据框（需包含age列）
        gender_col: 性别列名（2018年为Igender，2016年为gender）

    返回:
        筛选后的子样本（副本）
    """
    mask = (
        (df[gender_col] == 2.0)
        & (df["I1_3_2"].isin([2.0, 4.0]))
        & (df["age"] >= 15)
        & (df["age"] <= 64)
    )
    result = df[mask].copy()
    logger.info(
        "  农村女性子样本（%s==2, I1_3_2∈{2,4}, age∈[15,64]）：%d 人",
        gender_col, len(result),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 目标矩计算函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_m1_unemployment(rf18: pd.DataFrame) -> dict:
    """
    M1：失业率（ILO三分法）

    判断逻辑：
        就业    = I3_1 == 1
        失业    = I3_1 == 2  AND  I3c_5 == 1（最近3个月找过工作）
        非劳动力 = I3_1 == 2  AND  I3c_5 != 1

    返回:
        dict with value, n, note
    """
    emp = (rf18["I3_1"] == 1)
    unemp = (rf18["I3_1"] == 2) & (rf18["I3c_5"] == 1)
    emp_n = int(emp.sum())
    unemp_n = int(unemp.sum())
    lf_n = emp_n + unemp_n  # 劳动力人口

    if lf_n == 0:
        raise ValueError("M1计算失败：劳动力样本为0，无法计算失业率。")
    value = unemp_n / lf_n
    logger.info(
        "M1 unemployment_rate = %.4f  (就业:%d, 失业:%d, 劳动力:%d)",
        value, emp_n, unemp_n, lf_n,
    )
    return {
        "value": round(float(value), 4),
        "n": lf_n,
        "employed_n": emp_n,
        "unemployed_n": unemp_n,
        "description": "失业率（ILO定义：失业/劳动力，失业=无工作且最近3个月积极找工作）",
        "confidence": "HIGH",
        "note": f"劳动参与率={emp_n/len(rf18)*100:.1f}%；宽泛非就业率（包含非劳动力）={(rf18['I3_1']==2).sum()/len(rf18)*100:.1f}%",
    }


def compute_wage_moments(rf18: pd.DataFrame) -> dict:
    """
    M2/M3/M7：工资相关矩（就业者子样本）

    数据来源：
        I3a_6（2017年总收入，元/年），折算为月收入（/12）
        使用IQR 2.5倍法过滤异常值

    返回:
        dict with m2, m3, m7各自的结果
    """
    employed = rf18[rf18["I3_1"] == 1].copy()
    annual = employed["I3a_6"].dropna()
    annual_pos = annual[(annual > 0) & (annual < 2_400_000)]
    if len(annual_pos) == 0:
        raise ValueError("M2/M3/M7计算失败：工资样本为空（I3a_6有效值为0）。")

    # IQR 2.5倍过滤
    q1, q3 = annual_pos.quantile(0.25), annual_pos.quantile(0.75)
    iqr_range = q3 - q1
    annual_clean = annual_pos[
        (annual_pos >= q1 - 2.5 * iqr_range)
        & (annual_pos <= q3 + 2.5 * iqr_range)
    ]
    if len(annual_clean) == 0:
        raise ValueError("M2/M3/M7计算失败：IQR过滤后工资样本为空。")
    monthly = annual_clean / 12

    m2 = float(monthly.mean())
    m3 = float(monthly.std())
    p25 = float(monthly.quantile(0.25))
    p75 = float(monthly.quantile(0.75))
    if p25 <= 0:
        raise ValueError("M7计算失败：工资P25<=0，无法计算P75/P25比值。")
    m7 = p75 / p25
    n = int(len(monthly))

    logger.info(
        "M2 mean_wage = %.0f 元/月  M3 std_wage = %.0f  M7 iqr_ratio = %.2f  (n=%d)",
        m2, m3, m7, n,
    )
    common = {"n": n, "confidence": "HIGH"}
    return {
        "mean_wage": {**common, "value": round(m2, 1),
                      "description": "月工资均值（I3a_6年收入/12，IQR2.5倍过滤）"},
        "std_wage": {**common, "value": round(m3, 1),
                     "description": "月工资标准差（元）"},
        "wage_iqr_ratio": {**common, "value": round(m7, 3),
                           "description": "工资P75/P25比值（工资分散度指标）"},
    }


def compute_hours_moments(rf18: pd.DataFrame) -> dict:
    """
    M4/M8：工时相关矩（就业者子样本）

    数据来源：
        I3a_1（一周工作时长，小时/周）
        过滤范围：[1, 112] 小时

    返回:
        dict with m4, m8 的结果
    """
    employed = rf18[rf18["I3_1"] == 1].copy()
    hours = employed["I3a_1"].dropna()
    hours_clean = hours[(hours >= 1) & (hours <= 112)]
    if len(hours_clean) == 0:
        raise ValueError("M4/M8计算失败：工时样本为空（I3a_1过滤后为0）。")

    m4 = float(hours_clean.mean())
    m8 = float(hours_clean.std())
    n = int(len(hours_clean))

    logger.info("M4 mean_hours = %.1f h/w  M8 std_hours = %.1f  (n=%d)", m4, m8, n)
    common = {"n": n, "confidence": "HIGH"}
    return {
        "mean_weekly_hours": {**common, "value": round(m4, 1),
                              "description": "平均周工时（I3a_1，就业农村女性，1-112小时过滤）"},
        "std_weekly_hours": {**common, "value": round(m8, 1),
                             "description": "工时标准差（小时/周）"},
    }


def compute_transition_rates(rf16: pd.DataFrame, rf18: pd.DataFrame) -> dict:
    """
    M5/M6：就业转移率和分离率（面板匹配计算）

    面板匹配机制：
        追访样本的IID编码在2016→2018年间保持不变，
        通过IID_key精确字符串匹配实现跨波追踪。

    就业状态定义：
        2016年：I3a_7_0 == 1（有工作=就业）
        2018年：I3_1 == 1（2017年以来工作过=就业）

    说明：
        2018年的 I3a_7_0 缺失率较高，会导致面板样本过小。
        因此M5/M6采用 I3_1 口径，并在输出中记录该口径差异。

    月度转换公式（两波间隔24个月）：
        f_monthly = 1 - (1 - F_2yr)^(1/24)

    返回:
        dict with m5, m6 的结果
    """
    min_base_n = 10

    # IID精确匹配（保留原始状态变量，不允许将缺失静默映射为0）
    panel = pd.merge(
        rf16[["IID_key", "I3a_7_0"]],
        rf18[["IID_key", "I3_1", "I3a_7_0"]],
        on="IID_key",
        how="inner",
        suffixes=("_16", "_18"),
    )
    panel_n = len(panel)
    logger.info("面板追踪匹配：%d 人（农村女性）", panel_n)

    # 统计口径差异：2018年“当前无工作”但“2017年以来工作过”的人数
    gap_count = int(((panel["I3a_7_0_18"] == 2.0) & (panel["I3_1"] == 1.0)).sum())
    logger.info(
        "口径差异统计：2018当前无工作但2017年以来工作过的人数 = %d",
        gap_count,
    )

    # 仅保留状态变量完整的面板样本，避免缺失值被当作无工作（crash early）
    panel_valid = panel.dropna(subset=["I3a_7_0_16", "I3_1"]).copy()
    panel_valid_n = len(panel_valid)
    logger.info("状态变量完整的面板样本：%d 人", panel_valid_n)

    if panel_valid_n < 30:
        raise ValueError(
            f"M5/M6计算失败：有效面板样本仅 {panel_valid_n} 人，低于最低要求30。"
        )

    # 显式校验编码，若存在非(1,2)编码直接报错
    valid_codes_16 = set(panel_valid["I3a_7_0_16"].unique().tolist())
    valid_codes_18 = set(panel_valid["I3_1"].unique().tolist())
    if not valid_codes_16.issubset({1.0, 2.0}) or not valid_codes_18.issubset({1.0, 2.0}):
        raise ValueError(
            "M5/M6计算失败：就业状态变量存在非{1,2}编码，"
            f"I3a_7_0_16={valid_codes_16}, I3_1={valid_codes_18}"
        )

    panel_valid["emp16"] = panel_valid["I3a_7_0_16"].map({1.0: 1, 2.0: 0})
    panel_valid["emp18"] = panel_valid["I3_1"].map({1.0: 1, 2.0: 0})

    # M5：无工作→就业
    unemp16 = panel_valid[panel_valid["emp16"] == 0]
    if len(unemp16) < min_base_n:
        raise ValueError(
            f"M5计算失败：2016无工作基数仅 {len(unemp16)} 人，"
            f"低于最低要求{min_base_n}。"
        )
    F = float((unemp16["emp18"] == 1).mean())
    m5 = 1 - (1 - F) ** (1 / 24)
    logger.info(
        "M5 job_finding_rate = %.4f/月  (2年F=%.3f，n=%d)", m5, F, len(unemp16)
    )

    # M6：就业→无工作
    emp16 = panel_valid[panel_valid["emp16"] == 1]
    if len(emp16) < min_base_n:
        raise ValueError(
            f"M6计算失败：2016就业基数仅 {len(emp16)} 人，"
            f"低于最低要求{min_base_n}。"
        )
    S = float((emp16["emp18"] == 0).mean())
    m6 = 1 - (1 - S) ** (1 / 24)
    logger.info(
        "M6 separation_rate = %.4f/月  (2年S=%.3f，n=%d)", m6, S, len(emp16)
    )

    return {
        "job_finding_rate": {
            "value": round(float(m5), 4),
            "n": int(len(unemp16)),
            "confidence": "MEDIUM" if len(unemp16) >= 30 else "LOW",
            "description": "月度就业转移率（2016无工作→2018工作经历，24个月折算）",
        },
        "separation_rate": {
            "value": round(float(m6), 4),
            "n": int(len(emp16)),
            "confidence": "HIGH" if len(emp16) >= 100 else "MEDIUM",
            "description": "月度分离率（2016就业→2018无工作经历，24个月折算）",
        },
        "diagnostics": {
            "panel_matched_n": int(panel_n),
            "panel_valid_n": int(panel_valid_n),
            "definition_gap_n": int(gap_count),
        },
    }


def compute_digital_ratio(rf18: pd.DataFrame) -> dict:
    """
    M9：数字技能岗位比例（辅助矩，对应模型D维度）

    定义：
        就业农村女性中，工作中经常或有时使用互联网的比例
        I3a1_19_4 ∈ {1, 2} → 数字技能岗位

    返回:
        dict with m9 的结果
    """
    employed = rf18[rf18["I3_1"] == 1].copy()
    if "I3a1_19_4" not in employed.columns:
        raise KeyError("M9计算失败：2018数据缺少列 I3a1_19_4。")
    digital = employed["I3a1_19_4"].dropna()
    if len(digital) == 0:
        raise ValueError("M9计算失败：I3a1_19_4 全部缺失，无法计算数字岗位占比。")
    ratio = float((digital.isin([1.0, 2.0])).mean())
    n = int(len(digital))
    logger.info("M9 digital_job_ratio = %.3f  (n=%d)", ratio, n)
    return {
        "digital_job_ratio": {
            "value": round(ratio, 4),
            "n": n,
            "confidence": "MEDIUM",
            "description": "工作中经常/有时使用互联网的农村就业女性比例（数字素养D维度代理）",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 汇总与输出
# ─────────────────────────────────────────────────────────────────────────────

def build_yaml_output(m1: dict, wage: dict, hours: dict,
                      trans: dict, digital: dict, rf18: pd.DataFrame,
                      panel_n: int) -> dict:
    """
    汇总所有目标矩，构造YAML输出字典。

    参数:
        m1:      M1失业率结果
        wage:    M2/M3/M7工资矩结果
        hours:   M4/M8工时矩结果
        trans:   M5/M6转移率结果
        digital: M9数字素养结果
        rf18:    2018年农村女性子样本（用于统计元信息）
        panel_n: 面板匹配追踪人数

    返回:
        YAML格式的dict
    """

    # 转移率与截面口径诊断（仅用于解释，不作通过/失败判定）
    m5_val = trans["job_finding_rate"]["value"]
    m6_val = trans["separation_rate"]["value"]
    steady_state = m6_val / (m6_val + m5_val) if (m6_val + m5_val) > 0 else np.nan
    nonwork_rate_broad = float((rf18["I3_1"] == 2).sum() / len(rf18))

    return {
        "meta": {
            "description": "CLDS实算目标矩（农村女性，15-64岁）",
            "data_source": "CLDS 2018年个体问卷（主）+ CLDS 2016年（面板追踪）",
            "filter": "性别=女, 户口I1_3_2∈{2,4}（农业/失地农业）, 年龄15-64岁",
            "sample_size_2018": int(len(rf18)),
            "sample_employed_2018": int(m1["employed_n"]),
            "panel_tracked": panel_n,
            "M1_method": "ILO三分法: I3_1=1就业; I3_1=2且I3c_5=1失业; 其余=非劳动力",
            "M5M6_method": "IID精确匹配2016-2018，I3a_7_0(2016年当前状态)→I3_1(2018年工作经历)，24个月月度折算",
            "wage_method": "I3a_6（2017年总收入）/12，IQR2.5倍法过滤异常值",
            "transition_diagnostics": {
                **trans.get("diagnostics", {}),
                "steady_state_unemp": round(steady_state, 4),
                "cross_section_nonwork_rate": round(nonwork_rate_broad, 4),
                "note": "两者口径不同（转移率基于面板、截面基于2017年以来是否工作过），仅供诊断参考",
            },
        },
        "moments": {
            "unemployment_rate":   {"tag": "M1", **{k: v for k, v in m1.items() if k not in ["employed_n", "unemployed_n", "note"]}},
            "mean_wage":           {"tag": "M2", **wage["mean_wage"]},
            "std_wage":            {"tag": "M3", **wage["std_wage"]},
            "mean_weekly_hours":   {"tag": "M4", **hours["mean_weekly_hours"]},
            "job_finding_rate":    {"tag": "M5", **trans["job_finding_rate"]},
            "separation_rate":     {"tag": "M6", **trans["separation_rate"]},
            "wage_iqr_ratio":      {"tag": "M7", **wage["wage_iqr_ratio"]},
            "std_weekly_hours":    {"tag": "M8", **hours["std_weekly_hours"]},
            "digital_job_ratio":   {"tag": "M9", **digital["digital_job_ratio"]},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(output_path: str = DEFAULT_OUTPUT) -> None:
    """
    主计算流程。

    步骤：
        1. 加载2018年数据，筛选农村女性子样本
        2. 计算M1（失业率）
        3. 计算M2/M3/M7（工资矩）
        4. 计算M4/M8（工时矩）
        5. 加载2016年数据，筛选农村女性子样本
        6. 面板匹配，计算M5/M6（转移率）
        7. 计算M9（数字素养）
        8. 汇总、内部一致性验证、输出YAML

    参数:
        output_path: 输出YAML文件路径
    """
    logger.info("=" * 60)
    logger.info("CLDS目标矩计算开始")
    logger.info("=" * 60)

    # ── 步骤1：加载2018年数据 ──────────────────────────────────────────────
    df18 = load_2018(DTA_2018)
    rf18 = filter_rural_female(df18, gender_col="Igender")
    logger.info("2018年农村女性子样本总量: %d 人", len(rf18))

    # ── 步骤2-4：截面矩 ───────────────────────────────────────────────────
    logger.info("── 计算M1 失业率 ──")
    m1_result = compute_m1_unemployment(rf18)

    logger.info("── 计算M2/M3/M7 工资矩 ──")
    wage_result = compute_wage_moments(rf18)

    logger.info("── 计算M4/M8 工时矩 ──")
    hours_result = compute_hours_moments(rf18)

    # ── 步骤5-6：加载2016年数据，面板匹配 ──────────────────────────────────
    logger.info("── 加载2016年数据（面板追踪）──")
    df16 = load_2016(DTA_2016)
    rf16 = filter_rural_female(df16, gender_col="gender")

    logger.info("── 计算M5/M6 转移率（面板匹配）──")
    trans_result = compute_transition_rates(rf16, rf18)

    # 记录面板匹配人数（用于元信息）
    panel_matched = pd.merge(
        rf16[["IID_key"]], rf18[["IID_key"]], on="IID_key", how="inner"
    )
    panel_n = len(panel_matched)

    # ── 步骤7：数字素养 ───────────────────────────────────────────────────
    logger.info("── 计算M9 数字素养 ──")
    digital_result = compute_digital_ratio(rf18)

    # ── 步骤8：内部一致性验证 + 输出 ─────────────────────────────────────
    yaml_data = build_yaml_output(
        m1_result, wage_result, hours_result,
        trans_result, digital_result, rf18, panel_n,
    )

    # 打印转移率与截面口径诊断信息
    diag = yaml_data["meta"]["transition_diagnostics"]
    logger.info(
        "转移率诊断：s/(s+f)=%.3f，截面非工作率=%.3f，定义差异人数=%s",
        diag["steady_state_unemp"],
        diag["cross_section_nonwork_rate"],
        diag.get("definition_gap_n", "NA"),
    )

    # 输出汇总表格
    logger.info("=" * 60)
    logger.info("【最终结果】CLDS实算目标矩")
    logger.info("=" * 60)
    for key, mmt in yaml_data["moments"].items():
        logger.info("  %s %-20s = %-12.4f  (n=%s)", mmt["tag"], key, mmt["value"], mmt.get("n", "?"))

    # 写入YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info("✅ 目标矩已写入: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="计算CLDS农村女性劳动力市场目标矩（M1-M9），输出YAML配置文件"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"输出YAML路径（默认：{DEFAULT_OUTPUT}）",
    )
    args = parser.parse_args()

    # 检查数据文件是否存在
    for fpath, name in [(DTA_2018, "2018年数据"), (DTA_2016, "2016年数据")]:
        if not os.path.exists(fpath):
            logger.error("找不到%s: %s", name, fpath)
            sys.exit(1)

    main(output_path=args.output)
