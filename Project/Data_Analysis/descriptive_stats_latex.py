import pandas as pd
import numpy as np


# 读取数据
df = pd.read_csv('cleaned_data.csv')

# 移除空行
df = df.dropna(how='all')

# 定义连续变量和分类变量，以及它们的赋分范围
continuous_vars = [
    ('年龄', '实际年龄'),
    ('孩子数量', '0-3'),
    ('家务劳动时间', '小时/天'),
    ('闲暇时间', '小时/天'),
    ('每周期望工作天数', '天/周'),
    ('每天期望工作时数', '小时/天'),
    ('每月期望收入', '元/月'),
    ('工作保险重要性', '0-3'),
    ('劳动合同重要性', '0-3'),
    ('学历', '0-6'),
    ('累计工作年限', '年'),
    ('工作能力评分', '分值'),
    ('数字素养评分', '分值')
]

categorical_vars = [
    ('常住地', '分类'),
    ('工作意愿', '分类'),
    ('全职或兼职', '分类'),
    ('线上或线下', '分类'),
    ('是否曾工作', '0-1')
]

# 计算连续变量的描述性统计
print("=" * 80)
print("连续变量描述性统计")
print("=" * 80)

stats_list = []
for var, range_info in continuous_vars:
    if var in df.columns:
        data = df[var].dropna()
        stats = {
            '变量': f"{var}（{range_info}）",
            '样本量': len(data),
            '均值': data.mean(),
            '标准差': data.std(),
            '最小值': data.min(),
            '25分位数': data.quantile(0.25),
            '中位数': data.median(),
            '75分位数': data.quantile(0.75),
            '最大值': data.max()
        }
        stats_list.append(stats)

stats_df = pd.DataFrame(stats_list)

# 生成LaTeX表格 - 连续变量（使用booktabs三线表）
latex_continuous = r"""\begin{table}[htbp]
\centering
\caption{连续变量描述性统计}
\label{tab:descriptive_continuous}
\begin{tabular}{lcccccccc}
\toprule
变量 & 样本量 & 均值 & 标准差 & 最小值 & 25\% & 中位数 & 75\% & 最大值 \\
\midrule
"""

for _, row in stats_df.iterrows():
    latex_continuous += f"{row['变量']} & {row['样本量']:.0f} & {row['均值']:.2f} & {row['标准差']:.2f} & {row['最小值']:.2f} & {row['25分位数']:.2f} & {row['中位数']:.2f} & {row['75分位数']:.2f} & {row['最大值']:.2f} \\\\\n"

latex_continuous += r"""\bottomrule
\end{tabular}
\end{table}
"""

# 计算分类变量的频数统计
print("\n" + "=" * 80)
print("分类变量频数统计")
print("=" * 80)

# 创建合并的分类变量表格
latex_categorical = r"""\begin{table}[htbp]
\centering
\caption{分类变量频数统计}
\label{tab:descriptive_categorical}
\begin{tabular}{llcc}
\toprule
变量 & 类别 & 频数 & 百分比(\%) \\
\midrule
"""

for var, range_info in categorical_vars:
    if var in df.columns:
        print(f"\n{var}:")
        freq = df[var].value_counts().sort_index()
        print(freq)
        
        total = freq.sum()
        var_display = f"{var}（{range_info}）"
        
        # 第一行包含变量名
        first_row = True
        for idx, value in freq.items():
            percentage = (value / total) * 100
            if first_row:
                latex_categorical += f"{var_display} & {idx} & {value} & {percentage:.2f} \\\\\n"
                first_row = False
            else:
                latex_categorical += f" & {idx} & {value} & {percentage:.2f} \\\\\n"
        
        # 添加小计行
        latex_categorical += r"\cmidrule(lr){2-4}" + "\n"
        latex_categorical += f" & 小计 & {total} & 100.00 \\\\\n"
        latex_categorical += r"\midrule" + "\n"

# 移除最后一个\midrule，替换为\bottomrule
latex_categorical = latex_categorical.rstrip("\n").rstrip(r"\midrule") + "\n"
latex_categorical += r"""\bottomrule
\end{tabular}
\end{table}
"""

# 保存表格代码片段（用于插入到现有文档）
with open('descriptive_stats_tables.tex', 'w', encoding='utf-8') as f:
    f.write("% 数据描述性统计分析表格代码片段\n")
    f.write("% 生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
    f.write(latex_continuous)
    f.write("\n\n")
    f.write(latex_categorical)

# 生成完整的可编译LaTeX文档
latex_complete = r"""\documentclass[12pt,a4paper]{article}
\usepackage{ctex}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}

% 字体设置 - 使用更美观的字体
\setCJKmainfont{SimSun}[BoldFont=SimHei,ItalicFont=KaiTi]
\setCJKsansfont{Microsoft YaHei}
\setCJKmonofont{FangSong}

% 页面设置
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

% 表格行距
\renewcommand{\arraystretch}{1.2}

\title{\textbf{数据描述性统计分析报告}}
\author{}
\date{""" + pd.Timestamp.now().strftime('%Y年%m月%d日') + r"""}

\begin{document}

\maketitle

\section{概述}

本报告对cleaned\_data.csv数据集进行了全面的描述性统计分析。数据集共包含300个样本，涵盖了被调查者的基本信息、工作意愿、工作能力等多个维度的数据。

\section{连续变量描述性统计}

表\ref{tab:descriptive_continuous}展示了13个连续变量的描述性统计结果，包括样本量、均值、标准差、最小值、四分位数和最大值。变量名后括号中标注了各变量的赋分范围或单位。

"""

latex_complete += latex_continuous

latex_complete += r"""

\section{分类变量频数统计}

表\ref{tab:descriptive_categorical}展示了5个分类变量的频数分布和百分比统计。变量名后括号中标注了该变量的类型或赋分范围。

"""

latex_complete += latex_categorical

latex_complete += r"""

\section{主要发现}

\begin{itemize}
    \item 样本平均年龄为36.98岁，标准差为6.07岁，年龄分布相对集中。
    \item 被调查者平均有1.57个孩子，每天平均花费6.83小时进行家务劳动。
    \item 76.67\%的被调查者希望工作，其中65\%倾向于全职工作。
    \item 73.67\%的被调查者偏好线下工作，仅26.33\%偏好线上工作。
    \item 每月期望收入平均为4520.77元，范围从1400元到8000元。
    \item 92.33\%的被调查者有工作经历，平均累计工作年限为7.97年。
    \item 工作能力平均评分为25.02分，数字素养平均评分为8.62分。
    \item 工作保险重要性和劳动合同重要性平均评分分别为1.95和2.16（0-3分制）。
    \item 学历平均为3.53（0-6分制），相当于高中至大专水平。
\end{itemize}

\end{document}
"""

with open('descriptive_stats.tex', 'w', encoding='utf-8') as f:
    f.write(latex_complete)

# 打印到控制台
print("\n" + "=" * 80)
print("LaTeX文档已生成")
print("=" * 80)

# 同时保存一个可读的统计摘要
with open('descriptive_stats_summary.txt', 'w', encoding='utf-8') as f:
    f.write("数据描述性统计分析摘要\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"总样本量: {len(df)}\n\n")
    
    f.write("连续变量统计:\n")
    f.write("-" * 80 + "\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("分类变量统计:\n")
    f.write("-" * 80 + "\n")
    for var, range_info in categorical_vars:
        if var in df.columns:
            f.write(f"\n{var}（{range_info}）:\n")
            freq = df[var].value_counts().sort_index()
            total = freq.sum()
            for idx, value in freq.items():
                percentage = (value / total) * 100
                f.write(f"  {idx}: {value} ({percentage:.2f}%)\n")

print("\n描述性统计分析完成！")
print("生成文件:")
print("  - descriptive_stats.tex (完整的可编译LaTeX文档)")
print("  - descriptive_stats_tables.tex (表格代码片段，可插入现有文档)")
print("  - descriptive_stats_summary.txt (文本摘要)")
print("\n编译说明:")
print("  使用 xelatex 或 pdflatex 编译 descriptive_stats.tex:")
print("  xelatex descriptive_stats.tex")

