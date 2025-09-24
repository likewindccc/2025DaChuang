import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
plt.rcParams['axes.titlesize'] = 14      # 图标题字体大小
plt.rcParams['axes.labelsize'] = 12      # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12     # x轴刻度标签
plt.rcParams['ytick.labelsize'] = 12     # y轴刻度标签

df = pd.read_excel(r"C:\Users\21515\Desktop\杂物\Library\2025 大创\问卷数据清洗版3.22.xlsx")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
df.rename(columns={ 
    "3您的年龄是": "年龄", 
    "5您有多少个孩子": "孩子数量", 
    "6您一般每天会在家务劳动如买菜做饭做卫生照顾孩子等上花费多少个小时": "家务时长", 
    "7除了睡觉时间外您每天的闲暇时间即没有事情做的时间有多少个小时": "闲暇时间", 
    "期望周工作时数": "期望每周工作时数", 
    "15您最希望这份工作每个月的收入有多少元": "期望月收入", 
    "16.您觉得这份工作中的社会保险（如基本养老保险、基本医疗保险、失业保险、工伤保险和生育保险）重要吗？":"保险重要性",
    "17.您觉得这份工作中签订劳动合同重要吗？":"劳动合同重要性",
    "18您的学历是": "学历",
    "20您的累计工作年限即参加工作的年数一共多少年可具体到半年即05不计算中途失业待业的情况": "累计工作年限", 
    "八项工作能力总评":"工作技能总评",
    "数字素养总评分":"数字素养总评"
}, inplace=True)

category_dict = {
    "期望工作时间": [
        "期望每周工作时数" 
    ],
    "期望工作待遇": [
        "期望月收入",
        "保险重要性",
        "劳动合同重要性"
    ],
    "工作能力": [
        "工作技能总评"  
    ],
    "数字素养": [
        "数字素养总评"  
    ],
    "控制变量": [
        "年龄",
        "孩子数量",
        "学历", 
        "累计工作年限"
    ]
}

# ① 对每个大类别，直接使用原始变量值求和，构造聚合变量（命名为 "_原始和"）
for cat, subvars in category_dict.items():
    # 筛选存在的变量
    valid_subvars = [var for var in subvars if var in df.columns]
    if not valid_subvars:
        continue
    sum_col = cat + "聚合"
    df[sum_col] = df[valid_subvars].sum(axis=1)

############################################
# 2. 未归一化变量的可视化：原始子变量 & 聚合变量
############################################

# (a) 针对所有原始子变量绘制箱线图和直方图 + KDE
for cat, subvars in category_dict.items():
    for var in subvars:
        if var not in df.columns:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        sns.boxplot(x=df[var], ax=axes[0], color='lightgreen')
        sns.histplot(df[var], kde=True, ax=axes[1])
        
        axes[0].set_title(f'{var} - 箱线图')
        axes[1].set_title(f'{var} - 直方图及KDE')
        axes[1].set_ylabel('频数')
        
        plt.tight_layout()
        plt.savefig(f'images/{var}_箱线图_直方图.png', dpi=300)
        plt.close()

# (b) 针对各大类别聚合变量（"_原始和"）绘制箱线图和直方图 + KDE
for cat in category_dict.keys():
    sum_col = cat + "聚合"
    if sum_col not in df.columns:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    sns.boxplot(x=df[sum_col], ax=axes[0], color='lightgreen')
    sns.histplot(df[sum_col], kde=True, ax=axes[1])
    
    axes[0].set_title(f'{cat} - 箱线图')
    axes[1].set_title(f'{cat} - 直方图及KDE')
    axes[1].set_ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(f'images/{cat}_箱线图_直方图.png', dpi=300)
    plt.close()

############################################
# 3. 相关性热力图 & 描述性统计结果输出
############################################

# (a) 针对各大类别聚合变量的相关性热力图
agg_cols = [cat + "聚合" for cat in category_dict.keys() if cat + "聚合" in df.columns]
if agg_cols:
    corr_matrix_agg = df[agg_cols].corr()
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(corr_matrix_agg, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title("聚合变量相关性热力图")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("images/聚合变量相关性热力图.png", dpi=300)
    plt.close()

# (b) 针对所有涉及的原始子变量的相关性热力图
numeric_original = []
for subvars in category_dict.values():
    for var in subvars:
        if var in df.columns:
            numeric_original.append(var)
numeric_original = list(set(numeric_original))
if numeric_original:
    corr_matrix_orig = df[numeric_original].corr()
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(corr_matrix_orig, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("子变量相关性热力图")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("images/子变量相关性热力图.png", dpi=300)
    plt.close()

# (c) 输出聚合变量的描述性统计结果为 Excel 表格
stats_unscaled = df[agg_cols].describe().T
stats_unscaled['variance'] = df[agg_cols].var()
stats_unscaled.to_excel("未归一化聚合变量描述性统计结果.xlsx")

# (d) 输出所有子变量的描述性统计结果为 Excel 表格
stats_original = df[numeric_original].describe().T
stats_original['variance'] = df[numeric_original].var()
stats_original.to_excel("未归一化子变量描述性统计结果.xlsx")

for cat, subvars in category_dict.items():
    for var in subvars:
        # 跳过可能的缺失或非数值列
        if var not in df.columns:
            continue
        
        col_min = df[var].min()
        col_max = df[var].max()
        # 避免除以0
        if col_max == col_min:
            df[var + "_归一化变量"] = 0.0
        else:
            df[var + "_归一化变量"] = (df[var] - col_min) / (col_max - col_min)

for cat, subvars in category_dict.items():
    # 收集本类别下所有子变量的 _归一化变量 列
    norm_cols = [var + "_归一化变量" for var in subvars if var + "_归一化变量" in df.columns]
    
    # 若 norm_cols 为空，跳过
    if not norm_cols:
        continue
    
    # 1) 求和
    sum_col = cat + "_sum"
    df[sum_col] = df[norm_cols].sum(axis=1)
    
    # 2) 再做一次 Min-Max
    cat_min = df[sum_col].min()
    cat_max = df[sum_col].max()
    norm_col = cat + "_归一化变量"
    if cat_min == cat_max:
        df[norm_col] = 0.0
    else:
        df[norm_col] = (df[sum_col] - cat_min) / (cat_max - cat_min)

# 针对所有子变量的 _归一化变量 列，做箱线图+直方图
for cat, subvars in category_dict.items():
    for var in subvars:
        norm_var = var + "_归一化变量"
        if norm_var not in df.columns:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        axes[1].set_ylabel('频数')
        
        sns.boxplot(x=df[norm_var], ax=axes[0],color='lightgreen')
        sns.histplot(df[norm_var], kde=True, ax=axes[1])
        
        axes[0].set_title(f'{norm_var} 箱线图')
        axes[1].set_title(f'{norm_var} 直方图及KDE')
        
        plt.tight_layout()
        plt.savefig(f'images/{norm_var}_箱线图_直方图.png', dpi=300)
        plt.close()

for cat in category_dict.keys():
    norm_col = cat + "_归一化变量"
    if norm_col not in df.columns:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    axes[1].set_ylabel('频数')
    
    sns.boxplot(x=df[norm_col], ax=axes[0], color='lightgreen')
    sns.histplot(df[norm_col], kde=True, ax=axes[1])
    
    axes[0].set_title(f'{cat} 箱线图')
    axes[1].set_title(f'{cat} 直方图及KDE')
    
    plt.tight_layout()
    plt.savefig(f'images/{norm_col}_箱线图_直方图.png', dpi=300)
    plt.close()

categorical_cols = {
    '4您的常住地在': '常住地',
    '8您现在希望拥有工作吗': '是否希望工作',
    '11您希望它是全职还是兼职工作': '工作性质',
    '12您希望它是线上的还是线下的': '工作形式',
    '19您以前有参加过工作吗': '是否有工作经历',
}

# 重命名（若尚未完成）
for col, new_name in categorical_cols.items():
    if col in df.columns:
        df.rename(columns={col: new_name}, inplace=True)

# 继续你的分类变量可视化
for col in categorical_cols.values():
    if col not in df.columns:
        continue
    
    plt.figure(figsize=(8,5))
    ax = sns.countplot(x=df[col], order=df[col].value_counts().index, color='skyblue')
    total = len(df)
    
    # 在柱上标注百分比
    for p in ax.patches:
        height = p.get_height()
        pct = height / total * 100
        plt.text(p.get_x() + p.get_width() / 2, height + 0.01,
                 f'{pct:.2f}%', ha="center", fontsize=12)
    
    plt.title(f'{col}分布情况', fontsize=14)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'images/{col}_分布情况.png', dpi=300)
    plt.close()
