import jieba
import collections
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = r"D:\Document\WeChat Files\wxid_pjmx8c4tmzod12\FileStorage\File\2025-03\促进农村妇女创业就业.txt"

# 读取文件内容，若编码不是utf-8，可修改为对应的编码格式
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

words = jieba.lcut(text)

# 根据需要过滤掉一些无效词，例如只保留长度大于1的词
words = [word for word in words if len(word) > 1]

# 统计词频
word_counts = collections.Counter(words)

# 将统计结果转换为DataFrame，并按照频数从高到低排序
df = pd.DataFrame(word_counts.items(), columns=['词语', '频数'])
df = df.sort_values(by='频数', ascending=False)

print("文本分析结果表格（前20个高频词）：")
print(df.head(20))
output_file_path = r"D:\Document\WeChat Files\wxid_pjmx8c4tmzod12\FileStorage\File\2025-03\词频统计结果.xlsx"
df.to_excel(output_file_path, index=False)

# 可视化：绘制高频词条形图（取前20个词）
top_n = 20
df_top = df.head(top_n)

plt.figure(figsize=(10, 6))
plt.bar(df_top['词语'], df_top['频数'])
plt.xticks(rotation=45, ha='right')
plt.xlabel("词语")
plt.text(1.0, -0.15, '数据来源：中华全国妇女联合会', verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=10, color='gray')
plt.ylabel("频数")
plt.bar(df_top['词语'], df_top['频数'], color='skyblue')
plt.title("文本分析 - 高频词统计")
plt.tight_layout()
bar_chart_path = r"D:\Document\WeChat Files\wxid_pjmx8c4tmzod12\FileStorage\File\2025-03\高频词条形图.png"
plt.savefig(bar_chart_path, dpi=300)
plt.show()

# 从CSV文件读取词频数据
csv_file_path = r"D:\Document\WeChat Files\wxid_pjmx8c4tmzod12\FileStorage\File\2025-03\词频统计结果(1)(1).csv"
df_csv = pd.read_csv(csv_file_path, encoding='gbk')

# 将CSV数据转换为词频字典
word_counts_csv = dict(zip(df_csv['词语'], df_csv['频数']))

# 生成词云
wordcloud_csv = WordCloud(font_path='simhei.ttf', width=800, height=400, background_color='white').generate_from_frequencies(word_counts_csv)

# 显示词云图
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_csv, interpolation='bilinear')
plt.axis('off')
plt.title("文本分析 - 词云图")
plt.text(1.0, -0.06, '数据来源：中华全国妇女联合会', verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=10, color='gray')
wordcloud_csv_path = r"D:\Document\WeChat Files\wxid_pjmx8c4tmzod12\FileStorage\File\2025-03\词云图_from_csv.png"
plt.savefig(wordcloud_csv_path, dpi=300)
plt.show()
