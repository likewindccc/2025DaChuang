"""
问卷数据可视化器

提供问卷数据的描述性统计和探索性可视化功能
包括相关性热力图、分布图、关系图等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from matplotlib import font_manager

# 设置中文字体 - 优先使用微软雅黑
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SurveyDataVisualizer:
    """问卷数据可视化类"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化
        
        参数:
            output_dir: 输出目录，默认为OUTPUT目录
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'OUTPUT'
        
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'survey_analysis'
        
        # 创建输出目录
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载数据
        
        参数:
            data_path: 数据文件路径
        
        返回:
            数据框
        """
        df = pd.read_csv(data_path, encoding='utf-8')
        df = df.dropna(how='all')
        print(f"✓ 数据加载成功，样本量: {len(df)}")
        return df
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        continuous_vars: Optional[List[str]] = None,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (14, 12),
        save_name: str = 'correlation_heatmap'
    ) -> str:
        """
        绘制连续变量相关性热力图
        
        参数:
            df: 数据框
            continuous_vars: 连续变量列表，如果为None则自动选择数值型列
            method: 相关系数方法 ('pearson', 'spearman', 'kendall')
            figsize: 图形大小
            save_name: 保存文件名
        
        返回:
            保存路径
        """
        # 自动选择连续变量
        if continuous_vars is None:
            continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除编号列
            continuous_vars = [col for col in continuous_vars 
                             if col not in ['编号', 'ID', 'id']]
        
        # 选择数据
        data = df[continuous_vars].copy()
        
        # 计算相关系数矩阵
        corr_matrix = data.corr(method=method)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图 - 只显示左下角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0,
            linecolor='none',
            cbar_kws={
                "shrink": 0.8,
                "label": "相关系数"
            },
            annot=True,
            fmt='.2f',
            annot_kws={'size': 12, 'weight': 'bold'},
            ax=ax
        )
        
        # 关闭网格
        ax.grid(False)
        ax.set_axisbelow(False)
        
        # 旋转标签并设置字体
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', 
                          fontproperties='Microsoft YaHei')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, 
                          fontproperties='Microsoft YaHei')
        
        # 设置colorbar标签字体
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('相关系数', fontname='Microsoft YaHei')
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Microsoft YaHei')
        
        plt.tight_layout()
        
        # 保存图形
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"✓ 相关性热力图已保存: {save_path}")
        
        plt.close()
        
        return str(save_path)
    
    def plot_correlation_heatmap_annotated(
        self,
        df: pd.DataFrame,
        var_dict: Optional[dict] = None,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (16, 14),
        save_name: str = 'correlation_heatmap_annotated'
    ) -> str:
        """
        绘制带变量说明的相关性热力图
        
        参数:
            df: 数据框
            var_dict: 变量名称映射字典 {原列名: 显示名称}
            method: 相关系数方法
            figsize: 图形大小
            save_name: 保存文件名
        
        返回:
            保存路径
        """
        # 如果没有提供变量字典，使用默认映射
        if var_dict is None:
            var_dict = {
                '年龄': '年龄',
                '孩子数量': '孩子数量',
                '家务劳动时间': '家务时间',
                '闲暇时间': '闲暇时间',
                '每周期望工作天数': '期望工作天数',
                '每天期望工作时数': '期望工作时数',
                '每月期望收入': '期望收入',
                '工作保险重要性': '保险重要性',
                '劳动合同重要性': '合同重要性',
                '学历': '学历',
                '累计工作年限': '工作年限',
                '工作能力评分': '工作能力',
                '数字素养评分': '数字素养'
            }
        
        # 选择存在的变量
        available_vars = [col for col in var_dict.keys() if col in df.columns]
        data = df[available_vars].copy()
        
        # 重命名列
        display_names = [var_dict[col] for col in available_vars]
        data.columns = display_names
        
        # 计算相关系数
        corr_matrix = data.corr(method=method)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 生成颜色映射
        cmap = sns.diverging_palette(240, 10, n=256, as_cmap=True)
        
        # 绘制左下角热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0,
            linecolor='none',
            cbar_kws={
                "shrink": 0.75,
                "label": "相关系数",
                "pad": 0.02
            },
            annot=True,
            fmt='.2f',
            annot_kws={'size': 13, 'weight': 'bold'},
            ax=ax
        )
        
        # 关闭网格
        ax.grid(False)
        ax.set_axisbelow(False)
        
        # 旋转标签并设置字体
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11,
                          fontproperties='Microsoft YaHei')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11,
                          fontproperties='Microsoft YaHei')
        
        # 设置colorbar标签字体
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('相关系数', fontname='Microsoft YaHei', fontsize=11)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Microsoft YaHei')
        
        plt.tight_layout()
        
        # 保存图形
        save_path = self.figures_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ 标注版相关性热力图已保存: {save_path}")
        
        plt.close()
        
        return str(save_path)
    
    def generate_correlation_report(
        self,
        df: pd.DataFrame,
        continuous_vars: Optional[List[str]] = None,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        生成相关性分析报告
        
        参数:
            df: 数据框
            continuous_vars: 连续变量列表
            threshold: 显著相关性阈值
        
        返回:
            相关性报告数据框
        """
        # 自动选择连续变量
        if continuous_vars is None:
            continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
            continuous_vars = [col for col in continuous_vars 
                             if col not in ['编号', 'ID', 'id']]
        
        data = df[continuous_vars].copy()
        corr_matrix = data.corr(method='pearson')
        
        # 提取强相关关系
        strong_corr = []
        n = len(continuous_vars)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corr.append({
                        '变量1': continuous_vars[i],
                        '变量2': continuous_vars[j],
                        '相关系数': corr_val,
                        '强度': '强' if abs(corr_val) >= 0.7 else '中' if abs(corr_val) >= 0.5 else '弱',
                        '方向': '正相关' if corr_val > 0 else '负相关'
                    })
        
        report_df = pd.DataFrame(strong_corr)
        if not report_df.empty:
            report_df = report_df.sort_values('相关系数', key=abs, ascending=False)
        
        # 保存报告
        report_path = self.figures_dir / 'correlation_report.csv'
        report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"✓ 相关性报告已保存: {report_path}")
        
        return report_df


def main():
    """主函数 - 示例用法"""
    # 初始化可视化器
    visualizer = SurveyDataVisualizer()
    
    # 数据路径
    data_path = Path(__file__).parent.parent.parent.parent / 'Project' / 'Data_Analysis' / 'cleaned_data.csv'
    
    # 加载数据
    df = visualizer.load_data(str(data_path))
    
    # 绘制Pearson相关性热力图
    print("\n生成Pearson相关性热力图...")
    visualizer.plot_correlation_heatmap(
        df,
        method='pearson',
        save_name='survey_correlation_heatmap_pearson'
    )
    
    # 生成相关性报告
    print("\n生成相关性分析报告...")
    report = visualizer.generate_correlation_report(df, threshold=0.3)
    
    if not report.empty:
        print("\n显著相关关系Top 10:")
        print(report.head(10).to_string(index=False))
    else:
        print("\n未发现显著相关关系（阈值0.3）")
    
    print("\n✓ 可视化分析完成！")


if __name__ == '__main__':
    main()

