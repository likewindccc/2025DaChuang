"""
Population Generator Utilities

提供数据处理、验证、可视化等通用工具函数。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
from scipy import stats
import psutil
import time
from functools import wraps

# 配置日志
logger = logging.getLogger(__name__)

# 配置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def timer(func):
    """装饰器：记录函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 执行时间: {execution_time:.4f}秒")
        
        return result
    return wrapper


def memory_monitor(func):
    """装饰器：监控函数内存使用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        logger.info(f"{func.__name__} 内存使用: {memory_delta:.2f}MB "
                   f"(总计: {memory_after:.2f}MB)")
        
        return result
    return wrapper


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_data_quality(data: pd.DataFrame, 
                            required_columns: List[str],
                            data_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        全面的数据质量检查
        
        Args:
            data: 待验证数据
            required_columns: 必需列名
            data_bounds: 数据边界
            
        Returns:
            验证报告字典
        """
        report = {
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        # 1. 基本结构检查
        if data.empty:
            report['is_valid'] = False
            report['issues'].append("数据为空")
            return report
        
        # 2. 列完整性检查
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            report['is_valid'] = False
            report['issues'].append(f"缺少必需列: {missing_columns}")
        
        # 3. 数据类型检查
        for col in required_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    report['is_valid'] = False
                    report['issues'].append(f"列 '{col}' 不是数值类型")
        
        # 4. 缺失值检查
        missing_counts = data[required_columns].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_ratio = total_missing / (len(data) * len(required_columns))
            if missing_ratio > 0.05:  # 超过5%缺失
                report['is_valid'] = False
                report['issues'].append(f"缺失值过多: {missing_ratio:.2%}")
            else:
                report['recommendations'].append("建议处理少量缺失值")
        
        # 5. 数据边界检查
        for col, (min_val, max_val) in data_bounds.items():
            if col in data.columns:
                col_data = data[col].dropna()
                out_of_bounds = ((col_data < min_val) | (col_data > max_val)).sum()
                
                if out_of_bounds > 0:
                    ratio = out_of_bounds / len(col_data)
                    if ratio > 0.01:  # 超过1%超界
                        report['is_valid'] = False
                        report['issues'].append(
                            f"列 '{col}' 有 {ratio:.2%} 数据超出边界 [{min_val}, {max_val}]"
                        )
        
        # 6. 分布异常检查
        for col in required_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                
                if len(col_data) > 10:
                    # 异常值检测（IQR方法）
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outliers = ((col_data < Q1 - 1.5 * IQR) | 
                               (col_data > Q3 + 1.5 * IQR)).sum()
                    
                    outlier_ratio = outliers / len(col_data)
                    if outlier_ratio > 0.1:  # 超过10%异常值
                        report['recommendations'].append(
                            f"列 '{col}' 异常值较多: {outlier_ratio:.2%}"
                        )
        
        # 7. 统计摘要
        report['statistics'] = {
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'missing_ratio': total_missing / (len(data) * len(required_columns)),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return report
    
    @staticmethod
    def suggest_data_cleaning(data: pd.DataFrame, 
                            validation_report: Dict[str, Any]) -> List[str]:
        """根据验证报告建议数据清洗方案"""
        suggestions = []
        
        if not validation_report['is_valid']:
            for issue in validation_report['issues']:
                if "缺失值" in issue:
                    suggestions.append("使用插值法或删除含缺失值的行")
                elif "超出边界" in issue:
                    suggestions.append("检查数据录入错误或调整边界设置")
                elif "数据类型" in issue:
                    suggestions.append("转换数据类型或检查数据格式")
        
        # 添加一般性建议
        if validation_report['statistics']['missing_ratio'] > 0:
            suggestions.append("考虑使用插值法填补缺失值")
        
        if validation_report['statistics']['n_rows'] < 1000:
            suggestions.append("数据量较小，建议增加样本以提高模型稳定性")
        
        return suggestions


class DistributionAnalyzer:
    """分布分析器"""
    
    @staticmethod
    def analyze_marginal_distributions(data: pd.DataFrame, 
                                     columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        分析边际分布
        
        Args:
            data: 数据
            columns: 要分析的列名
            
        Returns:
            每列的分布分析结果
        """
        results = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            col_data = data[col].dropna()
            
            if len(col_data) < 10:
                results[col] = {'error': '数据量不足进行分布分析'}
                continue
            
            # 基本统计量
            stats_dict = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data)
            }
            
            # 分布拟合测试
            distributions = ['norm', 'gamma', 'beta', 'lognorm', 'uniform']
            best_dist = None
            best_p_value = 0
            
            for dist_name in distributions:
                try:
                    if dist_name == 'beta':
                        # Beta分布需要数据在[0,1]区间
                        if col_data.min() >= 0 and col_data.max() <= 1:
                            dist = getattr(stats, dist_name)
                            params = dist.fit(col_data)
                            _, p_value = stats.kstest(col_data, lambda x: dist.cdf(x, *params))
                        else:
                            continue
                    else:
                        dist = getattr(stats, dist_name)
                        params = dist.fit(col_data)
                        _, p_value = stats.kstest(col_data, lambda x: dist.cdf(x, *params))
                    
                    if p_value > best_p_value:
                        best_p_value = p_value
                        best_dist = {
                            'name': dist_name,
                            'params': params,
                            'p_value': p_value
                        }
                
                except Exception as e:
                    logger.warning(f"拟合 {dist_name} 分布失败: {e}")
            
            stats_dict['best_distribution'] = best_dist
            results[col] = stats_dict
        
        return results
    
    @staticmethod
    def analyze_correlations(data: pd.DataFrame, 
                           columns: List[str]) -> Dict[str, Any]:
        """分析变量间相关性"""
        if len(columns) < 2:
            return {'error': '至少需要2个变量进行相关性分析'}
        
        subset_data = data[columns].dropna()
        
        if len(subset_data) < 10:
            return {'error': '有效数据量不足'}
        
        # Pearson相关系数
        pearson_corr = subset_data.corr(method='pearson')
        
        # Spearman相关系数（秩相关）
        spearman_corr = subset_data.corr(method='spearman')
        
        # 找出强相关变量对
        strong_correlations = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                pearson_val = abs(pearson_corr.iloc[i, j])
                spearman_val = abs(spearman_corr.iloc[i, j])
                
                if pearson_val > 0.7 or spearman_val > 0.7:
                    strong_correlations.append({
                        'var1': columns[i],
                        'var2': columns[j],
                        'pearson': pearson_corr.iloc[i, j],
                        'spearman': spearman_corr.iloc[i, j]
                    })
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'strong_correlations': strong_correlations,
            'multicollinearity_detected': len(strong_correlations) > 0
        }


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_generation_comparison(real_data: pd.DataFrame,
                                 generated_data: pd.DataFrame,
                                 columns: List[str],
                                 save_path: Optional[str] = None) -> None:
        """
        绘制真实数据与生成数据的对比图
        
        Args:
            real_data: 真实数据
            generated_data: 生成数据
            columns: 要比较的列名
            save_path: 保存路径
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 每行最多3个子图
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            row = i // 3
            col_idx = i % 3
            ax = axes[row, col_idx]
            
            # 绘制分布对比
            if col in real_data.columns and col in generated_data.columns:
                real_values = real_data[col].dropna()
                gen_values = generated_data[col].dropna()
                
                # 直方图
                ax.hist(real_values, bins=30, alpha=0.7, label='真实数据', 
                       density=True, color='blue')
                ax.hist(gen_values, bins=30, alpha=0.7, label='生成数据', 
                       density=True, color='red')
                
                ax.set_title(f'{col} 分布对比')
                ax.set_xlabel(col)
                ax.set_ylabel('密度')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_cols, n_rows * 3):
            row = i // 3
            col_idx = i % 3
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"对比图已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(correlation_matrix: pd.DataFrame,
                               title: str = "相关性热力图",
                               save_path: Optional[str] = None) -> None:
        """绘制相关性热力图"""
        plt.figure(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"热力图已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_qq_plots(data: pd.DataFrame,
                     columns: List[str],
                     distributions: List[str] = None,
                     save_path: Optional[str] = None) -> None:
        """绘制Q-Q图检验分布拟合效果"""
        if distributions is None:
            distributions = ['norm'] * len(columns)
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, (col, dist_name) in enumerate(zip(columns, distributions)):
            if col not in data.columns:
                continue
            
            ax = axes[i]
            col_data = data[col].dropna()
            
            try:
                dist = getattr(stats, dist_name)
                stats.probplot(col_data, dist=dist, plot=ax)
                ax.set_title(f'{col} Q-Q图 ({dist_name})')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'绘图失败:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{col} Q-Q图 (失败)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Q-Q图已保存到: {save_path}")
        
        plt.show()


class FileManager:
    """文件管理工具"""
    
    @staticmethod
    def save_generation_results(data: pd.DataFrame,
                              metadata: Dict[str, Any],
                              output_dir: str,
                              filename_prefix: str = "generated_population") -> Dict[str, str]:
        """
        保存生成结果
        
        Args:
            data: 生成的数据
            metadata: 元数据信息
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # 保存数据
        data_file = output_path / f"{filename_prefix}_{timestamp}.csv"
        data.to_csv(data_file, index=False, encoding='utf-8-sig')
        saved_files['data'] = str(data_file)
        
        # 保存元数据
        metadata_file = output_path / f"{filename_prefix}_metadata_{timestamp}.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        saved_files['metadata'] = str(metadata_file)
        
        # 保存摘要统计
        summary_file = output_path / f"{filename_prefix}_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("生成数据摘要统计\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {metadata.get('generation_time', 'N/A')}\n")
            f.write(f"数据量: {len(data)} 行\n")
            f.write(f"特征数: {len(data.columns)} 列\n\n")
            f.write("描述性统计:\n")
            f.write(str(data.describe()))
        saved_files['summary'] = str(summary_file)
        
        logger.info(f"生成结果已保存到: {output_dir}")
        return saved_files
    
    @staticmethod
    def load_generation_config(config_path: str) -> Dict[str, Any]:
        """加载生成配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_file.suffix.lower() == '.yaml':
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError("配置文件格式不支持，请使用YAML或JSON格式")


def compute_data_quality_score(data: pd.DataFrame,
                             validation_report: Dict[str, Any]) -> float:
    """
    计算数据质量得分 (0-1)
    
    Args:
        data: 数据
        validation_report: 验证报告
        
    Returns:
        质量得分
    """
    score = 1.0
    
    # 缺失值惩罚
    missing_ratio = validation_report['statistics'].get('missing_ratio', 0)
    score -= missing_ratio * 0.3
    
    # 验证问题惩罚
    n_issues = len(validation_report['issues'])
    score -= min(n_issues * 0.1, 0.5)
    
    # 数据量奖励
    n_rows = validation_report['statistics'].get('n_rows', 0)
    if n_rows < 100:
        score -= 0.2
    elif n_rows > 10000:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def create_generation_summary(generator_name: str,
                            n_agents: int,
                            execution_time: float,
                            data_quality_score: float,
                            config: Dict[str, Any]) -> Dict[str, Any]:
    """创建生成摘要"""
    return {
        'generator_name': generator_name,
        'generation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'n_agents_generated': n_agents,
        'execution_time_seconds': execution_time,
        'data_quality_score': data_quality_score,
        'config_used': config,
        'system_info': {
            'python_version': f"{psutil.python_version}",
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'cpu_count': psutil.cpu_count()
        }
    }
