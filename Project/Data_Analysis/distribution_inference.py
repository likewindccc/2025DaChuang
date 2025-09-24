# -*- coding: utf-8 -*-
"""
农村女性就业市场主体特征分布推断模块

本模块实现基于最大似然估计(MLE)和Anderson-Darling检验的概率分布拟合分析，
为农村女性就业市场数据提供理论分布模型。

主要功能：
1. 多种概率分布的参数估计 (MLE)
2. 拟合优度检验 (Anderson-Darling)
3. 模型选择与比较 (AIC/BIC)
4. 分布推断结果报告生成

Author: Claude-4 AI Assistant
Date: 2024-09-24
Version: 1.3.0
对应研究计划第4.2节：市场主体特征的确定
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings

# 配置警告过滤
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# ==================== 常量定义 ====================
# 数值计算常量
NUMERICAL_EPSILON = 1e-6          # 数值稳定性常量，避免边界值
LOGNORM_SHIFT = 0.1               # 对数正态分布零值处理偏移量  
MIN_SAMPLE_SIZE = 10              # 最小样本量要求
CLIP_EPSILON = 1e-10              # CDF裁剪阈值，避免极值

# Anderson-Darling统计量调整参数
AD_ADJUSTMENT_COEF1 = 0.75        # 样本量调整系数1
AD_ADJUSTMENT_COEF2 = 2.25        # 样本量调整系数2

# p值计算阈值
P_VALUE_THRESHOLD_HIGH = 0.6      # 高阈值
P_VALUE_THRESHOLD_MID = 0.34      # 中阈值  
P_VALUE_THRESHOLD_LOW = 0.2       # 低阈值

# 支持的概率分布族
SUPPORTED_DISTRIBUTIONS = {
    'norm': stats.norm,              # 正态分布
    'gamma': stats.gamma,            # 伽马分布
    'expon': stats.expon,            # 指数分布
    'weibull_min': stats.weibull_min, # 威布尔分布
    'lognorm': stats.lognorm,        # 对数正态分布
    'beta': stats.beta,              # 贝塔分布
    'uniform': stats.uniform,        # 均匀分布
    'pareto': stats.pareto,          # 帕累托分布
    'chi2': stats.chi2,              # 卡方分布
    'genextreme': stats.genextreme   # 广义极值分布
}

class DistributionFitter:
    """
    概率分布拟合与统计检验类
    
    该类实现了多种概率分布的参数估计和拟合优度检验，主要用于农村女性就业市场数据的
    分布建模。支持的分析包括：
    - 最大似然估计(MLE)参数估计
    - Anderson-Darling拟合优度检验  
    - AIC/BIC模型选择
    - 统计推断结果比较
    
    Attributes:
        distributions (Dict): 支持的概率分布字典
        results (Dict): 存储各变量的分布拟合结果
        
    Example:
        >>> fitter = DistributionFitter()
        >>> fitter.fit_variable(data, '变量名')
        >>> fitter.create_comparison_table()
    """
    
    def __init__(self) -> None:
        """
        初始化分布拟合器
        
        使用预定义的常量SUPPORTED_DISTRIBUTIONS初始化支持的分布族，
        并创建空的结果存储字典。
        """
        # 使用全局常量定义的分布族
        self.distributions: Dict[str, Any] = SUPPORTED_DISTRIBUTIONS.copy()
        
        # 存储每个变量的拟合结果
        # 结构: {变量名: {分布名: {参数, 统计量, 指标}}}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    def mle_estimation(self, data: np.ndarray, dist_name: str) -> Optional[Tuple[float, ...]]:
        """
        最大似然估计 (MLE) 参数估计
        
        对指定的概率分布进行最大似然参数估计。针对不同分布类型的数据要求，
        进行相应的预处理（如Beta分布的标准化、对数正态分布的零值处理）。
        
        Args:
            data (np.ndarray): 输入数据数组
            dist_name (str): 分布名称，必须在支持的分布列表中
            
        Returns:
            Optional[Tuple[float, ...]]: 估计的参数元组，失败时返回None
            
        Note:
            - Beta分布：数据标准化到[0,1]区间
            - 对数正态分布：零值添加偏移量避免数值问题
            - 均匀分布：直接使用数据范围计算参数
        """
        try:
            dist = self.distributions[dist_name]
            
            # ========== 特殊分布的参数估计处理 ==========
            if dist_name == 'beta':
                # Beta分布：标准化到[0,1]区间并避免边界值
                data_range = data.max() - data.min()
                if data_range == 0:  # 处理常数数据的情况
                    return None
                data_scaled = (data - data.min()) / data_range
                data_scaled = np.clip(data_scaled, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
                params = dist.fit(data_scaled, floc=0, fscale=1)
                
            elif dist_name == 'uniform':
                # 均匀分布：直接使用数据范围
                params = (data.min(), data.max() - data.min())
                
            elif dist_name == 'lognorm':
                # 对数正态分布：添加偏移量避免零值和负值
                data_shifted = data + LOGNORM_SHIFT
                params = dist.fit(data_shifted)
                
            elif dist_name == 'pareto':
                # 帕累托分布：固定位置参数为0
                params = dist.fit(data, floc=0)
                
            else:
                # 标准MLE估计（正态、伽马、指数等）
                params = dist.fit(data)
            
            return params
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"    ❌ MLE估计失败 ({dist_name}): {str(e)}")
            return None
        except Exception as e:
            print(f"    ❌ MLE估计未知错误 ({dist_name}): {str(e)}")
            return None
    
    def anderson_darling_test(self, data: np.ndarray, dist_name: str, 
                            params: Tuple[float, ...]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Anderson-Darling拟合优度检验
        
        计算Anderson-Darling统计量来检验数据是否符合指定的概率分布。
        该检验对分布的尾部更加敏感，适用于检验分布拟合的质量。
        
        Args:
            data (np.ndarray): 原始数据数组
            dist_name (str): 分布名称
            params (Tuple[float, ...]): 分布参数
            
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: 
                (原始AD统计量, 调整后AD统计量, p值)，失败时返回(None, None, None)
                
        Note:
            - AD统计量越小，拟合越好
            - p值>0.05通常认为接受原假设（数据符合分布）
            - 针对不同分布类型进行相应的数据预处理
        """
        try:
            dist = self.distributions[dist_name]
            n = len(data)
            
            # ========== 根据分布类型准备数据 ==========
            if dist_name == 'beta':
                # Beta分布：标准化处理
                data_range = data.max() - data.min()
                if data_range == 0:
                    return None, None, None
                data_processed = (data - data.min()) / data_range
                data_processed = np.clip(data_processed, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
                
            elif dist_name == 'lognorm':
                # 对数正态分布：添加偏移量
                data_processed = data + LOGNORM_SHIFT
                
            else:
                # 其他分布：使用原始数据
                data_processed = data.copy()
            
            # 对数据排序并计算理论CDF值
            data_sorted = np.sort(data_processed)
            F = dist.cdf(data_sorted, *params)
            
            # 避免极值，确保数值稳定性
            F = np.clip(F, CLIP_EPSILON, 1 - CLIP_EPSILON)
            
            # ========== 计算AD统计量 ==========
            i = np.arange(1, n + 1)
            # AD统计量标准公式
            ad_stat = -n - np.sum((2*i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
            
            # 样本量调整（提高小样本的准确性）
            ad_stat_adj = ad_stat * (1 + AD_ADJUSTMENT_COEF1/n + AD_ADJUSTMENT_COEF2/n**2)
            
            # ========== 计算p值（近似公式） ==========
            p_value = self._calculate_ad_p_value(ad_stat_adj)
            
            return ad_stat, ad_stat_adj, p_value
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"    ❌ AD检验计算失败 ({dist_name}): {str(e)}")
            return None, None, None
        except Exception as e:
            print(f"    ❌ AD检验未知错误 ({dist_name}): {str(e)}")
            return None, None, None
    
    def _calculate_ad_p_value(self, ad_stat_adj: float) -> float:
        """
        计算Anderson-Darling统计量的p值
        
        使用分段近似公式计算p值，基于统计文献中的标准方法。
        
        Args:
            ad_stat_adj (float): 调整后的AD统计量
            
        Returns:
            float: p值，范围[0,1]
        """
        # 根据AD统计量大小选择不同的近似公式
        if ad_stat_adj >= P_VALUE_THRESHOLD_HIGH:
            # 高统计量值：指数衰减公式
            p_value = np.exp(1.2937 - 5.709*ad_stat_adj + 0.0186*(ad_stat_adj**2))
        elif ad_stat_adj >= P_VALUE_THRESHOLD_MID:
            # 中等统计量值：修正指数公式
            p_value = np.exp(0.9177 - 4.279*ad_stat_adj - 1.38*(ad_stat_adj**2))
        elif ad_stat_adj >= P_VALUE_THRESHOLD_LOW:
            # 较小统计量值：互补指数公式
            p_value = 1 - np.exp(-8.318 + 42.796*ad_stat_adj - 59.938*(ad_stat_adj**2))
        else:
            # 极小统计量值：高阶多项式公式
            p_value = 1 - np.exp(-13.436 + 101.14*ad_stat_adj - 223.73*(ad_stat_adj**2))
        
        # 确保p值在有效范围内
        return np.clip(p_value, 0.0, 1.0)
    
    def calculate_goodness_of_fit(self, data: np.ndarray, dist_name: str, 
                                params: Tuple[float, ...]) -> Tuple[Optional[float], Optional[float], 
                                                                   Optional[float], Optional[float]]:
        """
        计算分布拟合的各种优度指标
        
        计算对数似然、AIC、BIC和伪R²等统计指标，用于评估和比较不同分布的拟合效果。
        
        Args:
            data (np.ndarray): 原始数据数组
            dist_name (str): 分布名称
            params (Tuple[float, ...]): 分布参数
            
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]: 
                (对数似然值, AIC值, BIC值, 伪R²值)，失败时返回(None, None, None, None)
                
        Note:
            - AIC/BIC越小表示模型越好
            - 伪R²越接近1表示拟合越好
            - 对数似然值越大表示拟合越好
        """
        try:
            dist = self.distributions[dist_name]
            n = len(data)
            k = len(params)  # 参数个数
            
            # ========== 根据分布类型计算对数似然 ==========
            if dist_name == 'beta':
                # Beta分布：使用标准化数据
                data_range = data.max() - data.min()
                if data_range == 0:
                    return None, None, None, None
                data_scaled = (data - data.min()) / data_range
                data_scaled = np.clip(data_scaled, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
                log_likelihood = np.sum(dist.logpdf(data_scaled, *params))
                
                # 计算理论均值（还原到原始尺度）
                theoretical_mean = dist.mean(*params) * data_range + data.min()
                ss_res = np.sum((data - theoretical_mean)**2)
                
            elif dist_name == 'lognorm':
                # 对数正态分布：使用偏移数据
                data_shifted = data + LOGNORM_SHIFT
                log_likelihood = np.sum(dist.logpdf(data_shifted, *params))
                
                # 计算理论均值（减去偏移量）
                theoretical_mean = dist.mean(*params) - LOGNORM_SHIFT
                ss_res = np.sum((data - theoretical_mean)**2)
                
            else:
                # 标准分布：直接使用原始数据
                log_likelihood = np.sum(dist.logpdf(data, *params))
                theoretical_mean = dist.mean(*params)
                ss_res = np.sum((data - theoretical_mean)**2)
            
            # ========== 计算模型选择指标 ==========
            # AIC (Akaike Information Criterion)
            aic = 2*k - 2*log_likelihood
            
            # BIC (Bayesian Information Criterion)  
            bic = k*np.log(n) - 2*log_likelihood
            
            # ========== 计算伪R² (McFadden's R²) ==========
            ss_tot = np.sum((data - np.mean(data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # 处理数值异常
            if not np.isfinite(log_likelihood):
                log_likelihood = -np.inf
            if not np.isfinite(aic):
                aic = np.inf
            if not np.isfinite(bic):
                bic = np.inf
            if not np.isfinite(r_squared):
                r_squared = 0.0
                
            return log_likelihood, aic, bic, r_squared
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"    ❌ 拟合优度计算失败 ({dist_name}): {str(e)}")
            return None, None, None, None
        except Exception as e:
            print(f"    ❌ 拟合优度未知错误 ({dist_name}): {str(e)}")
            return None, None, None, None
    
    def fit_variable(self, data: np.ndarray, var_name: str) -> Dict[str, Dict[str, Any]]:
        """
        对单个变量拟合所有支持的概率分布
        
        执行完整的分布拟合分析流程，包括参数估计、拟合检验和模型比较。
        为每个分布计算MLE参数、AD检验统计量、AIC/BIC等指标。
        
        Args:
            data (np.ndarray): 变量数据数组
            var_name (str): 变量名称，用于结果标识和输出
            
        Returns:
            Dict[str, Dict[str, Any]]: 各分布的拟合结果字典
                结构: {分布名: {参数, 统计量, 拟合指标}}
                
        Note:
            - 自动跳过拟合失败的分布
            - 结果按AIC排序，选择最佳拟合分布
            - 所有结果保存到self.results中
        """
        print(f"\n🎯 分析变量: {var_name}")
        print("="*60)
        
        # ========== 数据描述性统计 ==========
        n_samples = len(data)
        data_mean = data.mean()
        data_std = data.std()
        data_min, data_max = data.min(), data.max()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        print(f"数据统计: N={n_samples}, 均值={data_mean:.3f}, 标准差={data_std:.3f}")
        print(f"范围: [{data_min:.3f}, {data_max:.3f}]")
        print(f"偏度: {skewness:.3f}, 峰度: {kurtosis:.3f}")
        
        # 初始化结果字典
        results: Dict[str, Dict[str, Any]] = {}
        successful_fits = 0
        
        # ========== 遍历所有分布进行拟合 ==========
        for dist_name in self.distributions.keys():
            print(f"\n📊 拟合分布: {dist_name}")
            
            # Step 1: MLE参数估计
            params = self.mle_estimation(data, dist_name)
            if params is None:
                continue
            
            print(f"  参数估计: {[f'{p:.4f}' for p in params]}")
            
            # Step 2: Anderson-Darling拟合检验
            ad_stat, ad_stat_adj, p_value = self.anderson_darling_test(data, dist_name, params)
            if ad_stat is None:
                continue
            
            print(f"  AD统计量: {ad_stat:.4f} (调整后: {ad_stat_adj:.4f})")
            print(f"  p值: {p_value:.4f}")
            
            # Step 3: 拟合优度指标计算
            log_like, aic, bic, r2 = self.calculate_goodness_of_fit(data, dist_name, params)
            if log_like is not None:
                print(f"  对数似然: {log_like:.4f}")
                print(f"  AIC: {aic:.4f}, BIC: {bic:.4f}")
                print(f"  伪R²: {r2:.4f}")
                successful_fits += 1
            
            # Step 4: 存储完整结果
            results[dist_name] = {
                'params': params,
                'ad_statistic': ad_stat,
                'ad_statistic_adj': ad_stat_adj, 
                'p_value': p_value,
                'log_likelihood': log_like,
                'aic': aic,
                'bic': bic,
                'r_squared': r2
            }
        
        # ========== 模型选择：寻找最佳拟合 ==========
        if results:
            # 筛选成功拟合的分布
            valid_results = {k: v for k, v in results.items() 
                           if v['aic'] is not None and np.isfinite(v['aic'])}
            
            if valid_results:
                # 按AIC选择最佳模型
                best_dist_name = min(valid_results.keys(), 
                                   key=lambda x: valid_results[x]['aic'])
                best_aic = valid_results[best_dist_name]['aic']
                
                print(f"\n🏆 最佳拟合分布: {best_dist_name} (AIC = {best_aic:.4f})")
                print(f"✅ 成功拟合 {successful_fits}/{len(self.distributions)} 个分布")
            else:
                print(f"\n⚠️ 所有分布拟合均未成功")
        
        # 保存结果到实例变量
        self.results[var_name] = results
        return results
    
    def create_comparison_table(self) -> None:
        """
        生成各变量分布拟合结果的汇总比较表
        
        按变量分组显示所有拟合分布的统计指标，包括AIC、BIC、AD统计量、p值等。
        结果按AIC升序排列，便于模型选择和比较。
        
        Note:
            - 表格显示格式化的数值结果
            - 根据p值(>0.05)给出接受/拒绝的结论
            - 空值和无效结果自动跳过
        """
        print("\n" + "="*80)
        print("📋 分布拟合结果汇总表")
        print("="*80)
        
        for var_name, var_results in self.results.items():
            print(f"\n变量: {var_name}")
            print("-"*60)
            print(f"{'分布':<12} {'AIC':<8} {'BIC':<8} {'AD统计量':<10} {'p值':<8} {'结论'}")
            print("-"*60)
            
            # 按AIC升序排序（越小越好）
            sorted_results = sorted(
                var_results.items(),
                key=lambda x: x[1]['aic'] if x[1]['aic'] is not None and np.isfinite(x[1]['aic']) else float('inf')
            )
            
            # 显示每个分布的拟合结果
            for dist_name, result in sorted_results:
                aic = result.get('aic')
                bic = result.get('bic')
                ad_stat = result.get('ad_statistic_adj')
                p_val = result.get('p_value')
                
                # 只显示完整有效的结果
                if all(x is not None and np.isfinite(x) for x in [aic, bic, ad_stat, p_val]):
                    # 假设检验结论（H0：数据符合该分布）
                    conclusion = "接受" if p_val > 0.05 else "拒绝"
                    print(f"{dist_name:<12} {aic:<8.2f} {bic:<8.2f} {ad_stat:<10.4f} {p_val:<8.4f} {conclusion}")


def main() -> Optional[DistributionFitter]:
    """
    主程序入口：执行完整的农村女性就业市场数据分布推断分析
    
    程序流程：
    1. 加载清洗后的数据文件
    2. 创建复合状态变量（如每周工作时长）
    3. 定义关键变量（状态变量+控制变量）
    4. 对每个变量进行多分布拟合分析
    5. 生成汇总比较表
    
    Returns:
        Optional[DistributionFitter]: 成功时返回拟合器对象，失败时返回None
        
    Note:
        - 数据文件应为UTF-8编码的CSV格式
        - 变量定义严格对应研究计划第4.2节
        - 自动跳过数据点不足(N<10)的变量
    """
    print("🔍 农村女性就业市场主体特征分布推断")
    print("基于MLE参数估计与Anderson-Darling检验")
    print("="*60)
    
    # ========== Step 1: 数据加载与验证 ==========
    try:
        df = pd.read_csv("cleaned_data.csv", encoding='utf-8-sig')
        print(f"✓ 成功加载数据：{df.shape[0]}个样本，{df.shape[1]}个变量")
        
        # 基本数据质量检查
        if df.shape[0] < MIN_SAMPLE_SIZE:
            print(f"❌ 样本量不足：需要至少{MIN_SAMPLE_SIZE}个样本")
            return None
            
    except FileNotFoundError:
        print("❌ 数据文件 'cleaned_data.csv' 未找到")
        return None
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    # ========== Step 2: 创建分布拟合器 ==========
    fitter = DistributionFitter()
    
    # ========== Step 3: 构造复合状态变量 ==========
    # T = 工作时间投入 = 每周期望工作天数 × 每天期望工作时数
    df['每周工作时长'] = df['每周期望工作天数'] * df['每天期望工作时数']
    print("✓ 创建复合状态变量：每周工作时长 = 工作天数 × 工作时数")
    
    # ========== Step 4: 定义分析变量集合 ==========
    # 严格按照研究计划第4.2节定义的变量体系
    key_variables = {
        # ===== 核心状态变量 x = (T, S, D, W) =====
        '每周工作时长': df['每周工作时长'],            # T - 工作时间投入（复合变量）
        '工作能力评分': df['工作能力评分'],            # S - 工作能力水平  
        '数字素养评分': df['数字素养评分'],            # D - 数字素养
        '每月期望收入': df['每月期望收入'],            # W - 期望工作待遇
        
        # ===== 控制变量 σ =====
        '年龄': df['年龄'],                          # 人口统计学控制变量
        '累计工作年限': df['累计工作年限'],            # 工作经验控制变量
        '家务劳动时间': df['家务劳动时间'],            # 时间配置控制变量
        '闲暇时间': df['闲暇时间'],                   # 时间配置控制变量
        
        # ===== 原始构成变量（用于验证） =====
        '每周期望工作天数': df['每周期望工作天数'],    # T的构成要素1
        '每天期望工作时数': df['每天期望工作时数']     # T的构成要素2
    }
    
    print(f"✓ 定义{len(key_variables)}个关键变量待分析")
    
    # ========== Step 5: 批量分布拟合分析 ==========
    analyzed_count = 0
    skipped_count = 0
    
    for var_name, data in key_variables.items():
        # 数据预处理：移除缺失值
        clean_data = data.dropna()
        
        # 样本量检查
        if len(clean_data) < MIN_SAMPLE_SIZE:
            print(f"⚠️  变量 '{var_name}' 数据点不足({len(clean_data)}<{MIN_SAMPLE_SIZE})，跳过分析")
            skipped_count += 1
            continue
        
        # 执行分布拟合分析
        fitter.fit_variable(clean_data.values, var_name)
        analyzed_count += 1
    
    # ========== Step 6: 生成分析报告 ==========
    if analyzed_count > 0:
        fitter.create_comparison_table()
        
        print(f"\n✅ 分布推断分析完成!")
        print(f"📊 成功分析: {analyzed_count} 个变量")
        if skipped_count > 0:
            print(f"⚠️  跳过: {skipped_count} 个变量（数据不足）")
        print("="*60)
        
        return fitter
    else:
        print("\n❌ 没有变量满足分析条件")
        return None

if __name__ == "__main__":
    fitter = main()
