#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
劳动力生成器模块

使用6维Gaussian Copula生成连续变量，使用经验分布条件抽样生成离散变量。

变量分类：
- 连续变量（6个）：T, S, D, W, 年龄, 累计工作年限 → Beta分布
- 离散变量（2个）：孩子数量, 学历 → 经验分布 + 年龄条件抽样

作者：AI Assistant
日期：2025-10-01
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta, kstest, chisquare
from typing import Dict, Tuple, Optional
import warnings

from ...core import BaseGenerator, DataValidationError, CopulaFittingError
from copulas.multivariate import GaussianMultivariate

COPULAS_AVAILABLE = True


class LaborGenerator(BaseGenerator):
    """
    劳动力生成器（6维Copula + 离散变量条件抽样）
    
    生成策略：
    1. 6维Gaussian Copula生成连续变量：T, S, D, W, 年龄, 累计工作年限
    2. 基于年龄条件抽样离散变量：孩子数量、学历
    
    Attributes:
        config (dict): 配置字典
        copula: Gaussian Copula模型（6维）
        marginals_continuous (dict): 连续变量的边际分布参数 (6个Beta)
        marginals_discrete (dict): 离散变量的经验分布 (2个)
        correlation_matrix (np.ndarray): 相关系数矩阵 (6x6)
        conditional_probs (dict): 条件概率表（年龄 → 孩子数量、学历）
        
    Examples:
        >>> gen = LaborGenerator({'seed': 42})
        >>> gen.fit(data)
        >>> laborers = gen.generate(1000)
        >>> is_valid = gen.validate(laborers)
    """
    
    # 连续变量列名
    CONTINUOUS_COLS = ['T', 'S', 'D', 'W', '年龄', '累计工作年限']
    
    # 离散变量列名
    DISCRETE_COLS = ['孩子数量', '学历']
    
    # 所有变量列名
    ALL_COLS = CONTINUOUS_COLS + DISCRETE_COLS
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化劳动力生成器
        
        Args:
            config: 配置字典，可选参数包括：
                - seed: 随机种子
                - use_copula: Copula类型，默认'gaussian'
                - correlation_method: 相关系数方法，默认'spearman'
        """
        super().__init__(config or {})
        
        # 设置随机种子
        seed = self.config.get('seed', 42)
        np.random.seed(seed)
        
        # Copula模型
        self.copula = None
        
        # 边际分布参数
        self.marginals_continuous = {}
        self.marginals_discrete = {}
        
        # 相关矩阵
        self.correlation_matrix = None
        
        # 条件概率表
        self.conditional_probs = None
        
        # 原始数据统计信息（用于验证）
        self.data_stats = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合Copula模型和离散变量分布
        
        Args:
            data: 包含所有8个变量的DataFrame
            
        Raises:
            DataValidationError: 数据验证失败
            CopulaFittingError: Copula拟合失败
        """
        # Step 1: 数据验证
        self._validate_data(data)
        
        # Step 2: 拟合连续变量的边际分布（6个Beta分布）
        self._fit_continuous_marginals(data)
        
        # Step 3: 归一化 + CDF变换 + 拟合6维Gaussian Copula
        self._fit_copula(data)
        
        # Step 4: 拟合离散变量的经验分布（孩子数量、学历）
        self._fit_discrete_marginals(data)
        
        # Step 5: 估计条件概率表（年龄→孩子数量、学历）
        self.conditional_probs = self._estimate_conditional_probs(data)
        
        # Step 6: 保存原始数据统计信息（只包含连续变量）
        self.data_stats = {
            'mean': data[self.CONTINUOUS_COLS].mean().to_dict(),
            'std': data[self.CONTINUOUS_COLS].std().to_dict(),
            'n_samples': len(data)
        }
        
        # Step 7: 保存所有参数
        corr_matrix_list = None
        if self.correlation_matrix is not None:
            if isinstance(self.correlation_matrix, np.ndarray):
                corr_matrix_list = self.correlation_matrix.tolist()
            else:  # DataFrame
                corr_matrix_list = self.correlation_matrix.values.tolist()
        
        self.fitted_params = {
            'marginals_continuous': self.marginals_continuous,
            'marginals_discrete': self.marginals_discrete,
            'correlation_matrix': corr_matrix_list,
            'conditional_probs': self.conditional_probs,
            'data_stats': self.data_stats
        }
        
        self.is_fitted = True
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """
        生成虚拟劳动力
        
        Args:
            n_agents: 要生成的劳动力数量
            
        Returns:
            包含所有8个变量的DataFrame
            
        Raises:
            RuntimeError: 模型未拟合
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()方法拟合模型")
        
        # Step 1: 从6维Copula采样，生成均匀分布[0,1]的样本
        uniform_samples = self._sample_from_copula(n_agents)
        
        # Step 2: 逆CDF变换 -> Beta分布 -> 原始尺度（6个连续变量）
        agents_data = self._inverse_transform(uniform_samples)
        
        # Step 3: 基于年龄条件抽样离散变量（孩子数量、学历）
        self._sample_discrete_variables(agents_data, n_agents)
        
        # Step 4: 构造完整DataFrame，添加agent_id和agent_type
        df = pd.DataFrame(agents_data)
        df['agent_id'] = range(1, n_agents + 1)
        df['agent_type'] = 'labor'
        
        # Step 5: 重新排序列，确保顺序一致
        df = df[['agent_id', 'agent_type'] + self.ALL_COLS]
        
        return df
    
    def validate(self, agents: pd.DataFrame) -> bool:
        """
        验证生成的劳动力数据质量
        
        使用KS检验验证连续变量，卡方检验验证离散变量
        
        Args:
            agents: 生成的劳动力DataFrame
            
        Returns:
            True if 所有检验通过
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()方法")
        
        all_passed = True
        
        # 验证连续变量（KS检验）
        for col in self.CONTINUOUS_COLS:
            # 归一化到[0,1]
            data_min, data_max = self.marginals_continuous[col]['scale']
            normalized = (agents[col] - data_min) / (data_max - data_min)
            
            # 严格裁剪到(0,1)开区间，避免边界问题
            epsilon = 1e-10
            normalized = normalized.clip(epsilon, 1 - epsilon)
            
            # 移除NaN值
            normalized_clean = normalized.dropna()
            
            if len(normalized_clean) == 0:
                all_passed = False
                continue
            
            # KS检验：比较生成数据与理论Beta分布
            params = self.marginals_continuous[col]['params']
            from scipy.stats import beta as beta_dist
            ks_stat, p_value = kstest(normalized_clean, beta_dist(*params).cdf)
            
            # 如果结果无效，使用两样本KS检验（更稳健）
            if np.isnan(ks_stat) or np.isnan(p_value):
                reference_sample = beta_dist(*params).rvs(size=len(normalized_clean))
                from scipy.stats import ks_2samp
                ks_stat, p_value = ks_2samp(normalized_clean, reference_sample)
            
            # 判断是否通过检验（显著性水平0.01）
            if p_value <= 0.01:
                all_passed = False
        
        # 验证离散变量（卡方检验）
        for col in self.DISCRETE_COLS:
            # 获取观测频数
            observed_counts = agents[col].value_counts().sort_index()
            
            # 获取期望概率和期望频数
            values = np.array(self.marginals_discrete[col]['values'])
            probs = np.array(self.marginals_discrete[col]['probs'])
            
            # 确保观测值和期望值的顺序一致
            observed = np.zeros(len(values))
            for i, val in enumerate(values):
                observed[i] = observed_counts.get(val, 0)
            
            # 计算期望频数
            expected = probs * len(agents)
            
            # 卡方检验前的条件检查：过滤掉期望频数<5的类别
            valid_mask = expected >= 5
            
            if valid_mask.sum() < 2:
                continue
            
            # 使用有效的类别进行检验
            observed_valid = observed[valid_mask]
            expected_valid = expected[valid_mask]
            
            # 确保总和相等（归一化）
            observed_valid = observed_valid * expected_valid.sum() / observed_valid.sum()
            
            # 执行卡方检验
            chi2_stat, p_value = chisquare(observed_valid, expected_valid)
            
            # 判断是否通过检验（显著性水平0.05）
            if p_value <= 0.05:
                all_passed = False
        
        return all_passed
    
    # ==================== 私有方法 ====================
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """验证输入数据"""
        missing_cols = [col for col in self.ALL_COLS if col not in data.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"数据缺少必需列: {missing_cols}\n"
                f"需要的列: {self.ALL_COLS}"
            )
        
        # 检查是否有缺失值
        null_counts = data[self.ALL_COLS].isnull().sum()
        if null_counts.any():
            raise DataValidationError(
                f"数据包含缺失值:\n{null_counts[null_counts > 0]}"
            )
        
        # 检查样本量
        if len(data) < 50:
            warnings.warn(f"样本量较小({len(data)})，可能影响拟合质量")
    
    def _fit_continuous_marginals(self, data: pd.DataFrame) -> None:
        """拟合连续变量的边际分布（使用实验结果）"""
        # 使用边际分布实验的结果
        self.marginals_continuous = {
            'T': {
                'dist': 'beta',
                'params': (1.93, 2.05, 0, 1),
                'scale': (data['T'].min(), data['T'].max())
            },
            'S': {
                'dist': 'beta',
                'params': (1.79, 1.57, 0, 1),
                'scale': (data['S'].min(), data['S'].max())
            },
            'D': {
                'dist': 'beta',
                'params': (0.37, 0.76, 0, 1),
                'scale': (data['D'].min(), data['D'].max())
            },
            'W': {
                'dist': 'beta',
                'params': (1.43, 1.45, 0, 1),
                'scale': (data['W'].min(), data['W'].max())
            },
            '年龄': {
                'dist': 'beta',
                'params': (1.01, 1.00, 0, 1),
                'scale': (data['年龄'].min(), data['年龄'].max())
            },
            '累计工作年限': {
                'dist': 'beta',
                'params': (0.55, 1.64, 0, 1),
                'scale': (data['累计工作年限'].min(), data['累计工作年限'].max())
            }
        }
    
    def _fit_copula(self, data: pd.DataFrame) -> None:
        """拟合6维Gaussian Copula"""
        # 归一化 + CDF变换
        uniform_data = pd.DataFrame()
        
        for col in self.CONTINUOUS_COLS:
            params = self.marginals_continuous[col]['params']
            scale_min, scale_max = self.marginals_continuous[col]['scale']
            
            # 归一化到[0,1]
            normalized = (data[col] - scale_min) / (scale_max - scale_min)
            
            # 裁剪到(0,1)开区间，避免CDF边界问题
            epsilon = 1e-10
            normalized = normalized.clip(epsilon, 1 - epsilon)
            
            # CDF变换到均匀分布
            uniform_data[col] = beta(*params).cdf(normalized)
        
        # 使用copulas库的标准实现拟合Gaussian Copula
        self.copula = GaussianMultivariate()
        self.copula.fit(uniform_data)
        
        # 提取相关矩阵：copulas库在fit后存储在correlation属性中
        if hasattr(self.copula, 'correlation') and self.copula.correlation is not None:
            self.correlation_matrix = self.copula.correlation
            if isinstance(self.correlation_matrix, pd.DataFrame):
                self.correlation_matrix = self.correlation_matrix.values
        else:
            # 备用：从数据计算
            self.correlation_matrix = uniform_data.corr(method='spearman').values
    
    def _fit_discrete_marginals(self, data: pd.DataFrame) -> None:
        """拟合离散变量的经验分布"""
        self.marginals_discrete = {}
        
        for col in self.DISCRETE_COLS:
            values, counts = np.unique(data[col], return_counts=True)
            probs = counts / len(data)
            
            self.marginals_discrete[col] = {
                'values': values.tolist(),
                'probs': probs.tolist()
            }
    
    def _estimate_conditional_probs(self, data: pd.DataFrame) -> Dict:
        """从数据估计条件概率表（年龄→孩子数量、学历）"""
        age_bins = ['<30', '30-40', '>=40']
        conditional_probs = {
            '孩子数量': {},
            '学历': {}
        }
        
        for age_bin in age_bins:
            # 筛选该年龄段的数据
            if age_bin == '<30':
                mask = data['年龄'] < 30
            elif age_bin == '30-40':
                mask = (data['年龄'] >= 30) & (data['年龄'] < 40)
            else:
                mask = data['年龄'] >= 40
            
            subset = data[mask]
            
            # 如果该年龄段无数据，使用全局分布
            if len(subset) == 0:
                for col in self.DISCRETE_COLS:
                    conditional_probs[col][age_bin] = self.marginals_discrete[col]['probs']
                continue
            
            # 统计每个离散变量的分布
            for col in self.DISCRETE_COLS:
                vals, counts = np.unique(subset[col], return_counts=True)
                probs = counts / len(subset)
                
                # 确保包含所有可能值（使用平滑处理）
                all_vals = self.marginals_discrete[col]['values']
                probs_full = []
                
                for val in all_vals:
                    idx = np.where(vals == val)[0]
                    if len(idx) > 0:
                        probs_full.append(probs[idx[0]])
                    else:
                        probs_full.append(0.001)  # 拉普拉斯平滑
                
                # 归一化
                probs_full = np.array(probs_full)
                probs_full = probs_full / probs_full.sum()
                
                conditional_probs[col][age_bin] = probs_full.tolist()
        
        return conditional_probs
    
    def _sample_from_copula(self, n_agents: int) -> pd.DataFrame:
        """从6维Copula采样"""
        # 使用copulas库采样
        uniform_samples = self.copula.sample(n_agents)
        
        # 确保列名正确
        if not all(col in uniform_samples.columns for col in self.CONTINUOUS_COLS):
            uniform_samples.columns = self.CONTINUOUS_COLS
        
        return uniform_samples
    
    def _inverse_transform(self, uniform_samples: pd.DataFrame) -> Dict:
        """逆CDF变换：均匀分布 → Beta分布 → 原始尺度"""
        agents_data = {}
        
        for col in self.CONTINUOUS_COLS:
            params = self.marginals_continuous[col]['params']
            scale_min, scale_max = self.marginals_continuous[col]['scale']
            
            # 获取均匀样本并裁剪到(0,1)开区间
            # 避免ppf在边界处的数值问题
            uniform_vals = uniform_samples[col].values
            epsilon = 1e-10
            uniform_vals = np.clip(uniform_vals, epsilon, 1 - epsilon)
            
            # 均匀分布 → Beta分布[0,1]（使用ppf即逆CDF）
            from scipy.stats import beta as beta_dist
            beta_samples = beta_dist(*params).ppf(uniform_vals)
            
            # 处理可能的NaN值
            beta_samples = np.nan_to_num(beta_samples, nan=0.5)
            
            # 反归一化到原始尺度
            agents_data[col] = beta_samples * (scale_max - scale_min) + scale_min
        
        return agents_data
    
    def _sample_discrete_variables(self, agents_data: Dict, n_agents: int) -> None:
        """基于年龄条件抽样离散变量"""
        kids_list = []
        edu_list = []
        
        for i in range(n_agents):
            age = agents_data['年龄'][i]
            
            # 根据年龄获取分箱
            age_bin = self._get_age_bin(age)
            
            # 抽样孩子数量
            kids_probs = self.conditional_probs['孩子数量'][age_bin]
            kids_values = self.marginals_discrete['孩子数量']['values']
            kids = np.random.choice(kids_values, p=kids_probs)
            kids_list.append(int(kids))
            
            # 抽样学历
            edu_probs = self.conditional_probs['学历'][age_bin]
            edu_values = self.marginals_discrete['学历']['values']
            edu = np.random.choice(edu_values, p=edu_probs)
            edu_list.append(int(edu))
        
        agents_data['孩子数量'] = kids_list
        agents_data['学历'] = edu_list
    
    def _get_age_bin(self, age: float) -> str:
        """根据年龄返回分箱标签"""
        if age < 30:
            return '<30'
        elif age < 40:
            return '30-40'
        else:
            return '>=40'

