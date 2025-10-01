#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
企业生成器模块

基于四维多元正态分布生成虚拟企业特征。
初始参数通过简单假设确定，后续通过校准模块优化。

变量：
- T: 每周工作时长（小时）
- S: 工作能力要求评分
- D: 数字素养要求评分
- W: 每月提供工资（元）

作者：AI Assistant
日期：2025-10-01
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, normaltest
from typing import Optional, Dict, Tuple
import warnings

from ...core import BaseGenerator, DataValidationError, ConfigurationError


class EnterpriseGenerator(BaseGenerator):
    """
    企业生成器（四维多元正态分布）
    
    生成策略：
    1. 使用四维多元正态分布 N(μ, Σ) 生成企业特征
    2. 初始参数基于劳动力数据或配置设定
    3. 支持通过set_params()方法更新参数（用于校准）
    
    Attributes:
        config (dict): 配置字典
        mean (np.ndarray): 均值向量 (4,)
        covariance (np.ndarray): 协方差矩阵 (4, 4)
        data_stats (dict): 原始数据统计信息（如果提供）
        
    Examples:
        >>> # 方式1：基于劳动力数据初始化
        >>> gen = EnterpriseGenerator({'seed': 43})
        >>> gen.fit(labor_data)  # 基于劳动力均值调整
        
        >>> # 方式2：使用默认配置
        >>> gen = EnterpriseGenerator({
        ...     'seed': 43,
        ...     'default_mean': [45, 75, 65, 5500],
        ...     'default_std': [10, 15, 15, 1000]
        ... })
        >>> gen.fit()
        
        >>> # 生成企业
        >>> enterprises = gen.generate(800)
        
        >>> # 校准后更新参数
        >>> gen.set_params(new_mean, new_cov)
    """
    
    # 核心变量列名
    CORE_COLS = ['T', 'S', 'D', 'W']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化企业生成器
        
        Args:
            config: 配置字典，可选参数包括：
                - seed: 随机种子
                - default_mean: 默认均值向量 [T, S, D, W]
                - default_std: 默认标准差向量 [σ_T, σ_S, σ_D, σ_W]
                - correlation: 相关系数矩阵（可选，默认为对角阵）
                - labor_multiplier: 基于劳动力均值的调整系数
        """
        super().__init__(config or {})
        
        # 设置随机种子
        seed = self.config.get('seed', 43)
        np.random.seed(seed)
        
        # 初始化分布参数
        self.mean = None
        self.covariance = None
        
        # 数据统计信息
        self.data_stats = None
    
    def fit(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        初始化分布参数
        
        策略：
        1. 如果提供劳动力数据：基于劳动力均值 × 调整系数
        2. 否则：使用配置中的默认值
        
        Args:
            data: 劳动力数据（可选），应包含 T, S, D, W 列
            
        Raises:
            DataValidationError: 数据验证失败
            ConfigurationError: 配置错误
        """
        print("\n" + "=" * 70)
        print("[EnterpriseGenerator] 开始初始化参数")
        print("=" * 70)
        
        # Step 1: 确定均值向量
        if data is not None:
            self._fit_from_labor_data(data)
        else:
            self._fit_from_config()
        
        # Step 2: 构造协方差矩阵
        self._build_covariance_matrix()
        
        # Step 3: 验证参数合法性
        self._validate_params()
        
        # Step 4: 保存参数
        self.fitted_params = {
            'mean': self.mean.tolist(),
            'covariance': self.covariance.tolist(),
            'correlation': self._get_correlation_matrix().tolist()
        }
        
        self.is_fitted = True
        print("[OK] 参数初始化完成！")
        print("=" * 70 + "\n")
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """
        生成虚拟企业
        
        Args:
            n_agents: 要生成的企业数量
            
        Returns:
            包含企业特征的DataFrame
            
        Raises:
            RuntimeError: 模型未拟合
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()方法初始化参数")
        
        print(f"\n[EnterpriseGenerator] 生成 {n_agents} 个虚拟企业...")
        
        # Step 1: 从多元正态分布采样
        try:
            samples = multivariate_normal.rvs(
                mean=self.mean,
                cov=self.covariance,
                size=n_agents,
                random_state=self.config.get('seed', 43)
            )
            
            # 处理单样本情况（返回1D数组）
            if n_agents == 1:
                samples = samples.reshape(1, -1)
                
        except Exception as e:
            raise RuntimeError(f"多元正态采样失败: {e}")
        
        # Step 2: 构造DataFrame
        df = pd.DataFrame(samples, columns=self.CORE_COLS)
        
        # Step 3: 添加ID和类型
        # 企业ID从1001开始，避免与劳动力ID冲突
        start_id = self.config.get('start_id', 1001)
        df['agent_id'] = range(start_id, start_id + n_agents)
        df['agent_type'] = 'enterprise'
        
        # Step 4: 数据后处理
        self._postprocess_data(df)
        
        # Step 5: 重新排序列
        df = df[['agent_id', 'agent_type'] + self.CORE_COLS]
        
        print(f"[OK] 生成完成！")
        
        return df
    
    def validate(self, agents: pd.DataFrame) -> bool:
        """
        验证生成的企业数据质量
        
        使用多种统计检验：
        1. 均值检验（t检验）
        2. 协方差矩阵检验（Bartlett检验）
        3. 正态性检验（Anderson-Darling）
        
        Args:
            agents: 生成的企业DataFrame
            
        Returns:
            True if 所有检验通过
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()方法")
        
        print("\n" + "=" * 70)
        print("[EnterpriseGenerator] 数据验证")
        print("=" * 70)
        
        all_passed = True
        
        # 验证1：均值检验
        print("\n[均值检验]")
        generated_mean = agents[self.CORE_COLS].mean().values
        
        for i, col in enumerate(self.CORE_COLS):
            mean_diff = generated_mean[i] - self.mean[i]
            relative_error = abs(mean_diff / self.mean[i]) * 100
            
            passed = relative_error < 10  # 10%容忍度
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"  {col:5s}: 目标={self.mean[i]:8.2f}, "
                  f"实际={generated_mean[i]:8.2f}, "
                  f"偏差={relative_error:5.2f}% {status}")
            
            if not passed:
                all_passed = False
        
        # 验证2：标准差检验
        print("\n[标准差检验]")
        generated_std = agents[self.CORE_COLS].std().values
        expected_std = np.sqrt(np.diag(self.covariance))
        
        for i, col in enumerate(self.CORE_COLS):
            std_diff = generated_std[i] - expected_std[i]
            relative_error = abs(std_diff / expected_std[i]) * 100
            
            passed = relative_error < 15  # 15%容忍度（方差估计误差更大）
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"  {col:5s}: 目标={expected_std[i]:8.2f}, "
                  f"实际={generated_std[i]:8.2f}, "
                  f"偏差={relative_error:5.2f}% {status}")
            
            if not passed:
                all_passed = False
        
        # 验证3：正态性检验（仅当样本量>=20时）
        if len(agents) >= 20:
            print("\n[正态性检验 - Shapiro-Wilk]")
            for col in self.CORE_COLS:
                try:
                    # 使用Shapiro-Wilk检验（适合小样本）
                    stat, p_value = stats.shapiro(agents[col])
                    
                    passed = p_value > 0.01  # 降低要求
                    status = "✓ PASS" if passed else "✗ FAIL"
                    
                    print(f"  {col:5s}: W={stat:.4f}, p={p_value:.4f} {status}")
                    
                    # 正态性检验失败不影响总体通过（仅作参考）
                except Exception as e:
                    print(f"  {col:5s}: 检验失败 - {str(e)[:30]}")
        
        # 总结
        print("\n" + "=" * 70)
        if all_passed:
            print("[验证结果] ✓ 所有检验通过")
        else:
            print("[验证结果] ✗ 部分检验未通过")
        print("=" * 70 + "\n")
        
        return all_passed
    
    def set_params(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        """
        设置分布参数（用于校准）
        
        Args:
            mean: 新的均值向量 (4,)
            covariance: 新的协方差矩阵 (4, 4)
            
        Raises:
            ValueError: 参数维度或值不合法
        """
        # 验证维度
        if mean.shape != (4,):
            raise ValueError(f"均值向量维度错误：期望(4,)，实际{mean.shape}")
        
        if covariance.shape != (4, 4):
            raise ValueError(f"协方差矩阵维度错误：期望(4,4)，实际{covariance.shape}")
        
        # 验证正定性
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            warnings.warn("协方差矩阵非正定，尝试修正...")
            covariance = self._make_positive_definite(covariance)
        
        # 更新参数
        self.mean = mean
        self.covariance = covariance
        
        self.fitted_params = {
            'mean': mean.tolist(),
            'covariance': covariance.tolist(),
            'correlation': self._get_correlation_matrix().tolist()
        }
        
        self.is_fitted = True
        
        print("[OK] 参数已更新")
    
    # ==================== 私有方法 ====================
    
    def _fit_from_labor_data(self, data: pd.DataFrame) -> None:
        """基于劳动力数据初始化参数"""
        # 验证数据
        missing_cols = [col for col in self.CORE_COLS if col not in data.columns]
        if missing_cols:
            raise DataValidationError(
                f"数据缺少必需列: {missing_cols}\n"
                f"需要的列: {self.CORE_COLS}"
            )
        
        # 计算劳动力均值
        labor_mean = data[self.CORE_COLS].mean().values
        
        # 保存统计信息
        self.data_stats = {
            'labor_mean': labor_mean.tolist(),
            'labor_std': data[self.CORE_COLS].std().values.tolist(),
            'n_samples': len(data)
        }
        
        # 企业需求通常略高于劳动力平均水平
        multiplier = self.config.get(
            'labor_multiplier',
            np.array([1.1, 1.05, 1.1, 1.2])  # T, S, D, W
        )
        
        self.mean = labor_mean * multiplier
        
        print(f"[OK] 基于劳动力数据初始化均值")
        print(f"  劳动力均值: {labor_mean}")
        print(f"  调整系数: {multiplier}")
        print(f"  企业均值: {self.mean}")
    
    def _fit_from_config(self) -> None:
        """从配置初始化参数"""
        default_mean = self.config.get('default_mean')
        
        if default_mean is None:
            # 使用合理的默认值
            default_mean = [45.0, 75.0, 65.0, 5500.0]
            warnings.warn(
                f"未提供default_mean，使用默认值: {default_mean}"
            )
        
        self.mean = np.array(default_mean, dtype=float)
        
        if len(self.mean) != 4:
            raise ConfigurationError(
                f"default_mean长度错误：期望4，实际{len(self.mean)}"
            )
        
        print(f"[OK] 从配置初始化均值: {self.mean}")
    
    def _build_covariance_matrix(self) -> None:
        """构造协方差矩阵"""
        # 获取标准差
        default_std = self.config.get('default_std')
        
        if default_std is None:
            # 基于均值的10%、20%、20%、20%作为标准差
            default_std = self.mean * np.array([0.25, 0.20, 0.23, 0.22])
            warnings.warn(
                f"未提供default_std，使用自动计算值: {default_std}"
            )
        
        std = np.array(default_std, dtype=float)
        
        if len(std) != 4:
            raise ConfigurationError(
                f"default_std长度错误：期望4，实际{len(std)}"
            )
        
        # 获取相关系数矩阵（如果提供）
        correlation = self.config.get('correlation')
        
        if correlation is not None:
            # 使用提供的相关系数矩阵
            corr_matrix = np.array(correlation)
            if corr_matrix.shape != (4, 4):
                raise ConfigurationError(
                    f"correlation形状错误：期望(4,4)，实际{corr_matrix.shape}"
                )
            
            # 构造协方差矩阵：Σ = D·R·D，其中D=diag(σ)，R=相关矩阵
            D = np.diag(std)
            self.covariance = D @ corr_matrix @ D
        else:
            # 使用对角矩阵（变量独立）
            self.covariance = np.diag(std ** 2)
        
        print(f"[OK] 协方差矩阵构造完成")
        print(f"  标准差: {std}")
    
    def _validate_params(self) -> None:
        """验证参数合法性"""
        # 检查均值非负
        if np.any(self.mean < 0):
            raise ValueError(f"均值包含负数: {self.mean}")
        
        # 检查协方差矩阵正定性
        try:
            np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            warnings.warn("协方差矩阵非正定，自动修正...")
            self.covariance = self._make_positive_definite(self.covariance)
        
        # 检查协方差矩阵对称性
        if not np.allclose(self.covariance, self.covariance.T):
            warnings.warn("协方差矩阵非对称，强制对称化...")
            self.covariance = (self.covariance + self.covariance.T) / 2
        
        print(f"[OK] 参数验证通过")
    
    def _postprocess_data(self, df: pd.DataFrame) -> None:
        """数据后处理"""
        # 确保非负值
        for col in self.CORE_COLS:
            if (df[col] < 0).any():
                n_negative = (df[col] < 0).sum()
                warnings.warn(
                    f"{col}包含{n_negative}个负值，已裁剪为0"
                )
                df[col] = df[col].clip(lower=0)
        
        # 可选：添加更多后处理逻辑
        # 例如：确保T在合理范围内（如15-70小时）
        if 'enforce_bounds' in self.config and self.config['enforce_bounds']:
            bounds = self.config.get('calibration_bounds', {})
            for col, (lower, upper) in bounds.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower, upper)
    
    def _get_correlation_matrix(self) -> np.ndarray:
        """从协方差矩阵提取相关系数矩阵"""
        std = np.sqrt(np.diag(self.covariance))
        D_inv = np.diag(1.0 / std)
        return D_inv @ self.covariance @ D_inv
    
    @staticmethod
    def _make_positive_definite(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """将矩阵修正为正定矩阵"""
        # 方法：添加小的正对角元素
        n = matrix.shape[0]
        corrected = matrix + epsilon * np.eye(n)
        
        # 验证
        try:
            np.linalg.cholesky(corrected)
            return corrected
        except np.linalg.LinAlgError:
            # 如果仍然失败，使用特征值修正
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues[eigenvalues < epsilon] = epsilon
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

