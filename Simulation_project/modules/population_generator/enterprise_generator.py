"""
Enterprise Generator

基于四维多元正态分布的企业主体生成器。
企业的四个关键属性：工作时长要求(T_req)、技能要求(S_req)、数字化要求(D_req)、提供薪资(W_offer)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import time
import warnings
from pathlib import Path

# 科学计算库
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

# 本地模块导入
from .base import AgentGenerator, GenerationSummary, GenerationError, ValidationError
from .config import EnterpriseGeneratorConfig, MultivariateNormalConfig
from .utils import (timer, memory_monitor, DataValidator, DistributionAnalyzer, 
                   compute_data_quality_score, create_generation_summary)
from .optimization import (fast_multivariate_normal_sample, fast_multivariate_normal_logpdf,
                          fast_cholesky_decomposition, check_numba_availability)

# 配置日志
logger = logging.getLogger(__name__)


class MultivariateNormalEngine:
    """多元正态分布引擎"""
    
    def __init__(self, config: MultivariateNormalConfig):
        """
        初始化多元正态分布引擎
        
        Args:
            config: 多元正态分布配置
        """
        self.config = config
        
        # 分布参数
        self.mean_vector = None
        self.covariance_matrix = None
        self.precision_matrix = None  # 协方差矩阵的逆
        self.cholesky_factor = None   # Cholesky分解
        self.log_det_cov = None       # 协方差矩阵的对数行列式
        
        # 拟合状态
        self.is_fitted = False
        self.fitting_stats = {}
        
        logger.info("初始化多元正态分布引擎")
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        拟合多元正态分布参数
        
        Args:
            data: 训练数据 [n_samples, 4]
            
        Returns:
            拟合结果统计
        """
        if data.shape[1] != self.config.dimensions:
            raise ValueError(f"数据维度必须为 {self.config.dimensions}")
        
        n_samples, n_features = data.shape
        
        if n_samples < n_features + 1:
            raise ValueError(f"样本数量 ({n_samples}) 必须大于特征数量 ({n_features})")
        
        logger.info(f"拟合多元正态分布，样本数: {n_samples}")
        
        # 计算均值向量
        self.mean_vector = np.mean(data, axis=0)
        
        # 计算协方差矩阵
        if self.config.regularization > 0:
            # 使用正则化协方差估计
            cov_estimator = LedoitWolf(assume_centered=False)
            cov_estimator.fit(data)
            self.covariance_matrix = cov_estimator.covariance_
            shrinkage = cov_estimator.shrinkage_
            logger.info(f"使用Ledoit-Wolf收缩，收缩系数: {shrinkage:.4f}")
        else:
            # 经验协方差矩阵
            self.covariance_matrix = np.cov(data, rowvar=False, bias=False)
        
        # 添加正则化项确保正定性
        regularization = max(self.config.regularization, 1e-8)
        self.covariance_matrix += regularization * np.eye(n_features)
        
        # 验证协方差矩阵的正定性
        try:
            self.cholesky_factor = cholesky(self.covariance_matrix, lower=True)
            self.log_det_cov = 2 * np.sum(np.log(np.diag(self.cholesky_factor)))
        except LinAlgError:
            logger.warning("协方差矩阵非正定，使用特征值修正")
            self.covariance_matrix = self._repair_covariance_matrix(self.covariance_matrix)
            self.cholesky_factor = cholesky(self.covariance_matrix, lower=True)
            self.log_det_cov = 2 * np.sum(np.log(np.diag(self.cholesky_factor)))
        
        # 计算精度矩阵（协方差矩阵的逆）
        try:
            self.precision_matrix = np.linalg.inv(self.covariance_matrix)
        except LinAlgError:
            logger.warning("协方差矩阵不可逆，使用伪逆")
            self.precision_matrix = np.linalg.pinv(self.covariance_matrix)
        
        # 计算拟合统计
        log_likelihood = self._compute_log_likelihood(data)
        aic = 2 * self._get_n_params() - 2 * log_likelihood
        bic = self._get_n_params() * np.log(n_samples) - 2 * log_likelihood
        
        self.fitting_stats = {
            'n_samples': n_samples,
            'mean_vector': self.mean_vector.copy(),
            'covariance_matrix': self.covariance_matrix.copy(),
            'eigenvalues': np.linalg.eigvals(self.covariance_matrix),
            'condition_number': np.linalg.cond(self.covariance_matrix),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'regularization_used': regularization
        }
        
        self.is_fitted = True
        logger.info("多元正态分布拟合完成")
        
        return self.fitting_stats
    
    def _repair_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """修复非正定的协方差矩阵"""
        # 特征值分解
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 将负特征值设为小正数
        min_eigenval = 1e-6
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # 重构协方差矩阵
        repaired_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return repaired_cov
    
    def _compute_log_likelihood(self, data: np.ndarray) -> float:
        """计算对数似然"""
        if check_numba_availability():
            try:
                log_probs = fast_multivariate_normal_logpdf(
                    data, self.mean_vector, self.precision_matrix, self.log_det_cov
                )
                return np.sum(log_probs)
            except:
                pass
        
        # 备选：使用scipy
        return np.sum(stats.multivariate_normal.logpdf(
            data, mean=self.mean_vector, cov=self.covariance_matrix
        ))
    
    def _get_n_params(self) -> int:
        """获取参数数量"""
        n_features = self.config.dimensions
        # 均值向量参数 + 协方差矩阵独立参数
        return n_features + n_features * (n_features + 1) // 2
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        从多元正态分布采样
        
        Args:
            n_samples: 样本数量
            random_state: 随机种子
            
        Returns:
            样本数据 [n_samples, 4]
        """
        if not self.is_fitted:
            raise RuntimeError("多元正态分布尚未拟合")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # 优先使用numba优化版本
        if check_numba_availability():
            try:
                return fast_multivariate_normal_sample(
                    n_samples, self.mean_vector, self.cholesky_factor, 
                    random_state or np.random.randint(0, 2**31)
                )
            except Exception as e:
                logger.warning(f"numba采样失败，使用备选方案: {e}")
        
        # 备选方案：使用numpy
        return np.random.multivariate_normal(
            self.mean_vector, self.covariance_matrix, n_samples
        )
    
    def probability_density(self, data: np.ndarray) -> np.ndarray:
        """计算概率密度"""
        if not self.is_fitted:
            raise RuntimeError("多元正态分布尚未拟合")
        
        # 使用numba优化版本（如果可用）
        if check_numba_availability():
            try:
                return np.exp(fast_multivariate_normal_logpdf(
                    data, self.mean_vector, self.precision_matrix, self.log_det_cov
                ))
            except:
                pass
        
        # 备选方案：使用scipy
        return stats.multivariate_normal.pdf(
            data, mean=self.mean_vector, cov=self.covariance_matrix
        )
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """获取分布参数"""
        if not self.is_fitted:
            raise RuntimeError("多元正态分布尚未拟合")
        
        return {
            'mean_vector': self.mean_vector.copy(),
            'covariance_matrix': self.covariance_matrix.copy(),
            'precision_matrix': self.precision_matrix.copy(),
            'cholesky_factor': self.cholesky_factor.copy()
        }


class ParameterCalibrator:
    """参数校准器"""
    
    def __init__(self, config: MultivariateNormalConfig):
        """
        初始化参数校准器
        
        Args:
            config: 多元正态分布配置
        """
        self.config = config
        self.calibration_history = []
    
    def calibrate_from_moments(self, 
                              target_means: np.ndarray,
                              target_covariances: np.ndarray) -> Dict[str, np.ndarray]:
        """
        基于目标矩进行参数校准
        
        Args:
            target_means: 目标均值向量
            target_covariances: 目标协方差矩阵
            
        Returns:
            校准后的参数
        """
        # 验证输入
        if len(target_means) != self.config.dimensions:
            raise ValueError(f"均值向量维度必须为 {self.config.dimensions}")
        
        if target_covariances.shape != (self.config.dimensions, self.config.dimensions):
            raise ValueError(f"协方差矩阵形状必须为 ({self.config.dimensions}, {self.config.dimensions})")
        
        # 应用边界约束
        calibrated_means = self._apply_mean_bounds(target_means)
        calibrated_covariances = self._apply_covariance_constraints(target_covariances)
        
        # 确保协方差矩阵正定
        calibrated_covariances = self._ensure_positive_definite(calibrated_covariances)
        
        # 记录校准历史
        calibration_record = {
            'timestamp': time.time(),
            'original_means': target_means.copy(),
            'calibrated_means': calibrated_means.copy(),
            'original_covariances': target_covariances.copy(),
            'calibrated_covariances': calibrated_covariances.copy(),
            'method': 'moments'
        }
        self.calibration_history.append(calibration_record)
        
        logger.info("基于矩的参数校准完成")
        
        return {
            'mean_vector': calibrated_means,
            'covariance_matrix': calibrated_covariances
        }
    
    def calibrate_from_data(self, 
                           reference_data: np.ndarray,
                           target_statistics: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        基于参考数据和目标统计量进行校准
        
        Args:
            reference_data: 参考数据
            target_statistics: 目标统计量字典
            
        Returns:
            校准后的参数
        """
        # 计算参考数据的基础统计量
        ref_means = np.mean(reference_data, axis=0)
        ref_cov = np.cov(reference_data, rowvar=False)
        
        # 根据目标统计量调整参数
        adjusted_means = ref_means.copy()
        adjusted_cov = ref_cov.copy()
        
        # 调整均值（如果提供）
        if 'target_means' in target_statistics:
            target_means = np.array(target_statistics['target_means'])
            adjusted_means = target_means
        
        # 调整方差（如果提供）
        if 'target_variances' in target_statistics:
            target_vars = np.array(target_statistics['target_variances'])
            # 保持相关性结构，只调整方差
            corr_matrix = np.corrcoef(reference_data, rowvar=False)
            std_devs = np.sqrt(target_vars)
            adjusted_cov = np.outer(std_devs, std_devs) * corr_matrix
        
        # 调整相关性（如果提供）
        if 'target_correlations' in target_statistics:
            target_corr = np.array(target_statistics['target_correlations'])
            std_devs = np.sqrt(np.diag(adjusted_cov))
            adjusted_cov = np.outer(std_devs, std_devs) * target_corr
        
        return self.calibrate_from_moments(adjusted_means, adjusted_cov)
    
    def _apply_mean_bounds(self, means: np.ndarray) -> np.ndarray:
        """应用均值边界约束"""
        calibrated_means = means.copy()
        
        feature_names = ['T_req', 'S_req', 'D_req', 'W_offer']
        
        for i, feature in enumerate(feature_names):
            if feature in self.config.mean_bounds:
                min_val, max_val = self.config.mean_bounds[feature]
                calibrated_means[i] = np.clip(calibrated_means[i], min_val, max_val)
        
        return calibrated_means
    
    def _apply_covariance_constraints(self, covariance: np.ndarray) -> np.ndarray:
        """应用协方差矩阵约束"""
        calibrated_cov = covariance.copy()
        
        # 应用最小方差约束
        for i in range(self.config.dimensions):
            if calibrated_cov[i, i] < self.config.min_variance:
                calibrated_cov[i, i] = self.config.min_variance
        
        # 应用最大相关系数约束
        for i in range(self.config.dimensions):
            for j in range(i + 1, self.config.dimensions):
                # 计算相关系数
                corr = calibrated_cov[i, j] / np.sqrt(calibrated_cov[i, i] * calibrated_cov[j, j])
                
                # 约束相关系数
                if abs(corr) > self.config.max_correlation:
                    sign = np.sign(corr)
                    max_cov = sign * self.config.max_correlation * np.sqrt(
                        calibrated_cov[i, i] * calibrated_cov[j, j]
                    )
                    calibrated_cov[i, j] = calibrated_cov[j, i] = max_cov
        
        return calibrated_cov
    
    def _ensure_positive_definite(self, covariance: np.ndarray) -> np.ndarray:
        """确保协方差矩阵正定"""
        try:
            # 尝试Cholesky分解检验正定性
            cholesky(covariance, lower=True)
            return covariance
        except LinAlgError:
            # 修复非正定矩阵
            logger.warning("协方差矩阵非正定，进行修复")
            
            # 特征值分解
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
            
            # 将负特征值设为小正数
            min_eigenval = self.config.min_variance
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # 重构正定矩阵
            repaired_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return repaired_cov


class EnterpriseGenerator(AgentGenerator):
    """
    企业主体生成器
    
    基于四维多元正态分布生成企业主体，四个属性为：
    - T_req: 工作时长要求
    - S_req: 技能要求  
    - D_req: 数字化要求
    - W_offer: 提供薪资
    """
    
    def __init__(self, config: Dict[str, Any], random_state: Optional[int] = None):
        """
        初始化企业生成器
        
        Args:
            config: 配置字典
            random_state: 随机种子
        """
        super().__init__(config, random_state)
        
        # 转换配置
        if isinstance(config, dict):
            self.enterprise_config = EnterpriseGeneratorConfig(**config.get('enterprise_config', {}))
        else:
            self.enterprise_config = config
        
        # 初始化组件
        self.mvn_engine = MultivariateNormalEngine(self.enterprise_config.mvn_config)
        self.parameter_calibrator = ParameterCalibrator(self.enterprise_config.mvn_config)
        
        # 数据验证器
        self.data_validator = DataValidator()
        
        # 拟合状态
        self.fitted_stats = {}
        
        logger.info(f"初始化企业生成器，随机种子: {random_state}")
    
    def get_required_columns(self) -> List[str]:
        """获取必需的数据列"""
        return self.enterprise_config.required_columns
    
    @timer
    @memory_monitor
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合企业生成器
        
        Args:
            data: 企业数据，包含四个必需列
        """
        logger.info("开始拟合企业生成器...")
        
        # 验证输入数据
        self._validate_input_data(data)
        
        # 数据质量检查
        validation_report = self.data_validator.validate_data_quality(
            data, self.get_required_columns(), self.enterprise_config.data_bounds
        )
        
        if not validation_report['is_valid']:
            logger.warning(f"数据质量问题: {validation_report['issues']}")
            if len(validation_report['issues']) > 2:
                raise ValidationError("数据质量问题过多，无法进行拟合")
        
        # 提取需要的列
        required_cols = self.get_required_columns()
        fitting_data = data[required_cols].dropna()
        
        if len(fitting_data) < 20:
            raise ValueError("有效数据量不足，至少需要20个观测值")
        
        logger.info(f"使用 {len(fitting_data)} 个样本进行拟合")
        
        # 拟合多元正态分布
        fitting_array = fitting_data.values
        mvn_stats = self.mvn_engine.fit(fitting_array)
        
        # 如果启用校准，可以在这里添加校准逻辑
        if self.enterprise_config.enable_calibration:
            calibration_result = self._perform_calibration(fitting_array)
            mvn_stats.update(calibration_result)
        
        # 保存拟合统计信息
        self.fitted_stats = {
            'n_samples': len(fitting_data),
            'data_shape': fitting_data.shape,
            'column_names': required_cols,
            'mvn_statistics': mvn_stats,
            'data_bounds': self.enterprise_config.data_bounds,
            'validation_report': validation_report,
            'calibration_enabled': self.enterprise_config.enable_calibration
        }
        
        # 标记为已拟合
        self.is_fitted = True
        logger.info("企业生成器拟合完成")
    
    def _perform_calibration(self, data: np.ndarray) -> Dict[str, Any]:
        """执行参数校准"""
        logger.info("执行参数校准...")
        
        # 使用默认参数作为目标
        default_means = np.array(self.enterprise_config.mvn_config.default_mean)
        default_cov = np.array(self.enterprise_config.mvn_config.default_cov_matrix)
        
        # 基于数据和默认参数的加权平均
        data_means = np.mean(data, axis=0)
        data_cov = np.cov(data, rowvar=False)
        
        # 校准权重（可配置）
        weight_data = 0.7
        weight_default = 0.3
        
        target_means = weight_data * data_means + weight_default * default_means
        target_cov = weight_data * data_cov + weight_default * default_cov
        
        # 执行校准
        calibrated_params = self.parameter_calibrator.calibrate_from_moments(
            target_means, target_cov
        )
        
        # 更新MVN引擎参数
        self.mvn_engine.mean_vector = calibrated_params['mean_vector']
        self.mvn_engine.covariance_matrix = calibrated_params['covariance_matrix']
        
        # 重新计算相关参数
        try:
            self.mvn_engine.cholesky_factor = cholesky(
                self.mvn_engine.covariance_matrix, lower=True
            )
            self.mvn_engine.precision_matrix = np.linalg.inv(
                self.mvn_engine.covariance_matrix
            )
            self.mvn_engine.log_det_cov = 2 * np.sum(
                np.log(np.diag(self.mvn_engine.cholesky_factor))
            )
        except LinAlgError as e:
            logger.error(f"校准后参数计算失败: {e}")
            raise
        
        return {
            'calibration_performed': True,
            'calibrated_means': calibrated_params['mean_vector'],
            'calibrated_covariances': calibrated_params['covariance_matrix']
        }
    
    @timer
    @memory_monitor
    def generate(self, n_agents: int, **kwargs) -> pd.DataFrame:
        """
        生成企业主体
        
        Args:
            n_agents: 企业数量
            **kwargs: 其他参数
            
        Returns:
            生成的企业DataFrame
        """
        self._check_fitted()
        
        if n_agents <= 0:
            raise ValueError("生成数量必须大于0")
        
        logger.info(f"开始生成 {n_agents} 个企业主体...")
        start_time = time.time()
        
        # 从多元正态分布采样
        random_state = kwargs.get('random_state', self.random_state)
        samples = self.mvn_engine.sample(n_agents, random_state)
        
        # 创建DataFrame
        column_names = self.get_required_columns()
        generated_data = pd.DataFrame(samples, columns=column_names)
        
        # 数据后处理
        generated_data = self._post_process_data(generated_data)
        
        # 验证生成数据
        is_valid, validation_report = self.validate(generated_data)
        if not is_valid:
            logger.warning(f"生成数据验证问题: {validation_report}")
        
        # 记录生成历史
        execution_time = time.time() - start_time
        data_quality_score = compute_data_quality_score(generated_data, validation_report)
        
        summary = GenerationSummary(
            n_agents=n_agents,
            generation_time=execution_time,
            data_quality_score=data_quality_score,
            distribution_stats=self._compute_generation_stats(generated_data),
            validation_passed=is_valid,
            memory_usage_mb=generated_data.memory_usage(deep=True).sum() / 1024 / 1024
        )
        
        self._record_generation(summary)
        
        logger.info(f"企业生成完成: {n_agents} 个主体，用时 {execution_time:.2f} 秒")
        return generated_data
    
    def _post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据后处理"""
        result = data.copy()
        
        # 应用数据边界约束
        for col, (min_val, max_val) in self.enterprise_config.data_bounds.items():
            if col in result.columns:
                # 使用clip确保数据在边界内
                result[col] = np.clip(result[col], min_val, max_val)
        
        # 数据类型和格式优化
        # W_offer (工资) 四舍五入到整数
        if 'W_offer' in result.columns:
            result['W_offer'] = result['W_offer'].round().astype(int)
        
        # T_req (工作时长) 保留1位小数
        if 'T_req' in result.columns:
            result['T_req'] = result['T_req'].round(1)
        
        # S_req 和 D_req (技能和数字化要求) 保留2位小数
        for col in ['S_req', 'D_req']:
            if col in result.columns:
                result[col] = result[col].round(2)
                # 确保在[0,1]范围内
                result[col] = np.clip(result[col], 0, 1)
        
        return result
    
    def validate(self, agents: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """验证生成数据的质量"""
        # 使用数据验证器
        validation_report = self.data_validator.validate_data_quality(
            agents, self.get_required_columns(), self.enterprise_config.data_bounds
        )
        
        # 额外的企业特定验证
        additional_checks = self._additional_validation_checks(agents)
        validation_report.update(additional_checks)
        
        # 综合判断
        is_valid = (validation_report['is_valid'] and 
                   validation_report.get('distribution_quality', 0) > 0.8)
        
        return is_valid, validation_report
    
    def _additional_validation_checks(self, agents: pd.DataFrame) -> Dict[str, Any]:
        """额外的验证检查"""
        checks = {}
        
        # 多元正态分布拟合度检验
        if hasattr(self, 'fitted_stats') and self.mvn_engine.is_fitted:
            # Mahalanobis距离检验
            generated_array = agents[self.get_required_columns()].values
            
            # 计算Mahalanobis距离
            diff = generated_array - self.mvn_engine.mean_vector
            mahalanobis_dist = np.sum(
                (diff @ self.mvn_engine.precision_matrix) * diff, axis=1
            )
            
            # Mahalanobis距离应该服从卡方分布
            expected_chi2_quantile = stats.chi2.ppf(0.95, df=4)  # 4维
            outlier_ratio = np.mean(mahalanobis_dist > expected_chi2_quantile)
            
            # 正常情况下应该有约5%的异常值
            quality_score = 1.0 - abs(outlier_ratio - 0.05) / 0.05
            checks['distribution_quality'] = max(0, min(1, quality_score))
            checks['outlier_ratio'] = outlier_ratio
            
            # 多元正态性检验（Shapiro-Wilk的多元扩展）
            if len(generated_array) <= 1000:  # 避免计算量过大
                try:
                    # 使用主成分变换后的单变量正态性检验
                    from sklearn.decomposition import PCA
                    pca = PCA()
                    pc_scores = pca.fit_transform(generated_array)
                    
                    # 对前两个主成分进行正态性检验
                    normality_pvalues = []
                    for i in range(min(2, pc_scores.shape[1])):
                        _, p_val = stats.shapiro(pc_scores[:, i])
                        normality_pvalues.append(p_val)
                    
                    checks['multivariate_normality_pvalue'] = np.mean(normality_pvalues)
                except Exception as e:
                    logger.debug(f"多元正态性检验失败: {e}")
                    checks['multivariate_normality_pvalue'] = 0.5
        
        return checks
    
    def _compute_generation_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算生成数据的统计信息"""
        stats_dict = {}
        
        for col in data.columns:
            col_data = data[col]
            stats_dict[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'skewness': float(stats.skew(col_data)),
                'kurtosis': float(stats.kurtosis(col_data))
            }
        
        # 添加相关性信息
        correlation_matrix = data.corr()
        stats_dict['correlation_matrix'] = correlation_matrix.to_dict()
        
        return stats_dict
    
    def set_parameters(self, 
                      mean_vector: np.ndarray,
                      covariance_matrix: np.ndarray) -> None:
        """
        直接设置分布参数（用于校准）
        
        Args:
            mean_vector: 均值向量
            covariance_matrix: 协方差矩阵
        """
        if len(mean_vector) != 4:
            raise ValueError("均值向量维度必须为4")
        
        if covariance_matrix.shape != (4, 4):
            raise ValueError("协方差矩阵形状必须为(4, 4)")
        
        # 使用校准器验证和调整参数
        calibrated_params = self.parameter_calibrator.calibrate_from_moments(
            mean_vector, covariance_matrix
        )
        
        # 设置MVN引擎参数
        self.mvn_engine.mean_vector = calibrated_params['mean_vector']
        self.mvn_engine.covariance_matrix = calibrated_params['covariance_matrix']
        
        # 重新计算相关参数
        try:
            self.mvn_engine.cholesky_factor = cholesky(
                self.mvn_engine.covariance_matrix, lower=True
            )
            self.mvn_engine.precision_matrix = np.linalg.inv(
                self.mvn_engine.covariance_matrix
            )
            self.mvn_engine.log_det_cov = 2 * np.sum(
                np.log(np.diag(self.mvn_engine.cholesky_factor))
            )
            
            # 标记为已拟合
            self.mvn_engine.is_fitted = True
            self.is_fitted = True
            
            logger.info("企业生成器参数设置完成")
            
        except LinAlgError as e:
            raise ValueError(f"参数设置失败，协方差矩阵问题: {e}")
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """获取当前分布参数"""
        if not self.is_fitted:
            raise RuntimeError("生成器尚未拟合")
        
        return self.mvn_engine.get_parameters()
