"""
Labor Agent Generator

基于Copula模型的劳动力主体生成器，重构自现有的copula_agent_generator.py
支持多种Copula模型和边际分布，具有高度的可配置性和扩展性。
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
from sklearn.preprocessing import StandardScaler

# Copula相关库
try:
    from copulas.multivariate import GaussianMultivariate
    from copulas.multivariate.vine import VineCopula
    from copulas.univariate import Univariate
    COPULAS_AVAILABLE = True
except ImportError as e:
    COPULAS_AVAILABLE = False
    logger.warning(f"Copulas库导入失败: {e}")

# 本地模块导入
from .base import AgentGenerator, GenerationSummary, GenerationError, ValidationError
from .config import LaborGeneratorConfig, CopulaConfig
from .utils import (timer, memory_monitor, DataValidator, DistributionAnalyzer, 
                   compute_data_quality_score, create_generation_summary)
from .optimization import (fast_uniform_to_marginal, fast_correlation_to_covariance,
                          fast_cholesky_decomposition, check_numba_availability)

# 配置日志和警告
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class MarginalDistributionEstimator:
    """边际分布估计器"""
    
    def __init__(self, candidate_distributions: List[str] = None):
        """
        初始化边际分布估计器
        
        Args:
            candidate_distributions: 候选分布列表
        """
        if candidate_distributions is None:
            candidate_distributions = ['norm', 'gamma', 'beta', 'lognorm', 'uniform']
        
        self.candidate_distributions = candidate_distributions
        self.fitted_distributions = {}
        
    def fit_best_distribution(self, data: np.ndarray, column_name: str) -> Dict[str, Any]:
        """
        为单列数据拟合最佳分布
        
        Args:
            data: 一维数据数组
            column_name: 列名
            
        Returns:
            最佳分布信息字典
        """
        if len(data) < 10:
            raise ValueError(f"数据量不足，至少需要10个观测值，当前: {len(data)}")
        
        # 清理数据
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            raise ValueError("数据全部为缺失值")
        
        best_distribution = None
        best_aic = np.inf
        best_params = None
        
        logger.info(f"为列 '{column_name}' 拟合边际分布，候选分布: {self.candidate_distributions}")
        
        for dist_name in self.candidate_distributions:
            try:
                # 获取分布对象
                distribution = getattr(stats, dist_name)
                
                # 特殊处理Beta分布（需要数据在[0,1]区间）
                if dist_name == 'beta':
                    if clean_data.min() < 0 or clean_data.max() > 1:
                        # 数据标准化到[0,1]
                        data_min, data_max = clean_data.min(), clean_data.max()
                        if data_max > data_min:
                            normalized_data = (clean_data - data_min) / (data_max - data_min)
                            # 稍微收缩以避免边界问题
                            normalized_data = normalized_data * 0.999 + 0.0005
                        else:
                            continue  # 跳过常数数据
                    else:
                        normalized_data = clean_data
                    
                    # 拟合参数
                    params = distribution.fit(normalized_data)
                    
                    # 计算对数似然
                    log_likelihood = np.sum(distribution.logpdf(normalized_data, *params))
                    
                    # 如果数据被标准化，需要记录变换参数
                    if 'normalized_data' in locals() and not np.array_equal(normalized_data, clean_data):
                        params = params + (data_min, data_max)  # 添加变换参数
                
                else:
                    # 正常拟合
                    params = distribution.fit(clean_data)
                    log_likelihood = np.sum(distribution.logpdf(clean_data, *params))
                
                # 计算AIC
                n_params = len(params)
                aic = 2 * n_params - 2 * log_likelihood
                
                logger.debug(f"  {dist_name}: AIC={aic:.4f}, params={params}")
                
                # 更新最佳分布
                if aic < best_aic:
                    best_aic = aic
                    best_distribution = dist_name
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"拟合 {dist_name} 分布失败: {e}")
                continue
        
        if best_distribution is None:
            # 备选方案：使用正态分布
            logger.warning(f"所有分布拟合失败，使用正态分布作为备选")
            best_distribution = 'norm'
            best_params = (clean_data.mean(), clean_data.std())
            best_aic = np.inf
        
        result = {
            'distribution': best_distribution,
            'params': best_params,
            'aic': best_aic,
            'data_range': (clean_data.min(), clean_data.max()),
            'n_samples': len(clean_data)
        }
        
        self.fitted_distributions[column_name] = result
        logger.info(f"列 '{column_name}' 最佳分布: {best_distribution} (AIC: {best_aic:.4f})")
        
        return result
    
    def transform_to_uniform(self, data: np.ndarray, column_name: str) -> np.ndarray:
        """
        将数据转换为均匀分布
        
        Args:
            data: 原始数据
            column_name: 列名
            
        Returns:
            转换后的均匀分布数据
        """
        if column_name not in self.fitted_distributions:
            raise ValueError(f"列 '{column_name}' 尚未拟合分布")
        
        dist_info = self.fitted_distributions[column_name]
        dist_name = dist_info['distribution']
        params = dist_info['params']
        
        try:
            distribution = getattr(stats, dist_name)
            
            # 特殊处理Beta分布的逆变换
            if dist_name == 'beta' and len(params) > 2:
                # 包含变换参数的情况
                data_min, data_max = params[-2], params[-1]
                beta_params = params[:-2]
                
                # 先标准化到[0,1]
                normalized_data = (data - data_min) / (data_max - data_min)
                normalized_data = np.clip(normalized_data, 0.0005, 0.9995)
                
                # 转换为均匀分布
                uniform_data = distribution.cdf(normalized_data, *beta_params)
            else:
                # 正常转换
                uniform_data = distribution.cdf(data, *params)
            
            # 确保在[0,1]范围内
            uniform_data = np.clip(uniform_data, 1e-10, 1 - 1e-10)
            
            return uniform_data
            
        except Exception as e:
            logger.error(f"转换列 '{column_name}' 到均匀分布失败: {e}")
            # 使用经验分布函数作为备选
            return self._empirical_cdf(data)
    
    def _empirical_cdf(self, data: np.ndarray) -> np.ndarray:
        """经验分布函数"""
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        uniform_data = np.empty_like(data)
        for i, value in enumerate(data):
            rank = np.searchsorted(sorted_data, value, side='right')
            uniform_data[i] = rank / (n + 1)  # 避免0和1
        
        return uniform_data


class CopulaEngine:
    """Copula模型引擎"""
    
    def __init__(self, config: CopulaConfig):
        """
        初始化Copula引擎
        
        Args:
            config: Copula配置
        """
        self.config = config
        self.fitted_copulas = {}
        self.best_copula = None
        self.best_copula_name = None
        
        if not COPULAS_AVAILABLE:
            raise ImportError("Copulas库不可用，请安装：pip install copulas")
    
    def fit_copula_models(self, uniform_data: np.ndarray, column_names: List[str]) -> Dict[str, Any]:
        """
        拟合多个Copula模型并选择最佳模型
        
        Args:
            uniform_data: 均匀分布数据 [n_samples, n_features]
            column_names: 列名
            
        Returns:
            拟合结果字典
        """
        if uniform_data.shape[1] != len(column_names):
            raise ValueError("数据维度与列名数量不匹配")
        
        # 转换为DataFrame（copulas库要求）
        uniform_df = pd.DataFrame(uniform_data, columns=column_names)
        
        logger.info(f"开始拟合Copula模型，候选模型: {self.config.candidate_models}")
        
        best_score = -np.inf
        fitting_results = {}
        
        for model_name in self.config.candidate_models:
            try:
                logger.info(f"拟合 {model_name} Copula...")
                
                # 创建Copula模型
                if model_name == 'Gaussian':
                    copula = GaussianMultivariate()
                elif model_name in ['RegularVine', 'CenterVine', 'DirectVine']:
                    vine_type = model_name.replace('Vine', '').lower()
                    copula = VineCopula(vine_type)
                else:
                    logger.warning(f"未知的Copula模型: {model_name}")
                    continue
                
                # 拟合模型
                start_time = time.time()
                copula.fit(uniform_df)
                fit_time = time.time() - start_time
                
                # 计算拟合质量得分
                try:
                    # 计算对数似然
                    log_likelihood = copula.probability_density(uniform_df)
                    total_log_likelihood = np.sum(np.log(log_likelihood + 1e-10))
                    
                    # 计算AIC/BIC
                    n_params = self._estimate_copula_params(copula, model_name)
                    n_samples = len(uniform_df)
                    
                    aic = 2 * n_params - 2 * total_log_likelihood
                    bic = n_params * np.log(n_samples) - 2 * total_log_likelihood
                    
                    # 选择评分标准
                    if self.config.selection_criterion == 'aic':
                        score = -aic
                    elif self.config.selection_criterion == 'bic':
                        score = -bic
                    else:  # log_likelihood
                        score = total_log_likelihood
                    
                    fitting_results[model_name] = {
                        'copula': copula,
                        'log_likelihood': total_log_likelihood,
                        'aic': aic,
                        'bic': bic,
                        'score': score,
                        'n_params': n_params,
                        'fit_time': fit_time,
                        'success': True
                    }
                    
                    # 更新最佳模型
                    if score > best_score:
                        best_score = score
                        self.best_copula = copula
                        self.best_copula_name = model_name
                    
                    logger.info(f"  {model_name}: AIC={aic:.4f}, BIC={bic:.4f}, "
                              f"LogLik={total_log_likelihood:.4f}")
                
                except Exception as e:
                    logger.warning(f"{model_name} Copula拟合成功但评估失败: {e}")
                    # 仍然保存拟合结果，但不进行评估
                    fitting_results[model_name] = {
                        'copula': copula,
                        'fit_time': fit_time,
                        'success': True,
                        'evaluation_error': str(e)
                    }
                    
                    # 如果没有其他成功的模型，使用这个作为备选
                    if self.best_copula is None:
                        self.best_copula = copula
                        self.best_copula_name = model_name
                
            except Exception as e:
                logger.warning(f"拟合 {model_name} Copula失败: {e}")
                fitting_results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 检查是否有成功拟合的模型
        if self.best_copula is None:
            raise GenerationError("所有Copula模型拟合失败")
        
        logger.info(f"最佳Copula模型: {self.best_copula_name}")
        
        self.fitted_copulas = fitting_results
        return fitting_results
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        从最佳Copula模型采样
        
        Args:
            n_samples: 样本数量
            
        Returns:
            均匀分布样本 [n_samples, n_features]
        """
        if self.best_copula is None:
            raise RuntimeError("尚未拟合Copula模型")
        
        try:
            # 使用copulas库采样
            samples_df = self.best_copula.sample(n_samples)
            return samples_df.values
        
        except Exception as e:
            logger.error(f"Copula采样失败: {e}")
            # 备选方案：使用多元正态分布
            logger.warning("使用多元正态分布作为备选采样方案")
            return self._fallback_gaussian_sample(n_samples)
    
    def _estimate_copula_params(self, copula, model_name: str) -> int:
        """估计Copula模型的参数数量"""
        if model_name == 'Gaussian':
            # 高斯Copula：相关矩阵的独立参数
            n_vars = len(copula.columns)
            return n_vars * (n_vars - 1) // 2
        elif 'Vine' in model_name:
            # Vine Copula：更复杂的参数结构
            n_vars = len(copula.columns) if hasattr(copula, 'columns') else 4
            return n_vars * (n_vars - 1)  # 简化估计
        else:
            return 1  # 默认值
    
    def _fallback_gaussian_sample(self, n_samples: int) -> np.ndarray:
        """备选的高斯Copula采样"""
        n_features = 4  # 默认特征数
        
        # 生成相关矩阵
        rho = 0.3  # 默认相关系数
        correlation_matrix = np.eye(n_features)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                correlation_matrix[i, j] = correlation_matrix[j, i] = rho
        
        # 多元正态分布采样
        mvn_samples = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_samples
        )
        
        # 转换为均匀分布
        uniform_samples = stats.norm.cdf(mvn_samples)
        
        return uniform_samples


class LaborAgentGenerator(AgentGenerator):
    """
    劳动力主体生成器
    
    基于Copula模型生成具有复杂相关性的虚拟劳动力个体。
    """
    
    def __init__(self, config: Dict[str, Any], random_state: Optional[int] = None):
        """
        初始化劳动力生成器
        
        Args:
            config: 配置字典，将转换为LaborGeneratorConfig
            random_state: 随机种子
        """
        super().__init__(config, random_state)
        
        # 转换配置
        if isinstance(config, dict):
            self.labor_config = LaborGeneratorConfig(**config.get('labor_config', {}))
        else:
            self.labor_config = config
        
        # 初始化组件
        self.marginal_estimator = MarginalDistributionEstimator(
            self.labor_config.copula_config.marginal_distributions
        )
        self.copula_engine = CopulaEngine(self.labor_config.copula_config)
        
        # 数据验证器
        self.data_validator = DataValidator()
        
        # 拟合状态
        self.fitted_data_stats = {}
        
        logger.info(f"初始化劳动力生成器，随机种子: {random_state}")
    
    def get_required_columns(self) -> List[str]:
        """获取必需的数据列"""
        return self.labor_config.required_columns
    
    @timer
    @memory_monitor
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合生成器模型
        
        Args:
            data: 用于拟合的真实劳动力数据
        """
        logger.info("开始拟合劳动力生成器...")
        
        # 验证输入数据
        self._validate_input_data(data)
        
        # 数据质量检查
        validation_report = self.data_validator.validate_data_quality(
            data, self.get_required_columns(), self.labor_config.data_bounds
        )
        
        if not validation_report['is_valid']:
            logger.warning(f"数据质量问题: {validation_report['issues']}")
            if len(validation_report['issues']) > 2:
                raise ValidationError("数据质量问题过多，无法进行拟合")
        
        # 提取需要的列
        required_cols = self.get_required_columns()
        fitting_data = data[required_cols].dropna()
        
        if len(fitting_data) < 50:
            raise ValueError("有效数据量不足，至少需要50个观测值")
        
        logger.info(f"使用 {len(fitting_data)} 个样本进行拟合")
        
        # Step 1: 拟合边际分布
        logger.info("Step 1: 拟合边际分布...")
        uniform_data = np.empty_like(fitting_data.values)
        
        for i, col in enumerate(required_cols):
            col_data = fitting_data[col].values
            
            # 拟合最佳分布
            self.marginal_estimator.fit_best_distribution(col_data, col)
            
            # 转换为均匀分布
            uniform_data[:, i] = self.marginal_estimator.transform_to_uniform(col_data, col)
        
        # Step 2: 拟合Copula模型
        logger.info("Step 2: 拟合Copula模型...")
        copula_results = self.copula_engine.fit_copula_models(uniform_data, required_cols)
        
        # 保存拟合统计信息
        self.fitted_data_stats = {
            'n_samples': len(fitting_data),
            'data_shape': fitting_data.shape,
            'marginal_distributions': self.marginal_estimator.fitted_distributions,
            'copula_results': copula_results,
            'best_copula': self.copula_engine.best_copula_name,
            'data_bounds': self.labor_config.data_bounds,
            'validation_report': validation_report
        }
        
        # 标记为已拟合
        self.is_fitted = True
        logger.info("劳动力生成器拟合完成")
    
    @timer
    @memory_monitor
    def generate(self, n_agents: int, **kwargs) -> pd.DataFrame:
        """
        生成虚拟劳动力主体
        
        Args:
            n_agents: 生成的主体数量
            **kwargs: 其他参数
            
        Returns:
            生成的劳动力DataFrame
        """
        self._check_fitted()
        
        if n_agents <= 0:
            raise ValueError("生成数量必须大于0")
        
        logger.info(f"开始生成 {n_agents} 个劳动力主体...")
        start_time = time.time()
        
        # 检查是否需要批量生成
        batch_size = kwargs.get('batch_size', self.labor_config.batch_size)
        
        if n_agents > batch_size:
            return self._generate_in_batches(n_agents, batch_size, **kwargs)
        
        # 单批次生成
        generated_data = self._generate_single_batch(n_agents)
        
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
        
        logger.info(f"劳动力生成完成: {n_agents} 个主体，用时 {execution_time:.2f} 秒")
        return generated_data
    
    def _generate_single_batch(self, n_agents: int) -> pd.DataFrame:
        """生成单批次数据"""
        required_cols = self.get_required_columns()
        
        # Step 1: 从Copula采样均匀分布数据
        uniform_samples = self.copula_engine.sample(n_agents)
        
        # Step 2: 转换为边际分布
        generated_data = np.empty_like(uniform_samples)
        
        for i, col in enumerate(required_cols):
            # 获取分布信息
            dist_info = self.marginal_estimator.fitted_distributions[col]
            dist_name = dist_info['distribution']
            params = dist_info['params']
            
            # 使用numba优化的转换（如果可用）
            if check_numba_availability():
                # 编码分布类型
                dist_type_map = {'norm': 0, 'gamma': 1, 'beta': 2, 'lognorm': 3}
                dist_type = dist_type_map.get(dist_name, 0)
                
                try:
                    generated_data[:, i] = fast_uniform_to_marginal(
                        uniform_samples[:, i], 
                        np.array(params[:2]),  # 只取前两个参数
                        dist_type
                    )
                except:
                    # 备选方案：使用scipy
                    generated_data[:, i] = self._scipy_inverse_transform(
                        uniform_samples[:, i], dist_name, params
                    )
            else:
                # 使用scipy进行逆变换
                generated_data[:, i] = self._scipy_inverse_transform(
                    uniform_samples[:, i], dist_name, params
                )
        
        # 创建DataFrame
        result_df = pd.DataFrame(generated_data, columns=required_cols)
        
        # 数据后处理
        result_df = self._post_process_data(result_df)
        
        return result_df
    
    def _scipy_inverse_transform(self, uniform_data: np.ndarray, 
                                dist_name: str, params: tuple) -> np.ndarray:
        """使用scipy进行逆变换"""
        try:
            distribution = getattr(stats, dist_name)
            
            # 特殊处理Beta分布
            if dist_name == 'beta' and len(params) > 2:
                # 包含变换参数的情况
                data_min, data_max = params[-2], params[-1]
                beta_params = params[:-2]
                
                # 先进行Beta逆变换
                beta_samples = distribution.ppf(uniform_data, *beta_params)
                
                # 然后逆变换到原始范围
                return beta_samples * (data_max - data_min) + data_min
            else:
                # 正常逆变换
                return distribution.ppf(uniform_data, *params)
        
        except Exception as e:
            logger.warning(f"逆变换失败 ({dist_name}): {e}，使用线性变换")
            # 备选：线性变换到合理范围
            data_bounds = self.labor_config.data_bounds
            col_bounds = list(data_bounds.values())[0]  # 使用第一个边界作为默认
            return uniform_data * (col_bounds[1] - col_bounds[0]) + col_bounds[0]
    
    def _generate_in_batches(self, total_agents: int, batch_size: int, **kwargs) -> pd.DataFrame:
        """分批次生成大量数据"""
        batches = []
        remaining = total_agents
        batch_num = 0
        
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            batch_num += 1
            
            logger.info(f"生成第 {batch_num} 批，数量: {current_batch_size}")
            
            batch_data = self._generate_single_batch(current_batch_size)
            batches.append(batch_data)
            
            remaining -= current_batch_size
        
        # 合并所有批次
        result = pd.concat(batches, ignore_index=True)
        return result
    
    def _post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据后处理：边界检查和修正"""
        result = data.copy()
        
        # 应用数据边界约束
        for col, (min_val, max_val) in self.labor_config.data_bounds.items():
            if col in result.columns:
                # 使用clip确保数据在边界内
                result[col] = np.clip(result[col], min_val, max_val)
        
        # 数据类型优化
        for col in result.columns:
            if col in ['age', 'education']:
                result[col] = result[col].round().astype(int)
        
        return result
    
    def validate(self, agents: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """验证生成数据的质量"""
        # 使用数据验证器
        validation_report = self.data_validator.validate_data_quality(
            agents, self.get_required_columns(), self.labor_config.data_bounds
        )
        
        # 额外的生成特定验证
        additional_checks = self._additional_validation_checks(agents)
        validation_report.update(additional_checks)
        
        # 综合判断
        is_valid = (validation_report['is_valid'] and 
                   validation_report.get('distribution_similarity', 0) > 0.7)
        
        return is_valid, validation_report
    
    def _additional_validation_checks(self, agents: pd.DataFrame) -> Dict[str, Any]:
        """额外的验证检查"""
        checks = {}
        
        # 分布相似性检查
        if hasattr(self, 'fitted_data_stats'):
            similarity_scores = []
            
            for col in self.get_required_columns():
                if col in agents.columns:
                    # 比较生成数据与原始数据的分布
                    generated_values = agents[col].values
                    
                    # KS检验
                    original_dist_info = self.fitted_data_stats['marginal_distributions'][col]
                    dist_name = original_dist_info['distribution']
                    params = original_dist_info['params']
                    
                    try:
                        distribution = getattr(stats, dist_name)
                        ks_stat, p_value = stats.kstest(
                            generated_values, 
                            lambda x: distribution.cdf(x, *params)
                        )
                        
                        # p值越大，分布越相似
                        similarity_scores.append(p_value)
                        
                    except Exception:
                        similarity_scores.append(0.5)  # 默认中等相似度
            
            checks['distribution_similarity'] = np.mean(similarity_scores) if similarity_scores else 0.5
        
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
        
        return stats_dict
