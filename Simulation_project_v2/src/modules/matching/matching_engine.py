"""
匹配引擎

集成偏好计算和Gale-Shapley算法，提供高层接口用于批量匹配模拟。
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging

from .preference import (
    compute_labor_preference_matrix,
    compute_enterprise_preference_matrix,
    compute_preference_rankings
)
from .gale_shapley import (
    gale_shapley,
    verify_stability,
    compute_matching_statistics
)
from .matching_result import MatchingResult


logger = logging.getLogger(__name__)


class MatchingEngine:
    """
    匹配引擎：整合偏好计算和GS算法
    
    提供高层接口用于：
    - 单轮匹配
    - 批量模拟
    - 参数调整
    - 结果统计
    """
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        初始化匹配引擎
        
        Args:
            config: 配置字典
            config_path: 配置文件路径（YAML）
        """
        if config is None and config_path is None:
            # 使用默认配置
            default_config_path = Path(__file__).parent.parent.parent.parent / "config" / "default" / "matching.yaml"
            if default_config_path.exists():
                config_path = str(default_config_path)
            else:
                config = self._default_config()
        
        if config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        # 提取偏好参数
        self.labor_pref_params = self.config.get('preference', {}).get('labor', {})
        self.enterprise_pref_params = self.config.get('preference', {}).get('enterprise', {})
        
        # 算法配置
        self.algorithm_config = self.config.get('algorithm', {})
        
        # 验证配置
        self.validation_config = self.config.get('validation', {})
        
        logger.info("匹配引擎初始化完成")
    
    @staticmethod
    def _default_config() -> Dict:
        """返回默认配置"""
        return {
            'preference': {
                'labor': {
                    'gamma_0': 1.0,
                    'gamma_1': 0.01,
                    'gamma_2': 0.5,
                    'gamma_3': 0.5,
                    'gamma_4': 0.001
                },
                'enterprise': {
                    'beta_0': 0.0,
                    'beta_1': 0.5,
                    'beta_2': 1.0,
                    'beta_3': 1.0,
                    'beta_4': -0.001
                }
            },
            'algorithm': {
                'method': 'gale_shapley',
                'proposer': 'labor'
            },
            'validation': {
                'verify_stability': True,
                'compute_quality': True
            }
        }
    
    def match(
        self,
        labor_agents: pd.DataFrame,
        enterprise_agents: pd.DataFrame,
        verify_stability: Optional[bool] = None
    ) -> MatchingResult:
        """
        执行单轮匹配
        
        Args:
            labor_agents: 劳动力DataFrame，必须包含列 [T, S, D, W]
            enterprise_agents: 企业DataFrame，必须包含列 [T, S, D, W]
            verify_stability: 是否验证稳定性（None则使用配置）
        
        Returns:
            MatchingResult对象
        """
        # 数据验证
        self._validate_dataframes(labor_agents, enterprise_agents)
        
        # 提取特征
        labor_features = labor_agents[['T', 'S', 'D', 'W']].values.astype(np.float32)
        enterprise_features = enterprise_agents[['T', 'S', 'D', 'W']].values.astype(np.float32)
        
        logger.info(f"开始匹配：{len(labor_features)}个劳动力 × {len(enterprise_features)}个企业")
        
        # Step 1: 计算偏好矩阵
        labor_pref = compute_labor_preference_matrix(
            labor_features,
            enterprise_features,
            **self.labor_pref_params
        )
        
        enterprise_pref = compute_enterprise_preference_matrix(
            enterprise_features,
            labor_features,
            **self.enterprise_pref_params
        )
        
        logger.debug("偏好矩阵计算完成")
        
        # Step 2: 转换为偏好排序
        labor_pref_order = compute_preference_rankings(labor_pref)
        enterprise_pref_order = compute_preference_rankings(enterprise_pref)
        
        # Step 3: 执行GS算法
        matching = gale_shapley(labor_pref_order, enterprise_pref_order)
        
        logger.info(f"匹配完成：{np.sum(matching != -1)}个成功匹配")
        
        # Step 4: 验证稳定性
        if verify_stability is None:
            verify_stability = self.validation_config.get('verify_stability', True)
        
        if verify_stability:
            from .gale_shapley import verify_stability as verify_fn
            is_stable, unstable_pairs = verify_fn(matching, labor_pref_order, enterprise_pref_order)
        else:
            is_stable, unstable_pairs = True, []
        
        # Step 5: 计算统计信息
        statistics = compute_matching_statistics(matching, labor_features, enterprise_features)
        
        # Step 6: 构造结果对象
        result = MatchingResult(
            labor_agents=labor_agents,
            enterprise_agents=enterprise_agents,
            matching=matching,
            labor_preference=labor_pref,
            enterprise_preference=enterprise_pref,
            is_stable=is_stable,
            unstable_pairs=unstable_pairs,
            statistics=statistics
        )
        
        return result
    
    def batch_match(
        self,
        labor_agents_list: List[pd.DataFrame],
        enterprise_agents_list: List[pd.DataFrame],
        verbose: bool = False
    ) -> List[MatchingResult]:
        """
        批量匹配（多个场景）
        
        Args:
            labor_agents_list: 劳动力DataFrame列表
            enterprise_agents_list: 企业DataFrame列表
            verbose: 是否显示详细进度
        
        Returns:
            MatchingResult列表
        """
        if len(labor_agents_list) != len(enterprise_agents_list):
            raise ValueError("劳动力和企业列表长度必须相同")
        
        results = []
        n_scenarios = len(labor_agents_list)
        
        logger.info(f"开始批量匹配：{n_scenarios}个场景")
        
        for i, (labor, enterprise) in enumerate(zip(labor_agents_list, enterprise_agents_list)):
            if verbose:
                print(f"\r批量匹配进度: {i+1}/{n_scenarios}", end='', flush=True)
            
            result = self.match(labor, enterprise)
            results.append(result)
        
        if verbose:
            print()  # 换行
        
        logger.info(f"批量匹配完成：{n_scenarios}个场景")
        
        return results
    
    def update_preference_params(
        self,
        labor_params: Optional[Dict] = None,
        enterprise_params: Optional[Dict] = None
    ):
        """
        更新偏好函数参数
        
        Args:
            labor_params: 劳动力偏好参数（部分或全部）
            enterprise_params: 企业偏好参数（部分或全部）
        """
        if labor_params is not None:
            self.labor_pref_params.update(labor_params)
            logger.info(f"更新劳动力偏好参数: {labor_params}")
        
        if enterprise_params is not None:
            self.enterprise_pref_params.update(enterprise_params)
            logger.info(f"更新企业偏好参数: {enterprise_params}")
    
    def _validate_dataframes(
        self,
        labor_agents: pd.DataFrame,
        enterprise_agents: pd.DataFrame
    ):
        """验证输入DataFrame的格式"""
        required_cols = ['T', 'S', 'D', 'W']
        
        # 检查劳动力DataFrame
        missing_labor = set(required_cols) - set(labor_agents.columns)
        if missing_labor:
            raise ValueError(f"劳动力DataFrame缺少列: {missing_labor}")
        
        # 检查企业DataFrame
        missing_enterprise = set(required_cols) - set(enterprise_agents.columns)
        if missing_enterprise:
            raise ValueError(f"企业DataFrame缺少列: {missing_enterprise}")
        
        # 检查数据类型
        for col in required_cols:
            if not np.issubdtype(labor_agents[col].dtype, np.number):
                raise ValueError(f"劳动力DataFrame的列'{col}'必须是数值类型")
            if not np.issubdtype(enterprise_agents[col].dtype, np.number):
                raise ValueError(f"企业DataFrame的列'{col}'必须是数值类型")
        
        # 检查空值
        if labor_agents[required_cols].isnull().any().any():
            raise ValueError("劳动力DataFrame包含空值")
        if enterprise_agents[required_cols].isnull().any().any():
            raise ValueError("企业DataFrame包含空值")
    
    def compute_batch_statistics(self, results: List[MatchingResult]) -> Dict:
        """
        计算批量匹配的汇总统计
        
        Args:
            results: MatchingResult列表
        
        Returns:
            汇总统计字典
        """
        if not results:
            return {}
        
        # 提取各场景的统计指标
        match_rates = [r.statistics['match_rate'] for r in results]
        unemployment_rates = [r.statistics['unemployment_rate'] for r in results]
        stability_rates = [1.0 if r.is_stable else 0.0 for r in results]
        
        # 计算匹配质量
        qualities = [r.compute_match_quality() for r in results]
        avg_labor_satisfaction = np.mean([q['avg_labor_satisfaction'] for q in qualities])
        avg_enterprise_satisfaction = np.mean([q['avg_enterprise_satisfaction'] for q in qualities])
        
        summary = {
            'n_scenarios': len(results),
            'avg_match_rate': float(np.mean(match_rates)),
            'std_match_rate': float(np.std(match_rates)),
            'avg_unemployment_rate': float(np.mean(unemployment_rates)),
            'std_unemployment_rate': float(np.std(unemployment_rates)),
            'stability_rate': float(np.mean(stability_rates)),
            'avg_labor_satisfaction': float(avg_labor_satisfaction),
            'avg_enterprise_satisfaction': float(avg_enterprise_satisfaction)
        }
        
        return summary
    
    def __repr__(self) -> str:
        return f"MatchingEngine(algorithm={self.algorithm_config.get('method', 'gale_shapley')})"

