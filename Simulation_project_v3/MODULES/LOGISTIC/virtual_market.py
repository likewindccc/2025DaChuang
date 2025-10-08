#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟市场生成模块

从POPULATION模块的分布参数采样生成虚拟劳动力和企业。

功能:
    - 加载劳动力分布参数（Copula + 离散分布）
    - 加载企业分布参数（多元正态分布）
    - 采样生成虚拟双边市场个体
    - 应用努力水平更新劳动力特征
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml


class VirtualMarket:
    """
    虚拟市场生成器
    
    职责:
        1. 加载劳动力和企业的分布参数
        2. 根据给定的努力水平和市场紧张度生成虚拟市场
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化虚拟市场生成器"""
        self.config = config
        np.random.seed(config['random_seed'])
        
        # 加载分布参数
        self._load_distributions()
    
    def _load_distributions(self) -> None:
        """加载劳动力和企业的分布参数"""
        # 加载劳动力分布参数
        labor_params_path = "OUTPUT/population/labor_distribution_params.pkl"
        with open(labor_params_path, 'rb') as f:
            labor_params = pickle.load(f)
        
        # 加载Copula模型对象（用于连续变量采样）
        self.labor_copula = labor_params['copula_model']
        
        # 离散变量的经验分布
        self.discrete_dist = labor_params['discrete_dist']
        
        # 读取企业分布参数（从population_config.yaml）
        pop_config_path = "CONFIG/population_config.yaml"
        with open(pop_config_path, 'r', encoding='utf-8') as f:
            pop_config = yaml.safe_load(f)
        
        ent_config = pop_config['enterprise_distribution']
        
        # 构建均值向量
        mean_dict = ent_config['mean']
        self.ent_mean = np.array([
            mean_dict['T_req'],
            mean_dict['S_req'],
            mean_dict['D_req'],
            mean_dict['W_offer']
        ])
        
        # 构建协方差矩阵
        std_dict = ent_config['std']
        std_vector = np.array([
            std_dict['T_req'],
            std_dict['S_req'],
            std_dict['D_req'],
            std_dict['W_offer']
        ])
        
        corr_dict = ent_config['correlation']
        corr_matrix = np.array([
            [1.0,              corr_dict['T_S'], corr_dict['T_D'], corr_dict['T_W']],
            [corr_dict['T_S'], 1.0,              corr_dict['S_D'], corr_dict['S_W']],
            [corr_dict['T_D'], corr_dict['S_D'], 1.0,              corr_dict['D_W']],
            [corr_dict['T_W'], corr_dict['S_W'], corr_dict['D_W'], 1.0]
        ])
        
        self.ent_cov = np.outer(std_vector, std_vector) * corr_matrix
    
    def generate_laborers(
        self,
        n_laborers: int,
        theta: float = 1.0
    ) -> pd.DataFrame:
        """
        生成虚拟劳动力
        
        Args:
            n_laborers: 劳动力数量
            theta: 市场紧张度（用于记录，Logit回归需要）
        
        Returns:
            劳动力DataFrame，包含列：id, T, S, D, W, age, edu, children, theta
            
        说明:
            - 直接从分布采样，不应用effort更新
            - effort的作用在MFG模块中通过状态更新实现
            - 匹配函数λ(x,σ,θ)不包含effort作为自变量
        """
        # 1. 从Copula采样连续变量
        continuous_samples = self.labor_copula.sample(n_laborers)
        
        # 2. 从离散分布采样
        edu_values = list(self.discrete_dist['edu'].keys())
        edu_probs = list(self.discrete_dist['edu'].values())
        edu_samples = np.random.choice(edu_values, size=n_laborers, p=edu_probs)
        
        children_values = list(self.discrete_dist['children'].keys())
        children_probs = list(self.discrete_dist['children'].values())
        children_samples = np.random.choice(
            children_values,
            size=n_laborers,
            p=children_probs
        )
        
        # 3. 构建DataFrame（使用原始采样值，不应用effort更新）
        laborers = pd.DataFrame({
            'id': np.arange(n_laborers),
            'T': continuous_samples['T'].values,
            'S': continuous_samples['S'].values,
            'D': continuous_samples['D'].values,
            'W': continuous_samples['W'].values,
            'age': continuous_samples['age'].values,
            'edu': edu_samples,
            'children': children_samples,
            'theta': theta  # 记录市场紧张度（Logit回归需要）
        })
        
        return laborers
    
    def generate_enterprises(
        self,
        n_enterprises: int,
        theta: float = 1.0
    ) -> pd.DataFrame:
        """
        生成虚拟企业
        
        Args:
            n_enterprises: 企业数量
            theta: 市场紧张度（岗位数/求职者数）
        
        Returns:
            企业DataFrame，包含列：id, T_req, S_req, D_req, W_offer, theta
        """
        # 从多元正态分布采样
        samples = np.random.multivariate_normal(
            mean=self.ent_mean,
            cov=self.ent_cov,
            size=n_enterprises
        )
        
        # 确保所有值为正（截断负值）
        samples = np.maximum(samples, 0.1)
        
        # 构建DataFrame
        enterprises = pd.DataFrame({
            'id': np.arange(n_enterprises),
            'T_req': samples[:, 0],
            'S_req': samples[:, 1],
            'D_req': samples[:, 2],
            'W_offer': samples[:, 3],
            'theta': theta  # 记录市场紧张度
        })
        
        return enterprises
    
    def generate_market(
        self,
        n_laborers: int,
        theta: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成一个完整的虚拟市场
        
        Args:
            n_laborers: 劳动力数量
            theta: 市场紧张度
        
        Returns:
            (laborers, enterprises): 劳动力和企业的DataFrame
        """
        # 根据theta计算企业数量
        n_enterprises = int(n_laborers * theta)
        
        # 生成劳动力和企业
        laborers = self.generate_laborers(n_laborers, theta)
        enterprises = self.generate_enterprises(n_enterprises, theta)
        
        return laborers, enterprises


def load_config(config_path: str = "CONFIG/logistic_config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

