#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
劳动力分布模块

分开建模劳动力特征的联合分布：
    - 连续变量 (T, S, D, W, age): 使用Gaussian Copula
    - 离散变量 (edu, children): 记录经验分布（频率分布）

功能:
    - 从预处理数据拟合连续和离散分布
    - 保存参数供LOGISTIC模块使用
    - LOGISTIC模块负责加载参数并采样
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any
import yaml

from copulas.multivariate import GaussianMultivariate


class LaborDistribution:
    """
    劳动力分布类
    
    职责:
        1. 连续变量 (T, S, D, W, age) 用Gaussian Copula建模
        2. 离散变量 (edu, children) 记录经验分布
        3. 保存两部分参数到文件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化劳动力分布"""
        self.config = config
        self.copula_model = None  # 连续变量的Copula模型
        self.discrete_dist = None  # 离散变量的经验分布
        np.random.seed(config['random_seed'])
    
    def _load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            包含所有变量的DataFrame
        """
        # 读取数据文件
        data_path = self.config['labor_distribution']['data_path']
        df = pd.read_csv(data_path, encoding='utf-8')
        
        # 获取变量配置
        variables = self.config['labor_distribution']['variables']
        
        # 计算T: 工作时长 = 每周工作天数 × 每天工作小时数
        col1, col2 = variables['T']
        df['T'] = df[col1] * df[col2]
        
        # 提取其他核心变量
        df['S'] = df[variables['S']]
        df['D'] = df[variables['D']]
        df['W'] = df[variables['W']]
        
        # 提取控制变量
        df['age'] = df['年龄']
        df['edu'] = df['学历']
        df['children'] = df['孩子数量']
        
        return df[['T', 'S', 'D', 'W', 'age', 'edu', 'children']]
    
    def fit(self) -> None:
        """
        拟合分布
        
        步骤:
            1. 加载数据
            2. 连续变量 (T, S, D, W, age) 用Copula拟合
            3. 离散变量 (edu, children) 记录频率分布
        """
        # 加载数据
        data = self._load_data()
        
        # 1. 拟合连续变量的Copula
        continuous_vars = ['T', 'S', 'D', 'W', 'age']
        self.copula_model = GaussianMultivariate()
        self.copula_model.fit(data[continuous_vars])
        
        # 2. 记录离散变量的经验分布（频率分布）
        self.discrete_dist = {
            'edu': data['edu'].value_counts(normalize=True).to_dict(),
            'children': data['children'].value_counts(normalize=True).to_dict()
        }
    
    def save_params(self) -> None:
        """
        保存参数到文件（硬编码路径）
        
        保存内容:
            - 连续变量的Copula参数
            - 离散变量的经验分布
        """
        # 硬编码保存路径
        filepath = "OUTPUT/population/labor_distribution_params.pkl"
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 打包所有参数
        params = {
            'copula_params': self.copula_model.to_dict(),
            'discrete_dist': self.discrete_dist
        }
        
        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)


def load_config(config_path: str = "CONFIG/population_config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
