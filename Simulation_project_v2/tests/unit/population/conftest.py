#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Population模块测试配置和共享fixtures

作者：AI Assistant
日期：2025-10-01
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


@pytest.fixture
def sample_labor_data():
    """
    生成模拟的劳动力调研数据
    
    Returns:
        pd.DataFrame: 包含300个样本的劳动力数据
    """
    np.random.seed(42)
    n = 300
    
    # 连续变量（使用Beta分布生成归一化数据，再映射到真实范围）
    from scipy.stats import beta
    
    T_norm = beta(2, 2).rvs(n)  # 每周工作时长（归一化）
    T = T_norm * (70 - 15) + 15  # 映射到 [15, 70]
    
    S_norm = beta(1.5, 2).rvs(n)  # 工作能力评分
    S = S_norm * 100  # 映射到 [0, 100]
    
    D_norm = beta(1.2, 3).rvs(n)  # 数字素养评分
    D = D_norm * 100
    
    W_norm = beta(2, 1.5).rvs(n)  # 期望工资
    W = W_norm * (10000 - 3000) + 3000  # 映射到 [3000, 10000]
    
    age_norm = beta(2, 2).rvs(n)  # 年龄
    age = age_norm * (60 - 18) + 18  # 映射到 [18, 60]
    
    work_years_norm = beta(1.5, 3).rvs(n)  # 累计工作年限
    work_years = work_years_norm * 40  # 映射到 [0, 40]
    
    # 离散变量
    kids = np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.4, 0.2, 0.1])
    
    # 学历：0-6映射（与真实问卷一致）
    # 0:未上过学, 1:小学, 2:初中, 3:高中/中专/职高, 4:大学专科, 5:大学本科, 6:硕士及以上
    edu = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n, 
                          p=[0.02, 0.05, 0.08, 0.15, 0.25, 0.35, 0.10])
    
    # 构造DataFrame
    data = pd.DataFrame({
        'T': T,
        'S': S,
        'D': D,
        'W': W,
        '年龄': age,
        '累计工作年限': work_years,
        '孩子数量': kids,
        '学历': edu
    })
    
    return data


@pytest.fixture
def sample_config():
    """
    标准测试配置
    
    Returns:
        dict: 配置字典
    """
    return {
        'seed': 42,
        'verbose': False
    }


@pytest.fixture
def enterprise_config():
    """
    企业生成器测试配置
    
    Returns:
        dict: 企业配置字典
    """
    return {
        'seed': 43,
        'default_mean': [45.0, 75.0, 65.0, 5500.0],
        'default_std': [11.0, 15.0, 15.0, 1100.0]
    }


@pytest.fixture
def real_cleaned_data():
    """
    加载真实的清洗后数据（如果存在）
    
    Returns:
        pd.DataFrame or None: 真实数据或None
    """
    data_path = 'data/input/cleaned_data.csv'
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, encoding='utf-8-sig')
        
        # 构造复合变量
        if '每周期望工作天数' in data.columns and '每天期望工作时数' in data.columns:
            data['每周工作时长'] = data['每周期望工作天数'] * data['每天期望工作时数']
        
        # 重命名
        name_mapping = {
            '每周工作时长': 'T',
            '工作能力评分': 'S',
            '数字素养评分': 'D',
            '每月期望收入': 'W'
        }
        data = data.rename(columns=name_mapping)
        
        return data
    else:
        return None


@pytest.fixture
def mock_fitted_labor_generator(sample_labor_data):
    """
    返回已拟合的LaborGenerator
    
    Returns:
        LaborGenerator: 已拟合的生成器
    """
    from src.modules.population import LaborGenerator
    
    gen = LaborGenerator({'seed': 42})
    gen.fit(sample_labor_data)
    
    return gen


@pytest.fixture
def mock_fitted_enterprise_generator(enterprise_config):
    """
    返回已拟合的EnterpriseGenerator（配置驱动）
    
    Returns:
        EnterpriseGenerator: 已拟合的生成器
    """
    from src.modules.population import EnterpriseGenerator
    
    gen = EnterpriseGenerator(enterprise_config)
    gen.fit()
    
    return gen


@pytest.fixture(autouse=True)
def cleanup_output_files():
    """
    测试后清理输出文件
    """
    yield
    
    # 测试完成后删除临时文件
    temp_files = [
        'tests_output_labor.csv',
        'tests_output_enterprise.csv',
        'test_params.json'
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass  # 忽略删除失败

