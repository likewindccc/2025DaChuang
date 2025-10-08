#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POPULATION模块

人口分布模块：建立劳动力特征分布模型。

核心类:
    - LaborDistribution: 劳动力分布（基于Copula + 经验分布）

说明:
    - 企业分布使用四维正态分布假设，参数直接在配置文件中定义
    - LOGISTIC模块直接读取配置文件生成企业样本
"""

from .labor_distribution import LaborDistribution, load_config

__all__ = [
    'LaborDistribution',
    'load_config'
]

__version__ = '1.0.0'
