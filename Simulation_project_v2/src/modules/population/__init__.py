#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Population模块 - 虚拟人口生成

负责生成虚拟劳动力和企业个体，用于匹配模拟和MFG求解。

主要组件：
- LaborGenerator: 劳动力生成器（6维Copula + 离散变量条件抽样）
- EnterpriseGenerator: 企业生成器（四维正态分布）

作者：AI Assistant
日期：2025-10-01
版本：v2.0
"""

from .labor_generator import LaborGenerator
from .enterprise_generator import EnterpriseGenerator

__all__ = [
    'LaborGenerator',
    'EnterpriseGenerator',
]

__version__ = '2.0.0'
__author__ = 'AI Assistant'
