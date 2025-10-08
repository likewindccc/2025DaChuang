#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOGISTIC模块

匹配与匹配函数构建模块。

核心类:
    - VirtualMarket: 虚拟市场生成器

核心函数:
    - perform_matching: 执行GS匹配算法

说明:
    - 从POPULATION模块的分布参数生成虚拟劳动力和企业
    - 执行GS匹配算法
    - 回归构建匹配函数
"""

from .virtual_market import VirtualMarket, load_config
from .gs_matching import perform_matching
from .match_function import MatchFunction

__all__ = [
    'VirtualMarket',
    'load_config',
    'perform_matching',
    'MatchFunction'
]

__version__ = '1.0.0'

