#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation Modules - 仿真核心模块

包含所有核心仿真模块：
- population: 虚拟人口生成
- matching: 匹配引擎
- estimation: 参数估计
- mfg: 平均场博弈求解器
- calibration: 模型校准

作者：AI Assistant
日期：2025-10-01
"""

from . import population

__all__ = [
    'population',
    # 'matching',     # 待实现
    # 'estimation',   # 待实现
    # 'mfg',          # 待实现
    # 'calibration',  # 待实现
]
