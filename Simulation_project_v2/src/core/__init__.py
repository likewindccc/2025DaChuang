#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core模块 - 项目核心基础设施

提供所有模块共享的核心组件，包括：
- 数据结构：Agent, MatchingPair, SimulationState, MFGEquilibrium
- 基础类：BaseGenerator, BaseSimulator
- 异常：7个细粒度异常类
- 类型：类型别名和类型检查工具

模块架构:
    core/
    ├── data_structures.py   - 核心数据结构
    ├── exceptions.py        - 异常体系
    ├── types.py             - 类型定义
    ├── base_generator.py    - 生成器基类
    ├── base_simulator.py    - 模拟器基类
    └── __init__.py          - 公共接口（本文件）

使用示例:
    >>> from src.core import (
    ...     Agent,
    ...     MatchingPair,
    ...     BaseGenerator,
    ...     DataValidationError,
    ...     AgentFeatures
    ... )
    >>> 
    >>> # 创建Agent
    >>> labor = Agent(
    ...     agent_id=1,
    ...     agent_type='labor',
    ...     T=40.0, S=75.0, D=60.0, W=4500.0
    ... )
    >>> 
    >>> # 转换为NumPy数组
    >>> features: AgentFeatures = labor.to_array()
    >>> 
    >>> # 异常处理
    >>> if not (0 <= labor.S <= 100):
    ...     raise DataValidationError("技能评分超出范围")

版本历史:
    v0.1.0 (2025-09-30) - 初始版本
        - 实现4个核心数据结构
        - 实现7个异常类
        - 实现类型系统
        - 实现BaseGenerator和BaseSimulator
"""

__version__ = '0.1.0'
__author__ = 'Simulation Team'
__date__ = '2025-09-30'


# ============================================================================
# 数据结构
# ============================================================================

from .data_structures import (
    Agent,
    MatchingPair,
    SimulationState,
    MFGEquilibrium,
)


# ============================================================================
# 基础类
# ============================================================================

from .base_generator import BaseGenerator
from .base_simulator import BaseSimulator


# ============================================================================
# 异常
# ============================================================================

from .exceptions import (
    SimulationError,
    DataValidationError,
    CopulaFittingError,
    MatchingError,
    ConvergenceError,
    ConfigurationError,
    CalibrationError,
    get_exception_class,
    EXCEPTION_MAP,
)


# ============================================================================
# 类型
# ============================================================================

from .types import (
    # 基础类型
    AgentID,
    TimeStep,
    ParameterDict,
    
    # NumPy数组类型
    AgentFeatures,
    PreferenceMatrix,
    ValueFunction,
    PolicyFunction,
    Distribution,
    CovarianceMatrix,
    CorrelationMatrix,
    
    # 函数类型
    ObjectiveFunction,
    MatchFunction,
    UtilityFunction,
    
    # 配置类型
    Config,
    ExperimentConfig,
    
    # 复合类型
    AgentPool,
    MatchingResult,
    SimulationMetrics,
    
    # 类型检查工具
    is_valid_agent_features,
    is_valid_distribution,
    is_symmetric_matrix,
)


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__date__',
    
    # 数据结构
    'Agent',
    'MatchingPair',
    'SimulationState',
    'MFGEquilibrium',
    
    # 基础类
    'BaseGenerator',
    'BaseSimulator',
    
    # 异常
    'SimulationError',
    'DataValidationError',
    'CopulaFittingError',
    'MatchingError',
    'ConvergenceError',
    'ConfigurationError',
    'CalibrationError',
    'get_exception_class',
    'EXCEPTION_MAP',
    
    # 基础类型
    'AgentID',
    'TimeStep',
    'ParameterDict',
    
    # NumPy数组类型
    'AgentFeatures',
    'PreferenceMatrix',
    'ValueFunction',
    'PolicyFunction',
    'Distribution',
    'CovarianceMatrix',
    'CorrelationMatrix',
    
    # 函数类型
    'ObjectiveFunction',
    'MatchFunction',
    'UtilityFunction',
    
    # 配置类型
    'Config',
    'ExperimentConfig',
    
    # 复合类型
    'AgentPool',
    'MatchingResult',
    'SimulationMetrics',
    
    # 类型检查工具
    'is_valid_agent_features',
    'is_valid_distribution',
    'is_symmetric_matrix',
]


# ============================================================================
# 模块级文档
# ============================================================================

def get_module_info() -> dict:
    """
    获取Core模块信息
    
    Returns:
        模块信息字典
    
    Examples:
        >>> from src.core import get_module_info
        >>> info = get_module_info()
        >>> print(f"Core模块版本: {info['version']}")
        Core模块版本: 0.1.0
    """
    return {
        'name': 'core',
        'version': __version__,
        'author': __author__,
        'date': __date__,
        'components': {
            'data_structures': ['Agent', 'MatchingPair', 'SimulationState', 'MFGEquilibrium'],
            'base_classes': ['BaseGenerator', 'BaseSimulator'],
            'exceptions': [
                'SimulationError', 'DataValidationError', 'CopulaFittingError',
                'MatchingError', 'ConvergenceError', 'ConfigurationError', 'CalibrationError'
            ],
            'types': [
                'AgentFeatures', 'PreferenceMatrix', 'ValueFunction', 'Distribution',
                'ObjectiveFunction', 'MatchFunction', 'Config'
            ]
        }
    }


def print_module_summary():
    """
    打印Core模块摘要
    
    Examples:
        >>> from src.core import print_module_summary
        >>> print_module_summary()
        ========================================
        Core模块 v0.1.0
        ========================================
        数据结构: 4个
        基础类:   2个
        异常:     7个
        类型:     20+个
        ========================================
    """
    print("=" * 70)
    print(f"Core模块 v{__version__}")
    print("=" * 70)
    print(f"作者:     {__author__}")
    print(f"日期:     {__date__}")
    print("-" * 70)
    print("组件统计:")
    print("  - 数据结构: 4个 (Agent, MatchingPair, SimulationState, MFGEquilibrium)")
    print("  - 基础类:   2个 (BaseGenerator, BaseSimulator)")
    print("  - 异常:     7个 (SimulationError及6个子类)")
    print("  - 类型:     20+个 (AgentFeatures, PreferenceMatrix, Config等)")
    print("=" * 70)


# ============================================================================
# 初始化检查
# ============================================================================

def _run_integrity_check():
    """运行完整性检查"""
    try:
        # 检查所有导出是否可用
        for name in __all__:
            if name.startswith('__'):
                continue
            if name not in globals():
                raise ImportError(f"导出的符号 '{name}' 不可用")
        
        # 检查数据结构
        test_agent = Agent(1, 'labor', 40.0, 75.0, 60.0, 4500.0)
        assert test_agent.to_array().shape == (4,)
        
        # 检查异常
        assert issubclass(DataValidationError, SimulationError)
        
        return True
    
    except Exception as e:
        print(f"[警告] Core模块完整性检查失败: {e}")
        return False


# 在模块加载时运行完整性检查
_integrity_ok = _run_integrity_check()

if not _integrity_ok:
    print("[警告] Core模块可能未正确初始化，请检查依赖")