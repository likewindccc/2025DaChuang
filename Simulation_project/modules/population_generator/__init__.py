"""
Population Generator Module

该模块负责生成虚拟主体池：
- 劳动力主体：基于Copula模型生成具有复杂相关性的虚拟个体
- 企业主体：基于四维多元正态分布生成企业池

主要组件：
- AgentGenerator: 抽象基类，定义统一接口
- LaborAgentGenerator: 劳动力主体生成器
- EnterpriseGenerator: 企业主体生成器
"""

from .base import AgentGenerator
from .labor_generator import LaborAgentGenerator
from .enterprise_generator import EnterpriseGenerator
from .config import PopulationConfig

__version__ = "1.0.0"

__all__ = [
    'AgentGenerator',
    'LaborAgentGenerator',
    'EnterpriseGenerator', 
    'PopulationConfig'
]
