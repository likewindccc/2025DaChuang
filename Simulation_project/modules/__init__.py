"""
EconLab模块包初始化
提供核心计算模块的统一入口
"""

__version__ = "1.0.0"
__author__ = "EconLab Development Team"

# 模块导入
from .population_generator import AgentGenerator, LaborAgentGenerator, EnterpriseGenerator

__all__ = [
    'AgentGenerator',
    'LaborAgentGenerator', 
    'EnterpriseGenerator'
]
