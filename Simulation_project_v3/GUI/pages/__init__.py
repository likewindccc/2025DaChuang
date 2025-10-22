"""
页面模块

包含3个主要页面：
- ConfigPage: 参数配置页
- SimulationPage: 仿真运行页
- ResultsPage: 结果分析页
"""

from .config_page import ConfigPage
from .simulation_page import SimulationPage
from .results_page import ResultsPage

__all__ = ['ConfigPage', 'SimulationPage', 'ResultsPage']

