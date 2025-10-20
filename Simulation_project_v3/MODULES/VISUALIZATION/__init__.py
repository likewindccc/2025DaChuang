"""
可视化模块

提供MFG、Calibration、Simulation和Data模块的可视化功能
支持静态图（Matplotlib）和交互式图（Plotly）
"""

from .style_config import COLORS, FIGURE_SIZE, FONT_CONFIG, setup_matplotlib_style, initialize_styles
from .data_visualizer import DataVisualizer
from .mfg_visualizer import MFGVisualizer
from .calibration_visualizer import CalibrationVisualizer
from .simulation_visualizer import SimulationVisualizer
from .dashboard_builder import DashboardBuilder

__all__ = [
    'COLORS',
    'FIGURE_SIZE',
    'FONT_CONFIG',
    'setup_matplotlib_style',
    'initialize_styles',
    'DataVisualizer',
    'MFGVisualizer',
    'CalibrationVisualizer',
    'SimulationVisualizer',
    'DashboardBuilder'
]

__version__ = '1.0.0'

