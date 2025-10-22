"""
自定义组件模块

包含：
- ParameterWidget: 参数输入组件（标签+滑块+输入框）
- ChartWidget: 图表展示组件（matplotlib嵌入）
- LogWidget: 日志显示组件（彩色日志）
"""

from .parameter_widget import ParameterWidget
from .chart_widget import ChartWidget
from .log_widget import LogWidget

__all__ = ['ParameterWidget', 'ChartWidget', 'LogWidget']

