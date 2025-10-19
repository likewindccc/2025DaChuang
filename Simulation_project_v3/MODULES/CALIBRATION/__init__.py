"""
CALIBRATION模块

功能：
1. 目标矩（Target Moments）管理
2. SMM目标函数实现
3. SMM校准器（两阶段校准）
4. 断点续跑支持
5. 结果分析与可视化

主要类：
- TargetMoments: 目标矩管理类
- ObjectiveFunction: SMM目标函数类
- SMMCalibrator: SMM校准器核心类
- OptimizationUtils: 优化辅助工具类

使用示例：
    from MODULES.CALIBRATION import SMMCalibrator
    
    calibrator = SMMCalibrator('CONFIG/calibration_config.yaml')
    result = calibrator.calibrate()
"""

from .target_moments import TargetMoments
from .objective_function import ObjectiveFunction, create_weight_matrix
from .smm_calibrator import SMMCalibrator
from .optimization_utils import OptimizationUtils

__all__ = [
    'TargetMoments',
    'ObjectiveFunction',
    'create_weight_matrix',
    'SMMCalibrator',
    'OptimizationUtils',
]

__version__ = '1.0.0'
__author__ = 'Simulation Project v3 Team'

