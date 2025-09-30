#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core异常体系模块

定义项目中所有自定义异常，采用细粒度异常设计，便于精确捕获和处理错误。

异常层次结构：
    SimulationError (基类)
    ├── DataValidationError        - 数据验证失败
    ├── CopulaFittingError         - Copula拟合失败
    ├── MatchingError              - 匹配算法错误
    ├── ConvergenceError           - MFG不收敛
    ├── ConfigurationError         - 配置错误
    └── CalibrationError           - 参数校准错误

使用示例：
    >>> from src.core.exceptions import DataValidationError
    >>> 
    >>> if value < 0:
    ...     raise DataValidationError(f"值不能为负：{value}")
"""


class SimulationError(Exception):
    """
    模拟系统基础异常
    
    所有自定义异常的基类。用于统一捕获所有模拟相关错误。
    
    Examples:
        >>> try:
        ...     # 某些模拟操作
        ...     pass
        ... except SimulationError as e:
        ...     print(f"模拟错误：{e}")
    
    Notes:
        - 不应直接抛出此异常，应使用具体的子类
        - 可以用于捕获所有模拟相关异常
    """
    pass


class DataValidationError(SimulationError):
    """
    数据验证失败
    
    用于以下场景：
    - 输入数据格式错误
    - 数据范围超出预期
    - 缺失必需字段
    - 数据类型不匹配
    
    Examples:
        >>> # 检查数据范围
        >>> if not (0 <= skill_score <= 100):
        ...     raise DataValidationError(
        ...         f"技能评分应在0-100之间，得到：{skill_score}"
        ...     )
        
        >>> # 检查必需字段
        >>> required_columns = ['T', 'S', 'D', 'W']
        >>> missing = set(required_columns) - set(data.columns)
        >>> if missing:
        ...     raise DataValidationError(
        ...         f"数据缺少必需列：{missing}"
        ...     )
    """
    pass


class CopulaFittingError(SimulationError):
    """
    Copula拟合失败
    
    用于以下场景：
    - Copula模型不收敛
    - 参数估计失败
    - 边际分布拟合失败
    - 依赖结构检验不通过
    
    Examples:
        >>> try:
        ...     copula_params = fit_copula(data)
        ... except Exception as e:
        ...     raise CopulaFittingError(
        ...         f"Gaussian Copula拟合失败：{e}"
        ...     ) from e
        
        >>> # 参数超出合理范围
        >>> if not (-1 <= rho <= 1):
        ...     raise CopulaFittingError(
        ...         f"相关系数超出范围[-1,1]：{rho}"
        ...     )
    """
    pass


class MatchingError(SimulationError):
    """
    匹配算法错误
    
    用于以下场景：
    - 匹配算法不收敛
    - 偏好矩阵计算失败
    - Gale-Shapley算法异常
    - 匹配结果不稳定
    
    Examples:
        >>> # 匹配算法不收敛
        >>> if iteration >= max_iterations:
        ...     raise MatchingError(
        ...         f"匹配算法在{max_iterations}次迭代后仍未收敛"
        ...     )
        
        >>> # 偏好矩阵计算失败
        >>> if np.any(np.isnan(preference_matrix)):
        ...     raise MatchingError(
        ...         "偏好矩阵包含NaN值，无法进行匹配"
        ...     )
    """
    pass


class ConvergenceError(SimulationError):
    """
    MFG不收敛
    
    用于以下场景：
    - 贝尔曼方程迭代不收敛
    - Kolmogorov Forward Equation (KFE) 演化不稳定
    - Mean Field Equilibrium (MFE) 均衡求解失败
    - 值函数更新震荡
    
    Examples:
        >>> # 贝尔曼方程不收敛
        >>> if not converged and iteration >= max_iterations:
        ...     raise ConvergenceError(
        ...         f"贝尔曼方程在{max_iterations}次迭代后仍未收敛，"
        ...         f"当前误差：{error:.6f}，阈值：{tolerance:.6f}"
        ...     )
        
        >>> # MFE求解失败
        >>> if mfe_error > tolerance:
        ...     raise ConvergenceError(
        ...         f"MFE均衡求解失败，误差={mfe_error:.6f} > {tolerance:.6f}"
        ...     )
    """
    pass


class ConfigurationError(SimulationError):
    """
    配置错误
    
    用于以下场景：
    - 配置文件格式错误
    - 必需参数缺失
    - 参数值不合法
    - 参数类型错误
    
    Examples:
        >>> # 检查必需参数
        >>> required_keys = ['grid_size', 'max_iterations', 'tolerance']
        >>> missing = set(required_keys) - set(config.keys())
        >>> if missing:
        ...     raise ConfigurationError(
        ...         f"配置缺少必需参数：{missing}"
        ...     )
        
        >>> # 参数范围检查
        >>> if config['tolerance'] <= 0:
        ...     raise ConfigurationError(
        ...         f"tolerance必须为正，得到：{config['tolerance']}"
        ...     )
        
        >>> # 参数类型检查
        >>> if not isinstance(config['grid_size'], int):
        ...     raise ConfigurationError(
        ...         f"grid_size必须是整数，得到：{type(config['grid_size'])}"
        ...     )
    """
    pass


class CalibrationError(SimulationError):
    """
    参数校准错误
    
    用于以下场景：
    - 遗传算法不收敛
    - 目标函数计算失败
    - 参数搜索空间不合理
    - 适应度函数异常
    
    Examples:
        >>> # 遗传算法不收敛
        >>> if generation >= max_generations and best_fitness > target_fitness:
        ...     raise CalibrationError(
        ...         f"遗传算法在{max_generations}代后仍未达到目标适应度，"
        ...         f"当前最佳：{best_fitness:.4f}，目标：{target_fitness:.4f}"
        ...     )
        
        >>> # 目标函数计算失败
        >>> try:
        ...     fitness = compute_fitness(params)
        ... except Exception as e:
        ...     raise CalibrationError(
        ...         f"目标函数计算失败（参数={params}）：{e}"
        ...     ) from e
        
        >>> # 参数超出搜索空间
        >>> if not all(lb <= p <= ub for p, lb, ub in zip(params, lower_bounds, upper_bounds)):
        ...     raise CalibrationError(
        ...         f"参数超出搜索空间：{params}"
        ...     )
    """
    pass


# 异常映射字典（用于从字符串创建异常）
EXCEPTION_MAP = {
    'SimulationError': SimulationError,
    'DataValidationError': DataValidationError,
    'CopulaFittingError': CopulaFittingError,
    'MatchingError': MatchingError,
    'ConvergenceError': ConvergenceError,
    'ConfigurationError': ConfigurationError,
    'CalibrationError': CalibrationError,
}


def get_exception_class(name: str) -> type:
    """
    根据名称获取异常类
    
    Args:
        name: 异常类名称
    
    Returns:
        异常类
    
    Raises:
        KeyError: 异常类不存在
    
    Examples:
        >>> exc_class = get_exception_class('DataValidationError')
        >>> raise exc_class("数据验证失败")
    """
    if name not in EXCEPTION_MAP:
        raise KeyError(f"未知的异常类：{name}")
    return EXCEPTION_MAP[name]
