#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成器抽象基类模块

定义所有生成器（劳动力生成器、企业生成器）的标准接口和生命周期。

生命周期:
    1. __init__(config)     - 初始化配置
    2. fit(data)            - 拟合参数
    3. generate(n_agents)   - 生成虚拟个体
    4. validate(agents)     - 验证生成质量
    5. save_params(path)    - 保存参数（可选）
    6. load_params(path)    - 加载参数（可选）

使用示例:
    >>> class LaborGenerator(BaseGenerator):
    ...     def fit(self, data):
    ...         # 实现Copula拟合逻辑
    ...         self.fitted_params = {...}
    ...         self.is_fitted = True
    ...     
    ...     def generate(self, n_agents):
    ...         if not self.is_fitted:
    ...             raise RuntimeError("必须先调用fit()")
    ...         # 生成逻辑
    ...         return agents_df
    ...     
    ...     def validate(self, agents):
    ...         # KS检验等
    ...         return True
    >>> 
    >>> generator = LaborGenerator(config)
    >>> generator.fit(training_data)
    >>> virtual_agents = generator.generate(1000)
    >>> is_valid = generator.validate(virtual_agents)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import pickle
from pathlib import Path

from .exceptions import ConfigurationError


class BaseGenerator(ABC):
    """
    生成器抽象基类
    
    定义所有生成器（劳动力、企业）的标准接口。
    
    子类必须实现：
    - fit(data): 拟合参数
    - generate(n_agents): 生成虚拟个体
    - validate(agents): 验证生成质量
    
    Attributes:
        config: 配置字典
        is_fitted: 是否已拟合
        fitted_params: 拟合后的参数
    
    Examples:
        >>> # 创建子类
        >>> class MyGenerator(BaseGenerator):
        ...     def fit(self, data: pd.DataFrame) -> None:
        ...         self.fitted_params = {'mean': data.mean()}
        ...         self.is_fitted = True
        ...     
        ...     def generate(self, n_agents: int) -> pd.DataFrame:
        ...         if not self.is_fitted:
        ...             raise RuntimeError("必须先调用fit()")
        ...         return pd.DataFrame(...)
        ...     
        ...     def validate(self, agents: pd.DataFrame) -> bool:
        ...         return True
        
        >>> # 使用
        >>> gen = MyGenerator({'seed': 42})
        >>> gen.fit(training_data)
        >>> agents = gen.generate(1000)
        >>> gen.validate(agents)
        True
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            config: 配置字典，应包含：
                - seed: 随机种子（可选）
                - 其他生成器特定参数
        
        Raises:
            ConfigurationError: 配置参数不合法
        """
        self.config = config
        self.is_fitted = False
        self.fitted_params: Optional[Dict[str, Any]] = None
        
        # 验证配置（子类可覆盖）
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        验证配置参数
        
        子类可以覆盖此方法来实现自定义验证逻辑
        
        Raises:
            ConfigurationError: 配置不合法
        """
        # 默认验证：检查config是否为字典
        if not isinstance(self.config, dict):
            raise ConfigurationError(
                f"config必须是字典类型，得到：{type(self.config)}"
            )
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合生成器参数
        
        从训练数据中学习分布参数、依赖结构等。
        
        Args:
            data: 训练数据，应包含所有必需的列
        
        Raises:
            ValueError: 数据格式不正确
            CopulaFittingError: 拟合失败
        
        Notes:
            - 拟合成功后必须设置 self.is_fitted = True
            - 拟合参数应存储在 self.fitted_params
        
        Examples:
            >>> def fit(self, data: pd.DataFrame) -> None:
            ...     # 检查必需列
            ...     required_cols = ['T', 'S', 'D', 'W']
            ...     if not all(col in data.columns for col in required_cols):
            ...         raise ValueError("数据缺少必需列")
            ...     
            ...     # 拟合Copula
            ...     self.fitted_params = fit_copula(data[required_cols])
            ...     self.is_fitted = True
        """
        pass
    
    @abstractmethod
    def generate(self, n_agents: int) -> pd.DataFrame:
        """
        生成虚拟个体
        
        使用拟合的参数生成指定数量的虚拟个体。
        
        Args:
            n_agents: 生成数量，必须为正整数
        
        Returns:
            包含虚拟个体的DataFrame，至少包含列：
            - agent_id: 个体ID
            - T, S, D, W: 核心特征
            - 可能包含其他控制变量
        
        Raises:
            RuntimeError: 未先调用fit()
            ValueError: n_agents不是正整数
        
        Examples:
            >>> def generate(self, n_agents: int) -> pd.DataFrame:
            ...     if not self.is_fitted:
            ...         raise RuntimeError("必须先调用fit()")
            ...     
            ...     if n_agents <= 0:
            ...         raise ValueError(f"n_agents必须为正，得到：{n_agents}")
            ...     
            ...     # 生成逻辑
            ...     agents = sample_from_copula(n_agents, self.fitted_params)
            ...     return pd.DataFrame(agents)
        """
        pass
    
    @abstractmethod
    def validate(self, agents: pd.DataFrame) -> bool:
        """
        验证生成的个体质量
        
        使用统计检验（如KS检验）验证生成的个体是否与训练数据分布一致。
        
        Args:
            agents: 待验证的个体数据
        
        Returns:
            是否通过验证
        
        Examples:
            >>> def validate(self, agents: pd.DataFrame) -> bool:
            ...     # KS检验
            ...     for col in ['T', 'S', 'D', 'W']:
            ...         ks_stat, p_value = kstest(...)
            ...         if p_value < 0.05:
            ...             return False
            ...     return True
        """
        pass
    
    def save_params(self, filepath: str) -> None:
        """
        保存拟合后的参数
        
        默认使用pickle序列化，子类可覆盖以支持其他格式（如JSON、YAML）。
        
        Args:
            filepath: 保存路径
        
        Raises:
            RuntimeError: 未先调用fit()
            IOError: 保存失败
        
        Examples:
            >>> gen.fit(data)
            >>> gen.save_params('models/labor_generator_params.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()再保存参数")
        
        # 确保目录存在
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存参数
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_params, f)
        
        print(f"[保存] 参数已保存到: {filepath}")
    
    def load_params(self, filepath: str) -> None:
        """
        加载已保存的参数
        
        默认使用pickle反序列化，子类可覆盖以支持其他格式。
        
        Args:
            filepath: 参数文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            IOError: 加载失败
        
        Examples:
            >>> gen = LaborGenerator(config)
            >>> gen.load_params('models/labor_generator_params.pkl')
            >>> agents = gen.generate(1000)  # 无需再调用fit()
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"参数文件不存在：{filepath}")
        
        # 加载参数
        with open(filepath, 'rb') as f:
            self.fitted_params = pickle.load(f)
        
        self.is_fitted = True
        print(f"[加载] 参数已从 {filepath} 加载")
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取拟合后的参数
        
        Returns:
            参数字典的副本
        
        Raises:
            RuntimeError: 未先调用fit()
        
        Examples:
            >>> gen.fit(data)
            >>> params = gen.get_params()
            >>> print(params.keys())
            dict_keys(['marginals', 'copula_corr', 'copula_type'])
        """
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()才能获取参数")
        
        # 返回副本，防止外部修改
        return self.fitted_params.copy()
    
    def __repr__(self) -> str:
        """
        字符串表示
        
        Returns:
            描述性字符串
        """
        status = "已拟合" if self.is_fitted else "未拟合"
        return f"{self.__class__.__name__}(status={status})"
