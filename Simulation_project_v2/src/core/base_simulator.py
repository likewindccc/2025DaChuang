#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟器抽象基类模块

定义所有模拟器（匹配引擎、MFG求解器）的标准接口和生命周期。

生命周期:
    1. __init__(config)     - 初始化配置
    2. setup(**kwargs)      - 准备模拟（加载数据、初始化参数）
    3. run()                - 执行模拟
    4. get_results()        - 获取格式化结果
    5. save_results(path)   - 保存结果（可选）

使用示例:
    >>> class MatchingEngine(BaseSimulator):
    ...     def setup(self, laborers, enterprises):
    ...         self.laborers = laborers
    ...         self.enterprises = enterprises
    ...         self.is_setup = True
    ...     
    ...     def run(self):
    ...         if not self.is_setup:
    ...             raise RuntimeError("必须先调用setup()")
    ...         # GS算法...
    ...         self.results = matchings
    ...         self.is_complete = True
    ...         return matchings
    ...     
    ...     def get_results(self):
    ...         if not self.is_complete:
    ...             raise RuntimeError("模拟未完成")
    ...         return {'matchings': self.results, 'metrics': ...}
    >>> 
    >>> engine = MatchingEngine(config)
    >>> engine.setup(laborers=labor_list, enterprises=ent_list)
    >>> matchings = engine.run()
    >>> results = engine.get_results()
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

from .exceptions import ConfigurationError


class BaseSimulator(ABC):
    """
    模拟器抽象基类
    
    定义所有模拟器（匹配引擎、MFG求解器）的标准接口。
    
    子类必须实现：
    - setup(**kwargs): 准备模拟
    - run(): 执行模拟
    - get_results(): 获取结果
    
    Attributes:
        config: 配置字典
        is_setup: 是否已准备就绪
        is_complete: 模拟是否完成
        results: 模拟结果
    
    Examples:
        >>> # 创建子类
        >>> class MySimulator(BaseSimulator):
        ...     def setup(self, data):
        ...         self.data = data
        ...         self.is_setup = True
        ...     
        ...     def run(self):
        ...         if not self.is_setup:
        ...             raise RuntimeError("必须先调用setup()")
        ...         self.results = {'output': ...}
        ...         self.is_complete = True
        ...         return self.results
        ...     
        ...     def get_results(self):
        ...         if not self.is_complete:
        ...             raise RuntimeError("模拟未完成")
        ...         return self.results
        
        >>> # 使用
        >>> sim = MySimulator({'max_iter': 1000})
        >>> sim.setup(data=my_data)
        >>> output = sim.run()
        >>> results = sim.get_results()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟器
        
        Args:
            config: 配置字典，应包含：
                - max_iterations: 最大迭代次数（可选）
                - tolerance: 收敛容忍度（可选）
                - seed: 随机种子（可选）
                - 其他模拟器特定参数
        
        Raises:
            ConfigurationError: 配置参数不合法
        """
        self.config = config
        self.is_setup = False
        self.is_complete = False
        self.results: Optional[Any] = None
        
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
    def setup(self, **kwargs) -> None:
        """
        准备模拟
        
        加载数据、初始化参数、分配资源等。
        
        Args:
            **kwargs: 模拟器特定的参数
                例如：laborers, enterprises, initial_state等
        
        Raises:
            ValueError: 参数不正确
        
        Notes:
            - 准备完成后必须设置 self.is_setup = True
        
        Examples:
            >>> def setup(
            ...     self,
            ...     laborers: List[Agent],
            ...     enterprises: List[Agent]
            ... ) -> None:
            ...     # 验证输入
            ...     if not laborers or not enterprises:
            ...         raise ValueError("劳动力和企业列表不能为空")
            ...     
            ...     # 初始化
            ...     self.laborers = laborers
            ...     self.enterprises = enterprises
            ...     self.preference_matrix = self._compute_preferences()
            ...     
            ...     self.is_setup = True
        """
        pass
    
    @abstractmethod
    def run(self) -> Any:
        """
        执行模拟
        
        运行核心算法，计算结果。
        
        Returns:
            模拟结果（具体类型由子类定义）
        
        Raises:
            RuntimeError: 未先调用setup()
        
        Notes:
            - 执行前必须检查 self.is_setup
            - 执行完成后必须设置 self.is_complete = True
            - 结果应存储在 self.results
        
        Examples:
            >>> def run(self) -> List[MatchingPair]:
            ...     if not self.is_setup:
            ...         raise RuntimeError("必须先调用setup()")
            ...     
            ...     # 执行GS算法
            ...     matchings = self._gale_shapley()
            ...     
            ...     # 保存结果
            ...     self.results = matchings
            ...     self.is_complete = True
            ...     
            ...     return matchings
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        获取格式化的结果
        
        将模拟结果整理为结构化的字典，便于分析和可视化。
        
        Returns:
            结果字典，应包含：
            - 主要输出（如matchings, equilibrium等）
            - 指标（如unemployment_rate, avg_wage等）
            - 元数据（如iterations, convergence_error等）
        
        Raises:
            RuntimeError: 模拟未完成
        
        Examples:
            >>> def get_results(self) -> Dict[str, Any]:
            ...     if not self.is_complete:
            ...         raise RuntimeError("模拟未完成，无法获取结果")
            ...     
            ...     return {
            ...         'matchings': self.results,
            ...         'metrics': {
            ...             'unemployment_rate': 0.1,
            ...             'avg_wage': 4800.0,
            ...             'theta': 1.0
            ...         },
            ...         'metadata': {
            ...             'iterations': 50,
            ...             'converged': True
            ...         }
            ...     }
        """
        pass
    
    def save_results(self, filepath: str, format: str = 'pickle') -> None:
        """
        保存模拟结果
        
        支持多种格式：pickle（默认）、json、csv等。
        
        Args:
            filepath: 保存路径
            format: 保存格式，'pickle' | 'json' | 'custom'
        
        Raises:
            RuntimeError: 模拟未完成
            ValueError: 不支持的格式
            IOError: 保存失败
        
        Examples:
            >>> sim.run()
            >>> sim.save_results('results/matching_results.pkl')
            >>> sim.save_results('results/matching_results.json', format='json')
        """
        if not self.is_complete:
            raise RuntimeError("模拟未完成，无法保存结果")
        
        # 确保目录存在
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"[保存] 结果已保存到: {filepath} (pickle格式)")
        
        elif format == 'json':
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                # 注意：需要确保results是JSON可序列化的
                json.dump(self.get_results(), f, indent=2, ensure_ascii=False)
            print(f"[保存] 结果已保存到: {filepath} (JSON格式)")
        
        else:
            raise ValueError(f"不支持的保存格式：{format}")
    
    def load_results(self, filepath: str, format: str = 'pickle') -> None:
        """
        加载已保存的结果
        
        Args:
            filepath: 结果文件路径
            format: 文件格式，'pickle' | 'json'
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的格式
            IOError: 加载失败
        
        Examples:
            >>> sim = MatchingEngine(config)
            >>> sim.load_results('results/matching_results.pkl')
            >>> results = sim.get_results()
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"结果文件不存在：{filepath}")
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
            self.is_complete = True
            print(f"[加载] 结果已从 {filepath} 加载 (pickle格式)")
        
        elif format == 'json':
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                self.results = loaded_data
            self.is_complete = True
            print(f"[加载] 结果已从 {filepath} 加载 (JSON格式)")
        
        else:
            raise ValueError(f"不支持的加载格式：{format}")
    
    def reset(self) -> None:
        """
        重置模拟器状态
        
        清除所有运行结果，允许重新配置和运行。
        
        Examples:
            >>> sim.run()
            >>> # ... 分析结果 ...
            >>> sim.reset()
            >>> sim.setup(new_data)
            >>> sim.run()  # 新的模拟
        """
        self.is_setup = False
        self.is_complete = False
        self.results = None
        print("[重置] 模拟器已重置")
    
    def __repr__(self) -> str:
        """
        字符串表示
        
        Returns:
            描述性字符串
        """
        if self.is_complete:
            status = "已完成"
        elif self.is_setup:
            status = "已准备"
        else:
            status = "未初始化"
        
        return f"{self.__class__.__name__}(status={status})"
