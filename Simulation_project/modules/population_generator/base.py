"""
Population Generator Base Classes

定义主体生成器的抽象基类，确保统一的接口和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class GenerationSummary:
    """生成结果摘要统计"""
    n_agents: int
    generation_time: float
    data_quality_score: float
    distribution_stats: Dict[str, Any]
    validation_passed: bool
    memory_usage_mb: float


class AgentGenerator(ABC):
    """
    主体生成器抽象基类
    
    定义所有主体生成器必须实现的核心接口，确保：
    1. 统一的数据格式和接口
    2. 可配置的参数管理
    3. 数据验证和质量保证
    4. 性能监控和统计
    """
    
    def __init__(self, config: Dict[str, Any], random_state: Optional[int] = None):
        """
        初始化生成器
        
        Args:
            config: 配置参数字典
            random_state: 随机种子，用于结果复现
        """
        self.config = config
        self.random_state = random_state
        self.is_fitted = False
        self.generation_history = []
        
        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.info(f"初始化{self.__class__.__name__}，随机种子: {random_state}")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合生成器模型
        
        根据真实数据拟合生成模型的参数，为后续生成做准备。
        
        Args:
            data: 用于拟合的真实数据，必须包含所需的特征列
            
        Raises:
            ValueError: 数据格式不符合要求
            RuntimeError: 拟合过程失败
        """
        pass
    
    @abstractmethod
    def generate(self, n_agents: int, **kwargs) -> pd.DataFrame:
        """
        生成指定数量的虚拟主体
        
        Args:
            n_agents: 需要生成的主体数量
            **kwargs: 其他生成参数
            
        Returns:
            包含生成主体的DataFrame，列名与真实数据一致
            
        Raises:
            ValueError: 参数不合法
            RuntimeError: 生成过程失败
        """
        pass
    
    @abstractmethod
    def validate(self, agents: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        验证生成数据的质量
        
        Args:
            agents: 待验证的主体数据
            
        Returns:
            (验证是否通过, 详细验证报告)
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list:
        """
        获取生成器需要的数据列名
        
        Returns:
            必需的列名列表
        """
        pass
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        获取生成器的摘要统计信息
        
        Returns:
            包含模型状态、生成历史等信息的字典
        """
        return {
            'class_name': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'random_state': self.random_state,
            'generation_count': len(self.generation_history),
            'total_generated': sum(h.n_agents for h in self.generation_history)
        }
    
    def reset(self) -> None:
        """重置生成器状态"""
        self.is_fitted = False
        self.generation_history.clear()
        logger.info(f"{self.__class__.__name__} 已重置")
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        验证输入数据的基本格式
        
        Args:
            data: 待验证的数据
            
        Raises:
            ValueError: 数据格式不符合要求
        """
        if data.empty:
            raise ValueError("输入数据不能为空")
        
        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")
        
        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"列 '{col}' 必须是数值类型")
        
        # 检查缺失值
        if data[required_columns].isnull().any().any():
            raise ValueError("数据包含缺失值，请先进行数据清洗")
        
        logger.info(f"输入数据验证通过: {len(data)}行 x {len(required_columns)}列")
    
    def _record_generation(self, summary: GenerationSummary) -> None:
        """记录生成历史"""
        self.generation_history.append(summary)
        logger.info(f"生成完成: {summary.n_agents}个主体，用时{summary.generation_time:.2f}秒")
    
    def _check_fitted(self) -> None:
        """检查模型是否已拟合"""
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} 尚未拟合，请先调用fit()方法")


class BatchGenerator:
    """
    批量生成管理器
    
    用于处理大规模主体生成，避免内存溢出。
    """
    
    def __init__(self, generator: AgentGenerator, batch_size: int = 1000):
        """
        初始化批量生成器
        
        Args:
            generator: 已拟合的主体生成器
            batch_size: 每批生成的主体数量
        """
        self.generator = generator
        self.batch_size = batch_size
        
        if not generator.is_fitted:
            raise ValueError("生成器必须先进行拟合")
    
    def generate_in_batches(self, total_agents: int, **kwargs) -> pd.DataFrame:
        """
        分批生成大量主体
        
        Args:
            total_agents: 总共需要生成的主体数量
            **kwargs: 传递给生成器的其他参数
            
        Returns:
            包含所有生成主体的DataFrame
        """
        if total_agents <= 0:
            raise ValueError("主体数量必须大于0")
        
        batches = []
        remaining = total_agents
        batch_num = 0
        
        logger.info(f"开始批量生成 {total_agents} 个主体，批次大小: {self.batch_size}")
        
        while remaining > 0:
            current_batch_size = min(self.batch_size, remaining)
            batch_num += 1
            
            logger.info(f"生成第 {batch_num} 批，数量: {current_batch_size}")
            
            batch_data = self.generator.generate(current_batch_size, **kwargs)
            batches.append(batch_data)
            
            remaining -= current_batch_size
        
        # 合并所有批次
        result = pd.concat(batches, ignore_index=True)
        logger.info(f"批量生成完成，总计: {len(result)} 个主体")
        
        return result


class GeneratorFactory:
    """
    生成器工厂类
    
    用于创建和管理不同类型的主体生成器。
    """
    
    _generators = {}
    
    @classmethod
    def register_generator(cls, name: str, generator_class: type) -> None:
        """注册新的生成器类型"""
        if not issubclass(generator_class, AgentGenerator):
            raise ValueError("生成器必须继承自AgentGenerator")
        
        cls._generators[name] = generator_class
        logger.info(f"注册生成器: {name}")
    
    @classmethod
    def create_generator(cls, 
                        name: str, 
                        config: Dict[str, Any], 
                        random_state: Optional[int] = None) -> AgentGenerator:
        """
        创建指定类型的生成器
        
        Args:
            name: 生成器类型名称
            config: 配置参数
            random_state: 随机种子
            
        Returns:
            生成器实例
        """
        if name not in cls._generators:
            raise ValueError(f"未知的生成器类型: {name}")
        
        generator_class = cls._generators[name]
        return generator_class(config, random_state)
    
    @classmethod
    def list_generators(cls) -> list:
        """列出所有可用的生成器类型"""
        return list(cls._generators.keys())


# 导出的异常类
class GenerationError(Exception):
    """生成过程中的自定义异常"""
    pass


class ValidationError(Exception):
    """数据验证失败的自定义异常"""
    pass
