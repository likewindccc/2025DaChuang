"""
Population Generator Configuration System

提供统一的配置管理，支持参数验证和默认值设置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import yaml
from pathlib import Path


@dataclass
class CopulaConfig:
    """Copula模型配置"""
    # 候选模型类型
    candidate_models: List[str] = field(default_factory=lambda: [
        'Gaussian', 'RegularVine', 'CenterVine', 'DirectVine'
    ])
    
    # 模型选择标准 ('aic', 'bic', 'log_likelihood')
    selection_criterion: str = 'aic'
    
    # 边际分布候选
    marginal_distributions: List[str] = field(default_factory=lambda: [
        'norm', 'gamma', 'beta', 'lognorm', 'uniform'
    ])
    
    # Vine Copula参数
    vine_structure: str = 'regular'  # 'regular', 'center', 'direct'
    
    # 数值优化参数
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # 并行化设置
    n_jobs: int = -1  # -1表示使用所有CPU核心


@dataclass
class LaborGeneratorConfig:
    """劳动力生成器配置"""
    # 必需的数据列
    required_columns: List[str] = field(default_factory=lambda: [
        'T', 'S', 'D', 'W', 'age', 'education'
    ])
    
    # 数据验证范围
    data_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'T': (0, 80),       # 工作时间 0-80小时/周
        'S': (0, 1),        # 技能水平 0-1
        'D': (0, 1),        # 数字素养 0-1
        'W': (1000, 8000),  # 期望工资 1000-8000元
        'age': (16, 65),    # 年龄 16-65岁
        'education': (0, 4) # 教育水平 0-4（小学到研究生）
    })
    
    # Copula模型配置
    copula_config: CopulaConfig = field(default_factory=CopulaConfig)
    
    # 生成质量控制
    quality_threshold: float = 0.8  # 数据质量阈值
    max_generation_attempts: int = 5  # 最大生成尝试次数
    
    # 内存管理
    batch_size: int = 1000
    cache_marginals: bool = True


@dataclass
class MultivariateNormalConfig:
    """多元正态分布配置"""
    # 分布维度（企业的4个属性）
    dimensions: int = 4
    
    # 默认参数（如果不进行校准）
    default_mean: List[float] = field(default_factory=lambda: [40.0, 0.5, 0.5, 3500.0])
    default_cov_matrix: List[List[float]] = field(default_factory=lambda: [
        [100.0,  0.1,   0.1,   200.0],  # T的方差和协方差
        [0.1,    0.05,  0.02,  50.0],   # S的方差和协方差
        [0.1,    0.02,  0.05,  50.0],   # D的方差和协方差
        [200.0,  50.0,  50.0,  500000.0] # W的方差和协方差
    ])
    
    # 参数约束
    mean_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'T_req': (20, 60),     # 工作时间要求
        'S_req': (0.1, 0.9),   # 技能要求
        'D_req': (0.1, 0.9),   # 数字化要求
        'W_offer': (2000, 6000) # 提供薪资
    })
    
    # 协方差矩阵约束
    min_variance: float = 1e-6  # 最小方差
    max_correlation: float = 0.9  # 最大相关系数
    
    # 数值稳定性
    regularization: float = 1e-8  # 正则化系数


@dataclass
class EnterpriseGeneratorConfig:
    """企业生成器配置"""
    # 必需的数据列
    required_columns: List[str] = field(default_factory=lambda: [
        'T_req', 'S_req', 'D_req', 'W_offer'
    ])
    
    # 数据验证范围
    data_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'T_req': (20, 60),     # 工作时间要求
        'S_req': (0, 1),       # 技能要求
        'D_req': (0, 1),       # 数字化要求
        'W_offer': (1500, 7000) # 提供薪资
    })
    
    # 多元正态分布配置
    mvn_config: MultivariateNormalConfig = field(default_factory=MultivariateNormalConfig)
    
    # 校准参数
    enable_calibration: bool = True
    calibration_method: str = 'mle'  # 'mle', 'mom'
    
    # 生成质量控制
    quality_threshold: float = 0.9
    validation_samples: int = 1000


@dataclass
class OptimizationConfig:
    """numba优化配置"""
    # 启用JIT编译
    enable_jit: bool = True
    
    # 并行化设置
    enable_parallel: bool = True
    parallel_threshold: int = 1000  # 数据量超过此值时启用并行
    
    # 编译缓存
    cache_compiled_functions: bool = True
    
    # 性能监控
    enable_performance_monitoring: bool = True
    benchmark_iterations: int = 3


@dataclass
class PopulationConfig:
    """主体生成模块总配置"""
    # 子模块配置
    labor_config: LaborGeneratorConfig = field(default_factory=LaborGeneratorConfig)
    enterprise_config: EnterpriseGeneratorConfig = field(default_factory=EnterpriseGeneratorConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # 全局设置
    random_seed: Optional[int] = None
    log_level: str = 'INFO'
    
    # 输出设置
    output_format: str = 'pandas'  # 'pandas', 'numpy'
    save_generation_logs: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PopulationConfig':
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PopulationConfig':
        """从字典创建配置对象"""
        # 这里可以添加复杂的配置转换逻辑
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """将配置保存为YAML文件"""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
    
    def validate(self) -> None:
        """验证配置参数的合法性"""
        # 验证随机种子
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError("随机种子必须为非负整数")
        
        # 验证日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ValueError(f"日志级别必须为: {valid_log_levels}")
        
        # 验证输出格式
        valid_formats = ['pandas', 'numpy']
        if self.output_format not in valid_formats:
            raise ValueError(f"输出格式必须为: {valid_formats}")
        
        # 验证子配置
        self._validate_labor_config()
        self._validate_enterprise_config()
        self._validate_optimization_config()
    
    def _validate_labor_config(self) -> None:
        """验证劳动力生成器配置"""
        config = self.labor_config
        
        # 验证数据边界
        for col, (min_val, max_val) in config.data_bounds.items():
            if min_val >= max_val:
                raise ValueError(f"列 '{col}' 的数据边界无效: [{min_val}, {max_val}]")
        
        # 验证质量阈值
        if not 0 <= config.quality_threshold <= 1:
            raise ValueError("质量阈值必须在 [0, 1] 范围内")
        
        # 验证批次大小
        if config.batch_size <= 0:
            raise ValueError("批次大小必须大于0")
    
    def _validate_enterprise_config(self) -> None:
        """验证企业生成器配置"""
        config = self.enterprise_config
        
        # 验证数据边界
        for col, (min_val, max_val) in config.data_bounds.items():
            if min_val >= max_val:
                raise ValueError(f"列 '{col}' 的数据边界无效: [{min_val}, {max_val}]")
        
        # 验证多元正态分布配置
        mvn_config = config.mvn_config
        
        # 验证维度
        if mvn_config.dimensions != 4:
            raise ValueError("企业属性必须为4维")
        
        # 验证默认均值向量
        if len(mvn_config.default_mean) != mvn_config.dimensions:
            raise ValueError("默认均值向量维度不匹配")
        
        # 验证默认协方差矩阵
        cov_matrix = np.array(mvn_config.default_cov_matrix)
        if cov_matrix.shape != (mvn_config.dimensions, mvn_config.dimensions):
            raise ValueError("默认协方差矩阵维度不匹配")
        
        # 检查协方差矩阵正定性
        try:
            np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("默认协方差矩阵不是正定矩阵")
    
    def _validate_optimization_config(self) -> None:
        """验证优化配置"""
        config = self.optimization_config
        
        # 验证并行阈值
        if config.parallel_threshold <= 0:
            raise ValueError("并行阈值必须大于0")
        
        # 验证基准测试迭代次数
        if config.benchmark_iterations <= 0:
            raise ValueError("基准测试迭代次数必须大于0")


def create_default_config() -> PopulationConfig:
    """创建默认配置"""
    return PopulationConfig()


def load_config(config_path: str) -> PopulationConfig:
    """从文件加载配置"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if config_file.suffix.lower() == '.yaml':
        return PopulationConfig.from_yaml(config_path)
    else:
        raise ValueError("目前只支持YAML格式的配置文件")


def save_default_config(output_path: str) -> None:
    """保存默认配置到文件"""
    default_config = create_default_config()
    default_config.to_yaml(output_path)
