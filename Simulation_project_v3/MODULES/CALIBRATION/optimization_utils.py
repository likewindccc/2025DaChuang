import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class OptimizationUtils:
    """
    优化辅助工具类
    
    功能：
    1. 管理参数边界
    2. 生成初始值（多种策略）
    3. 参数变换（对数/逻辑变换）
    4. 参数向量与字典的相互转换
    
    属性：
        param_config: 参数配置列表
        param_names: 参数名称列表
        param_bounds: 参数边界列表
        param_initial: 参数初始值数组
    """
    
    def __init__(self, calibration_config: Dict):
        """
        初始化优化辅助工具
        
        参数:
            calibration_config: 校准配置字典
        """
        self.param_config = calibration_config['parameters']
        self.param_names = [p['name'] for p in self.param_config]
        self.n_params = len(self.param_names)
        
        # 提取参数边界和初始值
        self.param_bounds = []
        self.param_initial = []
        self.config_paths = {}
        
        for param in self.param_config:
            self.param_bounds.append(tuple(param['bounds']))
            self.param_initial.append(param['initial_value'])
            self.config_paths[param['name']] = param['config_path']
        
        self.param_initial = np.array(self.param_initial)
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """
        获取参数边界列表
        
        返回:
            [(lower, upper), ...] 边界元组列表
        """
        return self.param_bounds.copy()
    
    def get_initial_values(
        self, 
        strategy: str = 'baseline'
    ) -> np.ndarray:
        """
        生成初始值
        
        参数:
            strategy: 初始值策略
                - 'baseline': 使用配置文件中的初始值
                - 'random': 在边界内随机采样
                - 'midpoint': 使用边界中点
                - 'sensitivity': 使用敏感性分析结果（待实现）
        
        返回:
            初始值数组，形状为(n_params,)
        """
        if strategy == 'baseline':
            return self.param_initial.copy()
        
        elif strategy == 'random':
            # 在边界内随机采样
            initial = np.zeros(self.n_params)
            for i, (lower, upper) in enumerate(self.param_bounds):
                initial[i] = np.random.uniform(lower, upper)
            return initial
        
        elif strategy == 'midpoint':
            # 使用边界中点
            initial = np.zeros(self.n_params)
            for i, (lower, upper) in enumerate(self.param_bounds):
                initial[i] = (lower + upper) / 2.0
            return initial
        
        elif strategy == 'sensitivity':
            # 从敏感性分析结果读取（暂时返回baseline）
            # TODO: 实现从敏感性分析结果文件读取最优值
            return self.param_initial.copy()
        
        else:
            raise ValueError(f"不支持的初始值策略: {strategy}")
    
    def vector_to_dict(self, params_vector: np.ndarray) -> Dict[str, float]:
        """
        将参数向量转换为参数字典
        
        参数:
            params_vector: 参数向量，形状为(n_params,)
        
        返回:
            {param_name: value} 字典
        """
        if len(params_vector) != self.n_params:
            raise ValueError(
                f"参数向量长度不匹配：期望 {self.n_params}, "
                f"实际 {len(params_vector)}"
            )
        
        return {
            name: float(value) 
            for name, value in zip(self.param_names, params_vector)
        }
    
    def dict_to_vector(self, params_dict: Dict[str, float]) -> np.ndarray:
        """
        将参数字典转换为参数向量
        
        参数:
            params_dict: {param_name: value} 字典
        
        返回:
            参数向量，形状为(n_params,)
        """
        return np.array([
            params_dict[name] for name in self.param_names
        ])
    
    def clip_to_bounds(self, params_vector: np.ndarray) -> np.ndarray:
        """
        将参数向量裁剪到边界内
        
        参数:
            params_vector: 参数向量
        
        返回:
            裁剪后的参数向量
        """
        clipped = params_vector.copy()
        
        for i, (lower, upper) in enumerate(self.param_bounds):
            clipped[i] = np.clip(clipped[i], lower, upper)
        
        return clipped
    
    def check_bounds(self, params_vector: np.ndarray) -> bool:
        """
        检查参数向量是否在边界内
        
        参数:
            params_vector: 参数向量
        
        返回:
            如果所有参数都在边界内则返回True
        """
        for i, (lower, upper) in enumerate(self.param_bounds):
            if not (lower <= params_vector[i] <= upper):
                return False
        return True
    
    def get_config_path(self, param_name: str) -> str:
        """
        获取参数在mfg_config.yaml中的路径
        
        参数:
            param_name: 参数名称
        
        返回:
            配置路径字符串，如 'economics.rho'
        """
        return self.config_paths[param_name]
    
    def get_param_names(self) -> List[str]:
        """
        获取参数名称列表
        
        返回:
            参数名称列表
        """
        return self.param_names.copy()
    
    def get_n_params(self) -> int:
        """
        获取参数数量
        
        返回:
            参数总数
        """
        return self.n_params
    
    def print_parameter_info(self) -> None:
        """
        打印参数信息（用于调试）
        """
        print("\n" + "="*80)
        print("校准参数信息")
        print("="*80)
        
        for i, param in enumerate(self.param_config):
            print(f"\n参数 {i+1}: {param['name']}")
            print(f"  显示名称: {param['display_name']}")
            print(f"  配置路径: {param['config_path']}")
            print(f"  边界: {param['bounds']}")
            print(f"  初始值: {param['initial_value']}")
            if param.get('description'):
                print(f"  说明: {param['description']}")
        
        print("\n" + "="*80)
    
    def generate_multiple_initial_values(
        self, 
        n_trials: int = 5,
        strategy: str = 'random'
    ) -> List[np.ndarray]:
        """
        生成多组初始值（用于鲁棒性测试）
        
        参数:
            n_trials: 生成的初始值组数
            strategy: 初始值策略
        
        返回:
            初始值数组列表
        """
        initial_values_list = []
        
        # 第一组始终使用baseline
        initial_values_list.append(self.get_initial_values('baseline'))
        
        # 其余使用指定策略
        for _ in range(n_trials - 1):
            initial_values_list.append(self.get_initial_values(strategy))
        
        return initial_values_list


def update_mfg_config_with_params(
    mfg_config_path: Path,
    params_dict: Dict[str, float],
    param_utils: OptimizationUtils,
    save_path: Optional[Path] = None
) -> Dict:
    """
    使用校准参数更新MFG配置
    
    参数:
        mfg_config_path: mfg_config.yaml路径
        params_dict: 参数字典 {param_name: value}
        param_utils: OptimizationUtils实例
        save_path: 保存路径（如果为None则不保存）
    
    返回:
        更新后的配置字典
    """
    # 加载MFG配置
    with open(mfg_config_path, 'r', encoding='utf-8') as f:
        mfg_config = yaml.safe_load(f)
    
    # 更新参数
    for param_name, param_value in params_dict.items():
        config_path = param_utils.get_config_path(param_name)
        keys = config_path.split('.')
        
        # 导航到目标位置
        target = mfg_config
        for key in keys[:-1]:
            target = target[key]
        
        # 更新值
        target[keys[-1]] = float(param_value)
    
    # 如果指定了保存路径，则保存配置
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(mfg_config, f, allow_unicode=True, 
                     default_flow_style=False)
    
    return mfg_config


def validate_parameters(
    params_vector: np.ndarray,
    param_utils: OptimizationUtils
) -> Tuple[bool, Optional[str]]:
    """
    验证参数向量的有效性
    
    参数:
        params_vector: 参数向量
        param_utils: OptimizationUtils实例
    
    返回:
        (is_valid, error_message)
        - is_valid: 参数是否有效
        - error_message: 如果无效，返回错误信息；否则为None
    """
    # 检查维度
    if len(params_vector) != param_utils.get_n_params():
        return False, f"参数维度不匹配：期望 {param_utils.get_n_params()}, 实际 {len(params_vector)}"
    
    # 检查是否包含NaN或Inf
    if np.any(np.isnan(params_vector)):
        return False, "参数包含NaN值"
    
    if np.any(np.isinf(params_vector)):
        return False, "参数包含Inf值"
    
    # 检查边界
    if not param_utils.check_bounds(params_vector):
        out_of_bounds = []
        bounds = param_utils.get_parameter_bounds()
        param_names = param_utils.get_param_names()
        
        for i, (lower, upper) in enumerate(bounds):
            if not (lower <= params_vector[i] <= upper):
                out_of_bounds.append(
                    f"{param_names[i]}={params_vector[i]:.4f} "
                    f"(边界: [{lower}, {upper}])"
                )
        
        return False, f"参数超出边界: {', '.join(out_of_bounds)}"
    
    return True, None

