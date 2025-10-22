"""
配置管理器

负责读取、写入、验证YAML配置文件
"""

import yaml
from pathlib import Path
from .path_helper import get_resource_path


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_dir="CONFIG"):
        """
        初始化配置管理器
        
        参数:
            config_dir: 配置文件目录，默认为"CONFIG"
        """
        self.config_dir = get_resource_path(config_dir)
        self.current_config = None
        self.current_config_name = None
    
    def load_config(self, config_name="mfg_config.yaml"):
        """
        加载配置文件
        
        参数:
            config_name: 配置文件名
        
        返回:
            配置字典
        """
        config_path = Path(self.config_dir) / config_name
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.current_config = config
        self.current_config_name = config_name
        
        return config
    
    def save_config(self, config_dict, config_name=None):
        """
        保存配置到文件
        
        参数:
            config_dict: 配置字典
            config_name: 配置文件名，如果为None则使用当前文件名
        """
        if config_name is None:
            config_name = self.current_config_name
        
        config_path = Path(self.config_dir) / config_name
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, 
                     default_flow_style=False)
        
        self.current_config = config_dict
        self.current_config_name = config_name
    
    def get_default_config(self):
        """
        获取默认配置
        
        返回:
            默认配置字典
        """
        return self.load_config("mfg_config.yaml")
    
    def validate_config(self, config_dict):
        """
        验证配置参数
        
        参数:
            config_dict: 配置字典
        
        返回:
            (is_valid, error_message)
        """
        # 检查必需的顶层键
        required_keys = ['economics', 'market', 'population']
        for key in required_keys:
            if key not in config_dict:
                return False, f"缺少必需的配置项: {key}"
        
        # 检查经济参数
        economics = config_dict.get('economics', {})
        
        # 检查rho范围
        rho = economics.get('rho')
        if rho is not None:
            if not (0.3 <= rho <= 0.6):
                return False, f"rho必须在0.3-0.6之间，当前值: {rho}"
        
        # 检查kappa范围
        kappa = economics.get('kappa')
        if kappa is not None:
            if not (1000.0 <= kappa <= 4000.0):
                return False, f"kappa必须在1000-4000之间，当前值: {kappa}"
        
        # 检查alpha范围
        disutility_T = economics.get('disutility_T', {})
        alpha = disutility_T.get('alpha')
        if alpha is not None:
            if not (0.1 <= alpha <= 0.6):
                return False, f"alpha必须在0.1-0.6之间，当前值: {alpha}"
        
        # 检查市场参数
        market = config_dict.get('market', {})
        target_theta = market.get('target_theta')
        if target_theta is not None:
            if target_theta <= 0:
                return False, f"target_theta必须大于0，当前值: {target_theta}"
        
        # 检查人口参数
        population = config_dict.get('population', {})
        n_individuals = population.get('n_individuals')
        if n_individuals is not None:
            if n_individuals < 100:
                return False, f"个体数量必须至少为100，当前值: {n_individuals}"
        
        return True, ""
    
    def get_parameter_value(self, key_path):
        """
        根据键路径获取参数值
        
        参数:
            key_path: 点分隔的键路径，如 "economics.rho"
        
        返回:
            参数值，如果不存在返回None
        """
        if self.current_config is None:
            return None
        
        keys = key_path.split('.')
        value = self.current_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def set_parameter_value(self, key_path, new_value):
        """
        根据键路径设置参数值
        
        参数:
            key_path: 点分隔的键路径，如 "economics.rho"
            new_value: 新值
        """
        if self.current_config is None:
            self.current_config = {}
        
        keys = key_path.split('.')
        config = self.current_config
        
        # 遍历到倒数第二个键
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最后一个键的值
        config[keys[-1]] = new_value

