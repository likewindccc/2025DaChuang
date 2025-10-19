import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


class TargetMoments:
    """
    目标矩管理类
    
    功能：
    1. 加载目标矩数据（从target_moments.yaml）
    2. 计算模拟数据的矩（从MFG均衡结果）
    3. 提供矩向量生成接口（用于SMM计算）
    
    属性：
        config_path: 目标矩配置文件路径
        target_moments: 目标矩字典 {moment_name: value}
        moment_names: 矩名称列表（保持顺序）
    """
    
    def __init__(self, config_path: str):
        """
        初始化目标矩管理器
        
        参数:
            config_path: target_moments.yaml配置文件路径
        """
        self.config_path = Path(config_path)
        self.target_moments = {}
        self.moment_names = []
        self.moment_metadata = {}
        
        # 加载目标矩配置
        self._load_config()
    
    def _load_config(self) -> None:
        """
        从YAML文件加载目标矩配置
        
        配置文件结构：
        moments:
            unemployment_rate:
                value: 0.048
                unit: 比例
                ...
            mean_wage:
                value: 4500.0
                ...
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        moments_config = config['moments']
        
        # 按配置文件顺序提取矩
        for moment_name, moment_info in moments_config.items():
            self.target_moments[moment_name] = moment_info['value']
            self.moment_names.append(moment_name)
            self.moment_metadata[moment_name] = {
                'unit': moment_info.get('unit', ''),
                'source': moment_info.get('source', ''),
                'confidence_interval': moment_info.get(
                    'confidence_interval', 
                    None
                )
            }
    
    def get_target_moments(self) -> Dict[str, float]:
        """
        获取目标矩字典
        
        返回:
            {moment_name: value} 字典
        """
        return self.target_moments.copy()
    
    def get_target_vector(self) -> np.ndarray:
        """
        获取目标矩向量（按照moment_names顺序）
        
        返回:
            numpy数组，形状为(n_moments,)
        """
        return np.array([
            self.target_moments[name] for name in self.moment_names
        ])
    
    def compute_simulated_moments(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> Dict[str, float]:
        """
        从MFG均衡结果计算模拟矩
        
        参数:
            individuals: 个体均衡状态DataFrame，包含列：
                - employed: 就业状态（0/1）
                - W: 工资
                - T, S, D等状态变量
            eq_info: 均衡信息字典，包含：
                - final_statistics: 最终统计信息
        
        返回:
            {moment_name: value} 字典
        """
        simulated_moments = {}
        
        # 计算失业率
        if 'unemployment_rate' in self.moment_names:
            unemployment_rate = eq_info['final_statistics']['unemployment_rate']
            simulated_moments['unemployment_rate'] = unemployment_rate
        
        # 计算平均工资（仅就业者）
        if 'mean_wage' in self.moment_names:
            employed_individuals = individuals[individuals['employed'] == 1]
            mean_wage = employed_individuals['W'].mean()
            simulated_moments['mean_wage'] = mean_wage
        
        # 计算工资标准差（仅就业者）
        if 'std_wage' in self.moment_names:
            employed_individuals = individuals[individuals['employed'] == 1]
            std_wage = employed_individuals['W'].std()
            simulated_moments['std_wage'] = std_wage
        
        return simulated_moments
    
    def get_simulated_vector(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> np.ndarray:
        """
        计算模拟矩向量（按照moment_names顺序）
        
        参数:
            individuals: 个体均衡状态DataFrame
            eq_info: 均衡信息字典
        
        返回:
            numpy数组，形状为(n_moments,)
        """
        simulated_moments = self.compute_simulated_moments(
            individuals, 
            eq_info
        )
        
        return np.array([
            simulated_moments[name] for name in self.moment_names
        ])
    
    def compute_moment_difference(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> np.ndarray:
        """
        计算矩差异向量：m_sim - m_target
        
        参数:
            individuals: 个体均衡状态DataFrame
            eq_info: 均衡信息字典
        
        返回:
            差异向量，形状为(n_moments,)
        """
        target_vec = self.get_target_vector()
        simulated_vec = self.get_simulated_vector(individuals, eq_info)
        
        return simulated_vec - target_vec
    
    def get_moment_comparison(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> pd.DataFrame:
        """
        生成矩对比表（用于报告和诊断）
        
        参数:
            individuals: 个体均衡状态DataFrame
            eq_info: 均衡信息字典
        
        返回:
            DataFrame，包含列：
            - moment_name: 矩名称
            - target_value: 目标值
            - simulated_value: 模拟值
            - difference: 差异（sim - target）
            - relative_error: 相对误差（%）
            - unit: 单位
        """
        target_moments = self.get_target_moments()
        simulated_moments = self.compute_simulated_moments(
            individuals, 
            eq_info
        )
        
        comparison_data = []
        
        for name in self.moment_names:
            target_val = target_moments[name]
            sim_val = simulated_moments[name]
            diff = sim_val - target_val
            
            # 计算相对误差（避免除零）
            if abs(target_val) > 1e-10:
                rel_error = (diff / target_val) * 100
            else:
                rel_error = np.nan
            
            comparison_data.append({
                'moment_name': name,
                'target_value': target_val,
                'simulated_value': sim_val,
                'difference': diff,
                'relative_error': rel_error,
                'unit': self.moment_metadata[name]['unit']
            })
        
        return pd.DataFrame(comparison_data)
    
    def print_moment_comparison(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> None:
        """
        打印矩对比表（格式化输出）
        
        参数:
            individuals: 个体均衡状态DataFrame
            eq_info: 均衡信息字典
        """
        comparison_df = self.get_moment_comparison(individuals, eq_info)
        
        print("\n" + "="*80)
        print("矩对比分析")
        print("="*80)
        
        for _, row in comparison_df.iterrows():
            print(f"\n{row['moment_name']}:")
            print(f"  目标值: {row['target_value']:.4f} {row['unit']}")
            print(f"  模拟值: {row['simulated_value']:.4f} {row['unit']}")
            print(f"  差异: {row['difference']:.4f}")
            
            if not np.isnan(row['relative_error']):
                print(f"  相对误差: {row['relative_error']:.2f}%")
        
        print("\n" + "="*80)
    
    def get_n_moments(self) -> int:
        """
        获取矩的数量
        
        返回:
            矩的总数
        """
        return len(self.moment_names)
    
    def get_moment_names(self) -> List[str]:
        """
        获取矩名称列表
        
        返回:
            矩名称列表（保持配置文件中的顺序）
        """
        return self.moment_names.copy()

