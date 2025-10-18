#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场模拟器

负责在不同政策场景下调用MFG求解器，进行批量模拟和结果对比。

核心功能：
1. 场景管理：定义和加载多个模拟场景
2. 配置调整：根据政策参数调整人口分布和MFG参数
3. 批量运行：依次运行所有场景并汇总结果
4. 结果保存：保存详细结果和对比汇总表
"""

import yaml
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, Optional

from MODULES.MFG import solve_equilibrium


class MarketSimulator:
    """
    市场模拟器
    
    管理多场景模拟，包括：
    - 加载场景配置
    - 调整参数（人口分布、MFG参数）
    - 批量运行MFG均衡求解
    - 汇总和保存结果
    """
    
    def __init__(self, config_path: str):
        """
        初始化市场模拟器
        
        参数:
            config_path: SIMULATOR配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 项目根目录
        self.root_dir = Path(__file__).parent.parent.parent
        
        # 配置文件路径
        self.mfg_config_path = self.root_dir / 'CONFIG' / 'mfg_config.yaml'
        self.mfg_config_backup_path = self.root_dir / 'CONFIG' / 'mfg_config_backup.yaml'
        
        # 输出目录
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份原始配置文件
        self._backup_mfg_config()
    
    def _backup_mfg_config(self) -> None:
        """备份原始MFG配置文件"""
        shutil.copy(self.mfg_config_path, self.mfg_config_backup_path)
    
    def _restore_mfg_config(self) -> None:
        """恢复原始MFG配置文件"""
        shutil.copy(self.mfg_config_backup_path, self.mfg_config_path)
    
    def _adjust_population_params(self, adjustments: Optional[Dict]) -> Dict:
        """
        准备人口分布调整参数
        
        由于实际的人口分布是从数据拟合的Copula模型，无法直接修改配置文件中的均值参数。
        因此，这个方法只是返回调整参数，让MFG模块在采样后进行调整。
        
        参数:
            adjustments: 调整参数字典，包含mean_S_multiplier和mean_D_multiplier
        
        返回:
            调整参数字典，如果没有调整则返回None
        """
        return adjustments
    
    def _adjust_mfg_params(self, adjustments: Optional[Dict]) -> None:
        """
        根据政策调整MFG状态更新系数
        
        参数:
            adjustments: 调整参数字典，包含gamma_S_multiplier和gamma_D_multiplier
        """
        if adjustments is None:
            return
        
        # 读取MFG配置
        with open(self.mfg_config_path, 'r', encoding='utf-8') as f:
            mfg_config = yaml.safe_load(f)
        
        # 应用调整
        if 'gamma_S_multiplier' in adjustments:
            original_gamma_S = mfg_config['economics']['state_update']['gamma_S']
            mfg_config['economics']['state_update']['gamma_S'] = \
                original_gamma_S * adjustments['gamma_S_multiplier']
        
        if 'gamma_D_multiplier' in adjustments:
            original_gamma_D = mfg_config['economics']['state_update']['gamma_D']
            mfg_config['economics']['state_update']['gamma_D'] = \
                original_gamma_D * adjustments['gamma_D_multiplier']
        
        # 保存修改后的配置
        with open(self.mfg_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(mfg_config, f, allow_unicode=True, default_flow_style=False)
    
    
    def run_scenario(
        self, 
        scenario_name: str, 
        scenario_config: Dict
    ) -> Dict:
        """
        运行单个场景
        
        参数:
            scenario_name: 场景名称
            scenario_config: 场景配置字典
        
        返回:
            场景结果字典，包含所有关键指标
        """
        print(f"\n{'='*80}")
        print(f"运行场景: {scenario_config['name']}")
        print(f"描述: {scenario_config['description']}")
        print(f"{'='*80}")
        
        # 1. 恢复原始配置文件（确保每个场景独立）
        self._restore_mfg_config()
        
        # 2. 准备人口分布调整参数（如果有）
        # 这些参数将传递给MFG模块，在采样后应用调整
        population_adjustment = self._adjust_population_params(
            scenario_config.get('population_adjustment')
        )
        
        # 3. 应用状态更新系数调整（如果有）
        # 通过修改gamma系数实现"提升学习能力"
        self._adjust_mfg_params(scenario_config.get('state_update_adjustment'))
        
        # 4. 运行MFG均衡求解
        # MFG会在初始化人口后应用population_adjustment
        individuals_eq, eq_info = solve_equilibrium(
            population_adjustment=population_adjustment
        )
        
        # 5. 立即恢复配置文件，避免影响下一个场景
        self._restore_mfg_config()
        
        # 6. 读取价值函数和策略（a_optimal单独保存在policy文件中）
        policy_path = self.root_dir / 'OUTPUT' / 'mfg' / 'equilibrium_policy.csv'
        policy_df = pd.read_csv(policy_path)
        
        # 7. 提取关键结果
        result = {
            'scenario_name': scenario_name,
            'scenario_display_name': scenario_config['name'],
            'policy_type': scenario_config.get('policy_type', 'none'),
            'converged': eq_info['converged'],
            'iterations': eq_info['iterations'],
            'unemployment_rate': eq_info['final_statistics']['unemployment_rate'],
            'mean_T': eq_info['final_statistics']['mean_T'],
            'mean_S': eq_info['final_statistics']['mean_S'],
            'mean_D': eq_info['final_statistics']['mean_D'],
            'mean_W': eq_info['final_statistics']['mean_W'],
            'mean_wage_employed': eq_info['final_statistics']['mean_wage_employed'],
            'mean_effort': policy_df['a_optimal'].mean(),
            'mean_value_U': policy_df['V_U'].mean(),
            'mean_value_E': policy_df['V_E'].mean(),
        }
        
        # 8. 保存详细结果（如果需要）
        if self.config['output']['save_detailed_results']:
            self._save_scenario_results(scenario_name, individuals_eq, eq_info)
        
        print(f"\n场景 '{scenario_config['name']}' 运行完成")
        print(f"  失业率: {result['unemployment_rate']*100:.2f}%")
        print(f"  平均工资: {result['mean_wage_employed']:.2f}")
        print(f"  平均努力: {result['mean_effort']:.4f}")
        
        return result
    
    def _save_scenario_results(
        self, 
        scenario_name: str, 
        individuals: pd.DataFrame,
        eq_info: Dict
    ) -> None:
        """
        保存单个场景的详细结果
        
        参数:
            scenario_name: 场景名称
            individuals: 个体均衡状态数据
            eq_info: 均衡求解信息
        """
        # 创建场景输出目录
        scenario_dir = self.output_dir / f'scenario_{scenario_name}'
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存个体均衡状态
        individuals.to_csv(
            scenario_dir / 'equilibrium_individuals.csv', 
            index=False
        )
        
        # 保存迭代历史
        if 'history' in eq_info:
            history_df = pd.DataFrame(eq_info['history'])
            history_df.to_csv(
                scenario_dir / 'equilibrium_history.csv', 
                index=False
            )
        
        # 保存汇总信息
        summary = {
            'scenario_name': scenario_name,
            'converged': eq_info['converged'],
            'iterations': eq_info['iterations'],
            'final_statistics': eq_info['final_statistics']
        }
        
        import pickle
        with open(scenario_dir / 'equilibrium_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
    
    def run_batch(self) -> pd.DataFrame:
        """
        批量运行所有场景
        
        遍历配置中的所有场景，依次运行并汇总结果。
        
        返回:
            场景对比汇总表 DataFrame
        """
        print("\n" + "="*80)
        print("开始批量场景模拟")
        print("="*80)
        
        results = []
        
        # 遍历所有场景
        for scenario_name, scenario_config in self.config['scenarios'].items():
            result = self.run_scenario(scenario_name, scenario_config)
            results.append(result)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存对比汇总表
        if self.config['output']['save_comparison_table']:
            comparison_path = self.output_dir / 'scenario_comparison.csv'
            results_df.to_csv(comparison_path, index=False)
            print(f"\n场景对比汇总表已保存至: {comparison_path}")
        
        # 恢复原始配置（确保最终状态正确）
        self._restore_mfg_config()
        
        print("\n" + "="*80)
        print("批量场景模拟完成")
        print("="*80)
        
        return results_df

