#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG均衡求解器（主控制器）

实现Bellman方程和KFE的交替迭代，求解平均场博弈的稳态均衡。

算法流程（研究计划4.6节）：
1. 初始化：生成N个个体，所有人失业，然后随机匹配一次
2. 外层迭代（直到收敛）：
   a) 计算市场紧张度 θ_t = V / U_t
   b) 求解Bellman方程 → 得到最优策略 a*(x) 和价值函数 V(x)
   c) 用a*求解KFE → 更新人口分布 m_{t+1}
   d) 检查收敛：
      - 价值函数相对变化：|ΔV|/|V| < ε_V (0.01)
      - 平均努力水平变化：|mean(a_{t+1}) - mean(a_t)| < ε_a (0.01)
      - 失业率变化：|u_{t+1} - u_t| < ε_u (0.001)

收敛后得到平均场均衡（MFE）：个体策略与市场状态自洽。
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

from .bellman_solver import BellmanSolver, load_match_function_model
from .kfe_solver import KFESolver


class EquilibriumSolver:
    """
    MFG均衡求解器
    
    负责：
    1. 初始化人口（基于POPULATION模块的分布）
    2. 协调Bellman和KFE的交替迭代
    3. 监控收敛过程
    4. 保存均衡结果
    """
    
    def __init__(
        self, 
        config_path: str, 
        population_adjustment: Optional[Dict] = None,
        save_results: bool = True
    ):
        """
        初始化均衡求解器
        
        参数:
            config_path: MFG配置文件路径
            population_adjustment: 人口分布调整参数（可选）
            save_results: 是否保存结果文件（并行校准时应设为False）
        """
        self.save_results = save_results
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载匹配函数模型
        model_path = self.config['paths']['match_function_model']
        self.match_model = load_match_function_model(model_path)
        
        # 初始化Bellman和KFE求解器
        self.bellman_solver = BellmanSolver(self.config, self.match_model)
        self.kfe_solver = KFESolver(self.config, self.match_model)
        
        # 提取参数
        self.n_individuals = self.config['population']['n_individuals']
        self.target_theta = self.config['market']['target_theta']  # 【修改】使用外生市场紧张度
        self.max_outer_iter = self.config['equilibrium']['max_outer_iter']
        self.damping_factor = self.config['equilibrium']['damping_factor']  # 【新增】阻尼因子
        self.epsilon_V = self.config['equilibrium']['convergence']['epsilon_V']
        self.epsilon_a = self.config['equilibrium']['convergence']['epsilon_a']
        self.epsilon_u = self.config['equilibrium']['convergence']['epsilon_u']
        self.use_relative_tol = self.config['equilibrium']['convergence']['use_relative_tol']  # 【新增】相对阈值标志
        
        # 【新增】人口分布调整参数
        self.population_adjustment = population_adjustment
        
        # 输出目录
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 历史记录
        self.history = {
            'iteration': [],
            'theta': [],
            'unemployment_rate': [],
            'mean_T': [],
            'mean_S': [],
            'mean_D': [],
            'mean_W': [],
            'mean_wage_employed': [],
            'mean_value_U': [],
            'mean_value_E': [],
            'mean_effort': [],
            'convergence_V': [],
            'convergence_a': [],
            'convergence_u': []
        }
    
    def initialize_population(self) -> pd.DataFrame:
        """
        初始化人口
        
        步骤（研究计划市场初始化）：
        1. 从POPULATION模块的分布中采样N个个体
        2. 所有个体初始状态为失业（employment_status='unemployed'）
        3. 运行一次随机匹配（effort=0，基于匹配函数λ）
        4. 根据匹配结果确定初始就业状态
        
        返回:
            individuals: DataFrame，包含所有个体的初始状态
        """
        print("=" * 80)
        print("初始化人口")
        print("=" * 80)
        
        # 加载劳动力分布模型
        from MODULES.POPULATION import LaborDistribution
        pop_config_path = "CONFIG/population_config.yaml"
        with open(pop_config_path, 'r', encoding='utf-8') as f:
            pop_config = yaml.safe_load(f)
        
        labor_model = LaborDistribution(pop_config)
        labor_model.fit()
        
        # 采样N个个体
        print(f"从人口分布中采样 {self.n_individuals} 个个体...")
        
        # 采样连续变量（T, S, D, W, age）
        continuous_samples = labor_model.copula_model.sample(self.n_individuals)
        
        # 采样离散变量（education, children）
        edu_values = list(labor_model.discrete_dist['edu'].keys())
        edu_probs = list(labor_model.discrete_dist['edu'].values())
        edu_samples = np.random.choice(
            edu_values, size=self.n_individuals, p=edu_probs
        )
        
        children_values = list(labor_model.discrete_dist['children'].keys())
        children_probs = list(labor_model.discrete_dist['children'].values())
        children_samples = np.random.choice(
            children_values, size=self.n_individuals, p=children_probs
        )
        
        # 组合为DataFrame
        individuals = continuous_samples.copy()
        individuals['education'] = edu_samples
        individuals['children'] = children_samples
        
        # 【新增】记录每个个体的初始T值（作为其理想工作时间）
        self.initial_T = individuals['T'].values.copy()
        print(f"记录初始T值：均值 = {self.initial_T.mean():.2f} 小时/周")
        print()
        
        # 【新增】应用人口分布调整（如果有）
        if self.population_adjustment is not None:
            print("应用人口分布调整（培训政策）...")
            if 'mean_S_multiplier' in self.population_adjustment:
                multiplier = self.population_adjustment['mean_S_multiplier']
                individuals['S'] = individuals['S'] * multiplier
                print(f"  技能水平S × {multiplier}")
            
            if 'mean_D_multiplier' in self.population_adjustment:
                multiplier = self.population_adjustment['mean_D_multiplier']
                individuals['D'] = individuals['D'] * multiplier
                print(f"  数字素养D × {multiplier}")
            print()
        
        # 初始化就业状态（所有人失业）
        individuals['employment_status'] = 'unemployed'
        individuals['current_wage'] = 0.0
        
        print(f"初始化完成：{self.n_individuals} 个个体，全部失业")
        print()
        
        # 运行一次初始匹配（effort=0）
        print("运行初始随机匹配...")
        initial_effort = pd.Series(
            np.zeros(self.n_individuals), 
            index=individuals.index
        )
        
        # 【修改】使用外生市场紧张度
        theta_initial = self.target_theta
        print(f"初始市场紧张度 θ = {theta_initial:.4f} (外生参数)")
        
        # 计算匹配概率
        lambda_probs = self.kfe_solver.compute_match_probabilities(
            individuals, initial_effort, theta_initial
        )
        
        # 模拟随机匹配
        matched_mask = np.random.random(self.n_individuals) < lambda_probs
        n_matched = matched_mask.sum()
        
        # 更新就业状态
        individuals.loc[matched_mask, 'employment_status'] = 'employed'
        individuals.loc[matched_mask, 'current_wage'] = individuals.loc[
            matched_mask, 'W'
        ]
        
        n_employed = (individuals['employment_status'] == 'employed').sum()
        initial_u_rate = 1.0 - n_employed / self.n_individuals
        
        print(f"初始匹配完成：{n_matched} 人匹配成功")
        print(f"初始失业率 = {initial_u_rate*100:.2f}%")
        print()
        
        return individuals
    
    def solve(
        self, 
        individuals: Optional[pd.DataFrame] = None,
        verbose: bool = True,
        callback: Optional[callable] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        求解MFG均衡
        
        参数:
            individuals: 初始人口（如为None，则自动初始化）
            verbose: 是否输出详细信息
            callback: 进度回调函数，签名为callback(iteration, stats)
            
        返回:
            (individuals_equilibrium, equilibrium_info): 均衡状态和统计信息
        """
        # 初始化人口
        if individuals is None:
            individuals = self.initialize_population()
        
        print("=" * 80)
        print("开始MFG均衡求解")
        print("=" * 80)
        print(f"最大迭代轮数: {self.max_outer_iter}")
        print(f"阻尼因子: {self.damping_factor} (V_new = {self.damping_factor}*V_computed + {1-self.damping_factor}*V_old)")
        if self.use_relative_tol:
            print(f"收敛阈值: ε_V={self.epsilon_V} (相对), ε_a={self.epsilon_a}, ε_u={self.epsilon_u}")
        else:
            print(f"收敛阈值: ε_V={self.epsilon_V}, ε_a={self.epsilon_a}, ε_u={self.epsilon_u}")
        print()
        
        # 初始化上一轮的值（用于收敛判断）
        prev_V_U = None
        prev_V_E = None
        prev_a_optimal = None
        prev_u_rate = None
        
        # 外层MFG均衡迭代
        for outer_iter in range(self.max_outer_iter):
            if verbose:
                print(f"{'='*80}")
                print(f"外层迭代 {outer_iter + 1}/{self.max_outer_iter}")
                print(f"{'='*80}")
            
            # 计算当前统计量
            n_unemployed = (individuals['employment_status'] == 'unemployed').sum()
            n_employed = (individuals['employment_status'] == 'employed').sum()
            u_rate = n_unemployed / self.n_individuals
            
            # 【修改】使用外生市场紧张度（不再根据V/U计算）
            theta = self.target_theta
            
            if verbose:
                print(f"失业数: {n_unemployed}, 就业数: {n_employed}")
                print(f"失业率: {u_rate*100:.2f}%")
                print(f"市场紧张度 θ = {theta:.4f}")
                print()
            
            # 步骤1: 求解Bellman方程
            if verbose:
                print("步骤1: 求解Bellman方程...")
            
            V_U_computed, V_E_computed, a_optimal = self.bellman_solver.solve(
                individuals, theta, self.initial_T
            )
            
            # 【新增】阻尼更新机制：平滑价值函数变化
            if outer_iter > 0 and prev_V_U is not None:
                V_U = self.damping_factor * V_U_computed + (1 - self.damping_factor) * prev_V_U
                V_E = self.damping_factor * V_E_computed + (1 - self.damping_factor) * prev_V_E
                if verbose:
                    print(f"  应用阻尼更新（α={self.damping_factor}）")
            else:
                V_U = V_U_computed.copy()
                V_E = V_E_computed.copy()
            
            if verbose:
                mean_V_U = V_U[individuals['employment_status'] == 'unemployed'].mean()
                mean_V_E = V_E[individuals['employment_status'] == 'employed'].mean()
                mean_a = a_optimal[individuals['employment_status'] == 'unemployed'].mean()
                print(f"  平均失业价值函数: {mean_V_U:.2f}")
                print(f"  平均就业价值函数: {mean_V_E:.2f}")
                print(f"  平均最优努力: {mean_a:.4f}")
                print()
            
            # 步骤2: 求解KFE（人口演化）
            if verbose:
                print("步骤2: 求解KFE（人口演化）...")
            
            individuals_next, stats = self.kfe_solver.evolve(
                individuals, a_optimal, theta
            )
            
            if verbose:
                print(f"  演化后失业率: {stats['unemployment_rate']*100:.2f}%")
                print(f"  平均T: {stats['mean_T']:.2f}")
                print(f"  平均S: {stats['mean_S']:.2f}")
                print(f"  平均D: {stats['mean_D']:.2f}")
                print(f"  平均W: {stats['mean_W']:.2f}")
                print()
            
            # 步骤3: 检查收敛（先计算指标）
            if outer_iter > 0:
                # 计算价值函数变化
                diff_V_U_abs = np.abs(V_U - prev_V_U).max()
                diff_V_E_abs = np.abs(V_E - prev_V_E).max()
                
                # 【修改】使用相对阈值判断价值函数收敛
                if self.use_relative_tol:
                    # 相对变化：|ΔV| / (|V| + 1e-10)
                    V_U_magnitude = np.abs(V_U).mean() + 1e-10
                    V_E_magnitude = np.abs(V_E).mean() + 1e-10
                    diff_V_U_rel = diff_V_U_abs / V_U_magnitude
                    diff_V_E_rel = diff_V_E_abs / V_E_magnitude
                    diff_V = max(diff_V_U_rel, diff_V_E_rel)
                else:
                    # 绝对变化
                    diff_V = max(diff_V_U_abs, diff_V_E_abs)
                
                # 【修改】计算平均努力水平变化（而非最大值变化）
                # 原因：a是离散化的（步长0.1），单个体变化至少0.1，但平均值是连续的
                mean_a_current = a_optimal.mean()
                mean_a_prev = prev_a_optimal.mean()
                diff_a = abs(mean_a_current - mean_a_prev)
                
                # 计算失业率变化（绝对值）
                diff_u = abs(u_rate - prev_u_rate)
            else:
                # 第一轮没有前置
                diff_V = np.nan
                diff_a = np.nan
                diff_u = np.nan
            
            # 记录收敛指标（所有轮次都记录）
            self.history['convergence_V'].append(diff_V)
            self.history['convergence_a'].append(diff_a)
            self.history['convergence_u'].append(diff_u)
            
            # 判断是否收敛
            if outer_iter > 0:
                if verbose:
                    print(f"收敛检查:")
                    if self.use_relative_tol:
                        print(f"  |ΔV|/|V| = {diff_V:.6f} (阈值: {self.epsilon_V}, 相对)")
                    else:
                        print(f"  |ΔV| = {diff_V:.6f} (阈值: {self.epsilon_V})")
                    print(f"  |Δmean(a)| = {diff_a:.6f} (阈值: {self.epsilon_a})")
                    print(f"  |Δu| = {diff_u:.6f} (阈值: {self.epsilon_u})")
                    print()
                
                if (diff_V < self.epsilon_V and 
                    diff_a < self.epsilon_a and 
                    diff_u < self.epsilon_u):
                    print(f"{'='*80}")
                    print(f"均衡已收敛！迭代 {outer_iter + 1} 轮")
                    print(f"{'='*80}")
                    print(f"最终失业率: {u_rate*100:.2f}%")
                    print(f"最终市场紧张度: {theta:.4f}")
                    print()
                    
                    # 【修复】收敛时也要记录最后一次历史
                    self.history['iteration'].append(outer_iter + 1)
                    self.history['theta'].append(theta)
                    self.history['unemployment_rate'].append(u_rate)
                    self.history['mean_T'].append(stats['mean_T'])
                    self.history['mean_S'].append(stats['mean_S'])
                    self.history['mean_D'].append(stats['mean_D'])
                    self.history['mean_W'].append(stats['mean_W'])
                    self.history['mean_wage_employed'].append(stats.get('mean_wage_employed', 0))
                    
                    if n_unemployed > 0:
                        mean_V_U = V_U[individuals['employment_status'] == 'unemployed'].mean()
                        mean_a = a_optimal[individuals['employment_status'] == 'unemployed'].mean()
                    else:
                        mean_V_U = 0
                        mean_a = 0
                    
                    if n_employed > 0:
                        mean_V_E = V_E[individuals['employment_status'] == 'employed'].mean()
                    else:
                        mean_V_E = 0
                    
                    self.history['mean_value_U'].append(mean_V_U)
                    self.history['mean_value_E'].append(mean_V_E)
                    self.history['mean_effort'].append(mean_a)
                    
                    # 保存均衡状态
                    self._save_equilibrium(
                        individuals_next, V_U, V_E, a_optimal, 
                        outer_iter + 1, converged=True
                    )
                    
                    return individuals_next, {
                        'converged': True,
                        'iterations': outer_iter + 1,
                        'final_unemployment_rate': u_rate,
                        'final_theta': theta,
                        'final_statistics': stats,
                        'history': self.history
                    }
            
            # 记录历史
            self.history['iteration'].append(outer_iter + 1)
            self.history['theta'].append(theta)
            self.history['unemployment_rate'].append(u_rate)
            self.history['mean_T'].append(stats['mean_T'])
            self.history['mean_S'].append(stats['mean_S'])
            self.history['mean_D'].append(stats['mean_D'])
            self.history['mean_W'].append(stats['mean_W'])
            self.history['mean_wage_employed'].append(stats.get('mean_wage_employed', 0))
            
            if n_unemployed > 0:
                mean_V_U = V_U[individuals['employment_status'] == 'unemployed'].mean()
                mean_a = a_optimal[individuals['employment_status'] == 'unemployed'].mean()
            else:
                mean_V_U = 0
                mean_a = 0
            
            if n_employed > 0:
                mean_V_E = V_E[individuals['employment_status'] == 'employed'].mean()
            else:
                mean_V_E = 0
            
            self.history['mean_value_U'].append(mean_V_U)
            self.history['mean_value_E'].append(mean_V_E)
            self.history['mean_effort'].append(mean_a)
            
            # 调用进度回调函数（供GUI使用）
            if callback is not None:
                callback_stats = {
                    'unemployment_rate': u_rate,
                    'theta': theta,
                    'mean_wage': individuals['current_wage'].mean(),
                    'mean_T': stats['mean_T'],
                    'mean_S': stats['mean_S'],
                    'diff_V': diff_V if outer_iter > 0 else 0,
                    'diff_u': diff_u if outer_iter > 0 else 0
                }
                callback(outer_iter + 1, callback_stats)
            
            # 更新状态
            individuals = individuals_next.copy()
            prev_V_U = V_U.copy()
            prev_V_E = V_E.copy()
            prev_a_optimal = a_optimal.copy()
            prev_u_rate = u_rate
        
        # 未收敛
        print(f"{'='*80}")
        print(f"警告：达到最大迭代次数 {self.max_outer_iter}，均衡未收敛")
        print(f"{'='*80}")
        
        # 保存最终状态
        self._save_equilibrium(
            individuals, V_U, V_E, a_optimal, 
            self.max_outer_iter, converged=False
        )
        
        return individuals, {
            'converged': False,
            'iterations': self.max_outer_iter,
            'final_unemployment_rate': u_rate,
            'final_theta': theta,
            'final_statistics': stats,
            'history': self.history
        }
    
    def _save_equilibrium(
        self,
        individuals: pd.DataFrame,
        V_U: pd.Series,
        V_E: pd.Series,
        a_optimal: pd.Series,
        iterations: int,
        converged: bool
    ):
        """
        保存均衡结果
        
        参数:
            individuals: 均衡时的个体状态
            V_U, V_E: 价值函数
            a_optimal: 最优努力
            iterations: 迭代轮数
            converged: 是否收敛
        """
        if not self.save_results:
            return
        
        print("保存均衡结果...")
        
        # 保存个体状态
        individuals_path = self.output_dir / "equilibrium_individuals.csv"
        individuals.to_csv(individuals_path, index=False, encoding='utf-8-sig')
        print(f"  个体状态已保存至: {individuals_path}")
        
        # 保存价值函数和策略
        policy_df = pd.DataFrame({
            'V_U': V_U,
            'V_E': V_E,
            'a_optimal': a_optimal
        })
        policy_path = self.output_dir / "equilibrium_policy.csv"
        policy_df.to_csv(policy_path, index=True, encoding='utf-8-sig')
        print(f"  价值函数和策略已保存至: {policy_path}")
        
        # 保存历史记录
        history_df = pd.DataFrame(self.history)
        history_path = self.output_dir / "equilibrium_history.csv"
        history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
        print(f"  迭代历史已保存至: {history_path}")
        
        # 保存汇总信息
        summary = {
            'converged': converged,
            'iterations': iterations,
            'n_individuals': self.n_individuals,
            'target_theta': self.target_theta,  # 【修改】保存外生市场紧张度
            'final_unemployment_rate': self.history['unemployment_rate'][-1],
            'final_theta': self.history['theta'][-1]
        }
        
        summary_path = self.output_dir / "equilibrium_summary.pkl"
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f"  汇总信息已保存至: {summary_path}")
        print()


def solve_equilibrium(
    config_path: str = "CONFIG/mfg_config.yaml",
    population_adjustment: Optional[Dict] = None,
    save_results: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    求解MFG均衡的便捷函数
    
    参数:
        config_path: MFG配置文件路径
        population_adjustment: 人口分布调整参数（可选）
            - mean_S_multiplier: 技能水平S的倍数调整
            - mean_D_multiplier: 数字素养D的倍数调整
        
    返回:
        (individuals_equilibrium, equilibrium_info): 均衡状态和统计信息
    """
    solver = EquilibriumSolver(config_path, population_adjustment, save_results)
    return solver.solve()

