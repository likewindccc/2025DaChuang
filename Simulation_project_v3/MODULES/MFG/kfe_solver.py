#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KFE演化求解器（Numba加速版本）

【实现方法说明】
研究计划中的KFE是理论公式：m^U(x,t) 表示状态空间上的密度函数。
本实现采用【蒙特卡洛模拟】：不显式计算密度函数，而是模拟N个具体个体。
数学等价性：大数定律保证经验分布收敛到理论密度函数。
优势：避免维度灾难（4D网格需10^8个点），易于实现，Numba加速性能优异。
"""
"""
KFE演化求解器（Numba加速版本）

基于个体的蒙特卡洛方法模拟人口分布的动态演化。

设计架构：
1. Numba加速的核心演化函数（纯NumPy数组，@njit装饰）
2. Python包装类（数据准备、模型调用、结果整理）

核心功能：
1. 根据贝尔曼求解器得到的最优策略a*，更新个体状态
2. 根据匹配概率λ和离职率μ，更新个体就业状态
3. 计算宏观统计量（失业率、市场紧张度）

研究计划公式（4.1.2节）：
- 失业者分布演化：
  m^U(x_{t+1}, t+1) = Σ[(1-λ)*I(x_{t+1}|x_t,a*)*m^U(x_t,t)] + μ*m^E(x_t,t)
  
- 就业者分布演化：
  m^E(x_{t+1}, t+1) = Σ[λ*I(x_{t+1}|x_t,a*)*m^U(x_t,t)] + (1-μ)*m^E(x_t,t)
  
其中：
- λ：匹配成功概率（失业→就业）
- μ：外生离职率（就业→失业）
- I(x_{t+1}|x_t,a*)：示性函数，表示状态转移
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from numba import njit, prange


# =============================================================================
# Numba加速的核心演化函数
# =============================================================================

@njit
def simulate_employment_transition(
    is_unemployed: bool,
    lambda_prob: float,
    mu_prob: float
) -> bool:
    """
    模拟就业状态转换（Numba加速）
    
    转换规则：
    - 失业者：以概率λ转为就业，以概率(1-λ)保持失业
    - 就业者：以概率μ转为失业，以概率(1-μ)保持就业
    
    参数：
        is_unemployed: 当前是否失业
        lambda_prob: 匹配成功概率（仅失业者使用）
        mu_prob: 离职率（仅就业者使用）
        
    返回：
        下一期是否失业
    """
    if is_unemployed:
        # 失业者：以概率λ找到工作
        # np.random.random() 返回 [0,1) 之间的均匀分布随机数
        if np.random.random() < lambda_prob:
            return False  # 转为就业
        else:
            return True   # 保持失业
    else:
        # 就业者：以概率μ离职
        if np.random.random() < mu_prob:
            return True   # 转为失业
        else:
            return False  # 保持就业


@njit(parallel=True)
def simulate_population_evolution(
    # 当前状态
    employment_status: np.ndarray,  # shape: (N,), True=失业, False=就业
    T: np.ndarray, S: np.ndarray, D: np.ndarray, W: np.ndarray,  # shape: (N,)
    current_wage: np.ndarray,  # shape: (N,), 就业者当前工资
    age: np.ndarray, edu: np.ndarray, children: np.ndarray,  # 固定特征
    # 策略与概率
    optimal_effort: np.ndarray,  # shape: (N,), 失业者的最优努力
    lambda_probs: np.ndarray,    # shape: (N,), 失业者的匹配概率
    mu_probs: np.ndarray,        # shape: (N,), 就业者的离职率
    # 状态更新参数
    T_max_population: float, W_min_population: float,
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    模拟整个人口的演化（Numba并行加速）
    
    对每个个体：
    1. 更新就业状态（失业/就业转换）
    2. 更新状态变量 (T, S, D, W)
    3. 更新当前工资（如果新匹配）
    
    【重要】：匹配成功后，就业收入 = 个体的期望工资W（状态变量）
    
    返回：
        (employment_status_new, T_new, S_new, D_new, W_new, current_wage_new)
    """
    N = len(employment_status)
    
    # 初始化下一期状态
    employment_status_new = np.empty(N, dtype=np.bool_)
    T_new = np.zeros(N)
    S_new = np.zeros(N)
    D_new = np.zeros(N)
    W_new = np.zeros(N)
    current_wage_new = np.zeros(N)
    
    # 并行循环：对每个个体进行演化
    for i in prange(N):
        is_unemployed = employment_status[i]
        
        # 1. 更新就业状态
        if is_unemployed:
            # 失业者：尝试匹配
            lambda_i = lambda_probs[i]
            employment_status_new[i] = simulate_employment_transition(
                True, lambda_i, 0.0
            )
            
            # 2. 更新状态变量（使用最优努力）
            a_opt = optimal_effort[i]
            T_new[i] = T[i] + gamma_T * a_opt * (T_max_population - T[i])
            W_new[i] = max(W_min_population, W[i] - gamma_W * a_opt)
            S_new[i] = S[i] + gamma_S * a_opt * (1.0 - S[i])
            D_new[i] = D[i] + gamma_D * a_opt * (1.0 - D[i])
            
            # 3. 更新当前工资
            if not employment_status_new[i]:
                # 匹配成功，就业收入 = 个体的期望工资W
                current_wage_new[i] = W_new[i]
            else:
                # 仍然失业
                current_wage_new[i] = 0.0
        else:
            # 就业者：面临离职风险
            mu_i = mu_probs[i]
            employment_status_new[i] = simulate_employment_transition(
                False, 0.0, mu_i
            )
            
            # 2. 状态变量（就业者不付出努力，状态不变）
            T_new[i] = T[i]
            S_new[i] = S[i]
            D_new[i] = D[i]
            W_new[i] = W[i]
            
            # 3. 更新当前工资
            if employment_status_new[i]:
                # 离职，失去工资
                current_wage_new[i] = 0.0
            else:
                # 保持就业，工资不变
                current_wage_new[i] = current_wage[i]
    
    return (employment_status_new, T_new, S_new, D_new, W_new,
            current_wage_new)


# =============================================================================
# Python包装类
# =============================================================================

class KFESolver:
    """
    KFE演化求解器（Python包装）
    
    负责：
    1. 准备演化所需的概率（λ, μ）
    2. 调用Numba加速的演化函数
    3. 计算宏观统计量（失业率、市场紧张度）
    4. 整理结果（NumPy数组 → DataFrame）
    """
    
    def __init__(self, config: Dict, match_function_model):
        """
        初始化KFE求解器
        
        参数:
            config: MFG配置字典
            match_function_model: 已训练的匹配函数模型
        """
        self.config = config
        self.match_model = match_function_model
        
        # 状态更新系数
        self.gamma_T = config['economics']['state_update']['gamma_T']
        self.gamma_W = config['economics']['state_update']['gamma_W']
        self.gamma_S = config['economics']['state_update']['gamma_S']
        self.gamma_D = config['economics']['state_update']['gamma_D']
        
        # 离职率参数
        sep = config['economics']['separation_rate']
        self.eta0 = sep['eta0']
        self.eta_T = sep['eta_T']
        self.eta_S = sep['eta_S']
        self.eta_D = sep['eta_D']
        self.eta_W = sep['eta_W']
        self.eta_age = sep['eta_age']
        self.eta_edu = sep['eta_edu']
        self.eta_children = sep['eta_children']
        
        # 市场参数
        self.vacancy = config['market']['vacancy']
    
    def compute_separation_rates(
        self,
        individuals: pd.DataFrame
    ) -> np.ndarray:
        """
        计算就业者的离职率（基于标准化变量）
        
        公式：μ(x, σ_i) = 1 / (1 + exp(-η'Z_std))
        其中Z_std是标准化后的变量
        
        步骤：
        1. 计算群体层面的均值和标准差
        2. 对每个个体的变量进行标准化
        3. 计算离职率
        
        参数:
            individuals: 个体DataFrame
            
        返回:
            mu_probs: shape (N,), 每个个体的离职率
        """
        N = len(individuals)
        mu_probs = np.zeros(N)
        
        # 计算群体统计量（用于标准化）
        T_mean, T_std = individuals['T'].mean(), individuals['T'].std()
        S_mean, S_std = individuals['S'].mean(), individuals['S'].std()
        D_mean, D_std = individuals['D'].mean(), individuals['D'].std()
        W_mean, W_std = individuals['W'].mean(), individuals['W'].std()
        age_mean, age_std = individuals['age'].mean(), individuals['age'].std()
        edu_mean, edu_std = individuals['education'].mean(), individuals['education'].std()
        children_mean, children_std = individuals['children'].mean(), individuals['children'].std()
        
        for idx, row in individuals.iterrows():
            # 仅就业者需要计算离职率
            if row['employment_status'] == 'employed':
                # 标准化变量
                T_std_val = (row['T'] - T_mean) / (T_std + 1e-10)
                S_std_val = (row['S'] - S_mean) / (S_std + 1e-10)
                D_std_val = (row['D'] - D_mean) / (D_std + 1e-10)
                W_std_val = (row['W'] - W_mean) / (W_std + 1e-10)
                age_std_val = (row['age'] - age_mean) / (age_std + 1e-10)
                edu_std_val = (row['education'] - edu_mean) / (edu_std + 1e-10)
                children_std_val = (row['children'] - children_mean) / (children_std + 1e-10)
                
                # 计算线性组合
                z = (self.eta0 +
                     self.eta_T * T_std_val +
                     self.eta_S * S_std_val +
                     self.eta_D * D_std_val +
                     self.eta_W * W_std_val +
                     self.eta_age * age_std_val +
                     self.eta_edu * edu_std_val +
                     self.eta_children * children_std_val)
                
                # Logistic函数
                mu = 1.0 / (1.0 + np.exp(-z))
                mu_probs[idx] = mu
            else:
                # 失业者的离职率为0（不适用）
                mu_probs[idx] = 0.0
        
        return mu_probs
    
    def compute_match_probabilities(
        self,
        individuals: pd.DataFrame,
        optimal_effort: pd.Series,
        theta: float
    ) -> np.ndarray:
        """
        计算失业者的匹配概率
        
        【与BellmanSolver区别】：
        - BellmanSolver.compute_match_probabilities_batch()：
          为每个个体 × 每个努力水平a 计算λ → 返回 (N, 11) 矩阵
          用于值迭代中枚举所有a找最优
          
        - KFESolver.compute_match_probabilities()：
          为每个个体仅计算最优努力a* 对应的λ → 返回 (N,) 向量
          用于人口演化时的随机转换
        
        参数:
            individuals: 个体DataFrame
            optimal_effort: 失业者的最优努力（从BellmanSolver获得）
            theta: 市场紧张度
            
        返回:
            lambda_probs: shape (N,), 每个个体的匹配概率
        """
        N = len(individuals)
        lambda_probs = np.zeros(N)
        
        # 预先计算sigma（控制变量综合指标）
        age_raw = individuals['age'].values
        edu_raw = individuals['education'].values
        children_raw = individuals['children'].values
        
        # 第一次MinMax标准化
        age_min, age_max = age_raw.min(), age_raw.max()
        age_norm = (age_raw - age_min) / (age_max - age_min + 1e-10)
        
        edu_min, edu_max = edu_raw.min(), edu_raw.max()
        edu_norm = (edu_raw - edu_min) / (edu_max - edu_min + 1e-10)
        
        children_min, children_max = children_raw.min(), children_raw.max()
        children_norm = (children_raw - children_min) / (
            children_max - children_min + 1e-10
        )
        
        # 求和
        sigma_sum = age_norm + edu_norm + children_norm
        
        # 第二次MinMax标准化
        sigma_min, sigma_max = sigma_sum.min(), sigma_sum.max()
        sigma = (sigma_sum - sigma_min) / (sigma_max - sigma_min + 1e-10)
        
        # 计算群体统计边界（用于状态更新）
        T_max_population = individuals['T'].max()
        W_min_population = individuals['W'].min()
        
        # 对每个失业者计算匹配概率
        for idx, row in individuals.iterrows():
            if row['employment_status'] == 'unemployed':
                # 根据最优努力更新状态
                a_opt = optimal_effort[idx]
                
                T_new = (row['T'] + self.gamma_T * a_opt *
                        (T_max_population - row['T']))
                S_new = row['S'] + self.gamma_S * a_opt * (1.0 - row['S'])
                D_new = row['D'] + self.gamma_D * a_opt * (1.0 - row['D'])
                W_new = max(W_min_population, row['W'] - self.gamma_W * a_opt)
                
                # 构建预测输入
                X = pd.DataFrame([{
                    'const': 1.0,
                    'T': T_new,
                    'S': S_new,
                    'D': D_new,
                    'W': W_new,
                    'sigma': sigma[idx],
                    'theta': theta
                }])
                
                # 预测匹配概率
                lambda_probs[idx] = self.match_model.predict(X).iloc[0]
            else:
                # 就业者的匹配概率为0（不适用）
                lambda_probs[idx] = 0.0
        
        return lambda_probs
    
    def evolve(
        self,
        individuals: pd.DataFrame,
        optimal_effort: pd.Series,
        theta: float
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        执行一期人口演化（主接口）
        
        参数:
            individuals: 当期个体DataFrame，必须包含：
                - employment_status: 'unemployed'/'employed'
                - T, S, D, W: 状态变量
                - current_wage: 就业者当前工资
                - age, education, children: 固定特征
            optimal_effort: 失业者的最优努力（从BellmanSolver获得）
            theta: 当期市场紧张度
            
        返回:
            (individuals_next, statistics): 下一期个体DataFrame和统计信息
        """
        N = len(individuals)
        
        # 1. 计算概率
        print("计算匹配概率和离职率...")
        lambda_probs = self.compute_match_probabilities(
            individuals, optimal_effort, theta
        )
        mu_probs = self.compute_separation_rates(individuals)
        
        # 2. 准备NumPy数组
        employment_status = (
            individuals['employment_status'] == 'unemployed'
        ).values
        
        T = individuals['T'].values
        S = individuals['S'].values
        D = individuals['D'].values
        W = individuals['W'].values
        current_wage = individuals['current_wage'].values
        age = individuals['age'].values
        edu = individuals['education'].values
        children = individuals['children'].values
        
        # 群体统计边界
        T_max_population = T.max()
        W_min_population = W.min()
        
        # 3. 调用Numba加速的演化函数
        print("开始人口演化（Numba加速）...")
        (employment_status_new, T_new, S_new, D_new, W_new,
         current_wage_new) = simulate_population_evolution(
            employment_status, T, S, D, W, current_wage,
            age, edu, children,
            optimal_effort.values, lambda_probs, mu_probs,
            T_max_population, W_min_population,
            self.gamma_T, self.gamma_W, self.gamma_S, self.gamma_D
        )
        
        # 4. 整理为DataFrame
        individuals_next = individuals.copy()
        individuals_next['employment_status'] = np.where(
            employment_status_new, 'unemployed', 'employed'
        )
        individuals_next['T'] = T_new
        individuals_next['S'] = S_new
        individuals_next['D'] = D_new
        individuals_next['W'] = W_new
        individuals_next['current_wage'] = current_wage_new
        
        # 5. 计算宏观统计量
        n_unemployed = employment_status_new.sum()
        n_employed = N - n_unemployed
        unemployment_rate = n_unemployed / N
        theta_next = self.vacancy / n_unemployed if n_unemployed > 0 else np.inf
        
        statistics = {
            'n_unemployed': n_unemployed,
            'n_employed': n_employed,
            'unemployment_rate': unemployment_rate,
            'theta': theta_next,
            'mean_T': T_new.mean(),
            'mean_S': S_new.mean(),
            'mean_D': D_new.mean(),
            'mean_W': W_new.mean(),
            'mean_wage_employed': current_wage_new[~employment_status_new].mean()
        }
        
        print(f"演化完成：失业率={unemployment_rate:.2%}, "
              f"市场紧张度θ={theta_next:.3f}")
        
        return individuals_next, statistics

