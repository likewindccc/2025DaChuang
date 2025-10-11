#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝尔曼方程求解器（Numba加速版本）

基于个体的蒙特卡洛方法求解失业者和就业者的值函数和最优努力策略。

设计架构：
1. Numba加速的核心计算函数（纯NumPy数组，@njit装饰）
2. Python包装类（数据准备、模型调用、结果整理）

性能优化：
- 核心双层循环使用@njit加速
- 匹配概率λ预先批量计算
- 值迭代主循环在numba内部

研究计划公式（4.1.1节）：
- 失业者：V^U_t(x,σ_i) = max_a {[b(x) - 0.5*κ*a²] + ρ[λ*V^E_{t+1} + (1-λ)*V^U_{t+1}]}
- 就业者：V^E_t(x,σ_i) = ω(x,σ_i) + ρ[μ*V^U_{t+1} + (1-μ)*V^E_{t+1}]
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple
from numba import njit, prange


# =============================================================================
# Numba加速的核心计算函数
# =============================================================================

@njit
def update_state_numba(
    T: float, S: float, D: float, W: float,
    a: float,
    T_max_population: float,  # 群体中T的最大值
    W_min_population: float,  # 群体中W的最小值
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float
) -> Tuple[float, float, float, float]:
    """
    状态更新函数（Numba加速）
    
    研究计划公式（4.3节）：
    - T_{t+1} = T_t + γ_T*a_t*(T_max - T_t)
    - W_{t+1} = max(W_min, W_t - γ_W*a_t)
    - S_{t+1} = S_t + γ_S*a_t*(1 - S_t)
    - D_{t+1} = D_t + γ_D*a_t*(1 - D_t)
    
    其中：
    - T_max = 当前群体中所有失业者的T的最大值
    - W_min = 当前群体中所有失业者的W的最小值
    """
    # 状态更新（使用群体统计边界）
    T_new = T + gamma_T * a * (T_max_population - T)
    W_new = max(W_min_population, W - gamma_W * a)
    S_new = S + gamma_S * a * (1.0 - S)
    D_new = D + gamma_D * a * (1.0 - D)
    
    return T_new, S_new, D_new, W_new


@njit
def compute_separation_rate_numba(
    T: float, S: float, D: float, W: float,
    age: float, education: float, children: float,
    # 群体统计量（用于标准化）
    T_mean: float, T_std: float,
    S_mean: float, S_std: float,
    D_mean: float, D_std: float,
    W_mean: float, W_std: float,
    age_mean: float, age_std: float,
    edu_mean: float, edu_std: float,
    children_mean: float, children_std: float,
    # 系数
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float
) -> float:
    """
    计算外生离职率 μ(x, σ_i)（Numba加速，基于标准化变量）
    
    步骤：
    1. 先对所有变量进行标准化：x_std = (x - mean) / std
    2. 计算线性组合：z = η'Z_std
    3. 应用logistic函数：μ = 1 / (1 + exp(-z))
    """
    # 标准化
    T_std_val = (T - T_mean) / (T_std + 1e-10)
    S_std_val = (S - S_mean) / (S_std + 1e-10)
    D_std_val = (D - D_mean) / (D_std + 1e-10)
    W_std_val = (W - W_mean) / (W_std + 1e-10)
    age_std_val = (age - age_mean) / (age_std + 1e-10)
    edu_std_val = (education - edu_mean) / (edu_std + 1e-10)
    children_std_val = (children - children_mean) / (children_std + 1e-10)
    
    # 计算线性组合
    z = (eta0 + 
         eta_T * T_std_val + 
         eta_S * S_std_val + 
         eta_D * D_std_val + 
         eta_W * W_std_val +
         eta_age * age_std_val + 
         eta_edu * edu_std_val + 
         eta_children * children_std_val)
    
    # Logistic函数
    mu = 1.0 / (1.0 + np.exp(-z))
    return mu


@njit(parallel=True)
def solve_unemployed_bellman_numba(
    # 状态变量（N个失业者）
    T_U: np.ndarray,  # shape: (N_U,)
    S_U: np.ndarray,
    D_U: np.ndarray,
    W_U: np.ndarray,
    # 固定特征
    age_U: np.ndarray,
    edu_U: np.ndarray,
    children_U: np.ndarray,
    # 匹配概率（预计算）shape: (N_U, n_effort)
    lambda_probs: np.ndarray,
    # 下期值函数（上一轮迭代的近似值）
    # 注：值迭代算法的核心思想是用"上一轮的值函数"作为"下期值函数"的近似
    # 通过不断迭代，这个近似会逐渐收敛到真实的均衡值函数
    V_U_next: np.ndarray,  # shape: (N_U,) 上一轮迭代的失业值函数
    V_E_next: np.ndarray,  # shape: (N_U,) 上一轮迭代的就业值函数
    # 努力水平网格
    a_grid: np.ndarray,  # shape: (n_effort,)
    # 参数
    rho: float, kappa: float, b0: float,
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解失业者贝尔曼方程（Numba并行加速）
    
    值迭代逻辑：
    1. 输入的V_U_next和V_E_next是上一轮迭代的值函数（初始为全零）
    2. 基于这个"下期值函数的近似"，求解当前期的最优决策和值函数
    3. 在外层循环中不断迭代，直到值函数收敛
    
    对每个失业个体，枚举努力水平，找最优a*和V^U
    
    返回:
        (V_U_t, a_optimal): shape均为(N_U,) - 本轮迭代计算出的值函数和最优努力
    """
    N_U = len(T_U)
    n_effort = len(a_grid)
    
    V_U_t = np.zeros(N_U)
    a_optimal = np.zeros(N_U)
    
    # 并行循环遍历每个失业个体
    for i in prange(N_U):
        max_value = -np.inf
        best_a = 0.0
        
        # 失业收益（固定）
        b = b0
        
        # 枚举所有努力水平
        for j in range(n_effort):
            a = a_grid[j]
            
            # 努力成本
            effort_cost = 0.5 * kappa * a * a
            
            # 即时效用
            instant_utility = b - effort_cost
            
            # 匹配概率（预计算的）
            lambda_prob = lambda_probs[i, j]
            
            # 下期期望价值
            V_next_expected = (
                lambda_prob * V_E_next[i] +
                (1.0 - lambda_prob) * V_U_next[i]
            )
            
            # 总价值
            total_value = instant_utility + rho * V_next_expected
            
            # 更新最优值
            if total_value > max_value:
                max_value = total_value
                best_a = a
        
        V_U_t[i] = max_value
        a_optimal[i] = best_a
    
    return V_U_t, a_optimal


@njit(parallel=True)
def solve_employed_bellman_numba(
    # 状态变量（N个就业者）
    T_E: np.ndarray,
    S_E: np.ndarray,
    D_E: np.ndarray,
    W_E: np.ndarray,
    # 当前工资收入
    current_wage_E: np.ndarray,  # shape: (N_E,) 就业者当前的工资收入
    # 固定特征
    age_E: np.ndarray,
    edu_E: np.ndarray,
    children_E: np.ndarray,
    # 群体统计量（用于标准化）
    T_mean: float, T_std: float,
    S_mean: float, S_std: float,
    D_mean: float, D_std: float,
    W_mean: float, W_std: float,
    age_mean: float, age_std: float,
    edu_mean: float, edu_std: float,
    children_mean: float, children_std: float,
    # 下期值函数
    V_U_next: np.ndarray,  # shape: (N_E,)
    V_E_next: np.ndarray,
    # 参数
    rho: float,
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float
) -> np.ndarray:
    """
    求解就业者贝尔曼方程（Numba并行加速）
    
    公式：V^E_t = ω + ρ[μ*V^U_{t+1} + (1-μ)*V^E_{t+1}]
    其中：ω = 个体当前匹配到的企业工资（current_wage）
    
    返回:
        V_E_t: shape (N_E,)
    """
    N_E = len(T_E)
    V_E_t = np.zeros(N_E)
    
    # 并行循环
    for i in prange(N_E):
        # 就业效用 = 个体当前的工资收入
        omega = current_wage_E[i]
        
        # 计算离职率（需要传入标准化所需的统计量）
        mu = compute_separation_rate_numba(
            T_E[i], S_E[i], D_E[i], W_E[i],
            age_E[i], edu_E[i], children_E[i],
            T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
            age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        
        # 下期期望价值
        V_next_expected = mu * V_U_next[i] + (1.0 - mu) * V_E_next[i]
        
        # 当期价值
        V_E_t[i] = omega + rho * V_next_expected
    
    return V_E_t


@njit
def value_iteration_numba(
    # 失业者数据
    T_U: np.ndarray, S_U: np.ndarray, D_U: np.ndarray, W_U: np.ndarray,
    age_U: np.ndarray, edu_U: np.ndarray, children_U: np.ndarray,
    lambda_probs_U: np.ndarray,  # shape: (N_U, n_effort)
    # 就业者数据
    T_E: np.ndarray, S_E: np.ndarray, D_E: np.ndarray, W_E: np.ndarray,
    current_wage_E: np.ndarray,  # 就业者当前工资
    age_E: np.ndarray, edu_E: np.ndarray, children_E: np.ndarray,
    # 索引映射
    unemployed_indices: np.ndarray,  # shape: (N_U,) 失业者在总体中的索引
    employed_indices: np.ndarray,    # shape: (N_E,) 就业者在总体中的索引
    # 努力网格
    a_grid: np.ndarray,
    # 参数
    rho: float, kappa: float, b0: float,
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float,
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float,
    # 迭代参数
    max_iter: int, tol: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    值迭代主循环（Numba加速）
    
    算法逻辑：
    1. 初始化：V_U = V_E = 0（第0轮的"下期值函数"猜测）
    2. 第k轮迭代：
       a) 保存第k-1轮的值函数为V_old（作为"下期值函数"的近似）
       b) 基于V_old，求解贝尔曼方程，得到第k轮的新值函数V_new
       c) 检查收敛：max|V_new - V_old| < tol
    3. 重复迭代，直到收敛或达到最大迭代次数
    
    这是一个不动点迭代过程：V = T(V)，其中T是贝尔曼算子
    理论保证：在满足压缩映射条件下，该迭代必收敛到唯一不动点
    
    修正：所有N个个体都维护V_U和V_E，避免索引错误
    
    返回:
        (V_U, V_E, a_optimal, iterations, max_diff)
    """
    N = len(unemployed_indices) + len(employed_indices)
    N_U = len(T_U)
    N_E = len(T_E)
    
    # 【修正】所有个体都维护V_U和V_E（长度都是N）
    V_U_all = np.zeros(N)  # 所有个体的失业价值函数
    V_E_all = np.zeros(N)  # 所有个体的就业价值函数
    a_optimal = np.zeros(N_U)
    
    # 计算群体统计量（用于标准化）
    # 合并所有个体的数据来计算统计量
    all_T = np.concatenate((T_U, T_E)) if N_U > 0 and N_E > 0 else (T_U if N_U > 0 else T_E)
    all_S = np.concatenate((S_U, S_E)) if N_U > 0 and N_E > 0 else (S_U if N_U > 0 else S_E)
    all_D = np.concatenate((D_U, D_E)) if N_U > 0 and N_E > 0 else (D_U if N_U > 0 else D_E)
    all_W = np.concatenate((W_U, W_E)) if N_U > 0 and N_E > 0 else (W_U if N_U > 0 else W_E)
    all_age = np.concatenate((age_U, age_E)) if N_U > 0 and N_E > 0 else (age_U if N_U > 0 else age_E)
    all_edu = np.concatenate((edu_U, edu_E)) if N_U > 0 and N_E > 0 else (edu_U if N_U > 0 else edu_E)
    all_children = np.concatenate((children_U, children_E)) if N_U > 0 and N_E > 0 else (children_U if N_U > 0 else children_E)
    
    T_mean, T_std = all_T.mean(), all_T.std()
    S_mean, S_std = all_S.mean(), all_S.std()
    D_mean, D_std = all_D.mean(), all_D.std()
    W_mean, W_std = all_W.mean(), all_W.std()
    age_mean, age_std = all_age.mean(), all_age.std()
    edu_mean, edu_std = all_edu.mean(), all_edu.std()
    children_mean, children_std = all_children.mean(), all_children.std()
    
    for iteration in range(max_iter):
        # 保存上一轮的值函数（作为"下期值函数"的近似）
        V_U_old_all = V_U_all.copy()
        V_E_old_all = V_E_all.copy()
        
        # 【修正】基于V_old，求解失业者贝尔曼方程
        if N_U > 0:
            # 提取失业者的V_U_old和V_E_old（使用索引映射）
            V_U_old_for_unemployed = V_U_old_all[unemployed_indices]
            V_E_old_for_unemployed = V_E_old_all[unemployed_indices]
            
            V_U_new, a_optimal = solve_unemployed_bellman_numba(
                T_U, S_U, D_U, W_U, age_U, edu_U, children_U,
                lambda_probs_U, V_U_old_for_unemployed, V_E_old_for_unemployed, a_grid,
                rho, kappa, b0, gamma_T, gamma_W, gamma_S, gamma_D
            )
            
            # 将结果写回V_U_all（对应位置）
            V_U_all[unemployed_indices] = V_U_new
        
        # 【修正】基于V_old，求解就业者贝尔曼方程
        if N_E > 0:
            # 提取就业者的V_U_old和V_E_old（使用索引映射）
            V_U_old_for_employed = V_U_old_all[employed_indices]
            V_E_old_for_employed = V_E_old_all[employed_indices]
            
            V_E_new = solve_employed_bellman_numba(
                T_E, S_E, D_E, W_E, current_wage_E,
                age_E, edu_E, children_E,
                T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
                age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
                V_U_old_for_employed, V_E_old_for_employed, rho,
                eta0, eta_T, eta_S, eta_D, eta_W,
                eta_age, eta_edu, eta_children
            )
            
            # 将结果写回V_E_all（对应位置）
            V_E_all[employed_indices] = V_E_new
        
        # 【添加】数值稳定性：限制价值函数范围，防止溢出
        V_U_all = np.clip(V_U_all, -1e6, 1e6)
        V_E_all = np.clip(V_E_all, -1e6, 1e6)
        
        # 检查收敛性：max|V_new - V_old|
        diff_U = np.max(np.abs(V_U_all - V_U_old_all))
        diff_E = np.max(np.abs(V_E_all - V_E_old_all))
        max_diff = max(diff_U, diff_E)
        
        if max_diff < tol:
            return V_U_all, V_E_all, a_optimal, iteration + 1, max_diff
    
    # 未收敛
    return V_U_all, V_E_all, a_optimal, max_iter, max_diff


# =============================================================================
# Python包装类
# =============================================================================

class BellmanSolver:
    """
    贝尔曼方程求解器（Python包装）
    
    负责：
    1. 加载匹配函数模型
    2. 准备数据（DataFrame → NumPy数组）
    3. 批量计算匹配概率
    4. 调用Numba加速函数
    5. 整理结果（NumPy数组 → DataFrame/Series）
    """
    
    def __init__(self, config: Dict, match_function_model):
        """
        初始化求解器
        
        参数:
            config: MFG配置字典
            match_function_model: 已训练的匹配函数模型
        """
        self.config = config
        self.match_model = match_function_model
        
        # 提取参数
        self.rho = config['economics']['rho']
        self.kappa = config['economics']['kappa']
        self.b0 = config['economics']['unemployment_benefit']['b0']
        self.mean_wage = config['economics']['employment_utility']['mean_wage']
        
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
        
        # 努力网格
        self.a_grid = np.linspace(
            config['effort']['a_min'],
            config['effort']['a_max'],
            config['effort']['a_points']
        )
        
        # 迭代参数
        self.max_iter = config['value_iteration']['max_iter']
        self.tol = config['value_iteration']['tol']
    
    def compute_match_probabilities_batch(
        self,
        individuals: pd.DataFrame,
        a_grid: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """
        批量计算匹配概率
        
        对每个个体和每个努力水平，计算匹配概率λ
        
        参数:
            individuals: 个体DataFrame（必须包含age, education, children列）
            a_grid: 努力水平数组
            theta: 市场紧张度
            
        返回:
            lambda_probs: shape (N, n_effort)
        """
        N = len(individuals)
        n_effort = len(a_grid)
        lambda_probs = np.zeros((N, n_effort))
        
        # 预先计算sigma（控制变量综合指标）
        # 公式：σ = MinMax(MinMax(age) + MinMax(edu) + MinMax(children))
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
        
        for j, a in enumerate(a_grid):
            # 为当前努力水平更新所有个体状态
            T_new_list = []
            S_new_list = []
            D_new_list = []
            W_new_list = []
            
            for idx, row in individuals.iterrows():
                # 更新状态（使用群体统计边界）
                T_new, S_new, D_new, W_new = update_state_numba(
                    row['T'], row['S'], row['D'], row['W'], a,
                    T_max_population, W_min_population,
                    self.gamma_T, self.gamma_W, self.gamma_S, self.gamma_D
                )
                
                T_new_list.append(T_new)
                S_new_list.append(S_new)
                D_new_list.append(D_new)
                W_new_list.append(W_new)
            
            # 构建批量预测输入
            X = pd.DataFrame({
                'const': np.ones(N),
                'T': T_new_list,
                'S': S_new_list,
                'D': D_new_list,
                'W': W_new_list,
                'sigma': sigma,  # 所有努力水平共享同一个sigma
                'theta': np.full(N, theta)
            })
            
            # 批量预测
            lambda_probs[:, j] = self.match_model.predict(X)
        
        return lambda_probs
    
    def solve(
        self,
        individuals: pd.DataFrame,
        theta: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        求解贝尔曼方程（主接口）
        
        参数:
            individuals: 个体DataFrame，必须包含：
                - employment_status: 就业状态（'unemployed'/'employed'）
                - current_wage: 就业者当前工资收入（失业者为NaN或0）
                - T, S, D, W: 状态变量
                - age, education, children: 固定特征
            theta: 市场紧张度
            
        返回:
            (V_U, V_E, a_optimal): 均为pd.Series，索引与individuals一致
        """
        # 分离失业和就业个体
        mask_U = individuals['employment_status'] == 'unemployed'
        mask_E = individuals['employment_status'] == 'employed'
        
        individuals_U = individuals[mask_U].copy()
        individuals_E = individuals[mask_E].copy()
        
        N_U = len(individuals_U)
        N_E = len(individuals_E)
        
        print(f"失业者: {N_U}, 就业者: {N_E}")
        
        # 批量计算失业者的匹配概率
        print("批量计算匹配概率...")
        if N_U > 0:
            lambda_probs_U = self.compute_match_probabilities_batch(
                individuals_U, self.a_grid, theta
            )
        else:
            lambda_probs_U = np.zeros((0, len(self.a_grid)))
        
        # 准备NumPy数组
        if N_U > 0:
            T_U = individuals_U['T'].values
            S_U = individuals_U['S'].values
            D_U = individuals_U['D'].values
            W_U = individuals_U['W'].values
            age_U = individuals_U['age'].values
            edu_U = individuals_U['education'].values
            children_U = individuals_U['children'].values
        else:
            T_U = S_U = D_U = W_U = age_U = edu_U = children_U = np.array([])
        
        if N_E > 0:
            T_E = individuals_E['T'].values
            S_E = individuals_E['S'].values
            D_E = individuals_E['D'].values
            W_E = individuals_E['W'].values
            # 就业者当前工资收入
            current_wage_E = individuals_E['current_wage'].values
            age_E = individuals_E['age'].values
            edu_E = individuals_E['education'].values
            children_E = individuals_E['children'].values
        else:
            T_E = S_E = D_E = W_E = np.array([])
            current_wage_E = np.array([])
            age_E = edu_E = children_E = np.array([])
        
        # 索引映射
        unemployed_indices = individuals_U.index.values
        employed_indices = individuals_E.index.values
        
        # 调用Numba加速的值迭代
        print("开始值迭代（Numba加速）...")
        V_U_array, V_E_array, a_optimal_array, iterations, max_diff = (
            value_iteration_numba(
                T_U, S_U, D_U, W_U, age_U, edu_U, children_U,
                lambda_probs_U,
                T_E, S_E, D_E, W_E, current_wage_E,
                age_E, edu_E, children_E,
                unemployed_indices, employed_indices, self.a_grid,
                self.rho, self.kappa, self.b0,
                self.gamma_T, self.gamma_W, self.gamma_S, self.gamma_D,
                self.eta0, self.eta_T, self.eta_S, self.eta_D, self.eta_W,
                self.eta_age, self.eta_edu, self.eta_children,
                self.max_iter, self.tol
            )
        )
        
        print(f"值迭代完成：迭代{iterations}轮，最大差异={max_diff:.6f}")
        
        # 【修正】整理结果为pd.Series
        # V_U_array和V_E_array现在长度都是N（所有个体）
        N = len(individuals)
        
        # 【调试】检查返回的数组
        print(f"  V_U_array长度: {len(V_U_array)}, 前5个值: {V_U_array[:5]}")
        print(f"  V_E_array长度: {len(V_E_array)}, 前5个值: {V_E_array[:5]}")
        print(f"  失业者索引: {unemployed_indices[:5] if len(unemployed_indices)>0 else '无'}")
        
        V_U = pd.Series(V_U_array, index=individuals.index)
        V_E = pd.Series(V_E_array, index=individuals.index)
        
        # a_optimal只对失业者有意义，就业者设为0
        a_optimal_all = np.zeros(N)
        if N_U > 0:
            a_optimal_all[unemployed_indices] = a_optimal_array
        a_optimal = pd.Series(a_optimal_all, index=individuals.index)
        
        return V_U, V_E, a_optimal


def load_match_function_model(model_path: str):
    """
    加载已训练的匹配函数模型
    
    参数:
        model_path: 模型文件路径
        
    返回:
        statsmodels Logit模型对象
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
