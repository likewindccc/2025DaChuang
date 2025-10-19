#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝尔曼方程求解器V2（统一处理版本，Numba加速）

【重构】核心改进：
- 所有个体同时计算V_U和V_E（无论当前是失业还是就业）
- V_U[i]: 个体i处于失业状态的价值
- V_E[i]: 个体i处于就业状态的价值
- 解决了原版本中状态转换时价值函数不连续的问题

设计架构：
1. Numba加速的核心计算函数（纯NumPy数组，@njit装饰）
2. Python包装类（数据准备、模型调用、结果整理）
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple
from numba import njit, prange
import yaml


# =============================================================================
# Numba加速的核心计算函数
# =============================================================================

@njit
def update_state_numba(
    T: float, S: float, D: float, W: float,
    a: float,
    T_max_population: float,
    W_min_population: float,
    S_min_population: float, S_max_population: float,
    D_min_population: float, D_max_population: float,
    gamma_T: float, gamma_W: float, gamma_S: float, gamma_D: float
) -> Tuple[float, float, float, float]:
    """
    状态更新函数（Numba加速）
    
    对S和D进行MinMax标准化到[0,1]，应用更新公式后再反标准化
    """
    # T和W直接更新
    T_new = T + gamma_T * a * (T_max_population - T)
    W_new = max(W_min_population, W - gamma_W * a)
    
    # S: MinMax标准化 -> 更新 -> 反标准化
    S_range = S_max_population - S_min_population
    if S_range > 1e-10:
        S_norm = (S - S_min_population) / S_range  # 标准化到[0,1]
        S_norm_new = S_norm + gamma_S * a * (1.0 - S_norm)  # 更新
        S_new = S_norm_new * S_range + S_min_population  # 反标准化
    else:
        S_new = S  # 如果所有人S相同，保持不变
    
    # D: MinMax标准化 -> 更新 -> 反标准化
    D_range = D_max_population - D_min_population
    if D_range > 1e-10:
        D_norm = (D - D_min_population) / D_range  # 标准化到[0,1]
        D_norm_new = D_norm + gamma_D * a * (1.0 - D_norm)  # 更新
        D_new = D_norm_new * D_range + D_min_population  # 反标准化
    else:
        D_new = D  # 如果所有人D相同，保持不变
    
    return T_new, S_new, D_new, W_new


@njit
def compute_separation_rate_numba(
    T: float, S: float, D: float, W: float,
    age: float, education: float, children: float,
    T_mean: float, T_std: float,
    S_mean: float, S_std: float,
    D_mean: float, D_std: float,
    W_mean: float, W_std: float,
    age_mean: float, age_std: float,
    edu_mean: float, edu_std: float,
    children_mean: float, children_std: float,
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float
) -> float:
    """计算外生离职率 μ(x, σ_i)"""
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
    
    # 限制在合理范围
    mu = max(0.01, min(0.99, mu))
    
    return mu


@njit(parallel=True)
def solve_bellman_unified_numba(
    # 状态变量（N个个体）
    T: np.ndarray, S: np.ndarray, D: np.ndarray, W: np.ndarray,
    # 固定特征
    age: np.ndarray, edu: np.ndarray, children: np.ndarray,
    # 当前状态
    is_unemployed: np.ndarray,  # bool数组
    current_wage: np.ndarray,
    # 匹配概率
    lambda_probs: np.ndarray,  # shape: (N, n_effort)
    # 下期值函数
    V_U_next: np.ndarray,
    V_E_next: np.ndarray,
    # 群体统计量
    T_mean: float, T_std: float,
    S_mean: float, S_std: float,
    D_mean: float, D_std: float,
    W_mean: float, W_std: float,
    age_mean: float, age_std: float,
    edu_mean: float, edu_std: float,
    children_mean: float, children_std: float,
    # 努力网格
    a_grid: np.ndarray,
    # 参数
    rho: float, kappa: float, b0: float,
    # T的负效用参数
    initial_T: np.ndarray, disutility_T_enabled: bool, alpha_T: float,
    # 离职率参数
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一求解所有个体的贝尔曼方程
    
    返回: (V_U, V_E, a_optimal)
    """
    N = len(T)
    n_effort = len(a_grid)
    
    V_U = np.zeros(N)
    V_E = np.zeros(N)
    a_optimal = np.zeros(N)
    
    # 失业收益
    b = b0
    
    # 并行循环
    for i in prange(N):
        # =====================================
        # 1. 计算V_U[i]（失业状态价值）
        # =====================================
        max_value_U = -np.inf
        best_a = 0.0
        
        for j in range(n_effort):
            a = a_grid[j]
            
            # 努力成本（二次函数）
            effort_cost = 0.5 * kappa * a * a
            
            # T的负效用（work-leisure tradeoff）
            # 当T偏离个体的初始T值（理想工作时间）时产生负效用
            if disutility_T_enabled:
                disutility_T = alpha_T * (T[i] - initial_T[i]) ** 2
            else:
                disutility_T = 0.0
            
            # 即时效用 = 失业救济金 - 努力成本 - T的负效用
            instant_utility = b - effort_cost - disutility_T
            
            # 匹配概率
            lambda_prob = lambda_probs[i, j]
            
            # 下期期望价值
            V_next_expected = (
                lambda_prob * V_E_next[i] +
                (1.0 - lambda_prob) * V_U_next[i]
            )
            
            # 总价值
            total_value = instant_utility + rho * V_next_expected
            
            if total_value > max_value_U:
                max_value_U = total_value
                best_a = a
        
        V_U[i] = max_value_U
        a_optimal[i] = best_a
    
        # =====================================
        # 2. 计算V_E[i]（就业状态价值）
        # =====================================
        # 就业效用
        if is_unemployed[i]:
            # 失业者：用期望工资W[i]
            omega = W[i]
        else:
            # 就业者：用实际工资
            omega = current_wage[i]
        
        # 计算离职率
        mu = compute_separation_rate_numba(
            T[i], S[i], D[i], W[i],
            age[i], edu[i], children[i],
            T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
            age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
            eta0, eta_T, eta_S, eta_D, eta_W,
            eta_age, eta_edu, eta_children
        )
        
        # 下期期望价值
        V_next_expected_E = mu * V_U_next[i] + (1.0 - mu) * V_E_next[i]
        
        # 当期价值
        V_E[i] = omega + rho * V_next_expected_E
    
    return V_U, V_E, a_optimal


@njit
def value_iteration_unified_numba(
    # 所有个体数据
    T: np.ndarray, S: np.ndarray, D: np.ndarray, W: np.ndarray,
    age: np.ndarray, edu: np.ndarray, children: np.ndarray,
    is_unemployed: np.ndarray,
    current_wage: np.ndarray,
    lambda_probs: np.ndarray,
    # 努力网格
    a_grid: np.ndarray,
    # 参数
    rho: float, kappa: float, b0: float,
    # T的负效用参数
    initial_T: np.ndarray, disutility_T_enabled: bool, alpha_T: float,
    # 离职率参数
    eta0: float, eta_T: float, eta_S: float, eta_D: float, eta_W: float,
    eta_age: float, eta_edu: float, eta_children: float,
    # 迭代参数
    max_iter: int, tol: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    值迭代主循环（Numba JIT加速）
    
    使用不动点迭代法求解贝尔曼方程：
    1. 初始化 V_U^0 = 0, V_E^0 = 0
    2. 迭代更新：V^{k+1} = T(V^k)，其中T是贝尔曼算子
    3. 直到收敛：‖V^{k+1} - V^k‖_∞ < tol
    
    算法特点：
    - 所有个体同时计算V_U和V_E（统一处理，避免索引错误）
    - 使用Numba JIT编译，提速10-100倍
    - 自动检测收敛，避免无限循环
    
    返回: (V_U, V_E, a_optimal, iterations, max_diff)
        - V_U: 失业状态价值函数（N维数组）
        - V_E: 就业状态价值函数（N维数组）
        - a_optimal: 最优努力水平（N维数组，仅对失业者有意义）
        - iterations: 实际迭代次数
        - max_diff: 最终收敛误差
    """
    N = len(T)
    
    # 初始化价值函数和策略（从0开始）
    V_U_all = np.zeros(N)
    V_E_all = np.zeros(N)
    a_optimal = np.zeros(N)
    
    # 计算群体统计量（用于离职率标准化）
    # μ(x, σ) = 1/(1+exp(-η'Z))，Z为标准化后的特征
    T_mean, T_std = T.mean(), T.std()
    S_mean, S_std = S.mean(), S.std()
    D_mean, D_std = D.mean(), D.std()
    W_mean, W_std = W.mean(), W.std()
    age_mean, age_std = age.mean(), age.std()
    edu_mean, edu_std = edu.mean(), edu.std()
    children_mean, children_std = children.mean(), children.std()
    
    # 值迭代主循环
    for iteration in range(max_iter):
        # 保存上一轮价值函数（用于收敛判断）
        V_U_old = V_U_all.copy()
        V_E_old = V_E_all.copy()
        
        # 求解贝尔曼方程：V^{k+1} = T(V^k)
        # 对所有N个个体同时计算新的V_U, V_E, a_optimal
        V_U_all, V_E_all, a_optimal = solve_bellman_unified_numba(
            T, S, D, W, age, edu, children,
            is_unemployed, current_wage,
            lambda_probs, V_U_old, V_E_old,
                T_mean, T_std, S_mean, S_std, D_mean, D_std, W_mean, W_std,
                age_mean, age_std, edu_mean, edu_std, children_mean, children_std,
            a_grid, rho, kappa, b0,
            initial_T, disutility_T_enabled, alpha_T,
            eta0, eta_T, eta_S, eta_D, eta_W, eta_age, eta_edu, eta_children
        )
        
        # 限制范围（防止数值溢出）
        V_U_all = np.clip(V_U_all, -1e6, 1e6)
        V_E_all = np.clip(V_E_all, -1e6, 1e6)
        
        # 检查收敛（使用无穷范数）
        # ‖V^{k+1} - V^k‖_∞ = max_i |V_i^{k+1} - V_i^k|
        diff_U = np.max(np.abs(V_U_all - V_U_old))
        diff_E = np.max(np.abs(V_E_all - V_E_old))
        max_diff = max(diff_U, diff_E)
        
        # 如果收敛，提前退出
        if max_diff < tol:
            return V_U_all, V_E_all, a_optimal, iteration + 1, max_diff
    
    # 达到最大迭代次数（未收敛）
    return V_U_all, V_E_all, a_optimal, max_iter, max_diff


# =============================================================================
# Python包装类
# =============================================================================

class BellmanSolver:
    """贝尔曼方程求解器"""
    
    def __init__(self, config, match_function_model):
        """
        初始化贝尔曼求解器
        
        参数:
            config: 配置字典（不是路径，而是已加载的配置）
            match_function_model: 已训练的匹配函数模型
        """
        
        # ==================================================
        # 1. 努力网格设置
        # ==================================================
        # 将连续的努力空间[a_min, a_max]离散化为有限个点
        # 用于在贝尔曼方程中枚举搜索最优努力水平
        effort_cfg = config['effort']
        self.a_grid = np.linspace(
            effort_cfg['a_min'],      # 最小努力水平（通常为0）
            effort_cfg['a_max'],      # 最大努力水平（通常为1）
            effort_cfg['a_points']    # 离散化点数（通常为11个）
        )
        
        # ==================================================
        # 2. 核心经济参数
        # ==================================================
        econ = config['economics']
        
        # 贴现因子（未来收益的偏好程度）
        self.rho = econ['rho']  # ρ ∈ (0,1)，越大越重视未来
        
        # 努力成本系数（effort_cost = 0.5 * kappa * a^2）
        self.kappa = econ['kappa']  # κ > 0，越大努力成本越高
        
        # 失业救济金（固定收益）
        self.b0 = econ['unemployment_benefit']['b0']  # b0 >= 0
        
        # T的负效用函数参数（work-leisure tradeoff）
        disutility_T_cfg = econ.get('disutility_T', {'enabled': False, 'alpha': 0.0})
        self.disutility_T_enabled = disutility_T_cfg.get('enabled', False)
        self.alpha_T = disutility_T_cfg.get('alpha', 0.0)  # α > 0，负效用系数
        
        # ==================================================
        # 3. 状态更新系数（研究计划4.3节）
        # ==================================================
        # 描述个体通过努力a改变状态的速度
        # x_{t+1} = x_t + γ*a_t*(x_bound - x_t)
        state_update = econ['state_update']
        self.gamma_T = state_update['gamma_T']  # 工作时长更新速度
        self.gamma_W = state_update['gamma_W']  # 期望工资更新速度
        self.gamma_S = state_update['gamma_S']  # 工作能力更新速度
        self.gamma_D = state_update['gamma_D']  # 数字素养更新速度
        
        # ==================================================
        # 4. 离职率函数系数（研究计划中的μ函数）
        # ==================================================
        # μ(x, σ) = 1/(1+exp(-η'Z))，Z为标准化后的特征向量
        sep_rate = econ['separation_rate']
        self.eta0 = sep_rate['eta0']              # 截距项
        self.eta_T = sep_rate['eta_T']            # T的系数（工作时长越长越稳定）
        self.eta_S = sep_rate['eta_S']            # S的系数（技能越高越稳定）
        self.eta_D = sep_rate['eta_D']            # D的系数（数字素养越高越稳定）
        self.eta_W = sep_rate['eta_W']            # W的系数（期望工资越高越不稳定）
        self.eta_age = sep_rate['eta_age']        # age的系数（年龄越大越稳定）
        self.eta_edu = sep_rate['eta_edu']        # edu的系数（教育越高越稳定）
        self.eta_children = sep_rate['eta_children']  # children的系数（孩子越多越不稳定）
        
        # ==================================================
        # 5. 值迭代算法参数
        # ==================================================
        val_iter = config['value_iteration']
        self.max_iter = val_iter['max_iter']  # 最大迭代次数（防止不收敛）
        self.tol = val_iter['tol']            # 收敛阈值（‖V_{k+1} - V_k‖ < tol）
        
        # ==================================================
        # 6. 匹配函数模型
        # ==================================================
        # 已训练的Logit模型：λ(x', σ, θ) = P(matched=1|x', σ, θ)
        self.match_function_model = match_function_model
    
    def compute_match_probabilities_batch(
        self,
        individuals: pd.DataFrame,
        a_grid: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """
        批量计算匹配概率λ(x', σ, θ)
        
        对于每个个体i和每个努力水平a_j：
        1. 根据努力a_j更新状态：x' = update(x, a_j)
        2. 计算控制变量σ（age, edu, children的标准化综合指标）
        3. 调用匹配函数模型：λ = MatchFunction(x', σ, θ)
        
        参数:
            individuals: 当前个体状态DataFrame
            a_grid: 努力水平网格
            theta: 市场紧张度
            
        返回:
            lambda_probs: shape (N, n_effort)，每个个体在每个努力水平下的匹配概率
        """
        N = len(individuals)
        n_effort = len(a_grid)
        lambda_probs = np.zeros((N, n_effort))
        
        # ==================================================
        # 步骤1: 计算控制变量σ（研究计划中的二次MinMax标准化）
        # ==================================================
        # σ = MinMax(MinMax(age) + MinMax(edu) + MinMax(children))
        
        age_raw = individuals['age'].values
        edu_raw = individuals['education'].values
        children_raw = individuals['children'].values
        
        # 第一次MinMax标准化：分别对age, edu, children标准化到[0,1]
        age_min, age_max = age_raw.min(), age_raw.max()
        age_norm = (age_raw - age_min) / (age_max - age_min + 1e-10)
        
        edu_min, edu_max = edu_raw.min(), edu_raw.max()
        edu_norm = (edu_raw - edu_min) / (edu_max - edu_min + 1e-10)
        
        children_min, children_max = children_raw.min(), children_raw.max()
        children_norm = (children_raw - children_min) / (
            children_max - children_min + 1e-10
        )
        
        # 求和后再次MinMax标准化（第二次标准化）
        sigma_sum = age_norm + edu_norm + children_norm
        sigma_min, sigma_max = sigma_sum.min(), sigma_sum.max()
        sigma = (sigma_sum - sigma_min) / (sigma_max - sigma_min + 1e-10)
        
        # ==================================================
        # 步骤2: 计算群体统计边界（用于状态更新）
        # ==================================================
        # 这些边界用于状态更新函数中的MinMax标准化
        T_max_population = individuals['T'].max()  # 群体最大工作时长
        W_min_population = individuals['W'].min()  # 群体最低期望工资
        S_min = individuals['S'].min()  # 群体最低工作能力
        S_max = individuals['S'].max()  # 群体最高工作能力
        D_min = individuals['D'].min()  # 群体最低数字素养
        D_max = individuals['D'].max()  # 群体最高数字素养
        
        # ==================================================
        # 步骤3: 对每个努力水平a_j，计算所有个体的匹配概率
        # ==================================================
        for j, a in enumerate(a_grid):
            # 存储所有个体在努力a下的新状态
            T_new_list = []
            S_new_list = []
            D_new_list = []
            W_new_list = []
            
            # 对每个个体，根据努力a更新状态
            for idx, row in individuals.iterrows():
                # 调用Numba加速的状态更新函数
                # x' = x + γ*a*(x_bound - x)
                T_new, S_new, D_new, W_new = update_state_numba(
                    row['T'], row['S'], row['D'], row['W'], a,
                    T_max_population, W_min_population,
                    S_min, S_max, D_min, D_max,
                    self.gamma_T, self.gamma_W, self.gamma_S, self.gamma_D
                )
                
                T_new_list.append(T_new)
                S_new_list.append(S_new)
                D_new_list.append(D_new)
                W_new_list.append(W_new)
            
            # 构建匹配函数输入（更新后的状态x' + 控制变量σ + 市场紧张度θ）
            X = pd.DataFrame({
                'const': np.ones(N),  # 常数项（匹配函数需要）
                'T': T_new_list,      # 更新后的工作时长
                'S': S_new_list,      # 更新后的工作能力
                'D': D_new_list,      # 更新后的数字素养
                'W': W_new_list,      # 更新后的期望工资
                'sigma': sigma,       # 控制变量综合指标
                'theta': np.full(N, theta)  # 市场紧张度（所有人相同）
            })
            
            # 调用训练好的匹配函数模型预测匹配概率
            # λ(x', σ, θ) = 1 / (1 + exp(-β'X))  (Logistic回归)
            logit_values = self.match_function_model.predict(X)
            probs = 1 / (1 + np.exp(-logit_values))
            lambda_probs[:, j] = probs
        
        return lambda_probs
    
    def solve(
        self,
        individuals: pd.DataFrame,
        theta: float,
        initial_T: np.ndarray
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        求解贝尔曼方程，得到价值函数和最优策略
        
        算法流程（研究计划4.4节）：
        1. 批量计算所有个体在所有努力水平下的匹配概率 λ(x', σ, θ)
        2. 使用值迭代算法求解贝尔曼方程：
           - 失业者：V_U(x) = max_a { b - c(a) - disutility_T(T, T_ideal) + ρ*[λ*V_E(x') + (1-λ)*V_U(x')] }
           - 就业者：V_E(x) = ω + ρ*[μ*V_U(x') + (1-μ)*V_E(x')]
        3. 返回收敛的价值函数和最优策略
        
        参数:
            individuals: 当前个体状态DataFrame（包含T,S,D,W等）
            theta: 市场紧张度（V/U）
            initial_T: 每个个体的初始T值（作为理想工作时间）
            
        返回:
            V_U: 失业状态价值函数（每个个体的失业价值）
            V_E: 就业状态价值函数（每个个体的就业价值）
            a_optimal: 最优努力水平（失业者的最优选择）
        """
        N = len(individuals)
        
        print(f"求解贝尔曼方程（N={N}）...")
        
        # ==================================================
        # 步骤1: 批量计算匹配概率矩阵
        # ==================================================
        # 对于每个个体i和每个努力水平a_j，计算λ_ij = λ(x_i', σ_i, θ)
        # 输出shape: (N, n_effort)
        print("批量计算匹配概率...")
        lambda_probs = self.compute_match_probabilities_batch(
            individuals, self.a_grid, theta
        )
        
        # ==================================================
        # 步骤2: 准备数据（转为NumPy数组以便Numba加速）
        # ==================================================
        # 状态变量
        T = individuals['T'].values
        S = individuals['S'].values
        D = individuals['D'].values
        W = individuals['W'].values
        
        # 固定特征（用于离职率计算）
        age = individuals['age'].values
        edu = individuals['education'].values
        children = individuals['children'].values
        
        # 当前就业状态
        is_unemployed = (
            individuals['employment_status'] == 'unemployed'
        ).values
        
        # 当前工资（就业者有实际工资，失业者为0）
        current_wage = individuals['current_wage'].fillna(0).values
        
        # ==================================================
        # 步骤3: 调用Numba加速的值迭代算法
        # ==================================================
        # 使用Bellman不动点迭代求解：
        # 重复直到收敛 {
        #   V_U_new, V_E_new, a_new = solve_bellman(V_U_old, V_E_old)
        #   if ‖V_new - V_old‖ < tol: break
        # }
        print("开始值迭代（Numba加速）...")
        V_U_array, V_E_array, a_optimal_array, iterations, max_diff = (
            value_iteration_unified_numba(
                # 状态变量
                T, S, D, W, age, edu, children,
                is_unemployed, current_wage,
                # 匹配概率矩阵
                lambda_probs, self.a_grid,
                # 经济参数
                self.rho, self.kappa, self.b0,
                # T的负效用参数
                initial_T, self.disutility_T_enabled, self.alpha_T,
                # 离职率参数
                self.eta0, self.eta_T, self.eta_S, self.eta_D, self.eta_W,
                self.eta_age, self.eta_edu, self.eta_children,
                # 算法参数
                self.max_iter, self.tol
            )
        )
        
        print(f"值迭代完成：迭代{iterations}轮，最大差异={max_diff:.6f}")
        
        # ==================================================
        # 步骤4: 输出统计信息（用于调试和监控）
        # ==================================================
        print(f"  【V2版本】V_U统计: min={V_U_array.min():.2f}, max={V_U_array.max():.2f}, mean={V_U_array.mean():.2f}")
        print(f"  【V2版本】V_E统计: min={V_E_array.min():.2f}, max={V_E_array.max():.2f}, mean={V_E_array.mean():.2f}")
        print(f"  【V2版本】a统计: min={a_optimal_array.min():.4f}, max={a_optimal_array.max():.4f}, mean={a_optimal_array.mean():.4f}")
        
        # ==================================================
        # 步骤5: 转换为pandas Series（保持与individuals的索引对应）
        # ==================================================
        V_U = pd.Series(V_U_array, index=individuals.index)
        V_E = pd.Series(V_E_array, index=individuals.index)
        a_optimal = pd.Series(a_optimal_array, index=individuals.index)
        
        return V_U, V_E, a_optimal


def load_match_function_model(model_path: str):
    """
    加载已训练的匹配函数模型
    
    匹配函数是在LOGISTIC模块中训练的Logit回归模型：
    λ(x, σ, θ) = P(matched=1 | T, S, D, W, sigma, theta)
    
    参数:
        model_path: 模型文件路径（.pkl格式）
        
    返回:
        已训练的statsmodels Logit模型对象
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

