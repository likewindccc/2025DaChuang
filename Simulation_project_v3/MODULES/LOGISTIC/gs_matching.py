#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gale-Shapley稳定匹配算法模块（Numba加速版本）

实现有限轮次的双边稳定匹配算法，模拟真实就业市场的摩擦。

核心功能:
    - 计算双边偏好得分（劳动力→企业、企业→劳动力）
    - 执行有限轮次的GS匹配（模拟市场摩擦）
    - 返回匹配结果

技术特点:
    - 使用numba JIT编译加速核心计算循环
    - MinMax标准化保证各项贡献量级一致
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Any


@njit
def compute_laborer_preferences_core(
    S_i_raw: np.ndarray,
    D_i_raw: np.ndarray,
    T_req_raw: np.ndarray,
    S_req_raw: np.ndarray,
    D_req_raw: np.ndarray,
    W_offer_raw: np.ndarray,
    gamma_0: float,
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
    gamma_4: float
) -> np.ndarray:
    """
    计算劳动力偏好得分核心循环（numba加速）
    
    改进版：先计算原始偏好项，再对每一项单独标准化
    
    Args:
        S_i_raw到W_offer_raw: 劳动力和企业特征（原始值）
        gamma_0到gamma_4: 偏好函数参数
    
    Returns:
        偏好矩阵，shape (n_laborers, n_enterprises)
    """
    n_laborers = len(S_i_raw)
    n_enterprises = len(T_req_raw)
    
    # 初始化各项原始矩阵
    term_T = np.zeros((n_laborers, n_enterprises))
    term_S = np.zeros((n_laborers, n_enterprises))
    term_D = np.zeros((n_laborers, n_enterprises))
    term_W = np.zeros((n_laborers, n_enterprises))
    
    # 计算各项原始值
    for i in range(n_laborers):
        for j in range(n_enterprises):
            term_T[i, j] = T_req_raw[j]
            term_S[i, j] = max(0.0, S_req_raw[j] - S_i_raw[i])
            term_D[i, j] = max(0.0, D_req_raw[j] - D_i_raw[i])
            term_W[i, j] = W_offer_raw[j]
    
    # 对每一项分别MinMax标准化
    T_min, T_max = term_T.min(), term_T.max()
    S_min, S_max = term_S.min(), term_S.max()
    D_min, D_max = term_D.min(), term_D.max()
    W_min, W_max = term_W.min(), term_W.max()
    
    term_T_norm = (term_T - T_min) / (T_max - T_min + 1e-10)
    term_S_norm = (term_S - S_min) / (S_max - S_min + 1e-10)
    term_D_norm = (term_D - D_min) / (D_max - D_min + 1e-10)
    term_W_norm = (term_W - W_min) / (W_max - W_min + 1e-10)
    
    # 加权求和
    preferences = np.zeros((n_laborers, n_enterprises))
    for i in range(n_laborers):
        for j in range(n_enterprises):
            preferences[i, j] = (
                gamma_0
                - gamma_1 * term_T_norm[i, j]
                - gamma_2 * term_S_norm[i, j]
                - gamma_3 * term_D_norm[i, j]
                + gamma_4 * term_W_norm[i, j]
            )
    
    return preferences


@njit
def compute_enterprise_preferences_core(
    T_i_raw: np.ndarray,
    S_i_raw: np.ndarray,
    D_i_raw: np.ndarray,
    W_i_raw: np.ndarray,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    beta_4: float
) -> np.ndarray:
    """
    计算企业偏好得分核心循环（numba加速）
    
    改进版：先计算原始偏好项，再对每一项单独标准化
    
    Args:
        T_i_raw到W_i_raw: 劳动力特征（原始值），shape (n_laborers,)
        beta_0到beta_4: 偏好函数参数
    
    Returns:
        偏好得分向量，shape (n_laborers,)
    """
    n_laborers = len(T_i_raw)
    
    # 各项就是原始值（企业偏好函数更简单）
    term_T = T_i_raw.copy()
    term_S = S_i_raw.copy()
    term_D = D_i_raw.copy()
    term_W = W_i_raw.copy()
    
    # 对每一项分别MinMax标准化
    T_min, T_max = term_T.min(), term_T.max()
    S_min, S_max = term_S.min(), term_S.max()
    D_min, D_max = term_D.min(), term_D.max()
    W_min, W_max = term_W.min(), term_W.max()
    
    term_T_norm = (term_T - T_min) / (T_max - T_min + 1e-10)
    term_S_norm = (term_S - S_min) / (S_max - S_min + 1e-10)
    term_D_norm = (term_D - D_min) / (D_max - D_min + 1e-10)
    term_W_norm = (term_W - W_min) / (W_max - W_min + 1e-10)
    
    # 加权求和
    preferences_base = np.zeros(n_laborers)
    for i in range(n_laborers):
        preferences_base[i] = (
            beta_0
            + beta_1 * term_T_norm[i]
            + beta_2 * term_S_norm[i]
            + beta_3 * term_D_norm[i]
            + beta_4 * term_W_norm[i]
        )
    
    return preferences_base


@njit
def gale_shapley_matching_core(
    laborer_pref_ranks: np.ndarray,
    enterprise_pref_ranks: np.ndarray,
    max_rounds: int
) -> np.ndarray:
    """
    GS匹配算法核心循环（numba加速）
    
    Args:
        laborer_pref_ranks: 劳动力偏好排序，shape (n_laborers, n_enterprises)
        enterprise_pref_ranks: 企业偏好排序，shape (n_laborers,)
        max_rounds: 最大匹配轮数
    
    Returns:
        匹配结果，shape (n_laborers,)，值为企业ID（-1表示未匹配）
        
    算法流程:
        1. 劳动力按偏好排序向企业申请
        2. 企业比较新旧申请者，保留偏好更高者
        3. 重复max_rounds轮或直到无新匹配
    """
    n_laborers = laborer_pref_ranks.shape[0]
    n_enterprises = laborer_pref_ranks.shape[1]
    
    # 初始化匹配状态
    laborer_match = np.full(n_laborers, -1, dtype=np.int32)
    enterprise_match = np.full(n_enterprises, -1, dtype=np.int32)
    laborer_next_proposal = np.zeros(n_laborers, dtype=np.int32)
    
    # 执行有限轮次匹配
    for round_idx in range(max_rounds):
        # 检查是否所有劳动力都已匹配
        all_matched = True
        for i in range(n_laborers):
            if laborer_match[i] == -1:
                all_matched = False
                break
        
        if all_matched:
            break
        
        # 未匹配劳动力向下一个偏好企业申请
        for laborer_id in range(n_laborers):
            # 跳过已匹配的劳动力
            if laborer_match[laborer_id] != -1:
                continue
            
            # 检查是否还有可申请的企业
            if laborer_next_proposal[laborer_id] >= n_enterprises:
                continue
            
            # 获取下一个申请的企业
            enterprise_id = laborer_pref_ranks[laborer_id, laborer_next_proposal[laborer_id]]
            laborer_next_proposal[laborer_id] += 1
            
            # 企业决策：接受或拒绝
            current_match = enterprise_match[enterprise_id]
            
            if current_match == -1:
                # 企业未匹配，直接接受
                enterprise_match[enterprise_id] = laborer_id
                laborer_match[laborer_id] = enterprise_id
            else:
                # 企业已匹配，比较新旧申请者
                # 找到current_match和laborer_id在企业偏好中的位置
                current_rank = -1
                new_rank = -1
                for rank in range(n_laborers):
                    if enterprise_pref_ranks[rank] == current_match:
                        current_rank = rank
                    if enterprise_pref_ranks[rank] == laborer_id:
                        new_rank = rank
                    if current_rank != -1 and new_rank != -1:
                        break
                
                if new_rank < current_rank:
                    # 新申请者更优，替换
                    laborer_match[current_match] = -1
                    enterprise_match[enterprise_id] = laborer_id
                    laborer_match[laborer_id] = enterprise_id
    
    return laborer_match


def perform_matching(
    laborers: pd.DataFrame,
    enterprises: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    执行完整的GS匹配流程
    
    Args:
        laborers: 劳动力DataFrame
        enterprises: 企业DataFrame
        config: 配置字典（包含偏好参数和max_rounds）
    
    Returns:
        匹配结果DataFrame
        
    流程:
        1. 提取并标准化特征（MinMax标准化）
        2. 计算双边偏好矩阵（使用numba加速）
        3. 执行GS匹配算法（使用numba加速）
        4. 返回匹配结果
    """
    # 读取配置参数
    gs_config = config['gs_matching']
    laborer_params = gs_config['laborer_preference']
    enterprise_params = gs_config['enterprise_preference']
    max_rounds = gs_config['max_rounds']
    
    # 提取特征（原始值）
    # 劳动力特征
    S_i_raw = laborers['S'].values
    D_i_raw = laborers['D'].values
    T_i_raw = laborers['T'].values
    W_i_raw = laborers['W'].values
    
    # 企业特征
    T_req_raw = enterprises['T_req'].values
    S_req_raw = enterprises['S_req'].values
    D_req_raw = enterprises['D_req'].values
    W_offer_raw = enterprises['W_offer'].values
    
    # 计算偏好矩阵（使用numba加速，标准化在内部完成）
    laborer_prefs = compute_laborer_preferences_core(
        S_i_raw, D_i_raw,
        T_req_raw, S_req_raw, D_req_raw, W_offer_raw,
        laborer_params['gamma_0'], laborer_params['gamma_1'],
        laborer_params['gamma_2'], laborer_params['gamma_3'],
        laborer_params['gamma_4']
    )
    
    enterprise_prefs_base = compute_enterprise_preferences_core(
        T_i_raw, S_i_raw, D_i_raw, W_i_raw,
        enterprise_params['beta_0'], enterprise_params['beta_1'],
        enterprise_params['beta_2'], enterprise_params['beta_3'],
        enterprise_params['beta_4']
    )
    
    # 生成偏好排序
    laborer_pref_ranks = np.argsort(-laborer_prefs, axis=1).astype(np.int32)
    enterprise_pref_ranks = np.argsort(-enterprise_prefs_base).astype(np.int32)
    
    # 执行GS匹配（使用numba加速）
    laborer_match = gale_shapley_matching_core(
        laborer_pref_ranks,
        enterprise_pref_ranks,
        max_rounds
    )
    
    # 构建结果DataFrame
    result = laborers.copy()
    result['enterprise_id'] = laborer_match
    result['matched'] = (laborer_match != -1).astype(int)
    
    return result
