#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
偏好函数各项贡献度分析
分析劳动力和企业偏好函数中各项的实际影响程度
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import VirtualMarket, load_config


def analyze_preference_contributions():
    """分析偏好函数各项的贡献度"""
    print("=" * 80)
    print("偏好函数各项贡献度分析")
    print("=" * 80)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 生成一个虚拟市场
    market_gen = VirtualMarket(config)
    laborers, enterprises = market_gen.generate_market(n_laborers=2000, theta=1.0)
    
    print(f"\n虚拟市场规模:")
    print(f"  劳动力: {len(laborers)}")
    print(f"  企业: {len(enterprises)}")
    
    # 提取原始数据
    T_i = laborers['T'].values
    S_i = laborers['S'].values
    D_i = laborers['D'].values
    W_i = laborers['W'].values
    
    T_req = enterprises['T_req'].values
    S_req = enterprises['S_req'].values
    D_req = enterprises['D_req'].values
    W_offer = enterprises['W_offer'].values
    
    print("\n" + "=" * 80)
    print("劳动力偏好函数分析（新版本：先计算原始偏好项，再对每项标准化）")
    print("=" * 80)
    
    # 劳动力偏好函数参数
    gamma_0 = config['gs_matching']['laborer_preference']['gamma_0']
    gamma_1 = config['gs_matching']['laborer_preference']['gamma_1']
    gamma_2 = config['gs_matching']['laborer_preference']['gamma_2']
    gamma_3 = config['gs_matching']['laborer_preference']['gamma_3']
    gamma_4 = config['gs_matching']['laborer_preference']['gamma_4']
    
    print(f"\n当前系数:")
    print(f"  γ_0 (截距) = {gamma_0}")
    print(f"  γ_1 (T_req) = {gamma_1}")
    print(f"  γ_2 (S_req-S_i) = {gamma_2}")
    print(f"  γ_3 (D_req-D_i) = {gamma_3}")
    print(f"  γ_4 (W_offer) = {gamma_4}")
    
    # 新方法：先计算原始偏好项矩阵（以第一个劳动力为例）
    i = 0
    n_enterprises = len(T_req)
    
    # 步骤1: 计算原始偏好项
    term_T_raw = np.zeros(n_enterprises)
    term_S_raw = np.zeros(n_enterprises)
    term_D_raw = np.zeros(n_enterprises)
    term_W_raw = np.zeros(n_enterprises)
    
    for j in range(n_enterprises):
        term_T_raw[j] = T_req[j]
        term_S_raw[j] = max(0.0, S_req[j] - S_i[i])
        term_D_raw[j] = max(0.0, D_req[j] - D_i[i])
        term_W_raw[j] = W_offer[j]
    
    # 步骤2: 对每一项单独MinMax标准化
    T_min, T_max = term_T_raw.min(), term_T_raw.max()
    S_min, S_max = term_S_raw.min(), term_S_raw.max()
    D_min, D_max = term_D_raw.min(), term_D_raw.max()
    W_min, W_max = term_W_raw.min(), term_W_raw.max()
    
    term_T_norm = (term_T_raw - T_min) / (T_max - T_min + 1e-10)
    term_S_norm = (term_S_raw - S_min) / (S_max - S_min + 1e-10)
    term_D_norm = (term_D_raw - D_min) / (D_max - D_min + 1e-10)
    term_W_norm = (term_W_raw - W_min) / (W_max - W_min + 1e-10)
    
    # 步骤3: 加权求和
    term_T = -gamma_1 * term_T_norm
    term_S = -gamma_2 * term_S_norm
    term_D = -gamma_3 * term_D_norm
    term_W = gamma_4 * term_W_norm
    
    print(f"\n第一个劳动力对所有企业的偏好各项统计:")
    print(f"  T项 (-γ_1*T_j_norm):")
    print(f"    范围: [{term_T.min():.4f}, {term_T.max():.4f}]")
    print(f"    均值: {term_T.mean():.4f} ± {term_T.std():.4f}")
    
    print(f"  S项 (-γ_2*max(0,S_j-S_i)):")
    print(f"    范围: [{term_S.min():.4f}, {term_S.max():.4f}]")
    print(f"    均值: {term_S.mean():.4f} ± {term_S.std():.4f}")
    
    print(f"  D项 (-γ_3*max(0,D_j-D_i)):")
    print(f"    范围: [{term_D.min():.4f}, {term_D.max():.4f}]")
    print(f"    均值: {term_D.mean():.4f} ± {term_D.std():.4f}")
    
    print(f"  W项 (+γ_4*W_j_norm):")
    print(f"    范围: [{term_W.min():.4f}, {term_W.max():.4f}]")
    print(f"    均值: {term_W.mean():.4f} ± {term_W.std():.4f}")
    
    # 总偏好
    total_pref = gamma_0 + term_T + term_S + term_D + term_W
    print(f"\n  总偏好:")
    print(f"    范围: [{total_pref.min():.4f}, {total_pref.max():.4f}]")
    print(f"    均值: {total_pref.mean():.4f} ± {total_pref.std():.4f}")
    
    # 计算各项的贡献占比（用绝对值）
    abs_T = np.abs(term_T).mean()
    abs_S = np.abs(term_S).mean()
    abs_D = np.abs(term_D).mean()
    abs_W = np.abs(term_W).mean()
    total_abs = abs_T + abs_S + abs_D + abs_W
    
    print(f"\n各项平均贡献占比（绝对值）:")
    print(f"  T项: {abs_T/total_abs*100:.1f}%")
    print(f"  S项: {abs_S/total_abs*100:.1f}%")
    print(f"  D项: {abs_D/total_abs*100:.1f}%")
    print(f"  W项: {abs_W/total_abs*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("企业偏好函数分析（新版本：先计算原始偏好项，再对每项标准化）")
    print("=" * 80)
    
    # 企业偏好函数参数
    beta_0 = config['gs_matching']['enterprise_preference']['beta_0']
    beta_1 = config['gs_matching']['enterprise_preference']['beta_1']
    beta_2 = config['gs_matching']['enterprise_preference']['beta_2']
    beta_3 = config['gs_matching']['enterprise_preference']['beta_3']
    beta_4 = config['gs_matching']['enterprise_preference']['beta_4']
    
    print(f"\n当前系数:")
    print(f"  β_0 (截距) = {beta_0}")
    print(f"  β_1 (T) = {beta_1}")
    print(f"  β_2 (S) = {beta_2}")
    print(f"  β_3 (D) = {beta_3}")
    print(f"  β_4 (W) = {beta_4}")
    
    # 新方法：先计算原始偏好项
    # 步骤1: 各项就是原始值（企业偏好函数更简单）
    term_T_e_raw = T_i.copy()
    term_S_e_raw = S_i.copy()
    term_D_e_raw = D_i.copy()
    term_W_e_raw = W_i.copy()
    
    # 步骤2: 对每一项单独MinMax标准化
    T_e_min, T_e_max = term_T_e_raw.min(), term_T_e_raw.max()
    S_e_min, S_e_max = term_S_e_raw.min(), term_S_e_raw.max()
    D_e_min, D_e_max = term_D_e_raw.min(), term_D_e_raw.max()
    W_e_min, W_e_max = term_W_e_raw.min(), term_W_e_raw.max()
    
    term_T_e_norm = (term_T_e_raw - T_e_min) / (T_e_max - T_e_min + 1e-10)
    term_S_e_norm = (term_S_e_raw - S_e_min) / (S_e_max - S_e_min + 1e-10)
    term_D_e_norm = (term_D_e_raw - D_e_min) / (D_e_max - D_e_min + 1e-10)
    term_W_e_norm = (term_W_e_raw - W_e_min) / (W_e_max - W_e_min + 1e-10)
    
    # 步骤3: 加权求和
    term_T_e = beta_1 * term_T_e_norm
    term_S_e = beta_2 * term_S_e_norm
    term_D_e = beta_3 * term_D_e_norm
    term_W_e = beta_4 * term_W_e_norm
    
    print(f"\n第一个企业对所有劳动力的偏好各项统计:")
    print(f"  T项 (β_1*T_i_norm):")
    print(f"    范围: [{term_T_e.min():.4f}, {term_T_e.max():.4f}]")
    print(f"    均值: {term_T_e.mean():.4f} ± {term_T_e.std():.4f}")
    
    print(f"  S项 (β_2*S_i_norm):")
    print(f"    范围: [{term_S_e.min():.4f}, {term_S_e.max():.4f}]")
    print(f"    均值: {term_S_e.mean():.4f} ± {term_S_e.std():.4f}")
    
    print(f"  D项 (β_3*D_i_norm):")
    print(f"    范围: [{term_D_e.min():.4f}, {term_D_e.max():.4f}]")
    print(f"    均值: {term_D_e.mean():.4f} ± {term_D_e.std():.4f}")
    
    print(f"  W项 (β_4*W_i_norm):")
    print(f"    范围: [{term_W_e.min():.4f}, {term_W_e.max():.4f}]")
    print(f"    均值: {term_W_e.mean():.4f} ± {term_W_e.std():.4f}")
    
    # 总偏好
    total_pref_e = beta_0 + term_T_e + term_S_e + term_D_e + term_W_e
    print(f"\n  总偏好:")
    print(f"    范围: [{total_pref_e.min():.4f}, {total_pref_e.max():.4f}]")
    print(f"    均值: {total_pref_e.mean():.4f} ± {total_pref_e.std():.4f}")
    
    # 计算各项的贡献占比
    abs_T_e = np.abs(term_T_e).mean()
    abs_S_e = np.abs(term_S_e).mean()
    abs_D_e = np.abs(term_D_e).mean()
    abs_W_e = np.abs(term_W_e).mean()
    total_abs_e = abs_T_e + abs_S_e + abs_D_e + abs_W_e
    
    print(f"\n各项平均贡献占比（绝对值）:")
    print(f"  T项: {abs_T_e/total_abs_e*100:.1f}%")
    print(f"  S项: {abs_S_e/total_abs_e*100:.1f}%")
    print(f"  D项: {abs_D_e/total_abs_e*100:.1f}%")
    print(f"  W项: {abs_W_e/total_abs_e*100:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_preference_contributions()

