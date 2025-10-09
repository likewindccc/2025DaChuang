#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匹配函数回归模块

通过Logit回归拟合匹配函数λ(x,σ,θ)，建立劳动力特征与匹配概率的关系。

功能:
    - 生成多批次虚拟市场数据（覆盖不同theta场景）
    - 执行GS匹配得到匹配结果
    - 构建Logit回归数据集
    - 拟合Logit回归模型
    - 保存回归参数
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any
import statsmodels.api as sm

from .virtual_market import VirtualMarket
from .gs_matching import perform_matching


class MatchFunction:
    """
    匹配函数类
    
    职责:
        1. 生成多轮虚拟市场并执行匹配
        2. 构建Logit回归数据集
        3. 拟合匹配函数λ(x,σ,θ)
        4. 保存回归结果
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化匹配函数"""
        self.config = config
        self.market_generator = VirtualMarket(config)
        self.regression_data = None  # 回归数据集
        self.model = None  # Logit模型
        self.params = None  # 回归参数
        
        np.random.seed(config['random_seed'])
    
    def generate_training_data(self) -> pd.DataFrame:
        """
        生成训练数据
        
        根据配置生成多批次虚拟市场，执行GS匹配，构建回归数据集。
        
        Returns:
            包含(x, σ, θ, matched)的DataFrame
            
        数据生成策略:
            - 按照theta_scenarios的权重分配轮数
            - 每轮生成n_laborers个劳动力
            - 记录每个劳动力的特征和匹配结果
        """
        # 读取配置
        data_gen_config = self.config['data_generation']
        n_rounds = data_gen_config['n_rounds']
        theta_scenarios = data_gen_config['theta_scenarios']
        n_laborers = self.config['market_size']['n_laborers']
        
        # 根据权重分配轮数
        n_tight = int(n_rounds * theta_scenarios['tight']['weight'])
        n_balanced = int(n_rounds * theta_scenarios['balanced']['weight'])
        n_surplus = n_rounds - n_tight - n_balanced
        
        # 生成theta列表
        theta_list = []
        
        # 岗位紧张型市场
        theta_list.extend(
            np.random.uniform(
                theta_scenarios['tight']['min'],
                theta_scenarios['tight']['max'],
                n_tight
            )
        )
        
        # 均衡市场（0.9-1.1均匀分布）
        theta_list.extend(
            np.random.uniform(
                theta_scenarios['balanced']['min'],
                theta_scenarios['balanced']['max'],
                n_balanced
            )
        )
        
        # 岗位富余型市场
        theta_list.extend(
            np.random.uniform(
                theta_scenarios['surplus']['min'],
                theta_scenarios['surplus']['max'],
                n_surplus
            )
        )
        
        # 随机打乱
        np.random.shuffle(theta_list)
        
        # 收集所有轮次的数据
        all_data = []
        
        print(f"\n   开始生成{n_rounds}轮数据...")
        print(f"   - 岗位紧张型: {n_tight}轮 (theta∈[{theta_scenarios['tight']['min']}, {theta_scenarios['tight']['max']}])")
        print(f"   - 均衡市场: {n_balanced}轮 (theta∈[{theta_scenarios['balanced']['min']}, {theta_scenarios['balanced']['max']}])")
        print(f"   - 岗位富余型: {n_surplus}轮 (theta∈[{theta_scenarios['surplus']['min']}, {theta_scenarios['surplus']['max']}])")
        print()
        
        for round_idx, theta in enumerate(theta_list):
            # 生成虚拟市场
            laborers, enterprises = self.market_generator.generate_market(
                n_laborers=n_laborers,
                theta=theta
            )
            
            # 执行GS匹配（已集成numba加速）
            match_result = perform_matching(laborers, enterprises, self.config)
            
            # 每轮显示进度
            match_rate = match_result['matched'].mean() * 100
            print(f"   [{round_idx + 1}/{n_rounds}] theta={theta:.3f}, 匹配率={match_rate:.1f}%")
            
            # 提取回归所需变量
            # x: 劳动力特征 (T, S, D, W)
            # σ: 劳动力控制变量的综合指标
            # θ: 市场紧张度
            # matched: 匹配结果（0/1）
            
            # 计算控制变量σ（劳动力自身特征的综合指标）
            # σ = minmax(minmax(age) + minmax(edu) + minmax(children))
            
            # 提取控制变量
            age_raw = match_result['age'].values
            edu_raw = match_result['edu'].values
            children_raw = match_result['children'].values
            
            # 第一次MinMax标准化
            age_min, age_max = age_raw.min(), age_raw.max()
            age_norm = (age_raw - age_min) / (age_max - age_min + 1e-10)
            
            edu_min, edu_max = edu_raw.min(), edu_raw.max()
            edu_norm = (edu_raw - edu_min) / (edu_max - edu_min + 1e-10)
            
            children_min, children_max = children_raw.min(), children_raw.max()
            children_norm = (children_raw - children_min) / (children_max - children_min + 1e-10)
            
            # 求和
            sigma_sum = age_norm + edu_norm + children_norm
            
            # 第二次MinMax标准化
            sigma_min, sigma_max = sigma_sum.min(), sigma_sum.max()
            sigma = (sigma_sum - sigma_min) / (sigma_max - sigma_min + 1e-10)
            
            # 构建数据
            round_data = match_result[['T', 'S', 'D', 'W', 'theta', 'matched']].copy()
            round_data['sigma'] = sigma
            round_data['round'] = round_idx
            
            all_data.append(round_data)
        
        # 合并所有轮次数据
        print(f"\n   合并数据...")
        self.regression_data = pd.concat(all_data, ignore_index=True)
        
        # 显示数据生成完成信息
        total_matched = self.regression_data['matched'].sum()
        total_samples = len(self.regression_data)
        overall_match_rate = total_matched / total_samples * 100
        print(f"   [完成] 总样本: {total_samples}, 总匹配: {total_matched}, 总体匹配率: {overall_match_rate:.2f}%\n")
        
        return self.regression_data
    
    def fit(self) -> None:
        """
        拟合Logit回归模型
        
        使用已生成的训练数据（self.regression_data）进行拟合。
        
        回归方程:
            logit(P(matched=1)) = β_0 + β_1*T + β_2*S + β_3*D + β_4*W 
                                  + β_5*sigma + β_6*theta
        
        说明:
            - 因变量: matched (0/1)
            - 自变量: 劳动力核心特征(T,S,D,W) + 控制变量(σ) + 市场紧张度(θ)
            - σ: 劳动力控制变量综合指标（age, edu, children的二次标准化）
        """
        # 直接使用已生成的训练数据
        data = self.regression_data
        
        # 构建回归变量
        # 因变量
        y = data['matched']
        
        # 自变量
        X = data[[
            'T', 'S', 'D', 'W',  # 劳动力核心特征
            'sigma',  # 控制变量综合指标
            'theta'   # 市场紧张度
        ]]
        
        # 数据清洗：处理NaN和inf
        # 统计原始样本数
        n_original = len(data)
        
        # 将inf替换为nan，然后统一删除
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        y_clean = y.copy()
        
        # 找出包含nan的行
        mask = X_clean.isnull().any(axis=1)
        n_invalid = mask.sum()
        
        if n_invalid > 0:
            print(f"\n警告：发现{n_invalid}个异常样本（包含NaN或inf），已删除")
            print(f"  删除前样本数: {n_original}")
            print(f"  删除后样本数: {n_original - n_invalid}")
            print(f"  删除比例: {n_invalid/n_original*100:.2f}%")
            
            # 删除异常样本
            X_clean = X_clean[~mask]
            y_clean = y_clean[~mask]
        
        # 添加常数项
        X = sm.add_constant(X_clean)
        
        # 拟合Logit模型（使用清洗后的y）
        self.model = sm.Logit(y_clean, X).fit(disp=0)  # disp=0关闭优化输出
        self.params = self.model.params
    
    def save_results(self) -> None:
        """
        保存回归结果（硬编码路径）
        
        保存内容:
            - 回归参数
            - 模型摘要统计
            - 训练数据（可选）
        """
        # 硬编码保存路径
        output_dir = Path("OUTPUT/logistic")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存回归参数
        params_file = output_dir / "match_function_params.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'model_summary': self.model.summary2().tables[1]  # 参数表
            }, f)
        
        # 保存完整模型（用于预测）
        model_file = output_dir / "match_function_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测匹配概率
        
        Args:
            X: 特征DataFrame，需包含所有自变量（T, S, D, W, sigma, theta）
        
        Returns:
            匹配概率数组
            
        说明:
            - 添加常数项：在fit()时对训练数据添加了常数项，在predict()时也要对新数据添加
            - 这样才能保证预测时特征维度与训练时一致
        """
        # 添加常数项（与训练时保持一致）
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # 预测
        return self.model.predict(X_with_const)


def load_config(config_path: str = "CONFIG/logistic_config.yaml") -> Dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

