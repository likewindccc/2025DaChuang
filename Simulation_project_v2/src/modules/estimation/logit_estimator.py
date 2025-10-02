"""
Logit估计器

基于ABM生成的训练数据，使用Logit回归估计匹配函数λ(x, σ_i, a, θ)的参数。

匹配函数形式（原始研究计划）：
λ(x, σ_i, a, θ) = 1 / (1 + exp[-(δ_0 + δ_x'x + δ_σ'σ_i + δ_a·a + δ_θ·ln(θ))])

其中：
- δ_0: 截距项
- δ_x: 状态变量x的系数向量 (T, S, D, W)
- δ_σ: 固定特征σ的系数向量 (年龄, 学历, 孩子数, 累计工作年限)
- δ_a: 努力水平a的系数
- δ_θ: 市场松紧度ln(θ)的系数
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging
from pathlib import Path
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
except ImportError:
    raise ImportError("需要安装statsmodels: pip install statsmodels")


logger = logging.getLogger(__name__)


class LogitEstimator:
    """
    Logit估计器
    
    使用statsmodels对ABM生成的训练数据进行Logit回归，
    估计匹配函数λ的参数。
    """
    
    def __init__(self):
        """初始化Logit估计器"""
        self.model: Optional[Logit] = None
        self.result: Optional[sm.discrete.discrete_model.BinaryResultsWrapper] = None
        self.params: Optional[pd.Series] = None
        self.feature_names: Optional[List[str]] = None
        
        logger.info("Logit估计器初始化完成")
    
    def fit(
        self,
        data: pd.DataFrame,
        include_control_vars: bool = True,
        add_intercept: bool = True
    ) -> Dict:
        """
        拟合Logit模型
        
        Args:
            data: ABM生成的训练数据
            include_control_vars: 是否包含控制变量σ（年龄、学历等）
            add_intercept: 是否添加截距项
        
        Returns:
            拟合结果摘要字典
        """
        logger.info("开始Logit回归估计")
        
        # Step 1: 准备特征和目标变量
        X, y, feature_names = self._prepare_features(
            data,
            include_control_vars=include_control_vars
        )
        
        # Step 2: 添加截距项
        if add_intercept:
            X = sm.add_constant(X, has_constant='add')
            feature_names = ['const'] + feature_names
        
        self.feature_names = feature_names
        
        logger.info(f"特征数量: {len(feature_names)}")
        logger.info(f"样本数量: {len(y)}")
        logger.info(f"匹配率: {y.mean():.2%}")
        
        # Step 3: 拟合Logit模型
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            self.model = Logit(y, X)
            self.result = self.model.fit(disp=0)
        
        # Step 4: 提取参数（直接引用result.params，它已经是pandas Series）
        self.params = self.result.params
        
        logger.info("Logit回归估计完成")
        
        # Step 5: 生成摘要
        summary = self._generate_summary()
        
        return summary
    
    def _prepare_features(
        self,
        data: pd.DataFrame,
        include_control_vars: bool
    ) -> tuple:
        """
        准备回归特征
        
        Args:
            data: 原始数据
            include_control_vars: 是否包含控制变量
        
        Returns:
            (X, y, feature_names)
        """
        # 状态变量 x (T, S, D, W)
        state_features = ['labor_T', 'labor_S', 'labor_D', 'labor_W']
        
        # 控制变量 σ（从Population生成器中获取）
        # 注意：ABM数据中没有直接记录控制变量，需要从劳动力ID关联
        # 简化方案：使用劳动力与市场的差距作为代理变量
        control_features = []
        if include_control_vars:
            control_features = [
                'labor_market_gap_T',
                'labor_market_gap_S',
                'labor_market_gap_D',
                'labor_market_gap_W'
            ]
        
        # 努力水平 a
        effort_features = ['effort']
        
        # 市场松紧度 ln(θ)
        data['ln_theta'] = np.log(data['theta'])
        theta_features = ['ln_theta']
        
        # 合并所有特征
        all_features = (
            state_features +
            control_features +
            effort_features +
            theta_features
        )
        
        feature_names = (
            [f'delta_{f}' for f in state_features] +
            [f'delta_sigma_{f}' for f in control_features] +
            ['delta_a'] +
            ['delta_theta']
        )
        
        # 提取特征和目标
        X = data[all_features].values
        y = data['matched'].values
        
        return X, y, feature_names
    
    def _generate_summary(self) -> Dict:
        """
        生成估计结果摘要
        
        Returns:
            摘要字典
        """
        if self.result is None:
            raise RuntimeError("模型尚未拟合")
        
        # 将numpy数组转换为字典
        params_dict = {
            name: float(val)
            for name, val in zip(self.feature_names, self.params)
        }
        
        pvalues_dict = {
            name: float(val)
            for name, val in zip(self.feature_names, self.result.pvalues)
        }
        
        # 置信区间
        conf_int = self.result.conf_int()
        conf_int_dict = {
            name: [float(conf_int[i, 0]), float(conf_int[i, 1])]
            for i, name in enumerate(self.feature_names)
        }
        
        summary = {
            'params': params_dict,
            'pvalues': pvalues_dict,
            'conf_int': conf_int_dict,
            'pseudo_r2': float(self.result.prsquared),
            'log_likelihood': float(self.result.llf),
            'aic': float(self.result.aic),
            'bic': float(self.result.bic),
            'n_obs': int(self.result.nobs),
            'converged': self.result.mle_retvals['converged']
        }
        
        return summary
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测匹配概率
        
        Args:
            X: 特征DataFrame
        
        Returns:
            匹配概率数组
        """
        if self.result is None:
            raise RuntimeError("模型尚未拟合")
        
        # 准备特征
        X_prepared, _, _ = self._prepare_features(
            X,
            include_control_vars=True
        )
        
        # 添加截距项
        X_prepared = sm.add_constant(X_prepared, has_constant='add')
        
        # 预测
        proba = self.result.predict(X_prepared)
        
        return proba
    
    def print_summary(self):
        """打印详细估计结果"""
        if self.result is None:
            raise RuntimeError("模型尚未拟合")
        
        print("\n" + "=" * 80)
        print("Logit回归估计结果")
        print("=" * 80)
        print(self.result.summary())
        print("=" * 80)
    
    def save_params(self, save_path: str):
        """
        保存估计参数
        
        Args:
            save_path: 保存路径（JSON格式）
        """
        if self.params is None:
            raise RuntimeError("模型尚未拟合")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建参数字典（params是numpy数组）
        params_dict = {
            'params': {
                name: float(val)
                for name, val in zip(self.feature_names, self.params)
            },
            'feature_names': self.feature_names,
            'summary': self._generate_summary()
        }
        
        # 保存为JSON
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"参数已保存至: {save_path}")
    
    def load_params(self, load_path: str):
        """
        加载估计参数
        
        Args:
            load_path: 参数文件路径
        """
        import json
        
        with open(load_path, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
        
        # 加载参数（转换为numpy数组以保持一致性）
        self.feature_names = params_dict['feature_names']
        self.params = np.array([
            params_dict['params'][name]
            for name in self.feature_names
        ])
        
        logger.info(f"参数已从{load_path}加载")
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
        
        Returns:
            评估指标字典
        """
        if self.result is None:
            raise RuntimeError("模型尚未拟合")
        
        # 预测
        y_pred_proba = self.predict(test_data)
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = test_data['matched'].values
        
        # 计算评估指标
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'auc': float(roc_auc_score(y_true, y_pred_proba))
        }
        
        return metrics
    
    def __repr__(self) -> str:
        status = "已拟合" if self.result is not None else "未拟合"
        return f"LogitEstimator(status={status})"

