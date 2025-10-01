"""
匹配结果数据结构

封装匹配结果及相关信息，便于后续分析和使用。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class MatchingResult:
    """
    匹配结果数据类
    
    Attributes:
        labor_agents: 劳动力DataFrame
        enterprise_agents: 企业DataFrame
        matching: (n_labor,) 匹配结果数组
        labor_preference: (n_labor, n_enterprise) 劳动力偏好矩阵
        enterprise_preference: (n_enterprise, n_labor) 企业偏好矩阵
        is_stable: 是否稳定
        unstable_pairs: 不稳定匹配对列表
        statistics: 统计信息字典
    """
    
    labor_agents: pd.DataFrame
    enterprise_agents: pd.DataFrame
    matching: np.ndarray
    labor_preference: np.ndarray
    enterprise_preference: np.ndarray
    is_stable: bool
    unstable_pairs: List[Tuple[int, int]]
    statistics: Optional[Dict] = None
    
    def __post_init__(self):
        """初始化后计算统计信息"""
        if self.statistics is None:
            self.statistics = self._compute_statistics()
    
    def _compute_statistics(self) -> Dict:
        """计算匹配统计信息"""
        n_labor = len(self.matching)
        n_matched = np.sum(self.matching != -1)
        
        stats = {
            'n_labor': n_labor,
            'n_enterprise': len(self.enterprise_agents),
            'n_matched': int(n_matched),
            'n_unmatched': int(n_labor - n_matched),
            'match_rate': float(n_matched / n_labor) if n_labor > 0 else 0.0,
            'unemployment_rate': float((n_labor - n_matched) / n_labor) if n_labor > 0 else 0.0,
            'is_stable': self.is_stable,
            'n_unstable_pairs': len(self.unstable_pairs)
        }
        
        return stats
    
    def get_matched_pairs(self) -> pd.DataFrame:
        """
        获取所有匹配对的详细信息
        
        Returns:
            DataFrame包含：labor_id, enterprise_id, labor特征, enterprise特征
        """
        matched_idx = self.matching != -1
        matched_labor_ids = np.where(matched_idx)[0]
        matched_enterprise_ids = self.matching[matched_idx]
        
        # 构建匹配对DataFrame
        pairs = []
        for labor_id, enterprise_id in zip(matched_labor_ids, matched_enterprise_ids):
            pair = {
                'labor_id': labor_id,
                'enterprise_id': enterprise_id,
                'labor_T': self.labor_agents.iloc[labor_id]['T'],
                'labor_S': self.labor_agents.iloc[labor_id]['S'],
                'labor_D': self.labor_agents.iloc[labor_id]['D'],
                'labor_W': self.labor_agents.iloc[labor_id]['W'],
                'enterprise_T': self.enterprise_agents.iloc[enterprise_id]['T'],
                'enterprise_S': self.enterprise_agents.iloc[enterprise_id]['S'],
                'enterprise_D': self.enterprise_agents.iloc[enterprise_id]['D'],
                'enterprise_W': self.enterprise_agents.iloc[enterprise_id]['W']
            }
            pairs.append(pair)
        
        return pd.DataFrame(pairs)
    
    def get_unmatched_labor(self) -> pd.DataFrame:
        """
        获取未匹配劳动力的详细信息
        
        Returns:
            DataFrame包含未匹配劳动力的所有特征
        """
        unmatched_idx = self.matching == -1
        return self.labor_agents[unmatched_idx].copy()
    
    def get_vacant_enterprises(self) -> pd.DataFrame:
        """
        获取空缺职位的企业详细信息
        
        Returns:
            DataFrame包含有空缺职位企业的所有特征
        """
        # 找出未被匹配的企业
        matched_enterprise_ids = set(self.matching[self.matching != -1])
        all_enterprise_ids = set(range(len(self.enterprise_agents)))
        vacant_enterprise_ids = all_enterprise_ids - matched_enterprise_ids
        
        if len(vacant_enterprise_ids) == 0:
            return pd.DataFrame()
        
        return self.enterprise_agents.iloc[list(vacant_enterprise_ids)].copy()
    
    def compute_match_quality(self) -> Dict[str, float]:
        """
        计算匹配质量指标
        
        Returns:
            质量指标字典
        """
        if self.statistics['n_matched'] == 0:
            return {
                'avg_labor_satisfaction': 0.0,
                'avg_enterprise_satisfaction': 0.0,
                'avg_total_satisfaction': 0.0
            }
        
        matched_idx = self.matching != -1
        matched_labor_ids = np.where(matched_idx)[0]
        matched_enterprise_ids = self.matching[matched_idx]
        
        # 劳动力满意度：匹配到的企业在其偏好列表中的排名
        labor_satisfaction = []
        for i, j in zip(matched_labor_ids, matched_enterprise_ids):
            # 找到企业j在劳动力i偏好中的排名（越小越好）
            rank = np.where(np.argsort(-self.labor_preference[i]) == j)[0][0]
            # 转换为满意度（0-1，1最好）
            satisfaction = 1.0 - rank / len(self.enterprise_agents)
            labor_satisfaction.append(satisfaction)
        
        # 企业满意度：匹配到的劳动力在其偏好列表中的排名
        enterprise_satisfaction = []
        for i, j in zip(matched_labor_ids, matched_enterprise_ids):
            # 找到劳动力i在企业j偏好中的排名
            rank = np.where(np.argsort(-self.enterprise_preference[j]) == i)[0][0]
            # 转换为满意度
            satisfaction = 1.0 - rank / len(self.labor_agents)
            enterprise_satisfaction.append(satisfaction)
        
        return {
            'avg_labor_satisfaction': float(np.mean(labor_satisfaction)),
            'avg_enterprise_satisfaction': float(np.mean(enterprise_satisfaction)),
            'avg_total_satisfaction': float(np.mean(labor_satisfaction + enterprise_satisfaction))
        }
    
    def to_dict(self) -> Dict:
        """
        转换为字典格式（用于序列化）
        
        Returns:
            包含所有信息的字典
        """
        return {
            'matching': self.matching.tolist(),
            'is_stable': self.is_stable,
            'unstable_pairs': self.unstable_pairs,
            'statistics': self.statistics,
            'match_quality': self.compute_match_quality()
        }
    
    def summary(self) -> str:
        """
        生成匹配结果摘要
        
        Returns:
            摘要字符串
        """
        quality = self.compute_match_quality()
        
        summary = f"""
=== 匹配结果摘要 ===

基本信息:
- 劳动力总数: {self.statistics['n_labor']}
- 企业总数: {self.statistics['n_enterprise']}
- 匹配成功数: {self.statistics['n_matched']}
- 未匹配劳动力: {self.statistics['n_unmatched']}
- 匹配率: {self.statistics['match_rate']:.2%}
- 失业率: {self.statistics['unemployment_rate']:.2%}

稳定性:
- 是否稳定: {'是' if self.is_stable else '否'}
- 不稳定匹配对数: {self.statistics['n_unstable_pairs']}

匹配质量:
- 劳动力平均满意度: {quality['avg_labor_satisfaction']:.4f}
- 企业平均满意度: {quality['avg_enterprise_satisfaction']:.4f}
- 总体平均满意度: {quality['avg_total_satisfaction']:.4f}
"""
        return summary.strip()
    
    def __repr__(self) -> str:
        return f"MatchingResult(n_matched={self.statistics['n_matched']}, " \
               f"match_rate={self.statistics['match_rate']:.2%}, " \
               f"is_stable={self.is_stable})"

