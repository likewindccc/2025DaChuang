"""
ABM数据生成器

通过基于主体的建模(Agent-Based Modeling)生成用于Logit回归的训练数据。
核心思想：在不同市场紧张度θ和努力水平a下，执行多轮次Gale-Shapley匹配，
记录每个劳动力的特征、环境参数和匹配结果，为后续估计匹配函数提供数据。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

from ..population.labor_generator import LaborGenerator
from ..population.enterprise_generator import EnterpriseGenerator
from .matching_engine import MatchingEngine


logger = logging.getLogger(__name__)


class ABMDataGenerator:
    """
    ABM数据生成器
    
    整合Population生成器和Matching引擎，通过多轮次模拟生成
    用于Logit回归的训练数据集。
    """
    
    def __init__(
        self,
        labor_generator: Optional[LaborGenerator] = None,
        enterprise_generator: Optional[EnterpriseGenerator] = None,
        matching_engine: Optional[MatchingEngine] = None,
        seed: Optional[int] = None,
        auto_fit: bool = True
    ):
        """
        初始化ABM数据生成器
        
        Args:
            labor_generator: 劳动力生成器（None则使用默认，需已fit）
            enterprise_generator: 企业生成器（None则使用默认，需已fit）
            matching_engine: 匹配引擎（None则使用默认）
            seed: 随机种子（用于控制ABM模拟的随机性）
            auto_fit: 是否自动fit生成器（如果尚未fit）
        """
        self.labor_gen = labor_generator or LaborGenerator()
        self.enterprise_gen = enterprise_generator or EnterpriseGenerator()
        self.matching_engine = matching_engine or MatchingEngine()
        
        # 设置随机种子（影响ABM模拟流程）
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
        
        # 确保生成器已拟合
        if auto_fit:
            self._ensure_generators_fitted()
        
        self.data_records: List[Dict] = []
        
        logger.info("ABM数据生成器初始化完成")
    
    def _ensure_generators_fitted(self):
        """
        确保生成器已拟合（如果未拟合，则使用默认测试数据拟合）
        """
        # 检查劳动力生成器是否已拟合
        if not hasattr(self.labor_gen, 'is_fitted') or not self.labor_gen.is_fitted:
            logger.warning("劳动力生成器未拟合，使用默认测试数据拟合")
            test_labor_data = self._create_test_labor_data()
            self.labor_gen.fit(test_labor_data)
        
        # 检查企业生成器是否已拟合
        if not hasattr(self.enterprise_gen, 'is_fitted') or not self.enterprise_gen.is_fitted:
            logger.warning("企业生成器未拟合，使用默认测试数据拟合")
            test_enterprise_data = self._create_test_enterprise_data()
            self.enterprise_gen.fit(test_enterprise_data)
    
    def _create_test_labor_data(self) -> pd.DataFrame:
        """
        创建测试用劳动力数据
        
        Returns:
            测试数据DataFrame
        """
        np.random.seed(42 if self.seed is None else self.seed)
        n = 1000
        
        df = pd.DataFrame({
            '年龄': np.random.randint(18, 65, n),
            '孩子数量': np.random.randint(0, 4, n),
            '学历': np.random.randint(0, 7, n),
            '累计工作年限': np.random.randint(0, 45, n),
            'T': np.random.rand(n) * 60,
            'S': np.random.rand(n),
            'D': np.random.rand(n),
            'W': np.random.rand(n) * 5000 + 2000
        })
        
        return df
    
    def _create_test_enterprise_data(self) -> pd.DataFrame:
        """
        创建测试用企业数据
        
        Returns:
            测试数据DataFrame
        """
        np.random.seed(43 if self.seed is None else self.seed + 1)
        n = 500
        
        df = pd.DataFrame({
            'T': np.random.rand(n) * 60,
            'S': np.random.rand(n),
            'D': np.random.rand(n),
            'W': np.random.rand(n) * 6000 + 2500
        })
        
        return df
    
    def generate_training_data(
        self,
        theta_range: List[float] = None,
        effort_levels: List[float] = None,
        n_rounds_per_combination: int = 5,
        base_n_labor: int = 1000,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        生成训练数据集
        
        Args:
            theta_range: θ值列表（岗位数/求职者数）
            effort_levels: 努力水平a列表
            n_rounds_per_combination: 每个(θ,a)组合的模拟轮数
            base_n_labor: 基准劳动力数量
            verbose: 是否显示进度条
        
        Returns:
            训练数据DataFrame
        """
        # 默认扰动策略（基于原始研究计划）
        if theta_range is None:
            theta_range = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        if effort_levels is None:
            effort_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        total_simulations = (
            len(theta_range) * len(effort_levels) * n_rounds_per_combination
        )
        
        logger.info(
            f"开始生成训练数据: {len(theta_range)}个θ值 × "
            f"{len(effort_levels)}个努力水平 × {n_rounds_per_combination}轮 "
            f"= {total_simulations}次模拟"
        )
        
        self.data_records = []
        
        # 创建进度条
        if verbose:
            pbar = tqdm(
                total=total_simulations,
                desc="ABM数据生成",
                unit="轮"
            )
        
        # 嵌套循环：θ × a × rounds
        for theta in theta_range:
            for effort in effort_levels:
                for round_idx in range(n_rounds_per_combination):
                    self._simulate_one_round(
                        theta=theta,
                        effort=effort,
                        base_n_labor=base_n_labor,
                        round_idx=round_idx
                    )
                    
                    if verbose:
                        pbar.update(1)
        
        if verbose:
            pbar.close()
        
        # 转换为DataFrame
        df = pd.DataFrame(self.data_records)
        
        logger.info(
            f"数据生成完成: {len(df)}条记录, "
            f"匹配率={df['matched'].mean():.2%}"
        )
        
        return df
    
    def _simulate_one_round(
        self,
        theta: float,
        effort: float,
        base_n_labor: int,
        round_idx: int
    ):
        """
        执行单轮模拟（使用单轮匹配算法）
        
        关键变化：
        - 不再使用完整的GS算法（会迭代直到收敛）
        - 改用单轮匹配：所有劳动力同时向最偏好企业投递，企业择优
        - 这样即使θ>1，仍会有显著失业，真实反映匹配摩擦
        
        Args:
            theta: 市场松紧度（岗位数/求职者数）
            effort: 努力水平
            base_n_labor: 基准劳动力数量
            round_idx: 轮次索引
        """
        # 计算实际数量
        n_labor = base_n_labor
        n_enterprise = int(base_n_labor * theta)
        
        # 生成劳动力和企业
        # 注意：努力水平a是在匹配函数λ(x,σ_i,a,θ)中体现的外生参数
        # 而不是劳动力特征的一部分，因此在记录数据时单独保存
        labor_df = self.labor_gen.generate(n_agents=n_labor)
        enterprise_df = self.enterprise_gen.generate(n_agents=n_enterprise)
        
        # 执行固定轮次匹配（不再使用MatchingEngine，直接调用固定轮次匹配函数）
        from .preference import (
            compute_labor_preference_matrix,
            compute_enterprise_preference_matrix,
            compute_preference_rankings
        )
        from .gale_shapley import limited_rounds_matching
        
        # 提取特征
        labor_features = labor_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
        enterprise_features = enterprise_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
        
        # 计算偏好矩阵
        labor_pref = compute_labor_preference_matrix(
            labor_features, enterprise_features,
            **self.matching_engine.labor_pref_params
        )
        
        enterprise_pref = compute_enterprise_preference_matrix(
            enterprise_features, labor_features,
            **self.matching_engine.enterprise_pref_params
        )
        
        # 转换为排序
        labor_pref_order = compute_preference_rankings(labor_pref)
        enterprise_pref_order = compute_preference_rankings(enterprise_pref)
        
        # 执行固定轮次匹配（默认10轮）
        matching = limited_rounds_matching(labor_pref_order, enterprise_pref_order, max_rounds=10)
        
        # 记录每个劳动力的数据
        self._record_labor_data(
            labor_df=labor_df,
            enterprise_df=enterprise_df,
            matching=matching,
            theta=theta,
            effort=effort,
            round_idx=round_idx
        )
    
    def _record_labor_data(
        self,
        labor_df: pd.DataFrame,
        enterprise_df: pd.DataFrame,
        matching: np.ndarray,
        theta: float,
        effort: float,
        round_idx: int
    ):
        """
        记录劳动力数据
        
        将每个劳动力的特征、环境参数和匹配结果记录为一条样本。
        
        Args:
            labor_df: 劳动力DataFrame
            enterprise_df: 企业DataFrame
            matching: 匹配结果数组
            theta: 市场松紧度
            effort: 努力水平
            round_idx: 轮次索引
        """
        n_labor = len(labor_df)
        
        # 计算企业特征的市场平均值（用于刻画市场环境）
        enterprise_mean_T = enterprise_df['T'].mean()
        enterprise_mean_S = enterprise_df['S'].mean()
        enterprise_mean_D = enterprise_df['D'].mean()
        enterprise_mean_W = enterprise_df['W'].mean()
        enterprise_std_T = enterprise_df['T'].std()
        enterprise_std_S = enterprise_df['S'].std()
        enterprise_std_D = enterprise_df['D'].std()
        enterprise_std_W = enterprise_df['W'].std()
        
        for i in range(n_labor):
            # 劳动力特征
            T_i = labor_df.iloc[i]['T']
            S_i = labor_df.iloc[i]['S']
            D_i = labor_df.iloc[i]['D']
            W_i = labor_df.iloc[i]['W']
            
            # 匹配结果
            matched = 1 if matching[i] != -1 else 0
            
            # 如果匹配成功，记录匹配的企业特征
            if matched == 1:
                j = matching[i]
                T_j = enterprise_df.iloc[j]['T']
                S_j = enterprise_df.iloc[j]['S']
                D_j = enterprise_df.iloc[j]['D']
                W_j = enterprise_df.iloc[j]['W']
                
                # 计算特征差距
                gap_T = T_j - T_i
                gap_S = S_j - S_i
                gap_D = D_j - D_i
                gap_W = W_j - W_i
            else:
                T_j = np.nan
                S_j = np.nan
                D_j = np.nan
                W_j = np.nan
                gap_T = np.nan
                gap_S = np.nan
                gap_D = np.nan
                gap_W = np.nan
            
            # 构建记录
            record = {
                # 劳动力特征
                'labor_T': T_i,
                'labor_S': S_i,
                'labor_D': D_i,
                'labor_W': W_i,
                
                # 匹配的企业特征（如果有）
                'enterprise_T': T_j,
                'enterprise_S': S_j,
                'enterprise_D': D_j,
                'enterprise_W': W_j,
                
                # 特征差距
                'gap_T': gap_T,
                'gap_S': gap_S,
                'gap_D': gap_D,
                'gap_W': gap_W,
                
                # 市场环境统计量
                'market_mean_T': enterprise_mean_T,
                'market_mean_S': enterprise_mean_S,
                'market_mean_D': enterprise_mean_D,
                'market_mean_W': enterprise_mean_W,
                'market_std_T': enterprise_std_T,
                'market_std_S': enterprise_std_S,
                'market_std_D': enterprise_std_D,
                'market_std_W': enterprise_std_W,
                
                # 劳动力与市场平均的差距
                'labor_market_gap_T': T_i - enterprise_mean_T,
                'labor_market_gap_S': S_i - enterprise_mean_S,
                'labor_market_gap_D': D_i - enterprise_mean_D,
                'labor_market_gap_W': W_i - enterprise_mean_W,
                
                # 环境参数
                'theta': theta,
                'effort': effort,
                
                # 匹配结果（目标变量）
                'matched': matched,
                
                # 元数据
                'round_idx': round_idx,
                'labor_idx': i
            }
            
            self.data_records.append(record)
    
    def save_data(
        self,
        df: pd.DataFrame,
        save_path: str,
        format: str = 'csv'
    ):
        """
        保存数据
        
        Args:
            df: 数据DataFrame
            save_path: 保存路径
            format: 保存格式（'csv', 'parquet', 'pickle'）
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
        elif format == 'parquet':
            df.to_parquet(save_path, index=False)
        elif format == 'pickle':
            df.to_pickle(save_path)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logger.info(f"数据已保存至: {save_path}")
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        生成数据摘要统计
        
        Args:
            df: 训练数据DataFrame
        
        Returns:
            统计信息字典
        """
        summary = {
            'n_records': len(df),
            'n_matched': int(df['matched'].sum()),
            'match_rate': float(df['matched'].mean()),
            
            # 按θ分组统计
            'match_rate_by_theta': df.groupby('theta')['matched'].mean().to_dict(),
            
            # 按effort分组统计
            'match_rate_by_effort': df.groupby('effort')['matched'].mean().to_dict(),
            
            # 劳动力特征统计
            'labor_T_mean': float(df['labor_T'].mean()),
            'labor_S_mean': float(df['labor_S'].mean()),
            'labor_D_mean': float(df['labor_D'].mean()),
            'labor_W_mean': float(df['labor_W'].mean()),
            
            # theta分布
            'theta_distribution': df['theta'].value_counts().to_dict(),
            
            # effort分布
            'effort_distribution': df['effort'].value_counts().to_dict()
        }
        
        return summary
    
    def plot_match_rate_heatmap(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制匹配率热力图（θ × effort）
        
        Args:
            df: 训练数据DataFrame
            save_path: 保存路径（None则不保存）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("未安装matplotlib/seaborn，跳过绘图")
            return
        
        # 计算每个(θ, effort)组合的匹配率
        pivot_table = df.pivot_table(
            values='matched',
            index='effort',
            columns='theta',
            aggfunc='mean'
        )
        
        # 绘图
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.2%',
            cmap='YlGnBu',
            cbar_kws={'label': '匹配率'}
        )
        plt.title('匹配率热力图（θ × effort）', fontsize=14)
        plt.xlabel('市场松紧度 θ', fontsize=12)
        plt.ylabel('努力水平 a', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"热力图已保存至: {save_path}")
        
        plt.show()
    
    def __repr__(self) -> str:
        return (
            f"ABMDataGenerator("
            f"labor_gen={self.labor_gen}, "
            f"enterprise_gen={self.enterprise_gen}, "
            f"n_records={len(self.data_records)})"
        )

