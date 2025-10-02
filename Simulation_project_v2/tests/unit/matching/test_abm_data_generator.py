"""
测试ABM数据生成器
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.modules.matching.abm_data_generator import ABMDataGenerator
from src.modules.population.labor_generator import LaborGenerator
from src.modules.population.enterprise_generator import EnterpriseGenerator
from src.modules.matching.matching_engine import MatchingEngine


class TestABMDataGeneratorInit:
    """测试初始化"""
    
    def test_default_init(self):
        """测试默认初始化"""
        generator = ABMDataGenerator(seed=42)
        
        assert generator is not None
        assert generator.labor_gen is not None
        assert generator.enterprise_gen is not None
        assert generator.matching_engine is not None
        assert len(generator.data_records) == 0
    
    def test_custom_components_init(self):
        """测试使用自定义组件初始化"""
        labor_gen = LaborGenerator()
        enterprise_gen = EnterpriseGenerator()
        matching_engine = MatchingEngine()
        
        generator = ABMDataGenerator(
            labor_generator=labor_gen,
            enterprise_generator=enterprise_gen,
            matching_engine=matching_engine,
            seed=42
        )
        
        assert generator.labor_gen is labor_gen
        assert generator.enterprise_gen is enterprise_gen
        assert generator.matching_engine is matching_engine
        assert generator.seed == 42


class TestSimulateOneRound:
    """测试单轮模拟"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.generator = ABMDataGenerator(seed=42)
    
    def test_simulate_theta_less_than_one(self):
        """测试θ<1的情况（劳动力过剩）"""
        self.generator._simulate_one_round(
            theta=0.8,
            effort=0.5,
            base_n_labor=100,
            round_idx=0
        )
        
        # 应该有100条记录（每个劳动力一条）
        assert len(self.generator.data_records) == 100
        
        # 验证数据结构
        record = self.generator.data_records[0]
        assert 'labor_T' in record
        assert 'labor_S' in record
        assert 'theta' in record
        assert 'effort' in record
        assert 'matched' in record
        
        # θ<1时匹配率应该较低
        matched_count = sum(r['matched'] for r in self.generator.data_records)
        match_rate = matched_count / 100
        assert match_rate < 1.0  # 不会全部匹配
    
    def test_simulate_theta_greater_than_one(self):
        """测试θ>1的情况（企业过剩）"""
        self.generator._simulate_one_round(
            theta=1.2,
            effort=0.5,
            base_n_labor=100,
            round_idx=0
        )
        
        assert len(self.generator.data_records) == 100
        
        # θ>1时匹配率应该较高
        matched_count = sum(r['matched'] for r in self.generator.data_records)
        match_rate = matched_count / 100
        assert match_rate > 0.8  # 大部分应该匹配成功
    
    def test_record_structure(self):
        """测试记录数据结构"""
        self.generator._simulate_one_round(
            theta=1.0,
            effort=0.5,
            base_n_labor=10,
            round_idx=0
        )
        
        record = self.generator.data_records[0]
        
        # 劳动力特征
        assert 'labor_T' in record
        assert 'labor_S' in record
        assert 'labor_D' in record
        assert 'labor_W' in record
        
        # 市场环境
        assert 'market_mean_T' in record
        assert 'market_std_S' in record
        assert 'theta' in record
        assert 'effort' in record
        
        # 匹配结果
        assert 'matched' in record
        assert record['matched'] in [0, 1]
        
        # 元数据
        assert record['round_idx'] == 0
        assert 'labor_idx' in record


class TestGenerateTrainingData:
    """测试训练数据生成"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.generator = ABMDataGenerator(seed=42)
    
    def test_small_scale_generation(self):
        """测试小规模数据生成"""
        df = self.generator.generate_training_data(
            theta_range=[0.8, 1.0, 1.2],
            effort_levels=[0.0, 0.5, 1.0],
            n_rounds_per_combination=2,
            base_n_labor=50,
            verbose=False
        )
        
        # 3个θ × 3个effort × 2轮 × 50个劳动力 = 900条记录
        expected_records = 3 * 3 * 2 * 50
        assert len(df) == expected_records
        
        # 验证列
        required_cols = [
            'labor_T', 'labor_S', 'labor_D', 'labor_W',
            'theta', 'effort', 'matched'
        ]
        for col in required_cols:
            assert col in df.columns
    
    def test_theta_range_coverage(self):
        """测试θ值覆盖范围"""
        df = self.generator.generate_training_data(
            theta_range=[0.7, 1.0, 1.3],
            effort_levels=[0.5],
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
        
        unique_theta = df['theta'].unique()
        assert len(unique_theta) == 3
        assert 0.7 in unique_theta
        assert 1.0 in unique_theta
        assert 1.3 in unique_theta
    
    def test_effort_range_coverage(self):
        """测试effort值覆盖范围"""
        df = self.generator.generate_training_data(
            theta_range=[1.0],
            effort_levels=[0.0, 0.5, 1.0],
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
        
        unique_effort = df['effort'].unique()
        assert len(unique_effort) == 3
        assert 0.0 in unique_effort
        assert 0.5 in unique_effort
        assert 1.0 in unique_effort
    
    def test_match_rate_by_theta(self):
        """测试不同θ下的匹配率"""
        df = self.generator.generate_training_data(
            theta_range=[0.7, 1.0, 1.3],
            effort_levels=[0.5],
            n_rounds_per_combination=3,
            base_n_labor=100,
            verbose=False
        )
        
        match_rate_by_theta = df.groupby('theta')['matched'].mean()
        
        # θ越大，匹配率应该越高（总体趋势）
        assert match_rate_by_theta[0.7] < match_rate_by_theta[1.3]
    
    def test_default_parameters(self):
        """测试默认参数"""
        df = self.generator.generate_training_data(
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
        
        # 默认7个θ × 6个effort × 1轮 × 50 = 2100条记录
        expected_records = 7 * 6 * 1 * 50
        assert len(df) == expected_records


class TestDataQuality:
    """测试数据质量"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.generator = ABMDataGenerator(seed=42)
        self.df = self.generator.generate_training_data(
            theta_range=[0.8, 1.0, 1.2],
            effort_levels=[0.0, 0.5, 1.0],
            n_rounds_per_combination=2,
            base_n_labor=50,
            verbose=False
        )
    
    def test_no_missing_values_in_required_cols(self):
        """测试必需列无缺失值"""
        required_cols = [
            'labor_T', 'labor_S', 'labor_D', 'labor_W',
            'theta', 'effort', 'matched'
        ]
        
        for col in required_cols:
            assert self.df[col].isnull().sum() == 0
    
    def test_matched_values_valid(self):
        """测试matched列取值有效"""
        assert self.df['matched'].isin([0, 1]).all()
    
    def test_theta_values_valid(self):
        """测试theta值有效"""
        assert (self.df['theta'] > 0).all()
        assert (self.df['theta'] <= 2.0).all()  # 合理范围
    
    def test_effort_values_valid(self):
        """测试effort值有效"""
        assert (self.df['effort'] >= 0).all()
        assert (self.df['effort'] <= 1.0).all()
    
    def test_labor_features_valid(self):
        """测试劳动力特征有效"""
        assert (self.df['labor_T'] >= 0).all()
        assert (self.df['labor_S'] >= 0).all()
        assert (self.df['labor_D'] >= 0).all()
        assert (self.df['labor_W'] >= 0).all()


class TestSummaryStatistics:
    """测试摘要统计"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.generator = ABMDataGenerator(seed=42)
        self.df = self.generator.generate_training_data(
            theta_range=[0.8, 1.0, 1.2],
            effort_levels=[0.5],
            n_rounds_per_combination=2,
            base_n_labor=50,
            verbose=False
        )
    
    def test_summary_statistics(self):
        """测试生成摘要统计"""
        summary = self.generator.generate_summary_statistics(self.df)
        
        assert 'n_records' in summary
        assert 'match_rate' in summary
        assert 'match_rate_by_theta' in summary
        assert 'match_rate_by_effort' in summary
        
        assert summary['n_records'] == len(self.df)
        assert 0 <= summary['match_rate'] <= 1
    
    def test_match_rate_by_theta_summary(self):
        """测试按θ分组的匹配率统计"""
        summary = self.generator.generate_summary_statistics(self.df)
        
        match_rate_by_theta = summary['match_rate_by_theta']
        
        assert 0.8 in match_rate_by_theta
        assert 1.0 in match_rate_by_theta
        assert 1.2 in match_rate_by_theta


class TestSaveData:
    """测试数据保存"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.generator = ABMDataGenerator(seed=42)
        self.df = self.generator.generate_training_data(
            theta_range=[1.0],
            effort_levels=[0.5],
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
    
    def test_save_csv(self):
        """测试保存CSV格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_data.csv'
            self.generator.save_data(self.df, str(save_path), format='csv')
            
            assert save_path.exists()
            
            # 读取并验证
            loaded_df = pd.read_csv(save_path)
            assert len(loaded_df) == len(self.df)
    
    def test_save_parquet(self):
        """测试保存Parquet格式"""
        pytest.importorskip("pyarrow", reason="需要安装pyarrow")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_data.parquet'
            self.generator.save_data(self.df, str(save_path), format='parquet')
            
            assert save_path.exists()
            
            # 读取并验证
            loaded_df = pd.read_parquet(save_path)
            assert len(loaded_df) == len(self.df)
    
    def test_save_pickle(self):
        """测试保存Pickle格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_data.pkl'
            self.generator.save_data(self.df, str(save_path), format='pickle')
            
            assert save_path.exists()
            
            # 读取并验证
            loaded_df = pd.read_pickle(save_path)
            assert len(loaded_df) == len(self.df)
    
    def test_unsupported_format(self):
        """测试不支持的格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_data.xyz'
            
            with pytest.raises(ValueError, match="不支持的格式"):
                self.generator.save_data(self.df, str(save_path), format='xyz')


class TestReproducibility:
    """测试可重复性"""
    
    def test_same_seed_same_results(self):
        """测试相同seed产生相同结果"""
        gen1 = ABMDataGenerator(seed=42)
        df1 = gen1.generate_training_data(
            theta_range=[1.0],
            effort_levels=[0.5],
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
        
        gen2 = ABMDataGenerator(seed=42)
        df2 = gen2.generate_training_data(
            theta_range=[1.0],
            effort_levels=[0.5],
            n_rounds_per_combination=1,
            base_n_labor=50,
            verbose=False
        )
        
        # 比较关键列
        pd.testing.assert_series_equal(df1['labor_T'], df2['labor_T'])
        pd.testing.assert_series_equal(df1['matched'], df2['matched'])

