"""
测试匹配引擎
"""

import pytest
import numpy as np
import pandas as pd
from src.modules.matching.matching_engine import MatchingEngine


class TestMatchingEngineInit:
    """测试匹配引擎初始化"""
    
    def test_default_init(self):
        """测试使用默认配置初始化"""
        engine = MatchingEngine()
        
        assert engine is not None
        assert 'gamma_0' in engine.labor_pref_params
        assert 'beta_0' in engine.enterprise_pref_params
    
    def test_custom_config_init(self):
        """测试使用自定义配置初始化"""
        config = {
            'preference': {
                'labor': {'gamma_0': 2.0, 'gamma_1': 0.02},
                'enterprise': {'beta_0': 1.0, 'beta_1': 0.3}
            }
        }
        
        engine = MatchingEngine(config=config)
        
        assert engine.labor_pref_params['gamma_0'] == 2.0
        assert engine.labor_pref_params['gamma_1'] == 0.02


class TestMatchingEngineMatch:
    """测试单轮匹配"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.engine = MatchingEngine()
        
        # 创建测试数据
        np.random.seed(42)
        self.labor = pd.DataFrame({
            'T': np.random.rand(10) * 60,
            'S': np.random.rand(10),
            'D': np.random.rand(10),
            'W': np.random.rand(10) * 5000
        })
        
        self.enterprise = pd.DataFrame({
            'T': np.random.rand(5) * 60,
            'S': np.random.rand(5),
            'D': np.random.rand(5),
            'W': np.random.rand(5) * 6000
        })
    
    def test_basic_match(self):
        """测试基本匹配功能"""
        result = self.engine.match(self.labor, self.enterprise)
        
        assert result is not None
        assert len(result.matching) == 10
        assert result.statistics['n_labor'] == 10
        assert result.statistics['n_enterprise'] == 5
    
    def test_match_stability(self):
        """测试匹配稳定性"""
        result = self.engine.match(self.labor, self.enterprise, verify_stability=True)
        
        assert result.is_stable, "GS算法应该总是产生稳定匹配"
        assert len(result.unstable_pairs) == 0
    
    def test_match_statistics(self):
        """测试匹配统计"""
        result = self.engine.match(self.labor, self.enterprise)
        
        assert 'match_rate' in result.statistics
        assert 'unemployment_rate' in result.statistics
        assert 0 <= result.statistics['match_rate'] <= 1
        assert 0 <= result.statistics['unemployment_rate'] <= 1
    
    def test_match_quality(self):
        """测试匹配质量计算"""
        result = self.engine.match(self.labor, self.enterprise)
        quality = result.compute_match_quality()
        
        assert 'avg_labor_satisfaction' in quality
        assert 'avg_enterprise_satisfaction' in quality
        assert 0 <= quality['avg_labor_satisfaction'] <= 1
        assert 0 <= quality['avg_enterprise_satisfaction'] <= 1


class TestMatchingEngineValidation:
    """测试数据验证"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.engine = MatchingEngine()
    
    def test_missing_columns(self):
        """测试缺少必需列"""
        labor = pd.DataFrame({'T': [40], 'S': [0.5]})  # 缺少D, W
        enterprise = pd.DataFrame({'T': [40], 'S': [0.5], 'D': [0.5], 'W': [3000]})
        
        with pytest.raises(ValueError, match="缺少列"):
            self.engine.match(labor, enterprise)
    
    def test_null_values(self):
        """测试包含空值"""
        labor = pd.DataFrame({
            'T': [40, np.nan],
            'S': [0.5, 0.6],
            'D': [0.5, 0.6],
            'W': [3000, 4000]
        })
        enterprise = pd.DataFrame({
            'T': [40],
            'S': [0.5],
            'D': [0.5],
            'W': [3000]
        })
        
        with pytest.raises(ValueError, match="空值"):
            self.engine.match(labor, enterprise)
    
    def test_non_numeric_columns(self):
        """测试非数值列"""
        labor = pd.DataFrame({
            'T': ['40', '50'],  # 字符串而非数值
            'S': [0.5, 0.6],
            'D': [0.5, 0.6],
            'W': [3000, 4000]
        })
        enterprise = pd.DataFrame({
            'T': [40],
            'S': [0.5],
            'D': [0.5],
            'W': [3000]
        })
        
        with pytest.raises(ValueError, match="数值类型"):
            self.engine.match(labor, enterprise)


class TestBatchMatch:
    """测试批量匹配"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.engine = MatchingEngine()
        
        # 创建多个场景
        np.random.seed(42)
        self.labor_list = [
            pd.DataFrame({
                'T': np.random.rand(10) * 60,
                'S': np.random.rand(10),
                'D': np.random.rand(10),
                'W': np.random.rand(10) * 5000
            })
            for _ in range(3)
        ]
        
        self.enterprise_list = [
            pd.DataFrame({
                'T': np.random.rand(5) * 60,
                'S': np.random.rand(5),
                'D': np.random.rand(5),
                'W': np.random.rand(5) * 6000
            })
            for _ in range(3)
        ]
    
    def test_batch_match(self):
        """测试批量匹配功能"""
        results = self.engine.batch_match(self.labor_list, self.enterprise_list)
        
        assert len(results) == 3
        assert all(r.statistics['n_labor'] == 10 for r in results)
        assert all(r.statistics['n_enterprise'] == 5 for r in results)
    
    def test_batch_statistics(self):
        """测试批量统计"""
        results = self.engine.batch_match(self.labor_list, self.enterprise_list)
        summary = self.engine.compute_batch_statistics(results)
        
        assert summary['n_scenarios'] == 3
        assert 'avg_match_rate' in summary
        assert 'std_match_rate' in summary
        assert 'stability_rate' in summary
        assert summary['stability_rate'] == 1.0  # GS总是稳定
    
    def test_batch_length_mismatch(self):
        """测试列表长度不匹配"""
        with pytest.raises(ValueError, match="长度必须相同"):
            self.engine.batch_match(self.labor_list, self.enterprise_list[:2])


class TestParameterUpdate:
    """测试参数更新"""
    
    def test_update_labor_params(self):
        """测试更新劳动力偏好参数"""
        engine = MatchingEngine()
        original_gamma0 = engine.labor_pref_params['gamma_0']
        
        engine.update_preference_params(labor_params={'gamma_0': 2.0})
        
        assert engine.labor_pref_params['gamma_0'] == 2.0
        assert engine.labor_pref_params['gamma_0'] != original_gamma0
    
    def test_update_enterprise_params(self):
        """测试更新企业偏好参数"""
        engine = MatchingEngine()
        original_beta0 = engine.enterprise_pref_params['beta_0']
        
        engine.update_preference_params(enterprise_params={'beta_0': 1.0})
        
        assert engine.enterprise_pref_params['beta_0'] == 1.0
    
    def test_partial_update(self):
        """测试部分参数更新"""
        engine = MatchingEngine()
        
        engine.update_preference_params(
            labor_params={'gamma_0': 2.0},
            enterprise_params={'beta_4': -0.002}
        )
        
        assert engine.labor_pref_params['gamma_0'] == 2.0
        assert engine.enterprise_pref_params['beta_4'] == -0.002
        # 其他参数不变
        assert engine.labor_pref_params['gamma_1'] == 0.01


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建引擎
        engine = MatchingEngine()
        
        # 准备数据
        np.random.seed(42)
        labor = pd.DataFrame({
            'T': np.random.rand(100) * 60,
            'S': np.random.rand(100),
            'D': np.random.rand(100),
            'W': np.random.rand(100) * 5000
        })
        
        enterprise = pd.DataFrame({
            'T': np.random.rand(50) * 60,
            'S': np.random.rand(50),
            'D': np.random.rand(50),
            'W': np.random.rand(50) * 6000
        })
        
        # 执行匹配
        result = engine.match(labor, enterprise)
        
        # 验证结果
        assert result.is_stable
        assert result.statistics['match_rate'] > 0
        
        # 获取匹配详情
        matched_pairs = result.get_matched_pairs()
        assert len(matched_pairs) == result.statistics['n_matched']
        
        # 获取未匹配劳动力
        unmatched = result.get_unmatched_labor()
        assert len(unmatched) == result.statistics['n_unmatched']
    
    def test_different_theta_scenarios(self):
        """测试不同θ值场景"""
        engine = MatchingEngine()
        np.random.seed(42)
        
        # 场景1：θ<1（劳动力过剩）
        labor1 = pd.DataFrame({
            'T': np.random.rand(100) * 60,
            'S': np.random.rand(100),
            'D': np.random.rand(100),
            'W': np.random.rand(100) * 5000
        })
        enterprise1 = pd.DataFrame({
            'T': np.random.rand(50) * 60,
            'S': np.random.rand(50),
            'D': np.random.rand(50),
            'W': np.random.rand(50) * 6000
        })
        
        # 场景2：θ>1（企业过剩）
        labor2 = pd.DataFrame({
            'T': np.random.rand(50) * 60,
            'S': np.random.rand(50),
            'D': np.random.rand(50),
            'W': np.random.rand(50) * 5000
        })
        enterprise2 = pd.DataFrame({
            'T': np.random.rand(100) * 60,
            'S': np.random.rand(100),
            'D': np.random.rand(100),
            'W': np.random.rand(100) * 6000
        })
        
        result1 = engine.match(labor1, enterprise1)
        result2 = engine.match(labor2, enterprise2)
        
        # 场景1失业率应该更高
        assert result1.statistics['unemployment_rate'] > result2.statistics['unemployment_rate']
        # 场景2匹配率应该更高
        assert result2.statistics['match_rate'] > result1.statistics['match_rate']

