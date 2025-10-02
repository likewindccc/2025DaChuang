"""
测试匹配函数
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.modules.estimation.match_function import (
    MatchFunction,
    compute_match_probability_numba,
    compute_match_probability_batch_numba
)


class TestMatchFunctionInit:
    """测试初始化"""
    
    def test_init_without_params(self):
        """测试无参数初始化"""
        func = MatchFunction()
        
        assert func is not None
        assert func.params is None
        assert func.param_array is None
    
    def test_init_with_params(self):
        """测试带参数初始化"""
        params = {
            'const': 0.5,
            'delta_labor_T': 0.01,
            'delta_labor_S': 0.5,
            'delta_labor_D': 0.5,
            'delta_labor_W': 0.0001,
            'delta_sigma_labor_market_gap_T': 0.02,
            'delta_sigma_labor_market_gap_S': 0.3,
            'delta_sigma_labor_market_gap_D': 0.3,
            'delta_sigma_labor_market_gap_W': 0.0001,
            'delta_a': 0.8,
            'delta_theta': 1.5
        }
        
        func = MatchFunction(params=params)
        
        assert func.params is not None
        assert func.param_array is not None
        assert len(func.param_array) == 11


class TestMatchFunctionCompute:
    """测试匹配概率计算"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.params = {
            'const': 0.5,
            'delta_labor_T': 0.01,
            'delta_labor_S': 0.5,
            'delta_labor_D': 0.5,
            'delta_labor_W': 0.0001,
            'delta_sigma_labor_market_gap_T': 0.02,
            'delta_sigma_labor_market_gap_S': 0.3,
            'delta_sigma_labor_market_gap_D': 0.3,
            'delta_sigma_labor_market_gap_W': 0.0001,
            'delta_a': 0.8,
            'delta_theta': 1.5
        }
        
        self.func = MatchFunction(params=self.params)
    
    def test_compute_single_probability(self):
        """测试单个概率计算"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        a = 0.5
        theta = 1.0
        
        prob = self.func.compute_match_probability(x, sigma, a, theta)
        
        assert isinstance(prob, (float, np.floating))
        assert 0 <= prob <= 1
    
    def test_compute_batch_probability(self):
        """测试批量概率计算"""
        n = 100
        X = np.random.rand(n, 4) * np.array([60, 1, 1, 5000])
        Sigma = np.random.randn(n, 4) * np.array([10, 0.2, 0.2, 1000])
        a = np.random.rand(n)
        theta = np.random.uniform(0.7, 1.3, n)
        
        probs = self.func.compute_match_probability_batch(X, Sigma, a, theta)
        
        assert len(probs) == n
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_compute_without_params(self):
        """测试未加载参数时计算抛出错误"""
        func_empty = MatchFunction()
        
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        
        with pytest.raises(RuntimeError, match="参数尚未设置"):
            func_empty.compute_match_probability(x, sigma, 0.5, 1.0)


class TestMatchFunctionSaveLoad:
    """测试参数保存和加载"""
    
    def test_load_params(self):
        """测试加载参数"""
        # 创建临时参数文件
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = Path(tmpdir) / 'params.json'
            
            # 手动创建参数文件
            import json
            params_dict = {
                'params': {
                    'const': 0.5,
                    'delta_labor_T': 0.01,
                    'delta_labor_S': 0.5,
                    'delta_labor_D': 0.5,
                    'delta_labor_W': 0.0001,
                    'delta_sigma_labor_market_gap_T': 0.02,
                    'delta_sigma_labor_market_gap_S': 0.3,
                    'delta_sigma_labor_market_gap_D': 0.3,
                    'delta_sigma_labor_market_gap_W': 0.0001,
                    'delta_a': 0.8,
                    'delta_theta': 1.5
                },
                'feature_names': [
                    'const', 'delta_labor_T', 'delta_labor_S',
                    'delta_labor_D', 'delta_labor_W'
                ]
            }
            
            with open(params_path, 'w') as f:
                json.dump(params_dict, f)
            
            # 加载参数
            func = MatchFunction()
            func.load_params(str(params_path))
            
            assert func.params is not None
            assert func.param_array is not None


class TestMatchFunctionSample:
    """测试匹配结果抽样"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.params = {
            'const': 0.5,
            'delta_labor_T': 0.01,
            'delta_labor_S': 0.5,
            'delta_labor_D': 0.5,
            'delta_labor_W': 0.0001,
            'delta_sigma_labor_market_gap_T': 0.02,
            'delta_sigma_labor_market_gap_S': 0.3,
            'delta_sigma_labor_market_gap_D': 0.3,
            'delta_sigma_labor_market_gap_W': 0.0001,
            'delta_a': 0.8,
            'delta_theta': 1.5
        }
        
        self.func = MatchFunction(params=self.params)
    
    def test_sample_match_outcome(self):
        """测试抽样匹配结果"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        a = 0.5
        theta = 1.0
        
        outcome = self.func.sample_match_outcome(x, sigma, a, theta, random_seed=42)
        
        assert isinstance(outcome, (bool, np.bool_))
    
    def test_sample_reproducibility(self):
        """测试抽样可重复性"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        a = 0.5
        theta = 1.0
        
        outcome1 = self.func.sample_match_outcome(x, sigma, a, theta, random_seed=42)
        outcome2 = self.func.sample_match_outcome(x, sigma, a, theta, random_seed=42)
        
        assert outcome1 == outcome2
    
    def test_sample_distribution(self):
        """测试抽样分布接近理论概率"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        a = 0.5
        theta = 1.0
        
        # 理论概率
        expected_prob = self.func.compute_match_probability(x, sigma, a, theta)
        
        # 多次抽样
        n_samples = 10000
        np.random.seed(42)
        outcomes = [
            self.func.sample_match_outcome(x, sigma, a, theta)
            for _ in range(n_samples)
        ]
        
        # 实际概率
        actual_prob = np.mean(outcomes)
        
        # 验证接近（允许5%误差）
        assert abs(actual_prob - expected_prob) < 0.05


class TestNumbaFunctions:
    """测试Numba优化函数"""
    
    def test_compute_match_probability_numba(self):
        """测试Numba单个概率计算"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.array([5.0, 0.1, 0.05, 500.0])
        a = 0.5
        theta = 1.0
        
        params = np.array([
            0.5,  # delta_0
            0.01, 0.5, 0.5, 0.0001,  # delta_x
            0.02, 0.3, 0.3, 0.0001,  # delta_sigma
            0.8,  # delta_a
            1.5   # delta_theta
        ])
        
        prob = compute_match_probability_numba(x, sigma, a, theta, params)
        
        assert isinstance(prob, (float, np.floating))
        assert 0 <= prob <= 1
    
    def test_compute_match_probability_batch_numba(self):
        """测试Numba批量概率计算"""
        n = 100
        X = np.random.rand(n, 4) * np.array([60, 1, 1, 5000])
        Sigma = np.random.randn(n, 4) * np.array([10, 0.2, 0.2, 1000])
        a = np.random.rand(n)
        theta = np.random.uniform(0.7, 1.3, n)
        
        params = np.array([
            0.5,  # delta_0
            0.01, 0.5, 0.5, 0.0001,  # delta_x
            0.02, 0.3, 0.3, 0.0001,  # delta_sigma
            0.8,  # delta_a
            1.5   # delta_theta
        ])
        
        probs = compute_match_probability_batch_numba(X, Sigma, a, theta, params)
        
        assert len(probs) == n
        assert np.all((probs >= 0) & (probs <= 1))


class TestMatchFunctionBehavior:
    """测试匹配函数行为特性"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.params = {
            'const': 0.5,
            'delta_labor_T': -0.01,  # 工作时间越长，匹配概率越低
            'delta_labor_S': 0.5,     # 技能越高，匹配概率越高
            'delta_labor_D': 0.5,     # 数字素养越高，匹配概率越高
            'delta_labor_W': -0.0001, # 期望工资越高，匹配概率越低
            'delta_sigma_labor_market_gap_T': 0.0,
            'delta_sigma_labor_market_gap_S': 0.0,
            'delta_sigma_labor_market_gap_D': 0.0,
            'delta_sigma_labor_market_gap_W': 0.0,
            'delta_a': 0.8,           # 努力越高，匹配概率越高
            'delta_theta': 1.5        # 市场松紧度越大，匹配概率越高
        }
        
        self.func = MatchFunction(params=self.params)
    
    def test_effort_effect(self):
        """测试努力水平对匹配概率的影响"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.zeros(4)
        theta = 1.0
        
        prob_low_effort = self.func.compute_match_probability(x, sigma, a=0.2, theta=theta)
        prob_high_effort = self.func.compute_match_probability(x, sigma, a=0.8, theta=theta)
        
        # 努力越高，匹配概率应该越高
        assert prob_high_effort > prob_low_effort
    
    def test_theta_effect(self):
        """测试市场松紧度对匹配概率的影响"""
        x = np.array([40.0, 0.7, 0.6, 3000.0])
        sigma = np.zeros(4)
        a = 0.5
        
        prob_tight = self.func.compute_match_probability(x, sigma, a, theta=0.7)
        prob_loose = self.func.compute_match_probability(x, sigma, a, theta=1.3)
        
        # θ越大（岗位越多），匹配概率应该越高
        assert prob_loose > prob_tight

