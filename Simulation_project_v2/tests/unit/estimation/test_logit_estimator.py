"""
测试Logit估计器
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.modules.estimation.logit_estimator import LogitEstimator


class TestLogitEstimatorInit:
    """测试初始化"""
    
    def test_init(self):
        """测试基本初始化"""
        estimator = LogitEstimator()
        
        assert estimator is not None
        assert estimator.model is None
        assert estimator.result is None
        assert estimator.params is None


class TestLogitEstimatorFit:
    """测试模型拟合"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.estimator = LogitEstimator()
        
        # 创建测试数据
        np.random.seed(42)
        n = 1000
        
        self.data = pd.DataFrame({
            # 劳动力特征
            'labor_T': np.random.rand(n) * 60,
            'labor_S': np.random.rand(n),
            'labor_D': np.random.rand(n),
            'labor_W': np.random.rand(n) * 5000,
            
            # 劳动力与市场差距
            'labor_market_gap_T': np.random.randn(n) * 10,
            'labor_market_gap_S': np.random.randn(n) * 0.2,
            'labor_market_gap_D': np.random.randn(n) * 0.2,
            'labor_market_gap_W': np.random.randn(n) * 1000,
            
            # 环境参数
            'effort': np.random.rand(n),
            'theta': np.random.uniform(0.7, 1.3, n),
            
            # 匹配结果（模拟生成）
            'matched': np.random.randint(0, 2, n)
        })
    
    def test_fit_basic(self):
        """测试基本拟合功能"""
        summary = self.estimator.fit(self.data)
        
        assert self.estimator.model is not None
        assert self.estimator.result is not None
        assert self.estimator.params is not None
        assert 'params' in summary
        assert 'pseudo_r2' in summary
    
    def test_fit_with_control_vars(self):
        """测试包含控制变量的拟合"""
        summary = self.estimator.fit(
            self.data,
            include_control_vars=True
        )
        
        assert len(self.estimator.feature_names) == 11  # const + 4(x) + 4(sigma) + 1(a) + 1(theta)
        assert 'delta_sigma_labor_market_gap_T' in self.estimator.feature_names
    
    def test_fit_without_control_vars(self):
        """测试不包含控制变量的拟合"""
        summary = self.estimator.fit(
            self.data,
            include_control_vars=False
        )
        
        assert len(self.estimator.feature_names) == 7  # 1+4+1+1 (const + x + a + theta)
    
    def test_fit_without_intercept(self):
        """测试不添加截距项的拟合"""
        summary = self.estimator.fit(
            self.data,
            add_intercept=False
        )
        
        assert 'const' not in self.estimator.feature_names


class TestLogitEstimatorPredict:
    """测试预测功能"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.estimator = LogitEstimator()
        
        # 创建并拟合测试数据
        np.random.seed(42)
        n = 500
        
        self.train_data = pd.DataFrame({
            'labor_T': np.random.rand(n) * 60,
            'labor_S': np.random.rand(n),
            'labor_D': np.random.rand(n),
            'labor_W': np.random.rand(n) * 5000,
            'labor_market_gap_T': np.random.randn(n) * 10,
            'labor_market_gap_S': np.random.randn(n) * 0.2,
            'labor_market_gap_D': np.random.randn(n) * 0.2,
            'labor_market_gap_W': np.random.randn(n) * 1000,
            'effort': np.random.rand(n),
            'theta': np.random.uniform(0.7, 1.3, n),
            'matched': np.random.randint(0, 2, n)
        })
        
        self.estimator.fit(self.train_data)
        
        # 创建测试数据
        self.test_data = self.train_data.copy().iloc[:100]
    
    def test_predict(self):
        """测试预测功能"""
        proba = self.estimator.predict(self.test_data)
        
        assert len(proba) == len(self.test_data)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_predict_without_fit(self):
        """测试未拟合时预测抛出错误"""
        estimator_new = LogitEstimator()
        
        with pytest.raises(RuntimeError, match="模型尚未拟合"):
            estimator_new.predict(self.test_data)


class TestLogitEstimatorSaveLoad:
    """测试参数保存和加载"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.estimator = LogitEstimator()
        
        # 创建并拟合测试数据
        np.random.seed(42)
        n = 500
        
        self.data = pd.DataFrame({
            'labor_T': np.random.rand(n) * 60,
            'labor_S': np.random.rand(n),
            'labor_D': np.random.rand(n),
            'labor_W': np.random.rand(n) * 5000,
            'labor_market_gap_T': np.random.randn(n) * 10,
            'labor_market_gap_S': np.random.randn(n) * 0.2,
            'labor_market_gap_D': np.random.randn(n) * 0.2,
            'labor_market_gap_W': np.random.randn(n) * 1000,
            'effort': np.random.rand(n),
            'theta': np.random.uniform(0.7, 1.3, n),
            'matched': np.random.randint(0, 2, n)
        })
        
        self.estimator.fit(self.data)
    
    def test_save_params(self):
        """测试保存参数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'params.json'
            self.estimator.save_params(str(save_path))
            
            assert save_path.exists()
            
            # 验证文件内容
            import json
            with open(save_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'params' in loaded_data
            assert 'feature_names' in loaded_data
            assert 'summary' in loaded_data
    
    def test_load_params(self):
        """测试加载参数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'params.json'
            
            # 保存参数
            self.estimator.save_params(str(save_path))
            
            # 创建新估计器并加载参数
            estimator_new = LogitEstimator()
            estimator_new.load_params(str(save_path))
            
            # 验证参数一致
            assert estimator_new.params is not None
            assert len(estimator_new.params) == len(self.estimator.params)


class TestLogitEstimatorEvaluate:
    """测试模型评估"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.estimator = LogitEstimator()
        
        # 创建训练和测试数据
        np.random.seed(42)
        n_train = 800
        n_test = 200
        
        # 训练数据
        self.train_data = pd.DataFrame({
            'labor_T': np.random.rand(n_train) * 60,
            'labor_S': np.random.rand(n_train),
            'labor_D': np.random.rand(n_train),
            'labor_W': np.random.rand(n_train) * 5000,
            'labor_market_gap_T': np.random.randn(n_train) * 10,
            'labor_market_gap_S': np.random.randn(n_train) * 0.2,
            'labor_market_gap_D': np.random.randn(n_train) * 0.2,
            'labor_market_gap_W': np.random.randn(n_train) * 1000,
            'effort': np.random.rand(n_train),
            'theta': np.random.uniform(0.7, 1.3, n_train),
            'matched': np.random.randint(0, 2, n_train)
        })
        
        # 测试数据
        self.test_data = pd.DataFrame({
            'labor_T': np.random.rand(n_test) * 60,
            'labor_S': np.random.rand(n_test),
            'labor_D': np.random.rand(n_test),
            'labor_W': np.random.rand(n_test) * 5000,
            'labor_market_gap_T': np.random.randn(n_test) * 10,
            'labor_market_gap_S': np.random.randn(n_test) * 0.2,
            'labor_market_gap_D': np.random.randn(n_test) * 0.2,
            'labor_market_gap_W': np.random.randn(n_test) * 1000,
            'effort': np.random.rand(n_test),
            'theta': np.random.uniform(0.7, 1.3, n_test),
            'matched': np.random.randint(0, 2, n_test)
        })
        
        self.estimator.fit(self.train_data)
    
    def test_evaluate(self):
        """测试评估功能"""
        metrics = self.estimator.evaluate(self.test_data)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # 检查指标范围
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            assert 0 <= metrics[key] <= 1


class TestLogitEstimatorPrintSummary:
    """测试打印摘要"""
    
    def test_print_summary(self, capsys):
        """测试打印摘要功能"""
        estimator = LogitEstimator()
        
        # 创建并拟合数据
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            'labor_T': np.random.rand(n) * 60,
            'labor_S': np.random.rand(n),
            'labor_D': np.random.rand(n),
            'labor_W': np.random.rand(n) * 5000,
            'labor_market_gap_T': np.random.randn(n) * 10,
            'labor_market_gap_S': np.random.randn(n) * 0.2,
            'labor_market_gap_D': np.random.randn(n) * 0.2,
            'labor_market_gap_W': np.random.randn(n) * 1000,
            'effort': np.random.rand(n),
            'theta': np.random.uniform(0.7, 1.3, n),
            'matched': np.random.randint(0, 2, n)
        })
        
        estimator.fit(data)
        estimator.print_summary()
        
        # 验证输出
        captured = capsys.readouterr()
        assert 'Logit回归估计结果' in captured.out
        assert 'pseudo' in captured.out.lower() or 'r-squared' in captured.out.lower()

