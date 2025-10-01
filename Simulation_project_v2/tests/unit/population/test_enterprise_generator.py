#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EnterpriseGenerator单元测试

测试覆盖：
1. 初始化和配置
2. 双模式fit方法
3. generate方法
4. validate方法
5. set_params方法
6. 边界条件和异常处理

作者：AI Assistant
日期：2025-10-01
"""

import pytest
import pandas as pd
import numpy as np
import json

from src.modules.population import EnterpriseGenerator
from src.core import DataValidationError, ConfigurationError


class TestEnterpriseGeneratorInitialization:
    """测试EnterpriseGenerator初始化"""
    
    def test_init_with_default_config(self):
        """测试默认配置初始化"""
        gen = EnterpriseGenerator()
        
        assert gen.config is not None
        assert gen.is_fitted is False
        assert gen.mean is None
        assert gen.covariance is None
    
    def test_init_with_custom_config(self, enterprise_config):
        """测试自定义配置初始化"""
        gen = EnterpriseGenerator(enterprise_config)
        
        assert gen.config['seed'] == 43
        assert gen.config['default_mean'] == [45.0, 75.0, 65.0, 5500.0]
    
    def test_init_sets_random_seed(self):
        """测试随机种子设置"""
        gen1 = EnterpriseGenerator({'seed': 42})
        gen2 = EnterpriseGenerator({'seed': 42})
        
        assert gen1.config['seed'] == gen2.config['seed']


class TestEnterpriseGeneratorFitConfigMode:
    """测试配置驱动模式的fit方法"""
    
    def test_fit_without_data_uses_config(self, enterprise_config):
        """测试无数据时使用配置"""
        gen = EnterpriseGenerator(enterprise_config)
        gen.fit()
        
        assert gen.is_fitted is True
        assert gen.mean is not None
        assert gen.covariance is not None
        np.testing.assert_array_equal(gen.mean, [45.0, 75.0, 65.0, 5500.0])
    
    def test_fit_without_config_uses_defaults(self):
        """测试无配置时使用默认值"""
        gen = EnterpriseGenerator()
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen.fit()
            
            # 应该有警告提示使用默认值
            assert any("default_mean" in str(warning.message) for warning in w)
        
        assert gen.is_fitted is True
        assert gen.mean is not None
    
    def test_fit_config_mode_builds_covariance(self, enterprise_config):
        """测试配置模式构造协方差矩阵"""
        gen = EnterpriseGenerator(enterprise_config)
        gen.fit()
        
        assert gen.covariance.shape == (4, 4)
        
        # 检查对角元素（方差）
        expected_var = np.array(enterprise_config['default_std']) ** 2
        np.testing.assert_array_almost_equal(
            np.diag(gen.covariance),
            expected_var
        )
    
    def test_fit_config_mode_validates_params(self):
        """测试配置模式验证参数"""
        # 错误的mean长度
        config = {'default_mean': [45.0, 75.0]}  # 只有2个元素
        gen = EnterpriseGenerator(config)
        
        with pytest.raises(ConfigurationError):
            gen.fit()


class TestEnterpriseGeneratorFitLaborMode:
    """测试劳动力驱动模式的fit方法"""
    
    def test_fit_with_labor_data(self, sample_labor_data):
        """测试使用劳动力数据拟合"""
        gen = EnterpriseGenerator({'seed': 43})
        gen.fit(sample_labor_data)
        
        assert gen.is_fitted is True
        assert gen.mean is not None
        assert gen.data_stats is not None
    
    def test_fit_labor_mode_applies_multiplier(self, sample_labor_data):
        """测试劳动力模式应用调整系数"""
        labor_mean = sample_labor_data[['T', 'S', 'D', 'W']].mean().values
        
        multiplier = np.array([1.1, 1.05, 1.1, 1.2])
        gen = EnterpriseGenerator({
            'seed': 43,
            'labor_multiplier': multiplier
        })
        gen.fit(sample_labor_data)
        
        expected_mean = labor_mean * multiplier
        np.testing.assert_array_almost_equal(gen.mean, expected_mean)
    
    def test_fit_labor_mode_saves_stats(self, sample_labor_data):
        """测试劳动力模式保存统计信息"""
        gen = EnterpriseGenerator()
        gen.fit(sample_labor_data)
        
        assert 'labor_mean' in gen.data_stats
        assert 'labor_std' in gen.data_stats
        assert 'n_samples' in gen.data_stats
        assert gen.data_stats['n_samples'] == len(sample_labor_data)
    
    def test_fit_labor_mode_missing_columns(self):
        """测试劳动力数据缺少列时抛出异常"""
        incomplete_data = pd.DataFrame({
            'T': [40, 45, 50],
            'S': [20, 25, 30]
            # 缺少 D, W
        })
        
        gen = EnterpriseGenerator()
        
        with pytest.raises(DataValidationError):
            gen.fit(incomplete_data)


class TestEnterpriseGeneratorGenerate:
    """测试generate方法"""
    
    def test_generate_before_fit_raises_error(self):
        """测试未拟合就生成时抛出异常"""
        gen = EnterpriseGenerator()
        
        with pytest.raises(RuntimeError) as exc_info:
            gen.generate(100)
        
        assert "必须先调用fit" in str(exc_info.value)
    
    def test_generate_correct_number_of_enterprises(self, mock_fitted_enterprise_generator):
        """测试生成正确数量的企业"""
        gen = mock_fitted_enterprise_generator
        
        n = 100
        enterprises = gen.generate(n)
        
        assert len(enterprises) == n
        assert enterprises['agent_id'].nunique() == n
    
    def test_generate_correct_columns(self, mock_fitted_enterprise_generator):
        """测试生成的DataFrame包含正确的列"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(50)
        
        expected_cols = ['agent_id', 'agent_type', 'T', 'S', 'D', 'W']
        
        assert list(enterprises.columns) == expected_cols
    
    def test_generate_correct_agent_type(self, mock_fitted_enterprise_generator):
        """测试agent_type正确"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(50)
        
        assert all(enterprises['agent_type'] == 'enterprise')
    
    def test_generate_agent_id_starts_from_1001(self, mock_fitted_enterprise_generator):
        """测试agent_id从1001开始（避免与劳动力冲突）"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(50)
        
        assert enterprises['agent_id'].iloc[0] == 1001
        assert enterprises['agent_id'].iloc[-1] == 1050
    
    def test_generate_custom_start_id(self):
        """测试自定义起始ID"""
        config = {
            'seed': 43,
            'default_mean': [45.0, 75.0, 65.0, 5500.0],
            'default_std': [11.0, 15.0, 15.0, 1100.0],
            'start_id': 2000
        }
        
        gen = EnterpriseGenerator(config)
        gen.fit()
        enterprises = gen.generate(10)
        
        assert enterprises['agent_id'].iloc[0] == 2000
    
    def test_generate_no_negative_values(self, mock_fitted_enterprise_generator):
        """测试生成的值非负"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(500)
        
        # 所有变量应该非负
        for col in ['T', 'S', 'D', 'W']:
            assert enterprises[col].min() >= 0, f"{col}存在负值"
    
    def test_generate_reproducibility_with_seed(self, enterprise_config):
        """测试设置seed后的可重复性"""
        gen1 = EnterpriseGenerator(enterprise_config)
        gen1.fit()
        ent1 = gen1.generate(100)
        
        gen2 = EnterpriseGenerator(enterprise_config)
        gen2.fit()
        ent2 = gen2.generate(100)
        
        pd.testing.assert_frame_equal(ent1, ent2)


class TestEnterpriseGeneratorValidate:
    """测试validate方法"""
    
    def test_validate_before_fit_raises_error(self):
        """测试未拟合就验证时抛出异常"""
        gen = EnterpriseGenerator()
        
        with pytest.raises(RuntimeError):
            gen.validate(pd.DataFrame())
    
    def test_validate_with_generated_data(self, mock_fitted_enterprise_generator):
        """测试验证生成的数据"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(500)
        
        result = gen.validate(enterprises)
        
        assert isinstance(result, bool)
    
    def test_validate_checks_mean(self, mock_fitted_enterprise_generator):
        """测试验证检查均值"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(800)
        
        generated_mean = enterprises[['T', 'S', 'D', 'W']].mean().values
        
        # 均值偏差应该很小
        mean_diff = np.abs(generated_mean - gen.mean)
        relative_error = mean_diff / gen.mean
        
        assert np.all(relative_error < 0.15), "均值偏差过大"
    
    def test_validate_checks_std(self, mock_fitted_enterprise_generator):
        """测试验证检查标准差"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(800)
        
        generated_std = enterprises[['T', 'S', 'D', 'W']].std().values
        expected_std = np.sqrt(np.diag(gen.covariance))
        
        std_diff = np.abs(generated_std - expected_std)
        relative_error = std_diff / expected_std
        
        assert np.all(relative_error < 0.20), "标准差偏差过大"


class TestEnterpriseGeneratorSetParams:
    """测试set_params方法（用于校准）"""
    
    def test_set_params_updates_mean_and_cov(self, mock_fitted_enterprise_generator):
        """测试set_params更新参数"""
        gen = mock_fitted_enterprise_generator
        
        new_mean = np.array([50.0, 80.0, 70.0, 6000.0])
        new_cov = np.diag([13.0, 17.0, 17.0, 1300.0]) ** 2
        
        gen.set_params(new_mean, new_cov)
        
        np.testing.assert_array_equal(gen.mean, new_mean)
        np.testing.assert_array_equal(gen.covariance, new_cov)
    
    def test_set_params_validates_dimensions(self, mock_fitted_enterprise_generator):
        """测试set_params验证维度"""
        gen = mock_fitted_enterprise_generator
        
        # 错误的均值维度
        wrong_mean = np.array([50.0, 80.0])  # 只有2个元素
        new_cov = np.eye(4)
        
        with pytest.raises(ValueError) as exc_info:
            gen.set_params(wrong_mean, new_cov)
        
        assert "均值向量维度错误" in str(exc_info.value)
        
        # 错误的协方差维度
        correct_mean = np.array([50.0, 80.0, 70.0, 6000.0])
        wrong_cov = np.eye(3)  # 只有3x3
        
        with pytest.raises(ValueError) as exc_info:
            gen.set_params(correct_mean, wrong_cov)
        
        assert "协方差矩阵维度错误" in str(exc_info.value)
    
    def test_set_params_handles_non_positive_definite(self, mock_fitted_enterprise_generator):
        """测试set_params处理非正定矩阵"""
        gen = mock_fitted_enterprise_generator
        
        new_mean = np.array([50.0, 80.0, 70.0, 6000.0])
        
        # 创建一个非正定矩阵
        non_pd_cov = np.array([
            [100,  80,  60,  50],
            [ 80, 100,  70,  60],
            [ 60,  70, 100,  80],
            [ 50,  60,  80, 100]
        ])
        
        # 应该自动修正
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen.set_params(new_mean, non_pd_cov)
            
            # 可能有警告
            if len(w) > 0:
                assert any("非正定" in str(warning.message) for warning in w)
        
        # 参数应该被更新（已修正）
        assert gen.mean is not None
    
    def test_set_params_enables_generation(self, mock_fitted_enterprise_generator):
        """测试set_params后可以直接生成"""
        gen = mock_fitted_enterprise_generator
        
        new_mean = np.array([50.0, 80.0, 70.0, 6000.0])
        new_cov = np.diag([13.0, 17.0, 17.0, 1300.0]) ** 2
        
        gen.set_params(new_mean, new_cov)
        
        # 应该可以直接生成
        enterprises = gen.generate(100)
        
        assert len(enterprises) == 100


class TestEnterpriseGeneratorEdgeCases:
    """测试边界条件"""
    
    def test_generate_single_enterprise(self, mock_fitted_enterprise_generator):
        """测试生成单个企业"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(1)
        
        assert len(enterprises) == 1
        assert enterprises.loc[0, 'agent_id'] == 1001
    
    def test_generate_large_number(self, mock_fitted_enterprise_generator):
        """测试生成大量企业"""
        gen = mock_fitted_enterprise_generator
        enterprises = gen.generate(5000)
        
        assert len(enterprises) == 5000
        assert enterprises['agent_id'].is_unique
    
    def test_fit_with_extreme_multipliers(self, sample_labor_data):
        """测试极端调整系数"""
        # 非常高的调整系数
        gen = EnterpriseGenerator({
            'labor_multiplier': np.array([2.0, 2.0, 2.0, 3.0])
        })
        
        gen.fit(sample_labor_data)
        
        # 应该成功拟合
        assert gen.is_fitted is True
        
        # 企业均值应该是劳动力的2-3倍
        labor_mean = sample_labor_data[['T', 'S', 'D', 'W']].mean().values
        enterprise_mean = gen.mean
        
        ratio = enterprise_mean / labor_mean
        np.testing.assert_array_almost_equal(ratio, [2.0, 2.0, 2.0, 3.0])


class TestEnterpriseGeneratorCovariance:
    """测试协方差矩阵处理"""
    
    def test_covariance_with_custom_correlation(self):
        """测试使用自定义相关系数矩阵"""
        corr_matrix = [
            [1.0, 0.3, 0.2, 0.4],
            [0.3, 1.0, 0.5, 0.6],
            [0.2, 0.5, 1.0, 0.3],
            [0.4, 0.6, 0.3, 1.0]
        ]
        
        config = {
            'seed': 43,
            'default_mean': [45.0, 75.0, 65.0, 5500.0],
            'default_std': [11.0, 15.0, 15.0, 1100.0],
            'correlation': corr_matrix
        }
        
        gen = EnterpriseGenerator(config)
        gen.fit()
        
        # 协方差矩阵应该不是对角矩阵
        assert not np.allclose(gen.covariance, np.diag(np.diag(gen.covariance)))
        
        # 提取相关系数矩阵
        extracted_corr = gen._get_correlation_matrix()
        
        np.testing.assert_array_almost_equal(extracted_corr, corr_matrix, decimal=6)
    
    def test_covariance_is_symmetric(self, mock_fitted_enterprise_generator):
        """测试协方差矩阵对称性"""
        gen = mock_fitted_enterprise_generator
        
        assert np.allclose(gen.covariance, gen.covariance.T)
    
    def test_covariance_is_positive_definite(self, mock_fitted_enterprise_generator):
        """测试协方差矩阵正定性"""
        gen = mock_fitted_enterprise_generator
        
        # 正定矩阵的所有特征值应该>0
        eigenvalues = np.linalg.eigvalsh(gen.covariance)
        
        assert np.all(eigenvalues > 0), f"存在非正特征值: {eigenvalues}"


class TestEnterpriseGeneratorIntegration:
    """集成测试"""
    
    def test_full_workflow_config_mode(self, enterprise_config):
        """测试完整工作流程（配置模式）"""
        # Step 1: 创建
        gen = EnterpriseGenerator(enterprise_config)
        
        # Step 2: 拟合
        gen.fit()
        assert gen.is_fitted is True
        
        # Step 3: 生成
        enterprises = gen.generate(800)
        assert len(enterprises) == 800
        
        # Step 4: 验证
        is_valid = gen.validate(enterprises)
        assert isinstance(is_valid, bool)
    
    def test_full_workflow_labor_mode(self, sample_labor_data):
        """测试完整工作流程（劳动力模式）"""
        # Step 1: 创建
        gen = EnterpriseGenerator({'seed': 43})
        
        # Step 2: 拟合（基于劳动力数据）
        gen.fit(sample_labor_data)
        assert gen.is_fitted is True
        
        # Step 3: 生成
        enterprises = gen.generate(800)
        assert len(enterprises) == 800
        
        # Step 4: 验证
        is_valid = gen.validate(enterprises)
        assert isinstance(is_valid, bool)
    
    def test_calibration_workflow(self, enterprise_config):
        """测试校准工作流程"""
        # Step 1: 初始化
        gen = EnterpriseGenerator(enterprise_config)
        gen.fit()
        
        # Step 2: 初始生成
        ent_v1 = gen.generate(500)
        mean_v1 = ent_v1[['T', 'S', 'D', 'W']].mean().values
        
        # Step 3: 模拟校准（更新参数）
        new_mean = mean_v1 * 1.1  # 提升10%
        new_std = np.sqrt(np.diag(gen.covariance)) * 1.05
        new_cov = np.diag(new_std ** 2)
        
        gen.set_params(new_mean, new_cov)
        
        # Step 4: 校准后生成
        ent_v2 = gen.generate(500)
        mean_v2 = ent_v2[['T', 'S', 'D', 'W']].mean().values
        
        # Step 5: 验证参数更新效果
        assert np.all(mean_v2 > mean_v1), "校准后均值应该提升"


class TestEnterpriseGeneratorWithRealData:
    """使用真实数据的测试"""
    
    def test_fit_with_real_labor_data(self, real_cleaned_data):
        """测试使用真实劳动力数据拟合"""
        if real_cleaned_data is None:
            pytest.skip("真实数据不存在")
        
        gen = EnterpriseGenerator({'seed': 43})
        gen.fit(real_cleaned_data)
        
        assert gen.is_fitted is True
        
        # 生成并验证
        enterprises = gen.generate(800)
        assert len(enterprises) == 800


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

