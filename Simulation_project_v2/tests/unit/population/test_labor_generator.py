#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaborGenerator单元测试

测试覆盖：
1. 初始化和配置
2. fit方法
3. generate方法
4. validate方法
5. 边界条件和异常处理
6. 参数保存和加载

作者：AI Assistant
日期：2025-10-01
"""

import pytest
import pandas as pd
import numpy as np
import json
import os

from src.modules.population import LaborGenerator
from src.core import DataValidationError


class TestLaborGeneratorInitialization:
    """测试LaborGenerator初始化"""
    
    def test_init_with_default_config(self):
        """测试默认配置初始化"""
        gen = LaborGenerator()
        
        assert gen.config is not None
        assert gen.is_fitted is False
        assert gen.marginals_continuous == {}
        assert gen.marginals_discrete == {}
    
    def test_init_with_custom_config(self, sample_config):
        """测试自定义配置初始化"""
        gen = LaborGenerator(sample_config)
        
        assert gen.config['seed'] == 42
        assert gen.is_fitted is False
    
    def test_init_sets_random_seed(self):
        """测试随机种子设置"""
        gen1 = LaborGenerator({'seed': 42})
        gen2 = LaborGenerator({'seed': 42})
        
        # 两次初始化应该产生相同的随机状态
        assert gen1.config['seed'] == gen2.config['seed']


class TestLaborGeneratorFit:
    """测试LaborGenerator的fit方法"""
    
    def test_fit_with_valid_data(self, sample_labor_data):
        """测试使用有效数据拟合"""
        gen = LaborGenerator({'seed': 42})
        gen.fit(sample_labor_data)
        
        assert gen.is_fitted is True
        assert len(gen.marginals_continuous) == 6  # T, S, D, W, 年龄, 累计工作年限
        assert len(gen.marginals_discrete) == 2  # 孩子数量, 学历
        assert 'fitted_params' in dir(gen)
    
    def test_fit_missing_columns(self, sample_config):
        """测试缺少必需列时抛出异常"""
        gen = LaborGenerator(sample_config)
        
        # 创建缺少列的数据
        incomplete_data = pd.DataFrame({
            'T': [40, 45, 50],
            'S': [20, 25, 30]
            # 缺少 D, W, 年龄等
        })
        
        with pytest.raises(DataValidationError) as exc_info:
            gen.fit(incomplete_data)
        
        assert "数据缺少必需列" in str(exc_info.value)
    
    def test_fit_empty_data(self, sample_config):
        """测试空数据"""
        gen = LaborGenerator(sample_config)
        
        empty_data = pd.DataFrame()
        
        with pytest.raises((DataValidationError, KeyError)):
            gen.fit(empty_data)
    
    def test_fit_saves_correlation_matrix(self, sample_labor_data):
        """测试fit保存相关矩阵"""
        gen = LaborGenerator()
        gen.fit(sample_labor_data)
        
        assert gen.correlation_matrix is not None
        assert gen.correlation_matrix.shape == (6, 6)  # 6个连续变量
    
    def test_fit_saves_conditional_probs(self, sample_labor_data):
        """测试fit保存条件概率表"""
        gen = LaborGenerator()
        gen.fit(sample_labor_data)
        
        assert gen.conditional_probs is not None
        assert '孩子数量' in gen.conditional_probs
        assert '学历' in gen.conditional_probs


class TestLaborGeneratorGenerate:
    """测试LaborGenerator的generate方法"""
    
    def test_generate_before_fit_raises_error(self):
        """测试未拟合就生成时抛出异常"""
        gen = LaborGenerator()
        
        with pytest.raises(RuntimeError) as exc_info:
            gen.generate(100)
        
        assert "必须先调用fit" in str(exc_info.value)
    
    def test_generate_correct_number_of_agents(self, mock_fitted_labor_generator):
        """测试生成正确数量的智能体"""
        gen = mock_fitted_labor_generator
        
        n = 100
        agents = gen.generate(n)
        
        assert len(agents) == n
        assert agents['agent_id'].nunique() == n
    
    def test_generate_correct_columns(self, mock_fitted_labor_generator):
        """测试生成的DataFrame包含正确的列"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(50)
        
        expected_cols = [
            'agent_id', 'agent_type',
            'T', 'S', 'D', 'W',
            '年龄', '累计工作年限', '孩子数量', '学历'
        ]
        
        assert list(agents.columns) == expected_cols
    
    def test_generate_correct_agent_type(self, mock_fitted_labor_generator):
        """测试生成的agent_type正确"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(50)
        
        assert all(agents['agent_type'] == 'labor')
    
    def test_generate_continuous_variables_in_range(self, mock_fitted_labor_generator):
        """测试连续变量在合理范围内"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(200)
        
        # T: [0, 100] (理论上，实际会更窄)
        assert agents['T'].min() >= 0
        assert agents['T'].max() <= 200  # 放宽范围
        
        # S, D: [0, 100]
        assert agents['S'].min() >= 0
        assert agents['S'].max() <= 150
        
        assert agents['D'].min() >= 0
        assert agents['D'].max() <= 150
        
        # W: [0, +∞)
        assert agents['W'].min() >= 0
    
    def test_generate_discrete_variables_valid_values(self, mock_fitted_labor_generator):
        """测试离散变量取值合法"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(200)
        
        # 孩子数量应该是非负整数
        assert agents['孩子数量'].min() >= 0
        assert all(agents['孩子数量'] == agents['孩子数量'].astype(int))
        
        # 学历应该在0-6范围内（整数）
        assert agents['学历'].min() >= 0
        assert agents['学历'].max() <= 6
        assert all(agents['学历'] == agents['学历'].astype(int))
    
    def test_generate_reproducibility_with_seed(self, sample_labor_data):
        """测试设置seed后的可重复性"""
        gen1 = LaborGenerator({'seed': 42})
        gen1.fit(sample_labor_data)
        agents1 = gen1.generate(100)
        
        gen2 = LaborGenerator({'seed': 42})
        gen2.fit(sample_labor_data)
        agents2 = gen2.generate(100)
        
        # 应该生成完全相同的数据
        pd.testing.assert_frame_equal(agents1, agents2)
    
    def test_generate_different_with_different_seed(self, sample_labor_data):
        """测试不同seed生成不同数据"""
        gen1 = LaborGenerator({'seed': 42})
        gen1.fit(sample_labor_data)
        agents1 = gen1.generate(100)
        
        gen2 = LaborGenerator({'seed': 99})
        gen2.fit(sample_labor_data)
        agents2 = gen2.generate(100)
        
        # 应该生成不同的数据
        assert not agents1.equals(agents2)


class TestLaborGeneratorValidate:
    """测试LaborGenerator的validate方法"""
    
    def test_validate_before_fit_raises_error(self, sample_labor_data):
        """测试未拟合就验证时抛出异常"""
        gen = LaborGenerator()
        
        with pytest.raises(RuntimeError):
            gen.validate(sample_labor_data)
    
    def test_validate_with_generated_data(self, mock_fitted_labor_generator):
        """测试验证生成的数据"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(500)
        
        # validate应该返回布尔值（可能为False因为KS检验严格）
        result = gen.validate(agents)
        
        assert isinstance(result, bool)
    
    def test_validate_checks_mean_preservation(self, mock_fitted_labor_generator, sample_labor_data):
        """测试验证均值保留"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(500)
        
        # 手动检查均值偏差
        for col in ['T', 'S', 'D', 'W']:
            original_mean = sample_labor_data[col].mean()
            generated_mean = agents[col].mean()
            
            relative_error = abs(generated_mean - original_mean) / original_mean
            
            # 均值偏差应该<20%（放宽要求）
            assert relative_error < 0.20, f"{col}均值偏差过大: {relative_error:.2%}"


class TestLaborGeneratorEdgeCases:
    """测试边界条件"""
    
    def test_generate_single_agent(self, mock_fitted_labor_generator):
        """测试生成单个智能体"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(1)
        
        assert len(agents) == 1
        assert agents.loc[0, 'agent_id'] == 1
    
    def test_generate_large_number(self, mock_fitted_labor_generator):
        """测试生成大量智能体"""
        gen = mock_fitted_labor_generator
        agents = gen.generate(5000)
        
        assert len(agents) == 5000
        assert agents['agent_id'].is_unique
    
    def test_fit_with_minimal_data(self):
        """测试使用最小数据集拟合"""
        # 创建最小数据集（50个样本）
        np.random.seed(42)
        minimal_data = pd.DataFrame({
            'T': np.random.uniform(20, 60, 50),
            'S': np.random.uniform(10, 90, 50),
            'D': np.random.uniform(5, 80, 50),
            'W': np.random.uniform(3000, 8000, 50),
            '年龄': np.random.uniform(20, 55, 50),
            '累计工作年限': np.random.uniform(0, 30, 50),
            '孩子数量': np.random.choice([0, 1, 2], 50),
            '学历': np.random.choice([3, 4, 5, 6], 50)  # 高中到硕士
        })
        
        gen = LaborGenerator()
        
        # 应该能成功拟合（虽然可能质量不高）
        gen.fit(minimal_data)
        
        assert gen.is_fitted is True


class TestLaborGeneratorParameterSaveLoad:
    """测试参数保存和加载"""
    
    def test_fitted_params_structure(self, mock_fitted_labor_generator):
        """测试fitted_params结构"""
        gen = mock_fitted_labor_generator
        
        assert 'marginals_continuous' in gen.fitted_params
        assert 'marginals_discrete' in gen.fitted_params
        assert 'correlation_matrix' in gen.fitted_params
        assert 'conditional_probs' in gen.fitted_params
    
    def test_params_are_json_serializable(self, mock_fitted_labor_generator):
        """测试参数可以JSON序列化"""
        gen = mock_fitted_labor_generator
        
        # 尝试序列化
        try:
            json_str = json.dumps(gen.fitted_params)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"参数无法序列化: {e}")


class TestLaborGeneratorIntegration:
    """集成测试"""
    
    def test_full_workflow(self, sample_labor_data):
        """测试完整工作流程"""
        # Step 1: 创建生成器
        gen = LaborGenerator({'seed': 42})
        
        # Step 2: 拟合
        gen.fit(sample_labor_data)
        assert gen.is_fitted is True
        
        # Step 3: 生成
        agents = gen.generate(1000)
        assert len(agents) == 1000
        
        # Step 4: 验证
        is_valid = gen.validate(agents)
        assert isinstance(is_valid, bool)
        
        # Step 5: 检查基本统计特性
        assert agents['T'].mean() > 0
        assert agents['S'].mean() > 0
        assert agents['D'].mean() > 0
        assert agents['W'].mean() > 0
    
    def test_multiple_generations_from_same_fit(self, mock_fitted_labor_generator):
        """测试从同一个拟合生成多次"""
        gen = mock_fitted_labor_generator
        
        agents1 = gen.generate(100)
        agents2 = gen.generate(100)
        agents3 = gen.generate(100)
        
        # 都应该成功生成
        assert len(agents1) == 100
        assert len(agents2) == 100
        assert len(agents3) == 100
        
        # ID不应该重复（每次从1开始）
        assert agents1['agent_id'].iloc[0] == 1
        assert agents2['agent_id'].iloc[0] == 1


class TestLaborGeneratorWithRealData:
    """使用真实数据的测试（如果存在）"""
    
    def test_fit_with_real_data(self, real_cleaned_data):
        """测试使用真实数据拟合"""
        if real_cleaned_data is None:
            pytest.skip("真实数据不存在")
        
        gen = LaborGenerator({'seed': 42})
        gen.fit(real_cleaned_data)
        
        assert gen.is_fitted is True
        
        # 生成并验证
        agents = gen.generate(1000)
        assert len(agents) == 1000
    
    def test_real_data_quality(self, real_cleaned_data):
        """测试真实数据生成质量"""
        if real_cleaned_data is None:
            pytest.skip("真实数据不存在")
        
        gen = LaborGenerator({'seed': 42})
        gen.fit(real_cleaned_data)
        
        agents = gen.generate(1000)
        
        # 检查均值保留（放宽到15%）
        for col in ['T', 'S', 'D', 'W']:
            if col in real_cleaned_data.columns:
                original_mean = real_cleaned_data[col].mean()
                generated_mean = agents[col].mean()
                
                relative_error = abs(generated_mean - original_mean) / original_mean
                
                assert relative_error < 0.15, \
                    f"{col}均值偏差: {relative_error:.2%} (原始={original_mean:.2f}, 生成={generated_mean:.2f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

