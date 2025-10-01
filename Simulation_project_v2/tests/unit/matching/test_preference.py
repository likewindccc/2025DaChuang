"""
测试偏好计算模块
"""

import pytest
import numpy as np
from src.modules.matching.preference import (
    compute_labor_preference_matrix,
    compute_enterprise_preference_matrix,
    compute_preference_rankings
)


class TestLaborPreference:
    """测试劳动力偏好计算"""
    
    def test_basic_preference_calculation(self):
        """测试基本偏好计算"""
        # 1个劳动力，2个企业
        labor = np.array([[40, 0.5, 0.5, 3000]], dtype=np.float32)
        enterprises = np.array([
            [40, 0.6, 0.6, 3000],  # 要求稍高，工资相同
            [40, 0.3, 0.3, 3000]   # 要求低，工资相同
        ], dtype=np.float32)
        
        pref = compute_labor_preference_matrix(
            labor, enterprises,
            gamma_0=1.0, gamma_1=0.01, gamma_2=0.5,
            gamma_3=0.5, gamma_4=0.001
        )
        
        assert pref.shape == (1, 2)
        # 工资相同时，第2个企业应该偏好更高（要求低于自身，无惩罚）
        # 企业1有技能差距惩罚，企业2无惩罚
        assert pref[0, 1] > pref[0, 0]
    
    def test_skill_gap_penalty(self):
        """测试技能差距惩罚的不对称性"""
        labor = np.array([[40, 0.5, 0.5, 3000]], dtype=np.float32)
        
        # 企业1：技能要求高于劳动力
        # 企业2：技能要求低于劳动力
        enterprises = np.array([
            [40, 0.8, 0.5, 3000],  # S要求高
            [40, 0.3, 0.5, 3000]   # S要求低
        ], dtype=np.float32)
        
        pref = compute_labor_preference_matrix(
            labor, enterprises,
            gamma_0=1.0, gamma_1=0.01, gamma_2=0.5,
            gamma_3=0.5, gamma_4=0.001
        )
        
        # 企业1有惩罚（要求超过自身），企业2无惩罚
        assert pref[0, 1] > pref[0, 0]
    
    def test_work_time_penalty(self):
        """测试工作时长负面影响"""
        labor = np.array([[40, 0.5, 0.5, 3000]], dtype=np.float32)
        
        enterprises = np.array([
            [60, 0.5, 0.5, 3000],  # 长时间
            [40, 0.5, 0.5, 3000]   # 中等时间
        ], dtype=np.float32)
        
        pref = compute_labor_preference_matrix(
            labor, enterprises,
            gamma_0=1.0, gamma_1=0.01, gamma_2=0.5,
            gamma_3=0.5, gamma_4=0.001
        )
        
        # 工作时长短的更受偏好
        assert pref[0, 1] > pref[0, 0]
    
    def test_wage_preference(self):
        """测试工资正面影响"""
        labor = np.array([[40, 0.5, 0.5, 3000]], dtype=np.float32)
        
        enterprises = np.array([
            [40, 0.5, 0.5, 5000],  # 高工资
            [40, 0.5, 0.5, 3000]   # 低工资
        ], dtype=np.float32)
        
        pref = compute_labor_preference_matrix(
            labor, enterprises,
            gamma_0=1.0, gamma_1=0.01, gamma_2=0.5,
            gamma_3=0.5, gamma_4=0.001
        )
        
        # 工资高的更受偏好
        assert pref[0, 0] > pref[0, 1]


class TestEnterprisePreference:
    """测试企业偏好计算"""
    
    def test_basic_preference_calculation(self):
        """测试基本偏好计算"""
        enterprises = np.array([[40, 0.5, 0.5, 4000]], dtype=np.float32)
        labor = np.array([
            [40, 0.8, 0.8, 3000],  # 高能力，低要价
            [40, 0.3, 0.3, 5000]   # 低能力，高要价
        ], dtype=np.float32)
        
        pref = compute_enterprise_preference_matrix(
            enterprises, labor,
            beta_0=0.0, beta_1=0.5, beta_2=1.0,
            beta_3=1.0, beta_4=-0.001
        )
        
        assert pref.shape == (1, 2)
        # 第1个劳动力应该更受偏好（能力强，要价低）
        assert pref[0, 0] > pref[0, 1]
    
    def test_wage_penalty(self):
        """测试期望工资负面影响（β₄<0）"""
        enterprises = np.array([[40, 0.5, 0.5, 4000]], dtype=np.float32)
        
        labor = np.array([
            [40, 0.5, 0.5, 6000],  # 高要价
            [40, 0.5, 0.5, 3000]   # 低要价
        ], dtype=np.float32)
        
        pref = compute_enterprise_preference_matrix(
            enterprises, labor,
            beta_0=0.0, beta_1=0.5, beta_2=1.0,
            beta_3=1.0, beta_4=-0.001
        )
        
        # 低要价的更受偏好
        assert pref[0, 1] > pref[0, 0]
        
        # 验证差异来自工资项
        wage_diff = (3000 - 6000) * (-0.001)
        expected_diff = wage_diff
        actual_diff = pref[0, 1] - pref[0, 0]
        assert abs(actual_diff - expected_diff) < 0.01
    
    def test_skill_preference(self):
        """测试技能水平正面影响"""
        enterprises = np.array([[40, 0.5, 0.5, 4000]], dtype=np.float32)
        
        labor = np.array([
            [40, 0.9, 0.5, 3000],  # 高技能
            [40, 0.3, 0.5, 3000]   # 低技能
        ], dtype=np.float32)
        
        pref = compute_enterprise_preference_matrix(
            enterprises, labor,
            beta_0=0.0, beta_1=0.5, beta_2=1.0,
            beta_3=1.0, beta_4=-0.001
        )
        
        # 高技能的更受偏好
        assert pref[0, 0] > pref[0, 1]
    
    def test_work_time_preference(self):
        """测试工作时间正面影响"""
        enterprises = np.array([[40, 0.5, 0.5, 4000]], dtype=np.float32)
        
        labor = np.array([
            [50, 0.5, 0.5, 3000],  # 长时间
            [30, 0.5, 0.5, 3000]   # 短时间
        ], dtype=np.float32)
        
        pref = compute_enterprise_preference_matrix(
            enterprises, labor,
            beta_0=0.0, beta_1=0.5, beta_2=1.0,
            beta_3=1.0, beta_4=-0.001
        )
        
        # 长时间的更受偏好
        assert pref[0, 0] > pref[0, 1]


class TestPreferenceRankings:
    """测试偏好排序"""
    
    def test_ranking_conversion(self):
        """测试偏好分数转排序"""
        # 3×4偏好矩阵
        preference = np.array([
            [0.8, 0.5, 0.9, 0.3],  # 排序应该是: 2, 0, 1, 3
            [0.2, 0.7, 0.4, 0.6],  # 排序应该是: 1, 3, 2, 0
            [0.5, 0.5, 0.5, 0.5]   # 全部相同（实际排序取决于argsort稳定性）
        ], dtype=np.float32)
        
        rankings = compute_preference_rankings(preference)
        
        assert rankings.shape == (3, 4)
        assert rankings[0, 0] == 2  # 最高分0.9的索引
        assert rankings[1, 0] == 1  # 最高分0.7的索引
    
    def test_large_matrix_ranking(self):
        """测试大规模偏好矩阵排序"""
        n, m = 100, 50
        preference = np.random.rand(n, m).astype(np.float32)
        
        rankings = compute_preference_rankings(preference)
        
        assert rankings.shape == (n, m)
        # 验证每行都是0到m-1的排列
        for i in range(n):
            assert set(rankings[i]) == set(range(m))


class TestPreferenceIntegration:
    """测试偏好计算集成场景"""
    
    def test_realistic_scenario(self):
        """测试真实场景"""
        # 10个劳动力，5个企业
        np.random.seed(42)
        labor = np.random.rand(10, 4).astype(np.float32)
        labor[:, 0] *= 60  # T: 0-60小时
        labor[:, 3] *= 5000  # W: 0-5000元
        
        enterprises = np.random.rand(5, 4).astype(np.float32)
        enterprises[:, 0] *= 60
        enterprises[:, 3] *= 6000
        
        # 计算偏好
        labor_pref = compute_labor_preference_matrix(labor, enterprises)
        enterprise_pref = compute_enterprise_preference_matrix(enterprises, labor)
        
        assert labor_pref.shape == (10, 5)
        assert enterprise_pref.shape == (5, 10)
        
        # 转换为排序
        labor_rank = compute_preference_rankings(labor_pref)
        enterprise_rank = compute_preference_rankings(enterprise_pref)
        
        assert labor_rank.shape == (10, 5)
        assert enterprise_rank.shape == (5, 10)
    
    def test_parameter_sensitivity(self):
        """测试参数敏感性"""
        labor = np.array([[40, 0.5, 0.5, 3000]], dtype=np.float32)
        enterprises = np.array([[40, 0.7, 0.7, 4000]], dtype=np.float32)
        
        # 不同gamma_2值
        pref1 = compute_labor_preference_matrix(labor, enterprises, gamma_2=0.1)
        pref2 = compute_labor_preference_matrix(labor, enterprises, gamma_2=1.0)
        
        # gamma_2越大，技能差距惩罚越严重
        assert pref1[0, 0] > pref2[0, 0]

