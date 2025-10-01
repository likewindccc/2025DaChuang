"""
测试Gale-Shapley算法
"""

import pytest
import numpy as np
from src.modules.matching.gale_shapley import (
    gale_shapley,
    verify_stability,
    compute_matching_statistics
)


class TestGaleShapley:
    """测试Gale-Shapley算法"""
    
    def test_simple_matching(self):
        """测试简单匹配（2×2）"""
        # 劳动力偏好排序
        labor_pref = np.array([
            [0, 1],  # 劳动力0: 企业0 > 企业1
            [1, 0]   # 劳动力1: 企业1 > 企业0
        ], dtype=np.int32)
        
        # 企业偏好排序
        enterprise_pref = np.array([
            [0, 1],  # 企业0: 劳动力0 > 劳动力1
            [1, 0]   # 企业1: 劳动力1 > 劳动力0
        ], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        assert len(matching) == 2
        assert matching[0] == 0  # 劳动力0匹配企业0
        assert matching[1] == 1  # 劳动力1匹配企业1
    
    def test_all_matched(self):
        """测试全部匹配（n=m）"""
        np.random.seed(42)
        n = 10
        
        # 随机生成偏好排序
        labor_pref = np.argsort(np.random.rand(n, n), axis=1)[:, ::-1].astype(np.int32)
        enterprise_pref = np.argsort(np.random.rand(n, n), axis=1)[:, ::-1].astype(np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        assert len(matching) == n
        assert np.all(matching != -1)  # 全部匹配
        assert len(set(matching)) == n  # 无重复匹配
    
    def test_excess_labor(self):
        """测试劳动力过剩（n>m）"""
        # 5个劳动力，3个企业
        labor_pref = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
            [1, 2, 0]
        ], dtype=np.int32)
        
        enterprise_pref = np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1]
        ], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        assert len(matching) == 5
        n_matched = np.sum(matching != -1)
        assert n_matched == 3  # 最多3个匹配
        assert len(set(matching[matching != -1])) == 3  # 3个不同的企业
    
    def test_excess_enterprises(self):
        """测试企业过剩（n<m）"""
        # 3个劳动力，5个企业
        labor_pref = np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1]
        ], dtype=np.int32)
        
        enterprise_pref = np.array([
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2]
        ], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        assert len(matching) == 3
        assert np.all(matching != -1)  # 全部匹配
    
    def test_deterministic_result(self):
        """测试结果的确定性"""
        np.random.seed(123)
        n = 20
        
        labor_pref = np.argsort(np.random.rand(n, n), axis=1)[:, ::-1].astype(np.int32)
        enterprise_pref = np.argsort(np.random.rand(n, n), axis=1)[:, ::-1].astype(np.int32)
        
        # 运行两次
        matching1 = gale_shapley(labor_pref, enterprise_pref)
        matching2 = gale_shapley(labor_pref, enterprise_pref)
        
        # 结果应该完全一致
        assert np.array_equal(matching1, matching2)


class TestStabilityVerification:
    """测试稳定性验证"""
    
    def test_stable_matching(self):
        """测试稳定匹配"""
        labor_pref = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.int32)
        
        enterprise_pref = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        is_stable, unstable_pairs = verify_stability(matching, labor_pref, enterprise_pref)
        
        assert is_stable
        assert len(unstable_pairs) == 0
    
    def test_unstable_matching(self):
        """测试不稳定匹配"""
        labor_pref = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.int32)
        
        enterprise_pref = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.int32)
        
        # 人为构造不稳定匹配
        unstable_matching = np.array([1, 0], dtype=np.int32)
        
        is_stable, unstable_pairs = verify_stability(unstable_matching, labor_pref, enterprise_pref)
        
        assert not is_stable
        assert len(unstable_pairs) > 0
    
    def test_gs_always_stable(self):
        """测试GS算法总是产生稳定匹配"""
        np.random.seed(42)
        
        for _ in range(10):  # 测试10次随机情况
            n = np.random.randint(5, 20)
            m = np.random.randint(5, 20)
            
            labor_pref = np.argsort(np.random.rand(n, m), axis=1)[:, ::-1].astype(np.int32)
            enterprise_pref = np.argsort(np.random.rand(m, n), axis=1)[:, ::-1].astype(np.int32)
            
            matching = gale_shapley(labor_pref, enterprise_pref)
            is_stable, _ = verify_stability(matching, labor_pref, enterprise_pref)
            
            assert is_stable, "GS算法应该总是产生稳定匹配"


class TestMatchingStatistics:
    """测试匹配统计"""
    
    def test_basic_statistics(self):
        """测试基本统计信息"""
        matching = np.array([0, 1, -1, 2], dtype=np.int32)
        labor_features = np.random.rand(4, 4).astype(np.float32)
        enterprise_features = np.random.rand(3, 4).astype(np.float32)
        
        stats = compute_matching_statistics(matching, labor_features, enterprise_features)
        
        assert stats['n_labor'] == 4
        assert stats['n_enterprise'] == 3
        assert stats['n_matched'] == 3
        assert stats['n_unmatched'] == 1
        assert stats['match_rate'] == 0.75
        assert stats['unemployment_rate'] == 0.25
    
    def test_all_matched_statistics(self):
        """测试全部匹配的统计"""
        matching = np.array([0, 1, 2], dtype=np.int32)
        labor_features = np.random.rand(3, 4).astype(np.float32)
        enterprise_features = np.random.rand(3, 4).astype(np.float32)
        
        stats = compute_matching_statistics(matching, labor_features, enterprise_features)
        
        assert stats['n_matched'] == 3
        assert stats['match_rate'] == 1.0
        assert stats['unemployment_rate'] == 0.0
        assert 'matched_labor_avg_T' in stats
        assert 'matched_enterprise_avg_W' in stats
    
    def test_no_match_statistics(self):
        """测试无匹配的统计"""
        matching = np.array([-1, -1, -1], dtype=np.int32)
        labor_features = np.random.rand(3, 4).astype(np.float32)
        enterprise_features = np.random.rand(3, 4).astype(np.float32)
        
        stats = compute_matching_statistics(matching, labor_features, enterprise_features)
        
        assert stats['n_matched'] == 0
        assert stats['match_rate'] == 0.0
        assert stats['unemployment_rate'] == 1.0


class TestPerformance:
    """测试性能"""
    
    def test_large_scale_matching(self):
        """测试大规模匹配（10K×5K）"""
        import time
        
        n_labor = 1000  # 降低规模以加快测试
        n_enterprise = 500
        
        np.random.seed(42)
        labor_pref = np.argsort(np.random.rand(n_labor, n_enterprise), axis=1)[:, ::-1].astype(np.int32)
        enterprise_pref = np.argsort(np.random.rand(n_enterprise, n_labor), axis=1)[:, ::-1].astype(np.int32)
        
        start_time = time.time()
        matching = gale_shapley(labor_pref, enterprise_pref)
        elapsed_time = time.time() - start_time
        
        assert len(matching) == n_labor
        assert elapsed_time < 10.0, f"匹配耗时{elapsed_time:.2f}s，超过10秒限制"
        
        print(f"\n1K×500匹配耗时: {elapsed_time:.3f}秒")
    
    @pytest.mark.slow
    def test_very_large_scale(self):
        """测试超大规模匹配（标记为慢速测试）"""
        import time
        
        n_labor = 10000
        n_enterprise = 5000
        
        np.random.seed(42)
        labor_pref = np.argsort(np.random.rand(n_labor, n_enterprise), axis=1)[:, ::-1].astype(np.int32)
        enterprise_pref = np.argsort(np.random.rand(n_enterprise, n_labor), axis=1)[:, ::-1].astype(np.int32)
        
        start_time = time.time()
        matching = gale_shapley(labor_pref, enterprise_pref)
        elapsed_time = time.time() - start_time
        
        assert len(matching) == n_labor
        
        print(f"\n10K×5K匹配耗时: {elapsed_time:.3f}秒")


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_pair(self):
        """测试单一匹配对"""
        labor_pref = np.array([[0]], dtype=np.int32)
        enterprise_pref = np.array([[0]], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        assert matching[0] == 0
    
    def test_incompatible_preferences(self):
        """测试完全不兼容的偏好"""
        # 所有劳动力都最喜欢企业0，但企业0最不喜欢他们
        labor_pref = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=np.int32)
        
        enterprise_pref = np.array([
            [2, 1, 0],  # 企业0最不喜欢劳动力0
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=np.int32)
        
        matching = gale_shapley(labor_pref, enterprise_pref)
        
        # 应该仍然产生稳定匹配
        is_stable, _ = verify_stability(matching, labor_pref, enterprise_pref)
        assert is_stable

