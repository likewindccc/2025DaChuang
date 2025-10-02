"""
精确参数搜索 - Numba加速版

目标：
1. 使用真实数据（cleaned_data.csv）
2. 固定扰动 σ=0.3，单轮匹配
3. 搜索8个参数（每个4个值），共65536个配置
4. 各项影响在同一数量级（-e2或-e3）
5. Numba加速核心计算

评估标准：
- 匹配率对θ敏感度
- 匹配率单调递增
- 合理的匹配率范围（40%-75%）

测试规模：
- 每个θ: 500个样本
- 重复10次取平均
- θ值: [0.7, 1.0, 1.3]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from numba import njit, prange
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.population.labor_generator import LaborGenerator
from src.modules.population.enterprise_generator import EnterpriseGenerator


# ============================================================================
# Numba优化的核心函数
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def compute_labor_preference_numba(
    labor_features: np.ndarray,
    enterprise_features: np.ndarray,
    gamma_0: float,
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
    gamma_4: float
) -> np.ndarray:
    """
    计算劳动力偏好矩阵（Numba优化）
    
    P_ij = γ₀ - γ₁T_j - γ₂max(0, S_j-S_i) - γ₃max(0, D_j-D_i) + γ₄W_j
    """
    n_labor = labor_features.shape[0]
    n_enterprise = enterprise_features.shape[0]
    
    preference = np.zeros((n_labor, n_enterprise), dtype=np.float32)
    
    for i in prange(n_labor):
        labor_T = labor_features[i, 0]
        labor_S = labor_features[i, 1]
        labor_D = labor_features[i, 2]
        labor_W = labor_features[i, 3]
        
        for j in range(n_enterprise):
            ent_T = enterprise_features[j, 0]
            ent_S = enterprise_features[j, 1]
            ent_D = enterprise_features[j, 2]
            ent_W = enterprise_features[j, 3]
            
            score = gamma_0
            score -= gamma_1 * ent_T
            score -= gamma_2 * max(0.0, ent_S - labor_S)
            score -= gamma_3 * max(0.0, ent_D - labor_D)
            score += gamma_4 * ent_W
            
            preference[i, j] = score
    
    return preference


@njit(parallel=True, fastmath=True, cache=True)
def compute_enterprise_preference_numba(
    enterprise_features: np.ndarray,
    labor_features: np.ndarray,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    beta_4: float
) -> np.ndarray:
    """
    计算企业偏好矩阵（Numba优化）
    
    P_ji = β₀ + β₁T_i + β₂S_i + β₃D_i + β₄W_i
    """
    n_enterprise = enterprise_features.shape[0]
    n_labor = labor_features.shape[0]
    
    preference = np.zeros((n_enterprise, n_labor), dtype=np.float32)
    
    for j in prange(n_enterprise):
        for i in range(n_labor):
            labor_T = labor_features[i, 0]
            labor_S = labor_features[i, 1]
            labor_D = labor_features[i, 2]
            labor_W = labor_features[i, 3]
            
            score = beta_0
            score += beta_1 * labor_T
            score += beta_2 * labor_S
            score += beta_3 * labor_D
            score += beta_4 * labor_W
            
            preference[j, i] = score
    
    return preference


@njit(fastmath=True, cache=True)
def add_noise_numba(preference: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """添加随机扰动（Numba优化）"""
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std, preference.shape).astype(np.float32)
    return preference + noise


@njit(parallel=True, fastmath=True, cache=True)
def compute_preference_rankings_numba(preference: np.ndarray) -> np.ndarray:
    """
    将偏好矩阵转换为排序（Numba优化）
    返回: (n, m) 数组，每行是该行偏好的企业/劳动力ID排序（从高到低）
    """
    n = preference.shape[0]
    m = preference.shape[1]
    rankings = np.zeros((n, m), dtype=np.int32)
    
    for i in prange(n):
        rankings[i] = np.argsort(-preference[i])  # 降序排列
    
    return rankings


@njit(cache=True)
def single_round_matching_numba(
    labor_pref_order: np.ndarray,
    enterprise_pref_order: np.ndarray
) -> np.ndarray:
    """
    单轮匹配（Numba优化）
    
    Returns:
        matching: (n_labor,) 数组，matching[i] = j 表示劳动力i匹配到企业j，-1表示未匹配
    """
    n_labor = labor_pref_order.shape[0]
    n_enterprise = enterprise_pref_order.shape[0]
    
    matching = np.full(n_labor, -1, dtype=np.int32)
    reverse_matching = np.full(n_enterprise, -1, dtype=np.int32)
    
    # 建立企业对劳动力的排名查找表
    enterprise_rank = np.zeros((n_enterprise, n_labor), dtype=np.int32)
    for j in range(n_enterprise):
        for rank in range(n_labor):
            labor_id = enterprise_pref_order[j, rank]
            enterprise_rank[j, labor_id] = rank
    
    # 每个劳动力向最偏好的企业投递
    for i in range(n_labor):
        if labor_pref_order.shape[1] == 0:
            continue
        j = labor_pref_order[i, 0]  # 最偏好的企业
        
        if reverse_matching[j] == -1:
            # 企业j还没有匹配，直接接受
            matching[i] = j
            reverse_matching[j] = i
        else:
            # 企业j已经有匹配，比较偏好
            k = reverse_matching[j]
            rank_i = enterprise_rank[j, i]
            rank_k = enterprise_rank[j, k]
            
            if rank_i < rank_k:
                # 企业j更偏好劳动力i
                matching[k] = -1
                matching[i] = j
                reverse_matching[j] = i
            # 否则劳动力i未被接受，保持matching[i] = -1
    
    return matching


# ============================================================================
# 主搜索逻辑
# ============================================================================

def test_one_config(
    labor_gen,
    enterprise_gen,
    gamma_params,
    beta_params,
    noise_std=0.3,
    n_labor=500,
    theta_list=[0.7, 1.0, 1.3],
    n_repeats=10
):
    """
    测试单个参数配置
    
    Returns:
        dict: {theta: match_rate}
    """
    results = {theta: [] for theta in theta_list}
    
    for repeat in range(n_repeats):
        seed = repeat
        np.random.seed(seed)
        
        for theta in theta_list:
            n_enterprise = int(n_labor * theta)
            
            # 生成数据
            labor_df = labor_gen.generate(n_agents=n_labor)
            enterprise_df = enterprise_gen.generate(n_agents=n_enterprise)
            
            labor_features = labor_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
            enterprise_features = enterprise_df[['T', 'S', 'D', 'W']].values.astype(np.float32)
            
            # 计算偏好（Numba加速）
            labor_pref = compute_labor_preference_numba(
                labor_features, enterprise_features, **gamma_params
            )
            
            enterprise_pref = compute_enterprise_preference_numba(
                enterprise_features, labor_features, **beta_params
            )
            
            # 添加扰动（Numba加速）
            labor_pref = add_noise_numba(labor_pref, noise_std, seed * 1000 + int(theta * 10))
            enterprise_pref = add_noise_numba(enterprise_pref, noise_std, seed * 1000 + int(theta * 10) + 1)
            
            # 转换为排序（Numba加速）
            labor_pref_order = compute_preference_rankings_numba(labor_pref)
            enterprise_pref_order = compute_preference_rankings_numba(enterprise_pref)
            
            # 执行单轮匹配（Numba加速）
            matching = single_round_matching_numba(labor_pref_order, enterprise_pref_order)
            
            # 计算匹配率
            match_rate = np.sum(matching != -1) / n_labor
            results[theta].append(match_rate)
    
    # 计算平均匹配率
    avg_results = {theta: np.mean(results[theta]) for theta in theta_list}
    return avg_results


def score_config(match_rates):
    """
    评分函数
    
    标准：
    1. 匹配率对θ敏感（单调递增）
    2. 合理的匹配率范围
    """
    theta_list = [0.7, 1.0, 1.3]
    rates = [match_rates[theta] for theta in theta_list]
    
    # 评分1: 单调性（差值为正）
    diff_1 = rates[1] - rates[0]
    diff_2 = rates[2] - rates[1]
    monotonicity_score = (diff_1 + diff_2) if (diff_1 > 0 and diff_2 > 0) else -10
    
    # 评分2: 匹配率范围合理性（目标：40%-75%）
    range_score = 0
    for rate in rates:
        if 0.40 <= rate <= 0.75:
            range_score += 1
        else:
            range_score -= abs(rate - 0.575) * 2  # 惩罚偏离
    
    # 评分3: 变化幅度（希望有明显变化）
    variation = rates[2] - rates[0]
    variation_score = variation * 2 if variation > 0.1 else variation - 0.5
    
    total_score = monotonicity_score + range_score + variation_score
    
    return total_score


def main():
    """主搜索流程"""
    print("=" * 80)
    print("精确参数搜索 - Numba加速版")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # 步骤1: 加载真实数据并拟合生成器
    # ========================================================================
    print("\n步骤1: 加载真实数据并拟合生成器")
    print("-" * 80)
    
    real_data = pd.read_csv('data/input/cleaned_data.csv')
    print(f"[OK] 真实数据加载完成: {len(real_data)} 条记录")
    
    # 预处理数据：计算T, S, D, W
    print("\n预处理数据...")
    real_data['T'] = real_data['每周期望工作天数'] * real_data['每天期望工作时数']
    real_data['S'] = real_data['工作能力评分']
    real_data['D'] = real_data['数字素养评分']
    real_data['W'] = real_data['每月期望收入']
    print("[OK] 核心变量计算完成: T, S, D, W")
    
    labor_gen = LaborGenerator()
    enterprise_gen = EnterpriseGenerator()
    
    print("\n拟合劳动力生成器...")
    labor_gen.fit(real_data)
    print("[OK] 劳动力生成器拟合完成")
    
    print("\n拟合企业生成器...")
    enterprise_gen.fit(real_data)
    print("[OK] 企业生成器拟合完成")
    
    # ========================================================================
    # 步骤2: 定义搜索空间
    # ========================================================================
    print("\n步骤2: 定义搜索空间")
    print("-" * 80)
    
    # 劳动力偏好参数
    gamma_1_values = np.array([0.009091, 0.018182, 0.027273, 0.036364])
    gamma_2_values = np.array([0.011905, 0.023810, 0.035714, 0.047619])
    gamma_3_values = np.array([0.025000, 0.050000, 0.075000, 0.100000])
    gamma_4_values = np.array([0.000076, 0.000152, 0.000227, 0.000303])
    
    # 企业偏好参数
    beta_1_values = np.array([0.009091, 0.018182, 0.027273, 0.036364])
    beta_2_values = np.array([0.011905, 0.023810, 0.035714, 0.047619])
    beta_3_values = np.array([0.025000, 0.050000, 0.075000, 0.100000])
    beta_4_values = np.array([-0.000303, -0.000227, -0.000152, -0.000076])
    
    total_configs = 4 ** 8
    print(f"\n搜索空间:")
    print(f"  劳动力参数: γ₁×γ₂×γ₃×γ₄ = 4×4×4×4")
    print(f"  企业参数:   β₁×β₂×β₃×β₄ = 4×4×4×4")
    print(f"  总配置数:   {total_configs:,}")
    print(f"  固定参数:   扰动σ=0.3, 单轮匹配")
    print(f"  测试规模:   500样本×3个θ×10次重复")
    
    # ========================================================================
    # 步骤3: 预热Numba（编译JIT函数）
    # ========================================================================
    print("\n步骤3: 预热Numba JIT编译")
    print("-" * 80)
    
    print("生成测试数据...")
    test_labor = labor_gen.generate(n_agents=10)
    test_enterprise = enterprise_gen.generate(n_agents=10)
    test_labor_feat = test_labor[['T', 'S', 'D', 'W']].values.astype(np.float32)
    test_ent_feat = test_enterprise[['T', 'S', 'D', 'W']].values.astype(np.float32)
    
    print("预热Numba函数（首次编译）...")
    _ = compute_labor_preference_numba(test_labor_feat, test_ent_feat, 1.0, 0.01, 0.02, 0.05, 0.0001)
    _ = compute_enterprise_preference_numba(test_ent_feat, test_labor_feat, 0.0, 0.01, 0.02, 0.05, -0.0001)
    _ = add_noise_numba(np.ones((10, 10), dtype=np.float32), 0.3, 42)
    _ = compute_preference_rankings_numba(np.random.rand(10, 10).astype(np.float32))
    _ = single_round_matching_numba(np.zeros((10, 10), dtype=np.int32), np.zeros((10, 10), dtype=np.int32))
    print("[OK] Numba预热完成")
    
    # ========================================================================
    # 步骤4: 基准测试（估算总耗时）
    # ========================================================================
    print("\n步骤4: 基准测试")
    print("-" * 80)
    
    gamma_params = {
        'gamma_0': 1.0,
        'gamma_1': gamma_1_values[0],
        'gamma_2': gamma_2_values[0],
        'gamma_3': gamma_3_values[0],
        'gamma_4': gamma_4_values[0]
    }
    beta_params = {
        'beta_0': 0.0,
        'beta_1': beta_1_values[0],
        'beta_2': beta_2_values[0],
        'beta_3': beta_3_values[0],
        'beta_4': beta_4_values[0]
    }
    
    print("测试单个配置耗时...")
    start_time = time.time()
    _ = test_one_config(labor_gen, enterprise_gen, gamma_params, beta_params)
    elapsed = time.time() - start_time
    
    print(f"[OK] 单配置耗时: {elapsed:.2f} 秒")
    
    total_time_minutes = (elapsed * total_configs) / 60
    total_time_hours = total_time_minutes / 60
    
    print(f"\n预计总耗时:")
    print(f"  {total_time_minutes:.1f} 分钟")
    print(f"  {total_time_hours:.2f} 小时")
    
    # ========================================================================
    # 步骤5: 执行精确搜索
    # ========================================================================
    print("\n步骤5: 执行精确搜索")
    print("-" * 80)
    
    print(f"\n>> 开始精确搜索 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f">> 总配置数: {total_configs:,}")
    print(f">> 预计耗时: {total_time_hours:.2f} 小时")
    print("-" * 80)
    
    results_list = []
    config_count = 0
    search_start_time = time.time()
    last_print_time = search_start_time
    
    for gamma_1 in gamma_1_values:
        for gamma_2 in gamma_2_values:
            for gamma_3 in gamma_3_values:
                for gamma_4 in gamma_4_values:
                    for beta_1 in beta_1_values:
                        for beta_2 in beta_2_values:
                            for beta_3 in beta_3_values:
                                for beta_4 in beta_4_values:
                                    config_count += 1
                                    
                                    gamma_params = {
                                        'gamma_0': 1.0,
                                        'gamma_1': gamma_1,
                                        'gamma_2': gamma_2,
                                        'gamma_3': gamma_3,
                                        'gamma_4': gamma_4
                                    }
                                    
                                    beta_params = {
                                        'beta_0': 0.0,
                                        'beta_1': beta_1,
                                        'beta_2': beta_2,
                                        'beta_3': beta_3,
                                        'beta_4': beta_4
                                    }
                                    
                                    try:
                                        match_rates = test_one_config(
                                            labor_gen, enterprise_gen,
                                            gamma_params, beta_params
                                        )
                                        
                                        score = score_config(match_rates)
                                        
                                        results_list.append({
                                            'config_id': config_count,
                                            'gamma_1': gamma_1,
                                            'gamma_2': gamma_2,
                                            'gamma_3': gamma_3,
                                            'gamma_4': gamma_4,
                                            'beta_1': beta_1,
                                            'beta_2': beta_2,
                                            'beta_3': beta_3,
                                            'beta_4': beta_4,
                                            'match_rate_0.7': match_rates[0.7],
                                            'match_rate_1.0': match_rates[1.0],
                                            'match_rate_1.3': match_rates[1.3],
                                            'score': score
                                        })
                                        
                                        # 更频繁的进度输出
                                        current_time = time.time()
                                        if config_count % 10 == 0 or (current_time - last_print_time) > 30:  # 每10个或每30秒
                                            elapsed = current_time - search_start_time
                                            avg_time = elapsed / config_count
                                            remaining_configs = total_configs - config_count
                                            remaining_time = remaining_configs * avg_time
                                            
                                            best_score = max([r['score'] for r in results_list])
                                            best_config = max(results_list, key=lambda x: x['score'])
                                            
                                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 进度: {config_count}/{total_configs} ({config_count/total_configs*100:.2f}%)")
                                            print(f"  >> 已耗时: {elapsed/60:.1f}分钟, 剩余: {remaining_time/60:.1f}分钟 (~{remaining_time/3600:.1f}小时)")
                                            print(f"  >> 速度: {avg_time:.2f}秒/配置")
                                            print(f"  >> 当前最佳: 得分={best_score:.4f}, "
                                                  f"匹配率[{best_config['match_rate_0.7']:.1%}, {best_config['match_rate_1.0']:.1%}, {best_config['match_rate_1.3']:.1%}]")
                                            
                                            last_print_time = current_time
                                    
                                    except Exception as e:
                                        print(f"配置{config_count}失败: {e}")
                                        continue
    
    search_elapsed = time.time() - search_start_time
    print(f"\n{'=' * 80}")
    print(f">> 搜索完成！")
    print(f"{'=' * 80}")
    print(f"  [OK] 成功测试: {len(results_list):,} 个配置")
    print(f"  [OK] 实际耗时: {search_elapsed/60:.1f} 分钟 ({search_elapsed/3600:.2f} 小时)")
    print(f"  [OK] 平均速度: {search_elapsed/len(results_list):.2f} 秒/配置")
    
    # ========================================================================
    # 步骤6: 保存结果
    # ========================================================================
    print(f"\n步骤6: 保存结果")
    print("-" * 80)
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('score', ascending=False)
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'exact_search_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"[OK] 完整结果已保存: {output_path}")
    
    # 保存Top 100
    top_path = output_dir / 'exact_search_top100.csv'
    results_df.head(100).to_csv(top_path, index=False)
    print(f"[OK] Top 100结果已保存: {top_path}")
    
    # ========================================================================
    # 步骤7: 展示最佳配置
    # ========================================================================
    print(f"\n步骤7: 最佳配置")
    print("=" * 80)
    
    print(f"\n>> 【Top 10 最佳配置】\n")
    
    for i, row in enumerate(results_df.head(10).itertuples(), 1):
        print(f"--- 排名 #{i} --- (得分={row.score:.4f}) ---")
        print(f"  配置ID: {row.config_id}")
        print(f"  劳动力偏好:")
        print(f"     gamma_1={row.gamma_1:.6f}, gamma_2={row.gamma_2:.6f}")
        print(f"     gamma_3={row.gamma_3:.6f}, gamma_4={row.gamma_4:.6f}")
        print(f"  企业偏好:")
        print(f"     beta_1={row.beta_1:.6f}, beta_2={row.beta_2:.6f}")
        print(f"     beta_3={row.beta_3:.6f}, beta_4={row.beta_4:.6f}")
        print(f"  匹配率:")
        print(f"     theta=0.7 -> {getattr(row, 'match_rate_0.7'):.2%}")
        print(f"     theta=1.0 -> {getattr(row, 'match_rate_1.0'):.2%}")
        print(f"     theta=1.3 -> {getattr(row, 'match_rate_1.3'):.2%}")
        print()
    
    total_elapsed = time.time() - start_time
    print("=" * 80)
    print(f">> 全部完成！")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print("=" * 80)


if __name__ == "__main__":
    main()



