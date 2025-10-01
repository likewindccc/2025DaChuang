"""
Gale-Shapley稳定匹配算法

实现延迟接受算法（Deferred Acceptance Algorithm），保证产生稳定匹配。
"""

import numpy as np
from numba import njit
from typing import Tuple, List


@njit(cache=True)
def gale_shapley(
    labor_pref_order: np.ndarray,
    enterprise_pref_order: np.ndarray
) -> np.ndarray:
    """
    Gale-Shapley延迟接受算法（劳动力提议版本）
    
    算法流程：
    1. 初始化：所有劳动力为"自由"状态，所有企业职位空缺
    2. While 存在自由劳动力且其仍有企业未申请：
       - 选择一个自由劳动力i
       - i向其偏好列表中的下一个企业j提议
       - If 企业j职位空缺：
           匹配(i, j)
       - Else if 企业j更偏好i而非当前匹配的劳动力k：
           解除匹配(k, j)，k变为自由
           匹配(i, j)
       - Else：
           i继续为自由，尝试下一个企业
    3. 返回匹配结果
    
    性质：
    - 时间复杂度：O(n*m)，其中n为劳动力数，m为企业数
    - 保证产生稳定匹配（不存在blocking pair）
    - 劳动力最优稳定匹配（对劳动力而言是所有稳定匹配中最优的）
    - 企业最劣稳定匹配（对企业而言是所有稳定匹配中最劣的）
    
    Args:
        labor_pref_order: (n_labor, n_enterprise) 劳动力偏好排序
                         每行是该劳动力对所有企业的偏好排序（索引），
                         第0列是最偏好的企业
        enterprise_pref_order: (n_enterprise, n_labor) 企业偏好排序
                              每行是该企业对所有劳动力的偏好排序（索引），
                              第0列是最偏好的劳动力
    
    Returns:
        matching: (n_labor,) 匹配结果
                 matching[i] = j 表示劳动力i匹配到企业j
                 matching[i] = -1 表示劳动力i未匹配
    """
    n_labor = labor_pref_order.shape[0]
    n_enterprise = enterprise_pref_order.shape[0]
    
    # 初始化匹配结果
    matching = np.full(n_labor, -1, dtype=np.int32)  # 劳动力→企业
    reverse_matching = np.full(n_enterprise, -1, dtype=np.int32)  # 企业→劳动力
    
    # 记录每个劳动力下一个要提议的企业索引（在其偏好列表中的位置）
    next_proposal = np.zeros(n_labor, dtype=np.int32)
    
    # 构建企业的偏好排名映射（用于快速比较）
    # enterprise_rank[j, i] = 劳动力i在企业j偏好中的排名（0最好）
    enterprise_rank = np.zeros((n_enterprise, n_labor), dtype=np.int32)
    for j in range(n_enterprise):
        for rank in range(n_labor):
            labor_id = enterprise_pref_order[j, rank]
            enterprise_rank[j, labor_id] = rank
    
    # 自由劳动力队列（使用数组模拟）
    free_labor = np.arange(n_labor, dtype=np.int32)
    free_count = n_labor
    
    # 主循环
    while free_count > 0:
        # 取出一个自由劳动力
        i = free_labor[free_count - 1]
        free_count -= 1
        
        # 检查i是否已申请完所有企业
        if next_proposal[i] >= n_enterprise:
            continue  # i无法匹配，保持未匹配状态
        
        # i向其偏好列表中的下一个企业j提议
        j = labor_pref_order[i, next_proposal[i]]
        next_proposal[i] += 1
        
        if reverse_matching[j] == -1:
            # 企业j职位空缺，直接匹配
            matching[i] = j
            reverse_matching[j] = i
        else:
            # 企业j已有匹配的劳动力k
            k = reverse_matching[j]
            
            # 比较企业j对i和k的偏好（排名越小越好）
            rank_i = enterprise_rank[j, i]
            rank_k = enterprise_rank[j, k]
            
            if rank_i < rank_k:
                # 企业j更偏好i，解除与k的匹配
                matching[k] = -1
                free_labor[free_count] = k
                free_count += 1
                
                # 匹配i和j
                matching[i] = j
                reverse_matching[j] = i
            else:
                # 企业j拒绝i，i继续为自由
                free_labor[free_count] = i
                free_count += 1
    
    return matching


def verify_stability(
    matching: np.ndarray,
    labor_pref_order: np.ndarray,
    enterprise_pref_order: np.ndarray
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    验证匹配的稳定性
    
    稳定性定义：不存在blocking pair (i, j)，使得：
    - 劳动力i更偏好企业j而非当前匹配
    - 企业j更偏好劳动力i而非当前匹配
    
    Args:
        matching: (n_labor,) 匹配结果
        labor_pref_order: (n_labor, n_enterprise) 劳动力偏好排序
        enterprise_pref_order: (n_enterprise, n_labor) 企业偏好排序
    
    Returns:
        is_stable: 是否稳定
        unstable_pairs: 不稳定匹配对列表 [(labor_id, enterprise_id), ...]
    """
    n_labor = len(matching)
    n_enterprise = enterprise_pref_order.shape[0]
    
    # 构建反向匹配
    reverse_matching = {j: None for j in range(n_enterprise)}
    for i, j in enumerate(matching):
        if j != -1:
            reverse_matching[j] = i
    
    # 构建偏好排名映射
    labor_rank = np.zeros((n_labor, n_enterprise), dtype=np.int32)
    for i in range(n_labor):
        for rank in range(n_enterprise):
            j = labor_pref_order[i, rank]
            labor_rank[i, j] = rank
    
    enterprise_rank = np.zeros((n_enterprise, n_labor), dtype=np.int32)
    for j in range(n_enterprise):
        for rank in range(n_labor):
            i = enterprise_pref_order[j, rank]
            enterprise_rank[j, i] = rank
    
    unstable_pairs = []
    
    # 检查每对(i, j)
    for i in range(n_labor):
        for j in range(n_enterprise):
            current_match_i = matching[i]
            current_match_j = reverse_matching[j]
            
            # 如果i和j已经匹配，跳过
            if current_match_i == j:
                continue
            
            # 检查i是否更偏好j而不是当前匹配
            if current_match_i == -1:
                i_prefers_j = True  # i未匹配，任何企业都比未匹配好
            else:
                i_prefers_j = labor_rank[i, j] < labor_rank[i, current_match_i]
            
            # 检查j是否更偏好i而不是当前匹配
            if current_match_j is None:
                j_prefers_i = True  # j职位空缺，任何劳动力都比空缺好
            else:
                j_prefers_i = enterprise_rank[j, i] < enterprise_rank[j, current_match_j]
            
            # 如果双方都更偏好对方，形成blocking pair
            if i_prefers_j and j_prefers_i:
                unstable_pairs.append((i, j))
    
    return (len(unstable_pairs) == 0), unstable_pairs


def compute_matching_statistics(
    matching: np.ndarray,
    labor_features: np.ndarray,
    enterprise_features: np.ndarray
) -> dict:
    """
    计算匹配统计信息
    
    Args:
        matching: (n_labor,) 匹配结果
        labor_features: (n_labor, 4) 劳动力特征
        enterprise_features: (n_enterprise, 4) 企业特征
    
    Returns:
        stats: 统计信息字典
    """
    n_labor = len(matching)
    n_matched = np.sum(matching != -1)
    
    stats = {
        'n_labor': n_labor,
        'n_enterprise': len(enterprise_features),
        'n_matched': int(n_matched),
        'n_unmatched': int(n_labor - n_matched),
        'match_rate': float(n_matched / n_labor) if n_labor > 0 else 0.0,
        'unemployment_rate': float((n_labor - n_matched) / n_labor) if n_labor > 0 else 0.0
    }
    
    # 计算匹配的劳动力和企业的平均特征
    if n_matched > 0:
        matched_labor_idx = matching != -1
        matched_enterprise_idx = matching[matched_labor_idx]
        
        matched_labor_features = labor_features[matched_labor_idx]
        matched_enterprise_features = enterprise_features[matched_enterprise_idx]
        
        stats['matched_labor_avg_T'] = float(np.mean(matched_labor_features[:, 0]))
        stats['matched_labor_avg_S'] = float(np.mean(matched_labor_features[:, 1]))
        stats['matched_labor_avg_D'] = float(np.mean(matched_labor_features[:, 2]))
        stats['matched_labor_avg_W'] = float(np.mean(matched_labor_features[:, 3]))
        
        stats['matched_enterprise_avg_T'] = float(np.mean(matched_enterprise_features[:, 0]))
        stats['matched_enterprise_avg_S'] = float(np.mean(matched_enterprise_features[:, 1]))
        stats['matched_enterprise_avg_D'] = float(np.mean(matched_enterprise_features[:, 2]))
        stats['matched_enterprise_avg_W'] = float(np.mean(matched_enterprise_features[:, 3]))
    
    return stats

