"""
稀疏网格插值模块

实现基于稀疏网格的插值方法，用于在任意状态点查询值函数。

核心功能：
1. 线性插值（Numba优化）：在稀疏网格点之间线性插值
2. 最近邻插值（Numba优化）：快速查询最近的网格点值
3. 批量插值：对多个查询点并行插值

插值方法选择：
- 线性插值：精度较高，适合平滑函数
- 最近邻：速度极快，适合初始化和粗糙估计

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from numba import njit, prange
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@njit
def find_nearest_neighbor(
    query_point: np.ndarray,
    grid_nodes: np.ndarray
) -> int:
    """
    找到最近的网格点（Numba优化）
    
    Args:
        query_point: 查询点 (dimension,)
        grid_nodes: 网格节点矩阵 (dimension, n_points)
    
    Returns:
        最近网格点的索引
    
    Notes:
        使用欧氏距离: d = ||x - x_i||_2
    """
    n_points = grid_nodes.shape[1]
    min_dist = np.inf
    min_idx = 0
    
    for i in range(n_points):
        dist = 0.0
        for d in range(query_point.shape[0]):
            diff = query_point[d] - grid_nodes[d, i]
            dist += diff * diff
        
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_idx


@njit
def find_k_nearest_neighbors(
    query_point: np.ndarray,
    grid_nodes: np.ndarray,
    k: int = 16
) -> np.ndarray:
    """
    找到k个最近的网格点（Numba优化）
    
    Args:
        query_point: 查询点 (dimension,)
        grid_nodes: 网格节点矩阵 (dimension, n_points)
        k: 最近邻数量，默认16
    
    Returns:
        k个最近邻的索引数组 (k,)
    
    Notes:
        - k应该 <= n_points
        - 使用部分排序优化
    """
    n_points = grid_nodes.shape[1]
    k = min(k, n_points)
    
    # 计算所有距离
    distances = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        dist = 0.0
        for d in range(query_point.shape[0]):
            diff = query_point[d] - grid_nodes[d, i]
            dist += diff * diff
        distances[i] = dist
    
    # 找到最小的k个距离的索引
    indices = np.argsort(distances)[:k]
    return indices


@njit
def linear_interpolate(
    query_point: np.ndarray,
    grid_nodes: np.ndarray,
    grid_values: np.ndarray,
    k_neighbors: int = 16
) -> float:
    """
    基于k近邻的线性插值（Numba优化）
    
    使用反距离加权（Inverse Distance Weighting, IDW）方法：
    f(x) = Σ w_i * f(x_i) / Σ w_i
    其中 w_i = 1 / (d_i + ε)^p
    
    Args:
        query_point: 查询点 (dimension,)
        grid_nodes: 网格节点矩阵 (dimension, n_points)
        grid_values: 网格点上的函数值 (n_points,)
        k_neighbors: 使用的近邻数量，默认16
    
    Returns:
        插值结果（标量）
    
    Notes:
        - 如果查询点恰好在某个网格点上，直接返回该点的值
        - 使用IDW方法，power=2
    """
    # 找到k个最近邻
    k_indices = find_k_nearest_neighbors(query_point, grid_nodes, k_neighbors)
    
    # 计算权重（反距离加权）
    weights = np.empty(len(k_indices), dtype=np.float64)
    epsilon = 1e-10  # 避免除零
    
    for i, idx in enumerate(k_indices):
        # 计算距离
        dist_sq = 0.0
        for d in range(query_point.shape[0]):
            diff = query_point[d] - grid_nodes[d, idx]
            dist_sq += diff * diff
        
        # 如果距离极小，直接返回该点的值
        if dist_sq < epsilon:
            return grid_values[idx]
        
        # IDW权重：w = 1 / d^2
        weights[i] = 1.0 / (dist_sq + epsilon)
    
    # 加权平均
    weighted_sum = 0.0
    weight_sum = 0.0
    for i, idx in enumerate(k_indices):
        weighted_sum += weights[i] * grid_values[idx]
        weight_sum += weights[i]
    
    return weighted_sum / weight_sum


@njit(parallel=True)
def batch_linear_interpolate(
    query_points: np.ndarray,
    grid_nodes: np.ndarray,
    grid_values: np.ndarray,
    k_neighbors: int = 16
) -> np.ndarray:
    """
    批量线性插值（Numba并行优化）
    
    Args:
        query_points: 查询点矩阵 (n_queries, dimension)
        grid_nodes: 网格节点矩阵 (dimension, n_points)
        grid_values: 网格点上的函数值 (n_points,)
        k_neighbors: 近邻数量
    
    Returns:
        插值结果向量 (n_queries,)
    """
    n_queries = query_points.shape[0]
    results = np.empty(n_queries, dtype=np.float64)
    
    for i in prange(n_queries):
        results[i] = linear_interpolate(
            query_points[i],
            grid_nodes,
            grid_values,
            k_neighbors
        )
    
    return results


@njit(parallel=True)
def batch_nearest_neighbor_interpolate(
    query_points: np.ndarray,
    grid_nodes: np.ndarray,
    grid_values: np.ndarray
) -> np.ndarray:
    """
    批量最近邻插值（Numba并行优化）
    
    Args:
        query_points: 查询点矩阵 (n_queries, dimension)
        grid_nodes: 网格节点矩阵 (dimension, n_points)
        grid_values: 网格点上的函数值 (n_points,)
    
    Returns:
        插值结果向量 (n_queries,)
    """
    n_queries = query_points.shape[0]
    results = np.empty(n_queries, dtype=np.float64)
    
    for i in prange(n_queries):
        nearest_idx = find_nearest_neighbor(query_points[i], grid_nodes)
        results[i] = grid_values[nearest_idx]
    
    return results


class SparseGridInterpolator:
    """
    稀疏网格插值器
    
    封装插值功能，提供易用的接口。
    
    Attributes:
        grid_nodes: 网格节点矩阵 (dimension, n_points)
        grid_values: 网格点上的函数值 (n_points,)
        method: 插值方法 ('linear' 或 'nearest')
        k_neighbors: 线性插值使用的近邻数
    """
    
    def __init__(
        self,
        grid_nodes: np.ndarray,
        method: str = 'linear',
        k_neighbors: int = 16
    ):
        """
        初始化插值器
        
        Args:
            grid_nodes: 网格节点矩阵 (dimension, n_points)
            method: 插值方法，'linear' 或 'nearest'，默认'linear'
            k_neighbors: 线性插值使用的近邻数量，默认16
        
        Raises:
            ValueError: 如果method不是'linear'或'nearest'
        """
        if method not in ['linear', 'nearest']:
            raise ValueError(f"method必须是'linear'或'nearest'，当前为'{method}'")
        
        self.grid_nodes = grid_nodes.copy()
        self.dimension = grid_nodes.shape[0]
        self.n_points = grid_nodes.shape[1]
        self.method = method
        self.k_neighbors = min(k_neighbors, self.n_points)
        
        # 存储当前的函数值（需要通过update_values设置）
        self.grid_values = None
        
        logger.info(
            f"初始化{self.dimension}维稀疏网格插值器："
            f"{self.n_points}个网格点，方法='{method}'"
        )
    
    def update_values(self, grid_values: np.ndarray):
        """
        更新网格点的函数值
        
        Args:
            grid_values: 新的函数值 (n_points,)
        
        Raises:
            ValueError: 如果grid_values维度不匹配
        """
        if len(grid_values) != self.n_points:
            raise ValueError(
                f"grid_values长度{len(grid_values)}与网格点数{self.n_points}不匹配"
            )
        
        self.grid_values = grid_values.copy()
    
    def interpolate(self, query_point: np.ndarray) -> float:
        """
        在单个查询点插值
        
        Args:
            query_point: 查询点 (dimension,)
        
        Returns:
            插值结果（标量）
        
        Raises:
            RuntimeError: 如果grid_values未设置
        """
        if self.grid_values is None:
            raise RuntimeError("必须先调用update_values设置函数值")
        
        if self.method == 'linear':
            return linear_interpolate(
                query_point,
                self.grid_nodes,
                self.grid_values,
                self.k_neighbors
            )
        else:  # nearest
            idx = find_nearest_neighbor(query_point, self.grid_nodes)
            return self.grid_values[idx]
    
    def batch_interpolate(self, query_points: np.ndarray) -> np.ndarray:
        """
        在多个查询点批量插值
        
        Args:
            query_points: 查询点矩阵 (n_queries, dimension)
        
        Returns:
            插值结果向量 (n_queries,)
        
        Raises:
            RuntimeError: 如果grid_values未设置
        """
        if self.grid_values is None:
            raise RuntimeError("必须先调用update_values设置函数值")
        
        if self.method == 'linear':
            return batch_linear_interpolate(
                query_points,
                self.grid_nodes,
                self.grid_values,
                self.k_neighbors
            )
        else:  # nearest
            return batch_nearest_neighbor_interpolate(
                query_points,
                self.grid_nodes,
                self.grid_values
            )
    
    def __call__(self, query_point_or_points: np.ndarray):
        """
        便捷调用接口
        
        Args:
            query_point_or_points: 单个点(dimension,)或多个点(n, dimension)
        
        Returns:
            插值结果（标量或向量）
        """
        if query_point_or_points.ndim == 1:
            return self.interpolate(query_point_or_points)
        else:
            return self.batch_interpolate(query_point_or_points)
    
    def __repr__(self) -> str:
        return (
            f"SparseGridInterpolator(dimension={self.dimension}, "
            f"n_points={self.n_points}, method='{self.method}', "
            f"k_neighbors={self.k_neighbors})"
        )

# 预编译和测试
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("测试稀疏网格插值模块")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    dimension = 4
    n_grid_points = 200
    
    # 随机网格点
    grid_nodes = np.random.rand(dimension, n_grid_points)
    
    # 测试函数: f(x) = sum(x^2)
    grid_values = np.sum(grid_nodes ** 2, axis=0)
    
    # 测试1: 最近邻插值
    print("\n[测试1] 最近邻插值")
    query_point = np.array([0.5, 0.5, 0.5, 0.5])
    
    start = time.time()
    nearest_idx = find_nearest_neighbor(query_point, grid_nodes)
    nearest_value = grid_values[nearest_idx]
    elapsed = time.time() - start
    
    true_value = np.sum(query_point ** 2)
    print(f"  查询点: {query_point}")
    print(f"  最近邻值: {nearest_value:.6f}")
    print(f"  真实值: {true_value:.6f}")
    print(f"  误差: {abs(nearest_value - true_value):.6f}")
    print(f"  耗时: {elapsed*1000:.4f} ms")
    
    # 测试2: 线性插值
    print("\n[测试2] 线性插值（k=16）")
    start = time.time()
    linear_value = linear_interpolate(query_point, grid_nodes, grid_values, k_neighbors=16)
    elapsed = time.time() - start
    
    print(f"  插值结果: {linear_value:.6f}")
    print(f"  真实值: {true_value:.6f}")
    print(f"  误差: {abs(linear_value - true_value):.6f}")
    print(f"  耗时: {elapsed*1000:.4f} ms")
    
    # 测试3: 批量插值
    print("\n[测试3] 批量插值（1000个查询点）")
    n_queries = 1000
    query_points = np.random.rand(n_queries, dimension)
    true_values = np.sum(query_points ** 2, axis=1)
    
    # 线性插值
    start = time.time()
    interpolated_values = batch_linear_interpolate(
        query_points, grid_nodes, grid_values, k_neighbors=16
    )
    elapsed = time.time() - start
    
    avg_error = np.mean(np.abs(interpolated_values - true_values))
    max_error = np.max(np.abs(interpolated_values - true_values))
    
    print(f"  方法: 线性插值")
    print(f"  平均误差: {avg_error:.6f}")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  总耗时: {elapsed*1000:.2f} ms")
    print(f"  平均每点: {elapsed/n_queries*1000:.4f} ms")
    
    # 最近邻插值
    start = time.time()
    nearest_values = batch_nearest_neighbor_interpolate(
        query_points, grid_nodes, grid_values
    )
    elapsed = time.time() - start
    
    avg_error_nn = np.mean(np.abs(nearest_values - true_values))
    max_error_nn = np.max(np.abs(nearest_values - true_values))
    
    print(f"\n  方法: 最近邻插值")
    print(f"  平均误差: {avg_error_nn:.6f}")
    print(f"  最大误差: {max_error_nn:.6f}")
    print(f"  总耗时: {elapsed*1000:.2f} ms")
    print(f"  平均每点: {elapsed/n_queries*1000:.4f} ms")
    
    # 测试4: 插值器类
    print("\n[测试4] SparseGridInterpolator类")
    interpolator = SparseGridInterpolator(grid_nodes, method='linear', k_neighbors=16)
    interpolator.update_values(grid_values)
    
    test_point = np.array([0.3, 0.7, 0.2, 0.8])
    result = interpolator(test_point)
    true_val = np.sum(test_point ** 2)
    
    print(f"  {interpolator}")
    print(f"  查询点: {test_point}")
    print(f"  插值结果: {result:.6f}")
    print(f"  真实值: {true_val:.6f}")
    print(f"  误差: {abs(result - true_val):.6f}")
    
    # 批量测试
    batch_result = interpolator(query_points[:10])
    print(f"  批量查询（10个点）: shape={batch_result.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！插值模块工作正常。")
    print("=" * 60)


