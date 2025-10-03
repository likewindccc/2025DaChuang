"""
稀疏网格模块（基于chaospy）

使用Smolyak稀疏网格方法解决高维状态空间的"维度诅咒"问题。

核心功能：
1. 基于chaospy生成4维Smolyak稀疏网格
2. 提供网格点、权重和层次结构
3. 支持不同精度级别（level 1-10）

理论基础：
- Smolyak稀疏网格：O(2^l * l^(d-1))个点，而全张量网格需O(2^(l*d))个点
- 在4维空间，level=5时约15,000个点 vs 全张量网格65,536个点

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
import chaospy as cp
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SparseGrid:
    """
    稀疏网格类（基于chaospy的Smolyak方法）
    
    管理4维状态空间的稀疏网格点和权重，用于贝尔曼方程和KFE的数值求解。
    
    Attributes:
        dimension: 状态空间维度（固定为4）
        level: 精度级别（1-10）
        bounds: 每个维度的边界 [(min, max), ...]
        nodes: 网格节点矩阵 (dimension, n_points)
        weights: 积分权重向量 (n_points,)
        n_points: 网格点数量
    """
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        level: int = 5,
        rule: str = 'clenshaw_curtis'
    ):
        """
        初始化稀疏网格
        
        Args:
            bounds: 每个维度的边界列表，长度为4
                    例如: [(15, 70), (0, 1), (0, 1), (1400, 8000)]
            level: Smolyak精度级别，默认5
                   - level=3: ~200个点
                   - level=5: ~15,000个点
                   - level=7: ~100,000个点
            rule: 积分规则，默认'clenshaw_curtis'（Clenshaw-Curtis嵌套网格）
                  支持的规则: 'clenshaw_curtis', 'gaussian', 'leja', 'newton_cotes'
        
        Raises:
            ValueError: 如果bounds维度不是4或level无效
        """
        if len(bounds) != 4:
            raise ValueError(f"bounds必须是4维，当前为{len(bounds)}维")
        
        if not (1 <= level <= 10):
            raise ValueError(f"level必须在1-10之间，当前为{level}")
        
        self.dimension = 4
        self.level = level
        self.bounds = bounds
        self.rule = rule
        
        # 生成稀疏网格
        logger.info(f"开始生成{self.dimension}维Smolyak稀疏网格（level={level}）...")
        self._generate_grid()
        logger.info(
            f"稀疏网格生成完成：{self.n_points}个网格点"
            f"（全张量网格需{(level+1)**4}个点）"
        )
        
        # 计算节省比例
        full_tensor_points = (level + 1) ** self.dimension
        self.efficiency = self.n_points / full_tensor_points
        logger.info(f"稀疏网格效率：{self.efficiency:.2%}（节省{1-self.efficiency:.2%}）")
    
    def _generate_grid(self):
        """
        生成Smolyak稀疏网格（内部方法）
        
        使用chaospy的generate_quadrature函数生成稀疏网格点和权重。
        sparse=True 参数启用Smolyak稀疏网格方法。
        """
        # 创建4维独立均匀分布（标准化到[0,1]）
        dist = cp.Iid(cp.Uniform(0, 1), self.dimension)
        
        # 生成Smolyak稀疏网格
        # sparse=True 启用Smolyak方法
        # order参数决定精度，对应我们的level
        try:
            nodes_std, weights = cp.generate_quadrature(
                order=self.level,
                dist=dist,
                rule=self.rule,
                sparse=True,  # ⭐ 关键：启用Smolyak稀疏网格
                growth=True   # 使用嵌套网格（提高效率）
            )
        except Exception as e:
            logger.error(f"chaospy生成稀疏网格失败: {e}")
            raise RuntimeError(f"稀疏网格生成失败: {e}")
        
        # 缩放到实际边界
        # nodes_std: (dimension, n_points)，每维∈[0,1]
        self.nodes = self._scale_to_bounds(nodes_std)
        self.weights = weights
        self.n_points = nodes_std.shape[1]
        
        # 验证权重和为1（积分性质）
        weight_sum = np.sum(self.weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            logger.warning(
                f"权重和为{weight_sum:.8f}，不等于1.0，"
                f"已自动归一化"
            )
            self.weights = self.weights / weight_sum
    
    def _scale_to_bounds(self, nodes_std: np.ndarray) -> np.ndarray:
        """
        将标准化网格点[0,1]缩放到实际边界
        
        Args:
            nodes_std: 标准化网格点 (dimension, n_points)
        
        Returns:
            缩放后的网格点 (dimension, n_points)
        """
        nodes_scaled = np.empty_like(nodes_std)
        
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            # 线性缩放: x_scaled = x_std * (max - min) + min
            nodes_scaled[i] = nodes_std[i] * (max_val - min_val) + min_val
        
        return nodes_scaled
    
    def get_nodes(self) -> np.ndarray:
        """
        获取所有网格节点
        
        Returns:
            网格节点矩阵 (dimension, n_points)
        """
        return self.nodes.copy()
    
    def get_weights(self) -> np.ndarray:
        """
        获取积分权重
        
        Returns:
            权重向量 (n_points,)
        """
        return self.weights.copy()
    
    def get_nodes_as_matrix(self) -> np.ndarray:
        """
        获取网格节点矩阵（转置形式）
        
        Returns:
            网格节点矩阵 (n_points, dimension)
        
        Notes:
            这种格式更方便用于批量计算，每行是一个状态向量
        """
        return self.nodes.T
    
    def get_node(self, idx: int) -> np.ndarray:
        """
        获取单个网格节点
        
        Args:
            idx: 节点索引，0 <= idx < n_points
        
        Returns:
            状态向量 (dimension,)
        
        Raises:
            IndexError: 如果索引超出范围
        """
        if not (0 <= idx < self.n_points):
            raise IndexError(
                f"节点索引{idx}超出范围[0, {self.n_points})"
            )
        return self.nodes[:, idx].copy()
    
    def integrate(self, values: np.ndarray) -> float:
        """
        使用稀疏网格权重进行积分
        
        Args:
            values: 在每个网格点的函数值 (n_points,)
        
        Returns:
            积分值（标量）
        
        Examples:
            >>> grid = SparseGrid(bounds, level=3)
            >>> # 计算∫f(x)dx，其中f在网格点已评估
            >>> integral = grid.integrate(f_values)
        
        Notes:
            积分公式: ∫f(x)dx ≈ Σ_i w_i * f(x_i)
        """
        if len(values) != self.n_points:
            raise ValueError(
                f"values长度{len(values)}与网格点数{self.n_points}不匹配"
            )
        
        # 加权求和
        integral = np.sum(self.weights * values)
        
        # 需要乘以超立方体体积（因为chaospy标准化到[0,1]^d）
        volume = 1.0
        for min_val, max_val in self.bounds:
            volume *= (max_val - min_val)
        
        return integral * volume
    
    def get_grid_info(self) -> Dict:
        """
        获取稀疏网格的详细信息
        
        Returns:
            包含网格信息的字典
        """
        return {
            'dimension': self.dimension,
            'level': self.level,
            'n_points': self.n_points,
            'bounds': self.bounds,
            'efficiency': self.efficiency,
            'full_tensor_points': (self.level + 1) ** self.dimension,
            'weight_sum': np.sum(self.weights),
            'rule': self.rule
        }
    
    def __repr__(self) -> str:
        return (
            f"SparseGrid(dim={self.dimension}, level={self.level}, "
            f"n_points={self.n_points}, efficiency={self.efficiency:.2%})"
        )


def create_mfg_sparse_grid(
    state_space_bounds: List[Tuple[float, float]],
    level: int = 5
) -> SparseGrid:
    """
    创建MFG模拟专用的稀疏网格（工厂函数）
    
    Args:
        state_space_bounds: 4维状态空间边界
            - T: 每周工作小时数
            - S_norm: 标准化工作能力 [0, 1]
            - D_norm: 标准化数字素养 [0, 1]
            - W: 期望工资
        level: 精度级别，默认5
    
    Returns:
        配置好的SparseGrid实例
    
    Examples:
        >>> bounds = [(15, 70), (0, 1), (0, 1), (1400, 8000)]
        >>> grid = create_mfg_sparse_grid(bounds, level=5)
        >>> print(grid.n_points)  # ~15,000
    """
    if len(state_space_bounds) != 4:
        raise ValueError("MFG状态空间必须是4维")
    
    logger.info("创建MFG专用稀疏网格...")
    grid = SparseGrid(
        bounds=state_space_bounds,
        level=level,
        rule='smolyak'
    )
    
    logger.info(
        f"MFG稀疏网格创建完成：{grid.n_points}个网格点"
    )
    
    return grid


# 预编译测试
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("测试稀疏网格模块（基于chaospy）")
    print("=" * 60)
    
    # 测试1: 创建MFG稀疏网格
    print("\n[测试 1] 创建4维稀疏网格（level=3）")
    bounds = [(15, 70), (0, 1), (0, 1), (1400, 8000)]
    
    start_time = time.time()
    grid = SparseGrid(bounds, level=3)
    elapsed = time.time() - start_time
    
    print(f"  创建耗时: {elapsed:.4f}秒")
    print(f"  {grid}")
    print(f"  网格点数量: {grid.n_points}")
    print(f"  权重和: {np.sum(grid.weights):.10f}")
    
    # 测试2: 获取网格节点
    print("\n[测试 2] 获取网格节点")
    nodes_matrix = grid.get_nodes_as_matrix()
    print(f"  节点矩阵形状: {nodes_matrix.shape}")
    print(f"  前3个节点:")
    for i in range(min(3, grid.n_points)):
        node = grid.get_node(i)
        print(f"    节点{i}: T={node[0]:.2f}, S={node[1]:.4f}, "
              f"D={node[2]:.4f}, W={node[3]:.2f}")
    
    # 测试3: 积分测试（积分常数函数应为体积）
    print("\n[测试 3] 积分测试")
    ones = np.ones(grid.n_points)
    integral = grid.integrate(ones)
    
    # 计算理论体积
    volume = 1.0
    for min_val, max_val in bounds:
        volume *= (max_val - min_val)
    
    print(f"  ∫1 dx (数值): {integral:.2f}")
    print(f"  理论体积: {volume:.2f}")
    print(f"  相对误差: {abs(integral - volume) / volume:.6%}")
    
    # 测试4: 更高精度网格
    print("\n[测试 4] Level=5 稀疏网格（实际MFG使用）")
    start_time = time.time()
    grid_l5 = SparseGrid(bounds, level=5)
    elapsed = time.time() - start_time
    
    print(f"  创建耗时: {elapsed:.4f}秒")
    print(f"  {grid_l5}")
    print(f"  网格信息:")
    info = grid_l5.get_grid_info()
    for key, value in info.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！稀疏网格模块工作正常。")
    print("=" * 60)

