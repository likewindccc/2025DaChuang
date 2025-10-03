#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core数据结构模块

定义项目中所有核心数据结构，包括：
- Agent: 个体基类（劳动力/企业）
- MatchingPair: 匹配对
- SimulationState: 模拟状态
- MFGEquilibrium: MFG均衡结果

所有数据结构使用dataclass实现，提供类型安全和自动方法生成。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class Agent:
    """
    个体基类
    
    表示劳动力或企业的基本属性。
    
    Attributes:
        agent_id: 个体唯一标识
        agent_type: 个体类型，必须是 'labor' 或 'enterprise'
        T: 工作时长相关属性（劳动力期望/企业要求，单位：小时/周）
        S: 技能/要求属性（劳动力能力/企业要求，范围：0-100）
        D: 数字素养属性（劳动力水平/企业要求，范围：0-100）
        W: 工资相关属性（劳动力期望/企业提供，单位：元/月）
        additional_attrs: 其他控制变量（如年龄、教育等），灵活存储
    
    Examples:
        >>> # 创建劳动力
        >>> labor = Agent(
        ...     agent_id=1,
        ...     agent_type='labor',
        ...     T=40.0,
        ...     S=75.0,
        ...     D=60.0,
        ...     W=4500.0,
        ...     additional_attrs={'age': 25, 'education': 'bachelor'}
        ... )
        
        >>> # 创建企业
        >>> enterprise = Agent(
        ...     agent_id=1001,
        ...     agent_type='enterprise',
        ...     T=48.0,
        ...     S=70.0,
        ...     D=65.0,
        ...     W=5000.0
        ... )
        
        >>> # 转换为NumPy数组
        >>> labor_features = labor.to_array()
        >>> print(labor_features)  # [40.0, 75.0, 60.0, 4500.0]
    
    Notes:
        - to_array()顺序固定为 [T, S, D, W]
        - additional_attrs不参与数组转换，仅用于存储元数据
        - 所有属性在__post_init__中进行验证
    """
    
    agent_id: int
    agent_type: str
    T: float
    S: float
    D: float
    W: float
    additional_attrs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        数据验证
        
        验证规则：
        - agent_type必须是'labor'或'enterprise'
        - T不能为负
        - S和D必须在0-100之间
        - W必须为正
        
        Raises:
            ValueError: 参数不符合验证规则
        """
        # 类型验证
        if self.agent_type not in ('labor', 'enterprise'):
            raise ValueError(
                f"agent_type必须是'labor'或'enterprise'，得到：{self.agent_type}"
            )
        
        # 范围验证
        if self.T < 0:
            raise ValueError(f"工作时长不能为负：{self.T}")
        
        if not (0 <= self.S <= 100):
            raise ValueError(f"技能评分应在0-100之间：{self.S}")
        
        if not (0 <= self.D <= 100):
            raise ValueError(f"数字素养评分应在0-100之间：{self.D}")
        
        if self.W <= 0:
            raise ValueError(f"工资必须为正：{self.W}")
    
    def to_array(self) -> np.ndarray:
        """
        转换为NumPy数组（用于计算）
        
        顺序固定为 [T, S, D, W]
        
        Returns:
            形状为(4,)的NumPy数组
        
        Examples:
            >>> agent = Agent(1, 'labor', 40.0, 75.0, 60.0, 4500.0)
            >>> arr = agent.to_array()
            >>> print(arr)
            [   40.    75.    60.  4500.]
        """
        return np.array([self.T, self.S, self.D, self.W], dtype=np.float64)
    
    @classmethod
    def from_array(
        cls,
        agent_id: int,
        agent_type: str,
        arr: np.ndarray,
        additional_attrs: Optional[Dict[str, Any]] = None
    ) -> 'Agent':
        """
        从NumPy数组创建Agent
        
        Args:
            agent_id: 个体ID
            agent_type: 个体类型 ('labor' 或 'enterprise')
            arr: 形状为(4,)的NumPy数组，顺序为 [T, S, D, W]
            additional_attrs: 额外属性
        
        Returns:
            Agent实例
        
        Raises:
            ValueError: 数组形状不正确
        
        Examples:
            >>> arr = np.array([40.0, 75.0, 60.0, 4500.0])
            >>> agent = Agent.from_array(1, 'labor', arr)
            >>> print(agent.T, agent.S, agent.D, agent.W)
            40.0 75.0 60.0 4500.0
        """
        if arr.shape != (4,):
            raise ValueError(f"数组形状必须是(4,)，得到：{arr.shape}")
        
        return cls(
            agent_id=agent_id,
            agent_type=agent_type,
            T=float(arr[0]),
            S=float(arr[1]),
            D=float(arr[2]),
            W=float(arr[3]),
            additional_attrs=additional_attrs
        )


@dataclass
class MatchingPair:
    """
    匹配对
    
    表示一个劳动力与企业的匹配结果。
    
    Attributes:
        labor_id: 劳动力ID
        enterprise_id: 企业ID
        matched: 是否匹配成功
        match_quality: 匹配质量得分，范围0-1（可选）
        metadata: 其他元数据（如匹配时的市场紧张度等）
    
    Examples:
        >>> # 成功匹配
        >>> pair = MatchingPair(
        ...     labor_id=1,
        ...     enterprise_id=1001,
        ...     matched=True,
        ...     match_quality=0.85,
        ...     metadata={'theta': 1.2, 'round': 3}
        ... )
        
        >>> # 未匹配
        >>> unpaired = MatchingPair(
        ...     labor_id=2,
        ...     enterprise_id=-1,
        ...     matched=False
        ... )
    
    Notes:
        - enterprise_id=-1 通常表示未匹配到企业
        - match_quality用于评估匹配效果，越高越好
    """
    
    labor_id: int
    enterprise_id: int
    matched: bool
    match_quality: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        数据验证
        
        Raises:
            ValueError: match_quality不在0-1之间
        """
        if self.match_quality is not None:
            if not (0 <= self.match_quality <= 1):
                raise ValueError(
                    f"匹配质量应在0-1之间：{self.match_quality}"
                )


@dataclass
class SimulationState:
    """
    模拟状态
    
    记录某一时刻的完整模拟状态，用于跟踪模拟演化过程。
    
    Attributes:
        time_step: 当前时间步
        laborers: 劳动力列表
        enterprises: 企业列表
        matchings: 匹配对列表
        unemployment_rate: 失业率，范围0-1
        theta: 市场紧张度 (V/U)，职位数/失业人数
        additional_metrics: 其他指标（如平均工资、匹配率等）
    
    Examples:
        >>> state = SimulationState(
        ...     time_step=0,
        ...     laborers=[labor1, labor2],
        ...     enterprises=[ent1, ent2],
        ...     matchings=[pair1],
        ...     unemployment_rate=0.1,
        ...     theta=1.0,
        ...     additional_metrics={'avg_wage': 4800.0}
        ... )
        
        >>> print(f"时间步{state.time_step}: 失业率={state.unemployment_rate:.2%}")
        时间步0: 失业率=10.00%
    
    Notes:
        - unemployment_rate = (未匹配劳动力数) / (总劳动力数)
        - theta = (职位数) / (失业人数)，反映劳动力市场松紧程度
    """
    
    time_step: int
    laborers: List[Agent]
    enterprises: List[Agent]
    matchings: List[MatchingPair]
    unemployment_rate: float
    theta: float
    additional_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """
        数据验证
        
        Raises:
            ValueError: 参数超出合理范围
        """
        if not (0 <= self.unemployment_rate <= 1):
            raise ValueError(
                f"失业率应在0-1之间：{self.unemployment_rate}"
            )
        
        if self.theta < 0:
            raise ValueError(f"市场紧张度不能为负：{self.theta}")
        
        if self.time_step < 0:
            raise ValueError(f"时间步不能为负：{self.time_step}")


@dataclass
class MFGEquilibrium:
    """
    MFG均衡结果
    
    存储平均场博弈求解后的均衡状态，包括值函数、策略函数和人口分布。
    
    Attributes:
        value_function_U: 失业状态值函数，形状 (grid_size_T, grid_size_S)
        value_function_E: 就业状态值函数，形状 (grid_size_T, grid_size_S)
        policy_function: 最优努力策略函数 a*(T,S)，形状同上
        distribution_U: 失业人口分布 m_U(T,S)，形状同上
        distribution_E: 就业人口分布 m_E(T,S)，形状同上
        theta: 均衡市场紧张度
        converged: 是否收敛
        iterations: 迭代次数
        convergence_error: 收敛误差（可选）
    
    Examples:
        >>> # 构造MFG均衡结果
        >>> grid_size = (50, 50)
        >>> equilibrium = MFGEquilibrium(
        ...     value_function_U=np.zeros(grid_size),
        ...     value_function_E=np.ones(grid_size) * 10,
        ...     policy_function=np.ones(grid_size) * 0.5,
        ...     distribution_U=np.ones(grid_size) / grid_size[0] / grid_size[1],
        ...     distribution_E=np.ones(grid_size) / grid_size[0] / grid_size[1],
        ...     theta=1.0,
        ...     converged=True,
        ...     iterations=100,
        ...     convergence_error=1e-6
        ... )
        
        >>> print(f"迭代{equilibrium.iterations}次后收敛，误差={equilibrium.convergence_error:.2e}")
        迭代100次后收敛，误差=1.00e-06
    
    Notes:
        - 所有数组形状必须一致
        - distribution应满足归一化条件：sum(distribution) ≈ 1
        - policy_function的值域通常在[0, 1]之间（努力水平）
    """
    
    value_function_U: np.ndarray
    value_function_E: np.ndarray
    policy_function: np.ndarray
    distribution_U: np.ndarray
    distribution_E: np.ndarray
    theta: float
    converged: bool
    iterations: int
    convergence_error: Optional[float] = None
    
    def __post_init__(self):
        """
        数据验证
        
        验证规则：
        - 所有数组形状必须一致
        - theta不能为负
        - iterations不能为负
        
        Raises:
            ValueError: 参数不符合验证规则
        """
        # 验证数组形状一致性
        shapes = [
            self.value_function_U.shape,
            self.value_function_E.shape,
            self.policy_function.shape,
            self.distribution_U.shape,
            self.distribution_E.shape
        ]
        
        if len(set(shapes)) != 1:
            raise ValueError(
                f"所有数组形状必须一致，得到：{shapes}"
            )
        
        # 验证参数范围
        if self.theta < 0:
            raise ValueError(f"市场紧张度不能为负：{self.theta}")
        
        if self.iterations < 0:
            raise ValueError(f"迭代次数不能为负：{self.iterations}")
        
        if self.convergence_error is not None and self.convergence_error < 0:
            raise ValueError(
                f"收敛误差不能为负：{self.convergence_error}"
            )
    
    @property
    def grid_shape(self) -> tuple:
        """
        获取网格形状
        
        Returns:
            网格形状元组 (grid_size_T, grid_size_S)
        """
        return self.value_function_U.shape
    
    @property
    def total_unemployment(self) -> float:
        """
        计算总失业人口比例
        
        Returns:
            失业人口总量（积分）
        """
        return float(np.sum(self.distribution_U))
    
    @property
    def total_employment(self) -> float:
        """
        计算总就业人口比例
        
        Returns:
            就业人口总量（积分）
        """
        return float(np.sum(self.distribution_E))


@dataclass
class MFGEquilibriumSparseGrid:
    """
    MFG均衡结果（稀疏网格版本）
    
    存储使用Smolyak稀疏网格求解的MFG均衡状态。
    适用于4维状态空间 (T, S, D, W)。
    
    Attributes:
        grid_nodes: 稀疏网格节点，形状 (4, n_points)
        V_U: 失业状态值函数，形状 (n_points,)
        V_E: 就业状态值函数，形状 (n_points,)
        a_star: 最优努力策略，形状 (n_points,)
        m_U: 失业人口分布，形状 (n_points,)
        m_E: 就业人口分布，形状 (n_points,)
        unemployment_rate: 均衡失业率
        employment_rate: 均衡就业率
        theta: 均衡市场紧张度
        converged: 是否收敛
        n_iterations: 迭代次数
        convergence_metrics: 收敛指标（diff_V, diff_a, diff_u）
        total_time: 求解总耗时（秒）
        config: 配置参数副本
        history: 历史演化记录（可选）
    
    Examples:
        >>> # 从MFG求解器结果构造
        >>> result = mfg_simulator.solve()
        >>> equilibrium = MFGEquilibriumSparseGrid.from_solver_result(result)
        >>> 
        >>> print(f"均衡失业率: {equilibrium.unemployment_rate:.2%}")
        >>> print(f"是否收敛: {equilibrium.converged}")
        >>> print(f"网格点数: {equilibrium.n_points}")
    
    Notes:
        - grid_nodes存储的是标准化后的状态（S和D在[0,1]）
        - m_U + m_E 应该归一化到1
        - a_star的值域通常在[0, 1]之间
    """
    
    grid_nodes: np.ndarray
    V_U: np.ndarray
    V_E: np.ndarray
    a_star: np.ndarray
    m_U: np.ndarray
    m_E: np.ndarray
    unemployment_rate: float
    employment_rate: float
    theta: float
    converged: bool
    n_iterations: int
    convergence_metrics: Dict[str, float]
    total_time: float
    config: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        数据验证
        
        Raises:
            ValueError: 参数不符合验证规则
        """
        # 验证grid_nodes形状
        if self.grid_nodes.shape[0] != 4:
            raise ValueError(
                f"grid_nodes第一维必须是4（状态空间维度），得到：{self.grid_nodes.shape[0]}"
            )
        
        n_points = self.grid_nodes.shape[1]
        
        # 验证所有1D数组长度一致
        arrays = {
            'V_U': self.V_U,
            'V_E': self.V_E,
            'a_star': self.a_star,
            'm_U': self.m_U,
            'm_E': self.m_E
        }
        
        for name, arr in arrays.items():
            if len(arr) != n_points:
                raise ValueError(
                    f"{name}长度{len(arr)}与网格点数{n_points}不一致"
                )
        
        # 验证失业率和就业率
        if not (0 <= self.unemployment_rate <= 1):
            raise ValueError(f"失业率应在0-1之间：{self.unemployment_rate}")
        
        if not (0 <= self.employment_rate <= 1):
            raise ValueError(f"就业率应在0-1之间：{self.employment_rate}")
        
        # 验证失业率+就业率≈1（允许小误差）
        total_rate = self.unemployment_rate + self.employment_rate
        if not (0.99 <= total_rate <= 1.01):
            raise ValueError(
                f"失业率+就业率应≈1，得到：{total_rate}"
            )
        
        # 验证theta
        if self.theta < 0:
            raise ValueError(f"市场紧张度不能为负：{self.theta}")
        
        # 验证迭代次数
        if self.n_iterations < 0:
            raise ValueError(f"迭代次数不能为负：{self.n_iterations}")
    
    @property
    def n_points(self) -> int:
        """获取稀疏网格点数"""
        return self.grid_nodes.shape[1]
    
    @property
    def dimension(self) -> int:
        """获取状态空间维度（固定为4）"""
        return 4
    
    @property
    def grid_efficiency(self) -> float:
        """
        计算稀疏网格效率
        
        Returns:
            相对于全张量网格的点数比例
        """
        # 假设level=5，全张量网格为6^4=1296个点
        full_tensor_points = 1296
        return self.n_points / full_tensor_points
    
    def get_value_at_state(
        self,
        x: np.ndarray,
        employment_status: str = 'unemployed'
    ) -> float:
        """
        查询任意状态的值函数（最近邻插值）
        
        Args:
            x: 状态向量 (4,): [T, S_norm, D_norm, W]
            employment_status: 'unemployed' 或 'employed'
        
        Returns:
            插值后的值函数
        """
        # 找到最近的网格点
        distances = np.sum((self.grid_nodes - x.reshape(-1, 1))**2, axis=0)
        nearest_idx = np.argmin(distances)
        
        if employment_status == 'unemployed':
            return float(self.V_U[nearest_idx])
        else:
            return float(self.V_E[nearest_idx])
    
    def get_optimal_effort(self, x: np.ndarray) -> float:
        """
        查询任意状态的最优努力水平（最近邻插值）
        
        Args:
            x: 状态向量 (4,)
        
        Returns:
            最优努力水平
        """
        distances = np.sum((self.grid_nodes - x.reshape(-1, 1))**2, axis=0)
        nearest_idx = np.argmin(distances)
        return float(self.a_star[nearest_idx])
    
    def summary(self) -> str:
        """
        生成均衡结果摘要
        
        Returns:
            格式化的摘要字符串
        """
        summary_lines = [
            "=" * 60,
            "MFG均衡结果摘要（稀疏网格）",
            "=" * 60,
            f"状态空间维度: {self.dimension}D",
            f"稀疏网格点数: {self.n_points}",
            f"网格效率: {self.grid_efficiency:.2%}",
            "",
            f"均衡失业率: {self.unemployment_rate:.4f} ({self.unemployment_rate*100:.2f}%)",
            f"均衡就业率: {self.employment_rate:.4f} ({self.employment_rate*100:.2f}%)",
            f"市场紧张度θ: {self.theta:.4f}",
            "",
            f"价值函数统计:",
            f"  V^U - 均值: {np.mean(self.V_U):10.2f}, 标准差: {np.std(self.V_U):8.2f}",
            f"  V^E - 均值: {np.mean(self.V_E):10.2f}, 标准差: {np.std(self.V_E):8.2f}",
            "",
            f"最优努力策略:",
            f"  a* - 均值: {np.mean(self.a_star):.4f}, 最小: {np.min(self.a_star):.4f}, 最大: {np.max(self.a_star):.4f}",
            "",
            f"求解状态:",
            f"  是否收敛: {'✅ 是' if self.converged else '❌ 否'}",
            f"  迭代次数: {self.n_iterations}",
            f"  总耗时: {self.total_time:.2f}秒 ({self.total_time/60:.1f}分钟)",
            f"  平均每次迭代: {self.total_time/self.n_iterations:.2f}秒",
            "",
            f"收敛指标:",
            f"  diff_V: {self.convergence_metrics['diff_V']:.2e}",
            f"  diff_a: {self.convergence_metrics['diff_a']:.2e}",
            f"  diff_u: {self.convergence_metrics['diff_u']:.2e}",
            "=" * 60
        ]
        
        return "\n".join(summary_lines)
    
    @classmethod
    def from_solver_result(
        cls,
        result: Dict,
        config: Optional[Dict] = None
    ) -> 'MFGEquilibriumSparseGrid':
        """
        从MFG求解器结果构造均衡对象
        
        Args:
            result: MFGSimulator.solve()返回的结果字典
            config: 配置参数（可选）
        
        Returns:
            MFGEquilibriumSparseGrid实例
        
        Examples:
            >>> result = mfg_simulator.solve()
            >>> equilibrium = MFGEquilibriumSparseGrid.from_solver_result(result)
        """
        return cls(
            grid_nodes=result['grid_nodes'],
            V_U=result['V_U'],
            V_E=result['V_E'],
            a_star=result['a_star'],
            m_U=result['m_U'],
            m_E=result['m_E'],
            unemployment_rate=result['unemployment_rate'],
            employment_rate=result['employment_rate'],
            theta=result.get('theta', 1.0),  # 默认值
            converged=result['converged'],
            n_iterations=result['n_iterations'],
            convergence_metrics=result['final_metrics'],
            total_time=result['total_time'],
            config=config,
            history=result.get('history')
        )
    
    def save(self, filepath: str):
        """
        保存均衡结果到文件
        
        Args:
            filepath: 保存路径（.npz格式）
        """
        np.savez(
            filepath,
            grid_nodes=self.grid_nodes,
            V_U=self.V_U,
            V_E=self.V_E,
            a_star=self.a_star,
            m_U=self.m_U,
            m_E=self.m_E,
            unemployment_rate=np.array([self.unemployment_rate]),
            employment_rate=np.array([self.employment_rate]),
            theta=np.array([self.theta]),
            converged=np.array([self.converged]),
            n_iterations=np.array([self.n_iterations])
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'MFGEquilibriumSparseGrid':
        """
        从文件加载均衡结果
        
        Args:
            filepath: 文件路径（.npz格式）
        
        Returns:
            MFGEquilibriumSparseGrid实例
        """
        data = np.load(filepath)
        
        return cls(
            grid_nodes=data['grid_nodes'],
            V_U=data['V_U'],
            V_E=data['V_E'],
            a_star=data['a_star'],
            m_U=data['m_U'],
            m_E=data['m_E'],
            unemployment_rate=float(data['unemployment_rate'][0]),
            employment_rate=float(data['employment_rate'][0]),
            theta=float(data['theta'][0]),
            converged=bool(data['converged'][0]),
            n_iterations=int(data['n_iterations'][0]),
            convergence_metrics={'diff_V': 0.0, 'diff_a': 0.0, 'diff_u': 0.0},  # 默认值
            total_time=0.0  # 默认值
        )
