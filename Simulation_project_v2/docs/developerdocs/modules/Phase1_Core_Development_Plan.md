# Phase 1: Core 模块开发文档

**模块名称**: Core - 核心基础模块  
**开发阶段**: Phase 1, Week 1  
**创建日期**: 2025-09-30  
**状态**: 设计阶段，待用户审阅

---

## 📋 目录

- [1. 模块概述](#1-模块概述)
- [2. 设计决策](#2-设计决策)
- [3. 核心数据结构](#3-核心数据结构)
- [4. 基础类设计](#4-基础类设计)
- [5. 异常体系](#5-异常体系)
- [6. 类型系统](#6-类型系统)
- [7. API接口定义](#7-api接口定义)
- [8. 测试策略](#8-测试策略)
- [9. 实现计划](#9-实现计划)

---

## 1. 模块概述

### 1.1 职责

Core 模块是整个项目的基础，提供：
- **核心数据结构**：Agent、MatchingPair、SimulationState 等
- **抽象基类**：所有生成器和模拟器的基类
- **异常体系**：统一的异常处理
- **类型定义**：项目通用的类型别名

### 1.2 依赖关系

```
所有其他模块
    ↓ 依赖
Core 模块
    ↓ 依赖
Python标准库 + numpy + pandas
```

**重要**：Core 模块不依赖任何其他业务模块，确保最大的复用性。

### 1.3 边际分布实验结果（用于数据结构设计）

根据实验结果（修正数据后），确定核心变量的分布类型：

| 变量 | 分布类型 | 参数 | AIC | 说明 |
|------|---------|------|-----|------|
| 每周工作时长 (T) | Beta | α=1.93, β=2.05 | -66.72 | ✅ 对称型分布 |
| 工作能力评分 (S) | Beta | α=1.79, β=1.57 | -39.99 | ✅ 对称型分布 |
| 数字素养评分 (D) | Beta | α=0.37, β=0.76 | -313.78 | ✅ U型分布（α<1, β<1）|
| 每月期望收入 (W) | Beta | α=1.43, β=1.45 | -16.04 | ✅ 对称型分布 |

**数据修正说明**：
- 发现36个数字素养评分为0的样本（占12%）
- 将这些样本的值设为0.1，避免对数正态分布拟合时的log(0)问题
- 修正后，数字素养评分拟合为Beta分布，与其他变量保持一致
- **所有4个核心变量统一使用Beta分布，便于后续Copula建模**

---

## 2. 设计决策

### 2.1 数据结构方案：dataclass ✅

**选择**：Python 3.7+ 的 `dataclass`

**原因**：
- ✅ 轻量级，性能好
- ✅ 类型提示友好（配合 `@dataclass` 装饰器）
- ✅ 自动生成 `__init__`, `__repr__`, `__eq__` 等方法
- ✅ 与 NumPy/Pandas 兼容良好
- ✅ 可以添加自定义方法和属性验证

**不选择 Pydantic 的原因**：
- Pydantic 功能强大但依赖较重
- Core 模块需要保持轻量
- 数据验证可以在业务层实现

### 2.2 类型注解：严格标注 ✅

**策略**：
- 所有公共函数/方法：**必须**有类型注解
- 私有方法：**推荐**有类型注解
- Numba函数：**可选**（避免兼容性问题）

**示例**：
```python
from typing import List, Dict, Optional
from numpy.typing import NDArray
import numpy as np

def process_agents(
    agents: List['Agent'],
    config: Dict[str, Any]
) -> NDArray[np.float64]:
    ...
```

### 2.3 异常体系：细粒度 ✅

**策略**：继承自 `SimulationError` 基类，每个模块有专门的异常

**优点**：
- 精确捕获和处理错误
- 便于调试
- 更好的错误信息

---

## 3. 核心数据结构

### 3.1 Agent（个体基类）

**用途**：表示劳动力或企业个体

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Agent:
    """
    个体基类
    
    表示劳动力或企业的基本属性
    
    Attributes:
        agent_id: 个体唯一标识
        agent_type: 个体类型 ('labor' 或 'enterprise')
        T: 工作时长相关属性
        S: 技能/要求属性
        D: 数字素养属性
        W: 工资相关属性
        additional_attrs: 其他控制变量（如年龄、教育等）
    
    Examples:
        >>> labor = Agent(
        ...     agent_id=1,
        ...     agent_type='labor',
        ...     T=40.0,
        ...     S=75.0,
        ...     D=60.0,
        ...     W=4500.0
        ... )
    """
    agent_id: int
    agent_type: str  # 'labor' or 'enterprise'
    T: float  # 工作时长（劳动力期望 / 企业要求）
    S: float  # 技能（劳动力能力 / 企业要求）
    D: float  # 数字素养（劳动力水平 / 企业要求）
    W: float  # 工资（劳动力期望 / 企业提供）
    additional_attrs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """数据验证"""
        if self.agent_type not in ('labor', 'enterprise'):
            raise ValueError(f"agent_type必须是'labor'或'enterprise'，得到：{self.agent_type}")
        
        # 基本范围检查
        if self.T < 0:
            raise ValueError(f"工作时长不能为负：{self.T}")
        if not (0 <= self.S <= 100):
            raise ValueError(f"技能评分应在0-100之间：{self.S}")
        if not (0 <= self.D <= 100):
            raise ValueError(f"数字素养评分应在0-100之间：{self.D}")
        if self.W <= 0:
            raise ValueError(f"工资必须为正：{self.W}")
    
    def to_array(self) -> np.ndarray:
        """转换为NumPy数组（用于计算）"""
        return np.array([self.T, self.S, self.D, self.W])
    
    @classmethod
    def from_array(
        cls,
        agent_id: int,
        agent_type: str,
        arr: np.ndarray,
        additional_attrs: Optional[Dict[str, Any]] = None
    ) -> 'Agent':
        """从NumPy数组创建Agent"""
        return cls(
            agent_id=agent_id,
            agent_type=agent_type,
            T=float(arr[0]),
            S=float(arr[1]),
            D=float(arr[2]),
            W=float(arr[3]),
            additional_attrs=additional_attrs
        )
```

**设计要点**：
- ✅ 使用 `@dataclass` 自动生成方法
- ✅ `__post_init__` 中进行数据验证
- ✅ 提供 `to_array()` 和 `from_array()` 方便与NumPy互转
- ✅ `additional_attrs` 存储额外的控制变量（灵活性）

### 3.2 MatchingPair（匹配对）

```python
@dataclass
class MatchingPair:
    """
    匹配对
    
    表示一个劳动力与企业的匹配结果
    
    Attributes:
        labor_id: 劳动力ID
        enterprise_id: 企业ID
        matched: 是否匹配成功
        match_quality: 匹配质量得分（可选）
        metadata: 其他元数据（如匹配时的市场紧张度等）
    """
    labor_id: int
    enterprise_id: int
    matched: bool
    match_quality: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.match_quality is not None:
            if not (0 <= self.match_quality <= 1):
                raise ValueError(f"匹配质量应在0-1之间：{self.match_quality}")
```

### 3.3 SimulationState（模拟状态）

```python
@dataclass
class SimulationState:
    """
    模拟状态
    
    记录某一时刻的完整模拟状态
    
    Attributes:
        time_step: 当前时间步
        laborers: 劳动力列表
        enterprises: 企业列表
        matchings: 匹配对列表
        unemployment_rate: 失业率
        theta: 市场紧张度
        additional_metrics: 其他指标
    """
    time_step: int
    laborers: List[Agent]
    enterprises: List[Agent]
    matchings: List[MatchingPair]
    unemployment_rate: float
    theta: float  # 市场紧张度 V/U
    additional_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if not (0 <= self.unemployment_rate <= 1):
            raise ValueError(f"失业率应在0-1之间：{self.unemployment_rate}")
        if self.theta < 0:
            raise ValueError(f"市场紧张度不能为负：{self.theta}")
```

### 3.4 MFGEquilibrium（MFG均衡）

```python
@dataclass
class MFGEquilibrium:
    """
    MFG均衡结果
    
    存储平均场博弈求解后的均衡状态
    
    Attributes:
        value_function_U: 失业状态值函数
        value_function_E: 就业状态值函数
        policy_function: 最优努力策略函数
        distribution_U: 失业人口分布
        distribution_E: 就业人口分布
        theta: 均衡市场紧张度
        converged: 是否收敛
        iterations: 迭代次数
    """
    value_function_U: np.ndarray  # 形状: (grid_size_T, grid_size_S)
    value_function_E: np.ndarray
    policy_function: np.ndarray   # 最优努力水平 a*
    distribution_U: np.ndarray    # 人口分布
    distribution_E: np.ndarray
    theta: float
    converged: bool
    iterations: int
    
    def __post_init__(self):
        # 验证数组形状一致性
        shapes = [
            self.value_function_U.shape,
            self.value_function_E.shape,
            self.policy_function.shape,
            self.distribution_U.shape,
            self.distribution_E.shape
        ]
        if len(set(shapes)) != 1:
            raise ValueError(f"所有数组形状必须一致，得到：{shapes}")
```

---

## 4. 基础类设计

### 4.1 BaseGenerator（生成器基类）

**职责**：所有生成器（劳动力、企业）的抽象基类

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseGenerator(ABC):
    """
    生成器抽象基类
    
    定义所有生成器的标准接口
    
    子类必须实现：
    - fit(): 拟合参数
    - generate(): 生成虚拟个体
    - validate(): 验证生成质量
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.is_fitted = False
        self.fitted_params = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        拟合生成器参数
        
        Args:
            data: 训练数据
        
        Raises:
            ValueError: 数据格式不正确
        """
        pass
    
    @abstractmethod
    def generate(self, n_agents: int) -> pd.DataFrame:
        """
        生成虚拟个体
        
        Args:
            n_agents: 生成数量
        
        Returns:
            包含虚拟个体的DataFrame
        
        Raises:
            RuntimeError: 未先调用fit()
        """
        pass
    
    @abstractmethod
    def validate(self, agents: pd.DataFrame) -> bool:
        """
        验证生成的个体质量
        
        Args:
            agents: 待验证的个体数据
        
        Returns:
            是否通过验证
        """
        pass
    
    def save_params(self, filepath: str) -> None:
        """保存拟合后的参数"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()再保存参数")
        # 实现保存逻辑（默认实现，子类可覆盖）
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_params, f)
    
    def load_params(self, filepath: str) -> None:
        """加载已保存的参数"""
        import pickle
        with open(filepath, 'rb') as f:
            self.fitted_params = pickle.load(f)
        self.is_fitted = True
```

**设计要点**：
- ✅ 使用 ABC（抽象基类）确保子类实现必需方法
- ✅ 统一的生命周期：init → fit → generate → validate
- ✅ 状态管理：`is_fitted` 标志防止未拟合就生成
- ✅ 参数持久化：save/load 方法

### 4.2 BaseSimulator（模拟器基类）

**职责**：所有模拟器（匹配引擎、MFG求解器）的抽象基类

```python
class BaseSimulator(ABC):
    """
    模拟器抽象基类
    
    定义所有模拟器的标准接口
    
    子类必须实现：
    - setup(): 准备模拟
    - run(): 执行模拟
    - get_results(): 获取结果
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.is_setup = False
        self.is_complete = False
        self.results = None
    
    @abstractmethod
    def setup(self, **kwargs) -> None:
        """
        准备模拟
        
        加载数据、初始化参数等
        
        Raises:
            ValueError: 参数不正确
        """
        pass
    
    @abstractmethod
    def run(self) -> Any:
        """
        执行模拟
        
        Returns:
            模拟结果
        
        Raises:
            RuntimeError: 未先调用setup()
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        获取格式化的结果
        
        Returns:
            结果字典
        
        Raises:
            RuntimeError: 模拟未完成
        """
        pass
    
    def save_results(self, filepath: str) -> None:
        """保存模拟结果"""
        if not self.is_complete:
            raise RuntimeError("模拟未完成，无法保存结果")
        # 默认实现
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
```

---

## 5. 异常体系

### 5.1 异常层次结构

```python
# src/core/exceptions.py

class SimulationError(Exception):
    """
    模拟系统基础异常
    
    所有自定义异常的基类
    """
    pass


class DataValidationError(SimulationError):
    """
    数据验证失败
    
    用于：
    - 输入数据格式错误
    - 数据范围超出预期
    - 缺失必需字段
    """
    pass


class CopulaFittingError(SimulationError):
    """
    Copula拟合失败
    
    用于：
    - Copula模型不收敛
    - 参数估计失败
    """
    pass


class MatchingError(SimulationError):
    """
    匹配算法错误
    
    用于：
    - 匹配算法不收敛
    - 偏好矩阵计算失败
    """
    pass


class ConvergenceError(SimulationError):
    """
    MFG不收敛
    
    用于：
    - 贝尔曼方程迭代不收敛
    - KFE演化不稳定
    - MFE均衡求解失败
    """
    pass


class ConfigurationError(SimulationError):
    """
    配置错误
    
    用于：
    - 配置文件格式错误
    - 必需参数缺失
    - 参数值不合法
    """
    pass


class CalibrationError(SimulationError):
    """
    参数校准错误
    
    用于：
    - 遗传算法不收敛
    - 目标函数计算失败
    """
    pass
```

### 5.2 异常使用示例

```python
# 在数据验证中
if not (0 <= value <= 100):
    raise DataValidationError(
        f"技能评分应在0-100之间，得到：{value}"
    )

# 在Copula拟合中
try:
    params = copula.fit(data)
except Exception as e:
    raise CopulaFittingError(
        f"Copula拟合失败：{e}"
    ) from e

# 在MFG求解中
if not converged and iteration >= max_iterations:
    raise ConvergenceError(
        f"MFG在{max_iterations}次迭代后仍未收敛，"
        f"当前误差：{error:.6f}"
    )
```

---

## 6. 类型系统

### 6.1 类型别名定义

```python
# src/core/types.py

from typing import Dict, List, Tuple, Callable, Union
import numpy as np
from numpy.typing import NDArray

# 数据类型别名
AgentID = int
TimeStep = int
ParameterDict = Dict[str, float]

# NumPy数组类型
AgentFeatures = NDArray[np.float64]      # (n_agents, n_features)
PreferenceMatrix = NDArray[np.float64]   # (n_labor, n_enterprise)
ValueFunction = NDArray[np.float64]      # (grid_size_T, grid_size_S)
Distribution = NDArray[np.float64]       # (grid_size_T, grid_size_S)

# 函数类型
ObjectiveFunction = Callable[[np.ndarray], float]
MatchFunction = Callable[
    [np.ndarray, np.ndarray, float, float],  # (x, sigma, a, theta)
    float                                     # 返回匹配概率
]

# 配置类型
Config = Dict[str, Union[int, float, str, List, Dict]]
```

### 6.2 类型使用示例

```python
from src.core.types import (
    AgentFeatures,
    PreferenceMatrix,
    MatchFunction
)

def compute_preference(
    labor_features: AgentFeatures,
    enterprise_features: AgentFeatures
) -> PreferenceMatrix:
    """
    计算偏好矩阵
    
    Args:
        labor_features: 劳动力特征，形状 (n_labor, 4)
        enterprise_features: 企业特征，形状 (n_enterprise, 4)
    
    Returns:
        偏好矩阵，形状 (n_labor, n_enterprise)
    """
    # 实现...
    pass
```

---

## 7. API接口定义

### 7.1 核心接口总览

```python
# src/core/__init__.py

"""
Core模块公共接口
"""

# 数据结构
from .data_structures import (
    Agent,
    MatchingPair,
    SimulationState,
    MFGEquilibrium
)

# 基础类
from .base_generator import BaseGenerator
from .base_simulator import BaseSimulator

# 异常
from .exceptions import (
    SimulationError,
    DataValidationError,
    CopulaFittingError,
    MatchingError,
    ConvergenceError,
    ConfigurationError,
    CalibrationError
)

# 类型
from .types import (
    AgentID,
    TimeStep,
    ParameterDict,
    AgentFeatures,
    PreferenceMatrix,
    ValueFunction,
    Distribution,
    ObjectiveFunction,
    MatchFunction,
    Config
)

__all__ = [
    # 数据结构
    'Agent',
    'MatchingPair',
    'SimulationState',
    'MFGEquilibrium',
    # 基础类
    'BaseGenerator',
    'BaseSimulator',
    # 异常
    'SimulationError',
    'DataValidationError',
    'CopulaFittingError',
    'MatchingError',
    'ConvergenceError',
    'ConfigurationError',
    'CalibrationError',
    # 类型
    'AgentID',
    'TimeStep',
    'ParameterDict',
    'AgentFeatures',
    'PreferenceMatrix',
    'ValueFunction',
    'Distribution',
    'ObjectiveFunction',
    'MatchFunction',
    'Config',
]
```

---

## 8. 测试策略

### 8.1 测试覆盖目标

- **数据结构**：100%（所有验证逻辑）
- **基础类**：90%（抽象方法除外）
- **异常**：100%（确保正确抛出）
- **类型**：通过 mypy 检查

### 8.2 单元测试结构

```
tests/unit/core/
├── test_data_structures.py   # 测试Agent、MatchingPair等
├── test_base_generator.py    # 测试BaseGenerator
├── test_base_simulator.py    # 测试BaseSimulator
└── test_exceptions.py         # 测试异常
```

### 8.3 测试用例示例

```python
# tests/unit/core/test_data_structures.py

import pytest
import numpy as np
from src.core import Agent, DataValidationError

class TestAgent:
    """测试Agent数据结构"""
    
    def test_agent_creation_valid(self):
        """测试正常创建Agent"""
        agent = Agent(
            agent_id=1,
            agent_type='labor',
            T=40.0,
            S=75.0,
            D=60.0,
            W=4500.0
        )
        assert agent.agent_id == 1
        assert agent.agent_type == 'labor'
    
    def test_agent_invalid_type(self):
        """测试无效的agent_type"""
        with pytest.raises(ValueError, match="agent_type必须是"):
            Agent(
                agent_id=1,
                agent_type='invalid',
                T=40.0,
                S=75.0,
                D=60.0,
                W=4500.0
            )
    
    def test_agent_negative_work_hours(self):
        """测试负的工作时长"""
        with pytest.raises(ValueError, match="工作时长不能为负"):
            Agent(
                agent_id=1,
                agent_type='labor',
                T=-10.0,
                S=75.0,
                D=60.0,
                W=4500.0
            )
    
    def test_agent_to_array(self):
        """测试转换为NumPy数组"""
        agent = Agent(
            agent_id=1,
            agent_type='labor',
            T=40.0,
            S=75.0,
            D=60.0,
            W=4500.0
        )
        arr = agent.to_array()
        np.testing.assert_array_equal(
            arr,
            np.array([40.0, 75.0, 60.0, 4500.0])
        )
    
    def test_agent_from_array(self):
        """测试从NumPy数组创建"""
        arr = np.array([40.0, 75.0, 60.0, 4500.0])
        agent = Agent.from_array(
            agent_id=1,
            agent_type='labor',
            arr=arr
        )
        assert agent.T == 40.0
        assert agent.S == 75.0
        assert agent.D == 60.0
        assert agent.W == 4500.0
```

---

## 9. 实现计划

### 9.1 开发顺序

**第1步**：数据结构（1天）
- [ ] `data_structures.py` - Agent, MatchingPair, SimulationState, MFGEquilibrium
- [ ] 单元测试

**第2步**：异常体系（0.5天）
- [ ] `exceptions.py` - 所有异常类
- [ ] 单元测试

**第3步**：类型定义（0.5天）
- [ ] `types.py` - 类型别名
- [ ] mypy 检查

**第4步**：基础类（1天）
- [ ] `base_generator.py` - BaseGenerator
- [ ] `base_simulator.py` - BaseSimulator
- [ ] 单元测试

**第5步**：整合与文档（0.5天）
- [ ] `__init__.py` - 公共接口
- [ ] Docstring 完善
- [ ] 代码审查

**总计**：约 **3.5天**

### 9.2 验收标准

- [x] 所有文件符合 PEP8 规范
- [x] 所有公共接口有完整的 docstring
- [x] 所有数据结构有输入验证
- [x] 单元测试覆盖率 > 90%
- [x] mypy 类型检查通过
- [x] 所有测试通过

---

## 10. 依赖与环境

### 10.1 核心依赖

```python
# Core模块只依赖基础库
numpy >= 1.26.0
pandas >= 2.2.0
```

**不依赖**：
- scipy（业务层使用）
- numba（业务层使用）
- 其他业务模块

### 10.2 开发依赖

```python
# 测试
pytest >= 8.3.0
pytest-cov >= 5.0.0

# 类型检查
mypy >= 1.0.0
```

---

## 11. 后续模块接口预览

### 11.1 Population模块如何使用Core

```python
from src.core import BaseGenerator, Agent, DataValidationError
import pandas as pd

class LaborGenerator(BaseGenerator):
    """劳动力生成器（继承BaseGenerator）"""
    
    def fit(self, data: pd.DataFrame) -> None:
        # 验证数据
        if 'T' not in data.columns:
            raise DataValidationError("缺少'T'列")
        
        # Copula拟合...
        self.is_fitted = True
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit()")
        
        # 生成虚拟个体...
        # 返回DataFrame
        pass
    
    def validate(self, agents: pd.DataFrame) -> bool:
        # KS检验等...
        pass
```

### 11.2 Matching模块如何使用Core

```python
from src.core import (
    BaseSimulator,
    Agent,
    MatchingPair,
    SimulationState
)

class MatchingEngine(BaseSimulator):
    """匹配引擎（继承BaseSimulator）"""
    
    def setup(self, laborers: List[Agent], enterprises: List[Agent]) -> None:
        self.laborers = laborers
        self.enterprises = enterprises
        self.is_setup = True
    
    def run(self) -> SimulationState:
        if not self.is_setup:
            raise RuntimeError("必须先调用setup()")
        
        # GS算法...
        matchings = []  # List[MatchingPair]
        
        state = SimulationState(
            time_step=0,
            laborers=self.laborers,
            enterprises=self.enterprises,
            matchings=matchings,
            unemployment_rate=0.1,
            theta=1.0
        )
        
        self.is_complete = True
        return state
```

---

## 12. 风险与注意事项

### 12.1 潜在风险

1. **数据验证性能**
   - `__post_init__` 中的验证在大量创建对象时可能影响性能
   - 缓解：提供 `skip_validation=True` 选项（内部使用）

2. **NumPy数组转换开销**
   - `to_array()` / `from_array()` 频繁调用可能影响性能
   - 缓解：后续可考虑直接使用NumPy结构化数组

3. **类型注解与Numba冲突**
   - Numba函数不支持复杂类型注解
   - 缓解：Numba函数使用最小类型注解

### 12.2 未来优化方向

1. **性能优化**
   - 考虑使用 NumPy 结构化数组替代 dataclass（如果性能瓶颈）
   - 使用 `__slots__` 减少内存占用

2. **功能扩展**
   - 添加序列化/反序列化（JSON, Parquet）
   - 添加数据转换工具（DataFrame ↔ Agent列表）

---

## 13. 审阅清单

**请审阅以下设计决策**：

- [ ] **数据结构**：dataclass 方案是否合适？
- [ ] **验证逻辑**：`__post_init__` 中的验证是否足够/过度？
- [ ] **基础类接口**：`fit() / generate() / validate()` 三步骤是否合理？
- [ ] **异常粒度**：7个异常类是否足够？是否过细？
- [ ] **类型系统**：类型别名是否清晰？
- [ ] **实现计划**：3.5天是否合理？

**待确认问题**：

1. ~~**数字素养评分的Lognorm参数异常**~~ - ✅ 已解决（数据修正后为Beta分布）
2. **additional_attrs字段** - 是否需要预定义结构（如年龄、教育）？
3. **Agent的to_array顺序** - [T, S, D, W] 顺序是否固定？

---

**文档状态**: ✅ 完成，待用户审阅  
**预计实现时间**: 3.5天  
**下一步**: 用户审阅 → 实现代码 → 单元测试
