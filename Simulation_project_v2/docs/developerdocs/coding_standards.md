# 代码规范文档 (Coding Standards)

**项目**: Simulation_project_v2  
**版本**: 2.0  
**日期**: 2025-09-30

---

## 📋 目录

- [1. 总体原则](#1-总体原则)
- [2. PEP8规范](#2-pep8规范)
- [3. 命名约定](#3-命名约定)
- [4. 代码结构](#4-代码结构)
- [5. 文档字符串](#5-文档字符串)
- [6. 类型注解](#6-类型注解)
- [7. Numba优化规范](#7-numba优化规范)
- [8. 测试规范](#8-测试规范)
- [9. Git提交规范](#9-git提交规范)
- [10. 代码审查清单](#10-代码审查清单)

---

## 1. 总体原则

### 1.1 五大核心原则

根据项目规则，所有代码必须严格遵守以下原则：

1. **用户第一原则**: 代码实现严格符合研究计划，禁止自作主张修改
2. **需求确定原则**: 功能实现前必须确认需求
3. **模块化与简洁原则**: 模块清晰，无冗余代码
4. **代码易读性原则**: 中文注释齐全，结构清晰
5. **PEP8原则**: 严格遵守PEP8代码风格规范

### 1.2 "奥卡姆剃刀" 法则

> **"如无必要，勿增实体"**

- ❌ **禁止**: 不必要的 try-except 块
- ❌ **禁止**: 冗余的测试接口（测试脚本用完即删）
- ❌ **禁止**: 未使用的导入、变量、函数
- ✅ **鼓励**: 最简洁的实现方式

### 1.3 可读性 > 性能（除非在热点代码中）

**非热点代码**:
```python
# ✅ 推荐：可读性好
user_ages = [user.age for user in active_users if user.age >= 18]
```

**热点代码**:
```python
# ✅ 推荐：Numba优化，牺牲少许可读性换取性能
@njit(fastmath=True)
def compute_values_fast(data):
    result = np.empty(len(data))
    for i in range(len(data)):
        result[i] = data[i] * 2.0 + 1.0
    return result
```

---

## 2. PEP8规范

### 2.1 缩进与行宽

**严格执行**:
- ✅ **4个空格** 缩进（禁止Tab）
- ✅ 每行 **≤ 79** 个字符（代码）
- ✅ 每行 **≤ 72** 个字符（注释、文档字符串）

**示例**:
```python
# ❌ 错误：超过79字符
def very_long_function_name_with_many_parameters(parameter1, parameter2, parameter3, parameter4, parameter5):
    pass

# ✅ 正确：换行对齐
def very_long_function_name_with_many_parameters(
    parameter1: float,
    parameter2: float,
    parameter3: float,
    parameter4: float,
    parameter5: float
) -> float:
    pass
```

### 2.2 空行规范

- 顶级函数和类之间: **2个空行**
- 类内方法之间: **1个空行**
- 函数内逻辑块之间: **1个空行**

**示例**:
```python
class LaborGenerator:
    """劳动力生成器"""
    
    def __init__(self, config: Dict):
        """初始化"""
        self.config = config
        self.copula_engine = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """拟合Copula模型"""
        # 第一步：数据验证
        self._validate_data(data)
        
        # 第二步：边际分布估计
        marginals = self._estimate_marginals(data)
        
        # 第三步：Copula拟合
        self.copula_engine.fit(marginals)
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成虚拟个体"""
        pass


def standalone_function():
    """独立函数"""
    pass


class AnotherClass:
    """另一个类"""
    pass
```

### 2.3 空格使用

**运算符**:
```python
# ✅ 正确
a = b + c
result = (x * 2) + (y ** 3)

# ❌ 错误
a=b+c
result=(x*2)+(y**3)
```

**函数调用**:
```python
# ✅ 正确
func(a, b, c)
result = my_function(param1=10, param2=20)

# ❌ 错误
func( a,b,c )
result=my_function( param1 = 10 , param2 = 20 )
```

**列表、字典**:
```python
# ✅ 正确
my_list = [1, 2, 3, 4]
my_dict = {'key1': 'value1', 'key2': 'value2'}

# ❌ 错误
my_list=[1,2,3,4]
my_dict={'key1':'value1','key2':'value2'}
```

---

## 3. 命名约定

### 3.1 命名规范表

| 类型 | 命名风格 | 示例 | 说明 |
|------|---------|------|------|
| **模块/文件** | `lower_with_underscores` | `labor_generator.py` | 全小写+下划线 |
| **包/目录** | `lower_with_underscores` | `population/`, `matching/` | 全小写+下划线 |
| **类** | `CapWords` (驼峰) | `LaborGenerator`, `MFGSimulator` | 每个单词首字母大写 |
| **函数/方法** | `lower_with_underscores` | `compute_preference()` | 全小写+下划线 |
| **变量** | `lower_with_underscores` | `agent_count`, `theta_value` | 全小写+下划线 |
| **常量** | `UPPER_WITH_UNDERSCORES` | `MAX_ITERATIONS`, `DEFAULT_SEED` | 全大写+下划线 |
| **私有属性/方法** | `_leading_underscore` | `_validate_data()` | 单下划线前缀 |
| **魔法方法** | `__double_underscore__` | `__init__`, `__repr__` | 双下划线前后 |

### 3.2 变量命名最佳实践

**描述性命名**:
```python
# ❌ 不好：无意义的缩写
n, m, x, y = 10000, 5000, 0.5, 1.0

# ✅ 好：清晰的描述
n_labor = 10000
n_enterprise = 5000
match_probability = 0.5
market_tightness = 1.0
```

**数学符号映射**:
```python
# 对于数学公式中的符号，保持一致性
# 公式: λ(x, σ, a, θ)
@njit
def match_function(x, sigma, a, theta):
    """
    匹配函数 λ(x, σ, a, θ)
    
    Args:
        x: 状态变量
        sigma: 固定特征
        a: 努力水平
        theta: 市场紧张度
    """
    pass
```

**布尔变量**:
```python
# ✅ 使用 is_, has_, can_, should_ 前缀
is_fitted = False
has_converged = True
can_generate = True
should_validate = True

# ❌ 避免模糊的命名
fitted = False  # 不如 is_fitted 清晰
```

---

## 4. 代码结构

### 4.1 模块结构

每个Python文件遵循以下结构：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块名称 - 一行简短描述

详细描述（如果需要）

Author: 作者姓名
Date: 2025-09-30
"""

# 标准库导入
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 第三方库导入
import numpy as np
import pandas as pd
from numba import njit

# 本地模块导入
from src.core.base_generator import BaseGenerator
from src.utils.data_validation import validate_dataframe

# 模块级常量
DEFAULT_RANDOM_SEED = 42
MAX_ITERATIONS = 1000
TOLERANCE = 1e-6


class MyClass:
    """类的文档字符串"""
    pass


def my_function():
    """函数的文档字符串"""
    pass


@njit
def optimized_function():
    """Numba优化函数"""
    pass


if __name__ == "__main__":
    # 模块测试代码（仅用于快速测试，不要留在最终代码中）
    pass
```

### 4.2 类结构

```python
class LaborGenerator(BaseGenerator):
    """
    劳动力生成器
    
    基于Copula理论生成虚拟劳动力个体
    
    Attributes:
        config: 配置字典
        copula_engine: Copula引擎实例
        is_fitted: 是否已拟合
    
    Examples:
        >>> generator = LaborGenerator(config={})
        >>> generator.fit(training_data)
        >>> virtual_agents = generator.generate(10000)
    """
    
    # 1. 类属性/常量
    DEFAULT_N_SAMPLES = 10000
    
    # 2. 初始化方法
    def __init__(self, config: Dict):
        """
        初始化劳动力生成器
        
        Args:
            config: 配置字典，包含Copula类型、随机种子等
        """
        super().__init__(config)
        self.copula_engine = None
        self.is_fitted = False
    
    # 3. 公共方法（按重要性排序）
    def fit(self, data: pd.DataFrame) -> None:
        """拟合Copula模型"""
        pass
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成虚拟个体"""
        pass
    
    def validate(self, agents: pd.DataFrame) -> bool:
        """验证生成的个体质量"""
        pass
    
    # 4. 私有方法（按调用顺序）
    def _validate_data(self, data: pd.DataFrame) -> None:
        """验证输入数据"""
        pass
    
    def _estimate_marginals(self, data: pd.DataFrame) -> np.ndarray:
        """估计边际分布"""
        pass
    
    # 5. 特殊方法
    def __repr__(self) -> str:
        return f"LaborGenerator(fitted={self.is_fitted})"
```

---

## 5. 文档字符串

### 5.1 Docstring风格

**使用Google风格**（简洁清晰）：

```python
def compute_preference_matrix(
    labor_features: np.ndarray,
    enterprise_features: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算劳动力-企业偏好矩阵
    
    使用加权欧氏距离计算劳动力对每个企业的偏好分数，分数越高表示
    偏好越强。
    
    Args:
        labor_features: 劳动力特征矩阵，形状 (n_labor, n_features)
        enterprise_features: 企业特征矩阵，形状 (n_enterprise, n_features)
        weights: 特征权重，形状 (n_features,)，默认为None（等权重）
    
    Returns:
        偏好矩阵，形状 (n_labor, n_enterprise)
        
    Raises:
        ValueError: 如果特征维度不匹配
        
    Examples:
        >>> labor = np.array([[40, 0.6], [35, 0.7]])
        >>> enterprise = np.array([[38, 0.65], [42, 0.5]])
        >>> preferences = compute_preference_matrix(labor, enterprise)
        >>> preferences.shape
        (2, 2)
        
    Notes:
        - 该函数有Numba优化版本 `compute_preference_matrix_numba()`
        - 对于大规模计算（>1000个体），建议使用优化版本
    """
    if labor_features.shape[1] != enterprise_features.shape[1]:
        raise ValueError("特征维度不匹配")
    
    # 实现...
    pass
```

### 5.2 Docstring必需部分

| 函数类型 | 必需部分 | 可选部分 |
|---------|---------|---------|
| **公共函数** | 描述, Args, Returns | Raises, Examples, Notes |
| **私有函数** | 简短描述, Args | Returns |
| **简单函数** | 一行描述 | - |
| **Numba函数** | 描述, Args, Returns, Notes(性能) | - |

### 5.3 类的Docstring

```python
class MFGSimulator(BaseSimulator):
    """
    MFG模拟器（简化版）
    
    实现平均场博弈均衡求解，包含贝尔曼方程和KFE演化的迭代算法。
    采用离散状态空间和有限期简化策略。
    
    该模拟器核心算法包括：
    1. 贝尔曼方程值迭代
    2. KFE人口分布演化
    3. 市场紧张度更新
    4. 收敛性检验
    
    Attributes:
        config: 配置字典
        state_space: 离散状态空间 (StateSpace对象)
        bellman_solver: 贝尔曼方程求解器
        kfe_solver: KFE求解器
        match_function: 匹配函数 λ(x, σ, a, θ)
        is_converged: 是否已收敛
        equilibrium: MFG均衡（MFGEquilibrium对象）
    
    Config参数:
        grid_size: tuple
            状态空间网格大小，默认(50, 50)
        max_iterations: int
            最大迭代次数，默认500
        tolerance: float
            收敛容差，默认1e-6
        rho: float
            贴现因子，默认0.95
        kappa: float
            努力成本系数，默认1.0
    
    Examples:
        >>> config = load_config('config/default/mfg.yaml')
        >>> simulator = MFGSimulator(config)
        >>> simulator.setup(match_function=lambda_func)
        >>> equilibrium = simulator.run()
        >>> print(f"收敛: {simulator.is_converged}")
    
    References:
        - Lasry & Lions (2007). "Mean Field Games"
        - 研究计划文档: docs/研究计划.md
    """
    pass
```

---

## 6. 类型注解

### 6.1 何时使用类型注解

✅ **必须使用**:
- 所有公共函数/方法
- 函数参数
- 函数返回值

⚠️ **可选**:
- 简单的局部变量
- 私有方法（但推荐使用）

❌ **不使用**:
- Numba优化函数（可能导致编译问题）

### 6.2 类型注解示例

```python
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# 基本类型
def simple_function(x: int, y: float) -> float:
    return x + y

# 集合类型
def process_data(
    data: List[float],
    labels: Dict[str, int],
    optional_param: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    pass

# NumPy数组（推荐使用NDArray）
def compute_matrix(
    features: NDArray[np.float64]
) -> NDArray[np.float64]:
    pass

# Pandas DataFrame
def load_data(path: str) -> pd.DataFrame:
    pass

# 回调函数
def optimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]]
) -> np.ndarray:
    pass

# Union类型
def flexible_input(
    data: Union[np.ndarray, pd.DataFrame, List[float]]
) -> np.ndarray:
    pass

# 自定义类型
from src.core.data_structures import Agent, MatchingPair

def match_agents(
    laborers: List[Agent],
    enterprises: List[Agent]
) -> List[MatchingPair]:
    pass
```

### 6.3 类型别名

对于复杂类型，定义别名：

```python
# src/core/types.py
from typing import Dict, List, Tuple, Callable
import numpy as np

# 数据类型别名
AgentFeatures = np.ndarray       # (n_agents, n_features)
PreferenceMatrix = np.ndarray    # (n_labor, n_enterprise)
ParameterDict = Dict[str, float]

# 函数类型别名
ObjectiveFunction = Callable[[np.ndarray], float]
MatchFunction = Callable[[np.ndarray, np.ndarray, float, float], float]

# 使用别名
def compute_preferences(
    features: AgentFeatures
) -> PreferenceMatrix:
    pass
```

---

## 7. Numba优化规范

### 7.1 Numba使用原则

✅ **适合使用Numba**:
- 纯数值计算
- 循环密集
- NumPy数组操作
- 性能热点（通过profiling确认）

❌ **不适合使用Numba**:
- pandas DataFrame操作
- 字典/列表推导式
- 字符串处理
- 复杂对象操作

### 7.2 Numba装饰器配置

**标准配置**:
```python
from numba import njit, prange

# 纯数值计算（默认）
@njit
def simple_compute(x: np.ndarray) -> np.ndarray:
    pass

# 高性能优化（牺牲少许精度）
@njit(fastmath=True, cache=True)
def match_function(x, sigma, a, theta):
    """
    匹配函数（Numba优化）
    
    **性能**: 加速比 ~20x vs 纯Python
    **精度**: fastmath可能导致±1e-12误差（可接受）
    """
    pass

# 并行计算（适合大矩阵）
@njit(parallel=True)
def compute_preference_matrix_parallel(labor, enterprise):
    """
    偏好矩阵并行计算
    
    **性能**: 4核CPU加速比 ~15x
    **注意**: 数据量<1000时，并行开销可能大于收益
    """
    n_labor = labor.shape[0]
    n_enterprise = enterprise.shape[0]
    result = np.empty((n_labor, n_enterprise))
    
    for i in prange(n_labor):  # prange表示并行
        for j in range(n_enterprise):
            result[i, j] = compute_preference(labor[i], enterprise[j])
    
    return result
```

### 7.3 Numba代码规范

**类型一致性**:
```python
# ✅ 正确：类型一致
@njit
def good_function(x):
    result = 0.0  # float
    for i in range(len(x)):
        result += x[i]  # x[i]也是float
    return result

# ❌ 错误：类型混乱
@njit
def bad_function(x):
    result = 0  # int
    for i in range(len(x)):
        result += x[i]  # x[i]是float，类型推断失败
    return result
```

**避免Python对象**:
```python
# ❌ 错误：使用字典
@njit
def bad_with_dict(data):
    cache = {}  # Numba不支持字典
    # ...

# ✅ 正确：使用NumPy数组
@njit
def good_with_array(data):
    cache = np.empty(100)  # 使用数组
    # ...
```

### 7.4 性能验证

每个Numba优化函数必须有性能测试：

```python
# tests/benchmarks/test_numba_speedup.py
import time
import numpy as np
from src.modules.estimation.match_function import (
    match_function_python,
    match_function_numba
)

def test_match_function_speedup():
    """验证Numba加速效果"""
    x = np.array([40.0, 0.6, 0.5, 4000.0])
    sigma = np.array([25.0, 12.0])
    a = 0.5
    theta = 1.0
    
    # Python版本
    start = time.perf_counter()
    for _ in range(10000):
        result_py = match_function_python(x, sigma, a, theta)
    time_python = time.perf_counter() - start
    
    # Numba版本（预热）
    match_function_numba(x, sigma, a, theta)
    
    # Numba版本（计时）
    start = time.perf_counter()
    for _ in range(10000):
        result_nb = match_function_numba(x, sigma, a, theta)
    time_numba = time.perf_counter() - start
    
    speedup = time_python / time_numba
    
    print(f"Python: {time_python:.4f}s")
    print(f"Numba:  {time_numba:.4f}s")
    print(f"加速比: {speedup:.1f}x")
    
    assert speedup > 10, f"加速不足: {speedup:.1f}x < 10x"
    assert np.isclose(result_py, result_nb, atol=1e-10)
```

---

## 8. 测试规范

### 8.1 测试文件命名

```
tests/
├── unit/
│   ├── test_population.py          # 测试population模块
│   ├── test_matching.py
│   └── test_mfg.py
├── integration/
│   └── test_full_pipeline.py
└── benchmarks/
    └── test_numba_speedup.py
```

### 8.2 测试函数命名

```python
# tests/unit/test_population.py

def test_labor_generator_initialization():
    """测试劳动力生成器初始化"""
    pass

def test_labor_generator_fit_with_valid_data():
    """测试拟合：有效数据"""
    pass

def test_labor_generator_fit_with_invalid_data():
    """测试拟合：无效数据（应该抛出异常）"""
    pass

def test_labor_generator_generate_correct_size():
    """测试生成：输出规模正确"""
    pass

def test_labor_generator_generate_preserves_distribution():
    """测试生成：保持边际分布"""
    pass
```

### 8.3 测试结构

```python
import pytest
import numpy as np
import pandas as pd
from src.modules.population import LaborGenerator

@pytest.fixture
def sample_data():
    """测试数据fixture"""
    return pd.DataFrame({
        'T': np.random.uniform(0, 168, 1000),
        'S': np.random.uniform(0, 100, 1000),
        'D': np.random.uniform(0, 100, 1000),
        'W': np.random.uniform(2000, 8000, 1000)
    })

@pytest.fixture
def labor_generator():
    """生成器fixture"""
    config = {'random_seed': 42}
    return LaborGenerator(config)

def test_fit_and_generate(labor_generator, sample_data):
    """
    测试拟合后生成
    
    验证：
    1. 可以正常拟合
    2. 拟合后状态正确
    3. 可以生成指定数量的个体
    """
    # Arrange
    n_agents = 5000
    
    # Act
    labor_generator.fit(sample_data)
    virtual_agents = labor_generator.generate(n_agents)
    
    # Assert
    assert labor_generator.is_fitted
    assert len(virtual_agents) == n_agents
    assert set(virtual_agents.columns) == {'T', 'S', 'D', 'W'}

def test_generate_without_fit_raises_error(labor_generator):
    """测试未拟合就生成应该抛出异常"""
    with pytest.raises(RuntimeError, match="必须先调用fit"):
        labor_generator.generate(1000)

@pytest.mark.parametrize("n_agents", [100, 1000, 10000])
def test_generate_different_sizes(labor_generator, sample_data, n_agents):
    """参数化测试：不同生成规模"""
    labor_generator.fit(sample_data)
    virtual_agents = labor_generator.generate(n_agents)
    assert len(virtual_agents) == n_agents
```

### 8.4 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| 核心模块 (src/modules/) | > 90% |
| 工具函数 (src/utils/) | > 85% |
| 基类 (src/core/) | > 80% |
| 总体 | > 85% |

---

## 9. Git提交规范

### 9.1 Conventional Commits

所有commit message遵循以下格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型 (type)**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `refactor`: 代码重构（不改变功能）
- `test`: 添加测试
- `perf`: 性能优化
- `chore`: 构建/工具链变更

**范围 (scope)** (可选):
- `population`, `matching`, `estimation`, `mfg`, `calibration`
- `core`, `utils`, `config`, `docs`

**示例**:
```
feat(population): 实现Copula引擎核心算法

- 添加GaussianMultivariate Copula支持
- 实现伪观测值转换
- 添加边际分布估计器

Closes #12
```

```
fix(matching): 修复Gale-Shapley算法稳定性检验bug

在特殊情况下（劳动力数量>企业数量×2），稳定性检验逻辑错误。
现已修复。

Fixes #25
```

```
docs(developer): 完善API文档和Numba优化指南

- 更新architecture.md
- 添加Numba最佳实践
- 补充性能基准测试说明
```

### 9.2 提交粒度

- ✅ **推荐**: 每个逻辑功能一个commit
- ❌ **避免**: 一次commit包含多个不相关的修改
- ❌ **避免**: 频繁的"修复typo"类commit（应该合并到主commit）

---

## 10. 代码审查清单

### 10.1 提交前自查

**功能性**:
- [ ] 代码实现符合需求
- [ ] 边界条件处理正确
- [ ] 异常处理完整

**代码质量**:
- [ ] 符合PEP8规范（行宽、空格、命名）
- [ ] 无未使用的导入/变量
- [ ] 无print/pdb调试语句
- [ ] 无TODO/FIXME（或已记录在issue中）

**文档**:
- [ ] 所有公共函数有文档字符串
- [ ] 复杂逻辑有注释说明
- [ ] 更新相关文档（README、API文档）

**测试**:
- [ ] 添加单元测试
- [ ] 所有测试通过 (`pytest`)
- [ ] 覆盖率达标

**性能**:
- [ ] Numba优化函数有性能测试
- [ ] 无明显性能瓶颈（大数据量测试）

**Git**:
- [ ] Commit message符合规范
- [ ] 没有敏感信息（密码、密钥）

### 10.2 Code Review清单（Reviewer使用）

**可读性**:
- [ ] 代码逻辑清晰易懂
- [ ] 变量命名有意义
- [ ] 复杂算法有注释

**正确性**:
- [ ] 算法实现正确
- [ ] 边界条件处理合理
- [ ] 数值稳定性良好

**设计**:
- [ ] 模块职责单一
- [ ] 接口设计合理
- [ ] 代码复用性好

**测试**:
- [ ] 测试覆盖关键路径
- [ ] 测试用例有代表性
- [ ] Mock使用合理

---

## 11. 附录

### 11.1 常用工具配置

**Black（格式化工具）**:
```toml
# pyproject.toml
[tool.black]
line-length = 79  # PEP8标准
target-version = ['py312']
```

**Flake8（代码检查）**:
```ini
# .flake8
[flake8]
max-line-length = 79
extend-ignore = E203, W503
per-file-ignores =
    __init__.py:F401
```

### 11.2 IDE配置建议

**VSCode**:
```json
// settings.json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.rulers": [79],
    "editor.formatOnSave": true
}
```

---

**文档维护**: 随项目演进更新  
**最后更新**: 2025-09-30  
**审阅者**: 全体开发人员
