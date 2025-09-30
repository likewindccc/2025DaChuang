# 技术选型文档 (Tech Stack)

**项目**: Simulation_project_v2  
**版本**: 2.0  
**日期**: 2025-09-30

---

## 📋 目录

- [1. 总体技术栈](#1-总体技术栈)
- [2. 核心依赖库](#2-核心依赖库)
- [3. 开发工具](#3-开发工具)
- [4. 部署与打包](#4-部署与打包)
- [5. 版本兼容性](#5-版本兼容性)

---

## 1. 总体技术栈

### 1.1 技术栈概览

```
┌─────────────────────────────────────────────┐
│           应用层                             │
│  Python 3.12.5 + 命令行界面                  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           科学计算层                         │
│  NumPy + SciPy + Pandas                     │
│  + Numba (JIT加速)                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           专业算法层                         │
│  Copulas + DEAP + Statsmodels               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           可视化 & 开发工具层                │
│  Matplotlib + Plotly + Pytest + Sphinx      │
└─────────────────────────────────────────────┘
```

### 1.2 技术决策原则

1. **稳定性优先**: 使用成熟、广泛验证的库
2. **性能关键**: Numba强制优化热点代码
3. **学术严谨**: 算法实现符合研究计划
4. **可维护性**: 代码清晰，文档完善
5. **依赖复用**: 继承旧版 `requirements.txt`

---

## 2. 核心依赖库

### 2.1 基础科学计算

#### NumPy (1.26.4)
**用途**: 数组计算、线性代数  
**关键功能**:
- 多维数组操作
- 矩阵运算（协方差矩阵、Cholesky分解）
- 随机数生成

**使用场景**:
```python
# 状态空间离散化
state_grid = np.meshgrid(
    np.linspace(0, 1, 50),  # T维度
    np.linspace(0, 1, 50)   # S维度
)

# 协方差矩阵计算
cov_matrix = np.cov(data, rowvar=False)
```

#### Pandas (2.2.3)
**用途**: 数据处理、CSV读写  
**关键功能**:
- DataFrame数据结构
- 数据清洗与转换
- 描述性统计

**使用场景**:
```python
# 虚拟劳动力数据
labor_df = pd.DataFrame({
    'T': work_hours,
    'S': skill_scores,
    'D': digital_literacy,
    'W': expected_wage
})

# 数据验证
assert labor_df['T'].between(0, 168).all()  # 每周最多168小时
```

#### SciPy (1.14.1)
**用途**: 统计分布、优化算法  
**关键功能**:
- `scipy.stats`: 概率分布（Beta, Normal等）
- `scipy.optimize`: 数值优化
- `scipy.linalg`: 线性代数高级功能

**使用场景**:
```python
# Beta分布拟合
from scipy import stats
params = stats.beta.fit(data)

# Anderson-Darling检验
ad_stat, critical_values, significance_level = stats.anderson(data, 'norm')
```

---

### 2.2 性能优化

#### Numba (0.59.0+) ⭐ 强制使用
**用途**: JIT编译加速Python代码  
**加速目标**: 10倍以上  
**强制优化的模块**:
1. 匹配函数 λ(x, σ, a, θ)
2. 偏好矩阵计算
3. 贝尔曼方程迭代
4. KFE演化步骤

**使用示例**:
```python
from numba import njit, prange

@njit(fastmath=True, cache=True)
def match_function(x, sigma, a, theta):
    """
    匹配概率函数（Numba加速版本）
    加速比: ~20x vs 纯Python
    """
    logit = (
        delta_0 + 
        delta_x[0] * x[0] + delta_x[1] * x[1] + 
        delta_x[2] * x[2] + delta_x[3] * x[3] + 
        delta_a * a + 
        delta_theta * np.log(theta)
    )
    return 1.0 / (1.0 + np.exp(-logit))

@njit(parallel=True)
def compute_preference_matrix(labor_features, enterprise_features):
    """
    并行计算偏好矩阵
    加速比: ~15x vs 纯Python（4核CPU）
    """
    n_labor = labor_features.shape[0]
    n_enterprise = enterprise_features.shape[0]
    preferences = np.zeros((n_labor, n_enterprise))
    
    for i in prange(n_labor):
        for j in range(n_enterprise):
            preferences[i, j] = compute_preference(
                labor_features[i], 
                enterprise_features[j]
            )
    
    return preferences
```

**Numba使用规范**:
- ✅ **DO**: 纯数值计算、循环、NumPy数组
- ❌ **DON'T**: pandas DataFrame、字典、列表推导式
- ⚙️ **参数**: `fastmath=True`（牺牲少许精度换速度）, `cache=True`（缓存编译结果）

**性能基准**:
```python
# tests/benchmarks/test_numba_speedup.py
def test_match_function_speedup():
    """测试Numba加速效果"""
    # 纯Python版本
    time_python = benchmark(match_function_python, args)
    
    # Numba版本
    time_numba = benchmark(match_function_numba, args)
    
    speedup = time_python / time_numba
    assert speedup > 10, f"加速比不足: {speedup}x < 10x"
```

---

### 2.3 专业算法库

#### Copulas (0.12.3)
**用途**: 解决变量非独立性问题  
**关键功能**:
- `GaussianMultivariate`: 高斯Copula
- 边际分布与依赖结构分离

**使用场景**:
```python
from copulas.multivariate import GaussianMultivariate

# 拟合Copula
copula = GaussianMultivariate()
copula.fit(uniform_data)  # 伪观测值

# 生成相关样本
virtual_agents = copula.sample(10000)
```

**已知限制**:
- VineCopula存在`NotImplementedError`
- 当前只使用GaussianMultivariate

#### DEAP (1.4.1+)
**用途**: 遗传算法参数校准  
**关键功能**:
- 遗传算法框架
- 多种选择、交叉、变异算子
- 并行评估支持

**使用场景**:
```python
from deap import base, creator, tools, algorithms

# 定义适应度和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 遗传算法工具箱
toolbox = base.Toolbox()
toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
population = toolbox.population(n=100)
algorithms.eaSimple(
    population, toolbox,
    cxpb=0.7, mutpb=0.2,
    ngen=50, verbose=True
)
```

**校准配置**:
```yaml
# config/default/calibration.yaml
genetic_algorithm:
  population_size: 100
  n_generations: 50
  crossover_prob: 0.7
  mutation_prob: 0.2
  tournament_size: 3
  
  # 并行评估
  n_processes: 4  # CPU核心数
```

#### Statsmodels (0.14.4)
**用途**: Logit回归、统计建模  
**关键功能**:
- `Logit`: 二元Logit回归
- 模型诊断与评估

**使用场景**:
```python
import statsmodels.api as sm

# Logit回归拟合匹配函数
X = matching_data[['T', 'S', 'D', 'W', 'effort', 'theta']]
y = matching_data['matched']  # 0/1

X = sm.add_constant(X)  # 添加截距项
model = sm.Logit(y, X)
result = model.fit()

print(result.summary())  # 完整的统计报告
```

---

### 2.4 可视化

#### Matplotlib (3.9.2)
**用途**: 基础绘图、学术图表  
**关键功能**:
- 折线图、散点图、热力图
- 子图布局
- LaTeX公式支持

**配置**:
```python
# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 学术风格
plt.style.use('seaborn-v0_8-paper')
```

#### Seaborn (0.13.2)
**用途**: 统计可视化  
**关键功能**:
- 相关性热力图
- 分布图
- 配对图

**使用场景**:
```python
import seaborn as sns

# 相关性热力图
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1
)
```

#### Plotly (5.24.1)
**用途**: 交互式图表  
**关键功能**:
- 3D图
- 动态图表
- HTML导出

**使用场景**:
```python
import plotly.graph_objects as go

# 3D状态空间可视化
fig = go.Figure(data=[go.Scatter3d(
    x=data['T'],
    y=data['S'],
    z=data['W'],
    mode='markers'
)])
fig.write_html("results/figures/state_space_3d.html")
```

---

### 2.5 测试与质量保证

#### Pytest (8.3.3)
**用途**: 单元测试、集成测试  
**关键功能**:
- 测试发现与运行
- Fixtures（测试夹具）
- 参数化测试

**配置**:
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers

# 性能测试
markers =
    slow: 标记慢速测试
    benchmark: 性能基准测试
```

**使用示例**:
```python
# tests/unit/test_population.py
import pytest
from src.modules.population import LaborGenerator

@pytest.fixture
def sample_data():
    """测试数据fixture"""
    return pd.read_csv("data/input/labor_survey.csv")

def test_labor_generator_fit(sample_data):
    """测试劳动力生成器拟合"""
    generator = LaborGenerator(config={})
    generator.fit(sample_data)
    
    assert generator.is_fitted
    assert generator.copula_engine.best_copula is not None

@pytest.mark.parametrize("n_agents", [100, 1000, 10000])
def test_labor_generator_scale(sample_data, n_agents):
    """参数化测试：不同规模"""
    generator = LaborGenerator(config={})
    generator.fit(sample_data)
    
    virtual_agents = generator.generate(n_agents)
    assert len(virtual_agents) == n_agents
```

#### Pytest-Cov (5.0.0)
**用途**: 测试覆盖率  
**使用**:
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

**目标覆盖率**:
- 核心模块: > 90%
- 工具函数: > 80%
- 总体: > 85%

#### Pytest-Benchmark (4.0.0+)
**用途**: 性能基准测试  
**使用**:
```python
def test_match_function_performance(benchmark):
    """基准测试：匹配函数性能"""
    result = benchmark(
        match_function,
        x=np.array([40, 0.6, 0.5, 4000]),
        sigma=np.array([25, 12]),
        a=0.5,
        theta=1.0
    )
    assert result >= 0 and result <= 1
```

---

### 2.6 文档生成

#### Sphinx (8.0.2)
**用途**: 自动生成API文档  
**配置**:
```python
# docs/conf.py
extensions = [
    'sphinx.ext.autodoc',      # 自动从docstring生成文档
    'sphinx.ext.napoleon',     # 支持Google/NumPy风格docstring
    'sphinx.ext.viewcode',     # 源代码链接
    'sphinx.ext.mathjax',      # 数学公式
]

html_theme = 'sphinx_rtd_theme'  # ReadTheDocs主题
```

**生成命令**:
```bash
cd docs
sphinx-apidoc -o api ../src
make html
```

---

## 3. 开发工具

### 3.1 代码质量工具

#### Black (可选)
**用途**: 代码格式化  
**配置**:
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
```

#### Flake8 (可选)
**用途**: 代码风格检查  
**配置**:
```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv
```

### 3.2 依赖管理

#### 继承旧版依赖
```bash
# 复用已有依赖
cp ../requirements.txt ./requirements.txt
```

**依赖结构** (未来优化):
```
requirements/
├── base.txt        # 核心依赖
├── dev.txt         # 开发工具
└── docs.txt        # 文档生成
```

---

## 4. 部署与打包

### 4.1 虚拟环境

**Python版本**: 3.12.5  
**虚拟环境路径**: `D:\Python\venv\`

**激活命令**:
```powershell
D:\Python\.venv\Scripts\Activate.ps1
```

### 4.2 GUI (后期)

**框架选择**: PyQt6  
**当前阶段**: 命令行优先  
**GUI开发时间**: Phase 6+

### 4.3 打包 (后期)

**工具**: PyInstaller  
**目标**: 独立exe文件  
**打包命令** (未来):
```bash
pyinstaller --onefile --windowed src/main.py
```

---

## 5. 版本兼容性

### 5.1 Python版本要求

| Python版本 | 支持状态 | 说明 |
|-----------|---------|------|
| 3.12.x    | ✅ 推荐  | 开发与测试版本 |
| 3.11.x    | ✅ 兼容  | 应该可运行 |
| 3.10.x    | ⚠️ 可能  | 未测试 |
| < 3.10    | ❌ 不支持| Numba/类型注解限制 |

### 5.2 操作系统兼容性

| OS | 支持状态 | 说明 |
|----|---------|------|
| Windows 10/11 | ✅ 推荐 | 主要开发环境 |
| Linux | ✅ 兼容 | 服务器部署 |
| macOS | ⚠️ 未测试 | 理论兼容 |

### 5.3 关键库版本锁定

**必须锁定版本**:
```txt
numba>=0.59.0        # Numba核心功能
numpy>=1.26.0,<2.0   # 避免Numba兼容问题
```

**可以浮动版本**:
```txt
pandas>=2.2.0
scipy>=1.14.0
matplotlib>=3.9.0
```

---

## 6. 性能目标

### 6.1 Numba加速目标

| 函数 | Python基准 | Numba目标 | 加速比 |
|------|-----------|----------|--------|
| match_function | 10 ms | < 0.5 ms | > 20x |
| preference_matrix | 5 s | < 0.3 s | > 15x |
| bellman_iteration | 100 ms | < 10 ms | > 10x |

### 6.2 大规模计算目标

**测试规模**: 10,000 劳动力 × 5,000 企业

| 操作 | 时间目标 | 内存目标 |
|------|---------|---------|
| 虚拟个体生成 | < 5 s | < 1 GB |
| 单轮匹配 | < 30 s | < 2 GB |
| 完整MFG模拟 | < 10 min | < 4 GB |
| 校准（50代） | < 8 hours | < 8 GB |

---

## 7. 依赖升级策略

### 7.1 何时升级

- 🔴 **安全漏洞**: 立即升级
- 🟡 **重要新功能**: 评估后升级
- 🟢 **小版本更新**: 定期升级

### 7.2 升级流程

1. 检查更新: `pip list --outdated`
2. 阅读 Change Log
3. 在分支中测试升级
4. 运行完整测试套件
5. 合并到主分支

---

## 8. 附录

### 8.1 完整依赖列表

参见: `requirements.txt` (继承自旧版)

核心依赖:
```txt
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1
statsmodels==0.14.4
scikit-learn==1.5.2
numba>=0.59.0
copulas==0.12.3

matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1

pytest==8.3.3
pytest-cov==5.0.0

pyyaml==6.0.2
pydantic==2.9.2

# 遗传算法
deap>=1.4.1

# 文档
sphinx==8.0.2
```

### 8.2 相关资源

- [Numba文档](https://numba.pydata.org/numba-doc/latest/)
- [DEAP文档](https://deap.readthedocs.io/)
- [Copulas文档](https://sdv.dev/Copulas/)
- [Pytest文档](https://docs.pytest.org/)

---

**文档维护**: 随依赖变更更新  
**最后更新**: 2025-09-30  
**审阅者**: 技术负责人
