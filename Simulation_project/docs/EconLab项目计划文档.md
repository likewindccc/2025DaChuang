# EconLab经济学实验室项目计划文档

**版本:** 1.0  
**日期:** 2025年9月26日  
**项目名称:** 农村女性就业市场MFG模拟系统 (EconLab)  
**目标:** 构建专业级经济学仿真实验平台

---

## 1. 项目概述

### 1.1 项目定位
EconLab是一个融合平均场博弈（MFG）与基于主体建模（ABM）的经济学实验平台，专门用于农村女性就业市场的动态仿真分析。该平台将为经济学研究者、政策制定者提供强大的建模工具和直观的分析界面。

### 1.2 核心价值
- **理论突破**: 将MFG理论应用于劳动经济学研究
- **政策工具**: 为农村就业政策提供量化评估平台
- **教学资源**: 为经济学教育提供可视化仿真工具
- **研究平台**: 支持复杂经济现象的建模分析

### 1.3 技术特色
- **高性能计算**: numba + Cython混合优化
- **智能校准**: 遗传算法自动参数优化
- **专业界面**: PyQt6构建的现代化GUI
- **独立部署**: 一键安装的exe应用程序

---

## 2. 总体架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    EconLab经济学实验室                        │
├─────────────────────────────────────────────────────────────┤
│                     用户界面层 (GUI Layer)                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │  主控制面板  │  参数设置  │  实时监控  │  结果分析  │   │
│  │ Main Panel │ Parameters │ Monitor   │ Analysis  │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   应用逻辑层 (Application Layer)              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │  任务调度器  │  数据管理  │  配置管理  │  报告生成  │   │
│  │ Task Scheduler│Data Manager│Config Mgr │Report Gen │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   计算引擎层 (Compute Engine Layer)           │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │ 主体生成器  │  匹配引擎  │ 函数估计器 │  MFG求解器 │   │
│  │Population   │ Matching   │ Function   │    MFG     │   │
│  │ Generator   │ Engine     │ Estimator  │ Simulator  │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
│                           │                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │ 实验控制器  │  校准引擎  │  可视化引擎 │  性能监控  │   │
│  │Experiment   │Calibration │Visualization│Performance │   │
│  │Controller   │Engine      │Engine       │Monitor     │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   数据访问层 (Data Access Layer)             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │   配置文件   │   输入数据   │   输出结果   │   日志系统   │   │
│  │   Config    │ Input Data │ Output     │  Logging   │   │
│  │   Files     │           │ Results    │   System   │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈架构

```python
┌─── 用户界面 ───┐    ┌─── 计算优化 ───┐    ┌─── 数据处理 ───┐
│ PyQt6         │    │ numba (JIT)   │    │ pandas        │
│ matplotlib    │    │ Cython (AOT)  │    │ numpy         │
│ plotly        │    │ multiprocessing│    │ scipy         │
└───────────────┘    └───────────────┘    └───────────────┘

┌─── 算法工具 ───┐    ┌─── 系统工具 ───┐    ┌─── 部署工具 ───┐
│ DEAP (GA)     │    │ PyYAML        │    │ PyInstaller   │
│ scikit-learn  │    │ logging       │    │ UPX           │
│ statsmodels   │    │ json          │    │ NSIS          │
└───────────────┘    └───────────────┘    └───────────────┘
```

### 2.3 目录结构设计

```
EconLab/
├── econlab/                      # 主包目录
│   ├── __init__.py
│   ├── main.py                   # 程序入口
│   ├── config.py                 # 全局配置
│   ├── core/                     # 核心计算模块
│   │   ├── __init__.py
│   │   ├── population_generator/ # Module 1: 主体生成器
│   │   │   ├── __init__.py
│   │   │   ├── agent_generator.py
│   │   │   ├── copula_engine.py
│   │   │   └── enterprise_generator.py
│   │   ├── matching_engine/      # Module 2: 匹配引擎
│   │   │   ├── __init__.py
│   │   │   ├── gale_shapley.py
│   │   │   ├── preference_calculator.py
│   │   │   └── market_simulator.py
│   │   ├── function_estimator/   # Module 3: 函数估计器
│   │   │   ├── __init__.py
│   │   │   ├── logit_estimator.py
│   │   │   └── model_validator.py
│   │   ├── mfg_simulator/        # Module 4: MFG模拟器
│   │   │   ├── __init__.py
│   │   │   ├── bellman_solver.py
│   │   │   ├── kfe_solver.py
│   │   │   └── state_discretizer.py
│   │   └── experiment_controller/ # Module 5: 实验控制器
│   │       ├── __init__.py
│   │       ├── calibration_engine.py
│   │       ├── policy_analyzer.py
│   │       └── result_processor.py
│   ├── optimization/             # 优化算法模块
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py
│   │   ├── objective_functions.py
│   │   └── parameter_bounds.py
│   ├── gui/                      # 图形界面模块
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── parameter_panel.py
│   │   │   ├── monitor_panel.py
│   │   │   ├── result_panel.py
│   │   │   └── progress_dialog.py
│   │   └── visualization/
│   │       ├── __init__.py
│   │       ├── real_time_plotter.py
│   │       ├── static_charts.py
│   │       └── animation_engine.py
│   ├── data/                     # 数据管理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   └── data_exporter.py
│   └── utils/                    # 工具函数模块
│       ├── __init__.py
│       ├── decorators.py
│       ├── performance.py
│       └── logging_config.py
├── config/                       # 配置文件目录
│   ├── default_config.yaml
│   ├── calibration_bounds.yaml
│   └── gui_settings.yaml
├── data/                         # 数据目录
│   ├── input/
│   │   ├── cleaned_data.csv
│   │   └── validation_data.csv
│   ├── output/
│   └── cache/
├── results/                      # 结果输出目录
│   ├── reports/
│   ├── figures/
│   └── logs/
├── tests/                        # 测试目录
│   ├── unit_tests/
│   ├── integration_tests/
│   └── performance_tests/
├── docs/                         # 文档目录
│   ├── api/
│   ├── user_guide/
│   └── technical/
├── scripts/                      # 脚本目录
│   ├── build.py
│   ├── install_deps.py
│   └── run_tests.py
├── requirements.txt              # 依赖列表
├── setup.py                      # 安装脚本
├── README.md                     # 项目说明
└── EconLab.spec                  # PyInstaller配置
```

---

## 3. 核心模块详细设计

### 3.1 Module 1: PopulationGenerator (主体生成器)

#### 3.1.1 模块职责
- 基于Copula模型生成虚拟劳动力主体池
- 基于多元正态分布生成企业主体池
- 提供可配置的主体规模和属性分布

#### 3.1.2 核心类设计

```python
class AgentGenerator:
    """主体生成器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_state = None
    
    @abstractmethod
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成指定数量的主体"""
        pass
    
    @abstractmethod
    def validate_agents(self, agents: pd.DataFrame) -> bool:
        """验证生成的主体数据有效性"""
        pass

class LaborAgentGenerator(AgentGenerator):
    """劳动力主体生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.copula_model = None
        self.marginal_distributions = {}
    
    @numba.jit(nopython=True)
    def _generate_correlated_samples(self, n_samples: int) -> np.ndarray:
        """使用numba优化的相关样本生成"""
        pass
    
    def fit_copula_model(self, data: pd.DataFrame) -> None:
        """拟合Copula模型"""
        pass
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成劳动力主体"""
        pass

class EnterpriseGenerator(AgentGenerator):
    """企业主体生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mean_vector = None      # 4维均值向量
        self.covariance_matrix = None # 4×4协方差矩阵
    
    @numba.jit(nopython=True) 
    def _multivariate_normal_sample(self, n_samples: int) -> np.ndarray:
        """numba优化的多元正态分布采样"""
        pass
    
    def calibrate_distribution(self, parameters: Dict[str, float]) -> None:
        """校准多元正态分布参数"""
        pass
    
    def generate(self, n_agents: int) -> pd.DataFrame:
        """生成企业主体"""
        pass
```

#### 3.1.3 性能优化策略

```python
# 关键计算函数使用numba JIT编译
@numba.jit(nopython=True, parallel=True)
def fast_copula_transform(data: np.ndarray, marginal_cdfs: List) -> np.ndarray:
    """快速Copula变换"""
    result = np.empty_like(data)
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            result[i, j] = marginal_cdfs[j](data[i, j])
    return result

# 内存优化的批量生成
class BatchGenerator:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def generate_in_batches(self, total_agents: int) -> Iterator[pd.DataFrame]:
        """分批生成主体，避免内存溢出"""
        for i in range(0, total_agents, self.batch_size):
            batch_size = min(self.batch_size, total_agents - i)
            yield self._generate_batch(batch_size)
```

### 3.2 Module 2: MatchingEngine (匹配引擎)

#### 3.2.1 模块职责
- 实现Gale-Shapley稳定匹配算法
- 计算双边偏好排序
- 支持多θ值场景的批量模拟

#### 3.2.2 核心类设计

```python
class PreferenceCalculator:
    """偏好计算器"""
    
    def __init__(self, jobseeker_params: Dict, employer_params: Dict):
        self.gamma = jobseeker_params  # 求职者偏好参数
        self.beta = employer_params    # 企业偏好参数
    
    @numba.jit(nopython=True)
    def _compute_jobseeker_utility(self, 
                                  jobseeker_attrs: np.ndarray,
                                  job_attrs: np.ndarray) -> float:
        """计算求职者对岗位的效用"""
        # γ₀ - γ₁×T - γ₂×max(0,S_req-S_own) - γ₃×max(0,D_req-D_own) + γ₄×W
        pass
    
    @numba.jit(nopython=True)
    def _compute_employer_utility(self, 
                                 jobseeker_attrs: np.ndarray,
                                 job_attrs: np.ndarray) -> float:
        """计算企业对求职者的效用"""
        # β₀ + β₁×T + β₂×S + β₃×D - β₄×W
        pass
    
    @numba.jit(nopython=True, parallel=True)
    def compute_preference_matrix(self, 
                                 jobseekers: np.ndarray,
                                 enterprises: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """并行计算偏好矩阵"""
        pass

class GaleShapleyMatcher:
    """Gale-Shapley稳定匹配算法实现"""
    
    def __init__(self):
        self.preference_calculator = None
        self.matching_history = []
    
    @numba.jit(nopython=True)
    def _stable_matching_core(self, 
                             men_preferences: np.ndarray,
                             women_preferences: np.ndarray) -> np.ndarray:
        """numba优化的稳定匹配核心算法"""
        pass
    
    def match(self, 
             jobseekers: pd.DataFrame, 
             enterprises: pd.DataFrame,
             theta: float) -> pd.DataFrame:
        """执行匹配算法"""
        pass
    
    def validate_stability(self, 
                          matching_result: pd.DataFrame) -> bool:
        """验证匹配结果的稳定性"""
        pass

class MarketSimulator:
    """市场模拟器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matcher = GaleShapleyMatcher()
        self.preference_calculator = PreferenceCalculator(
            config['jobseeker_params'], 
            config['employer_params']
        )
    
    def simulate_multiple_scenarios(self, 
                                   jobseekers: pd.DataFrame,
                                   enterprises: pd.DataFrame,
                                   theta_values: List[float]) -> pd.DataFrame:
        """模拟多个θ值场景"""
        results = []
        for theta in theta_values:
            # 根据θ值调整参与匹配的主体数量
            active_enterprises = self._sample_enterprises(enterprises, theta)
            matching_result = self.matcher.match(jobseekers, active_enterprises, theta)
            results.append(matching_result)
        
        return pd.concat(results, ignore_index=True)
    
    def _sample_enterprises(self, 
                           enterprises: pd.DataFrame, 
                           theta: float) -> pd.DataFrame:
        """根据θ值采样企业"""
        pass
```

#### 3.2.3 算法优化

```python
# 内存高效的偏好矩阵计算
class SparsePreferenceMatrix:
    """稀疏偏好矩阵，节省内存"""
    
    def __init__(self, n_jobseekers: int, n_enterprises: int, top_k: int = 100):
        self.n_jobseekers = n_jobseekers
        self.n_enterprises = n_enterprises
        self.top_k = top_k  # 只保存每个主体的top-k偏好
        
    @numba.jit(nopython=True)
    def compute_top_k_preferences(self, utility_matrix: np.ndarray) -> np.ndarray:
        """只计算top-k偏好，减少内存使用"""
        pass

# 并行化的多场景模拟
from joblib import Parallel, delayed

def parallel_scenario_simulation(scenarios: List[Dict]) -> List[pd.DataFrame]:
    """并行执行多个场景的模拟"""
    return Parallel(n_jobs=-1)(
        delayed(simulate_single_scenario)(scenario) 
        for scenario in scenarios
    )
```

### 3.3 Module 3: MatchFunctionEstimator (匹配函数估计器)

#### 3.3.1 模块职责
- 整合多轮匹配结果为训练数据集
- 使用Logit回归估计匹配概率函数λ
- 模型验证和统计检验

#### 3.3.2 核心类设计

```python
class MatchingDataProcessor:
    """匹配数据预处理器"""
    
    def __init__(self):
        self.feature_columns = ['T', 'S', 'D', 'W', 'age', 'education']
        self.target_column = 'matched'
    
    def prepare_training_data(self, 
                             matching_results: List[pd.DataFrame]) -> pd.DataFrame:
        """准备Logit回归的训练数据"""
        # 合并多轮匹配结果
        combined_data = pd.concat(matching_results, ignore_index=True)
        
        # 特征工程
        features = self._extract_features(combined_data)
        
        # 数据清洗和验证
        clean_data = self._clean_and_validate(features)
        
        return clean_data
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征提取"""
        features = data.copy()
        
        # 添加努力水平特征
        features['effort_level'] = self._compute_effort_level(data)
        
        # 添加市场紧张度特征
        features['log_theta'] = np.log(data['market_tightness'])
        
        # 添加交互项
        features['T_theta'] = features['T'] * features['log_theta']
        features['effort_theta'] = features['effort_level'] * features['log_theta']
        
        return features
    
    @numba.jit(nopython=True)
    def _compute_effort_level(self, data: np.ndarray) -> np.ndarray:
        """计算努力水平（numba优化）"""
        pass

class LogitEstimator:
    """Logit回归估计器"""
    
    def __init__(self, regularization: str = 'l2', alpha: float = 0.01):
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.coefficients = {}
        self.model_stats = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """拟合Logit模型"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 拟合模型
        self.model = LogisticRegression(
            penalty=self.regularization,
            C=1/self.alpha,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # 提取系数
        self._extract_coefficients(X.columns)
        
        # 计算模型统计量
        self._compute_model_statistics(X_scaled, y)
    
    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """预测匹配概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    @numba.jit(nopython=True)
    def fast_predict(self, 
                    features: np.ndarray, 
                    coefficients: np.ndarray,
                    intercept: float) -> np.ndarray:
        """numba优化的快速预测"""
        # λ = 1/(1 + exp(-(δ₀ + δ'x)))
        linear_combination = intercept + np.dot(features, coefficients)
        return 1.0 / (1.0 + np.exp(-linear_combination))
    
    def _extract_coefficients(self, feature_names: List[str]) -> None:
        """提取回归系数"""
        self.coefficients = {
            'intercept': self.model.intercept_[0],
            'state_coeffs': {},
            'control_coeffs': {},
            'effort_coeff': 0.0,
            'tightness_coeff': 0.0
        }
        
        for i, name in enumerate(feature_names):
            coeff = self.model.coef_[0][i]
            if name in ['T', 'S', 'D', 'W']:
                self.coefficients['state_coeffs'][name] = coeff
            elif name == 'effort_level':
                self.coefficients['effort_coeff'] = coeff
            elif name == 'log_theta':
                self.coefficients['tightness_coeff'] = coeff
            else:
                self.coefficients['control_coeffs'][name] = coeff
    
    def _compute_model_statistics(self, X: np.ndarray, y: np.ndarray) -> None:
        """计算模型统计量"""
        from sklearn.metrics import log_loss, roc_auc_score
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        self.model_stats = {
            'log_likelihood': -log_loss(y, y_pred_proba) * len(y),
            'aic': 2 * len(self.model.coef_[0]) - 2 * (-log_loss(y, y_pred_proba) * len(y)),
            'bic': len(self.model.coef_[0]) * np.log(len(y)) - 2 * (-log_loss(y, y_pred_proba) * len(y)),
            'pseudo_r2': 1 - (log_loss(y, y_pred_proba) / log_loss(y, [y.mean()] * len(y))),
            'auc': roc_auc_score(y, y_pred_proba),
            'n_observations': len(y)
        }

class ModelValidator:
    """模型验证器"""
    
    def __init__(self):
        self.validation_results = {}
    
    def cross_validate(self, 
                      estimator: LogitEstimator,
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv_folds: int = 5) -> Dict[str, float]:
        """交叉验证"""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, log_loss
        
        # 定义评分函数
        def neg_log_loss(y_true, y_pred):
            return -log_loss(y_true, y_pred)
        
        scorer = make_scorer(neg_log_loss, needs_proba=True)
        
        # 执行交叉验证
        cv_scores = cross_val_score(
            estimator.model, 
            estimator.scaler.transform(X), 
            y, 
            cv=cv_folds, 
            scoring=scorer
        )
        
        self.validation_results = {
            'cv_mean_score': cv_scores.mean(),
            'cv_std_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        return self.validation_results
    
    def goodness_of_fit_test(self, 
                           estimator: LogitEstimator,
                           X: pd.DataFrame, 
                           y: pd.Series) -> Dict[str, Any]:
        """拟合优度检验"""
        # Hosmer-Lemeshow检验
        y_pred_proba = estimator.predict_probability(X)
        
        # 将概率分为10个区间
        deciles = pd.qcut(y_pred_proba, 10, duplicates='drop')
        
        # 计算每个区间的观测值和期望值
        observed = y.groupby(deciles).sum()
        expected = y_pred_proba.groupby(deciles).sum() 
        total = y.groupby(deciles).count()
        
        # 计算Hosmer-Lemeshow统计量
        hl_statistic = ((observed - expected) ** 2 / (expected * (1 - expected/total))).sum()
        
        return {
            'hosmer_lemeshow_stat': hl_statistic,
            'p_value': 1 - stats.chi2.cdf(hl_statistic, df=len(observed)-2),
            'decile_table': pd.DataFrame({
                'observed': observed,
                'expected': expected,
                'total': total
            })
        }
```

### 3.4 Module 4: MFGSimulator (MFG模拟器)

#### 3.4.1 模块职责
- 状态空间离散化
- 贝尔曼方程值迭代求解
- KFE人口分布演化
- 收敛性判断

#### 3.4.2 核心类设计

```python
class StateDiscretizer:
    """状态空间离散化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grid_points = config['discretization']
        self.state_bounds = {}
        self.grid_coordinates = {}
    
    def setup_state_space(self, 
                         agent_data: pd.DataFrame) -> None:
        """设置状态空间"""
        # 确定每个状态变量的边界
        for var in ['T', 'S', 'D', 'W']:
            self.state_bounds[var] = {
                'min': agent_data[var].min(),
                'max': agent_data[var].max()
            }
        
        # 生成网格坐标
        self._generate_grid_coordinates()
    
    def _generate_grid_coordinates(self) -> None:
        """生成网格坐标"""
        for var in ['T', 'S', 'D', 'W']:
            n_points = self.grid_points[f'n_grid_points_{var}']
            min_val = self.state_bounds[var]['min']
            max_val = self.state_bounds[var]['max']
            
            self.grid_coordinates[var] = np.linspace(min_val, max_val, n_points)
    
    @numba.jit(nopython=True)
    def state_to_index(self, state: np.ndarray) -> int:
        """将连续状态转换为离散索引"""
        pass
    
    @numba.jit(nopython=True)
    def index_to_state(self, index: int) -> np.ndarray:
        """将离散索引转换为状态"""
        pass
    
    def get_total_states(self) -> int:
        """获取总状态数"""
        return np.prod([self.grid_points[f'n_grid_points_{var}'] 
                       for var in ['T', 'S', 'D', 'W']])

class BellmanSolver:
    """贝尔曼方程求解器"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 discretizer: StateDiscretizer,
                 match_function: Callable):
        self.config = config
        self.discretizer = discretizer
        self.match_function = match_function
        
        # 贝尔曼方程参数
        self.rho = config['bellman_equation']['discount_factor_rho']
        self.kappa = config['bellman_equation']['effort_cost_kappa']
        
        # 价值函数
        self.V_unemployed = None
        self.V_employed = None
        self.optimal_effort = None
    
    def initialize_value_functions(self) -> None:
        """初始化价值函数"""
        n_states = self.discretizer.get_total_states()
        self.V_unemployed = np.zeros(n_states)
        self.V_employed = np.zeros(n_states)
        self.optimal_effort = np.zeros(n_states)
    
    @numba.jit(nopython=True)
    def _bellman_operator_unemployed(self, 
                                   state_index: int,
                                   V_unemployed: np.ndarray,
                                   V_employed: np.ndarray,
                                   theta: float) -> Tuple[float, float]:
        """失业状态的贝尔曼算子"""
        # 将索引转换为状态
        state = self.discretizer.index_to_state(state_index)
        
        # 网格搜索最优努力水平
        best_value = -np.inf
        best_effort = 0.0
        
        effort_grid = np.linspace(0, 1, 21)  # 0到1之间的21个点
        
        for effort in effort_grid:
            # 计算即时效用
            instant_utility = self._unemployment_benefit(state) - 0.5 * self.kappa * effort**2
            
            # 计算匹配概率
            match_prob = self.match_function(state, effort, theta)
            
            # 计算期望延续价值
            next_state = self._state_transition(state, effort)
            next_state_index = self.discretizer.state_to_index(next_state)
            
            expected_continuation = (match_prob * V_employed[next_state_index] + 
                                   (1 - match_prob) * V_unemployed[next_state_index])
            
            # 计算总价值
            total_value = instant_utility + self.rho * expected_continuation
            
            if total_value > best_value:
                best_value = total_value
                best_effort = effort
        
        return best_value, best_effort
    
    @numba.jit(nopython=True)
    def _bellman_operator_employed(self, 
                                 state_index: int,
                                 V_unemployed: np.ndarray,
                                 V_employed: np.ndarray) -> float:
        """就业状态的贝尔曼算子"""
        state = self.discretizer.index_to_state(state_index)
        
        # 就业效用
        employment_utility = self._employment_utility(state)
        
        # 外生离职概率
        separation_prob = self._separation_rate(state)
        
        # 期望延续价值
        expected_continuation = (separation_prob * V_unemployed[state_index] + 
                               (1 - separation_prob) * V_employed[state_index])
        
        return employment_utility + self.rho * expected_continuation
    
    @numba.jit(nopython=True)
    def _unemployment_benefit(self, state: np.ndarray) -> float:
        """失业救济"""
        # b(x) = b₀ + b₁×T + b₂×S + b₃×D + b₄×W
        b_coeffs = self.config['utility_functions']['unemployment_b_coeffs']
        return (b_coeffs[0] + 
                b_coeffs[1] * state[0] +  # T
                b_coeffs[2] * state[1] +  # S
                b_coeffs[3] * state[2] +  # D
                b_coeffs[4] * state[3])   # W
    
    @numba.jit(nopython=True)
    def _employment_utility(self, state: np.ndarray) -> float:
        """就业效用"""
        # ω(x) = ω₀ + ω₁×T + ω₂×S + ω₃×D + ω₄×W
        omega_coeffs = self.config['utility_functions']['employment_omega_coeffs']
        return (omega_coeffs[0] + 
                omega_coeffs[1] * state[0] +  # T
                omega_coeffs[2] * state[1] +  # S
                omega_coeffs[3] * state[2] +  # D
                omega_coeffs[4] * state[3])   # W
    
    @numba.jit(nopython=True)
    def _separation_rate(self, state: np.ndarray) -> float:
        """外生离职率"""
        # μ(x) = 1/(1 + exp(-(η₀ + η₁×T + η₂×S + η₃×D + η₄×W)))
        eta_coeffs = self.config['exogenous_separation_mu']['eta_coeffs']
        linear_part = (eta_coeffs[0] + 
                      eta_coeffs[1] * state[0] +  # T
                      eta_coeffs[2] * state[1] +  # S
                      eta_coeffs[3] * state[2] +  # D
                      eta_coeffs[4] * state[3])   # W
        return 1.0 / (1.0 + np.exp(-linear_part))
    
    @numba.jit(nopython=True)
    def _state_transition(self, state: np.ndarray, effort: float) -> np.ndarray:
        """状态转移函数"""
        # 根据努力水平更新状态
        new_state = state.copy()
        
        # T_{t+1} = T_t + γ_T * a_t * (T_max - T_t)
        gamma_T = 0.1  # 工作时间调整速度
        T_max = 70.0   # 最大工作时间
        new_state[0] = state[0] + gamma_T * effort * (T_max - state[0])
        
        # S_{t+1} = S_t + γ_S * a_t * (1 - S_t)
        gamma_S = 0.1  # 技能提升速度
        new_state[1] = state[1] + gamma_S * effort * (1 - state[1])
        
        # D_{t+1} = D_t + γ_D * a_t * (1 - D_t)
        gamma_D = 0.1  # 数字素养提升速度
        new_state[2] = state[2] + gamma_D * effort * (1 - state[2])
        
        # W_{t+1} = max(W_min, W_t - γ_W * a_t)
        gamma_W = 0.05  # 期望工资调整速度
        W_min = 1400.0  # 最低期望工资
        new_state[3] = max(W_min, state[3] - gamma_W * effort)
        
        return new_state
    
    def solve_value_iteration(self, 
                            theta: float,
                            max_iterations: int = 1000,
                            tolerance: float = 1e-6) -> bool:
        """值迭代求解"""
        self.initialize_value_functions()
        
        for iteration in range(max_iterations):
            # 保存旧值函数
            V_unemployed_old = self.V_unemployed.copy()
            V_employed_old = self.V_employed.copy()
            
            # 更新价值函数
            for state_idx in range(self.discretizer.get_total_states()):
                # 更新失业状态价值函数
                new_value, optimal_effort = self._bellman_operator_unemployed(
                    state_idx, V_unemployed_old, V_employed_old, theta
                )
                self.V_unemployed[state_idx] = new_value
                self.optimal_effort[state_idx] = optimal_effort
                
                # 更新就业状态价值函数
                self.V_employed[state_idx] = self._bellman_operator_employed(
                    state_idx, V_unemployed_old, V_employed_old
                )
            
            # 检查收敛性
            max_diff_unemployed = np.max(np.abs(self.V_unemployed - V_unemployed_old))
            max_diff_employed = np.max(np.abs(self.V_employed - V_employed_old))
            
            if max(max_diff_unemployed, max_diff_employed) < tolerance:
                print(f"值迭代在第{iteration+1}轮收敛")
                return True
        
        print(f"值迭代在{max_iterations}轮后未收敛")
        return False

class KFESolver:
    """KFE求解器"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 discretizer: StateDiscretizer,
                 bellman_solver: BellmanSolver):
        self.config = config
        self.discretizer = discretizer
        self.bellman_solver = bellman_solver
        
        # 人口分布
        self.m_unemployed = None
        self.m_employed = None
    
    def initialize_population_distribution(self, 
                                         initial_data: pd.DataFrame) -> None:
        """初始化人口分布"""
        n_states = self.discretizer.get_total_states()
        self.m_unemployed = np.zeros(n_states)
        self.m_employed = np.zeros(n_states)
        
        # 根据初始数据设置失业人口分布
        for _, agent in initial_data.iterrows():
            state = np.array([agent['T'], agent['S'], agent['D'], agent['W']])
            state_idx = self.discretizer.state_to_index(state)
            self.m_unemployed[state_idx] += 1.0 / len(initial_data)
    
    @numba.jit(nopython=True)
    def _kfe_step(self, 
                  m_unemployed: np.ndarray,
                  m_employed: np.ndarray,
                  optimal_effort: np.ndarray,
                  match_function: Callable,
                  theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """KFE单步演化"""
        n_states = len(m_unemployed)
        new_m_unemployed = np.zeros(n_states)
        new_m_employed = np.zeros(n_states)
        
        for state_idx in range(n_states):
            state = self.discretizer.index_to_state(state_idx)
            effort = optimal_effort[state_idx]
            
            # 计算匹配概率和离职概率
            match_prob = match_function(state, effort, theta)
            separation_prob = self.bellman_solver._separation_rate(state)
            
            # 状态转移
            next_state = self.bellman_solver._state_transition(state, effort)
            next_state_idx = self.discretizer.state_to_index(next_state)
            
            # 失业者流动
            unemployed_outflow = match_prob * m_unemployed[state_idx]
            unemployed_stay = (1 - match_prob) * m_unemployed[state_idx]
            
            # 就业者流动
            employed_outflow = separation_prob * m_employed[state_idx]
            employed_stay = (1 - separation_prob) * m_employed[state_idx]
            
            # 更新分布
            new_m_unemployed[next_state_idx] += unemployed_stay
            new_m_unemployed[state_idx] += employed_outflow
            
            new_m_employed[next_state_idx] += unemployed_outflow
            new_m_employed[next_state_idx] += employed_stay
        
        return new_m_unemployed, new_m_employed
    
    def solve_kfe(self, 
                  theta: float,
                  max_iterations: int = 1000,
                  tolerance: float = 1e-6) -> bool:
        """求解KFE"""
        for iteration in range(max_iterations):
            # 保存旧分布
            m_unemployed_old = self.m_unemployed.copy()
            m_employed_old = self.m_employed.copy()
            
            # KFE演化一步
            self.m_unemployed, self.m_employed = self._kfe_step(
                m_unemployed_old, 
                m_employed_old,
                self.bellman_solver.optimal_effort,
                self.bellman_solver.match_function,
                theta
            )
            
            # 检查收敛性
            max_diff_unemployed = np.max(np.abs(self.m_unemployed - m_unemployed_old))
            max_diff_employed = np.max(np.abs(self.m_employed - m_employed_old))
            
            if max(max_diff_unemployed, max_diff_employed) < tolerance:
                print(f"KFE在第{iteration+1}轮收敛")
                return True
        
        print(f"KFE在{max_iterations}轮后未收敛")
        return False
    
    def get_unemployment_rate(self) -> float:
        """计算失业率"""
        total_unemployed = np.sum(self.m_unemployed)
        total_employed = np.sum(self.m_employed)
        return total_unemployed / (total_unemployed + total_employed)

class MFGSimulator:
    """MFG模拟器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discretizer = StateDiscretizer(config)
        self.bellman_solver = None
        self.kfe_solver = None
        self.convergence_history = []
    
    def setup(self, 
              agent_data: pd.DataFrame,
              match_function: Callable) -> None:
        """设置模拟器"""
        # 设置状态空间
        self.discretizer.setup_state_space(agent_data)
        
        # 初始化求解器
        self.bellman_solver = BellmanSolver(
            self.config, self.discretizer, match_function
        )
        self.kfe_solver = KFESolver(
            self.config, self.discretizer, self.bellman_solver
        )
        
        # 初始化人口分布
        self.kfe_solver.initialize_population_distribution(agent_data)
    
    def solve_mfg_equilibrium(self, 
                            V: int,  # 岗位空缺数
                            max_outer_iterations: int = 100,
                            tolerance: float = 1e-4) -> Dict[str, Any]:
        """求解MFG均衡"""
        
        # 初始市场紧张度
        theta = V / np.sum(self.kfe_solver.m_unemployed)
        
        for outer_iter in range(max_outer_iterations):
            print(f"MFG外层迭代 {outer_iter + 1}, θ = {theta:.4f}")
            
            # Step 1: 给定θ，求解贝尔曼方程
            bellman_converged = self.bellman_solver.solve_value_iteration(theta)
            if not bellman_converged:
                print("贝尔曼方程未收敛，终止")
                break
            
            # Step 2: 给定最优策略，求解KFE
            kfe_converged = self.kfe_solver.solve_kfe(theta)
            if not kfe_converged:
                print("KFE未收敛，终止")
                break
            
            # Step 3: 更新市场紧张度
            new_total_unemployed = np.sum(self.kfe_solver.m_unemployed)
            new_theta = V / new_total_unemployed
            
            # 检查收敛性
            theta_diff = abs(new_theta - theta)
            unemployment_rate = self.kfe_solver.get_unemployment_rate()
            
            print(f"  失业率: {unemployment_rate:.4f}")
            print(f"  θ变化: {theta_diff:.6f}")
            
            self.convergence_history.append({
                'iteration': outer_iter + 1,
                'theta': theta,
                'unemployment_rate': unemployment_rate,
                'theta_diff': theta_diff
            })
            
            if theta_diff < tolerance:
                print(f"MFG在第{outer_iter + 1}轮收敛")
                
                return {
                    'converged': True,
                    'final_theta': new_theta,
                    'unemployment_rate': unemployment_rate,
                    'value_functions': {
                        'V_unemployed': self.bellman_solver.V_unemployed,
                        'V_employed': self.bellman_solver.V_employed
                    },
                    'optimal_policy': self.bellman_solver.optimal_effort,
                    'population_distribution': {
                        'm_unemployed': self.kfe_solver.m_unemployed,
                        'm_employed': self.kfe_solver.m_employed
                    },
                    'convergence_history': self.convergence_history
                }
            
            # 更新θ（使用阻尼更新避免振荡）
            damping = 0.5
            theta = damping * new_theta + (1 - damping) * theta
        
        # 未收敛
        return {
            'converged': False,
            'final_theta': theta,
            'unemployment_rate': self.kfe_solver.get_unemployment_rate(),
            'convergence_history': self.convergence_history
        }
```

### 3.5 Module 5: ExperimentController (实验控制器)

#### 3.5.1 模块职责
- 参数校准（遗传算法优化）
- 政策分析（多场景批量模拟）
- 结果处理和报告生成

#### 3.5.2 核心类设计

```python
class CalibrationEngine:
    """校准引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_values = config['calibration_target']
        self.parameter_bounds = config['calibration_params']
        self.ga_config = config.get('genetic_algorithm', {})
        
        # 遗传算法组件
        self.toolbox = None
        self.population = None
        self.logbook = None
    
    def setup_genetic_algorithm(self) -> None:
        """设置遗传算法"""
        from deap import algorithms, base, creator, tools
        
        # 定义优化问题（最小化）
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # 参数生成函数
        param_names = list(self.parameter_bounds.keys())
        for i, param_name in enumerate(param_names):
            bounds = self.parameter_bounds[param_name]
            self.toolbox.register(f"attr_{i}", 
                                 np.random.uniform, bounds[0], bounds[1])
        
        # 个体和种群生成
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             [getattr(self.toolbox, f"attr_{i}") 
                              for i in range(len(param_names))], n=1)
        
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        # 遗传算子
        self.toolbox.register("evaluate", self._objective_function)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, 
                             mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _objective_function(self, individual: List[float]) -> Tuple[float]:
        """目标函数：模拟结果与现实数据的差值"""
        try:
            # 将个体转换为参数字典
            parameters = self._individual_to_parameters(individual)
            
            # 运行MFG模拟
            simulation_result = self._run_simulation_with_parameters(parameters)
            
            # 计算与目标值的差距
            objective_value = 0.0
            
            # 失业率差距
            unemployment_diff = abs(simulation_result['unemployment_rate'] - 
                                  self.target_values['unemployment_rate'])
            objective_value += unemployment_diff ** 2
            
            # 可以添加更多目标指标
            # matching_rate_diff = abs(simulation_result['matching_rate'] - 
            #                          self.target_values['matching_rate'])
            # objective_value += matching_rate_diff ** 2
            
            return (objective_value,)
        
        except Exception as e:
            # 如果模拟失败，返回极大惩罚值
            print(f"模拟失败: {e}")
            return (1e6,)
    
    def _individual_to_parameters(self, individual: List[float]) -> Dict[str, float]:
        """将遗传算法个体转换为参数字典"""
        parameters = {}
        param_names = list(self.parameter_bounds.keys())
        
        for i, param_name in enumerate(param_names):
            # 确保参数在边界内
            bounds = self.parameter_bounds[param_name]
            value = np.clip(individual[i], bounds[0], bounds[1])
            parameters[param_name] = value
        
        return parameters
    
    def _run_simulation_with_parameters(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """使用给定参数运行模拟"""
        # 更新配置
        updated_config = self.config.copy()
        for param_name, value in parameters.items():
            # 处理嵌套参数路径，如 'mfg_simulator.bellman_equation.effort_cost_kappa'
            keys = param_name.split('.')
            current_dict = updated_config
            for key in keys[:-1]:
                current_dict = current_dict[key]
            current_dict[keys[-1]] = value
        
        # 运行完整的模拟流程
        # 这里需要调用完整的五模块流程
        simulation_pipeline = SimulationPipeline(updated_config)
        return simulation_pipeline.run_complete_simulation()
    
    def calibrate(self, 
                  population_size: int = 50,
                  n_generations: int = 100,
                  cxpb: float = 0.7,
                  mutpb: float = 0.2) -> Dict[str, Any]:
        """执行校准"""
        self.setup_genetic_algorithm()
        
        # 初始化种群
        self.population = self.toolbox.population(n=population_size)
        
        # 运行遗传算法
        from deap import algorithms
        
        self.population, self.logbook = algorithms.eaSimple(
            self.population, self.toolbox, 
            cxpb=cxpb, mutpb=mutpb, ngen=n_generations,
            stats=self._setup_statistics(),
            halloffame=tools.HallOfFame(1),
            verbose=True
        )
        
        # 获取最优个体
        best_individual = tools.selBest(self.population, 1)[0]
        best_parameters = self._individual_to_parameters(best_individual)
        
        return {
            'best_parameters': best_parameters,
            'best_fitness': best_individual.fitness.values[0],
            'logbook': self.logbook,
            'final_population': self.population
        }
    
    def _setup_statistics(self):
        """设置遗传算法统计"""
        from deap import tools
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        return stats

class PolicyAnalyzer:
    """政策分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_config = config.copy()
        self.policy_scenarios = config.get('policy_scenarios', [])
    
    def analyze_policy_scenarios(self) -> Dict[str, Any]:
        """分析多个政策场景"""
        results = {}
        
        # 运行基准场景
        print("运行基准场景...")
        baseline_result = self._run_scenario(self.baseline_config, "baseline")
        results['baseline'] = baseline_result
        
        # 运行政策场景
        for scenario in self.policy_scenarios:
            print(f"运行政策场景: {scenario['name']}")
            
            # 应用政策参数覆盖
            scenario_config = self._apply_policy_override(
                self.baseline_config.copy(), 
                scenario['params_override']
            )
            
            scenario_result = self._run_scenario(scenario_config, scenario['name'])
            results[scenario['name']] = scenario_result
        
        # 生成比较分析
        comparison = self._compare_scenarios(results)
        
        return {
            'scenario_results': results,
            'comparison_analysis': comparison
        }
    
    def _apply_policy_override(self, 
                              base_config: Dict[str, Any],
                              override_params: Dict[str, Any]) -> Dict[str, Any]:
        """应用政策参数覆盖"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(base_config, override_params)
        return base_config
    
    def _run_scenario(self, config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
        """运行单个政策场景"""
        # 运行完整模拟流程
        simulation_pipeline = SimulationPipeline(config)
        result = simulation_pipeline.run_complete_simulation()
        
        # 添加场景信息
        result['scenario_name'] = scenario_name
        result['config_used'] = config
        
        return result
    
    def _compare_scenarios(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同场景的结果"""
        baseline = results['baseline']
        comparison = {}
        
        for scenario_name, scenario_result in results.items():
            if scenario_name == 'baseline':
                continue
            
            # 计算关键指标的变化
            unemployment_change = (scenario_result['unemployment_rate'] - 
                                 baseline['unemployment_rate'])
            
            # 计算相对变化百分比
            unemployment_pct_change = (unemployment_change / 
                                     baseline['unemployment_rate']) * 100
            
            comparison[scenario_name] = {
                'unemployment_rate_change': unemployment_change,
                'unemployment_rate_pct_change': unemployment_pct_change,
                'policy_effectiveness': 'positive' if unemployment_change < 0 else 'negative'
            }
        
        return comparison

class ExperimentController:
    """实验控制器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_engine = CalibrationEngine(config)
        self.policy_analyzer = PolicyAnalyzer(config)
        self.results_manager = ResultsManager(config)
    
    def run_calibration_experiment(self) -> Dict[str, Any]:
        """运行校准实验"""
        print("开始参数校准...")
        
        calibration_result = self.calibration_engine.calibrate()
        
        # 保存校准结果
        self.results_manager.save_calibration_results(calibration_result)
        
        print("校准完成")
        return calibration_result
    
    def run_policy_analysis_experiment(self, 
                                     calibrated_params: Dict[str, float] = None) -> Dict[str, Any]:
        """运行政策分析实验"""
        print("开始政策分析...")
        
        # 如果提供了校准参数，更新配置
        if calibrated_params:
            self._update_config_with_calibrated_params(calibrated_params)
        
        policy_results = self.policy_analyzer.analyze_policy_scenarios()
        
        # 保存分析结果
        self.results_manager.save_policy_results(policy_results)
        
        print("政策分析完成")
        return policy_results
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """运行完整实验（校准+政策分析）"""
        # Step 1: 参数校准
        calibration_result = self.run_calibration_experiment()
        
        # Step 2: 使用校准参数进行政策分析
        best_params = calibration_result['best_parameters']
        policy_result = self.run_policy_analysis_experiment(best_params)
        
        # 综合结果
        complete_result = {
            'calibration': calibration_result,
            'policy_analysis': policy_result,
            'metadata': {
                'experiment_date': datetime.now().isoformat(),
                'config_used': self.config
            }
        }
        
        # 生成综合报告
        self.results_manager.generate_complete_report(complete_result)
        
        return complete_result
    
    def _update_config_with_calibrated_params(self, calibrated_params: Dict[str, float]) -> None:
        """使用校准参数更新配置"""
        for param_name, value in calibrated_params.items():
            keys = param_name.split('.')
            current_dict = self.config
            for key in keys[:-1]:
                current_dict = current_dict[key]
            current_dict[keys[-1]] = value

class ResultsManager:
    """结果管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config['io_paths']['results_output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_calibration_results(self, calibration_result: Dict[str, Any]) -> None:
        """保存校准结果"""
        # 保存最优参数
        params_file = self.results_dir / "calibrated_parameters.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_result['best_parameters'], f, 
                     indent=2, ensure_ascii=False)
        
        # 保存校准历史
        logbook_file = self.results_dir / "calibration_history.csv"
        logbook_df = pd.DataFrame(calibration_result['logbook'])
        logbook_df.to_csv(logbook_file, index=False)
        
        print(f"校准结果已保存到: {self.results_dir}")
    
    def save_policy_results(self, policy_results: Dict[str, Any]) -> None:
        """保存政策分析结果"""
        # 保存场景结果
        scenarios_file = self.results_dir / "policy_scenarios_results.json"
        with open(scenarios_file, 'w', encoding='utf-8') as f:
            json.dump(policy_results['scenario_results'], f, 
                     indent=2, ensure_ascii=False, default=str)
        
        # 保存比较分析
        comparison_file = self.results_dir / "policy_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(policy_results['comparison_analysis'], f, 
                     indent=2, ensure_ascii=False)
        
        print(f"政策分析结果已保存到: {self.results_dir}")
    
    def generate_complete_report(self, complete_result: Dict[str, Any]) -> None:
        """生成完整实验报告"""
        report_file = self.results_dir / "experiment_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            self._write_markdown_report(f, complete_result)
        
        print(f"完整实验报告已生成: {report_file}")
    
    def _write_markdown_report(self, f, complete_result: Dict[str, Any]) -> None:
        """写入Markdown格式的报告"""
        f.write("# EconLab实验报告\n\n")
        f.write(f"**生成时间**: {complete_result['metadata']['experiment_date']}\n\n")
        
        # 校准结果部分
        f.write("## 参数校准结果\n\n")
        calibration = complete_result['calibration']
        f.write(f"**最优适应度值**: {calibration['best_fitness']:.6f}\n\n")
        
        f.write("### 校准后的参数\n\n")
        for param, value in calibration['best_parameters'].items():
            f.write(f"- **{param}**: {value:.6f}\n")
        f.write("\n")
        
        # 政策分析结果部分
        f.write("## 政策分析结果\n\n")
        policy_analysis = complete_result['policy_analysis']
        
        for scenario_name, comparison in policy_analysis['comparison_analysis'].items():
            f.write(f"### {scenario_name}\n\n")
            f.write(f"- 失业率变化: {comparison['unemployment_rate_change']:.4f}\n")
            f.write(f"- 失业率相对变化: {comparison['unemployment_rate_pct_change']:.2f}%\n")
            f.write(f"- 政策效果: {comparison['policy_effectiveness']}\n\n")
```

---

## 4. 开发计划与时间安排

### 4.1 开发阶段划分

#### **第一阶段：核心计算引擎开发 (12-16周)**

**Week 1-4: Module 1 - PopulationGenerator**
- Week 1: 基础架构搭建，类设计实现
- Week 2: Copula引擎集成，企业生成器开发
- Week 3: numba优化实现，批量生成逻辑
- Week 4: 单元测试，性能测试，文档编写

**Week 5-8: Module 2 - MatchingEngine**
- Week 5: Gale-Shapley算法实现
- Week 6: 偏好计算器开发，numba优化
- Week 7: 多场景模拟器，并行化优化
- Week 8: 算法验证，性能测试

**Week 9-12: Module 3 - MatchFunctionEstimator**
- Week 9: 数据预处理器实现
- Week 10: Logit估计器开发
- Week 11: 模型验证器，交叉验证
- Week 12: 统计检验，结果输出

**Week 13-16: Module 4 - MFGSimulator**
- Week 13: 状态空间离散化，架构搭建
- Week 14: 贝尔曼方程求解器
- Week 15: KFE求解器，MFG主循环
- Week 16: 收敛性优化，数值稳定性调试

#### **第二阶段：优化与校准系统 (3-4周)**

**Week 17-20: Module 5 - ExperimentController + Optimization**
- Week 17: 遗传算法引擎开发
- Week 18: 校准系统集成测试
- Week 19: 政策分析器实现
- Week 20: 完整流程测试，性能优化

#### **第三阶段：用户界面开发 (6-8周)**

**Week 21-28: GUI Development**
- Week 21-23: PyQt6主窗口设计，基础界面
- Week 24-25: 参数设置面板，配置管理
- Week 26-27: 实时监控界面，进度显示
- Week 28: 结果展示界面，数据可视化

#### **第四阶段：可视化与集成 (4-6周)**

**Week 29-34: Visualization & Integration**
- Week 29-30: matplotlib + PyQt集成
- Week 31: 实时图表，动态可视化
- Week 32-33: 前后端集成，异步通信
- Week 34: 系统级测试，用户体验优化

#### **第五阶段：文档与打包 (3-4周)**

**Week 35-38: Documentation & Packaging**
- Week 35-36: 代码文档编写，API文档
- Week 37: 用户手册，技术说明文档
- Week 38: PyInstaller打包，分发测试

### 4.2 关键里程碑

| 里程碑 | 时间 | 交付内容 | 验收标准 |
|--------|------|----------|----------|
| Alpha版本 | Week 16 | 核心计算引擎 | 五模块基础功能完成，数值测试通过 |
| Beta版本 | Week 28 | 完整系统 | GUI界面完成，核心功能可用 |
| RC版本 | Week 34 | 集成系统 | 完整功能测试，性能达标 |
| 最终版本 | Week 38 | EconLab 1.0 | exe打包完成，文档齐全 |

### 4.3 技术风险缓解计划

#### **高风险项目应对策略**

1. **MFG数值收敛问题**
   - **应对**: 并行开发策略迭代法作为备选
   - **检查点**: Week 14进行收敛性测试
   - **备选方案**: 简化状态空间或使用近似算法

2. **numba优化兼容性**
   - **应对**: 关键函数优先测试numba兼容性
   - **检查点**: 每个模块完成后立即测试优化效果
   - **备选方案**: Cython实现关键性能瓶颈

3. **GUI性能问题**
   - **应对**: 早期进行GUI+计算引擎集成测试
   - **检查点**: Week 26测试大规模计算时的界面响应
   - **备选方案**: Web界面或分离式架构

### 4.4 质量保证计划

#### **测试策略**
- **单元测试**: 每个模块>=80%代码覆盖率
- **集成测试**: 模块间接口测试
- **性能测试**: 大规模数据性能基准
- **用户测试**: GUI可用性测试

#### **代码质量**
- **代码审查**: 关键算法代码审查
- **文档要求**: 每个函数必须有docstring
- **类型注解**: 所有公共接口必须有类型注解
- **日志记录**: 完整的运行日志系统

---

## 5. 总结

这份详细的项目计划文档涵盖了EconLab经济学实验室的完整技术架构、模块设计和开发计划。项目将按照"核心优先"的策略，首先确保五个核心模块的高质量实现，再逐步完善GUI界面和高级功能。

整个项目预计需要38周（约9个月）的开发时间，将交付一个功能完整、性能优异的经济学仿真平台。通过systematic的模块化设计和严格的质量控制，确保最终产品能够满足专业研究和教学应用的需求。

**下一步行动**: 请您确认这个详细的项目计划是否符合预期，我们即可开始第一阶段的开发工作。
