# MFG匹配率与离职率结构修复说明

## 1. 文档目的

本说明文档对应以下代码改动：

- `MODULES/MFG/matching_utils.py`
- `MODULES/MFG/bellman_solver.py`
- `MODULES/MFG/kfe_solver.py`
- `MODULES/MFG/equilibrium_solver.py`
- `CONFIG/mfg_config.yaml`
- `TESTS/test_transition_intercepts.py`

本轮修复聚焦三个结构问题：

1. 为匹配概率 `lambda` 提供“统一截距平移”的结构修正能力；
2. 为离职率截距 `eta0` 提供“按目标分离率反解”的结构修正能力；
3. 统一 Bellman 与 KFE 在 `S/D` 状态更新时的量纲与标准化规则。

---

## 2. 上下游关系

### 2.1 上游依赖

- `OUTPUT/logistic/match_function_model.pkl`
  - 提供匹配概率 Logit 模型系数。
- `CONFIG/mfg_config.yaml`
  - 提供状态更新、匹配率目标值、离职率目标值等结构参数与开关。
- `MODULES/POPULATION`
  - 负责生成基础人口样本。

### 2.2 下游使用方

- `MODULES/MFG/equilibrium_solver.py`
  - 调用 Bellman 与 KFE 完成均衡迭代。
- `MODULES/SIMULATOR/market_simulator.py`
  - 间接复用 MFG 求解链路运行政策场景。
- `run_paper_simulation.py`
  - 间接复用 MFG 求解链路生成论文表格与模拟输出。

---

## 3. 功能概览

### 3.1 `matching_utils.py`

该文件新增了三类公共能力：

- `compute_sigma_from_demographics()`
  - 统一构造人口统计控制变量 `sigma`，保证与 Logistic 训练阶段一致。
- `project_states_with_effort_grid()` / `project_states_with_effort_vector()`
  - 统一 `T/S/D/W` 在努力作用下的状态投影逻辑。
  - `S/D` 先做 Min-Max 标准化，再更新，再反标准化。
- `solve_logit_shift_for_target()`
  - 通过二分法反解 Logit 平移项，使平均概率逼近目标值。

### 3.2 `bellman_solver.py`

本次改动的核心点：

- 匹配概率从“直接使用 Logit 输出”改为“线性项 + 统一截距平移后再过 sigmoid”；
- 新增 `set_lambda_intercept_shift()` 与 `calibrate_lambda_intercept()`；
- 新增 `set_eta0()`，用于与 KFE 共用同一套离职率截距；
- 离职率 Numba 函数的下界从 `0.01` 改为 `1e-6`，避免结构上无法匹配低分离率目标；
- Bellman 侧的 `S/D` 状态投影改为复用统一工具函数。

补充说明：

- 当前正式配置中，`lambda/eta0` 的自动对齐开关默认是关闭的；
- 相关方法仍保留，供后续结构实验或专项诊断使用。

### 3.3 `kfe_solver.py`

本次改动的核心点：

- 匹配概率与 Bellman 共用同一套 `sigma` 构造与 `S/D` 状态投影；
- 新增 `calibrate_lambda_intercept()`，用于根据目标 job-finding rate 反解统一截距；
- 新增 `calibrate_eta0()`，用于根据目标 separation rate 反解离职率截距；
- `compute_separation_rates()` 改为与 Bellman 同口径的 sigmoid 计算方式；
- `statistics` 输出中新增 `lambda_intercept_shift` 与 `eta0`，方便追踪结构参数。

### 3.4 `equilibrium_solver.py`

本次改动的核心点：

- 新增 `_sync_transition_parameters()`，保证 Bellman 与 KFE 同步使用同一套结构截距；
- 新增 `_calibrate_transition_intercepts()`，负责在参考样本上先校准 `lambda`，再校准 `eta0`；
- 初始化人口时改成两步：
  - 第一步：全体失业样本上校准匹配概率截距，再做初始随机匹配；
  - 第二步：基于初始就业样本反解 `eta0`；
- 均衡输出与 `equilibrium_summary.pkl` 中新增 `lambda_intercept_shift` 与 `eta0`。

---

## 4. 关键调用流程

### 4.1 初始化阶段

1. `EquilibriumSolver.initialize_population()`
2. `_calibrate_transition_intercepts(..., calibrate_lambda=True, calibrate_eta0=False)`
3. `KFESolver.compute_match_probabilities()`
4. 初始随机匹配
5. `_calibrate_transition_intercepts(..., calibrate_lambda=False, calibrate_eta0=True)`

### 4.2 外层迭代阶段

1. `BellmanSolver.solve()`
2. `BellmanSolver.compute_match_probabilities_batch()`
3. `KFESolver.evolve()`
4. `KFESolver.compute_match_probabilities()`
5. `KFESolver.compute_separation_rates()`

---

## 5. 输入输出示例

### 5.1 配置输入示例

来自 `CONFIG/mfg_config.yaml` 的新增关键结构：

```yaml
economics:
  separation_rate:
    auto_calibrate_eta0: false
    target_rate: 0.0028
    eta0_bounds: [-20.0, 5.0]
market:
  match_probability:
    auto_calibrate_intercept: false
    target_rate: 0.0285
    intercept_shift: 0.0
    intercept_bounds: [-20.0, 5.0]
```

说明：

- `auto_calibrate_eta0 = false`
- `auto_calibrate_intercept = false`

表示这两套结构修正能力默认不进入当前主结果链路。
`TESTS/test_transition_intercepts.py` 会在临时配置中单独打开它们，用于验证实现本身未失效。

### 5.2 均衡输出示例

`eq_info` 现在会额外包含：

```python
{
    "converged": False,
    "iterations": 200,
    "final_unemployment_rate": 0.08,
    "final_theta": 1.5,
    "lambda_intercept_shift": -4.17,
    "eta0": -5.50,
    "final_statistics": {...},
    "history": {...}
}
```

### 5.3 测试输出含义

`TESTS/test_transition_intercepts.py` 会校验：

- `JOB_FINDING_MEAN` 是否贴近 `0.0285`
- `SEPARATION_MEAN` 是否贴近 `0.0028`
- Bellman 与 KFE 对失业者的 `lambda` 是否一致

---

## 6. 本轮结论

这轮修复解决的是“结构上根本对不齐”的问题，而不是直接保证最终论文数值一定漂亮：

- 现在模型具备把 `lambda` 压回目标量级的结构能力；
- 现在模型具备把 `eta0` 反解到目标分离率的结构能力；
- 现在 Bellman 与 KFE 的 `S/D` 更新规则一致；
- 当前正式配置默认关闭 `M5/M6` 对齐开关，后续是否启用取决于研究口径。

---

## 7. 更新日志

### 2026-03-13

- 新增 `matching_utils.py`，统一 `sigma` 构造、`S/D` 投影和 Logit 平移项反解。
- 修改 `bellman_solver.py`，引入匹配截距平移与更低的离职率下界。
- 修改 `kfe_solver.py`，支持自动校准 `lambda` 截距与 `eta0`。
- 修改 `equilibrium_solver.py`，在初始化阶段自动完成结构截距同步与校准。
- 修改 `mfg_config.yaml`，新增匹配/离职率自动校准配置。
- 新增 `test_transition_intercepts.py`，用于快速结构自检。

### 2026-03-14

- 因当前主口径不再使用 `M5/M6`，正式配置默认关闭
  `auto_calibrate_intercept` 与 `auto_calibrate_eta0`。
- 保留结构修正函数与专项测试，仅供后续诊断或扩展时开启。
