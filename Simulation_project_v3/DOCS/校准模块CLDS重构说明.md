# 校准模块CLDS重构说明

## 文档定位
- 模块范围：`MODULES/CALIBRATION/target_moments.py`、`objective_function.py`、`smm_calibrator.py`、`MODULES/MFG/kfe_solver.py`
- 目标：在不大改项目结构前提下，将校准流程切换到 CLDS M1-M8 口径，并落地 Step0-Step5 多步走。

## 上下游关系
- 上游依赖：
  - `CONFIG/target_moments.yaml`（CLDS目标矩 + 标准误）
  - `CONFIG/calibration_config.yaml`（权重、两阶段、Jacobian配置）
  - `MODULES/MFG/equilibrium_solver.py`（提供均衡求解入口）
- 下游产出：
  - `OUTPUT/calibration/jacobian_analysis.csv`
  - `OUTPUT/calibration/parameter_partition.yaml`
  - `OUTPUT/calibration/moment_covariance_stage5.csv`
  - `OUTPUT/calibration/calibration_stage_summary.yaml`
  - `OUTPUT/calibration/calibrated_parameters.yaml`

## 功能概览
- CLDS矩口径：正式支持 M1-M8（失业率、工资均值/标准差、工时均值/标准差、就业转移率、分离率、工资IQR比值）。
- 权重体系：支持 `inverse_variance_bootstrap`（Step4）与 `efficient_from_covariance`（Step5）。
- 多步流程：
  - Step0：数值微扰 Jacobian 敏感性预分析。
  - Step1：按敏感性阈值自动分类外部/内部参数。
  - Step2：外部参数固定（取当前基准值，后续可替换文献值）。
  - Step4：鲁棒权重阶段优化。
  - Step5：协方差逆权重精修阶段优化。

## 架构与调用链
- `TargetMoments`
  - 读取 CLDS 目标矩配置与元信息（含 `bootstrap_se`）。
  - 计算 MFG 输出对应的模拟矩向量。
- `ObjectiveFunction`
  - 计算 `J(θ) = (m_sim-m_target)'W(m_sim-m_target)`。
  - 支持按 PID 落盘历史（并行场景）。
- `SMMCalibrator`
  - 组织 Step0-Step5 全流程。
  - 自动 workers 解析：`auto/all/-1` => `os.cpu_count()`。
  - 支持内部参数优化 + 外部参数固定映射。
- `KFESolver.evolve`
  - 新增 `job_finding_rate`、`separation_rate` 的期望值与实现值统计，供 M5/M6 对接。

## 输入输出示例
- 目标矩配置输入（节选）
```yaml
moments:
  unemployment_rate:
    tag: M1
    value: 0.035        # 2026-03-10 由1.73% 调整为 3.5%（见校准目标矩裁剪说明.md）
    bootstrap_se: 0.010
  log_std_wage:         # 2026-03-10 由 std_wage（2591.6元）替换为对数工资标准差
    tag: M3
    value: 0.649
    bootstrap_se: 0.020
```

- 阶段摘要输出（节选）
```yaml
parameter_partition:
  internal: [rho, kappa, alpha_T, gamma_T, gamma_S]
  external: [gamma_D, gamma_W]
stage4:
  success: true
```

## 测试覆盖
- `TESTS/test_target_moments_clds.py`：M1-M8模拟矩与SE向量基础校验。
- `TESTS/test_weight_matrix_clds.py`：逆方差权重、协方差逆权重、对角自定义权重校验。
- `TESTS/test_calibration_pipeline_clds.py`：mock求解器下的 Step0-Step5 流程烟雾测试。
- `TESTS/verify_parallel_config.py`：workers=auto + Numba单线程配置验证。

## 已知限制
- `CONFIG/target_moments.yaml` 中 M4/M8 的 `bootstrap_se` 仍沿用 CLDS 原始口径的标准误（未随数值调整更新），精确 SE 需从修正后口径重新 Bootstrap 计算。
- Step2 外部参数值当前默认取基准配置值，如需文献值可在 Step1 覆盖配置中指定。
- 当前 `log_std_wage` 的目标值（0.649）由对数正态近似推导，非直接从 CLDS 微数据计算，存在分布假设误差。

## 更新日志
- **2026-03-10**（最新）
  - 目标矩数值全面调整（详见 `校准目标矩裁剪说明.md`）：M1→3.5%、M3→`log_std_wage`(0.649)、M4→52h、M8→16.7h。
  - 权重体系从 `inverse_variance_bootstrap` 切换为手工对角权重（`diagonal`）。
  - 新增并行保护机制（`NUMBA_NUM_THREADS` 自动计算）。
- **2026-03-02**
  - 切换到 CLDS M1-M8 目标矩与两阶段权重流程。
  - 新增 Jacobian 参数分类与阶段摘要输出。
  - KFE新增 M5/M6 所需转移率统计。
  - 新增 3 个 CLDS 相关测试脚本并更新并行验证脚本。
