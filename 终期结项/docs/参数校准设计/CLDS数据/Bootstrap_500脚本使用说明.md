# CLDS Bootstrap 500 使用说明

## 1. 脚本位置
- `终期结项/docs/参数校准设计/CLDS数据/clds_bootstrap_moments.py`

## 2. 功能说明
- 对 CLDS 口径的目标矩做 Bootstrap 标准误估计；
- 默认估计 M1-M8；
- 可选估计 M9（`--include-m9`）；
- 输出含 `bootstrap_se` 的 YAML 文件。

## 3. 默认输入输出
- 输入基准文件：`target_moments_clds.yaml`
- 默认输出文件：`target_moments_clds_bootstrap500.yaml`

## 4. 运行命令（500次）
```bash
cd "终期结项/docs/参数校准设计/CLDS数据"
python clds_bootstrap_moments.py --n-bootstrap 500 --seed 20260302
```

## 5. 常用参数
- `--n-bootstrap`：Bootstrap次数，默认 `500`
- `--seed`：随机种子，默认 `20260302`
- `--base-yaml`：基准目标矩 YAML 路径
- `--output`：输出 YAML 路径
- `--include-m9`：同时估计 M9 的 `bootstrap_se`
- `--quiet`：静默模式

## 6. 结果校验
```bash
grep -n "bootstrap_se" target_moments_clds_bootstrap500.yaml
```

## 7. 与 v3 校准对接
- 将输出 YAML 里的 `moments.*.bootstrap_se` 同步到：
  - `Simulation_project_v3/CONFIG/target_moments.yaml`
- 然后在 v3 中使用：
  - `weight_type: inverse_variance_bootstrap`
- 若你希望强制每个矩都必须有 `bootstrap_se`，将配置项设为：
  - `strict_bootstrap_se: true`

## 8. 异常说明
- 如果某个矩在多次抽样中有效次数不足（`valid_draws < 2`），脚本会直接报错退出；
- 常见原因是面板基数过小或某次抽样后口径无法计算，建议：
  - 提高样本稳定性（检查筛选口径）；
  - 增大 `n-bootstrap`；
  - 先观察日志中哪个矩失败频繁，再决定是否调整口径或保留该矩。
