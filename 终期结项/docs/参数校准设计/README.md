# 参数校准设计文档

**项目名称**：农村女性就业市场MFG模拟系统 v3.0  
**创建日期**：2025-07-01
**更新日期**：2026-03-03
**当前状态**：外部参数已定值，4维内部SMM校准待运行

---

## 文档目录

| 编号 | 文件名 | 内容概要 |
| --- | --- | --- |
| 01 | `01_校准方法论总览.md` | SMM方法原理、两阶段校准流程设计、实现状态评估、参考文献 |
| 02 | `02_目标矩选择与数据来源.md` | 当前问题诊断、目标矩选择原则、推荐8矩方案、各矩数据来源与目标值、权重矩阵设计 |
| 03 | `03_中国微观数据库可用性评估.md` | CLDS/CFPS/CHARLS/CHIP/统计年鉴的详细评估、综合对比矩阵、推荐数据使用策略 |
| 04 | `04_Jacobian敏感性分析方案.md` | Jacobian数值计算方案、弹性矩阵、参数分类决策标准、预期结果分析 |
| 05 | `05_CLDS数据结构与探索结果.md` | CLDS 2016+2018变量核验、M1-M9实算结果、口径差异诊断、更新日志 |
| 06 | `06_参数校准模块标准运行顺序.md` | 基于当前代码实现的标准运行顺序、输入输出、关键产物与常见注意事项 |

---

## 核心结论摘要

### 校准策略

当前代码采用 **Jacobian elbow 驱动的两阶段校准法**：

1. **Step 0**：Jacobian敏感性预分析（已完成，**elbow识别后已关闭**）
2. **Step 1**：参数分类（**已硬编码 override**，直接执行）
3. **Step 4**：鲁棒阶段优化（bootstrap逆方差权重，4维DE）
4. **Step 5**：高效阶段优化（基于协方差逆矩阵）
5. 输出阶段性文件与最终参数文件（并更新`mfg_config.yaml`）

### 参数分类（最终确定）

| 类型 | 参数 | 固定值/初始值 | 依据 |
|------|------|-------------|------|
| **外部（固定）** | `kappa` | **1500.0** | Christensen et al. (2005, *Econometrica*) |
| **外部（固定）** | `alpha_T` | **0.25** | Rogerson & Wallenius (2009, *JME*) |
| **外部（固定）** | `gamma_S` | **0.35** | Heckman, Lochner & Taber (1998, *AER*) |
| **内部（SMM优化）** | `gamma_W` | 0.15 | — |
| **内部（SMM优化）** | `gamma_T` | 0.30 | — |
| **内部（SMM优化）** | `gamma_D` | 0.45 | — |
| **内部（SMM优化）** | `rho` | 0.40 | — |

### 数据来源

- **首选**：CLDS（中国劳动力动态调查，15-64岁劳动年龄人口）
- **交叉验证**：CFPS（中国家庭追踪调查）
- **宏观矩**：国家统计局 / 统计年鉴
- **工资补充**：CHIP（中国家庭收入调查）

### 关键参考文献

- Andrews, Gentzkow & Shapiro (2017, QJE) — 参数敏感性度量方法
- Shimer (2005, AER) — 劳动搜索模型校准范式
- McFadden (1989, Econometrica) — SMM理论基础
- Hansen (1982, Econometrica) — GMM/过度识别检验
- Rogerson & Wallenius (2009, JME) — 工时负效用校准
- Heckman, Lochner & Taber (1998, AER) — 人力资本积累参数
- Christensen et al. (2005, Econometrica) — 搜寻成本校准

---

## 实现进度（截至 2026-03-03）

- [x] CLDS口径目标矩（M1-M8）接入 `CONFIG/target_moments.yaml`
- [x] `target_moments.py` 支持 M1-M8 模拟矩计算
- [x] `target_moments.yaml` 同步 500 次 bootstrap 标准误（`bootstrap_se`）
- [x] Step0 Jacobian敏感性分析（elbow识别完成，已关闭重跑）
- [x] Step1 参数分类：elbow结果硬编码至 `external_params_override`
- [x] 外部参数文献依据定值（kappa=1500, alpha_T=0.25, gamma_S=0.35）
- [x] 两阶段优化代码（Step4 bootstrap逆方差 + Step5协方差逆）
- [x] checkpoint与auto-resume机制
- [ ] **正式4维SMM校准运行**（内部参数：gamma_W/gamma_T/gamma_D/rho）
- [ ] 形成正式结果落盘文件（`calibrated_parameters.yaml`、`optimization_result.pkl`）
- [ ] Hansen J检验（过度识别检验）
- [ ] 完整真实数据长程校准并形成稳定最终结果
- [ ] 提升M5样本量（当前分母n=12，低置信度）
