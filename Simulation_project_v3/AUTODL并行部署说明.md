# AutoDL并行校准部署说明

## 📋 修改概述

为了在AutoDL云服务器上充分利用多核CPU进行高效校准，本项目进行了以下核心修改：

### 核心策略转变

**从Numba级并行 → 进程级并行**

- ❌ **原方案失败原因**：Nelder-Mead串行优化算法，每次只评估1个参数点，即使单个MFG求解用了Numba并行，128核心仍大部分空闲
- ✅ **新方案**：使用Differential Evolution并行优化算法，每代同时评估32个参数点，32个独立进程并行运行

## 🔧 修改的文件

### 1. 核心算法修改

**`CONFIG/calibration_config.yaml`**
```yaml
optimization:
  method: differential_evolution  # 改为并行优化算法
  options:
    popsize: 32   # 种群大小
    workers: 32   # 32进程并行
```

**`MODULES/CALIBRATION/smm_calibrator.py`**
- 添加 `differential_evolution` 支持
- 自动识别并行/串行算法

**`TESTS/test_calibration.py`**
- 添加Numba环境变量优化配置

### 2. AutoDL部署支持

**新增 `AutoDL_Deploy/` 文件夹**，包含：
- 运行脚本：`force_parallel_run.sh`（核心）
- 环境设置：`autodl_setup.sh`
- 监控工具：`monitor_script.sh`
- 通知工具：`email_notify.sh`
- 详细文档：多个md文档和教程

## 📊 预期效果

| 指标 | 原方案 | 新方案 | 改善 |
|------|--------|--------|------|
| 优化算法 | Nelder-Mead | Differential Evolution | 串行→并行 |
| 单次评估点数 | 1 | 32 | 32倍 |
| 进程数 | 1 | 32 | 32倍 |
| 预计运行时间 | ~7天 | ~0.5-1天 | **7-14倍加速** |
| CPU利用率 | <10% | 50-80% | 5-8倍提升 |

## 🚀 快速部署

### 步骤1：上传文件到AutoDL

使用WinSCP上传以下3个文件：
1. `TESTS/test_calibration.py`
2. `CONFIG/calibration_config.yaml`
3. `MODULES/CALIBRATION/smm_calibrator.py`

### 步骤2：运行校准

```bash
cd ~/Simulation_project_v3
bash AutoDL_Deploy/force_parallel_run.sh
```

### 步骤3：监控运行

```bash
htop  # 应看到30+个python进程
```

## 📚 详细文档

- **快速开始**：`AutoDL_Deploy/AutoDL快速开始.txt`
- **详细教程**：`AutoDL_Deploy/AUTODL使用教程.md`
- **并行原理**：`AutoDL_Deploy/并行优化方案.txt`
- **完整指南**：`AutoDL_Deploy/最终部署方案.md`
- **文件清单**：`AutoDL_Deploy/修改文件清单.txt`

## ✅ 验证并行成功的标志

- ✅ htop中看到30+个python进程同时运行
- ✅ Load average在50-100之间
- ✅ 日志中显示"使用并行差分进化算法"
- ✅ 单代迭代时间显著低于原方案

## 🔧 故障排查

如果并行未启动：
1. 确认上传了最新的 `calibration_config.yaml`
2. 检查配置中 `method: differential_evolution`
3. 查看日志开头的算法类型输出
4. 运行 `ps aux | grep python | wc -l` 检查进程数

---

**修改日期**：2025/01/20  
**版本**：v3.0 - 进程并行方案  
**适用场景**：AutoDL/矩池云等多核服务器环境

