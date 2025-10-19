# AutoDL云平台使用指南

本项目已为AutoDL云平台部署准备好了完整的辅助工具和教程。

## 📁 准备的文件清单

### 📖 教程文档
1. **AUTODL使用教程.md** - 完整的零基础教程（推荐从这里开始！）
2. **AutoDL快速开始.txt** - 快速参考指南
3. **README_AUTODL.md** - 本文件

### 🔧 自动化脚本
4. **autodl_setup.sh** - 环境自动配置脚本（在服务器上运行）
5. **run_calibration.sh** - 校准任务启动脚本（在服务器上运行）
6. **quick_test.sh** - 快速测试脚本（在服务器上运行）
7. **upload_to_autodl.ps1** - Windows上传脚本（可选，推荐用WinSCP）

## 🚀 快速开始（3步上手）

### Step 1: 阅读教程
打开 `AUTODL使用教程.md`，这是一份详细的零基础教程，涵盖：
- AutoDL注册与充值
- 实例创建与连接
- 代码上传方法
- 环境配置步骤
- 任务运行与监控
- 结果下载
- 常见问题解答
- Linux基础命令

### Step 2: 上传项目到AutoDL
推荐使用 **WinSCP**（图形界面，最简单）：
1. 下载WinSCP: https://winscp.net/
2. 连接到AutoDL实例
3. 拖拽整个文件夹到服务器

### Step 3: 在服务器上执行
```bash
# 进入项目目录
cd Simulation_project_v3

# 配置环境
chmod +x autodl_setup.sh
./autodl_setup.sh

# 快速测试
chmod +x quick_test.sh
./quick_test.sh

# 运行校准
chmod +x run_calibration.sh
./run_calibration.sh
```

## 💰 预算参考

| 项目 | 配置 | 单价 | 预计时长 | 总费用 |
|------|------|------|---------|--------|
| 环境测试 | 8核16GB | 0.2-0.3元/h | 0.5小时 | 0.1-0.15元 |
| 单次校准 | 8核16GB | 0.2-0.3元/h | 10小时 | 2-3元 |
| 完整开发 | 8核16GB | 0.2-0.3元/h | 30-50小时 | 6-15元 |

**建议首次充值**：50元（足够完成多次测试和校准）

## ⚠️ 重要提示

1. **必须使用screen运行长时间任务**，否则SSH断开程序会停止
2. **任务完成后及时关机**，避免产生不必要的费用
3. **定期下载checkpoint文件**到本地备份
4. **首次使用先运行 quick_test.sh** 确保环境配置正确

## 📋 完整流程检查清单

- [ ] 注册AutoDL账号并充值50元
- [ ] 创建8核16GB CPU实例（Ubuntu 20.04）
- [ ] 使用SSH或WinSCP连接成功
- [ ] 上传项目代码到 `/root/Simulation_project_v3/`
- [ ] 运行 `autodl_setup.sh` 配置环境
- [ ] 运行 `quick_test.sh` 验证环境
- [ ] 运行 `run_calibration.sh` 启动校准
- [ ] 定期检查日志：`tail -f OUTPUT/calibration/calibration_run.log`
- [ ] 下载结果到本地
- [ ] 关闭AutoDL实例

## 🆘 遇到问题？

### 常见问题速查

**Q1: SSH连接不上**
```bash
# 检查列表：
✓ 实例状态是否"运行中"？
✓ IP地址是否正确？
✓ 密码是否正确？（可在控制台重置）
✓ 网络是否正常？
```

**Q2: 依赖安装失败**
```bash
# 使用国内镜像源重试
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q3: 程序运行中断**
```bash
# 检查是否使用了screen
screen -ls  # 查看所有会话
screen -r calibration  # 重新连接
```

**Q4: 如何知道任务完成**
```bash
# 方法1: 查看日志
tail -f OUTPUT/calibration/calibration_run.log

# 方法2: 检查输出文件
ls -lh OUTPUT/calibration/calibrated_parameters.yaml

# 方法3: 连接screen会话
screen -r calibration
```

### 获取更多帮助

1. **AutoDL官方文档**: https://www.autodl.com/docs/
2. **AutoDL在线客服**: 网站右下角
3. **项目文档**: `DOCS/` 目录
4. **详细教程**: `AUTODL使用教程.md`

## 📊 脚本说明

### autodl_setup.sh（必须运行）
**功能**：自动配置服务器环境
- 更新系统软件包
- 安装必要工具（htop, screen, git等）
- 设置时区为北京时间
- 配置Python UTF-8编码
- 安装项目依赖（使用清华镜像加速）
- 验证环境配置

**运行时间**：5-10分钟

### run_calibration.sh（启动任务）
**功能**：在后台启动校准任务
- 创建screen会话
- 设置环境变量
- 运行校准脚本
- 记录运行日志

**特点**：SSH断开不影响运行

### quick_test.sh（推荐先运行）
**功能**：快速测试环境配置
- 测试Python环境
- 检查必需库
- 验证项目文件
- 运行简单测试
- 显示系统信息

**运行时间**：1-2分钟

## 🎓 Linux命令速记卡

```bash
# 目录操作
cd Simulation_project_v3  # 进入目录
pwd                       # 显示当前路径
ls                        # 列出文件
ls -lh                    # 详细列出（含大小）

# 文件查看
cat 文件名                # 显示全部内容
tail -20 文件名           # 显示最后20行
tail -f 文件名            # 实时查看（日志）

# Screen会话管理
screen -S 名称            # 创建会话
screen -ls               # 列出所有会话
screen -r 名称           # 重新连接
Ctrl+A, D                # 分离会话（程序继续运行）

# 系统监控
htop                     # 查看CPU/内存（按q退出）
df -h                    # 查看磁盘空间
free -h                  # 查看内存使用

# 脚本执行
chmod +x 脚本名.sh       # 添加执行权限
./脚本名.sh              # 运行脚本
```

## 🎯 推荐工作流

### 首次使用（完整测试）
```bash
# 1. 配置环境
./autodl_setup.sh

# 2. 快速测试
./quick_test.sh

# 3. 运行单个测试
python3 TESTS/test_population.py
python3 TESTS/test_equilibrium_solver.py

# 4. 确认无误后运行完整校准
./run_calibration.sh
```

### 日常使用（快速启动）
```bash
# 1. 进入目录
cd Simulation_project_v3

# 2. 直接启动校准
./run_calibration.sh

# 3. 分离SSH
screen -r calibration
# 按 Ctrl+A, D

# 4. 定期检查
tail -f OUTPUT/calibration/calibration_run.log
```

## 📈 性能优化建议

1. **选择合适的实例配置**
   - 最小：8核16GB（约0.2元/小时）
   - 推荐：16核32GB（约0.4元/小时，速度更快）

2. **使用国内镜像源**
   - 已在 `autodl_setup.sh` 中配置清华源
   - 大幅加快依赖安装速度

3. **启用Numba并行**
   - 项目已配置 `parallel=True`
   - 自动利用多核CPU加速

4. **监控资源使用**
   - 使用 `htop` 查看CPU使用率
   - 确保所有核心都在工作

## ✅ 成功标志

当您完成校准后，应该看到：

```
OUTPUT/calibration/
├── calibrated_parameters.yaml  # 校准后的参数
├── calibration_history.csv     # 评估历史
├── calibration.log             # 详细日志
├── optimization_result.pkl     # 优化结果
└── checkpoint_latest.pkl       # 最新断点
```

在 `calibration.log` 最后应该显示：
```
优化完成
最优SMM距离: 0.xxxxxx
```

---

**准备开始了吗？打开 `AUTODL使用教程.md` 开始您的云平台之旅！**

有任何问题随时询问！

