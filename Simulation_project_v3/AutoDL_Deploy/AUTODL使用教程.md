# AutoDL平台使用完整教程（零基础版）

**适用人群**：完全没有接触过云平台和Linux环境的Windows用户  
**项目**：Simulation_project_v3 - 农村女性就业市场MFG模拟系统  
**创建时间**：2025年10月

---

## 📋 准备工作清单

在开始之前，请确保：

- [ ] 有一个邮箱（用于注册）
- [ ] 准备50-100元充值（建议先充50元测试）
- [ ] 本地项目代码完整（在 `D:\Python\2025DaChuang\Simulation_project_v3\`）
- [ ] 安装了Chrome或Edge浏览器
- [ ] Windows 10/11系统（已安装PowerShell）

---

## 第一部分：注册与充值（约10分钟）

### Step 1: 注册AutoDL账号

1. **打开浏览器**，访问：https://www.autodl.com
2. **点击右上角"注册"**按钮
3. **填写注册信息**：
   - 邮箱地址（建议使用QQ邮箱或163邮箱）
   - 设置密码（建议包含大小写字母+数字）
   - 输入验证码
4. **验证邮箱**：
   - 登录您的邮箱
   - 找到AutoDL发来的验证邮件
   - 点击验证链接完成激活

### Step 2: 实名认证（可选但推荐）

1. **登录AutoDL**
2. **进入"个人中心"**（右上角头像 → 个人中心）
3. **点击"实名认证"**
4. **按提示上传身份证照片**并填写信息
5. **等待审核**（通常5分钟内完成）

> **为什么要实名认证？**
> - 部分优惠需要实名
> - 提高账户安全性
> - 获得更多支持服务

### Step 3: 账户充值

1. **进入"费用中心"**（右上角 → 费用中心）
2. **点击"充值"**按钮
3. **选择充值金额**：
   - 建议首次充值：**50元**（用于测试）
   - 预计消耗：2-5元/次校准运行
4. **选择支付方式**：
   - 支付宝
   - 微信支付
   - 银行卡
5. **完成支付**

---

## 第二部分：创建CPU实例（约5分钟）

### Step 4: 进入控制台

1. **登录AutoDL后**，点击顶部导航栏的"**控制台**"
2. 您会看到一个实例列表页面（刚开始是空的）

### Step 5: 创建新实例

1. **点击"创建实例"**或"租用新实例"按钮

2. **选择区域**：
   - 推荐选择：**华东** 或 **华北**（距离北京较近）
   - 看哪个区域有CPU实例可用

3. **选择实例类型**：
   - ⚠️ **注意**：不要选GPU实例！
   - 找到 **"CPU实例"** 或 **"纯CPU"** 选项卡
   - 选择配置：**8核CPU + 16GB内存** 或 **16核CPU + 32GB内存**

4. **选择镜像**：
   - 找到 **"基础镜像"** 或 **"官方镜像"**
   - 选择：**Ubuntu 20.04** 或 **Ubuntu 22.04**
   - 带Python 3.8/3.10的镜像更佳

5. **存储设置**：
   - 系统盘：**40GB**（足够用）
   - 数据盘：可选（我们的项目不需要）

6. **实例名称**：
   - 输入一个好记的名字，如：`MFG_Calibration`

7. **付费方式**：
   - 选择：**按量付费**（用多少付多少）
   - 不要选包月（不划算）

8. **确认创建**：
   - 检查配置无误
   - 查看预估费用（应该在0.1-0.3元/小时）
   - **点击"立即创建"**

9. **等待实例启动**：
   - 状态会从"创建中" → "启动中" → "运行中"
   - 通常需要**1-2分钟**

---

## 第三部分：连接实例（约10分钟）

### Step 6: 获取连接信息

1. **在实例列表中**，找到刚创建的实例
2. **记录以下信息**：
   - **IP地址**：如 `120.26.xxx.xxx`
   - **SSH端口**：通常是 `22` 或其他端口号
   - **用户名**：通常是 `root`
   - **密码**：通常在实例详情页可以看到或重置

> **提示**：点击实例名称可以查看详细信息

### Step 7: 使用PowerShell连接（Windows 10/11）

**Windows 10/11自带SSH客户端，不需要额外下载软件！**

1. **打开PowerShell**：
   - 按 `Win + X`
   - 选择"Windows PowerShell"或"终端"

2. **输入SSH连接命令**：
   ```powershell
   ssh root@120.26.xxx.xxx
   ```
   （将IP地址替换为您的实例IP）

3. **如果端口不是22**：
   ```powershell
   ssh -p 端口号 root@120.26.xxx.xxx
   ```

4. **首次连接会提示**：
   ```
   The authenticity of host '120.26.xxx.xxx' can't be established.
   Are you sure you want to continue connecting (yes/no)?
   ```
   **输入 `yes` 并按回车**

5. **输入密码**：
   - 粘贴AutoDL提供的密码（右键粘贴，看不见是正常的）
   - 按回车

6. **成功连接后**，您会看到类似这样的提示符：
   ```bash
   root@autodl-container-xxxx:~#
   ```

> **恭喜！您已经成功连接到云服务器了！**

---

## 第四部分：上传项目代码（约10分钟）

### 方法A：使用AutoDL的JupyterLab上传（推荐新手）

1. **在AutoDL控制台**，找到您的实例
2. **点击"打开JupyterLab"**按钮（或类似的Web访问按钮）
3. **在JupyterLab界面**：
   - 左侧是文件浏览器
   - 点击"上传"图标（向上箭头）
4. **上传项目文件夹**：
   - ⚠️ **不要一次性上传整个文件夹**（可能失败）
   - 先上传 `requirements.txt`
   - 再分别上传各个文件夹（CONFIG、MODULES、DATA、TESTS等）

### 方法B：使用WinSCP上传（更稳定）

1. **下载WinSCP**：
   - 访问：https://winscp.net/eng/download.php
   - 下载并安装（免费软件）

2. **打开WinSCP**：
   - 文件协议：SFTP
   - 主机名：您的实例IP
   - 端口：22（或AutoDL提供的端口）
   - 用户名：root
   - 密码：AutoDL提供的密码

3. **点击"登录"**

4. **上传文件**：
   - 左侧是本地文件（找到 `D:\Python\2025DaChuang\Simulation_project_v3\`）
   - 右侧是远程服务器（默认在 `/root/`）
   - **直接拖拽整个文件夹到右侧**
   - 等待上传完成（可能需要5-10分钟）

### 方法C：使用Git（最优雅）

如果您的代码已经上传到GitHub：

1. **在SSH连接中**，输入：
   ```bash
   git clone https://github.com/您的用户名/Simulation_project_v3.git
   ```

2. **等待下载完成**

---

## 第五部分：配置环境（约10分钟）

### Step 8: 运行自动配置脚本

1. **确认您已经SSH连接到实例**

2. **进入项目目录**：
   ```bash
   cd Simulation_project_v3
   ```

3. **查看文件是否上传完整**：
   ```bash
   ls
   ```
   应该能看到：CONFIG、MODULES、DATA、TESTS、requirements.txt等

4. **给脚本添加执行权限**：
   ```bash
   chmod +x autodl_setup.sh
   ```

5. **运行配置脚本**：
   ```bash
   ./autodl_setup.sh
   ```

6. **等待安装完成**（约5-10分钟）：
   - 您会看到安装进度信息
   - 最后会显示"环境配置完成！"

> **如果脚本执行失败**，请手动执行以下命令：
> ```bash
> apt-get update -y
> apt-get install -y python3-pip git htop screen
> pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 第六部分：运行校准任务（约10-20小时）

### Step 9: 启动校准任务

**方法1：使用自动化脚本（推荐）**

1. **给运行脚本添加执行权限**：
   ```bash
   chmod +x run_calibration.sh
   ```

2. **启动校准**：
   ```bash
   ./run_calibration.sh
   ```

3. **您会看到提示信息**：
   ```
   ✓ 校准任务已在后台启动！
   ```

**方法2：手动启动（用于学习）**

1. **创建screen会话**：
   ```bash
   screen -S calibration
   ```

2. **设置编码**：
   ```bash
   export PYTHONIOENCODING=utf-8
   ```

3. **运行校准**：
   ```bash
   python3 TESTS/test_calibration.py
   ```

4. **分离会话**：
   - 按 `Ctrl + A`
   - 再按 `D`
   - 您会回到普通命令行，但程序仍在后台运行

### Step 10: 监控运行状态

**查看日志**：
```bash
tail -f OUTPUT/calibration/calibration_run.log
```
（按 `Ctrl + C` 退出查看）

**查看CPU和内存使用**：
```bash
htop
```
（按 `q` 退出）

**重新连接到screen会话**：
```bash
screen -r calibration
```

**查看所有screen会话**：
```bash
screen -ls
```

---

## 第七部分：下载结果（约5分钟）

### Step 11: 下载输出文件

**方法1：使用WinSCP（推荐）**

1. **打开WinSCP并连接**（如前面所述）
2. **在右侧导航到**：`/root/Simulation_project_v3/OUTPUT/calibration/`
3. **选中所有文件**
4. **拖拽到左侧本地文件夹**
5. **等待下载完成**

**方法2：使用scp命令（在本地PowerShell中）**

```powershell
scp -r root@120.26.xxx.xxx:/root/Simulation_project_v3/OUTPUT/calibration D:\Python\2025DaChuang\Simulation_project_v3\OUTPUT\
```

---

## 第八部分：停止实例（节省费用）

### Step 12: 关闭实例

1. **确认已下载所有需要的文件**
2. **在AutoDL控制台**，找到您的实例
3. **点击"关机"或"停止"**按钮
4. **确认操作**

> **⚠️ 重要提示**：
> - 关机后数据仍保留，重新开机可继续使用
> - 关机后按小时计费停止
> - 如果完全不需要了，可以选择"释放"或"删除"（数据会被清除）

---

## 🔧 常见问题解答

### Q1: SSH连接不上怎么办？
**A**: 
- 检查实例是否处于"运行中"状态
- 确认IP地址和端口号正确
- 尝试在AutoDL控制台重置密码
- 检查本地网络是否正常

### Q2: 上传文件很慢怎么办？
**A**: 
- 使用WinSCP而不是JupyterLab
- 检查本地网络速度
- 考虑使用Git方式（需要先把代码传到GitHub）

### Q3: 安装依赖时出错怎么办？
**A**: 
```bash
# 使用国内镜像源
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 如何知道校准任务完成了？
**A**: 
- 查看日志最后是否有"校准结束时间"
- 查看 `OUTPUT/calibration/calibrated_parameters.yaml` 是否生成
- 使用 `screen -r calibration` 连接到会话查看

### Q5: SSH断开了程序会停止吗？
**A**: 
- 如果使用了 `screen`，程序**不会**停止
- 如果直接运行，程序**会**停止
- 这就是为什么我们使用screen的原因

### Q6: 费用会不会很贵？
**A**: 
- CPU实例约0.2-0.3元/小时
- 运行10小时约2-3元
- 记得任务完成后及时关机

---

## 📖 Linux基础命令速查

**文件操作**：
```bash
ls              # 列出当前目录文件
cd 目录名        # 进入目录
cd ..           # 返回上级目录
pwd             # 显示当前路径
mkdir 目录名     # 创建目录
rm 文件名        # 删除文件
```

**查看文件内容**：
```bash
cat 文件名       # 显示全部内容
head 文件名      # 显示前10行
tail 文件名      # 显示后10行
tail -f 文件名   # 实时查看（适合日志）
```

**系统监控**：
```bash
htop            # 查看CPU和内存（按q退出）
df -h           # 查看磁盘空间
free -h         # 查看内存使用
```

**进程管理**：
```bash
ps aux          # 查看所有进程
kill PID        # 结束进程（PID是进程号）
```

---

## 📞 获取帮助

如果遇到问题：

1. **AutoDL官方文档**：https://www.autodl.com/docs/
2. **AutoDL客服**：在网站右下角有在线客服
3. **本项目问题**：检查 `DOCS/` 文件夹中的文档

---

## ✅ 完整流程检查清单

- [ ] 注册AutoDL账号并充值
- [ ] 创建8核16GB CPU实例
- [ ] 使用SSH成功连接
- [ ] 上传项目代码
- [ ] 运行autodl_setup.sh配置环境
- [ ] 启动校准任务（使用screen）
- [ ] 监控任务运行状态
- [ ] 下载结果文件
- [ ] 关闭实例

---

**祝您使用顺利！有问题随时询问。**

