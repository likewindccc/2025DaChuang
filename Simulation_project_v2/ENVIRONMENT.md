# 虚拟环境配置说明

**项目**: Simulation_project_v2  
**虚拟环境路径**: `D:\Python\2025DaChuang\venv`

---

## ⚠️ 重要提醒

本项目使用**项目专属虚拟环境**，请勿使用全局环境！

```
✅ 正确的虚拟环境: D:\Python\2025DaChuang\venv
❌ 不要使用: D:\Python\.venv （这是全局环境）
```

---

## 激活虚拟环境

### 方法一：使用快捷脚本（推荐）

双击或命令行运行：
```bash
cd D:\Python\2025DaChuang\Simulation_project_v2
activate_env.bat
```

### 方法二：手动激活

**PowerShell**:
```powershell
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1
```

**CMD**:
```cmd
D:\Python\2025DaChuang\venv\Scripts\activate.bat
```

---

## 验证环境

激活后检查：

```bash
# 查看Python路径（应该指向项目venv）
where python
# 应输出: D:\Python\2025DaChuang\venv\Scripts\python.exe

# 查看Python版本
python --version
# 应输出: Python 3.12.5

# 查看已安装包
pip list
```

---

## 依赖安装

**首次使用需要安装依赖**：

```bash
# 1. 激活虚拟环境
D:\Python\2025DaChuang\venv\Scripts\Activate.ps1

# 2. 进入项目目录
cd D:\Python\2025DaChuang\Simulation_project_v2

# 3. 安装依赖
pip install -r ../requirements.txt

# 4. 验证关键包
python -c "import numpy, pandas, numba; print('核心依赖安装成功')"
```

---

## 常见问题

### Q: 为什么要使用项目专属虚拟环境？

A: 
- 隔离依赖，避免版本冲突
- 本项目与其他项目（如2025国赛）依赖可能不同
- 便于团队协作和环境复现

### Q: 如何确认使用了正确的环境？

A: 激活后运行：
```bash
python -c "import sys; print(sys.executable)"
```
应输出：`D:\Python\2025DaChuang\venv\Scripts\python.exe`

### Q: activate_env.bat 做了什么？

A: 
1. 激活 `D:\Python\2025DaChuang\venv`
2. 显示Python版本确认
3. 保持窗口打开，方便后续操作

---

## 维护说明

如需更新依赖：

```bash
# 激活环境
activate_env.bat

# 更新单个包
pip install --upgrade numpy

# 更新所有包（谨慎）
pip list --outdated
pip install --upgrade 包名

# 导出新的依赖清单（如果有更新）
pip freeze > requirements_new.txt
```

---

**最后更新**: 2025-09-30  
**维护者**: 项目开发团队
