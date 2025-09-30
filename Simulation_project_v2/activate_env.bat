@echo off
REM 激活Simulation_project_v2项目虚拟环境
REM 使用方法: 在项目根目录双击此文件或在命令行运行

echo ================================================
echo   Simulation_project_v2 环境激活脚本
echo ================================================
echo.
echo 正在激活虚拟环境: D:\Python\2025大创\venv
echo.

call "D:\Python\2025大创\venv\Scripts\activate.bat"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ 虚拟环境激活成功！
    echo.
    echo 当前Python版本:
    python --version
    echo.
    echo 项目目录: %CD%
    echo.
    echo 可以开始开发了！
    echo ================================================
) else (
    echo.
    echo ✗ 虚拟环境激活失败！
    echo 请检查路径: D:\Python\2025大创\venv
    echo ================================================
)

REM 保持窗口打开
cmd /k
