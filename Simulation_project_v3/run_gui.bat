@echo off
chcp 65001 >nul
echo ====================================
echo  EconLab 启动脚本
echo ====================================
echo.

cd /d "%~dp0"

echo 设置UTF-8编码...
set PYTHONIOENCODING=utf-8

echo 启动EconLab GUI...
D:\Python\2025DaChuang\venv\Scripts\python.exe GUI\app.py

pause

