# EconLab PowerShell启动脚本

Write-Host "====================================" -ForegroundColor Cyan
Write-Host " EconLab 启动脚本" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# 进入项目目录
Set-Location $PSScriptRoot

# 设置UTF-8编码
$env:PYTHONIOENCODING = "utf-8"

Write-Host "启动EconLab GUI..." -ForegroundColor Green

# 直接使用虚拟环境的Python
& "D:\Python\2025DaChuang\venv\Scripts\python.exe" GUI\app.py

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

