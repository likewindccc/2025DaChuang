# AutoDL项目上传脚本 (PowerShell)
# 用途：从Windows本地上传项目到AutoDL服务器

# ==================================================
# 配置区域（请修改以下信息）
# ==================================================

# AutoDL服务器信息（从AutoDL控制台获取）
$SERVER_IP = "120.26.xxx.xxx"  # 替换为您的实例IP
$SERVER_PORT = "22"             # SSH端口，通常是22
$SERVER_USER = "root"           # 用户名，通常是root

# 本地项目路径
$LOCAL_PATH = "D:\Python\2025DaChuang\Simulation_project_v3"

# 远程目标路径
$REMOTE_PATH = "/root/Simulation_project_v3"

# ==================================================
# 上传功能
# ==================================================

Write-Host "=========================================="
Write-Host "AutoDL项目上传工具"
Write-Host "=========================================="
Write-Host ""

Write-Host "本地路径: $LOCAL_PATH"
Write-Host "远程服务器: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
Write-Host "远程路径: $REMOTE_PATH"
Write-Host ""

# 提示用户确认
Write-Host "请确认以下信息是否正确："
Write-Host "1. 已在脚本中配置正确的服务器IP和端口"
Write-Host "2. 本地项目路径存在"
Write-Host "3. 已经能够SSH连接到服务器"
Write-Host ""

$confirmation = Read-Host "确认无误，开始上传？(输入 yes 继续)"

if ($confirmation -ne "yes") {
    Write-Host "取消上传。"
    exit
}

Write-Host ""
Write-Host "=========================================="
Write-Host "开始上传项目文件..."
Write-Host "=========================================="
Write-Host ""

# 使用scp上传（Windows 10/11自带）
Write-Host "提示：上传过程中需要输入服务器密码"
Write-Host ""

# 排除不必要的文件夹
$excludes = @(
    "__pycache__",
    ".git",
    "venv",
    ".venv",
    "*.pyc"
)

Write-Host "正在上传文件..."
Write-Host "（这可能需要几分钟，请耐心等待）"
Write-Host ""

# 使用tar压缩后上传（更快）
Write-Host "[方法1] 尝试使用tar压缩上传..."

# 创建临时压缩包
$tempFile = "$env:TEMP\simulation_project_v3.tar.gz"

# 切换到项目目录
Push-Location $LOCAL_PATH

# 检查是否安装了tar（Windows 10 1803+自带）
$tarExists = Get-Command tar -ErrorAction SilentlyContinue

if ($tarExists) {
    Write-Host "正在压缩项目文件..."
    
    # 排除不需要的文件
    tar -czf $tempFile `
        --exclude="__pycache__" `
        --exclude=".git" `
        --exclude="*.pyc" `
        --exclude=".venv" `
        --exclude="venv" `
        .
    
    Write-Host "✓ 压缩完成: $tempFile"
    Write-Host "正在上传到服务器..."
    
    # 上传压缩包
    if ($SERVER_PORT -eq "22") {
        scp $tempFile "${SERVER_USER}@${SERVER_IP}:/root/simulation_project_v3.tar.gz"
    } else {
        scp -P $SERVER_PORT $tempFile "${SERVER_USER}@${SERVER_IP}:/root/simulation_project_v3.tar.gz"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 上传成功！"
        Write-Host ""
        Write-Host "接下来请在SSH中执行以下命令解压："
        Write-Host ""
        Write-Host "  cd /root"
        Write-Host "  mkdir -p Simulation_project_v3"
        Write-Host "  tar -xzf simulation_project_v3.tar.gz -C Simulation_project_v3"
        Write-Host "  rm simulation_project_v3.tar.gz"
        Write-Host "  cd Simulation_project_v3"
        Write-Host "  chmod +x autodl_setup.sh"
        Write-Host "  ./autodl_setup.sh"
        Write-Host ""
    } else {
        Write-Host "✗ 上传失败，请检查网络连接和服务器信息"
    }
    
    # 清理临时文件
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    
} else {
    Write-Host "[方法2] tar不可用，使用scp直接上传..."
    Write-Host ""
    Write-Host "⚠️  这种方式较慢，建议使用WinSCP图形界面工具"
    Write-Host ""
    
    # 直接使用scp上传整个目录
    if ($SERVER_PORT -eq "22") {
        scp -r $LOCAL_PATH "${SERVER_USER}@${SERVER_IP}:/root/"
    } else {
        scp -P $SERVER_PORT -r $LOCAL_PATH "${SERVER_USER}@${SERVER_IP}:/root/"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 上传成功！"
    } else {
        Write-Host "✗ 上传失败"
    }
}

Pop-Location

Write-Host ""
Write-Host "=========================================="
Write-Host "上传完成！"
Write-Host "=========================================="
Write-Host ""
Write-Host "下一步："
Write-Host "1. 使用SSH连接到服务器："
Write-Host "   ssh ${SERVER_USER}@${SERVER_IP}"
Write-Host ""
Write-Host "2. 进入项目目录："
Write-Host "   cd Simulation_project_v3"
Write-Host ""
Write-Host "3. 运行配置脚本："
Write-Host "   chmod +x autodl_setup.sh"
Write-Host "   ./autodl_setup.sh"
Write-Host ""

Read-Host "按回车键退出"

