#!/bin/bash

# AutoDL环境配置脚本
# 用途：在AutoDL实例上自动配置Python环境和安装依赖

echo "=========================================="
echo "开始配置AutoDL环境"
echo "=========================================="

# 1. 更新系统
echo ""
echo "[步骤 1/6] 更新系统软件包..."
apt-get update -y
apt-get upgrade -y

# 2. 安装必要工具
echo ""
echo "[步骤 2/6] 安装必要工具..."
apt-get install -y htop screen git vim wget curl

# 3. 设置时区为北京时间
echo ""
echo "[步骤 3/6] 设置时区为北京时间..."
timedatectl set-timezone Asia/Shanghai
echo "当前时间: $(date '+%Y/%m/%d %H:%M')"

# 4. 设置Python编码
echo ""
echo "[步骤 4/6] 配置Python UTF-8编码..."
export PYTHONIOENCODING=utf-8
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc

# 5. 检查Python版本
echo ""
echo "[步骤 5/6] 检查Python环境..."
python3 --version
pip3 --version

# 6. 安装项目依赖
echo ""
echo "[步骤 6/6] 安装项目依赖包..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "依赖安装完成！"
else
    echo "警告: 未找到requirements.txt文件"
fi

# 7. 测试Numba加速
echo ""
echo "=========================================="
echo "测试环境配置"
echo "=========================================="
python3 -c "import numpy; print('NumPy版本:', numpy.__version__)"
python3 -c "import numba; print('Numba版本:', numba.__version__)"
python3 -c "import pandas; print('Pandas版本:', pandas.__version__)"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "接下来您可以："
echo "1. 运行测试: python3 TESTS/test_population.py"
echo "2. 运行校准: screen -S calibration"
echo "             python3 TESTS/test_calibration.py"
echo "3. 查看日志: tail -f OUTPUT/calibration/calibration.log"
echo ""

