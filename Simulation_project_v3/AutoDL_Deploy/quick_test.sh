#!/bin/bash

# 快速测试脚本
# 用途：在AutoDL上快速测试环境是否配置正确

echo "=========================================="
echo "环境快速测试"
echo "=========================================="

# 设置编码
export PYTHONIOENCODING=utf-8

# 测试1: Python环境
echo ""
echo "[测试 1/5] Python环境..."
python3 --version
if [ $? -eq 0 ]; then
    echo "✓ Python可用"
else
    echo "✗ Python不可用"
    exit 1
fi

# 测试2: 必需库
echo ""
echo "[测试 2/5] 检查必需库..."
python3 -c "
import numpy
import pandas
import scipy
import numba
import yaml
print('✓ 所有核心库已安装')
print(f'  - NumPy: {numpy.__version__}')
print(f'  - Pandas: {pandas.__version__}')
print(f'  - SciPy: {scipy.__version__}')
print(f'  - Numba: {numba.__version__}')
"

if [ $? -ne 0 ]; then
    echo "✗ 缺少必需库，请运行 autodl_setup.sh"
    exit 1
fi

# 测试3: 项目文件
echo ""
echo "[测试 3/5] 检查项目文件..."
required_dirs=("CONFIG" "MODULES" "DATA" "TESTS" "OUTPUT")
missing_dirs=()

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    echo "✓ 项目文件结构完整"
else
    echo "✗ 缺少目录: ${missing_dirs[*]}"
    echo "  请确保上传了完整的项目文件"
    exit 1
fi

# 测试4: 运行简单测试
echo ""
echo "[测试 4/5] 运行人口分布模块测试..."
timeout 60 python3 TESTS/test_population.py > /tmp/test_output.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 人口分布模块测试通过"
else
    echo "✗ 测试失败，查看日志："
    tail -20 /tmp/test_output.log
    exit 1
fi

# 测试5: CPU性能
echo ""
echo "[测试 5/5] CPU信息..."
echo "CPU核心数: $(nproc)"
echo "内存信息:"
free -h | grep Mem

echo ""
echo "=========================================="
echo "✓ 所有测试通过！环境配置正确"
echo "=========================================="
echo ""
echo "您现在可以："
echo "1. 运行完整测试: python3 TESTS/test_equilibrium_solver.py"
echo "2. 启动校准: ./run_calibration.sh"
echo ""

