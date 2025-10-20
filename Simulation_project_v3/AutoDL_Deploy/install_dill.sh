#!/bin/bash
# 安装loky库以支持并行计算中的闭包序列化

echo "=================================="
echo "安装loky库"
echo "=================================="

pip3 install loky>=3.4.0

echo ""
echo "验证安装："
python3 -c "import loky; print(f'loky版本: {loky.__version__}')"

echo ""
echo "安装完成！"

