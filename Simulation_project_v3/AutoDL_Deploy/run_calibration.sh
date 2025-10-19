#!/bin/bash

# AutoDL校准运行脚本
# 用途：在后台运行校准任务，避免SSH断开影响

echo "=========================================="
echo "启动校准任务"
echo "=========================================="

# 设置编码
export PYTHONIOENCODING=utf-8

# 创建输出目录
mkdir -p OUTPUT/calibration

# 使用screen在后台运行
echo ""
echo "正在创建screen会话..."
screen -dmS calibration bash -c "
    cd $(pwd)
    export PYTHONIOENCODING=utf-8
    echo '校准开始时间: $(date)' > OUTPUT/calibration/calibration_run.log
    python3 TESTS/test_calibration.py 2>&1 | tee -a OUTPUT/calibration/calibration_run.log
    echo '校准结束时间: $(date)' >> OUTPUT/calibration/calibration_run.log
"

echo ""
echo "✓ 校准任务已在后台启动！"
echo ""
echo "常用命令："
echo "  查看运行状态: screen -r calibration"
echo "  分离会话:     按 Ctrl+A 再按 D"
echo "  查看日志:     tail -f OUTPUT/calibration/calibration_run.log"
echo "  监控资源:     htop"
echo ""
echo "=========================================="

