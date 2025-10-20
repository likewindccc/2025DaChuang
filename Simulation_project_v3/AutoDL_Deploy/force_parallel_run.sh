#!/bin/bash
# 强制启用Numba并行的运行脚本

set -e

echo "=========================================="
echo "强制并行化校准运行脚本"
echo "=========================================="

# 1. 清除Numba缓存
echo "[步骤1] 清除Numba缓存..."
rm -rf ~/.cache/numba/*
rm -rf /tmp/numba_cache_*
echo "✓ Numba缓存已清除"

# 2. 设置环境变量（关键：禁用Numba并行，避免与DE的32进程冲突）
echo "[步骤2] 设置环境变量..."
export PYTHONIOENCODING=utf-8
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_DISABLE_JIT=0

echo "NUMBA_NUM_THREADS: $NUMBA_NUM_THREADS (设为1以配合DE的32进程并行)"
echo "NUMBA_THREADING_LAYER: $NUMBA_THREADING_LAYER"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# 3. 进入项目目录
cd ~/Simulation_project_v3

# 4. 先运行一个简单测试验证配置
echo "[步骤3] 运行配置验证..."
python3 -c "
import os
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
import numba
import numpy as np

print(f'Numba版本: {numba.__version__}')
print(f'线程数配置: {numba.config.NUMBA_NUM_THREADS} (串行模式)')
print(f'并行层: {numba.config.THREADING_LAYER}')

@numba.njit(parallel=True)
def test_parallel(n):
    result = np.zeros(n)
    for i in numba.prange(n):
        result[i] = i * i
    return result

print('开始并行测试...')
result = test_parallel(100000000)
print(f'测试完成，结果长度: {len(result)}')
print('如果CPU使用率飙升到多个核心，说明并行成功！')
"

echo "[步骤4] 启动校准任务..."
# 5. 杀掉旧的screen会话
screen -S calibration -X quit 2>/dev/null || true

# 6. 启动新的screen会话
screen -dmS calibration bash -c "
    export PYTHONIOENCODING=utf-8
    export NUMBA_NUM_THREADS=1
    export NUMBA_THREADING_LAYER=omp
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    
    cd ~/Simulation_project_v3
    python3 TESTS/test_calibration.py 2>&1 | tee OUTPUT/calibration/calibration_run.log
    
    echo '任务完成时间:' \$(date) >> OUTPUT/calibration/calibration_run.log
    
    # 任务完成后发送通知（如果配置了）
    if [ -f ~/Simulation_project_v3/AutoDL_Deploy/email_notify.sh ]; then
        bash ~/Simulation_project_v3/AutoDL_Deploy/email_notify.sh
    fi
"

echo "=========================================="
echo "✓ 校准任务已启动"
echo "=========================================="
echo ""
echo "监控命令："
echo "  screen -r calibration     # 查看运行界面"
echo "  htop                      # 查看CPU使用率（应该看到多个核心都在100%）"
echo "  tail -f OUTPUT/calibration/calibration_run.log  # 查看日志"
echo ""
echo "退出screen: Ctrl+A 然后按 D"
echo "=========================================="

