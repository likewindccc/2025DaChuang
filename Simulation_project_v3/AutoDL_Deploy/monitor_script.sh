#!/bin/bash

# AutoDL任务监控脚本
# 用途：生成实时监控报告，方便随时查看进度

REPORT_FILE="OUTPUT/calibration/progress_report.txt"
LOG_FILE="OUTPUT/calibration/calibration_run.log"

echo "=========================================="
echo "任务监控报告"
echo "=========================================="
echo ""
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 检查进程是否运行
echo "【运行状态】"
if pgrep -f "test_calibration.py" > /dev/null; then
    echo "✓ 校准任务正在运行中"
    RUNNING=true
else
    echo "✗ 校准任务未运行（可能已完成或出错）"
    RUNNING=false
fi
echo ""

# 2. 显示日志最后20行
echo "【最新日志】（最后20行）"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "日志文件不存在"
fi
echo ""

# 3. 检查输出文件
echo "【输出文件检查】"
if [ -f "OUTPUT/calibration/calibrated_parameters.yaml" ]; then
    echo "✓ 已生成 calibrated_parameters.yaml（任务可能完成）"
    echo "  文件大小: $(du -h OUTPUT/calibration/calibrated_parameters.yaml | cut -f1)"
    echo "  修改时间: $(stat -c '%y' OUTPUT/calibration/calibrated_parameters.yaml 2>/dev/null || stat -f '%Sm' OUTPUT/calibration/calibrated_parameters.yaml)"
else
    echo "✗ 尚未生成 calibrated_parameters.yaml"
fi

if [ -f "OUTPUT/calibration/calibration_history.csv" ]; then
    LINE_COUNT=$(wc -l < OUTPUT/calibration/calibration_history.csv)
    echo "✓ 历史记录: $LINE_COUNT 次评估"
else
    echo "✗ 尚未生成历史记录"
fi
echo ""

# 4. CPU和内存使用
echo "【系统资源】"
echo "CPU使用率:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  使用: " 100 - $1 "%"}'
echo "内存使用:"
free -h | grep Mem | awk '{print "  已用: " $3 " / 总计: " $2 " (" int($3/$2*100) "%)"}'
echo ""

# 5. 预估完成时间（基于历史数据）
echo "【进度估计】"
if [ -f "OUTPUT/calibration/calibration_history.csv" ] && [ "$RUNNING" = true ]; then
    CURRENT_EVALS=$(wc -l < OUTPUT/calibration/calibration_history.csv)
    CURRENT_EVALS=$((CURRENT_EVALS - 1))  # 减去表头
    
    if [ $CURRENT_EVALS -gt 0 ]; then
        echo "  当前评估次数: $CURRENT_EVALS"
        echo "  目标评估次数: ~1000（最大）"
        
        PERCENT=$((CURRENT_EVALS * 100 / 1000))
        echo "  进度: $PERCENT%"
        
        # 简单进度条
        FILLED=$((PERCENT / 2))
        BAR=$(printf '%*s' "$FILLED" | tr ' ' '█')
        EMPTY=$(printf '%*s' $((50 - FILLED)) | tr ' ' '░')
        echo "  [$BAR$EMPTY] $PERCENT%"
    else
        echo "  刚开始运行..."
    fi
else
    echo "  任务未运行或无历史数据"
fi
echo ""

# 6. 磁盘空间
echo "【磁盘空间】"
df -h . | tail -1 | awk '{print "  已用: " $3 " / 总计: " $2 " (" $5 " 使用率)"}'
echo ""

echo "=========================================="
echo "提示："
echo "  - 重新连接: screen -r calibration"
echo "  - 实时日志: tail -f $LOG_FILE"
echo "  - 刷新报告: ./monitor_script.sh"
echo "=========================================="

# 保存到文件
{
    echo "最后更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    if [ "$RUNNING" = true ]; then
        echo "状态: ✓ 运行中"
    else
        echo "状态: ✗ 未运行"
    fi
    
    if [ -f "OUTPUT/calibration/calibration_history.csv" ]; then
        EVALS=$(wc -l < OUTPUT/calibration/calibration_history.csv)
        EVALS=$((EVALS - 1))
        echo "评估次数: $EVALS"
    fi
    
    if [ -f "OUTPUT/calibration/calibrated_parameters.yaml" ]; then
        echo "结果: ✓ 已完成"
    else
        echo "结果: 进行中..."
    fi
} > "$REPORT_FILE"

