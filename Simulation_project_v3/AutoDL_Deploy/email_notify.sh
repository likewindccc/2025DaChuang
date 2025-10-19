#!/bin/bash

# 邮件通知脚本（可选）
# 用途：任务完成后发送邮件通知

# ==================================================
# 配置区域（需要配置邮件信息）
# ==================================================

# 注意：AutoDL服务器可能无法直接发送邮件
# 建议使用webhook或企业微信、钉钉等通知方式

# 这个脚本提供基本框架，实际使用需要配置第三方通知服务

WEBHOOK_URL="YOUR_WEBHOOK_URL"  # 企业微信/钉钉webhook

# ==================================================
# 检查任务是否完成
# ==================================================

check_completion() {
    if [ -f "OUTPUT/calibration/calibrated_parameters.yaml" ]; then
        return 0  # 完成
    else
        return 1  # 未完成
    fi
}

# ==================================================
# 发送通知（使用webhook）
# ==================================================

send_notification() {
    local message="$1"
    
    # 示例：使用curl发送webhook通知
    # 企业微信webhook格式
    curl -X POST "$WEBHOOK_URL" \
        -H 'Content-Type: application/json' \
        -d "{
            \"msgtype\": \"text\",
            \"text\": {
                \"content\": \"$message\"
            }
        }"
}

# ==================================================
# 持续监控
# ==================================================

echo "开始监控任务..."
echo "按 Ctrl+C 停止监控"

CHECK_INTERVAL=300  # 每5分钟检查一次

while true; do
    if check_completion; then
        MESSAGE="🎉 校准任务已完成！
        
时间: $(date '+%Y-%m-%d %H:%M:%S')
服务器: AutoDL实例
项目: Simulation_project_v3

请及时：
1. 下载结果文件
2. 关闭AutoDL实例"
        
        send_notification "$MESSAGE"
        echo "✓ 任务完成，通知已发送"
        break
    else
        echo "$(date '+%H:%M:%S') - 任务进行中..."
        sleep $CHECK_INTERVAL
    fi
done

