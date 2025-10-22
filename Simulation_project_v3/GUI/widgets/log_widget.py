"""
日志显示组件

彩色日志显示组件
"""

from PyQt6.QtWidgets import QTextEdit


class LogWidget(QTextEdit):
    """日志显示组件类"""
    
    def __init__(self):
        """初始化日志组件"""
        super().__init__()
        self.setReadOnly(True)
        
        # 设置样式
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
    
    def append_log(self, level, message):
        """
        添加一条日志
        
        参数:
            level: 日志级别 (INFO/WARN/ERROR)
            message: 日志消息
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            'INFO': '#1ABC9C',
            'WARN': '#F39C12',
            'ERROR': '#E74C3C'
        }
        color = color_map.get(level, '#ECF0F1')
        
        log_html = f'<span style="color: #7F8C8D;">[{timestamp}]</span> '
        log_html += f'<span style="color: {color}; font-weight: bold;">{level}</span> '
        log_html += f'<span>{message}</span>'
        
        self.append(log_html)
    
    def clear_log(self):
        """清空日志"""
        self.clear()

