"""
参数输入组件

组合标签、滑块和输入框的自定义参数输入组件
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                             QSlider, QLineEdit, QDoubleSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator


class ParameterWidget(QWidget):
    """参数输入组件类"""
    
    value_changed = pyqtSignal(float)
    
    def __init__(self, name, min_val, max_val, default_val, 
                 tooltip="", step=0.01, decimals=2):
        """
        初始化参数组件
        
        参数:
            name: 参数名称
            min_val: 最小值
            max_val: 最大值
            default_val: 默认值
            tooltip: 提示信息
            step: 步长
            decimals: 小数位数
        """
        super().__init__()
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals
        self.current_value = default_val
        
        self.init_ui()
        self.set_value(default_val)
        
        if tooltip:
            self.setToolTip(tooltip)
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 第一行：标签和数值输入框
        top_layout = QHBoxLayout()
        
        # 参数名称标签
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 600;
                color: #2C3E50;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
            }
        """)
        top_layout.addWidget(self.name_label)
        
        top_layout.addStretch()
        
        # 数值输入框
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(self.min_val)
        self.spinbox.setMaximum(self.max_val)
        self.spinbox.setSingleStep(self.step)
        self.spinbox.setDecimals(self.decimals)
        self.spinbox.setFixedWidth(120)
        self.spinbox.setFixedHeight(36)
        self.spinbox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 8px 12px;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                background-color: #F8F9FA;
                color: #2C3E50;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #1ABC9C;
                background-color: white;
            }
        """)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        top_layout.addWidget(self.spinbox)
        
        main_layout.addLayout(top_layout)
        
        # 第二行：滑块和范围标签
        slider_layout = QHBoxLayout()
        
        # 最小值标签
        self.min_label = QLabel(str(self.min_val))
        self.min_label.setStyleSheet("""
            font-size: 11px;
            color: #6C757D;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        slider_layout.addWidget(self.min_label)
        
        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        slider_steps = int((self.max_val - self.min_val) / self.step)
        self.slider.setMinimum(0)
        self.slider.setMaximum(slider_steps)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E5E8E8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1ABC9C;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #16A085;
            }
        """)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider, stretch=1)
        
        # 最大值标签
        self.max_label = QLabel(str(self.max_val))
        self.max_label.setStyleSheet("""
            font-size: 11px;
            color: #6C757D;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        slider_layout.addWidget(self.max_label)
        
        main_layout.addLayout(slider_layout)
        
        self.setLayout(main_layout)
    
    def _on_slider_changed(self, slider_value):
        """滑块值改变时的回调"""
        value = self.min_val + (slider_value * self.step)
        value = round(value, self.decimals)
        
        # 阻止信号循环
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        
        if value != self.current_value:
            self.current_value = value
            self.value_changed.emit(value)
    
    def _on_spinbox_changed(self, value):
        """输入框值改变时的回调"""
        value = round(value, self.decimals)
        
        # 更新滑块位置
        slider_value = int((value - self.min_val) / self.step)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        
        if value != self.current_value:
            self.current_value = value
            self.value_changed.emit(value)
    
    def get_value(self):
        """
        获取当前值
        
        返回:
            当前参数值
        """
        return self.current_value
    
    def set_value(self, value):
        """
        设置参数值
        
        参数:
            value: 要设置的值
        """
        value = max(self.min_val, min(self.max_val, value))
        value = round(value, self.decimals)
        
        self.current_value = value
        
        # 更新输入框
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        
        # 更新滑块
        slider_value = int((value - self.min_val) / self.step)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)

