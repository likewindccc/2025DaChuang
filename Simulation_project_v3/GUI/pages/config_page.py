"""
参数配置页

允许用户配置MFG模型的各项参数
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QLabel, QPushButton, 
                             QMessageBox, QSpinBox, QDoubleSpinBox,
                             QFileDialog)
from PyQt6.QtCore import Qt
from GUI.widgets import ParameterWidget


class ConfigPage(QWidget):
    """参数配置页面类"""
    
    def __init__(self, config_manager):
        """
        初始化参数配置页
        
        参数:
            config_manager: ConfigManager实例
        """
        super().__init__()
        self.config_manager = config_manager
        self.parameter_widgets = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 创建经济参数组
        economics_group = self.create_economics_group()
        layout.addWidget(economics_group)
        
        # 创建市场参数组
        market_group = self.create_market_group()
        layout.addWidget(market_group)
        
        # 创建按钮组
        button_layout = self.create_button_layout()
        layout.addLayout(button_layout)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 状态标签
        self.status_label = QLabel("当前配置: CONFIG/mfg_config.yaml")
        self.status_label.setStyleSheet("color: #7F8C8D; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # 加载默认配置
        self.load_default_config()
    
    def create_economics_group(self):
        """创建经济参数组"""
        group = QGroupBox("经济参数")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 贴现因子 ρ
        rho_widget = ParameterWidget(
            name="贴现因子 ρ",
            min_val=0.30,
            max_val=0.60,
            default_val=0.40,
            tooltip="个体对未来的重视程度，越大越重视未来",
            step=0.01,
            decimals=2
        )
        layout.addWidget(rho_widget)
        self.parameter_widgets['economics.rho'] = rho_widget
        
        # 努力成本系数 κ
        kappa_widget = ParameterWidget(
            name="努力成本系数 κ",
            min_val=1000.0,
            max_val=4000.0,
            default_val=2000.0,
            tooltip="努力的边际成本，越大个体越不愿意努力",
            step=100.0,
            decimals=0
        )
        layout.addWidget(kappa_widget)
        self.parameter_widgets['economics.kappa'] = kappa_widget
        
        # T负效用系数 α
        alpha_widget = ParameterWidget(
            name="T负效用系数 α",
            min_val=0.10,
            max_val=0.60,
            default_val=0.30,
            tooltip="工作时间偏离理想值的负效用系数",
            step=0.01,
            decimals=2
        )
        layout.addWidget(alpha_widget)
        self.parameter_widgets['economics.disutility_T.alpha'] = alpha_widget
        
        group.setLayout(layout)
        return group
    
    def create_market_group(self):
        """创建市场参数组"""
        group = QGroupBox("市场参数")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 个体数量
        n_individuals_layout = QHBoxLayout()
        n_individuals_label = QLabel("个体数量:")
        n_individuals_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #2C3E50;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        n_individuals_layout.addWidget(n_individuals_label)
        n_individuals_layout.addStretch()
        
        self.n_individuals_spinbox = QSpinBox()
        self.n_individuals_spinbox.setMinimum(100)
        self.n_individuals_spinbox.setMaximum(50000)
        self.n_individuals_spinbox.setSingleStep(1000)
        self.n_individuals_spinbox.setValue(10000)
        self.n_individuals_spinbox.setFixedWidth(140)
        self.n_individuals_spinbox.setFixedHeight(36)
        self.n_individuals_spinbox.setToolTip("模拟的个体数量")
        self.n_individuals_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 8px 12px;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                background-color: #F8F9FA;
            }
            QSpinBox:focus {
                border: 2px solid #1ABC9C;
                background-color: white;
            }
        """)
        n_individuals_layout.addWidget(self.n_individuals_spinbox)
        
        layout.addLayout(n_individuals_layout)
        
        # 目标市场紧张度 θ
        theta_layout = QHBoxLayout()
        theta_label = QLabel("目标市场紧张度 θ:")
        theta_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #2C3E50;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        theta_layout.addWidget(theta_label)
        theta_layout.addStretch()
        
        self.theta_spinbox = QDoubleSpinBox()
        self.theta_spinbox.setMinimum(0.5)
        self.theta_spinbox.setMaximum(3.0)
        self.theta_spinbox.setSingleStep(0.1)
        self.theta_spinbox.setValue(1.5)
        self.theta_spinbox.setDecimals(2)
        self.theta_spinbox.setFixedWidth(140)
        self.theta_spinbox.setFixedHeight(36)
        self.theta_spinbox.setToolTip("岗位空缺数与失业者数量的比值")
        self.theta_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                padding: 8px 12px;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                background-color: #F8F9FA;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #1ABC9C;
                background-color: white;
            }
        """)
        theta_layout.addWidget(self.theta_spinbox)
        
        layout.addLayout(theta_layout)
        
        # 最大迭代轮数
        max_iter_layout = QHBoxLayout()
        max_iter_label = QLabel("最大迭代轮数:")
        max_iter_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #2C3E50;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        max_iter_layout.addWidget(max_iter_label)
        max_iter_layout.addStretch()
        
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setMinimum(10)
        self.max_iter_spinbox.setMaximum(500)
        self.max_iter_spinbox.setSingleStep(10)
        self.max_iter_spinbox.setValue(100)
        self.max_iter_spinbox.setFixedWidth(140)
        self.max_iter_spinbox.setFixedHeight(36)
        self.max_iter_spinbox.setToolTip("MFG均衡求解的最大迭代次数")
        self.max_iter_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 8px 12px;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
                background-color: #F8F9FA;
            }
            QSpinBox:focus {
                border: 2px solid #1ABC9C;
                background-color: white;
            }
        """)
        max_iter_layout.addWidget(self.max_iter_spinbox)
        
        layout.addLayout(max_iter_layout)
        
        group.setLayout(layout)
        return group
    
    def create_button_layout(self):
        """创建按钮布局"""
        layout = QHBoxLayout()
        
        # 加载配置按钮
        load_btn = QPushButton("加载配置文件")
        load_btn.clicked.connect(self.load_config)
        load_btn.setMinimumHeight(35)
        layout.addWidget(load_btn)
        
        # 保存配置按钮
        save_btn = QPushButton("保存当前配置")
        save_btn.clicked.connect(self.save_config)
        save_btn.setMinimumHeight(35)
        layout.addWidget(save_btn)
        
        # 重置按钮
        reset_btn = QPushButton("重置为默认值")
        reset_btn.clicked.connect(self.reset_config)
        reset_btn.setMinimumHeight(35)
        layout.addWidget(reset_btn)
        
        return layout
    
    def load_default_config(self):
        """加载默认配置"""
        config = self.config_manager.load_config("mfg_config.yaml")
        self.update_ui_from_config(config)
        self.status_label.setText("当前配置: CONFIG/mfg_config.yaml")
    
    def update_ui_from_config(self, config):
        """
        从配置字典更新界面
        
        参数:
            config: 配置字典
        """
        # 更新经济参数
        rho = config.get('economics', {}).get('rho', 0.40)
        self.parameter_widgets['economics.rho'].set_value(rho)
        
        kappa = config.get('economics', {}).get('kappa', 2000.0)
        self.parameter_widgets['economics.kappa'].set_value(kappa)
        
        alpha = config.get('economics', {}).get('disutility_T', {}).get('alpha', 0.30)
        self.parameter_widgets['economics.disutility_T.alpha'].set_value(alpha)
        
        # 更新市场参数
        n_individuals = config.get('population', {}).get('n_individuals', 10000)
        self.n_individuals_spinbox.setValue(n_individuals)
        
        target_theta = config.get('market', {}).get('target_theta', 1.5)
        self.theta_spinbox.setValue(target_theta)
        
        max_iter = config.get('equilibrium', {}).get('max_outer_iter', 100)
        self.max_iter_spinbox.setValue(max_iter)
    
    def get_config_from_ui(self):
        """
        从界面获取配置
        
        返回:
            配置字典
        """
        config = self.config_manager.current_config.copy()
        
        # 更新经济参数
        self.config_manager.set_parameter_value(
            'economics.rho',
            self.parameter_widgets['economics.rho'].get_value()
        )
        self.config_manager.set_parameter_value(
            'economics.kappa',
            self.parameter_widgets['economics.kappa'].get_value()
        )
        self.config_manager.set_parameter_value(
            'economics.disutility_T.alpha',
            self.parameter_widgets['economics.disutility_T.alpha'].get_value()
        )
        
        # 更新市场参数
        self.config_manager.set_parameter_value(
            'population.n_individuals',
            self.n_individuals_spinbox.value()
        )
        self.config_manager.set_parameter_value(
            'market.target_theta',
            self.theta_spinbox.value()
        )
        self.config_manager.set_parameter_value(
            'equilibrium.max_outer_iter',
            self.max_iter_spinbox.value()
        )
        
        return self.config_manager.current_config
    
    def load_config(self):
        """加载配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载配置文件",
            "CONFIG",
            "YAML文件 (*.yaml *.yml)"
        )
        
        if file_path:
            config = self.config_manager.load_config(file_path)
            self.update_ui_from_config(config)
            self.status_label.setText(f"当前配置: {file_path}")
            QMessageBox.information(self, "成功", "配置文件加载成功")
    
    def save_config(self):
        """保存配置"""
        # 获取界面配置
        config = self.get_config_from_ui()
        
        # 验证配置
        is_valid, error_msg = self.config_manager.validate_config(config)
        if not is_valid:
            QMessageBox.warning(self, "验证失败", f"配置验证失败:\n{error_msg}")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存配置文件",
            "CONFIG/mfg_config_custom.yaml",
            "YAML文件 (*.yaml *.yml)"
        )
        
        if file_path:
            self.config_manager.save_config(config, file_path)
            self.status_label.setText(f"当前配置: {file_path}")
            QMessageBox.information(self, "成功", "配置文件保存成功")
    
    def reset_config(self):
        """重置配置为默认值"""
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要重置为默认配置吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.load_default_config()
            QMessageBox.information(self, "成功", "已重置为默认配置")

