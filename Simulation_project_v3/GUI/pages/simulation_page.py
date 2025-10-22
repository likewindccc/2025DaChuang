"""
仿真运行页

显示仿真运行进度和实时监控数据
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QLabel, QPushButton, 
                             QProgressBar, QTextEdit, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class SimulationPage(QWidget):
    """仿真运行页面类"""
    
    def __init__(self, use_demo_mode=True):
        """
        初始化仿真运行页
        
        参数:
            use_demo_mode: 是否使用演示模式（默认True，绕过Numba问题）
        """
        super().__init__()
        self.is_running = False
        self.simulation_finished_signal = None
        self.use_demo_mode = use_demo_mode
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 创建运行控制组
        control_group = self.create_control_group()
        layout.addWidget(control_group)
        
        # 创建实时监控组
        monitor_group = self.create_monitor_group()
        layout.addWidget(monitor_group)
        
        # 创建运行日志组
        log_group = self.create_log_group()
        layout.addWidget(log_group)
        
        self.setLayout(layout)
    
    def create_control_group(self):
        """创建运行控制组"""
        group = QGroupBox("运行控制")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 按钮行
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始仿真")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)
        self.start_btn.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_simulation)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("运行状态: 就绪 ⚪")
        self.status_label.setStyleSheet("font-size: 12px; margin-top: 5px;")
        layout.addWidget(self.status_label)
        
        group.setLayout(layout)
        return group
    
    def create_monitor_group(self):
        """创建实时监控组"""
        group = QGroupBox("实时监控")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 进度信息
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("当前迭代:"))
        self.iteration_label = QLabel("0 / 100")
        self.iteration_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.iteration_label)
        progress_layout.addStretch()
        layout.addLayout(progress_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #1ABC9C;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 关键指标 - 卡片式设计
        indicators_layout = QHBoxLayout()
        indicators_layout.setSpacing(15)
        
        # 失业率卡片
        unemployment_card = self.create_metric_card(
            "失业率", "--", "#E74C3C"
        )
        self.unemployment_label = unemployment_card.findChild(QLabel, "value")
        indicators_layout.addWidget(unemployment_card)
        
        # 收敛指标卡片
        convergence_card = self.create_metric_card(
            "收敛指标", "--", "#F39C12"
        )
        self.convergence_label = convergence_card.findChild(QLabel, "value")
        indicators_layout.addWidget(convergence_card)
        
        # 平均工资卡片
        wage_card = self.create_metric_card(
            "平均工资", "--", "#3498DB"
        )
        self.wage_label = wage_card.findChild(QLabel, "value")
        indicators_layout.addWidget(wage_card)
        
        layout.addLayout(indicators_layout)
        
        group.setLayout(layout)
        return group
    
    def create_metric_card(self, title, initial_value, color):
        """
        创建指标卡片
        
        参数:
            title: 卡片标题
            initial_value: 初始值
            color: 值的颜色
        
        返回:
            QFrame卡片组件
        """
        card = QFrame()
        card.setObjectName("metricCard")
        card.setStyleSheet(f"""
            QFrame#metricCard {{
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E9ECEF;
                padding: 20px;
            }}
            QFrame#metricCard:hover {{
                border: 1px solid #1ABC9C;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 13px;
            color: #6C757D;
            font-weight: 500;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 值
        value_label = QLabel(initial_value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {color};
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)
        
        card.setLayout(layout)
        return card
    
    def create_log_group(self):
        """创建运行日志组"""
        group = QGroupBox("运行日志")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_text)
        
        group.setLayout(layout)
        return group
    
    def set_config_manager(self, config_manager):
        """
        设置配置管理器（在主窗口初始化后调用）
        
        参数:
            config_manager: ConfigManager实例
        """
        self.config_manager = config_manager
    
    def start_simulation(self):
        """开始仿真"""
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # 获取配置
        config_dict = self.config_manager.current_config
        
        # 根据模式选择Worker
        if self.use_demo_mode:
            self.status_label.setText("运行状态: 演示模式运行中 🟢")
            self.append_log("INFO", "启动演示模式（使用模拟数据）...")
            from GUI.workers.demo_worker import DemoWorker
            self.worker = DemoWorker(config_dict)
        else:
            self.status_label.setText("运行状态: 运行中 🟢")
            self.append_log("INFO", "启动真实MFG求解...")
            from GUI.workers import SimulationWorker
            self.worker = SimulationWorker(config_dict)
        
        # 连接信号
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.log_message.connect(self.append_log)
        
        # 获取最大迭代次数
        max_iter = config_dict.get('equilibrium', {}).get('max_outer_iter', 100)
        self.progress_bar.setMaximum(max_iter)
        self.iteration_label.setText(f"0 / {max_iter}")
        
        # 启动线程
        self.worker.start()
    
    def pause_simulation(self):
        """暂停仿真"""
        self.is_running = False
        self.worker.pause()
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ 继续")
        self.pause_btn.setEnabled(False)
        self.status_label.setText("运行状态: 已暂停 🟡")
        self.append_log("WARN", "仿真已暂停")
    
    def stop_simulation(self):
        """停止仿真"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ 开始仿真")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("运行状态: 已停止 🔴")
        self.append_log("ERROR", "仿真已停止")
        self.reset_display()
    
    def on_progress_updated(self, iteration, stats):
        """
        处理进度更新
        
        参数:
            iteration: 当前迭代轮数
            stats: 统计数据字典
        """
        max_iter = self.progress_bar.maximum()
        
        # 更新进度条
        self.progress_bar.setValue(iteration)
        self.iteration_label.setText(f"{iteration} / {max_iter}")
        
        # 更新关键指标
        unemployment_rate = stats.get('unemployment_rate', 0)
        self.unemployment_label.setText(f"{unemployment_rate*100:.2f}%")
        
        diff_V = stats.get('diff_V', 0)
        self.convergence_label.setText(f"|ΔV|={diff_V:.4f}")
        
        mean_wage = stats.get('mean_wage', 0)
        self.wage_label.setText(f"{mean_wage:.0f}元")
    
    def on_simulation_finished(self, individuals, eq_info):
        """
        仿真完成处理
        
        参数:
            individuals: 均衡个体DataFrame
            eq_info: 均衡信息字典
        """
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ 开始仿真")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("运行状态: 完成 ✅")
        
        # 显示完成消息
        converged = eq_info.get('converged', False)
        if converged:
            self.append_log("INFO", "均衡求解收敛！")
        else:
            self.append_log("WARN", "达到最大迭代次数，未完全收敛")
        
        # 保存结果
        self.last_individuals = individuals
        self.last_eq_info = eq_info
        
        # 通知主窗口更新结果页
        if self.simulation_finished_signal is not None:
            self.simulation_finished_signal(individuals, eq_info)
    
    def on_error(self, error_msg):
        """
        错误处理
        
        参数:
            error_msg: 错误消息
        """
        self.append_log("ERROR", f"发生错误: {error_msg}")
        
        # 显示错误对话框
        QMessageBox.critical(
            self,
            "运行错误",
            f"仿真运行时发生错误:\n\n{error_msg}\n\n"
            "这可能是Numba/llvmlite的DLL加载问题。\n"
            "建议解决方法：\n"
            "1. 重启应用重试\n"
            "2. 重新安装Numba: pip install numba==0.59.0\n"
            "3. 安装Visual C++ Redistributable"
        )
        
        self.stop_simulation()
    
    def append_log(self, level, message):
        """
        添加日志
        
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
        
        self.log_text.append(log_html)
    
    def reset_display(self):
        """重置显示"""
        self.progress_bar.setValue(0)
        self.iteration_label.setText("0 / 100")
        self.unemployment_label.setText("--")
        self.convergence_label.setText("--")
        self.wage_label.setText("--")

