"""
主窗口

EconLab应用的主窗口，管理所有页面和全局功能
"""

from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QStatusBar,
                             QMenuBar, QToolBar, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon

from .pages import ConfigPage, SimulationPage, ResultsPage
from .utils import ConfigManager


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self, use_demo_mode=True):
        """
        初始化主窗口
        
        参数:
            use_demo_mode: 是否使用演示模式
        """
        super().__init__()
        
        self.use_demo_mode = use_demo_mode
        
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        
        # 初始化界面
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("EconLab - 农村女性就业市场仿真平台")
        self.setGeometry(100, 100, 1280, 800)
        self.setMinimumSize(1024, 720)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_tool_bar()
        
        # 创建标签页
        self.create_tab_widget()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 应用样式
        self.apply_style()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        new_action = QAction("新建配置", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开配置", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_config)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存配置", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 运行菜单
        run_menu = menubar.addMenu("运行(&R)")
        
        start_action = QAction("开始仿真", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self.start_simulation)
        run_menu.addAction(start_action)
        
        stop_action = QAction("停止仿真", self)
        stop_action.setShortcut("Shift+F5")
        stop_action.triggered.connect(self.stop_simulation)
        run_menu.addAction(stop_action)
        
        run_menu.addSeparator()
        
        # 模式切换
        self.demo_mode_action = QAction("演示模式（当前）", self)
        self.demo_mode_action.setCheckable(True)
        self.demo_mode_action.setChecked(True)
        self.demo_mode_action.triggered.connect(self.toggle_demo_mode)
        run_menu.addAction(self.demo_mode_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        doc_action = QAction("使用文档", self)
        doc_action.setShortcut("F1")
        doc_action.triggered.connect(self.show_documentation)
        help_menu.addAction(doc_action)
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 新建按钮
        new_action = QAction("📁 新建", self)
        new_action.triggered.connect(self.new_config)
        toolbar.addAction(new_action)
        
        # 保存按钮
        save_action = QAction("💾 保存", self)
        save_action.triggered.connect(self.save_config)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 运行按钮
        run_action = QAction("▶ 运行", self)
        run_action.triggered.connect(self.start_simulation)
        toolbar.addAction(run_action)
        
        # 停止按钮
        stop_action = QAction("⏹ 停止", self)
        stop_action.triggered.connect(self.stop_simulation)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # 帮助按钮
        help_action = QAction("❓ 帮助", self)
        help_action.triggered.connect(self.show_documentation)
        toolbar.addAction(help_action)
    
    def create_tab_widget(self):
        """创建标签页组件"""
        self.tab_widget = QTabWidget()
        
        # 创建三个页面
        self.config_page = ConfigPage(self.config_manager)
        self.simulation_page = SimulationPage(use_demo_mode=self.use_demo_mode)
        self.results_page = ResultsPage()
        
        # 传递config_manager到simulation_page
        self.simulation_page.set_config_manager(self.config_manager)
        
        # 连接仿真完成信号到结果更新和页面切换
        self.simulation_page.simulation_finished_signal = (
            self._on_simulation_completed
        )
        
        # 添加到标签页
        self.tab_widget.addTab(self.config_page, "参数配置")
        self.tab_widget.addTab(self.simulation_page, "仿真运行")
        self.tab_widget.addTab(self.results_page, "结果分析")
        
        # 设置为中心组件
        self.setCentralWidget(self.tab_widget)
    
    def _on_simulation_completed(self, individuals, eq_info):
        """
        仿真完成后的处理
        
        参数:
            individuals: 均衡个体DataFrame
            eq_info: 均衡信息字典
        """
        # 更新结果页数据
        self.results_page.update_results(individuals, eq_info)
        
        # 自动切换到结果分析页
        self.tab_widget.setCurrentIndex(2)
        
        # 更新状态栏
        self.statusBar.showMessage("仿真完成，结果已更新")
    
    def create_status_bar(self):
        """创建状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
    
    def apply_style(self):
        """应用全局样式"""
        from GUI.utils import get_resource_path
        from pathlib import Path
        
        # 尝试加载外部QSS文件
        qss_path = Path(get_resource_path("GUI/resources/styles/main.qss"))
        
        if qss_path.exists():
            with open(qss_path, 'r', encoding='utf-8') as f:
                style = f.read()
            self.setStyleSheet(style)
        else:
            # 备用内置样式
            style = """
            QMainWindow {
                background-color: #F5F6FA;
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            """
            self.setStyleSheet(style)
    
    # 菜单栏和工具栏的槽函数
    
    def new_config(self):
        """新建配置"""
        QMessageBox.information(self, "提示", "新建配置功能正在开发中...")
    
    def open_config(self):
        """打开配置"""
        QMessageBox.information(self, "提示", "打开配置功能正在开发中...")
    
    def save_config(self):
        """保存配置"""
        QMessageBox.information(self, "提示", "保存配置功能正在开发中...")
    
    def start_simulation(self):
        """开始仿真"""
        # 切换到仿真运行页
        self.tab_widget.setCurrentIndex(1)
        # 触发仿真页的开始按钮
        self.simulation_page.start_simulation()
    
    def stop_simulation(self):
        """停止仿真"""
        self.simulation_page.stop_simulation()
    
    def toggle_demo_mode(self):
        """切换演示模式"""
        is_demo = self.demo_mode_action.isChecked()
        self.simulation_page.use_demo_mode = is_demo
        
        if is_demo:
            self.demo_mode_action.setText("演示模式（当前）")
            self.statusBar.showMessage("已切换到演示模式（使用模拟数据）")
        else:
            self.demo_mode_action.setText("真实模式（当前）")
            self.statusBar.showMessage("已切换到真实模式（调用MFG求解器）")
    
    def show_documentation(self):
        """显示使用文档"""
        doc_text = """
        EconLab - 农村女性就业市场仿真平台
        
        使用步骤：
        1. 在"参数配置"页面调整模型参数
        2. 点击"开始仿真"运行MFG均衡求解
        3. 在"结果分析"页面查看可视化结果
        
        快捷键：
        - Ctrl+N: 新建配置
        - Ctrl+O: 打开配置
        - Ctrl+S: 保存配置
        - F5: 开始仿真
        - Shift+F5: 停止仿真
        - F1: 显示帮助
        
        详细文档请参考项目目录中的 GUI开发文档.md
        """
        QMessageBox.information(self, "使用文档", doc_text)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>EconLab v1.0</h2>
        <p>农村女性就业市场仿真平台</p>
        <p>基于平均场博弈(MFG)与主体建模(ABM)理论</p>
        <br>
        <p><b>开发团队：</b>李心泠团队</p>
        <p><b>指导教师：</b>李三希、林琳</p>
        <p><b>技术栈：</b>Python 3.11 + PyQt6 + Numba</p>
        <br>
        <p>© 2025 2025大创项目</p>
        """
        QMessageBox.about(self, "关于EconLab", about_text)
    
    def closeEvent(self, event):
        """
        窗口关闭事件
        
        参数:
            event: 关闭事件
        """
        reply = QMessageBox.question(
            self, 
            "确认退出",
            "确定要退出EconLab吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

