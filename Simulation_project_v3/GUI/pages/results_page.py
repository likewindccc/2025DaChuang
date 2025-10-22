"""
结果分析页

显示仿真结果的可视化图表和关键指标
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QLabel, QPushButton, 
                             QTabWidget, QMessageBox, QFileDialog, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from GUI.widgets import ChartWidget
import pandas as pd
from datetime import datetime


class ResultsPage(QWidget):
    """结果分析页面类"""
    
    def __init__(self):
        """初始化结果分析页"""
        super().__init__()
        self.individuals = None
        self.eq_info = None
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 创建图表区域
        chart_group = self.create_chart_group()
        layout.addWidget(chart_group, stretch=3)
        
        # 创建指标汇总组
        summary_group = self.create_summary_group()
        layout.addWidget(summary_group, stretch=1)
        
        # 创建导出按钮组
        export_layout = self.create_export_layout()
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
    
    def create_chart_group(self):
        """创建图表显示组"""
        group = QGroupBox("结果可视化")
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
        
        # 创建图表切换标签页
        self.chart_tabs = QTabWidget()
        
        # 创建4个图表组件
        self.unemployment_chart = ChartWidget()
        self.distribution_chart = ChartWidget()
        self.convergence_chart = ChartWidget()
        self.theta_chart = ChartWidget()
        
        self.chart_tabs.addTab(self.unemployment_chart, "失业率趋势")
        self.chart_tabs.addTab(self.distribution_chart, "状态分布")
        self.chart_tabs.addTab(self.convergence_chart, "收敛过程")
        self.chart_tabs.addTab(self.theta_chart, "市场紧张度")
        
        # 当切换标签时更新图表
        self.chart_tabs.currentChanged.connect(self.on_chart_tab_changed)
        
        layout.addWidget(self.chart_tabs)
        
        group.setLayout(layout)
        return group
    
    def create_summary_group(self):
        """创建关键指标汇总组"""
        group = QGroupBox("关键指标汇总")
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
        
        # 第一行指标
        row1 = QHBoxLayout()
        self.unemployment_card = self.create_indicator_card("最终失业率", "--", "%")
        self.iterations_card = self.create_indicator_card("收敛轮数", "--", "轮")
        self.wage_card = self.create_indicator_card("平均工资", "--", "元")
        row1.addWidget(self.unemployment_card)
        row1.addWidget(self.iterations_card)
        row1.addWidget(self.wage_card)
        layout.addLayout(row1)
        
        # 第二行指标
        row2 = QHBoxLayout()
        self.theta_card = self.create_indicator_card("市场紧张度", "--", "")
        self.T_card = self.create_indicator_card("平均T值", "--", "小时")
        self.S_card = self.create_indicator_card("平均S值", "--", "")
        row2.addWidget(self.theta_card)
        row2.addWidget(self.T_card)
        row2.addWidget(self.S_card)
        layout.addLayout(row2)
        
        group.setLayout(layout)
        return group
    
    def create_indicator_card(self, label_text, value_text, unit_text):
        """
        创建指标卡片
        
        参数:
            label_text: 指标名称
            value_text: 指标值
            unit_text: 单位
        
        返回:
            QFrame
        """
        card = QFrame()
        card.setObjectName("resultCard")
        card.setStyleSheet("""
            QFrame#resultCard {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E9ECEF;
                padding: 18px;
            }
            QFrame#resultCard:hover {
                border: 1px solid #1ABC9C;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # 指标名称
        label = QLabel(label_text)
        label.setStyleSheet("""
            font-size: 14px;
            color: #6C757D;
            font-weight: 500;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        # 指标值
        value = QLabel(f"{value_text} {unit_text}")
        value.setStyleSheet("""
            font-size: 26px;
            font-weight: bold;
            color: #2C3E50;
            font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
        """)
        value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value)
        
        card.setLayout(layout)
        return card
    
    def create_export_layout(self):
        """创建导出按钮布局"""
        layout = QHBoxLayout()
        
        # 导出CSV按钮
        csv_btn = QPushButton("📄 导出CSV数据")
        csv_btn.setMinimumHeight(35)
        csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(csv_btn)
        
        # 导出图表按钮
        chart_btn = QPushButton("📊 导出所有图表")
        chart_btn.setMinimumHeight(35)
        chart_btn.clicked.connect(self.export_charts)
        layout.addWidget(chart_btn)
        
        # 生成报告按钮
        report_btn = QPushButton("📝 生成完整报告")
        report_btn.setMinimumHeight(35)
        report_btn.clicked.connect(self.generate_report)
        layout.addWidget(report_btn)
        
        return layout
    
    def update_results(self, individuals, eq_info):
        """
        更新结果显示
        
        参数:
            individuals: 均衡个体DataFrame
            eq_info: 均衡信息字典
        """
        self.individuals = individuals
        self.eq_info = eq_info
        
        # 更新指标卡片
        self.update_summary_cards()
        
        # 更新图表
        self.update_all_charts()
    
    def update_summary_cards(self):
        """更新关键指标卡片"""
        # 更新失业率
        unemployment_rate = self.eq_info.get('final_unemployment_rate', 0)
        self.update_card_value(
            self.unemployment_card, 
            f"{unemployment_rate*100:.2f}"
        )
        
        # 更新收敛轮数
        iterations = self.eq_info.get('iterations', 0)
        self.update_card_value(self.iterations_card, str(iterations))
        
        # 更新平均工资
        mean_wage = self.individuals['current_wage'].mean()
        self.update_card_value(self.wage_card, f"{mean_wage:.0f}")
        
        # 更新市场紧张度
        theta = self.eq_info.get('final_theta', 0)
        self.update_card_value(self.theta_card, f"{theta:.2f}")
        
        # 更新平均T值
        mean_T = self.individuals['T'].mean()
        self.update_card_value(self.T_card, f"{mean_T:.1f}")
        
        # 更新平均S值
        mean_S = self.individuals['S'].mean()
        self.update_card_value(self.S_card, f"{mean_S:.2f}")
    
    def update_card_value(self, card, value_text):
        """
        更新指标卡片的值
        
        参数:
            card: 卡片widget
            value_text: 新的值文本
        """
        # 获取卡片中的value标签（第二个子组件）
        layout = card.layout()
        value_label = layout.itemAt(1).widget()
        
        # 获取单位
        old_text = value_label.text()
        parts = old_text.split()
        unit = parts[1] if len(parts) > 1 else ""
        
        # 更新文本
        value_label.setText(f"{value_text} {unit}")
    
    def update_all_charts(self):
        """更新所有图表"""
        # 更新失业率趋势图
        history = self.eq_info.get('history', {})
        if 'unemployment_rate' in history:
            iterations = list(range(len(history['unemployment_rate'])))
            u_rates = [u*100 for u in history['unemployment_rate']]
            self.unemployment_chart.plot_unemployment_trend(iterations, u_rates)
        
        # 更新状态分布图（默认显示T）
        if self.individuals is not None:
            self.distribution_chart.plot_state_distribution(
                self.individuals['T'].values, 
                'T'
            )
        
        # 更新收敛曲线
        if 'diff_V' in history and 'diff_u' in history:
            iterations = list(range(len(history['diff_V'])))
            self.convergence_chart.plot_convergence(
                iterations,
                history['diff_V'],
                history['diff_u']
            )
        
        # 更新市场紧张度图
        if 'theta' in history:
            iterations = list(range(len(history['theta'])))
            self.theta_chart.plot_market_tightness(
                iterations, 
                history['theta']
            )
    
    def on_chart_tab_changed(self, index):
        """
        图表标签页切换时的处理
        
        参数:
            index: 标签页索引
        """
        # 如果切换到状态分布页，可以添加下拉框选择不同状态变量
        pass
    
    def export_csv(self):
        """导出CSV数据"""
        if self.individuals is None:
            QMessageBox.warning(self, "警告", "暂无结果数据，请先运行仿真")
            return
        
        # 选择保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"equilibrium_results_{timestamp}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出CSV数据",
            f"OUTPUT/{default_filename}",
            "CSV文件 (*.csv)"
        )
        
        if file_path:
            self.individuals.to_csv(file_path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "成功", 
                                   f"数据已导出到:\n{file_path}")
    
    def export_charts(self):
        """导出所有图表"""
        if self.individuals is None:
            QMessageBox.warning(self, "警告", "暂无结果数据，请先运行仿真")
            return
        
        # 选择保存目录
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择图表保存目录",
            "OUTPUT"
        )
        
        if dir_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存失业率趋势图
            self.unemployment_chart.figure.savefig(
                f"{dir_path}/unemployment_trend_{timestamp}.png",
                dpi=300, bbox_inches='tight'
            )
            
            # 保存状态分布图
            self.distribution_chart.figure.savefig(
                f"{dir_path}/state_distribution_{timestamp}.png",
                dpi=300, bbox_inches='tight'
            )
            
            # 保存收敛曲线
            self.convergence_chart.figure.savefig(
                f"{dir_path}/convergence_{timestamp}.png",
                dpi=300, bbox_inches='tight'
            )
            
            # 保存市场紧张度图
            self.theta_chart.figure.savefig(
                f"{dir_path}/market_tightness_{timestamp}.png",
                dpi=300, bbox_inches='tight'
            )
            
            QMessageBox.information(self, "成功", 
                                   f"所有图表已导出到:\n{dir_path}")
    
    def generate_report(self):
        """生成完整报告"""
        if self.individuals is None:
            QMessageBox.warning(self, "警告", "暂无结果数据，请先运行仿真")
            return
        
        # 生成Markdown报告
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# MFG均衡求解结果报告

生成时间: {timestamp}

## 关键指标

- 最终失业率: {self.eq_info.get('final_unemployment_rate', 0)*100:.2f}%
- 收敛轮数: {self.eq_info.get('iterations', 0)}
- 市场紧张度: {self.eq_info.get('final_theta', 0):.2f}
- 平均工资: {self.individuals['current_wage'].mean():.0f}元

## 状态变量统计

- 平均T值: {self.individuals['T'].mean():.2f}小时/周
- 平均S值: {self.individuals['S'].mean():.3f}
- 平均D值: {self.individuals['D'].mean():.3f}
- 平均W值: {self.individuals['W'].mean():.0f}元

## 收敛状态

是否收敛: {'是' if self.eq_info.get('converged', False) else '否'}

---

*由EconLab自动生成*
"""
        
        # 保存报告
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存报告",
            f"OUTPUT/report_{timestamp_file}.md",
            "Markdown文件 (*.md);;文本文件 (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            QMessageBox.information(self, "成功", 
                                   f"报告已生成:\n{file_path}")

