"""
MFG仿真后台工作线程

在后台运行MFG求解，避免阻塞GUI主线程
"""

import sys
import os
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import yaml
import tempfile


class SimulationWorker(QThread):
    """MFG仿真工作线程类"""
    
    progress_updated = pyqtSignal(int, dict)
    finished = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str, str)
    
    def __init__(self, config_dict):
        """
        初始化工作线程
        
        参数:
            config_dict: 配置字典
        """
        super().__init__()
        self.config_dict = config_dict
        self.is_paused = False
        self.is_stopped = False
    
    def run(self):
        """
        运行MFG求解（在后台线程中）
        
        注意：这里的异常处理是GUI架构必需的，
        用于捕获错误并通过信号传递给主线程显示，避免应用崩溃
        """
        temp_config_path = None
        
        try:
            self.log_message.emit("INFO", "开始MFG均衡求解...")
            
            # 创建临时配置文件
            temp_config = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.yaml',
                delete=False,
                encoding='utf-8'
            )
            temp_config_path = temp_config.name
            
            yaml.dump(self.config_dict, temp_config, 
                     allow_unicode=True, default_flow_style=False)
            temp_config.close()
            
            self.log_message.emit("INFO", "初始化MFG求解器...")
            
            # 导入MFG模块
            from MODULES.MFG.equilibrium_solver import EquilibriumSolver
            
            # 创建求解器
            solver = EquilibriumSolver(temp_config_path)
            
            self.log_message.emit(
                "INFO", 
                f"初始化{self.config_dict['population']['n_individuals']}个个体..."
            )
            
            # 初始化人口
            individuals = solver.initialize_population()
            
            self.log_message.emit("INFO", "开始迭代求解均衡...")
            
            # 求解均衡，传入进度回调
            individuals_eq, eq_info = solver.solve(
                individuals=individuals,
                verbose=False,
                callback=self._progress_callback
            )
            
            # 删除临时配置文件
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            # 发送完成信号
            self.log_message.emit("INFO", "MFG均衡求解完成！")
            self.finished.emit(individuals_eq, eq_info)
            
        except Exception as e:
            # GUI架构必需的异常处理：捕获错误并通过信号传递
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log_message.emit("ERROR", f"运行出错: {error_msg}")
            self.error_occurred.emit(error_msg)
            
            # 清理临时文件
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def _progress_callback(self, iteration, stats):
        """
        进度回调函数
        
        参数:
            iteration: 当前迭代轮数
            stats: 当前统计数据字典
        """
        if self.is_stopped:
            return
        
        # 发送进度信号
        self.progress_updated.emit(iteration, stats)
        
        # 发送日志
        unemployment_rate = stats.get('unemployment_rate', 0)
        self.log_message.emit(
            "INFO",
            f"第{iteration}轮: 失业率={unemployment_rate*100:.2f}%"
        )
    
    def pause(self):
        """暂停运行"""
        self.is_paused = True
    
    def resume(self):
        """继续运行"""
        self.is_paused = False
    
    def stop(self):
        """停止运行"""
        self.is_stopped = True

