"""
演示模式工作线程

使用模拟数据进行演示，不调用真实的MFG求解器
适合展示界面效果，绕过Numba DLL问题
"""

import time
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class DemoWorker(QThread):
    """演示模式工作线程"""
    
    progress_updated = pyqtSignal(int, dict)
    finished = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str, str)
    
    def __init__(self, config_dict):
        """
        初始化演示线程
        
        参数:
            config_dict: 配置字典
        """
        super().__init__()
        self.config_dict = config_dict
        self.is_stopped = False
    
    def run(self):
        """运行模拟的MFG求解过程"""
        self.log_message.emit("INFO", "演示模式：开始模拟MFG均衡求解...")
        
        n_individuals = self.config_dict['population']['n_individuals']
        max_iter = self.config_dict['equilibrium']['max_outer_iter']
        
        self.log_message.emit("INFO", f"初始化{n_individuals}个个体...")
        time.sleep(0.5)
        
        # 生成模拟的个体数据
        individuals = self._generate_demo_individuals(n_individuals)
        
        self.log_message.emit("INFO", "开始迭代求解均衡...")
        
        # 模拟迭代过程
        for iteration in range(1, max_iter + 1):
            if self.is_stopped:
                self.log_message.emit("WARN", "演示已停止")
                return
            
            # 模拟每轮的统计数据
            stats = self._generate_iteration_stats(iteration, max_iter)
            
            # 发送进度
            self.progress_updated.emit(iteration, stats)
            
            # 模拟计算时间
            time.sleep(0.1)
            
            # 模拟收敛（在70-90%之间收敛）
            if iteration > max_iter * 0.7:
                if stats['diff_V'] < 0.01 and stats['diff_u'] < 0.001:
                    self.log_message.emit("INFO", f"第{iteration}轮收敛！")
                    break
        
        # 生成最终结果
        eq_info = self._generate_final_results(iteration, individuals)
        
        self.log_message.emit("INFO", "演示模式：MFG均衡求解完成！")
        self.finished.emit(individuals, eq_info)
    
    def _generate_demo_individuals(self, n):
        """生成模拟的个体数据"""
        individuals = pd.DataFrame({
            'T': np.random.normal(45, 8, n),
            'S': np.random.beta(2, 2, n),
            'D': np.random.beta(1.5, 2, n),
            'W': np.random.normal(4500, 1200, n),
            'current_wage': np.random.normal(4500, 1500, n),
            'employment_status': np.random.choice(
                ['employed', 'unemployed'], 
                n, 
                p=[0.95, 0.05]
            )
        })
        return individuals
    
    def _generate_iteration_stats(self, iteration, max_iter):
        """生成每轮迭代的统计数据"""
        # 模拟失业率从6%降到4%
        progress = iteration / max_iter
        u_rate = 0.06 - 0.02 * progress + np.random.normal(0, 0.003)
        u_rate = max(0.03, min(0.07, u_rate))
        
        # 模拟收敛指标逐渐变小
        diff_V = 0.05 * (1 - progress) + np.random.normal(0, 0.005)
        diff_V = max(0, diff_V)
        
        diff_u = 0.003 * (1 - progress) + np.random.normal(0, 0.0005)
        diff_u = max(0, diff_u)
        
        # 模拟工资
        mean_wage = 4400 + 200 * progress + np.random.normal(0, 50)
        
        stats = {
            'unemployment_rate': u_rate,
            'theta': 1.5,
            'mean_wage': mean_wage,
            'mean_T': 45.2,
            'mean_S': 0.72,
            'diff_V': diff_V,
            'diff_u': diff_u
        }
        
        return stats
    
    def _generate_final_results(self, iterations, individuals):
        """生成最终结果信息"""
        # 生成模拟的历史数据
        history = {
            'unemployment_rate': [],
            'theta': [],
            'diff_V': [],
            'diff_u': []
        }
        
        for i in range(1, iterations + 1):
            progress = i / iterations
            history['unemployment_rate'].append(
                0.06 - 0.02 * progress
            )
            history['theta'].append(1.5)
            history['diff_V'].append(0.05 * (1 - progress))
            history['diff_u'].append(0.003 * (1 - progress))
        
        eq_info = {
            'converged': True,
            'iterations': iterations,
            'final_unemployment_rate': 0.04,
            'final_theta': 1.5,
            'history': history
        }
        
        return eq_info
    
    def stop(self):
        """停止运行"""
        self.is_stopped = True

