"""
EconLab应用启动入口

农村女性就业市场MFG模拟系统 - 桌面应用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 关键修复：在导入PyQt6之前先导入numba
# 避免PyQt6的DLL与llvmlite的DLL冲突
print("正在加载Numba模块（在PyQt6之前）...")
try:
    import numba
    from MODULES.MFG.bellman_solver import BellmanSolver
    from MODULES.MFG.kfe_solver import KFESolver
    print("Numba模块加载成功！")
    NUMBA_AVAILABLE = True
except Exception as e:
    print(f"Numba加载失败: {e}")
    print("将使用演示模式")
    NUMBA_AVAILABLE = False

# 现在才导入PyQt6
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from GUI.main_window import MainWindow


def setup_warnings():
    """配置警告过滤"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*missing from font.*')


def exception_hook(exctype, value, traceback):
    """
    全局异常钩子
    
    捕获所有未处理的异常，防止应用崩溃
    这是GUI应用必需的异常处理机制
    """
    import traceback as tb
    error_msg = ''.join(tb.format_exception(exctype, value, traceback))
    print(f"未捕获的异常:\n{error_msg}")
    
    # 显示错误对话框
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.critical(
        None,
        "程序错误",
        f"发生未预期的错误:\n\n{exctype.__name__}: {str(value)}"
    )


def main():
    """应用主函数"""
    # 设置全局异常钩子（防止崩溃）
    sys.excepthook = exception_hook
    
    # 过滤中文字体警告
    setup_warnings()
    
    # 设置高DPI缩放
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("EconLab")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("农村女性就业市场动态演化机制团队")
    
    print("启动EconLab GUI...")
    
    # 创建并显示主窗口
    # 根据Numba是否可用，决定默认模式
    window = MainWindow(use_demo_mode=not NUMBA_AVAILABLE)
    
    if NUMBA_AVAILABLE:
        print("真实模式已启用")
    else:
        print("演示模式已启用（Numba不可用）")
    
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

