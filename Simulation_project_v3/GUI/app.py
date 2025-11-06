"""
EconLab应用启动入口

农村女性就业市场MFG模拟系统 - 桌面应用
"""

import sys
import os
from pathlib import Path

# 针对PyInstaller打包环境：添加DLL搜索路径（必须在导入numba之前）
if getattr(sys, 'frozen', False):
    # 应用被打包后运行的路径处理逻辑
    application_path = Path(sys.executable).parent
    internal_path = application_path / '_internal'
    log_path = application_path / 'runtime_boot.log'

    def _runtime_log(message: str) -> None:
        # 在打包环境下写入调试日志，便于排查 DLL 加载问题
        try:
            with open(log_path, 'a', encoding='utf-8') as fh:
                fh.write(f"{message}\n")
        except Exception:
            pass

    _runtime_log("==== 应用启动 ====")
    _runtime_log(f"application_path: {application_path}")
    _runtime_log(f"_internal exists: {internal_path.exists()}")
    
    # 方案1：添加到PATH环境变量（最可靠）
    llvmlite_binding_path = internal_path / 'llvmlite' / 'binding'
    if llvmlite_binding_path.exists():
        print(f"添加DLL搜索路径到PATH: {llvmlite_binding_path}")
        os.environ['PATH'] = str(llvmlite_binding_path) + os.pathsep + os.environ.get('PATH', '')
        _runtime_log(f"添加PATH: {llvmlite_binding_path}")
    else:
        _runtime_log(f"llvmlite_binding_path缺失: {llvmlite_binding_path}")
    
    # 添加 _internal 目录到PATH
    if internal_path.exists():
        print(f"添加DLL搜索路径到PATH: {internal_path}")
        os.environ['PATH'] = str(internal_path) + os.pathsep + os.environ.get('PATH', '')
        _runtime_log(f"添加PATH: {internal_path}")
    else:
        _runtime_log(f"_internal缺失: {internal_path}")
    
    # 方案2：使用os.add_dll_directory（Windows 10+）
    if hasattr(os, 'add_dll_directory'):
        if llvmlite_binding_path.exists():
            try:
                os.add_dll_directory(str(llvmlite_binding_path))
                print(f"使用add_dll_directory添加: {llvmlite_binding_path}")
                _runtime_log(f"add_dll_directory OK: {llvmlite_binding_path}")
            except Exception as e:
                print(f"add_dll_directory失败: {e}")
                _runtime_log(f"add_dll_directory失败(binding): {e}")
        
        if internal_path.exists():
            try:
                os.add_dll_directory(str(internal_path))
                print(f"使用add_dll_directory添加: {internal_path}")
                _runtime_log(f"add_dll_directory OK: {internal_path}")
            except Exception as e:
                print(f"add_dll_directory失败: {e}")
                _runtime_log(f"add_dll_directory失败(_internal): {e}")
    
    # 方案3：强制复制所有关键DLL到应用根目录（最后的保险）
    llvmlite_dll_source = llvmlite_binding_path / 'llvmlite.dll'
    llvmlite_dll_target = application_path / 'llvmlite.dll'
    try:
        import shutil
        # 总是复制，覆盖旧文件
        if llvmlite_dll_source.exists():
            shutil.copy2(str(llvmlite_dll_source), str(llvmlite_dll_target))
            print(f"复制llvmlite.dll到应用根目录: {llvmlite_dll_target}")
            _runtime_log(f"复制llvmlite.dll到应用根目录: {llvmlite_dll_target}")
        else:
            _runtime_log(f"llvmlite.dll源文件缺失: {llvmlite_dll_source}")
        
        # 同时也复制到 _internal 根目录（冗余）
        llvmlite_dll_target2 = internal_path / 'llvmlite.dll'
        if llvmlite_dll_source.exists():
            shutil.copy2(str(llvmlite_dll_source), str(llvmlite_dll_target2))
            print(f"复制llvmlite.dll到_internal目录: {llvmlite_dll_target2}")
            _runtime_log(f"复制llvmlite.dll到_internal目录: {llvmlite_dll_target2}")
    except Exception as e:
        print(f"复制DLL失败: {e}")
        _runtime_log(f"复制DLL失败: {e}")
# 添加项目根目录到Python路径
if getattr(sys, 'frozen', False):
    # 打包后的环境：资源在 _internal 目录
    application_path = Path(sys.executable).parent
    project_root = application_path / '_internal'
else:
    # 开发环境
    project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))

# 设置工作目录为项目根目录（确保相对路径正确）
os.chdir(str(project_root))

# 关键修复：在导入PyQt6之前先导入numba
# 避免PyQt6的DLL与llvmlite的DLL冲突
print("正在加载Numba模块（在PyQt6之前）...")
try:
    # 【关键修复】在打包环境中，手动预加载 llvmlite.dll
    if getattr(sys, 'frozen', False):
        import ctypes
        llvmlite_dll = project_root / 'llvmlite' / 'binding' / 'llvmlite.dll'
        if llvmlite_dll.exists():
            print(f"手动预加载 llvmlite.dll: {llvmlite_dll}")
            ctypes.CDLL(str(llvmlite_dll))
    
    import numba
    from MODULES.MFG.bellman_solver import BellmanSolver
    from MODULES.MFG.kfe_solver import KFESolver
    print("Numba模块加载成功！")
    NUMBA_AVAILABLE = True
except Exception as e:
    print(f"Numba加载失败: {e}")
    import traceback
    traceback.print_exc()
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

