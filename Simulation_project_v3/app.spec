# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
from PyInstaller.building.datastruct import Tree

# 收集 numba 和 llvmlite 的数据文件和动态库
try:
    numba_datas = collect_data_files('numba')
    numba_binaries = collect_dynamic_libs('numba')
    llvmlite_datas = collect_data_files('llvmlite')
    llvmlite_binaries = collect_dynamic_libs('llvmlite')
    
    # 手动添加 llvmlite 整个包作为Tree（不压缩）
    import llvmlite
    llvmlite_path = os.path.dirname(llvmlite.__file__)
    llvmlite_tree = Tree(llvmlite_path, prefix='llvmlite', excludes=['__pycache__', '*.pyc'])
    
    # 关键修复：明确将 llvmlite.dll 放到根目录（PyInstaller会找到）
    llvmlite_dll_path = os.path.join(llvmlite_path, 'binding', 'llvmlite.dll')
    if os.path.exists(llvmlite_dll_path):
        # 添加到 binaries，放到根目录（'.' 表示应用根目录）
        llvmlite_binaries.append((llvmlite_dll_path, '.'))
        print(f"Added llvmlite.dll to root directory: {llvmlite_dll_path}")
    
    # 【关键修复2】添加 MSVCP140 系列 DLL（llvmlite.dll 依赖）
    import PyQt6
    pyqt6_bin = os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'bin')
    for msvcp_dll in ['msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll']:
        msvcp_path = os.path.join(pyqt6_bin, msvcp_dll)
        if os.path.exists(msvcp_path):
            llvmlite_binaries.append((msvcp_path, '.'))
            print(f"Added {msvcp_dll} to root directory for llvmlite")
except Exception as e:
    print(f"Warning: Could not collect numba/llvmlite files: {e}")
    numba_datas = []
    numba_binaries = []
    llvmlite_datas = []
    llvmlite_binaries = []
    llvmlite_tree = []

block_cipher = None

a = Analysis(
    ['GUI/app.py'],
    pathex=[],
    binaries=numba_binaries + llvmlite_binaries,
    datas=[
        ('GUI/resources', 'resources'),
        ('CONFIG', 'CONFIG'),
        ('MODULES', 'MODULES'),
        ('OUTPUT', 'OUTPUT'),
    ] + numba_datas + llvmlite_datas,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
        'numpy',
        'scipy',
        'yaml',
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'pandas',
        'GUI.pages.config_page',
        'GUI.pages.simulation_page',
        'GUI.pages.results_page',
        'GUI.widgets.parameter_widget',
        'GUI.widgets.chart_widget',
        'GUI.widgets.metric_card',
        'GUI.widgets.colored_log_widget',
        'GUI.workers.simulation_worker',
        'GUI.workers.demo_worker',
        'GUI.utils.config_manager',
        'GUI.utils.style_manager',
        'numba',
        'numba.core',
        'numba.core.types',
        'numba.core.typing',
        'numba.core.typing.typeof',
        'numba.typed',
        'numba.typed.typedlist',
        'numba.typed.typeddict',
        'llvmlite',
        'llvmlite.binding',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EconLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='GUI/ChatGPT-Image-2025年10月29日-23_50_54.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    llvmlite_tree,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EconLab',
)

