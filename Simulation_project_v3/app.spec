# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller配置文件

用于将EconLab打包为Windows可执行文件
"""

block_cipher = None


a = Analysis(
    ['GUI/app.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 配置文件
        ('CONFIG', 'CONFIG'),
        # 预处理数据（如果需要）
        ('DATA/processed', 'DATA/processed'),
        # GUI资源
        ('GUI/resources', 'GUI/resources'),
        # 匹配函数模型（预训练模型）
        ('OUTPUT/logistic/match_function_model.pkl', 'OUTPUT/logistic'),
        # 人口分布参数
        ('OUTPUT/population/labor_distribution_params.pkl', 'OUTPUT/population'),
    ],
    hiddenimports=[
        # 核心计算库
        'numba',
        'scipy._lib.messagestream',
        'scipy.special.cython_special',
        'copulas',
        # GUI库
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        # matplotlib后端
        'matplotlib.backends.backend_qtagg',
        # 其他可能缺失的模块
        'statsmodels',
        'pandas',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不必要的库
        'matplotlib.tests',
        'pytest',
        'IPython',
        'notebook',
        'sphinx',
        'PIL',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EconLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

