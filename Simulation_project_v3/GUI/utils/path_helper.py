"""
路径处理工具

处理打包后的资源文件路径问题
"""

import sys
import os
from pathlib import Path


def get_resource_path(relative_path):
    """
    获取资源文件的绝对路径（兼容开发环境和打包后环境）
    
    参数:
        relative_path: 相对路径，如 "CONFIG/mfg_config.yaml"
    
    返回:
        资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # 打包后，资源文件在_MEIPASS临时目录
        base_path = sys._MEIPASS
    else:
        # 开发环境，返回项目根目录
        base_path = Path(__file__).parent.parent.parent
    
    return os.path.join(base_path, relative_path)


def get_output_dir():
    """
    获取输出目录路径
    
    返回:
        OUTPUT目录的绝对路径
    """
    return get_resource_path("OUTPUT")


def ensure_dir_exists(dir_path):
    """
    确保目录存在，不存在则创建
    
    参数:
        dir_path: 目录路径
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)

