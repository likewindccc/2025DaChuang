#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目目录结构初始化脚本
自动创建完整的目录结构和 __init__.py 文件
"""
import os
from pathlib import Path

# 获取当前脚本所在目录作为项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 完整的目录结构
DIRECTORIES = [
    # 源代码目录
    "src",
    "src/core",
    "src/modules",
    "src/modules/population",
    "src/modules/matching",
    "src/modules/estimation",
    "src/modules/mfg",
    "src/modules/calibration",
    "src/utils",
    
    # 测试目录
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/benchmarks",
    
    # 配置目录
    "config",
    "config/default",
    "config/experiments",
    
    # 数据目录
    "data",
    "data/input",
    "data/output",
    
    # 结果目录
    "results",
    "results/figures",
    "results/reports",
    "results/logs",
    
    # 文档目录
    "docs",
    "docs/userdocs",
    "docs/developerdocs",
    "docs/academicdocs",
]

# 需要创建 __init__.py 的目录
INIT_DIRS = [
    "src",
    "src/core",
    "src/modules",
    "src/modules/population",
    "src/modules/matching",
    "src/modules/estimation",
    "src/modules/mfg",
    "src/modules/calibration",
    "src/utils",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/benchmarks",
]

def create_directory_structure():
    """创建完整的目录结构"""
    print("=" * 70)
    print("🚀 Simulation_project_v2 目录结构初始化")
    print("=" * 70)
    print(f"\n📁 项目根目录: {PROJECT_ROOT}\n")
    
    created_count = 0
    
    # 创建所有目录
    for dir_path in DIRECTORIES:
        full_path = PROJECT_ROOT / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")
            created_count += 1
        else:
            print(f"  跳过(已存在): {dir_path}")
    
    print(f"\n📝 创建 __init__.py 文件...\n")
    
    # 创建 __init__.py 文件
    init_count = 0
    for dir_path in INIT_DIRS:
        full_path = PROJECT_ROOT / dir_path
        init_file = full_path / "__init__.py"
        
        if not init_file.exists():
            # 根据目录深度生成合适的文档字符串
            module_name = dir_path.replace("/", ".").replace("\\", ".")
            doc_string = f'"""{module_name} - 模块初始化文件"""\n'
            
            init_file.write_text(doc_string, encoding='utf-8')
            print(f"✓ 创建文件: {dir_path}/__init__.py")
            init_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ 完成! 创建了 {created_count} 个目录和 {init_count} 个初始化文件")
    print("=" * 70)

if __name__ == "__main__":
    try:
        create_directory_structure()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
