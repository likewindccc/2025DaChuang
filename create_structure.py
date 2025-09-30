"""
项目目录结构初始化脚本
自动创建Simulation_project_v2的完整目录结构
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(r"D:\Python\2025大创\Simulation_project_v2")

# 完整的目录结构定义
DIRECTORY_STRUCTURE = {
    "src": {
        "core": {},
        "modules": {
            "population": {},
            "matching": {},
            "estimation": {},
            "mfg": {},
            "calibration": {}
        },
        "utils": {}
    },
    "tests": {
        "unit": {},
        "integration": {},
        "benchmarks": {}
    },
    "config": {
        "default": {},
        "experiments": {}
    },
    "data": {
        "input": {},
        "output": {}
    },
    "results": {
        "figures": {},
        "reports": {},
        "logs": {}
    },
    "docs": {
        "userdocs": {},
        "developerdocs": {},
        "academicdocs": {}
    }
}

def create_directory_structure(base_path: Path, structure: dict):
    """递归创建目录结构"""
    for dir_name, sub_structure in structure.items():
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
        
        # 递归创建子目录
        if sub_structure:
            create_directory_structure(dir_path, sub_structure)
        
        # 创建__init__.py文件（仅src目录下）
        if str(dir_path).startswith(str(PROJECT_ROOT / "src")):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""模块: {dir_name}"""\n', encoding='utf-8')
                print(f"  └─ 创建文件: __init__.py")

def main():
    print("=" * 60)
    print("🚀 Simulation_project_v2 目录结构初始化")
    print("=" * 60)
    
    # 创建项目根目录
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 项目根目录: {PROJECT_ROOT}\n")
    
    # 创建完整目录结构
    create_directory_structure(PROJECT_ROOT, DIRECTORY_STRUCTURE)
    
    print("\n" + "=" * 60)
    print("✅ 目录结构创建完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
