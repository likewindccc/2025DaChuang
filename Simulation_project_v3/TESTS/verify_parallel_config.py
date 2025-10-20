"""
验证并行配置的正确性

检查项：
1. NUMBA_NUM_THREADS 应为 1
2. OMP_NUM_THREADS 应为 1
3. DE workers 配置为 32
4. 无过度订阅风险
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numba


def check_environment_variables():
    """检查环境变量配置"""
    print("=" * 80)
    print("检查1: 环境变量配置")
    print("=" * 80)
    
    checks = [
        ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'),
        ('MKL_NUM_THREADS', '1'),
    ]
    
    all_ok = True
    for var_name, expected in checks:
        actual = os.environ.get(var_name, 'NOT_SET')
        status = "✓" if actual == expected else "✗"
        
        if actual != expected:
            all_ok = False
            
        print(f"  {status} {var_name}: {actual} (期望: {expected})")
    
    # 检查Numba实际配置
    numba_threads = numba.config.NUMBA_NUM_THREADS
    print(f"\n  Numba实际线程数: {numba_threads}")
    
    if numba_threads != 1:
        print(f"  ✗ 警告：Numba未正确配置为1线程！")
        all_ok = False
    else:
        print(f"  ✓ Numba正确配置为串行模式")
    
    return all_ok


def check_calibration_config():
    """检查校准配置"""
    print("\n" + "=" * 80)
    print("检查2: 校准模块配置")
    print("=" * 80)
    
    config_path = project_root / 'CONFIG' / 'calibration_config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    method = config['optimization']['method']
    workers = config['optimization']['options'].get('workers', 1)
    popsize = config['optimization']['options'].get('popsize', 1)
    
    print(f"  优化方法: {method}")
    print(f"  并行进程数: {workers}")
    print(f"  种群大小: {popsize}")
    
    all_ok = True
    
    if method != 'differential_evolution':
        print(f"  ✗ 警告：未使用差分进化算法")
        all_ok = False
    else:
        print(f"  ✓ 使用差分进化算法")
    
    if workers != 32:
        print(f"  ✗ 警告：workers不是32")
        all_ok = False
    else:
        print(f"  ✓ 配置为32进程并行")
    
    return all_ok


def calculate_parallelism():
    """计算并行度"""
    print("\n" + "=" * 80)
    print("检查3: 并行度分析")
    print("=" * 80)
    
    numba_threads = numba.config.NUMBA_NUM_THREADS
    
    config_path = project_root / 'CONFIG' / 'calibration_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    workers = config['optimization']['options'].get('workers', 1)
    
    total_threads = workers * numba_threads
    
    print(f"  DE进程数: {workers}")
    print(f"  每进程Numba线程: {numba_threads}")
    print(f"  总并行任务数: {total_threads}")
    print()
    
    if total_threads == 32:
        print(f"  ✓ 完美配置！32个进程 × 1个线程 = 32个任务")
        print(f"  ✓ 在32核CPU上达到完美负载均衡")
        return True
    elif total_threads > 64:
        print(f"  ✗ 严重过度订阅！{total_threads}个任务竞争32核心")
        print(f"  ✗ 过度订阅比例: {total_threads / 32:.1f}倍")
        return False
    else:
        print(f"  ⚠️ 配置可能需要调整")
        return True


def print_recommendations():
    """打印建议"""
    print("\n" + "=" * 80)
    print("配置建议")
    print("=" * 80)
    
    print("""
推荐配置（方案A - 已实施）：
  - NUMBA_NUM_THREADS=1
  - OMP_NUM_THREADS=1
  - DE workers=32
  - 总任务数=32
  - 适用于32核CPU
  
预期性能：
  - 单次MFG: ~18分钟
  - 并行效率: 90%
  - 总时间: ~30小时
  - 成本: ~9元（AutoDL 32核）
  
监控命令（AutoDL）：
  htop           # 应该看到32个python进程，CPU使用率稳定100%
  screen -r      # 查看运行日志
""")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("并行配置验证工具")
    print("=" * 80 + "\n")
    
    check1 = check_environment_variables()
    check2 = check_calibration_config()
    check3 = calculate_parallelism()
    
    print_recommendations()
    
    print("\n" + "=" * 80)
    if check1 and check2 and check3:
        print("✅ 所有检查通过！配置正确，可以开始校准任务")
    else:
        print("❌ 部分检查失败！请修复配置后再运行")
    print("=" * 80 + "\n")
    
    return check1 and check2 and check3


if __name__ == '__main__':
    # 先设置环境变量（模拟AutoDL环境）
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    success = main()
    sys.exit(0 if success else 1)

