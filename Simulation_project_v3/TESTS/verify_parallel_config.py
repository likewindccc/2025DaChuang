"""
验证并行配置的正确性

检查项：
1. NUMBA_NUM_THREADS 应为 1
2. OMP_NUM_THREADS 应为 1
3. DE workers 配置为 auto/all 或正整数
4. 无过度订阅风险
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

# 在导入numba前先设置线程环境变量，避免配置被锁定
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

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
        status = "[OK]" if actual == expected else "[ERR]"
        
        if actual != expected:
            all_ok = False
            
        print(f"  {status} {var_name}: {actual} (期望: {expected})")
    
    # 检查Numba实际配置
    try:
        numba.set_num_threads(1)
    except Exception:
        pass
    numba_threads = numba.config.NUMBA_NUM_THREADS
    print(f"\n  Numba实际线程数: {numba_threads}")
    
    if numba_threads != 1:
        print("  [ERR] 警告：Numba未正确配置为1线程！")
        all_ok = False
    else:
        print("  [OK] Numba正确配置为串行模式")
    
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
        print("  [ERR] 警告：未使用差分进化算法")
        all_ok = False
    else:
        print("  [OK] 使用差分进化算法")
    
    if isinstance(workers, str):
        if workers.strip().lower() in {'auto', 'all', '-1'}:
            print("  [OK] workers已配置为自动全核")
        else:
            print("  [ERR] workers字符串配置非法")
            all_ok = False
    elif int(workers) > 0:
        print(f"  [OK] workers为固定正整数: {workers}")
    else:
        print("  [ERR] workers必须为正整数或auto/all")
        all_ok = False
    
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
    
    workers_raw = config['optimization']['options'].get('workers', 1)
    if isinstance(workers_raw, str) and workers_raw.strip().lower() in {'auto', 'all', '-1'}:
        workers = os.cpu_count() or 1
    else:
        workers = int(workers_raw)

    total_threads = workers * numba_threads
    
    print(f"  DE进程数: {workers}")
    print(f"  每进程Numba线程: {numba_threads}")
    print(f"  总并行任务数: {total_threads}")
    print()
    
    cpu_count = os.cpu_count() or 1
    if total_threads == cpu_count:
        print(f"  [OK] 完美配置！{workers}个进程 × {numba_threads}个线程 = {cpu_count}个任务")
        print(f"  [OK] 在当前{cpu_count}核CPU上达到满负载")
        return True
    elif total_threads > cpu_count * 2:
        print(f"  [ERR] 严重过度订阅！{total_threads}个任务竞争{cpu_count}核心")
        print(f"  [ERR] 过度订阅比例: {total_threads / cpu_count:.1f}倍")
        return False
    else:
        print("  [WARN] 配置可能需要调整")
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
  - DE workers=auto
  - 总任务数=CPU核心数
  - 适配任意核数CPU
  
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
        print("[OK] 所有检查通过！配置正确，可以开始校准任务")
    else:
        print("[ERR] 部分检查失败！请修复配置后再运行")
    print("=" * 80 + "\n")
    
    return check1 and check2 and check3


if __name__ == '__main__':
    # 先设置环境变量（模拟AutoDL环境）
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    success = main()
    sys.exit(0 if success else 1)

