import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.CALIBRATION import SMMCalibrator


def test_calibration_quick():
    """
    快速测试（5次迭代）
    
    目的：
    1. 验证流程完整性
    2. 验证断点保存/恢复
    3. 验证输出文件生成
    
    预计时间：30-40分钟
    """
    print("\n" + "="*80)
    print("CALIBRATION模块快速测试")
    print("="*80)
    print("测试类型：快速测试（5次迭代）")
    print("预计时间：30-40分钟")
    print("="*80)
    
    # 确认用户是否继续
    response = input("\n是否开始快速测试？(y/n): ")
    if response.lower() != 'y':
        print("测试已取消")
        return
    
    # 创建校准器
    config_path = "CONFIG/calibration_config.yaml"
    calibrator = SMMCalibrator(config_path)
    
    # 修改配置为快速测试模式
    calibrator.config['optimization']['options']['maxiter'] = 5
    calibrator.config['optimization']['options']['maxfev'] = 10
    
    print("\n快速测试配置：")
    print("  最大迭代次数: 5")
    print("  最大函数评估次数: 10")
    
    # 运行校准
    result = calibrator.calibrate()
    
    # 打印结果
    print("\n" + "="*80)
    print("快速测试完成")
    print("="*80)
    print(f"优化成功: {result.success}")
    print(f"函数评估次数: {result.nfev}")
    print(f"最优SMM距离: {result.fun:.6f}")
    
    # 检查输出文件
    output_dir = Path("OUTPUT/calibration")
    
    print("\n检查输出文件:")
    
    files_to_check = [
        'calibration_history.csv',
        'calibrated_parameters.yaml',
        'optimization_result.pkl'
    ]
    
    for filename in files_to_check:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (未找到)")
    
    print("\n" + "="*80)
    print("快速测试结束")
    print("="*80)


def test_calibration_small_scale():
    """
    小规模测试（20次迭代）
    
    目的：
    1. 验证收敛趋势
    2. 验证结果输出
    3. 测试断点续跑
    
    预计时间：2-3小时
    """
    print("\n" + "="*80)
    print("CALIBRATION模块小规模测试")
    print("="*80)
    print("测试类型：小规模测试（20次迭代）")
    print("预计时间：2-3小时")
    print("="*80)
    
    # 确认用户是否继续
    response = input("\n是否开始小规模测试？(y/n): ")
    if response.lower() != 'y':
        print("测试已取消")
        return
    
    # 创建校准器
    config_path = "CONFIG/calibration_config.yaml"
    calibrator = SMMCalibrator(config_path)
    
    # 修改配置为小规模测试
    calibrator.config['optimization']['options']['maxiter'] = 20
    calibrator.config['optimization']['options']['maxfev'] = 50
    
    print("\n小规模测试配置：")
    print("  最大迭代次数: 20")
    print("  最大函数评估次数: 50")
    print("  断点保存频率: 每10次评估")
    
    # 运行校准
    result = calibrator.calibrate()
    
    # 打印结果
    print("\n" + "="*80)
    print("小规模测试完成")
    print("="*80)
    print(f"优化成功: {result.success}")
    print(f"函数评估次数: {result.nfev}")
    print(f"最优SMM距离: {result.fun:.6f}")
    
    # 打印参数
    print("\n最优参数:")
    param_names = calibrator.param_utils.get_param_names()
    for name, value in zip(param_names, result.x):
        print(f"  {name}: {value:.6f}")
    
    print("\n" + "="*80)
    print("小规模测试结束")
    print("="*80)


def test_calibration_full():
    """
    完整测试（运行至收敛或200次迭代）
    
    目的：
    实际校准运行
    
    预计时间：4-20小时
    """
    print("\n" + "="*80)
    print("CALIBRATION模块完整测试")
    print("="*80)
    print("测试类型：完整校准（运行至收敛）")
    print("预计时间：4-20小时")
    print("="*80)
    
    # 确认用户是否继续
    response = input("\n是否开始完整校准？(y/n): ")
    if response.lower() != 'y':
        print("测试已取消")
        return
    
    # 创建校准器
    config_path = "CONFIG/calibration_config.yaml"
    calibrator = SMMCalibrator(config_path)
    
    print("\n完整测试配置：")
    print("  最大迭代次数: 200")
    print("  最大函数评估次数: 1000")
    print("  断点保存频率: 每10次评估")
    print("  自动恢复断点: 启用")
    
    # 运行校准
    result = calibrator.calibrate()
    
    # 打印结果
    print("\n" + "="*80)
    print("完整测试完成")
    print("="*80)
    print(f"优化成功: {result.success}")
    print(f"函数评估次数: {result.nfev}")
    print(f"最优SMM距离: {result.fun:.6f}")
    
    # 打印最优评估
    calibrator.obj_function.print_best_evaluation()
    
    print("\n" + "="*80)
    print("完整测试结束")
    print("="*80)


def main():
    """主测试入口"""
    print("\n" + "="*80)
    print("CALIBRATION模块测试")
    print("="*80)
    print("\n请选择测试类型：")
    print("  1. 快速测试（5次迭代，约30-40分钟）")
    print("  2. 小规模测试（20次迭代，约2-3小时）")
    print("  3. 完整测试（运行至收敛，约4-20小时）")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-3): ")
    
    if choice == '1':
        test_calibration_quick()
    elif choice == '2':
        test_calibration_small_scale()
    elif choice == '3':
        test_calibration_full()
    elif choice == '0':
        print("退出测试")
    else:
        print("无效选项")


if __name__ == '__main__':
    main()

