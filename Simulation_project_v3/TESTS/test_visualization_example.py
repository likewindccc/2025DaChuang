"""
可视化模块使用示例

演示如何使用VISUALIZATION模块生成各类图表
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from MODULES.VISUALIZATION import (
    initialize_styles,
    DataVisualizer,
    MFGVisualizer,
    CalibrationVisualizer,
    SimulationVisualizer,
    DashboardBuilder
)


def test_data_visualization():
    """测试数据可视化器"""
    print("\n=== 测试数据可视化器 ===")
    
    # 创建模拟数据
    n_individuals = 10000
    individuals = pd.DataFrame({
        'T': np.random.normal(45, 8, n_individuals),
        'S': np.random.beta(2, 5, n_individuals),
        'D': np.random.beta(2, 5, n_individuals),
        'W': np.random.lognormal(8.4, 0.3, n_individuals)
    })
    
    # 创建可视化器
    visualizer = DataVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 绘制初始分布
    static_path, interactive_path = visualizer.plot_initial_distribution(individuals)
    print(f"✓ 静态图: {static_path}")
    print(f"✓ 交互图: {interactive_path}")
    
    # 绘制Copula结构
    copula_path = visualizer.plot_copula_structure(individuals)
    print(f"✓ Copula图: {copula_path}")


def test_mfg_visualization():
    """测试MFG可视化器"""
    print("\n=== 测试MFG可视化器 ===")
    
    # 创建模拟数据
    n_individuals = 5000
    individuals = pd.DataFrame({
        'T': np.random.normal(47, 7, n_individuals),
        'S': np.random.beta(2.5, 4, n_individuals),
        'D': np.random.beta(2.5, 4, n_individuals),
        'W': np.random.lognormal(8.4, 0.3, n_individuals),
        'V_U': np.random.uniform(1000, 5000, n_individuals),
        'V_E': np.random.uniform(3000, 8000, n_individuals),
        'a_optimal': np.random.beta(3, 2, n_individuals)
    })
    
    convergence_history = pd.DataFrame({
        'iteration': range(1, 31),
        'diff_V': np.exp(-np.linspace(0, 5, 30)) * 0.5,
        'diff_a': np.exp(-np.linspace(0, 4, 30)) * 0.1,
        'diff_u': np.exp(-np.linspace(0, 3, 30)) * 0.01
    })
    
    # 创建可视化器
    visualizer = MFGVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 绘制收敛曲线
    conv_path = visualizer.plot_convergence_curves(convergence_history)
    print(f"✓ 收敛曲线: {conv_path}")
    
    # 绘制最优努力分布
    effort_path = visualizer.plot_optimal_effort_distribution(individuals)
    print(f"✓ 努力分布: {effort_path}")
    
    # 创建3D价值函数图
    value_3d_path = visualizer.create_interactive_value_function_3d(individuals, 'V_U')
    print(f"✓ 3D价值函数: {value_3d_path}")


def test_calibration_visualization():
    """测试校准可视化器"""
    print("\n=== 测试校准可视化器 ===")
    
    # 创建模拟数据
    n_evals = 100
    objective_history = pd.DataFrame({
        'evaluation': range(1, n_evals + 1),
        'smm_distance': np.exp(-np.linspace(0, 5, n_evals)) * 10 + np.random.rand(n_evals) * 0.5
    })
    
    parameter_history = pd.DataFrame({
        'evaluation': range(1, n_evals + 1),
        'rho': 0.75 - np.exp(-np.linspace(0, 3, n_evals)) * 0.35 + np.random.rand(n_evals) * 0.05,
        'kappa': 2000 + np.random.randn(n_evals) * 200,
        'smm_distance': objective_history['smm_distance'].values
    })
    
    # 创建可视化器
    visualizer = CalibrationVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 绘制目标函数历史
    obj_path = visualizer.plot_objective_history(objective_history)
    print(f"✓ 目标函数: {obj_path}")
    
    # 绘制参数轨迹
    param_path = visualizer.plot_parameter_traces(parameter_history, ['rho', 'kappa'])
    print(f"✓ 参数轨迹: {param_path}")
    
    # 绘制矩拟合
    target_moments = {'unemployment_rate': 0.15, 'mean_wage': 4500, 'std_wage': 1500}
    simulated_moments = {'unemployment_rate': 0.16, 'mean_wage': 4400, 'std_wage': 1550}
    moment_path = visualizer.plot_moment_fit(target_moments, simulated_moments)
    print(f"✓ 矩拟合: {moment_path}")


def test_simulation_visualization():
    """测试仿真可视化器"""
    print("\n=== 测试仿真可视化器 ===")
    
    # 创建模拟政策结果
    policy_results = {
        '基准': {'unemployment_rate': 0.15, 'mean_wage': 4500, 'mean_effort': 0.5},
        '技能培训': {'unemployment_rate': 0.12, 'mean_wage': 4800, 'mean_effort': 0.6},
        '数字素养': {'unemployment_rate': 0.13, 'mean_wage': 4700, 'mean_effort': 0.55}
    }
    
    # 创建可视化器
    visualizer = SimulationVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 绘制政策对比
    comp_path = visualizer.plot_policy_comparison(policy_results)
    print(f"✓ 政策对比: {comp_path}")
    
    # 创建交互式雷达图
    radar_path = visualizer.create_interactive_policy_radar(policy_results)
    print(f"✓ 雷达图: {radar_path}")


def test_dashboard():
    """测试仪表盘构建器"""
    print("\n=== 测试仪表盘构建器 ===")
    
    # 准备数据（复用前面的模拟数据）
    n_individuals = 2000
    individuals = pd.DataFrame({
        'T': np.random.normal(47, 7, n_individuals),
        'S': np.random.beta(2.5, 4, n_individuals),
        'D': np.random.beta(2.5, 4, n_individuals),
        'W': np.random.lognormal(8.4, 0.3, n_individuals),
        'V_U': np.random.uniform(1000, 5000, n_individuals),
        'V_E': np.random.uniform(3000, 8000, n_individuals),
        'a_optimal': np.random.beta(3, 2, n_individuals)
    })
    
    convergence_history = pd.DataFrame({
        'iteration': range(1, 21),
        'diff_V': np.exp(-np.linspace(0, 4, 20)) * 0.3,
        'diff_a': np.exp(-np.linspace(0, 3, 20)) * 0.08,
        'diff_u': np.exp(-np.linspace(0, 2, 20)) * 0.005
    })
    
    # 创建仪表盘构建器
    builder = DashboardBuilder(output_dir=project_root / 'OUTPUT')
    
    # 构建MFG仪表盘
    mfg_dashboard = builder.build_mfg_dashboard(individuals, convergence_history)
    print(f"✓ MFG仪表盘: {mfg_dashboard}")


if __name__ == "__main__":
    # 初始化样式
    print("初始化可视化样式...")
    initialize_styles()
    
    # 运行测试
    test_data_visualization()
    test_mfg_visualization()
    test_calibration_visualization()
    test_simulation_visualization()
    test_dashboard()
    
    print("\n=== 所有可视化测试完成！ ===")
    print(f"输出目录: {project_root / 'OUTPUT'}")

