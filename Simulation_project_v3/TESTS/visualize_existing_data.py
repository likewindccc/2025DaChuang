"""
使用OUTPUT中的真实数据生成可视化

这个脚本会读取OUTPUT目录中已有的数据，生成各类可视化图表
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.VISUALIZATION import (
    initialize_styles,
    DataVisualizer,
    MFGVisualizer,
    SimulationVisualizer,
    DashboardBuilder
)


def load_mfg_data():
    """加载MFG模块数据"""
    output_dir = project_root / 'OUTPUT' / 'mfg'
    
    print("📂 加载MFG数据...")
    
    # 读取个体基本数据（T, S, D, W等）
    individuals_path = output_dir / 'equilibrium_individuals.csv'
    policy_path = output_dir / 'equilibrium_policy.csv'
    
    individuals = None
    if individuals_path.exists() and policy_path.exists():
        # 读取基本属性
        base_data = pd.read_csv(individuals_path)
        # 读取策略和价值函数
        policy_data = pd.read_csv(policy_path)
        
        # 合并数据（按索引）
        individuals = pd.concat([base_data, policy_data[['V_U', 'V_E', 'a_optimal']]], axis=1)
        print(f"  ✓ 个体数据: {len(individuals)} 条记录")
        print(f"    - 包含列: {individuals.columns.tolist()}")
    elif individuals_path.exists():
        individuals = pd.read_csv(individuals_path)
        print(f"  ⚠️  仅有基本数据（无V_U, V_E, a_optimal）")
    else:
        print(f"  ✗ 未找到个体数据")
    
    # 读取收敛历史
    history_path = output_dir / 'equilibrium_history.csv'
    if history_path.exists():
        convergence_history = pd.read_csv(history_path)
        print(f"  ✓ 收敛历史: {len(convergence_history)} 次迭代")
    else:
        print(f"  ✗ 未找到 {history_path}")
        convergence_history = None
    
    return individuals, convergence_history


def load_simulation_data():
    """加载Simulation模块数据"""
    output_dir = project_root / 'OUTPUT' / 'simulation'
    
    print("\n📂 加载Simulation数据...")
    
    # 读取场景对比数据
    comparison_path = output_dir / 'scenario_comparison.csv'
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        print(f"  ✓ 场景对比: {len(comparison)} 个场景")
    else:
        print(f"  ✗ 未找到 {comparison_path}")
        comparison = None
    
    # 读取各场景的时间序列数据
    scenarios = ['baseline', 'training_high', 'training_low']
    time_series_data = {}
    
    for scenario in scenarios:
        scenario_dir = output_dir / f'scenario_{scenario}'
        history_path = scenario_dir / 'equilibrium_history.csv'
        
        if history_path.exists():
            df = pd.read_csv(history_path)
            # 添加时间列（假设每次迭代代表一个时间步）
            df['time'] = df['iteration']
            time_series_data[scenario] = df
            print(f"  ✓ {scenario}: {len(df)} 个时间点")
    
    return comparison, time_series_data


def visualize_mfg_data(individuals, convergence_history):
    """可视化MFG数据"""
    if individuals is None or convergence_history is None:
        print("\n⚠️  MFG数据不完整，跳过可视化")
        return
    
    print("\n🎨 开始MFG可视化...")
    visualizer = MFGVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 适配列名（实际文件使用convergence_X而不是diff_X）
    if 'convergence_V' in convergence_history.columns:
        convergence_history = convergence_history.rename(columns={
            'convergence_V': 'diff_V',
            'convergence_a': 'diff_a',
            'convergence_u': 'diff_u'
        })
    
    # 1. 收敛曲线
    print("  📈 生成收敛曲线...")
    conv_path = visualizer.plot_convergence_curves(convergence_history)
    
    # 2. 最优努力分布
    if 'a_optimal' in individuals.columns:
        print("  📊 生成最优努力分布...")
        effort_path = visualizer.plot_optimal_effort_distribution(individuals)
    
    # 3. 价值函数热力图
    if all(col in individuals.columns for col in ['T', 'S', 'D', 'W', 'V_U']):
        print("  🔥 生成价值函数热力图...")
        value_path = visualizer.plot_value_function_heatmap(individuals, 'V_U')
    
    # 4. 3D交互式价值函数
    if all(col in individuals.columns for col in ['T', 'S', 'D', 'W', 'V_U']):
        print("  🌐 生成3D交互式价值函数...")
        value_3d_path = visualizer.create_interactive_value_function_3d(individuals, 'V_U')
    
    print("  ✅ MFG可视化完成！")


def visualize_simulation_data(comparison, time_series_data):
    """可视化Simulation数据"""
    if comparison is None:
        print("\n⚠️  Simulation数据不完整，跳过可视化")
        return
    
    print("\n🎨 开始Simulation可视化...")
    visualizer = SimulationVisualizer(output_dir=project_root / 'OUTPUT')
    
    # 1. 政策效果对比
    if 'scenario' in comparison.columns:
        print("  📊 生成政策对比...")
        
        # 转换为需要的格式
        policy_results = {}
        for _, row in comparison.iterrows():
            scenario_name = row['scenario']
            policy_results[scenario_name] = {
                'unemployment_rate': row.get('unemployment_rate', 0),
                'mean_wage': row.get('mean_wage', 0),
                'mean_T': row.get('mean_T', 0)
            }
        
        if policy_results:
            comp_path = visualizer.plot_policy_comparison(policy_results)
            radar_path = visualizer.create_interactive_policy_radar(policy_results)
    
    # 2. 时间序列演化
    if time_series_data:
        print("  📈 生成时间序列...")
        
        # 假设每个场景的history有unemployment_rate列
        # 如果没有，可以从其他指标计算
        for scenario, df in time_series_data.items():
            if 'unemployment_rate' not in df.columns:
                # 可以从其他数据计算，这里先跳过
                continue
        
        # 暂时不生成时间序列（需要确认数据格式）
        print("  ℹ️  时间序列需要确认数据格式")
    
    print("  ✅ Simulation可视化完成！")


def create_dashboard(individuals, convergence_history):
    """创建整合仪表盘"""
    if individuals is None or convergence_history is None:
        print("\n⚠️  数据不完整，跳过仪表盘创建")
        return
    
    print("\n🎨 开始创建仪表盘...")
    builder = DashboardBuilder(output_dir=project_root / 'OUTPUT')
    
    # 适配列名
    if 'convergence_V' in convergence_history.columns:
        convergence_history = convergence_history.rename(columns={
            'convergence_V': 'diff_V',
            'convergence_a': 'diff_a',
            'convergence_u': 'diff_u'
        })
    
    # MFG仪表盘
    print("  📊 创建MFG仪表盘...")
    mfg_dashboard = builder.build_mfg_dashboard(individuals, convergence_history)
    
    print("  ✅ 仪表盘创建完成！")
    print(f"  🌐 可以在浏览器中打开: {mfg_dashboard}")


def main():
    """主函数"""
    print("=" * 60)
    print("  使用真实数据生成可视化")
    print("=" * 60)
    
    # 初始化样式
    print("\n🎨 初始化可视化样式...")
    initialize_styles()
    
    # 加载数据
    individuals, convergence_history = load_mfg_data()
    comparison, time_series_data = load_simulation_data()
    
    # 生成可视化
    visualize_mfg_data(individuals, convergence_history)
    visualize_simulation_data(comparison, time_series_data)
    
    # 创建仪表盘
    create_dashboard(individuals, convergence_history)
    
    print("\n" + "=" * 60)
    print("  ✅ 所有可视化完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {project_root / 'OUTPUT'}")
    print("\n📊 生成的内容：")
    print("  - OUTPUT/figures/mfg/       (静态图，PNG格式)")
    print("  - OUTPUT/interactive/mfg/   (交互式图表，HTML格式)")
    print("  - OUTPUT/dashboards/        (整合仪表盘，HTML格式)")
    print("\n💡 提示：")
    print("  1. HTML文件可以直接在浏览器中打开")
    print("  2. PNG文件适合插入论文/报告")
    print("  3. 可以将HTML文件复制到WEBSITE/charts/目录中嵌入网站")


if __name__ == "__main__":
    main()

