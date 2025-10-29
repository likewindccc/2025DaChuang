#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成所有可视化图表

功能：
- 使用VISUALIZATION模块生成所有模块的可视化图表
- 基于OUTPUT中现有的数据文件
- 只调用已实现的可视化方法
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from MODULES.VISUALIZATION import (
    DataVisualizer,
    MFGVisualizer,
    SimulationVisualizer,
    initialize_styles
)


def main():
    """主函数"""
    print("=" * 80)
    print("批量生成可视化图表")
    print("=" * 80)
    
    initialize_styles()
    
    total_charts = 0
    
    # 1. POPULATION模块可视化
    print("\n【1/4】生成POPULATION模块可视化...")
    total_charts += generate_population_visualizations()
    
    # 2. LOGISTIC模块可视化
    print("\n【2/4】LOGISTIC模块...")
    print("  ✅ 使用已有图表（test_match_function.py已生成）")
    
    # 3. MFG模块可视化
    print("\n【3/4】生成MFG模块可视化...")
    total_charts += generate_mfg_visualizations()
    
    # 4. SIMULATION模块可视化
    print("\n【4/4】生成SIMULATION模块可视化...")
    total_charts += generate_simulation_visualizations()
    
    print("\n" + "=" * 80)
    print(f"✅ 批量可视化完成！共生成 {total_charts} 张图表")
    print("=" * 80)
    print("\n输出目录：")
    print("  - 静态图：OUTPUT/figures/")
    print("  - 交互式图：OUTPUT/interactive/")
    print()


def generate_population_visualizations() -> int:
    """生成POPULATION模块可视化"""
    count = 0
    visualizer = DataVisualizer(output_dir='OUTPUT')
    
    try:
        # 加载分布参数并重新采样
        with open('OUTPUT/population/labor_distribution_params.pkl', 'rb') as f:
            labor_params = pickle.load(f)
        
        copula_model = labor_params['copula_model']
        samples = copula_model.sample(10000)
        
        discrete_dist = labor_params['discrete_dist']
        edu_values = list(discrete_dist['edu'].keys())
        edu_probs = list(discrete_dist['edu'].values())
        children_values = list(discrete_dist['children'].keys())
        children_probs = list(discrete_dist['children'].values())
        
        individuals = pd.DataFrame(samples, columns=['T', 'S', 'D', 'W', 'age'])
        individuals['edu'] = np.random.choice(edu_values, size=10000, p=edu_probs)
        individuals['children'] = np.random.choice(children_values, size=10000, p=children_probs)
        
        # 1. 初始劳动力分布
        print("  生成初始劳动力分布图...")
        static_path, interactive_path = visualizer.plot_initial_distribution(individuals)
        count += 2
        
        # 2. Copula相关性结构
        print("  生成Copula相关性结构图...")
        copula_path = visualizer.plot_copula_structure(individuals)
        count += 1
        
        print(f"  ✅ POPULATION: 生成{count}张图表")
        
    except Exception as e:
        print(f"  ⚠️ POPULATION可视化出错: {e}")
        import traceback
        traceback.print_exc()
    
    return count


def generate_mfg_visualizations() -> int:
    """生成MFG模块可视化"""
    count = 0
    visualizer = MFGVisualizer(output_dir='OUTPUT')
    
    try:
        # 加载数据
        individuals = pd.read_csv('OUTPUT/mfg/equilibrium_individuals.csv')
        history = pd.read_csv('OUTPUT/mfg/equilibrium_history.csv')
        value_dist = pd.read_csv('OUTPUT/mfg/value_distribution_full.csv')
        
        # 重命名列以匹配可视化器期望的列名
        history_renamed = history.rename(columns={
            'convergence_V': 'diff_V',
            'convergence_a': 'diff_a',
            'convergence_u': 'diff_u'
        })
        
        # 1. 收敛曲线
        print("  生成MFG收敛曲线...")
        conv_path = visualizer.plot_convergence_curves(history_renamed)
        count += 1
        
        # 2. 价值函数热力图（V_U, V_E, delta_V）
        print("  生成价值函数热力图（V_U）...")
        vu_path = visualizer.plot_value_function_heatmap(value_dist, 'V_U')
        count += 1
        
        print("  生成价值函数热力图（V_E）...")
        ve_path = visualizer.plot_value_function_heatmap(value_dist, 'V_E')
        count += 1
        
        print("  生成价值函数热力图（delta_V）...")
        delta_path = visualizer.plot_value_function_heatmap(value_dist, 'delta_V')
        count += 1
        
        # 3. 最优努力分布
        print("  生成最优努力分布图...")
        effort_path = visualizer.plot_optimal_effort_distribution(value_dist)
        count += 1
        
        # 4. 3D交互式价值函数
        print("  生成3D交互式价值函数（V_U）...")
        v3d_vu_path = visualizer.create_interactive_value_function_3d(value_dist, 'V_U')
        count += 1
        
        print("  生成3D交互式价值函数（V_E）...")
        v3d_ve_path = visualizer.create_interactive_value_function_3d(value_dist, 'V_E')
        count += 1
        
        print(f"  ✅ MFG: 生成{count}张图表")
        
    except Exception as e:
        print(f"  ⚠️ MFG可视化出错: {e}")
        import traceback
        traceback.print_exc()
    
    return count


def generate_simulation_visualizations() -> int:
    """生成SIMULATION模块可视化"""
    count = 0
    
    try:
        # SIMULATION模块的可视化方法需要特定的数据格式
        # 暂时跳过，后续根据需要补充
        print("  ✅ SIMULATION: 使用已有数据，可手动调用可视化方法")
        
    except Exception as e:
        print(f"  ⚠️ SIMULATION可视化出错: {e}")
    
    return count


if __name__ == '__main__':
    main()
