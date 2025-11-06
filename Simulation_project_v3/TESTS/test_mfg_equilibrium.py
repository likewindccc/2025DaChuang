#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG均衡求解器测试脚本（完整规模版）

功能：
1. 加载配置和匹配函数模型
2. 初始化10000个体的人口
3. 求解平均场博弈均衡（最多200轮迭代）
4. 生成详细分析和可视化结果

可视化方案：
==============
共生成6张独立图表（每张14×8英寸，300 DPI，微软雅黑字体）

【图1】失业率演化 (convergence_unemployment_rate.png)
  └─ 蓝色曲线：从初始失业率到均衡失业率的完整演化路径

【图2】市场紧张度演化 (convergence_theta.png)
  └─ 红色曲线：岗位供给/求职人数的动态变化

【图3】平均努力水平演化 (convergence_effort.png)
  └─ 绿色曲线：失业者求职努力随时间的变化趋势

【图4】收敛指标监控 (convergence_metrics.png)
  ├─ 紫色曲线：|ΔV| 价值函数变化量
  ├─ 橙色曲线：|Δa| 努力水平变化量
  └─ 红色虚线：收敛阈值（epsilon_V）

【图5】努力水平分布直方图 (effort_distribution_histogram.png)
  └─ 展示均衡时失业者努力选择的分布特征

【图6】技能-努力关系散点图 (effort_vs_skill.png)
  └─ 揭示个体技能与求职努力的相关关系

输出文件：
  数据文件：
    - OUTPUT/mfg/equilibrium_individuals.csv     (个体状态)
    - OUTPUT/mfg/equilibrium_policy.csv          (价值函数和策略)
    - OUTPUT/mfg/equilibrium_history.csv         (迭代历史)
    - OUTPUT/mfg/equilibrium_summary.pkl         (汇总信息)
    - OUTPUT/mfg/value_distribution_full.csv     (完整价值函数分布)
  
  可视化文件：
    - OUTPUT/mfg/convergence_unemployment_rate.png  (失业率演化)
    - OUTPUT/mfg/convergence_theta.png              (市场紧张度演化)
    - OUTPUT/mfg/convergence_effort.png             (努力水平演化)
    - OUTPUT/mfg/convergence_metrics.png            (收敛指标监控)
    - OUTPUT/mfg/effort_distribution_histogram.png  (努力分布直方图)
    - OUTPUT/mfg/effort_vs_skill.png                (技能-努力散点图)
"""

import sys
import os
import numpy as np
import pandas as pd

# 【重要】在无GUI环境（如AutoDL服务器）下必须先设置后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.MFG import solve_equilibrium

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 全局字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体
plt.rcParams['xtick.labelsize'] = 14  # X轴刻度字体
plt.rcParams['ytick.labelsize'] = 14  # Y轴刻度字体
plt.rcParams['legend.fontsize'] = 14  # 图例字体


def test_mfg_equilibrium_small():
    """
    完整规模测试：10000个体，200轮迭代
    完整求解MFG均衡并生成详细可视化
    """
    print("=" * 80)
    print("MFG均衡求解器 - 完整规模测试")
    print("=" * 80)
    print("配置：10000个体，最多200轮迭代")
    print("预计运行时间：20-40分钟")
    print()
    
    # 加载配置并修改为完整规模
    import yaml
    config_path = "CONFIG/mfg_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改为完整规模测试
    config['population']['n_individuals'] = 10000
    config['equilibrium']['max_outer_iter'] = 200
    config['market']['vacancy'] = 15000  # θ=1.5时的岗位数
    
    # 保存临时配置
    temp_config_path = "CONFIG/mfg_config_test.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        # 求解均衡
        individuals_eq, eq_info = solve_equilibrium(temp_config_path)
        
        # 输出结果
        print("\n" + "=" * 80)
        print("均衡求解完成")
        print("=" * 80)
        print(f"是否收敛: {eq_info['converged']}")
        print(f"迭代轮数: {eq_info['iterations']}")
        print(f"最终失业率: {eq_info['final_unemployment_rate']*100:.2f}%")
        print(f"最终市场紧张度: {eq_info['final_theta']:.4f}")
        print()
        
        # 分析个体状态
        print("=" * 80)
        print("个体状态统计")
        print("=" * 80)
        unemployed = individuals_eq[individuals_eq['employment_status'] == 'unemployed']
        employed = individuals_eq[individuals_eq['employment_status'] == 'employed']
        
        print(f"\n失业者数量: {len(unemployed)}")
        print(f"就业者数量: {len(employed)}")
        print()
        
        print("失业者状态变量:")
        print(unemployed[['T', 'S', 'D', 'W']].describe())
        print()
        
        print("就业者状态变量:")
        print(employed[['T', 'S', 'D', 'W']].describe())
        print()
        
        # 读取策略
        policy = pd.read_csv("OUTPUT/mfg/equilibrium_policy.csv")
        unemployed_policy = policy[individuals_eq['employment_status'] == 'unemployed']
        
        print("=" * 80)
        print("均衡策略统计")
        print("=" * 80)
        print("\n失业者最优努力水平:")
        print(unemployed_policy['a_optimal'].describe())
        print()
        
        print("失业者价值函数:")
        print(unemployed_policy['V_U'].describe())
        print()
        
        # 就业溢价分析
        value_dist = pd.read_csv("OUTPUT/mfg/value_distribution_full.csv")
        print("=" * 80)
        print("就业溢价分析 (Delta_V = V_E - V_U)")
        print("=" * 80)
        print(f"\n平均就业溢价: {value_dist['delta_V'].mean():.2f}")
        print(f"就业溢价标准差: {value_dist['delta_V'].std():.2f}")
        print(f"就业溢价范围: [{value_dist['delta_V'].min():.2f}, {value_dist['delta_V'].max():.2f}]")
        print()
        
        # 状态变量变化分析
        print("=" * 80)
        print("状态变量演化分析")
        print("=" * 80)
        history = eq_info['history']
        if len(history['iteration']) > 1:
            print(f"\n平均T变化: {history['mean_T'][0]:.2f} → {history['mean_T'][-1]:.2f} " +
                  f"(+{history['mean_T'][-1] - history['mean_T'][0]:.2f})")
            print(f"平均S变化: {history['mean_S'][0]:.2f} → {history['mean_S'][-1]:.2f} " +
                  f"(+{history['mean_S'][-1] - history['mean_S'][0]:.2f})")
            print(f"平均D变化: {history['mean_D'][0]:.2f} → {history['mean_D'][-1]:.2f} " +
                  f"(+{history['mean_D'][-1] - history['mean_D'][0]:.2f})")
            print(f"失业率变化: {history['unemployment_rate'][0]*100:.2f}% → " +
                  f"{history['unemployment_rate'][-1]*100:.2f}% " +
                  f"({history['unemployment_rate'][-1]*100 - history['unemployment_rate'][0]*100:+.2f}pp)")
        print()
        
        # ============================================================
        # 可视化1：失业率演化
        # ============================================================
        print("\n生成可视化图表...")
        
        plt.figure(figsize=(14, 8))
        plt.plot(history['iteration'], 
                [u*100 for u in history['unemployment_rate']], 
                'b-', linewidth=3)
        plt.xlabel('迭代轮数', fontsize=18, fontweight='bold')
        plt.ylabel('失业率 (%)', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linewidth=1.5)
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/convergence_unemployment_rate.png', dpi=300, bbox_inches='tight')
        print("  ✓ 失业率演化图已保存")
        plt.close()
        
        # ============================================================
        # 可视化2：市场紧张度演化
        # ============================================================
        plt.figure(figsize=(14, 8))
        plt.plot(history['iteration'], history['theta'], 'r-', linewidth=3)
        plt.xlabel('迭代轮数', fontsize=18, fontweight='bold')
        plt.ylabel('市场紧张度 θ', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linewidth=1.5)
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/convergence_theta.png', dpi=300, bbox_inches='tight')
        print("  ✓ 市场紧张度演化图已保存")
        plt.close()
        
        # ============================================================
        # 可视化3：平均努力水平演化
        # ============================================================
        plt.figure(figsize=(14, 8))
        plt.plot(history['iteration'], history['mean_effort'], 'g-', linewidth=3)
        plt.xlabel('迭代轮数', fontsize=18, fontweight='bold')
        plt.ylabel('平均努力水平', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linewidth=1.5)
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/convergence_effort.png', dpi=300, bbox_inches='tight')
        print("  ✓ 平均努力水平演化图已保存")
        plt.close()
        
        # ============================================================
        # 可视化4：收敛指标监控
        # ============================================================
        valid_indices = [i for i, v in enumerate(history['convergence_V']) if not np.isnan(v)]
        if valid_indices:
            plt.figure(figsize=(14, 8))
            plt.semilogy(
                [history['iteration'][i] for i in valid_indices],
                [history['convergence_V'][i] for i in valid_indices],
                color='purple', linewidth=3, label='|ΔV| (价值函数变化)'
            )
            plt.semilogy(
                [history['iteration'][i] for i in valid_indices],
                [history['convergence_a'][i] for i in valid_indices],
                color='orange', linewidth=3, label='|Δa| (努力水平变化)'
            )
            plt.axhline(y=config['equilibrium']['convergence']['epsilon_V'], 
                       color='r', linestyle='--', linewidth=2, label='收敛阈值')
            plt.xlabel('迭代轮数', fontsize=18, fontweight='bold')
            plt.ylabel('收敛指标 (对数尺度)', fontsize=18, fontweight='bold')
            plt.legend(loc='best', fontsize=14)
            plt.grid(True, alpha=0.3, linewidth=1.5)
            plt.tight_layout()
            plt.savefig('OUTPUT/mfg/convergence_metrics.png', dpi=300, bbox_inches='tight')
            print("  ✓ 收敛指标监控图已保存")
            plt.close()
        
        # ============================================================
        # 可视化5：失业者最优努力水平分布
        # ============================================================
        plt.figure(figsize=(14, 8))
        plt.hist(unemployed_policy['a_optimal'], bins=20, 
                edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('最优努力水平', fontsize=18, fontweight='bold')
        plt.ylabel('频数', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linewidth=1.5, axis='y')
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/effort_distribution_histogram.png', dpi=300, bbox_inches='tight')
        print("  ✓ 努力水平分布直方图已保存")
        plt.close()
        
        # ============================================================
        # 可视化6：技能水平与最优努力的关系
        # ============================================================
        plt.figure(figsize=(14, 8))
        plt.scatter(unemployed['S'].values, unemployed_policy['a_optimal'].values, 
                   alpha=0.5, s=50, c='steelblue', edgecolors='navy', linewidth=0.5)
        plt.xlabel('技能水平 S', fontsize=18, fontweight='bold')
        plt.ylabel('最优努力水平', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linewidth=1.5)
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/effort_vs_skill.png', dpi=300, bbox_inches='tight')
        print("  ✓ 技能-努力关系散点图已保存")
        plt.close()
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        
        return individuals_eq, eq_info
        
    finally:
        # 删除临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_mfg_equilibrium_full():
    """
    完整规模测试：10000个体，100轮迭代
    完整求解MFG均衡
    """
    print("=" * 80)
    print("MFG均衡求解器 - 完整规模测试")
    print("=" * 80)
    print("警告：这可能需要几分钟到几十分钟的时间")
    print()
    
    # 使用默认配置
    individuals_eq, eq_info = solve_equilibrium()
    
    # 输出详细结果
    print("\n" + "=" * 80)
    print("均衡求解完成")
    print("=" * 80)
    print(f"是否收敛: {eq_info['converged']}")
    print(f"迭代轮数: {eq_info['iterations']}")
    print(f"最终失业率: {eq_info['final_unemployment_rate']*100:.2f}%")
    print(f"最终市场紧张度: {eq_info['final_theta']:.4f}")
    
    # 详细分析
    print("\n" + "=" * 80)
    print("详细统计分析")
    print("=" * 80)
    
    unemployed = individuals_eq[individuals_eq['employment_status'] == 'unemployed']
    employed = individuals_eq[individuals_eq['employment_status'] == 'employed']
    
    print(f"\n总人数: {len(individuals_eq)}")
    print(f"失业者: {len(unemployed)} ({len(unemployed)/len(individuals_eq)*100:.2f}%)")
    print(f"就业者: {len(employed)} ({len(employed)/len(individuals_eq)*100:.2f}%)")
    
    return individuals_eq, eq_info


if __name__ == '__main__':
    print("=" * 80)
    print("MFG均衡求解器 - 完整规模测试")
    print("=" * 80)
    print("\n配置信息:")
    print("  • 个体数量: 10,000")
    print("  • 最大迭代轮数: 200")
    print("  • 市场紧张度: θ = 1.5")
    print("  • 预计运行时间: 20-40分钟")
    print("\n输出内容:")
    print("  1. 均衡状态数据 (5个CSV文件)")
    print("  2. 可视化图表 (6张独立PNG，14×8尺寸，300 DPI)")
    print("     • 失业率演化")
    print("     • 市场紧张度演化")
    print("     • 努力水平演化")
    print("     • 收敛指标监控")
    print("     • 努力分布直方图")
    print("     • 技能-努力散点图")
    print("  3. 详细统计分析")
    print("\n按Enter键开始运行...")
    input()
    
    # 运行完整规模测试
    individuals, info = test_mfg_equilibrium_small()

