#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFG均衡求解器测试脚本

测试完整的MFG模块：
1. 加载配置和匹配函数模型
2. 初始化人口
3. 求解平均场博弈均衡
4. 分析和可视化结果
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.MFG import solve_equilibrium

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def test_mfg_equilibrium_small():
    """
    小规模测试：1000个体，20轮迭代
    快速验证功能是否正常
    """
    print("=" * 80)
    print("MFG均衡求解器 - 小规模测试")
    print("=" * 80)
    print()
    
    # 加载配置并修改为小规模
    import yaml
    config_path = "CONFIG/mfg_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改为小规模测试
    config['population']['n_individuals'] = 1000
    config['equilibrium']['max_outer_iter'] = 20
    config['market']['vacancy'] = 1000
    
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
        
        # 可视化收敛过程
        history = eq_info['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 失业率演化
        axes[0, 0].plot(history['iteration'], 
                       [u*100 for u in history['unemployment_rate']], 
                       'b-', linewidth=2)
        axes[0, 0].set_xlabel('迭代轮数')
        axes[0, 0].set_ylabel('失业率 (%)')
        axes[0, 0].set_title('失业率演化')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 市场紧张度演化
        axes[0, 1].plot(history['iteration'], history['theta'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('迭代轮数')
        axes[0, 1].set_ylabel('市场紧张度 θ')
        axes[0, 1].set_title('市场紧张度演化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 平均努力水平演化
        axes[1, 0].plot(history['iteration'], history['mean_effort'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('迭代轮数')
        axes[1, 0].set_ylabel('平均努力水平')
        axes[1, 0].set_title('平均努力水平演化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 收敛指标
        valid_indices = [i for i, v in enumerate(history['convergence_V']) if not np.isnan(v)]
        if valid_indices:
            axes[1, 1].semilogy(
                [history['iteration'][i] for i in valid_indices],
                [history['convergence_V'][i] for i in valid_indices],
                'purple', linewidth=2, label='|ΔV|'
            )
            axes[1, 1].semilogy(
                [history['iteration'][i] for i in valid_indices],
                [history['convergence_a'][i] for i in valid_indices],
                'orange', linewidth=2, label='|Δa|'
            )
            axes[1, 1].axhline(y=config['equilibrium']['convergence']['epsilon_V'], 
                              color='r', linestyle='--', label='收敛阈值')
            axes[1, 1].set_xlabel('迭代轮数')
            axes[1, 1].set_ylabel('收敛指标 (对数尺度)')
            axes[1, 1].set_title('收敛过程')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/test_convergence.png', dpi=150, bbox_inches='tight')
        print(f"收敛过程图已保存至: OUTPUT/mfg/test_convergence.png")
        plt.close()
        
        # 努力水平分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        axes[0].hist(unemployed_policy['a_optimal'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('最优努力水平')
        axes[0].set_ylabel('频数')
        axes[0].set_title('失业者最优努力水平分布')
        axes[0].grid(True, alpha=0.3)
        
        # 与状态变量的关系（以技能S为例）
        axes[1].scatter(unemployed['S'], unemployed_policy['a_optimal'], 
                       alpha=0.5, s=20)
        axes[1].set_xlabel('技能水平 S')
        axes[1].set_ylabel('最优努力水平')
        axes[1].set_title('技能水平与最优努力的关系')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUTPUT/mfg/test_effort_distribution.png', dpi=150, bbox_inches='tight')
        print(f"努力分布图已保存至: OUTPUT/mfg/test_effort_distribution.png")
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
    print("MFG均衡求解器测试\n")
    
    # 默认运行小规模测试
    # 如需完整测试，将下面改为 test_mfg_equilibrium_full()
    individuals, info = test_mfg_equilibrium_small()

