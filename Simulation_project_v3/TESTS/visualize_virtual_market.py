#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟市场三维可视化

展示劳动力和企业在特征空间中的分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from MODULES.LOGISTIC.virtual_market import VirtualMarket, load_config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_labor_market_3d(
    laborers: pd.DataFrame,
    enterprises: pd.DataFrame,
    features: tuple = ('T', 'S', 'W'),
    save_path: str = None
):
    """
    绘制劳动力市场三维散点图
    
    Args:
        laborers: 劳动力数据
        enterprises: 企业数据
        features: 三个特征维度，例如 ('T', 'S', 'W')
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 特征映射（劳动力特征 -> 企业特征）
    feature_map = {
        'T': 'T_req',
        'S': 'S_req',
        'D': 'D_req',
        'W': 'W_offer'
    }
    
    # 获取劳动力数据
    labor_x = laborers[features[0]].values
    labor_y = laborers[features[1]].values
    labor_z = laborers[features[2]].values
    
    # 获取企业数据
    ent_features = tuple(feature_map.get(f, f) for f in features)
    ent_x = enterprises[ent_features[0]].values
    ent_y = enterprises[ent_features[1]].values
    ent_z = enterprises[ent_features[2]].values
    
    # 绘制劳动力（蓝色）
    scatter1 = ax.scatter(
        labor_x, labor_y, labor_z,
        c='#3498db',
        marker='o',
        s=30,
        alpha=0.6,
        label='劳动力',
        edgecolors='w',
        linewidth=0.5
    )
    
    # 绘制企业（红色）
    scatter2 = ax.scatter(
        ent_x, ent_y, ent_z,
        c='#e74c3c',
        marker='^',
        s=40,
        alpha=0.6,
        label='企业岗位',
        edgecolors='w',
        linewidth=0.5
    )
    
    # 设置轴标签
    feature_names = {
        'T': '工作时长 (小时/周)',
        'T_req': '要求工作时长 (小时/周)',
        'S': '技能水平 (分)',
        'S_req': '要求技能水平 (分)',
        'D': '数字素养 (分)',
        'D_req': '要求数字素养 (分)',
        'W': '期望工资 (元/月)',
        'W_offer': '提供工资 (元/月)'
    }
    
    ax.set_xlabel(feature_names.get(features[0], features[0]), 
                  fontsize=16, labelpad=15, fontweight='bold')
    ax.set_ylabel(feature_names.get(features[1], features[1]), 
                  fontsize=16, labelpad=15, fontweight='bold')
    ax.set_zlabel(feature_names.get(features[2], features[2]), 
                  fontsize=16, labelpad=15, fontweight='bold')
    
    # 图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ 三维图已保存: {save_path}")
    
    plt.close()


def plot_multiple_views(
    laborers: pd.DataFrame,
    enterprises: pd.DataFrame,
    output_dir: Path
):
    """
    绘制多个视角的三维图
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. (T, S, W) - 核心三维
    print("\n生成视图1: 工作时长-技能-工资空间...")
    plot_labor_market_3d(
        laborers, enterprises,
        features=('T', 'S', 'W'),
        save_path=output_dir / 'market_3d_TSW.png'
    )
    
    # 2. (S, D, W) - 能力-数字素养-工资
    print("生成视图2: 技能-数字素养-工资空间...")
    plot_labor_market_3d(
        laborers, enterprises,
        features=('S', 'D', 'W'),
        save_path=output_dir / 'market_3d_SDW.png'
    )
    
    # 3. (T, D, W) - 工作时长-数字素养-工资
    print("生成视图3: 工作时长-数字素养-工资空间...")
    plot_labor_market_3d(
        laborers, enterprises,
        features=('T', 'D', 'W'),
        save_path=output_dir / 'market_3d_TDW.png'
    )
    
    # 4. (T, S, D) - 纯特征空间（不含工资）
    print("生成视图4: 工作时长-技能-数字素养空间...")
    plot_labor_market_3d(
        laborers, enterprises,
        features=('T', 'S', 'D'),
        save_path=output_dir / 'market_3d_TSD.png'
    )


def plot_market_statistics(
    laborers: pd.DataFrame,
    enterprises: pd.DataFrame,
    output_dir: Path
):
    """
    绘制市场统计对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 特征对应关系
    comparisons = [
        ('T', 'T_req', '工作时长 (小时/周)'),
        ('S', 'S_req', '技能水平 (分)'),
        ('D', 'D_req', '数字素养 (分)'),
        ('W', 'W_offer', '工资 (元/月)')
    ]
    
    for idx, (labor_col, ent_col, label) in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        
        # 绘制直方图
        ax.hist(laborers[labor_col], bins=30, alpha=0.6, 
                color='#3498db', label='劳动力', density=True)
        ax.hist(enterprises[ent_col], bins=30, alpha=0.6, 
                color='#e74c3c', label='企业岗位', density=True)
        
        # 添加均值线
        labor_mean = laborers[labor_col].mean()
        ent_mean = enterprises[ent_col].mean()
        ax.axvline(labor_mean, color='#2980b9', linestyle='--', 
                   linewidth=2, label=f'劳动力均值: {labor_mean:.1f}')
        ax.axvline(ent_mean, color='#c0392b', linestyle='--', 
                   linewidth=2, label=f'企业均值: {ent_mean:.1f}')
        
        ax.set_xlabel(label, fontsize=13, fontweight='bold')
        ax.set_ylabel('密度', fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'market_statistics.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✓ 统计对比图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("虚拟劳动力市场三维可视化")
    print("="*60)
    
    # 加载配置
    config = load_config("CONFIG/logistic_config.yaml")
    
    # 创建虚拟市场生成器
    print("\n初始化虚拟市场生成器...")
    market = VirtualMarket(config)
    
    # 生成市场数据
    print("\n生成虚拟市场数据...")
    n_laborers = 1000
    theta = 0.8  # 岗位富余市场
    
    laborers, enterprises = market.generate_market(n_laborers, theta)
    
    print(f"✓ 已生成 {len(laborers)} 个劳动力")
    print(f"✓ 已生成 {len(enterprises)} 个企业岗位")
    print(f"✓ 市场紧张度 θ = {theta}")
    
    # 输出目录
    output_dir = Path("OUTPUT/market_visualization")
    
    # 生成三维可视化
    print("\n" + "="*60)
    print("生成三维可视化...")
    print("="*60)
    plot_multiple_views(laborers, enterprises, output_dir)
    
    # 生成统计对比图
    print("\n生成统计对比图...")
    plot_market_statistics(laborers, enterprises, output_dir)
    
    # 输出市场基本统计
    print("\n" + "="*60)
    print("市场基本统计")
    print("="*60)
    
    print("\n【劳动力特征统计】")
    print(laborers[['T', 'S', 'D', 'W', 'age']].describe().round(2))
    
    print("\n【企业岗位特征统计】")
    print(enterprises[['T_req', 'S_req', 'D_req', 'W_offer']].describe().round(2))
    
    print("\n【特征匹配度分析】")
    print(f"工作时长匹配度: 劳动力均值={laborers['T'].mean():.1f}, "
          f"企业均值={enterprises['T_req'].mean():.1f}, "
          f"差距={abs(laborers['T'].mean() - enterprises['T_req'].mean()):.1f}")
    print(f"技能匹配度: 劳动力均值={laborers['S'].mean():.1f}, "
          f"企业均值={enterprises['S_req'].mean():.1f}, "
          f"差距={abs(laborers['S'].mean() - enterprises['S_req'].mean()):.1f}")
    print(f"数字素养匹配度: 劳动力均值={laborers['D'].mean():.1f}, "
          f"企业均值={enterprises['D_req'].mean():.1f}, "
          f"差距={abs(laborers['D'].mean() - enterprises['D_req'].mean()):.1f}")
    print(f"工资匹配度: 劳动力期望={laborers['W'].mean():.0f}, "
          f"企业提供={enterprises['W_offer'].mean():.0f}, "
          f"差距={abs(laborers['W'].mean() - enterprises['W_offer'].mean()):.0f}")
    
    print("\n" + "="*60)
    print("✓ 所有可视化完成！")
    print(f"✓ 输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

