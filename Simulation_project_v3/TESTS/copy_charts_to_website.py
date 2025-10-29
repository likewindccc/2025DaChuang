#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复制可视化图表到WEBSITE文件夹

将OUTPUT中生成的图表按模块分类复制到WEBSITE，供网页引用
"""

import shutil
from pathlib import Path


def main():
    """主函数"""
    print("=" * 80)
    print("复制可视化图表到WEBSITE")
    print("=" * 80)
    
    # 定义源目录和目标目录
    output_dir = Path('OUTPUT')
    website_dir = Path('WEBSITE')
    
    # 创建WEBSITE/charts目录结构
    charts_dir = website_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    for module in ['data', 'population', 'logistic', 'mfg', 'calibration', 'simulation']:
        (charts_dir / module / 'static').mkdir(parents=True, exist_ok=True)
        (charts_dir / module / 'interactive').mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # ========== 1. POPULATION/DATA模块 ==========
    print("\n【1/5】复制POPULATION模块图表...")
    
    # 静态图
    copy_if_exists(
        output_dir / 'figures/data/DATA_initial_distribution.png',
        charts_dir / 'population/static/initial_distribution.png'
    )
    copy_if_exists(
        output_dir / 'figures/data/DATA_copula_structure.png',
        charts_dir / 'population/static/copula_structure.png'
    )
    total_copied += 2
    
    # 交互式图
    copy_if_exists(
        output_dir / 'interactive/data/DATA_initial_distribution.html',
        charts_dir / 'population/interactive/initial_distribution.html'
    )
    total_copied += 1
    
    print(f"  ✅ 已复制 3 张图表")
    
    # ========== 2. LOGISTIC模块 ==========
    print("\n【2/5】复制LOGISTIC模块图表...")
    
    # 静态图
    copy_if_exists(
        output_dir / 'logistic/distribution_visualization.png',
        charts_dir / 'logistic/static/distribution_visualization.png'
    )
    copy_if_exists(
        output_dir / 'logistic/match_function_prediction_analysis.png',
        charts_dir / 'logistic/static/prediction_analysis.png'
    )
    copy_if_exists(
        output_dir / 'logistic/preference_score_distribution.png',
        charts_dir / 'logistic/static/preference_distribution.png'
    )
    total_copied += 3
    
    print(f"  ✅ 已复制 3 张图表")
    
    # ========== 3. MFG模块 ==========
    print("\n【3/5】复制MFG模块图表...")
    
    # 静态图（核心图表）
    mfg_static_charts = [
        ('MFG_convergence_curves.png', 'convergence_curves.png'),
        ('MFG_value_function_V_U_heatmap.png', 'value_function_V_U.png'),
        ('MFG_value_function_V_E_heatmap.png', 'value_function_V_E.png'),
        ('MFG_value_function_delta_V_heatmap.png', 'value_function_delta.png'),
        ('MFG_optimal_effort_distribution.png', 'effort_distribution.png'),
    ]
    
    for src_name, dst_name in mfg_static_charts:
        copy_if_exists(
            output_dir / f'figures/mfg/{src_name}',
            charts_dir / f'mfg/static/{dst_name}'
        )
    total_copied += len(mfg_static_charts)
    
    # 交互式图
    copy_if_exists(
        output_dir / 'interactive/mfg/MFG_value_function_V_U_3D.html',
        charts_dir / 'mfg/interactive/value_function_V_U_3D.html'
    )
    copy_if_exists(
        output_dir / 'interactive/mfg/MFG_value_function_V_E_3D.html',
        charts_dir / 'mfg/interactive/value_function_V_E_3D.html'
    )
    total_copied += 2
    
    # 额外的分析图（来自测试）
    copy_if_exists(
        output_dir / 'mfg/market_distribution_comparison.png',
        charts_dir / 'mfg/static/market_distribution_comparison.png'
    )
    copy_if_exists(
        output_dir / 'mfg/separation_rate_distribution_standardized.png',
        charts_dir / 'mfg/static/separation_rate_distribution.png'
    )
    total_copied += 2
    
    print(f"  ✅ 已复制 9 张图表")
    
    # ========== 4. CALIBRATION模块 ==========
    print("\n【4/5】CALIBRATION模块...")
    print("  ⏳ 校准尚未完成，暂无图表")
    
    # ========== 5. SIMULATION模块 ==========
    print("\n【5/5】SIMULATION模块...")
    print("  ⏳ 可视化图表待补充")
    
    print("\n" + "=" * 80)
    print(f"✅ 图表复制完成！共复制 {total_copied} 张图表")
    print("=" * 80)
    print(f"\n目标目录：WEBSITE/charts/")
    print("\n目录结构：")
    print("  WEBSITE/charts/")
    print("    ├── population/")
    print("    │   ├── static/       (PNG图片)")
    print("    │   └── interactive/  (HTML交互图)")
    print("    ├── logistic/")
    print("    ├── mfg/")
    print("    ├── calibration/")
    print("    └── simulation/")
    print()


def copy_if_exists(src: Path, dst: Path) -> bool:
    """
    如果源文件存在则复制
    
    参数:
        src: 源文件路径
        dst: 目标文件路径
    
    返回:
        是否成功复制
    """
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"    ✓ {src.name} → {dst}")
        return True
    else:
        print(f"    ✗ {src.name} (不存在)")
        return False


if __name__ == '__main__':
    main()

