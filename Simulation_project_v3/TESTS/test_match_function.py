#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
匹配函数回归模块测试

测试Logit回归拟合匹配函数λ(x,σ,θ)的完整流程。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 过滤字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.LOGISTIC import MatchFunction, load_config
import time


def test_match_function():
    """测试匹配函数回归"""
    print("=" * 80)
    print("匹配函数Logit回归测试")
    print("=" * 80)
    
    # 加载配置
    print("\n1. 加载配置...")
    config = load_config("CONFIG/logistic_config.yaml")
    print(f"   数据生成轮数: {config['data_generation']['n_rounds']}")
    print(f"   每轮劳动力数: {config['market_size']['n_laborers']}")
    
    # 初始化匹配函数
    print("\n2. 初始化匹配函数...")
    match_func = MatchFunction(config)
    print("   [完成]")
    
    # 生成训练数据
    print("\n3. 生成训练数据...")
    start_time = time.time()
    data = match_func.generate_training_data()
    data_gen_time = time.time() - start_time
    print(f"   数据生成耗时: {data_gen_time:.1f} 秒")
    print(f"\n   数据前5行:")
    print(data.head())
    
    # 拟合Logit回归
    print("\n4. 拟合Logit回归...")
    start_time = time.time()
    match_func.fit()
    fit_time = time.time() - start_time
    print(f"   [完成] 回归拟合耗时: {fit_time:.1f} 秒")
    
    # 显示回归结果
    print("\n5. 回归结果摘要:")
    print(f"   伪R²: {match_func.model.prsquared:.4f}")
    print(f"   AIC: {match_func.model.aic:.2f}")
    print(f"   BIC: {match_func.model.bic:.2f}")
    
    print("\n   回归参数:")
    print(match_func.params)
    
    # 保存结果
    print("\n6. 保存回归结果...")
    match_func.save_results()
    print("   保存路径: OUTPUT/logistic/match_function_params.pkl")
    print("   保存路径: OUTPUT/logistic/match_function_model.pkl")
    
    # 测试预测
    print("\n7. 测试预测功能...")
    test_X = data.iloc[:5][['T', 'S', 'D', 'W', 'sigma', 'theta']]
    pred_prob = match_func.predict(test_X)
    print(f"   前5个样本的预测匹配概率:")
    for i, prob in enumerate(pred_prob):
        actual = data.iloc[i]['matched']
        print(f"     样本{i}: 预测概率={prob:.4f}, 实际结果={actual}")
    
    # 分析所有样本的预测概率
    print("\n8. 分析预测概率分布...")
    all_X = data[['T', 'S', 'D', 'W', 'sigma', 'theta']]
    all_pred_prob = match_func.predict(all_X)
    
    # 筛选实际为1和0的样本
    matched_mask = data['matched'] == 1
    unmatched_mask = data['matched'] == 0
    matched_pred_prob = all_pred_prob[matched_mask]
    unmatched_pred_prob = all_pred_prob[unmatched_mask]
    
    if len(matched_pred_prob) > 0:
        print(f"   实际匹配样本（matched=1）的预测概率:")
        print(f"     样本数: {len(matched_pred_prob)}")
        print(f"     最小值: {matched_pred_prob.min():.6f}")
        print(f"     最大值: {matched_pred_prob.max():.6f}")
        print(f"     均值: {matched_pred_prob.mean():.6f}")
        print(f"     中位数: {np.median(matched_pred_prob):.6f}")
    else:
        print("   警告: 没有匹配样本（matched=1）")
    
    # 可视化预测概率分布
    print("\n9. 生成预测概率可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Logit回归预测概率分析', fontsize=16)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 所有样本的预测概率分布
    ax1 = axes[0, 0]
    ax1.hist(all_pred_prob, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax1.set_xlabel('预测匹配概率', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('所有样本的预测概率分布', fontsize=12)
    ax1.axvline(all_pred_prob.mean(), color='red', linestyle='--', 
                label=f'均值={all_pred_prob.mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按实际匹配结果分组的预测概率分布
    ax2 = axes[0, 1]
    ax2.hist(matched_pred_prob, bins=30, alpha=0.6, color='green', 
             edgecolor='black', label=f'实际匹配 (n={len(matched_pred_prob)})')
    ax2.hist(unmatched_pred_prob, bins=30, alpha=0.6, color='red', 
             edgecolor='black', label=f'实际未匹配 (n={len(unmatched_pred_prob)})')
    ax2.set_xlabel('预测匹配概率', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('预测概率分布（按实际结果分组）', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot对比
    ax3 = axes[1, 0]
    box_data = [matched_pred_prob, unmatched_pred_prob]
    bp = ax3.boxplot(box_data, tick_labels=['实际匹配', '实际未匹配'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax3.set_ylabel('预测匹配概率', fontsize=12)
    ax3.set_title('预测概率分布对比（箱线图）', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 预测概率统计汇总
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "预测概率统计汇总\n" + "="*40 + "\n\n"
    stats_text += f"总样本数: {len(all_pred_prob)}\n"
    stats_text += f"匹配样本数: {len(matched_pred_prob)} ({len(matched_pred_prob)/len(all_pred_prob)*100:.1f}%)\n"
    stats_text += f"未匹配样本数: {len(unmatched_pred_prob)} ({len(unmatched_pred_prob)/len(all_pred_prob)*100:.1f}%)\n\n"
    
    stats_text += "实际匹配样本 (matched=1):\n"
    stats_text += f"  均值: {matched_pred_prob.mean():.4f}\n"
    stats_text += f"  中位数: {np.median(matched_pred_prob):.4f}\n"
    stats_text += f"  最小值: {matched_pred_prob.min():.4f}\n"
    stats_text += f"  最大值: {matched_pred_prob.max():.4f}\n\n"
    
    stats_text += "实际未匹配样本 (matched=0):\n"
    stats_text += f"  均值: {unmatched_pred_prob.mean():.4f}\n"
    stats_text += f"  中位数: {np.median(unmatched_pred_prob):.4f}\n"
    stats_text += f"  最小值: {unmatched_pred_prob.min():.4f}\n"
    stats_text += f"  最大值: {unmatched_pred_prob.max():.4f}\n\n"
    
    stats_text += f"回归模型:\n"
    stats_text += f"  伪R²: {match_func.model.prsquared:.4f}\n"
    stats_text += f"  AIC: {match_func.model.aic:.2f}\n"
    stats_text += f"  BIC: {match_func.model.bic:.2f}"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图形
    output_dir = Path("OUTPUT/logistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "match_function_prediction_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   可视化图表已保存至: {output_file}")
    
    print("\n" + "=" * 80)
    print("[完成] 匹配函数回归测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_match_function()

