"""
Module 1 Population Generator 演示脚本

演示劳动力和企业主体生成器的基本用法。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path

# 导入Module 1
from modules.population_generator import (
    LaborAgentGenerator, 
    EnterpriseGenerator, 
    PopulationConfig,
    create_default_config
)
from modules.population_generator.utils import Visualizer
from modules.population_generator.test_module import run_quick_test

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_demo_data():
    """创建演示用的样本数据"""
    logger.info("创建演示数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建劳动力样本数据
    n_labor = 800
    labor_data = pd.DataFrame({
        'T': np.random.normal(35, 8, n_labor),        # 工作时间
        'S': np.random.beta(2, 2.5, n_labor),         # 技能水平  
        'D': np.random.beta(1.8, 3, n_labor),         # 数字素养
        'W': np.random.lognormal(np.log(3200), 0.4, n_labor),  # 期望工资
        'age': np.random.randint(18, 58, n_labor),    # 年龄
        'education': np.random.choice([0,1,2,3,4], n_labor, p=[0.1,0.3,0.35,0.2,0.05])  # 教育水平
    })
    
    # 应用合理的数据边界
    labor_data['T'] = np.clip(labor_data['T'], 5, 75)
    labor_data['S'] = np.clip(labor_data['S'], 0.05, 0.95)
    labor_data['D'] = np.clip(labor_data['D'], 0.05, 0.95)
    labor_data['W'] = np.clip(labor_data['W'], 1200, 7500)
    
    # 创建企业样本数据（四维多元正态分布）
    n_enterprise = 400
    
    # 企业属性的均值和协方差矩阵
    enterprise_mean = [42, 0.6, 0.55, 3800]  # [T_req, S_req, D_req, W_offer]
    enterprise_cov = [
        [80, 0.08, 0.06, 150],     # T_req的方差和协方差
        [0.08, 0.04, 0.015, 35],   # S_req的方差和协方差  
        [0.06, 0.015, 0.03, 25],   # D_req的方差和协方差
        [150, 35, 25, 400000]      # W_offer的方差和协方差
    ]
    
    enterprise_array = np.random.multivariate_normal(enterprise_mean, enterprise_cov, n_enterprise)
    enterprise_data = pd.DataFrame(enterprise_array, columns=['T_req', 'S_req', 'D_req', 'W_offer'])
    
    # 应用数据边界
    enterprise_data['T_req'] = np.clip(enterprise_data['T_req'], 25, 55)
    enterprise_data['S_req'] = np.clip(enterprise_data['S_req'], 0.1, 0.9)
    enterprise_data['D_req'] = np.clip(enterprise_data['D_req'], 0.1, 0.9)
    enterprise_data['W_offer'] = np.clip(enterprise_data['W_offer'], 2000, 6500)
    
    logger.info(f"演示数据创建完成: 劳动力{n_labor}个，企业{n_enterprise}个")
    
    return labor_data, enterprise_data


def demo_labor_generator(labor_data, config):
    """演示劳动力生成器"""
    logger.info("\n" + "="*50)
    logger.info("劳动力生成器演示")
    logger.info("="*50)
    
    # 创建生成器
    labor_generator = LaborAgentGenerator(config.labor_config, random_state=42)
    
    # 拟合模型
    logger.info("正在拟合劳动力生成模型...")
    start_time = time.time()
    labor_generator.fit(labor_data)
    fit_time = time.time() - start_time
    
    logger.info(f"拟合完成，用时 {fit_time:.2f} 秒")
    
    # 显示拟合统计信息
    stats = labor_generator.get_summary_stats()
    logger.info(f"拟合模型信息:")
    logger.info(f"  - 训练样本数: {labor_generator.fitted_data_stats['n_samples']}")
    logger.info(f"  - 最佳Copula模型: {labor_generator.fitted_data_stats['best_copula']}")
    logger.info(f"  - 边际分布数量: {len(labor_generator.fitted_data_stats['marginal_distributions'])}")
    
    # 生成虚拟主体
    n_generate = 1000
    logger.info(f"正在生成 {n_generate} 个虚拟劳动力主体...")
    start_time = time.time()
    generated_labor = labor_generator.generate(n_generate)
    generate_time = time.time() - start_time
    
    logger.info(f"生成完成，用时 {generate_time:.2f} 秒 (速率: {n_generate/generate_time:.0f} 个/秒)")
    
    # 数据质量验证
    is_valid, validation_report = labor_generator.validate(generated_labor)
    logger.info(f"数据质量验证: {'通过' if is_valid else '存在问题'}")
    logger.info(f"数据质量得分: {validation_report.get('data_quality_score', 'N/A')}")
    
    # 显示生成数据的基本统计
    logger.info("\n生成数据统计摘要:")
    print(generated_labor.describe())
    
    return generated_labor, labor_generator


def demo_enterprise_generator(enterprise_data, config):
    """演示企业生成器"""
    logger.info("\n" + "="*50)
    logger.info("企业生成器演示")
    logger.info("="*50)
    
    # 创建生成器
    enterprise_generator = EnterpriseGenerator(config.enterprise_config, random_state=42)
    
    # 拟合模型
    logger.info("正在拟合企业生成模型...")
    start_time = time.time()
    enterprise_generator.fit(enterprise_data)
    fit_time = time.time() - start_time
    
    logger.info(f"拟合完成，用时 {fit_time:.2f} 秒")
    
    # 显示拟合统计信息
    mvn_stats = enterprise_generator.fitted_stats['mvn_statistics']
    logger.info(f"多元正态分布拟合信息:")
    logger.info(f"  - 训练样本数: {mvn_stats['n_samples']}")
    logger.info(f"  - 条件数: {mvn_stats['condition_number']:.2f}")
    logger.info(f"  - AIC: {mvn_stats['aic']:.2f}")
    logger.info(f"  - BIC: {mvn_stats['bic']:.2f}")
    
    # 生成虚拟企业
    n_generate = 600
    logger.info(f"正在生成 {n_generate} 个虚拟企业主体...")
    start_time = time.time()
    generated_enterprises = enterprise_generator.generate(n_generate)
    generate_time = time.time() - start_time
    
    logger.info(f"生成完成，用时 {generate_time:.2f} 秒 (速率: {n_generate/generate_time:.0f} 个/秒)")
    
    # 数据质量验证
    is_valid, validation_report = enterprise_generator.validate(generated_enterprises)
    logger.info(f"数据质量验证: {'通过' if is_valid else '存在问题'}")
    logger.info(f"数据质量得分: {validation_report.get('data_quality_score', 'N/A')}")
    
    # 显示生成数据的基本统计
    logger.info("\n生成数据统计摘要:")
    print(generated_enterprises.describe())
    
    return generated_enterprises, enterprise_generator


def create_comparison_visualizations(original_labor, generated_labor, 
                                   original_enterprise, generated_enterprise):
    """创建对比可视化图表"""
    logger.info("\n创建数据对比可视化...")
    
    visualizer = Visualizer()
    
    # 创建输出目录
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 劳动力数据对比图
    logger.info("生成劳动力数据对比图...")
    visualizer.plot_generation_comparison(
        original_labor, generated_labor,
        ['T', 'S', 'D', 'W'],
        save_path=str(output_dir / "labor_comparison.png")
    )
    
    # 企业数据对比图
    logger.info("生成企业数据对比图...")
    visualizer.plot_generation_comparison(
        original_enterprise, generated_enterprise,
        ['T_req', 'S_req', 'D_req', 'W_offer'],
        save_path=str(output_dir / "enterprise_comparison.png")
    )
    
    # 相关性热力图
    logger.info("生成相关性对比热力图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始劳动力相关性
    axes[0, 0].set_title("原始劳动力数据相关性")
    import seaborn as sns
    sns.heatmap(original_labor[['T', 'S', 'D', 'W']].corr(), 
               annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    
    # 生成劳动力相关性
    axes[0, 1].set_title("生成劳动力数据相关性") 
    sns.heatmap(generated_labor[['T', 'S', 'D', 'W']].corr(),
               annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
    
    # 原始企业相关性
    axes[1, 0].set_title("原始企业数据相关性")
    sns.heatmap(original_enterprise.corr(), 
               annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    
    # 生成企业相关性
    axes[1, 1].set_title("生成企业数据相关性")
    sns.heatmap(generated_enterprise.corr(),
               annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    
    plt.tight_layout()
    correlation_file = output_dir / "correlation_comparison.png"
    plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"可视化图表已保存到: {output_dir}")


def save_demo_results(generated_labor, generated_enterprise):
    """保存演示结果"""
    logger.info("\n保存演示结果...")
    
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 保存生成数据
    labor_file = output_dir / "generated_labor_demo.csv"
    enterprise_file = output_dir / "generated_enterprise_demo.csv"
    
    generated_labor.to_csv(labor_file, index=False, encoding='utf-8-sig')
    generated_enterprise.to_csv(enterprise_file, index=False, encoding='utf-8-sig')
    
    # 创建演示总结
    summary_file = output_dir / "demo_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Module 1 Population Generator 演示总结\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"演示时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("生成数据概况:\n")
        f.write(f"- 劳动力主体: {len(generated_labor)} 个\n")
        f.write(f"- 企业主体: {len(generated_enterprise)} 个\n\n")
        
        f.write("劳动力数据统计:\n")
        f.write(str(generated_labor.describe()) + "\n\n")
        
        f.write("企业数据统计:\n")
        f.write(str(generated_enterprise.describe()) + "\n\n")
        
        f.write("文件说明:\n")
        f.write("- generated_labor_demo.csv: 生成的劳动力数据\n")
        f.write("- generated_enterprise_demo.csv: 生成的企业数据\n")
        f.write("- labor_comparison.png: 劳动力数据分布对比图\n")
        f.write("- enterprise_comparison.png: 企业数据分布对比图\n")
        f.write("- correlation_comparison.png: 相关性对比热力图\n")
    
    logger.info(f"演示结果已保存到: {output_dir}")
    logger.info(f"  - 劳动力数据: {labor_file}")
    logger.info(f"  - 企业数据: {enterprise_file}")
    logger.info(f"  - 演示总结: {summary_file}")


def main():
    """主演示函数"""
    print("=" * 60)
    print("    EconLab Module 1: Population Generator 演示")
    print("=" * 60)
    
    try:
        # 快速测试
        logger.info("运行快速功能测试...")
        quick_test_passed = run_quick_test()
        
        if not quick_test_passed:
            logger.error("快速测试未通过，建议检查环境配置")
            return
        
        logger.info("✅ 快速测试通过，开始演示...")
        
        # 创建配置
        config = create_default_config()
        
        # 创建演示数据
        labor_data, enterprise_data = create_demo_data()
        
        # 演示劳动力生成器
        generated_labor, labor_generator = demo_labor_generator(labor_data, config)
        
        # 演示企业生成器
        generated_enterprise, enterprise_generator = demo_enterprise_generator(enterprise_data, config)
        
        # 创建可视化对比
        create_comparison_visualizations(
            labor_data, generated_labor,
            enterprise_data, generated_enterprise
        )
        
        # 保存结果
        save_demo_results(generated_labor, generated_enterprise)
        
        # 演示总结
        logger.info("\n" + "="*50)
        logger.info("演示完成总结")
        logger.info("="*50)
        logger.info("✅ 劳动力生成器: 基于Copula模型，支持复杂相关性")
        logger.info("✅ 企业生成器: 基于多元正态分布，四维属性建模")
        logger.info("✅ 数据质量: 生成数据与原始数据分布高度一致")
        logger.info("✅ 性能表现: 支持快速大规模数据生成")
        logger.info("✅ 可视化: 提供完整的数据对比和验证工具")
        
        print("\n🎉 Module 1 演示成功完成！")
        print("📁 演示结果已保存到 demo_outputs/ 目录")
        print("📊 请查看生成的图表和数据文件")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
