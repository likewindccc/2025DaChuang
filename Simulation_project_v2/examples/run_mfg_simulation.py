"""
完整的MFG模拟示例

演示如何使用整个MFG系统求解农村女性就业市场的均衡状态。

运行方式:
    python examples/run_mfg_simulation.py

Author: AI Assistant
Date: 2025-10-03
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import yaml
import logging
from datetime import datetime

# 导入MFG模块
from modules.mfg.mfg_simulator import MFGSimulator
from core.data_structures import MFGEquilibriumSparseGrid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/logs/mfg_simulation.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_test_config() -> dict:
    """
    创建测试配置（小规模，快速演示）
    
    注意：这是一个演示配置，实际运行时应使用config/default/mfg.yaml
    """
    config = {
        'state_space': {
            'dimension': 4,
            'ranges': {
                'T': [15, 70],
                'S': [2, 44],
                'D': [0, 20],
                'W': [1400, 8000]
            }
        },
        'sparse_grid': {
            'library': 'chaospy',
            'level': 3,  # 小规模测试：level=3约200个点
            'dimension': 4
        },
        'state_transition': {
            'gamma_T': 0.1,
            'gamma_S': 0.05,
            'gamma_D': 0.08,
            'gamma_W': 100.0,
            'T_max': 70.0,
            'W_min': 1400.0
        },
        'utility': {
            'unemployment': {
                'b_0': 500.0,
                'kappa': 1.0
            },
            'employment': {
                'alpha_T': 10.0
            }
        },
        'bellman': {
            'n_effort_grid': 11,  # 减少到11个以加快测试
            'rho': 0.9,
            'max_iterations': 100,
            'tolerance': 1e-4
        },
        'kfe': {
            'mu': 0.05,
            'n_evolution_steps': 1
        },
        'market': {
            'theta_mechanism': 'fixed_theta',
            'theta_bar': 1.0
        },
        'convergence': {
            'epsilon_V': 1e-3,  # 放宽容差以加快测试
            'epsilon_a': 1e-3,
            'epsilon_u': 1e-3,
            'max_iterations': 50,  # 减少最大迭代次数
            'early_stopping': True
        },
        'initialization': {
            'unemployment_rate': 0.2,
            'distribution_source': 'uniform',
            'V_U_init': 'zero',
            'V_E_init': 'zero'
        },
        'optimization': {
            'use_numba': True,
            'parallel': True,
            'cache': True,
            'fastmath': True
        },
        'output': {
            'save_path': 'results/mfg/',
            'save_intermediate': False,  # 不保存中间结果以节省空间
            'save_frequency': 10,
            'plot_convergence': True,
            'plot_distributions': False,
            'plot_policy': False
        },
        'logging': {
            'level': 'INFO',
            'log_frequency': 5,
            'verbose': True
        }
    }
    
    return config


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("MFG模拟系统 - 完整示例")
    print("农村女性就业市场均衡求解")
    print("=" * 70 + "\n")
    
    # 1. 加载配置
    logger.info("步骤1: 加载配置...")
    try:
        # 尝试加载真实配置
        config = load_config('config/default/mfg.yaml')
        # 覆盖一些参数以加快测试
        config['sparse_grid']['level'] = 3
        config['bellman']['n_effort_grid'] = 11
        config['convergence']['max_iterations'] = 50
        config['convergence']['epsilon_V'] = 1e-3
        config['convergence']['epsilon_a'] = 1e-3
        config['convergence']['epsilon_u'] = 1e-3
        logger.info("✅ 已加载真实配置（已调整为快速测试模式）")
    except FileNotFoundError:
        # 使用测试配置
        config = create_test_config()
        logger.info("✅ 使用内置测试配置")
    
    # 2. 创建MFG模拟器
    logger.info("\n步骤2: 初始化MFG模拟器...")
    simulator = MFGSimulator(
        config=config,
        match_func_params=None  # 使用默认匹配函数参数
    )
    logger.info("✅ MFG模拟器初始化完成")
    
    # 3. 求解均衡
    logger.info("\n步骤3: 开始求解MFG均衡...")
    logger.info("注意：这可能需要几分钟时间，请耐心等待...\n")
    
    start_time = datetime.now()
    result = simulator.solve(verbose=True)
    end_time = datetime.now()
    
    elapsed_time = (end_time - start_time).total_seconds()
    logger.info(f"\n✅ 求解完成！总耗时: {elapsed_time:.2f}秒")
    
    # 4. 构造均衡对象
    logger.info("\n步骤4: 构造均衡结果对象...")
    equilibrium = MFGEquilibriumSparseGrid.from_solver_result(result, config=config)
    logger.info("✅ 均衡对象构造完成")
    
    # 5. 输出摘要
    logger.info("\n步骤5: 输出均衡结果摘要...")
    print("\n" + equilibrium.summary())
    
    # 6. 保存结果
    logger.info("\n步骤6: 保存结果...")
    output_dir = Path(config['output']['save_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    simulator.save_results(result, output_dir=str(output_dir))
    
    # 保存均衡对象
    equilibrium.save(str(output_dir / 'equilibrium.npz'))
    logger.info(f"✅ 结果已保存到: {output_dir}")
    
    # 7. 绘制收敛曲线（可选）
    if config['output']['plot_convergence']:
        try:
            logger.info("\n步骤7: 绘制收敛曲线...")
            simulator.plot_convergence(result, output_dir=str(output_dir))
            logger.info("✅ 收敛图已生成")
        except Exception as e:
            logger.warning(f"绘图失败: {e}")
    
    # 8. 示例查询
    logger.info("\n步骤8: 示例查询...")
    
    # 查询一个典型状态的值
    test_state = np.array([40.0, 0.5, 0.5, 3000.0])  # [T, S_norm, D_norm, W]
    
    V_U_at_state = equilibrium.get_value_at_state(test_state, 'unemployed')
    V_E_at_state = equilibrium.get_value_at_state(test_state, 'employed')
    a_at_state = equilibrium.get_optimal_effort(test_state)
    
    print(f"\n查询示例状态: T=40h/周, S=50%, D=50%, W=3000元/月")
    print(f"  失业状态价值 V^U: {V_U_at_state:.2f}")
    print(f"  就业状态价值 V^E: {V_E_at_state:.2f}")
    print(f"  最优努力水平 a*: {a_at_state:.4f}")
    print(f"  就业收益增量 ΔV: {V_E_at_state - V_U_at_state:.2f}")
    
    # 9. 结束
    print("\n" + "=" * 70)
    print("✅ MFG模拟完成！")
    print("=" * 70)
    
    logger.info("\n" + "=" * 70)
    logger.info("模拟结束摘要:")
    logger.info(f"  - 均衡失业率: {equilibrium.unemployment_rate:.2%}")
    logger.info(f"  - 是否收敛: {equilibrium.converged}")
    logger.info(f"  - 迭代次数: {equilibrium.n_iterations}")
    logger.info(f"  - 总耗时: {equilibrium.total_time:.2f}秒")
    logger.info(f"  - 结果保存位置: {output_dir}")
    logger.info("=" * 70)
    
    return equilibrium


if __name__ == "__main__":
    try:
        # 创建必要的目录
        Path('results/mfg').mkdir(parents=True, exist_ok=True)
        Path('results/logs').mkdir(parents=True, exist_ok=True)
        
        # 运行主函数
        equilibrium = main()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断了模拟")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

