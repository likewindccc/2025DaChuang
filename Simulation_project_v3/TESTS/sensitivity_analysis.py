import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import copy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.MFG.equilibrium_solver import solve_equilibrium


def load_baseline_config():
    """加载基准配置"""
    config_path = project_root / "CONFIG" / "mfg_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def update_config_parameter(config, param_path, value):
    """
    更新配置参数
    
    参数:
        config: 配置字典
        param_path: 参数路径，如 'economics.kappa' 或 'economics.rho'
        value: 新参数值
    """
    keys = param_path.split('.')
    target = config
    for key in keys[:-1]:
        if key not in target:
            raise KeyError(f"配置路径 '{param_path}' 中的键 '{key}' 不存在。可用键: {list(target.keys())}")
        target = target[key]
    target[keys[-1]] = value


def save_config_temporarily(config, temp_path):
    """临时保存配置文件"""
    with open(temp_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def run_single_scenario(param_name, param_value, baseline_config):
    """
    运行单个敏感性分析场景
    
    参数:
        param_name: 参数路径字符串，如 'economics.kappa'
        param_value: 参数值
        baseline_config: 基准配置字典
    
    返回:
        结果字典，包含关键指标
    """
    print(f"\n{'='*80}")
    print(f"测试参数: {param_name} = {param_value}")
    print(f"{'='*80}")
    
    # 深拷贝基准配置
    config = copy.deepcopy(baseline_config)
    
    # 更新参数
    update_config_parameter(config, param_name, param_value)
    
    # 临时保存配置
    temp_config_path = project_root / "CONFIG" / "mfg_config_temp.yaml"
    save_config_temporarily(config, temp_config_path)
    
    # 运行MFG求解
    try:
        individuals_eq, eq_info = solve_equilibrium(str(temp_config_path))
        
        # 提取关键指标
        results = {
            'param_name': param_name,
            'param_value': param_value,
            'unemployment_rate': eq_info['final_statistics']['unemployment_rate'],
            'mean_T': individuals_eq['T'].mean(),
            'mean_S': individuals_eq['S'].mean(),
            'mean_D': individuals_eq['D'].mean(),
            'mean_W': individuals_eq['W'].mean(),
            'converged': eq_info.get('converged', False),
            'iterations': eq_info.get('iterations', 0)
        }
        
        # 读取策略文件获取努力水平
        policy_path = project_root / "OUTPUT" / "mfg" / "equilibrium_policy.csv"
        if policy_path.exists():
            policy_df = pd.read_csv(policy_path)
            results['mean_effort'] = policy_df['a_optimal'].mean()
        else:
            results['mean_effort'] = np.nan
        
        print(f"\n结果摘要:")
        print(f"  失业率: {results['unemployment_rate']:.4f}")
        print(f"  平均T: {results['mean_T']:.2f}")
        print(f"  平均S: {results['mean_S']:.2f}")
        print(f"  平均D: {results['mean_D']:.2f}")
        print(f"  平均努力: {results['mean_effort']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"运行失败: {str(e)}")
        return {
            'param_name': param_name,
            'param_value': param_value,
            'unemployment_rate': np.nan,
            'mean_T': np.nan,
            'mean_S': np.nan,
            'mean_D': np.nan,
            'mean_W': np.nan,
            'mean_effort': np.nan,
            'converged': False,
            'iterations': 0
        }
    
    finally:
        # 删除临时配置文件
        if temp_config_path.exists():
            temp_config_path.unlink()


def sensitivity_analysis_rho():
    """
    rho（贴现因子）敏感性分析
    
    rho越大，个体越重视未来，可能更愿意努力提升人力资本
    """
    print("\n" + "="*80)
    print("敏感性分析 1: rho（贴现因子）")
    print("="*80)
    
    baseline_config = load_baseline_config()
    
    # 测试不同的rho值
    rho_values = [0.70, 0.75, 0.80, 0.85]
    
    results = []
    for rho in rho_values:
        result = run_single_scenario('economics.rho', rho, baseline_config)
        results.append(result)
    
    return pd.DataFrame(results)


def sensitivity_analysis_kappa():
    """
    kappa（努力成本系数）敏感性分析
    
    kappa越大，努力成本越高，个体越不愿意努力
    """
    print("\n" + "="*80)
    print("敏感性分析 2: kappa（努力成本系数）")
    print("="*80)
    
    baseline_config = load_baseline_config()
    
    # 测试不同的kappa值
    kappa_values = [0.5, 0.8, 1.0, 1.5, 2.0]
    
    results = []
    for kappa in kappa_values:
        result = run_single_scenario('economics.kappa', kappa, baseline_config)
        results.append(result)
    
    return pd.DataFrame(results)


def sensitivity_analysis_gamma():
    """
    gamma系数（状态更新系数）敏感性分析
    
    gamma越大，努力对状态变量的影响越大
    测试同时缩放所有gamma系数
    """
    print("\n" + "="*80)
    print("敏感性分析 3: gamma系数（状态更新系数）")
    print("="*80)
    
    baseline_config = load_baseline_config()
    
    # 获取基准gamma值
    baseline_gamma_T = baseline_config['economics']['state_update']['gamma_T']
    baseline_gamma_S = baseline_config['economics']['state_update']['gamma_S']
    baseline_gamma_D = baseline_config['economics']['state_update']['gamma_D']
    baseline_gamma_W = baseline_config['economics']['state_update']['gamma_W']
    
    # 测试不同的缩放倍数
    scale_factors = [0.5, 1.0, 1.5, 2.0]
    
    results = []
    for scale in scale_factors:
        print(f"\n测试gamma缩放倍数: {scale}")
        
        # 深拷贝配置
        config = copy.deepcopy(baseline_config)
        
        # 同时缩放所有gamma
        config['economics']['state_update']['gamma_T'] = baseline_gamma_T * scale
        config['economics']['state_update']['gamma_S'] = baseline_gamma_S * scale
        config['economics']['state_update']['gamma_D'] = baseline_gamma_D * scale
        config['economics']['state_update']['gamma_W'] = baseline_gamma_W * scale
        
        # 保存临时配置
        temp_config_path = project_root / "CONFIG" / "mfg_config_temp.yaml"
        save_config_temporarily(config, temp_config_path)
        
        try:
            individuals_eq, eq_info = solve_equilibrium(str(temp_config_path))
            
            policy_path = project_root / "OUTPUT" / "mfg" / "equilibrium_policy.csv"
            policy_df = pd.read_csv(policy_path)
            
            result = {
                'param_name': 'gamma_scale',
                'param_value': scale,
                'unemployment_rate': eq_info['final_statistics']['unemployment_rate'],
                'mean_T': individuals_eq['T'].mean(),
                'mean_S': individuals_eq['S'].mean(),
                'mean_D': individuals_eq['D'].mean(),
                'mean_W': individuals_eq['W'].mean(),
                'mean_effort': policy_df['a_optimal'].mean(),
                'converged': eq_info.get('converged', False),
                'iterations': eq_info.get('iterations', 0)
            }
            
            print(f"结果: 失业率={result['unemployment_rate']:.4f}, "
                  f"T={result['mean_T']:.2f}, "
                  f"努力={result['mean_effort']:.4f}")
            
            results.append(result)
            
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    return pd.DataFrame(results)


def sensitivity_analysis_damping():
    """
    damping_factor（阻尼因子）敏感性分析
    
    damping_factor越大，值函数更新越激进
    """
    print("\n" + "="*80)
    print("敏感性分析 4: damping_factor（阻尼因子）")
    print("="*80)
    
    baseline_config = load_baseline_config()
    
    # 测试不同的阻尼因子
    damping_values = [0.1, 0.3, 0.5, 0.7]
    
    results = []
    for damping in damping_values:
        result = run_single_scenario(
            'equilibrium.damping_factor', 
            damping, 
            baseline_config
        )
        results.append(result)
    
    return pd.DataFrame(results)


def generate_report(all_results):
    """
    生成敏感性分析报告
    
    参数:
        all_results: 字典，包含各参数的敏感性分析结果DataFrame
    """
    print("\n" + "="*80)
    print("敏感性分析总结报告")
    print("="*80)
    
    output_dir = project_root / "OUTPUT" / "sensitivity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存所有结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for param_name, results_df in all_results.items():
        # 保存CSV
        csv_path = output_dir / f"sensitivity_{param_name}_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n{param_name} 结果已保存至: {csv_path}")
        
        # 打印关键统计
        print(f"\n{param_name} 敏感性分析:")
        print("-" * 80)
        
        # 计算变化范围
        if len(results_df) > 0:
            unemployment_range = (
                results_df['unemployment_rate'].max() - 
                results_df['unemployment_rate'].min()
            )
            effort_range = (
                results_df['mean_effort'].max() - 
                results_df['mean_effort'].min()
            )
            T_range = results_df['mean_T'].max() - results_df['mean_T'].min()
            
            print(f"失业率变化范围: {unemployment_range:.4f} "
                  f"({results_df['unemployment_rate'].min():.4f} - "
                  f"{results_df['unemployment_rate'].max():.4f})")
            print(f"努力水平变化范围: {effort_range:.4f} "
                  f"({results_df['mean_effort'].min():.4f} - "
                  f"{results_df['mean_effort'].max():.4f})")
            print(f"T变化范围: {T_range:.2f} "
                  f"({results_df['mean_T'].min():.2f} - "
                  f"{results_df['mean_T'].max():.2f})")
        
        # 打印详细表格
        print(f"\n详细结果:")
        display_cols = [
            'param_value', 
            'unemployment_rate', 
            'mean_effort', 
            'mean_T', 
            'mean_S'
        ]
        print(results_df[display_cols].to_string(index=False))
    
    # 合并所有结果
    all_results_df = pd.concat(all_results.values(), ignore_index=True)
    combined_path = output_dir / f"sensitivity_all_{timestamp}.csv"
    all_results_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"\n所有结果已保存至: {combined_path}")
    
    # 识别最敏感的参数
    print("\n" + "="*80)
    print("参数敏感性排名（按失业率变化范围）")
    print("="*80)
    
    sensitivity_ranking = []
    for param_name, results_df in all_results.items():
        if len(results_df) > 0 and not results_df['unemployment_rate'].isna().all():
            unemployment_range = (
                results_df['unemployment_rate'].max() - 
                results_df['unemployment_rate'].min()
            )
            sensitivity_ranking.append({
                'parameter': param_name,
                'unemployment_range': unemployment_range
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_ranking)
    sensitivity_df = sensitivity_df.sort_values(
        'unemployment_range', 
        ascending=False
    )
    
    print(sensitivity_df.to_string(index=False))


def main():
    """主函数：执行完整的敏感性分析"""
    print("="*80)
    print("MFG模型敏感性分析")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目路径: {project_root}")
    
    # 执行各参数的敏感性分析
    all_results = {}
    
    # 1. rho敏感性分析
    all_results['rho'] = sensitivity_analysis_rho()
    
    # 2. kappa敏感性分析
    all_results['kappa'] = sensitivity_analysis_kappa()
    
    # 3. gamma敏感性分析
    all_results['gamma'] = sensitivity_analysis_gamma()
    
    # 4. damping敏感性分析
    all_results['damping'] = sensitivity_analysis_damping()
    
    # 生成报告
    generate_report(all_results)
    
    print("\n" + "="*80)
    print("敏感性分析完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()

