"""
Population Generator Module Integration Test

测试Module 1的完整功能，包括：
1. 劳动力生成器测试
2. 企业生成器测试  
3. 性能基准测试
4. 数据质量验证
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 本地模块导入
from .config import PopulationConfig, create_default_config
from .labor_generator import LaborAgentGenerator
from .enterprise_generator import EnterpriseGenerator
from .utils import DataValidator, Visualizer, compute_data_quality_score
from .optimization import get_optimization_info, check_numba_availability

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModuleIntegrationTester:
    """模块集成测试器"""
    
    def __init__(self, config: PopulationConfig = None):
        """
        初始化测试器
        
        Args:
            config: 测试配置，默认使用default config
        """
        self.config = config or create_default_config()
        self.test_results = {}
        self.performance_results = {}
        
        # 创建测试输出目录
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("初始化模块集成测试器")
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """运行完整测试套件"""
        logger.info("开始运行完整测试套件...")
        
        test_suite_results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment_info': self._get_environment_info(),
            'tests_performed': {}
        }
        
        try:
            # 1. 环境和依赖测试
            logger.info("1. 环境和依赖测试")
            env_results = self._test_environment()
            test_suite_results['tests_performed']['environment'] = env_results
            
            # 2. 生成测试数据
            logger.info("2. 生成测试数据")
            test_data = self._generate_test_data()
            
            # 3. 劳动力生成器测试
            logger.info("3. 劳动力生成器测试")
            labor_results = self._test_labor_generator(test_data['labor_data'])
            test_suite_results['tests_performed']['labor_generator'] = labor_results
            
            # 4. 企业生成器测试
            logger.info("4. 企业生成器测试")
            enterprise_results = self._test_enterprise_generator(test_data['enterprise_data'])
            test_suite_results['tests_performed']['enterprise_generator'] = enterprise_results
            
            # 5. 性能基准测试
            logger.info("5. 性能基准测试")
            performance_results = self._test_performance()
            test_suite_results['tests_performed']['performance'] = performance_results
            
            # 6. 大规模生成测试
            logger.info("6. 大规模生成测试")
            scale_results = self._test_large_scale_generation()
            test_suite_results['tests_performed']['large_scale'] = scale_results
            
            # 7. 数据质量对比测试
            logger.info("7. 数据质量对比测试")
            quality_results = self._test_data_quality_comparison(
                test_data, labor_results, enterprise_results
            )
            test_suite_results['tests_performed']['data_quality'] = quality_results
            
            # 计算总体测试评分
            test_suite_results['overall_score'] = self._calculate_overall_score(
                test_suite_results['tests_performed']
            )
            
            # 生成测试报告
            self._generate_test_report(test_suite_results)
            
            logger.info("完整测试套件运行完成")
            return test_suite_results
            
        except Exception as e:
            logger.error(f"测试套件运行失败: {e}")
            test_suite_results['error'] = str(e)
            test_suite_results['overall_score'] = 0.0
            return test_suite_results
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        import sys
        import platform
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'numba_info': get_optimization_info(),
            'memory_available_gb': self._get_available_memory_gb()
        }
    
    def _get_available_memory_gb(self) -> float:
        """获取可用内存（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 0.0
    
    def _test_environment(self) -> Dict[str, Any]:
        """测试环境和依赖"""
        results = {'status': 'passed', 'issues': []}
        
        # 检查关键依赖
        try:
            from copulas.multivariate import GaussianMultivariate
            results['copulas_available'] = True
        except ImportError:
            results['copulas_available'] = False
            results['issues'].append('Copulas库不可用')
        
        # 检查numba优化
        results['numba_available'] = check_numba_availability()
        if not results['numba_available']:
            results['issues'].append('numba优化不可用')
        
        # 检查内存
        available_memory = self._get_available_memory_gb()
        if available_memory < 1.0:
            results['issues'].append(f'可用内存不足: {available_memory:.2f}GB')
        
        results['memory_sufficient'] = available_memory >= 1.0
        
        if results['issues']:
            results['status'] = 'warning'
        
        return results
    
    def _generate_test_data(self) -> Dict[str, pd.DataFrame]:
        """生成测试数据"""
        logger.info("生成测试数据...")
        
        # 生成劳动力测试数据
        np.random.seed(42)
        n_labor = 1000
        
        labor_data = pd.DataFrame({
            'T': np.random.normal(35, 10, n_labor),
            'S': np.random.beta(2, 3, n_labor),
            'D': np.random.beta(1.5, 2.5, n_labor),
            'W': np.random.lognormal(np.log(3000), 0.5, n_labor),
            'age': np.random.randint(18, 60, n_labor),
            'education': np.random.randint(0, 5, n_labor)
        })
        
        # 应用数据边界
        labor_data['T'] = np.clip(labor_data['T'], 0, 80)
        labor_data['S'] = np.clip(labor_data['S'], 0, 1)
        labor_data['D'] = np.clip(labor_data['D'], 0, 1)
        labor_data['W'] = np.clip(labor_data['W'], 1000, 8000)
        labor_data['age'] = np.clip(labor_data['age'], 16, 65)
        
        # 生成企业测试数据
        n_enterprise = 500
        
        # 使用多元正态分布生成相关数据
        mean = [40, 0.5, 0.5, 3500]
        cov = [[100, 0.1, 0.1, 200],
               [0.1, 0.05, 0.02, 50],
               [0.1, 0.02, 0.05, 50],
               [200, 50, 50, 500000]]
        
        enterprise_array = np.random.multivariate_normal(mean, cov, n_enterprise)
        enterprise_data = pd.DataFrame(enterprise_array, columns=['T_req', 'S_req', 'D_req', 'W_offer'])
        
        # 应用数据边界
        enterprise_data['T_req'] = np.clip(enterprise_data['T_req'], 20, 60)
        enterprise_data['S_req'] = np.clip(enterprise_data['S_req'], 0, 1)
        enterprise_data['D_req'] = np.clip(enterprise_data['D_req'], 0, 1)
        enterprise_data['W_offer'] = np.clip(enterprise_data['W_offer'], 1500, 7000)
        
        logger.info(f"生成测试数据完成: 劳动力{n_labor}个, 企业{n_enterprise}个")
        
        return {
            'labor_data': labor_data,
            'enterprise_data': enterprise_data
        }
    
    def _test_labor_generator(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试劳动力生成器"""
        results = {'status': 'failed', 'error': None}
        
        try:
            # 创建生成器
            generator = LaborAgentGenerator(self.config.labor_config, random_state=42)
            
            # 测试拟合
            start_time = time.time()
            generator.fit(test_data)
            fit_time = time.time() - start_time
            
            results['fit_time'] = fit_time
            results['fitted_successfully'] = generator.is_fitted
            
            # 测试生成
            n_test_generate = 500
            start_time = time.time()
            generated_data = generator.generate(n_test_generate)
            generate_time = time.time() - start_time
            
            results['generate_time'] = generate_time
            results['generated_count'] = len(generated_data)
            results['generation_rate'] = n_test_generate / generate_time  # 个/秒
            
            # 验证数据质量
            is_valid, validation_report = generator.validate(generated_data)
            results['validation_passed'] = is_valid
            results['data_quality_score'] = compute_data_quality_score(
                generated_data, validation_report
            )
            
            # 保存生成结果
            output_file = self.output_dir / "labor_generated_test.csv"
            generated_data.to_csv(output_file, index=False)
            results['output_saved'] = str(output_file)
            
            results['status'] = 'passed'
            results['generated_data'] = generated_data
            
            logger.info(f"劳动力生成器测试通过: 拟合{fit_time:.2f}s, "
                       f"生成{generate_time:.2f}s, 质量分{results['data_quality_score']:.3f}")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"劳动力生成器测试失败: {e}")
        
        return results
    
    def _test_enterprise_generator(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """测试企业生成器"""
        results = {'status': 'failed', 'error': None}
        
        try:
            # 创建生成器
            generator = EnterpriseGenerator(self.config.enterprise_config, random_state=42)
            
            # 测试拟合
            start_time = time.time()
            generator.fit(test_data)
            fit_time = time.time() - start_time
            
            results['fit_time'] = fit_time
            results['fitted_successfully'] = generator.is_fitted
            
            # 测试生成
            n_test_generate = 300
            start_time = time.time()
            generated_data = generator.generate(n_test_generate)
            generate_time = time.time() - start_time
            
            results['generate_time'] = generate_time
            results['generated_count'] = len(generated_data)
            results['generation_rate'] = n_test_generate / generate_time  # 个/秒
            
            # 验证数据质量
            is_valid, validation_report = generator.validate(generated_data)
            results['validation_passed'] = is_valid
            results['data_quality_score'] = compute_data_quality_score(
                generated_data, validation_report
            )
            
            # 保存生成结果
            output_file = self.output_dir / "enterprise_generated_test.csv"
            generated_data.to_csv(output_file, index=False)
            results['output_saved'] = str(output_file)
            
            results['status'] = 'passed'
            results['generated_data'] = generated_data
            
            logger.info(f"企业生成器测试通过: 拟合{fit_time:.2f}s, "
                       f"生成{generate_time:.2f}s, 质量分{results['data_quality_score']:.3f}")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"企业生成器测试失败: {e}")
        
        return results
    
    def _test_performance(self) -> Dict[str, Any]:
        """性能基准测试"""
        results = {}
        
        try:
            # 测试不同规模下的性能
            test_sizes = [100, 500, 1000, 2000]
            performance_data = []
            
            # 创建一次性测试数据
            test_data = self._generate_test_data()
            
            for size in test_sizes:
                logger.info(f"测试规模 {size}...")
                
                # 劳动力生成器性能测试
                labor_generator = LaborAgentGenerator(self.config.labor_config, random_state=42)
                labor_generator.fit(test_data['labor_data'])
                
                start_time = time.time()
                labor_generated = labor_generator.generate(size)
                labor_time = time.time() - start_time
                
                # 企业生成器性能测试
                enterprise_generator = EnterpriseGenerator(self.config.enterprise_config, random_state=42)
                enterprise_generator.fit(test_data['enterprise_data'])
                
                start_time = time.time()
                enterprise_generated = enterprise_generator.generate(size)
                enterprise_time = time.time() - start_time
                
                performance_data.append({
                    'size': size,
                    'labor_time': labor_time,
                    'labor_rate': size / labor_time,
                    'enterprise_time': enterprise_time,
                    'enterprise_rate': size / enterprise_time,
                    'total_time': labor_time + enterprise_time
                })
            
            results['performance_data'] = performance_data
            results['status'] = 'passed'
            
            # 计算性能评级
            avg_labor_rate = np.mean([p['labor_rate'] for p in performance_data])
            avg_enterprise_rate = np.mean([p['enterprise_rate'] for p in performance_data])
            
            results['avg_labor_generation_rate'] = avg_labor_rate
            results['avg_enterprise_generation_rate'] = avg_enterprise_rate
            
            # 性能评级 (个/秒)
            if avg_labor_rate > 1000:
                results['labor_performance_grade'] = 'excellent'
            elif avg_labor_rate > 500:
                results['labor_performance_grade'] = 'good'
            else:
                results['labor_performance_grade'] = 'fair'
            
            if avg_enterprise_rate > 2000:
                results['enterprise_performance_grade'] = 'excellent'
            elif avg_enterprise_rate > 1000:
                results['enterprise_performance_grade'] = 'good'
            else:
                results['enterprise_performance_grade'] = 'fair'
            
            logger.info(f"性能测试完成: 劳动力{avg_labor_rate:.0f}个/秒, "
                       f"企业{avg_enterprise_rate:.0f}个/秒")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"性能测试失败: {e}")
        
        return results
    
    def _test_large_scale_generation(self) -> Dict[str, Any]:
        """大规模生成测试"""
        results = {'status': 'failed'}
        
        try:
            # 测试大规模生成能力
            large_scale_size = 5000
            logger.info(f"大规模生成测试: {large_scale_size}个主体")
            
            # 准备数据
            test_data = self._generate_test_data()
            
            # 劳动力大规模生成
            labor_generator = LaborAgentGenerator(self.config.labor_config, random_state=42)
            labor_generator.fit(test_data['labor_data'])
            
            start_time = time.time()
            large_labor_data = labor_generator.generate(large_scale_size)
            labor_large_time = time.time() - start_time
            
            # 企业大规模生成
            enterprise_generator = EnterpriseGenerator(self.config.enterprise_config, random_state=42)
            enterprise_generator.fit(test_data['enterprise_data'])
            
            start_time = time.time()
            large_enterprise_data = enterprise_generator.generate(large_scale_size)
            enterprise_large_time = time.time() - start_time
            
            results.update({
                'status': 'passed',
                'target_size': large_scale_size,
                'labor_generated': len(large_labor_data),
                'enterprise_generated': len(large_enterprise_data),
                'labor_time': labor_large_time,
                'enterprise_time': enterprise_large_time,
                'labor_memory_mb': large_labor_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'enterprise_memory_mb': large_enterprise_data.memory_usage(deep=True).sum() / 1024 / 1024
            })
            
            # 保存大规模数据样本
            sample_size = 1000
            labor_sample = large_labor_data.sample(sample_size)
            enterprise_sample = large_enterprise_data.sample(sample_size)
            
            labor_sample.to_csv(self.output_dir / "large_scale_labor_sample.csv", index=False)
            enterprise_sample.to_csv(self.output_dir / "large_scale_enterprise_sample.csv", index=False)
            
            logger.info(f"大规模生成测试通过: 劳动力{labor_large_time:.1f}s, "
                       f"企业{enterprise_large_time:.1f}s")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"大规模生成测试失败: {e}")
        
        return results
    
    def _test_data_quality_comparison(self, 
                                    original_data: Dict[str, pd.DataFrame],
                                    labor_results: Dict[str, Any],
                                    enterprise_results: Dict[str, Any]) -> Dict[str, Any]:
        """数据质量对比测试"""
        results = {'status': 'failed'}
        
        try:
            comparison_results = {}
            
            # 劳动力数据对比
            if 'generated_data' in labor_results:
                labor_comparison = self._compare_distributions(
                    original_data['labor_data'], 
                    labor_results['generated_data']
                )
                comparison_results['labor'] = labor_comparison
                
                # 生成对比图
                visualizer = Visualizer()
                visualizer.plot_generation_comparison(
                    original_data['labor_data'],
                    labor_results['generated_data'],
                    ['T', 'S', 'D', 'W'],
                    save_path=str(self.output_dir / "labor_comparison.png")
                )
            
            # 企业数据对比
            if 'generated_data' in enterprise_results:
                enterprise_comparison = self._compare_distributions(
                    original_data['enterprise_data'],
                    enterprise_results['generated_data']
                )
                comparison_results['enterprise'] = enterprise_comparison
                
                # 生成对比图
                visualizer = Visualizer()
                visualizer.plot_generation_comparison(
                    original_data['enterprise_data'],
                    enterprise_results['generated_data'],
                    ['T_req', 'S_req', 'D_req', 'W_offer'],
                    save_path=str(self.output_dir / "enterprise_comparison.png")
                )
            
            # 计算总体质量得分
            quality_scores = []
            for comp in comparison_results.values():
                if 'overall_similarity' in comp:
                    quality_scores.append(comp['overall_similarity'])
            
            results.update({
                'status': 'passed',
                'comparison_results': comparison_results,
                'overall_quality_score': np.mean(quality_scores) if quality_scores else 0.0
            })
            
            logger.info(f"数据质量对比完成，总体得分: {results['overall_quality_score']:.3f}")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"数据质量对比测试失败: {e}")
        
        return results
    
    def _compare_distributions(self, original: pd.DataFrame, generated: pd.DataFrame) -> Dict[str, Any]:
        """比较原始数据和生成数据的分布"""
        comparison = {}
        
        common_columns = set(original.columns) & set(generated.columns)
        
        for col in common_columns:
            # KS检验
            from scipy.stats import ks_2samp
            ks_stat, p_value = ks_2samp(original[col].dropna(), generated[col].dropna())
            
            # 统计量对比
            orig_mean = original[col].mean()
            gen_mean = generated[col].mean()
            orig_std = original[col].std()
            gen_std = generated[col].std()
            
            comparison[col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': p_value,
                'mean_diff_pct': abs(gen_mean - orig_mean) / orig_mean * 100,
                'std_diff_pct': abs(gen_std - orig_std) / orig_std * 100,
                'similarity_score': max(0, 1 - ks_stat)  # 基于KS统计量的相似度
            }
        
        # 总体相似度
        similarity_scores = [comp['similarity_score'] for comp in comparison.values()]
        comparison['overall_similarity'] = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return comparison
    
    def _calculate_overall_score(self, test_results: Dict[str, Any]) -> float:
        """计算总体测试得分"""
        scores = []
        
        # 环境测试 (权重: 0.1)
        if test_results.get('environment', {}).get('status') == 'passed':
            scores.append(1.0 * 0.1)
        
        # 劳动力生成器测试 (权重: 0.3)
        labor_result = test_results.get('labor_generator', {})
        if labor_result.get('status') == 'passed':
            quality_score = labor_result.get('data_quality_score', 0)
            scores.append(quality_score * 0.3)
        
        # 企业生成器测试 (权重: 0.3)
        enterprise_result = test_results.get('enterprise_generator', {})
        if enterprise_result.get('status') == 'passed':
            quality_score = enterprise_result.get('data_quality_score', 0)
            scores.append(quality_score * 0.3)
        
        # 性能测试 (权重: 0.15)
        perf_result = test_results.get('performance', {})
        if perf_result.get('status') == 'passed':
            # 基于生成速率的得分
            labor_rate = perf_result.get('avg_labor_generation_rate', 0)
            enterprise_rate = perf_result.get('avg_enterprise_generation_rate', 0)
            perf_score = min(1.0, (labor_rate + enterprise_rate) / 2000)  # 归一化
            scores.append(perf_score * 0.15)
        
        # 数据质量对比 (权重: 0.15)
        quality_result = test_results.get('data_quality', {})
        if quality_result.get('status') == 'passed':
            quality_score = quality_result.get('overall_quality_score', 0)
            scores.append(quality_score * 0.15)
        
        return sum(scores)
    
    def _generate_test_report(self, results: Dict[str, Any]) -> None:
        """生成测试报告"""
        report_file = self.output_dir / "module_test_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Population Generator Module 测试报告\n\n")
            f.write(f"**生成时间**: {results['test_timestamp']}\n")
            f.write(f"**总体得分**: {results['overall_score']:.2f}/1.00\n\n")
            
            # 环境信息
            f.write("## 环境信息\n\n")
            env_info = results['environment_info']
            f.write(f"- Python版本: {env_info['python_version'].split()[0]}\n")
            f.write(f"- 平台: {env_info['platform']}\n")
            f.write(f"- NumPy版本: {env_info['numpy_version']}\n")
            f.write(f"- Pandas版本: {env_info['pandas_version']}\n")
            f.write(f"- 可用内存: {env_info['memory_available_gb']:.1f} GB\n")
            f.write(f"- numba可用: {env_info['numba_info']['numba_available']}\n\n")
            
            # 测试结果
            f.write("## 测试结果\n\n")
            for test_name, test_result in results['tests_performed'].items():
                status = test_result.get('status', 'unknown')
                status_emoji = '✅' if status == 'passed' else '❌' if status == 'failed' else '⚠️'
                f.write(f"### {test_name.replace('_', ' ').title()} {status_emoji}\n\n")
                
                if status == 'passed':
                    if test_name == 'labor_generator':
                        f.write(f"- 拟合时间: {test_result.get('fit_time', 0):.2f}秒\n")
                        f.write(f"- 生成速率: {test_result.get('generation_rate', 0):.0f}个/秒\n")
                        f.write(f"- 数据质量得分: {test_result.get('data_quality_score', 0):.3f}\n")
                    elif test_name == 'enterprise_generator':
                        f.write(f"- 拟合时间: {test_result.get('fit_time', 0):.2f}秒\n")
                        f.write(f"- 生成速率: {test_result.get('generation_rate', 0):.0f}个/秒\n")
                        f.write(f"- 数据质量得分: {test_result.get('data_quality_score', 0):.3f}\n")
                    elif test_name == 'performance':
                        f.write(f"- 劳动力生成速率: {test_result.get('avg_labor_generation_rate', 0):.0f}个/秒\n")
                        f.write(f"- 企业生成速率: {test_result.get('avg_enterprise_generation_rate', 0):.0f}个/秒\n")
                        f.write(f"- 劳动力性能评级: {test_result.get('labor_performance_grade', 'N/A')}\n")
                        f.write(f"- 企业性能评级: {test_result.get('enterprise_performance_grade', 'N/A')}\n")
                elif 'error' in test_result:
                    f.write(f"- 错误信息: {test_result['error']}\n")
                
                f.write("\n")
            
            f.write("## 结论\n\n")
            score = results['overall_score']
            if score >= 0.9:
                f.write("✅ **测试结果优秀**，模块功能完整且性能良好。\n")
            elif score >= 0.7:
                f.write("⚠️ **测试结果良好**，模块基本功能正常，部分方面需要优化。\n")
            else:
                f.write("❌ **测试结果需要改进**，存在较多问题需要修复。\n")
        
        logger.info(f"测试报告已生成: {report_file}")


def run_quick_test() -> bool:
    """运行快速测试"""
    logger.info("运行快速测试...")
    
    try:
        tester = ModuleIntegrationTester()
        
        # 生成小规模测试数据
        test_data = tester._generate_test_data()
        
        # 快速劳动力生成器测试
        labor_results = tester._test_labor_generator(test_data['labor_data'])
        if labor_results['status'] != 'passed':
            logger.error("劳动力生成器快速测试失败")
            return False
        
        # 快速企业生成器测试
        enterprise_results = tester._test_enterprise_generator(test_data['enterprise_data'])
        if enterprise_results['status'] != 'passed':
            logger.error("企业生成器快速测试失败")
            return False
        
        logger.info("快速测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"快速测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行完整测试套件
    tester = ModuleIntegrationTester()
    results = tester.run_full_test_suite()
    
    print(f"\n测试完成！总体得分: {results['overall_score']:.2f}/1.00")
    print(f"详细报告请查看: {tester.output_dir / 'module_test_report.md'}")
