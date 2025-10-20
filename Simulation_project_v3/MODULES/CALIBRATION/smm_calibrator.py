import os
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize, differential_evolution, OptimizeResult

from .target_moments import TargetMoments
from .objective_function import ObjectiveFunction, create_weight_matrix
from .optimization_utils import (
    OptimizationUtils,
    update_mfg_config_with_params,
    validate_parameters
)


class SMMCalibrator:
    """
    SMM校准器核心类
    
    功能：
    1. 管理完整的校准流程
    2. 调用scipy.optimize进行参数优化
    3. 实现断点续跑机制
    4. 保存中间结果和最终结果
    5. 生成诊断报告
    
    属性：
        config: 校准配置字典
        target_moments: 目标矩管理器
        param_utils: 参数工具类
        obj_function: 目标函数实例
        output_dir: 输出目录
        checkpoint_enabled: 是否启用断点续跑
    """
    
    def __init__(self, config_path: str):
        """
        初始化SMM校准器
        
        参数:
            config_path: calibration_config.yaml配置文件路径
        """
        self.config_path = Path(config_path)
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化输出目录
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化目标矩管理器
        target_moments_path = Path(
            self.config['target_moments']['config_file']
        )
        self.target_moments = TargetMoments(str(target_moments_path))
        
        # 初始化参数工具
        self.param_utils = OptimizationUtils(self.config)
        
        # 初始化权重矩阵
        n_moments = self.target_moments.get_n_moments()
        weight_type = self.config['target_moments']['weight_type']
        self.weight_matrix = create_weight_matrix(n_moments, weight_type)
        
        # 断点续跑设置
        self.checkpoint_enabled = self.config['checkpoint']['enabled']
        self.checkpoint_dir = Path(
            self.config['checkpoint']['checkpoint_dir']
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # MFG求解器设置
        self.mfg_config_path = Path(
            self.config['mfg_solver']['config_path']
        )
        
        # 初始化目标函数（在calibrate中创建，因为需要MFG求解器）
        self.obj_function = None
        
        # 优化结果
        self.result = None
    
    def _create_mfg_solver(self) -> callable:
        """
        创建MFG求解器函数
        
        返回:
            MFG求解函数，签名为 func(params_vector) -> (individuals, eq_info)
        """
        from MODULES.MFG.equilibrium_solver import solve_equilibrium
        
        mfg_config_path = self.mfg_config_path
        param_utils = self.param_utils
        output_dir = self.output_dir
        
        def mfg_solver(params_vector: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
            """
            MFG求解器包装函数
            
            参数:
                params_vector: 参数向量
            
            返回:
                (individuals, eq_info)
            """
            # 验证参数
            is_valid, error_msg = validate_parameters(
                params_vector, 
                param_utils
            )
            
            if not is_valid:
                raise ValueError(f"参数无效: {error_msg}")
            
            # 转换为参数字典
            params_dict = param_utils.vector_to_dict(params_vector)
            
            # 创建临时MFG配置文件（使用进程ID确保并行安全）
            temp_config_path = output_dir / f'mfg_config_temp_{os.getpid()}.yaml'
            
            try:
                update_mfg_config_with_params(
                    mfg_config_path,
                    params_dict,
                    param_utils,
                    temp_config_path
                )
                
                # 运行MFG求解（禁用文件保存以避免并发冲突）
                individuals, eq_info = solve_equilibrium(
                    str(temp_config_path),
                    save_results=False
                )
                
                return individuals, eq_info
                
            finally:
                # 确保删除临时配置文件（即使出错也要删除）
                if temp_config_path.exists():
                    temp_config_path.unlink()
        
        return mfg_solver
    
    def calibrate(
        self, 
        method: Optional[str] = None,
        initial_values: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """
        执行SMM校准
        
        参数:
            method: 优化方法（如果为None则使用配置文件中的方法）
            initial_values: 初始参数值（如果为None则使用配置文件中的初始值）
        
        返回:
            scipy.optimize.OptimizeResult对象
        """
        print("\n" + "="*80)
        print("开始SMM校准")
        print("="*80)
        print(f"配置文件: {self.config_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查是否有断点可以恢复
        if self.checkpoint_enabled and self.config['checkpoint']['auto_resume']:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is not None:
                print(f"\n发现断点文件: {checkpoint_path}")
                response = input("是否从断点恢复？(y/n): ")
                if response.lower() == 'y':
                    return self.resume_from_checkpoint(checkpoint_path)
        
        # 打印参数信息
        self.param_utils.print_parameter_info()
        
        # 打印目标矩信息
        print("\n" + "="*80)
        print("目标矩信息")
        print("="*80)
        target_moments = self.target_moments.get_target_moments()
        for name, value in target_moments.items():
            print(f"{name}: {value}")
        
        # 确定优化方法
        if method is None:
            method = self.config['optimization']['method']
        
        # 确定初始值
        if initial_values is None:
            initial_values = self.param_utils.get_initial_values('baseline')
        
        print(f"\n优化方法: {method}")
        print(f"初始参数: {initial_values}")
        
        # 创建MFG求解器
        print("\n初始化MFG求解器...")
        mfg_solver = self._create_mfg_solver()
        
        # 创建目标函数
        print("初始化目标函数...")
        self.obj_function = ObjectiveFunction(
            self.target_moments,
            self.weight_matrix,
            mfg_solver,
            self.output_dir
        )
        
        # 获取优化器选项
        options = self.config['optimization']['options'].copy()
        
        # 获取参数边界
        bounds = self.param_utils.get_parameter_bounds()
        
        # 创建回调函数（用于断点保存）
        def callback(xk):
            """优化过程中的回调函数"""
            if self.checkpoint_enabled:
                n_eval = self.obj_function.get_evaluation_count()
                save_freq = self.config['checkpoint']['save_frequency']
                
                if n_eval % save_freq == 0:
                    self._save_checkpoint(xk, None)
        
        # 运行优化
        print("\n" + "="*80)
        print("开始优化迭代...")
        print("="*80)
        
        # 根据方法选择优化器
        if method == 'differential_evolution':
            print(f"使用并行差分进化算法")
            print(f"种群大小: {options.get('popsize', 15)}")
            n_workers = options.get('workers', 1)
            print(f"并行进程数: {n_workers}")
            
            # differential_evolution的callback接口与minimize不同
            # 它接收 (xk, convergence) 而不是 (xk)
            def de_callback(xk, convergence=0):
                """差分进化算法的回调函数"""
                if self.checkpoint_enabled:
                    n_eval = self.obj_function.get_evaluation_count()
                    save_freq = self.config['checkpoint']['save_frequency']
                    if n_eval % save_freq == 0:
                        self._save_checkpoint(xk, None)
                return False  # 返回True会提前终止优化
            
            # differential_evolution支持的参数（过滤掉不兼容的参数）
            de_valid_params = {
                'maxiter', 'popsize', 'atol', 'tol', 'workers', 
                'updating', 'polish', 'strategy', 'recombination', 
                'mutation', 'seed', 'init', 'disp'
            }
            de_options = {k: v for k, v in options.items() if k in de_valid_params}
            
            # 直接传递workers整数，scipy会自动处理并行
            # 注意：需要使用支持闭包序列化的loky库
            if n_workers > 1:
                print(f"警告：并行模式下禁用checkpoint回调（避免进程冲突）")
                print(f"使用loky.get_reusable_executor进行闭包序列化")
                
                # 使用loky提供的executor，支持闭包序列化
                from loky import get_reusable_executor
                
                # 创建可重用的executor
                executor = get_reusable_executor(max_workers=n_workers)
                
                # 修改配置以使用executor.map
                de_options_copy = de_options.copy()
                de_options_copy['workers'] = executor.map
                # 保持updating='deferred'以确保并行评估整个种群
                
                result = differential_evolution(
                    func=self.obj_function,
                    bounds=bounds,
                    callback=None,
                    **de_options_copy
                )
            else:
                result = differential_evolution(
                    func=self.obj_function,
                    bounds=bounds,
                    callback=de_callback,
                    **de_options
                )
        else:
            print(f"使用{method}算法（串行）")
            result = minimize(
                fun=self.obj_function,
                x0=initial_values,
                method=method,
                bounds=bounds,
                options=options,
                callback=callback
            )
        
        self.result = result
        
        # 保存最终结果
        print("\n" + "="*80)
        print("优化完成")
        print("="*80)
        self._print_optimization_result(result)
        self._save_final_results(result)
        
        # 打印最优评估
        self.obj_function.print_best_evaluation()
        
        return result
    
    def resume_from_checkpoint(
        self, 
        checkpoint_path: Path
    ) -> OptimizeResult:
        """
        从断点恢复校准
        
        参数:
            checkpoint_path: 断点文件路径
        
        返回:
            OptimizeResult对象
        """
        print(f"\n从断点恢复: {checkpoint_path}")
        
        # 加载断点
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"断点信息:")
        print(f"  保存时间: {checkpoint['timestamp']}")
        print(f"  评估次数: {checkpoint['n_evaluations']}")
        print(f"  当前参数: {checkpoint['current_params']}")
        print(f"  最优SMM距离: {checkpoint['best_obj_value']:.6f}")
        
        # 重新创建MFG求解器和目标函数
        mfg_solver = self._create_mfg_solver()
        self.obj_function = ObjectiveFunction(
            self.target_moments,
            self.weight_matrix,
            mfg_solver,
            self.output_dir
        )
        
        # 恢复评估历史
        history_file = self.output_dir / 'calibration_history.csv'
        if history_file.exists():
            self.obj_function.load_history(history_file)
        
        # 从当前最优参数重新开始优化
        # 注意：scipy.optimize不支持完全恢复优化器状态
        # 这里简化为从最优参数重新开始
        initial_values = np.array(checkpoint['best_params'])
        
        return self.calibrate(initial_values=initial_values)
    
    def _save_checkpoint(
        self, 
        current_params: np.ndarray,
        result: Optional[OptimizeResult]
    ) -> None:
        """
        保存断点
        
        参数:
            current_params: 当前参数
            result: 优化结果（如果有）
        """
        best_eval = self.obj_function.get_best_evaluation()
        
        checkpoint = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_evaluations': self.obj_function.get_evaluation_count(),
            'current_params': current_params.tolist(),
            'current_obj_value': self.obj_function(current_params),
            'best_params': best_eval['params'],
            'best_obj_value': best_eval['smm_distance'],
            'result': result
        }
        
        # 保存到文件
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f'checkpoint_{timestamp_str}.pkl'
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # 更新最新断点链接
        latest_link = self.checkpoint_dir / 'checkpoint_latest.pkl'
        if latest_link.exists():
            latest_link.unlink()
        
        with open(latest_link, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # 清理旧断点
        self._cleanup_old_checkpoints()
        
        print(f"\n断点已保存: {checkpoint_file.name}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧断点文件（保留最近N个）"""
        keep_n = self.config['checkpoint']['keep_last_n']
        
        # 获取所有断点文件（排除latest链接）
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob('checkpoint_*.pkl') 
             if f.name != 'checkpoint_latest.pkl'],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # 删除多余的旧文件
        for old_checkpoint in checkpoints[keep_n:]:
            old_checkpoint.unlink()
            print(f"  已删除旧断点: {old_checkpoint.name}")
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """
        获取最新断点文件路径
        
        返回:
            断点文件路径，如果不存在则返回None
        """
        latest_link = self.checkpoint_dir / 'checkpoint_latest.pkl'
        
        if latest_link.exists():
            return latest_link
        else:
            return None
    
    def _print_optimization_result(self, result: OptimizeResult) -> None:
        """
        打印优化结果
        
        参数:
            result: scipy.optimize.OptimizeResult对象
        """
        print(f"优化状态: {'成功' if result.success else '失败'}")
        print(f"终止信息: {result.message}")
        print(f"函数评估次数: {result.nfev}")
        
        if hasattr(result, 'nit'):
            print(f"迭代次数: {result.nit}")
        
        print(f"\n最优参数:")
        param_names = self.param_utils.get_param_names()
        for name, value in zip(param_names, result.x):
            print(f"  {name}: {value:.6f}")
        
        print(f"\n最优SMM距离: {result.fun:.6f}")
    
    def _save_final_results(self, result: OptimizeResult) -> None:
        """
        保存最终校准结果
        
        参数:
            result: scipy.optimize.OptimizeResult对象
        """
        # 保存校准后的参数到YAML文件
        params_dict = self.param_utils.vector_to_dict(result.x)
        
        calibrated_params_file = self.output_dir / 'calibrated_parameters.yaml'
        
        output_data = {
            'calibration_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': self.config['optimization']['method'],
                'success': bool(result.success),
                'n_evaluations': result.nfev,
                'smm_distance': float(result.fun)
            },
            'parameters': params_dict
        }
        
        with open(calibrated_params_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, allow_unicode=True, 
                     default_flow_style=False)
        
        print(f"\n校准后的参数已保存至: {calibrated_params_file}")
        
        # 更新原始MFG配置文件（备份后）
        backup_path = self.mfg_config_path.parent / (
            self.mfg_config_path.stem + 
            f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
        )
        
        # 备份原配置
        import shutil
        shutil.copy(self.mfg_config_path, backup_path)
        print(f"原配置已备份至: {backup_path}")
        
        # 更新配置文件
        update_mfg_config_with_params(
            self.mfg_config_path,
            params_dict,
            self.param_utils,
            self.mfg_config_path
        )
        print(f"MFG配置已更新: {self.mfg_config_path}")
        
        # 保存优化结果对象
        result_file = self.output_dir / 'optimization_result.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"优化结果对象已保存至: {result_file}")

