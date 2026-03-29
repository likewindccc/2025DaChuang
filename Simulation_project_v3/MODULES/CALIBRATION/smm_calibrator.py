import os
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
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
        self.target_moments = TargetMoments(
            str(target_moments_path),
            selected_moments=self._extract_active_moment_names(),
        )
        
        # 初始化参数工具
        self.param_utils = OptimizationUtils(self.config)
        
        # 初始化权重矩阵（支持bootstrap逆方差）
        weight_type = self.config['target_moments']['weight_type']
        self.weight_matrix = create_weight_matrix(
            self.target_moments,
            weight_type=weight_type,
            custom_weights=self._extract_custom_moment_weights(),
            strict_bootstrap_se=bool(
                self.config['target_moments'].get('strict_bootstrap_se', False)
            )
        )
        
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
        
        # 目标函数实例（在calibrate中创建）
        self.obj_function = None
        
        # 优化结果
        self.result = None

        # 多步校准配置
        self.strategy_config = self.config.get('calibration_strategy', {})
        self.two_stage_config = self.config.get('two_stage_calibration', {})

        # Step1 参数分类结果（默认全内部）
        self.param_names = self.param_utils.get_param_names()
        self.parameter_partition = {
            'internal': self.param_names.copy(),
            'external': []
        }
        self._active_param_indices = list(range(len(self.param_names)))
        self._fixed_external_values = {}
        self._expand_params_for_history = None
    
    def _create_base_population_sample(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        预先生成并缓存一份基础人口样本，供整个校准过程复用。
        """
        from MODULES.MFG.equilibrium_solver import EquilibriumSolver

        base_solver = EquilibriumSolver(
            str(self.mfg_config_path),
            save_results=False
        )
        return base_solver.create_base_population_sample(verbose=False)

    def _create_mfg_solver(
        self,
        base_population: pd.DataFrame,
        base_initial_T: np.ndarray
    ) -> callable:
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
                    save_results=False,
                    base_population=base_population,
                    initial_T=base_initial_T,
                    verbose=False
                )
                
                return individuals, eq_info
                
            finally:
                # 确保删除临时配置文件（即使出错也要删除）
                if temp_config_path.exists():
                    temp_config_path.unlink()
        
        return mfg_solver

    def _extract_custom_moment_weights(self) -> Optional[Dict[str, float]]:
        """
        读取配置中各矩的手工权重（可选）。
        """
        moments_cfg = self.config.get('target_moments', {}).get('moments', [])
        if not isinstance(moments_cfg, list):
            return None

        custom_weights = {}
        for item in moments_cfg:
            if isinstance(item, dict) and 'name' in item and 'weight' in item:
                custom_weights[item['name']] = float(item['weight'])

        return custom_weights if custom_weights else None

    def _extract_active_moment_names(self) -> Optional[List[str]]:
        """
        读取当前校准阶段实际启用的目标矩列表。

        设计目标：
        1. 若 calibration_config.yaml 显式给出 moments 列表，则只加载这些矩；
        2. 若未给出，则默认回退为 target_moments.yaml 中的全部矩定义。
        """
        moments_cfg = self.config.get('target_moments', {}).get('moments', [])
        if not isinstance(moments_cfg, list) or len(moments_cfg) == 0:
            return None

        active_names = []
        for item in moments_cfg:
            if isinstance(item, dict) and 'name' in item:
                active_names.append(item['name'])

        return active_names if active_names else None

    def _resolve_workers(self, method: str, options: Dict) -> int:
        """
        解析 workers 配置。
        支持 auto/all/-1，并默认使用全部核心。
        """
        if method != 'differential_evolution':
            return 1

        workers_value = options.get('workers', 1)
        if isinstance(workers_value, str):
            workers_norm = workers_value.strip().lower()
            if workers_norm in {'auto', 'all', '-1'}:
                workers = os.cpu_count() or 1
            else:
                workers = int(workers_value)
        else:
            workers = int(workers_value)

        if workers <= 0:
            workers = os.cpu_count() or 1

        options['workers'] = workers
        return workers

    def _expand_to_full_vector(self, internal_vector: np.ndarray) -> np.ndarray:
        """
        将内部参数向量扩展为完整参数向量（外部参数固定）。
        """
        full_vector = np.zeros(len(self.param_names), dtype=float)
        for param_idx, param_name in enumerate(self.param_names):
            if param_idx in self._active_param_indices:
                internal_idx = self._active_param_indices.index(param_idx)
                full_vector[param_idx] = float(internal_vector[internal_idx])
            else:
                full_vector[param_idx] = float(self._fixed_external_values[param_name])
        return full_vector

    def _wrap_internal_solver(
        self,
        full_solver: Callable[[np.ndarray], Tuple[pd.DataFrame, Dict]]
    ) -> Callable[[np.ndarray], Tuple[pd.DataFrame, Dict]]:
        """
        把“完整参数求解器”包装为“内部参数求解器”。
        """
        def solver_internal(internal_params: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
            full_params = self._expand_to_full_vector(
                np.asarray(internal_params, dtype=float)
            )
            return full_solver(full_params)

        return solver_internal

    def _build_internal_bounds(self) -> List[Tuple[float, float]]:
        """
        获取内部参数对应的边界列表。
        """
        all_bounds = self.param_utils.get_parameter_bounds()
        return [all_bounds[i] for i in self._active_param_indices]

    def _run_jacobian_analysis(
        self,
        full_solver: Callable[[np.ndarray], Tuple[pd.DataFrame, Dict]],
        base_params: np.ndarray
    ) -> pd.DataFrame:
        """
        Step 0: Jacobian敏感性预分析（数值微扰法）。
        """
        step0_cfg = self.strategy_config.get('step0_jacobian', {})
        relative_step = float(step0_cfg.get('relative_step', 0.03))
        if relative_step <= 0:
            raise ValueError("step0_jacobian.relative_step 必须为正数")

        bounds = self.param_utils.get_parameter_bounds()
        target_vec = self.target_moments.get_target_vector()
        moment_names = self.target_moments.get_moment_names()

        n_params = len(self.param_names)
        n_moments = len(moment_names)

        jacobian = np.zeros((n_moments, n_params), dtype=float)
        elasticity = np.zeros((n_moments, n_params), dtype=float)

        for idx, param_name in enumerate(self.param_names):
            lower, upper = bounds[idx]
            span = max(upper - lower, 1.0e-8)
            step = relative_step * span

            plus_params = base_params.copy()
            minus_params = base_params.copy()
            plus_params[idx] = np.clip(base_params[idx] + step, lower, upper)
            minus_params[idx] = np.clip(base_params[idx] - step, lower, upper)

            denominator = plus_params[idx] - minus_params[idx]
            if abs(denominator) < 1.0e-12:
                continue

            individuals_plus, eq_plus = full_solver(plus_params)
            individuals_minus, eq_minus = full_solver(minus_params)

            m_plus = self.target_moments.get_simulated_vector(individuals_plus, eq_plus)
            m_minus = self.target_moments.get_simulated_vector(individuals_minus, eq_minus)

            gradient = (m_plus - m_minus) / denominator
            jacobian[:, idx] = gradient

            param_scale = max(abs(base_params[idx]), 1.0)
            moment_scale = np.maximum(np.abs(target_vec), 1.0e-8)
            elasticity[:, idx] = gradient * (param_scale / moment_scale)

            print(f"Jacobian完成: {param_name}, step={denominator:.6g}")

        rows = []
        for idx, param_name in enumerate(self.param_names):
            elastic_col = elasticity[:, idx]
            rows.append({
                'param_name': param_name,
                'max_abs_elasticity': float(np.max(np.abs(elastic_col))),
                'mean_abs_elasticity': float(np.mean(np.abs(elastic_col)))
            })

        summary_df = pd.DataFrame(rows).sort_values('max_abs_elasticity', ascending=True)
        for moment_idx, moment_name in enumerate(moment_names):
            summary_df[f'd_{moment_name}'] = jacobian[moment_idx, :]
            summary_df[f'elastic_{moment_name}'] = elasticity[moment_idx, :]

        output_file = Path(
            step0_cfg.get('output_file', str(self.output_dir / 'jacobian_analysis.csv'))
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        np.save(self.output_dir / 'jacobian_matrix.npy', jacobian)
        np.save(self.output_dir / 'jacobian_elasticity.npy', elasticity)

        print(f"Step0 Jacobian结果已保存: {output_file}")
        return summary_df

    def _classify_parameters(self, jacobian_summary: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Step 1: 基于敏感性阈值将参数分为外部/内部两类。
        """
        step1_cfg = self.strategy_config.get('step1_classification', {})
        threshold = float(step1_cfg.get('weak_sensitivity_threshold', 0.03))

        external_override = set(step1_cfg.get('external_params_override', []))
        internal_override = set(step1_cfg.get('internal_params_override', []))

        if external_override or internal_override:
            internal_set = set(internal_override)
            external_set = set(external_override)
            for name in self.param_names:
                if name not in internal_set and name not in external_set:
                    internal_set.add(name)
        else:
            internal_set = set()
            external_set = set()
            for _, row in jacobian_summary.iterrows():
                name = row['param_name']
                sensitivity = float(row['max_abs_elasticity'])
                if sensitivity < threshold:
                    external_set.add(name)
                else:
                    internal_set.add(name)

            if len(internal_set) == 0:
                strongest = jacobian_summary.sort_values(
                    'max_abs_elasticity',
                    ascending=False
                ).iloc[0]['param_name']
                internal_set.add(strongest)
                external_set.discard(strongest)

        partition = {
            'internal': [name for name in self.param_names if name in internal_set],
            'external': [name for name in self.param_names if name in external_set],
            'threshold': threshold
        }

        partition_file = self.output_dir / 'parameter_partition.yaml'
        with open(partition_file, 'w', encoding='utf-8') as f:
            yaml.dump(partition, f, allow_unicode=True, default_flow_style=False)

        print("Step1 参数分类完成:")
        print(f"  internal: {partition['internal']}")
        print(f"  external: {partition['external']}")
        return partition

    def _build_weight_matrix(
        self,
        weight_type: str,
        covariance_matrix: Optional[np.ndarray] = None,
        strict_bootstrap_se: bool = False,
        regularization: float = 1.0e-8
    ) -> np.ndarray:
        """
        构建权重矩阵（支持bootstrap逆方差和协方差逆）。
        """
        return create_weight_matrix(
            self.target_moments,
            weight_type=weight_type,
            custom_weights=self._extract_custom_moment_weights(),
            covariance_matrix=covariance_matrix,
            regularization=regularization,
            strict_bootstrap_se=strict_bootstrap_se
        )

    def _estimate_moment_covariance(
        self,
        internal_solver: Callable[[np.ndarray], Tuple[pd.DataFrame, Dict]],
        internal_params: np.ndarray,
        n_replicates: int
    ) -> np.ndarray:
        """
        估计矩差异的协方差矩阵（Step 5使用）。
        """
        n_replicates = max(2, int(n_replicates))
        diffs = []

        print(f"估计矩协方差: 重复求解 {n_replicates} 次")
        for rep in range(n_replicates):
            individuals, eq_info = internal_solver(internal_params)
            diff_vec = self.target_moments.compute_moment_difference(individuals, eq_info)
            diffs.append(diff_vec)
            print(f"  replicate {rep + 1}/{n_replicates} 完成")

        diff_matrix = np.vstack(diffs)
        covariance = np.cov(diff_matrix, rowvar=False, ddof=1)

        moment_names = self.target_moments.get_moment_names()
        pd.DataFrame(
            covariance,
            index=moment_names,
            columns=moment_names
        ).to_csv(self.output_dir / 'moment_covariance_stage5.csv', encoding='utf-8-sig')

        pd.DataFrame(
            diff_matrix,
            columns=moment_names
        ).to_csv(self.output_dir / 'moment_diff_draws_stage5.csv', index=False, encoding='utf-8-sig')

        return covariance

    def _optimize_stage(
        self,
        stage_name: str,
        method: str,
        options: Dict,
        internal_solver: Callable[[np.ndarray], Tuple[pd.DataFrame, Dict]],
        weight_matrix: np.ndarray,
        initial_internal: np.ndarray,
        internal_bounds: List[Tuple[float, float]]
    ) -> OptimizeResult:
        """
        执行单阶段优化（Step4/Step5通用）。
        """
        local_options = options.copy()
        n_workers = self._resolve_workers(method, local_options)
        use_pid_history = (method == 'differential_evolution' and n_workers > 1)

        print("\n" + "=" * 80)
        print(f"开始{stage_name}")
        print("=" * 80)
        print(f"优化方法: {method}")
        print(f"并行进程数: {n_workers}")
        print(f"初始内部参数: {initial_internal}")

        self.obj_function = ObjectiveFunction(
            self.target_moments,
            weight_matrix,
            internal_solver,
            self.output_dir,
            use_pid_history=use_pid_history
        )
        self._expand_params_for_history = self._expand_to_full_vector

        def callback(xk):
            if not self.checkpoint_enabled:
                return
            save_freq = int(self.config['checkpoint']['save_frequency'])
            n_eval = self.obj_function.get_evaluation_count()
            if n_eval > 0 and save_freq > 0 and n_eval % save_freq == 0:
                full_x = self._expand_to_full_vector(np.asarray(xk, dtype=float))
                self._save_checkpoint(full_x, None)

        if method == 'differential_evolution':
            de_callback_counter = {'count': 0}

            def de_callback(xk, convergence=0):
                _ = convergence
                de_callback_counter['count'] += 1
                if self.checkpoint_enabled:
                    save_freq = int(self.config['checkpoint']['save_frequency'])
                    if save_freq > 0 and de_callback_counter['count'] % save_freq == 0:
                        full_x = self._expand_to_full_vector(np.asarray(xk, dtype=float))
                        self._save_checkpoint(full_x, None)
                return False

            de_valid_params = {
                'maxiter', 'popsize', 'atol', 'tol', 'workers',
                'updating', 'polish', 'strategy', 'recombination',
                'mutation', 'seed', 'init', 'disp'
            }
            de_options = {
                key: value for key, value in local_options.items()
                if key in de_valid_params
            }

            if n_workers > 1:
                from loky import get_reusable_executor
                executor = get_reusable_executor(max_workers=n_workers)
                de_options['workers'] = executor.map

            result = differential_evolution(
                func=self.obj_function,
                bounds=internal_bounds,
                x0=initial_internal,
                callback=de_callback,
                **de_options
            )
        else:
            result = minimize(
                fun=self.obj_function,
                x0=initial_internal,
                method=method,
                bounds=internal_bounds,
                options=local_options,
                callback=callback
            )

        print(f"{stage_name}完成: success={result.success}, fun={float(result.fun):.6f}")
        return result

    def _expand_result_to_full(self, internal_result: OptimizeResult) -> OptimizeResult:
        """
        将内部参数优化结果映射为完整参数结果。
        """
        full_result = OptimizeResult(internal_result)
        full_result.x = self._expand_to_full_vector(
            np.asarray(internal_result.x, dtype=float)
        )
        return full_result
    
    def calibrate(
        self, 
        method: Optional[str] = None,
        initial_values: Optional[np.ndarray] = None,
        allow_auto_resume: bool = True
    ) -> OptimizeResult:
        """
        执行SMM校准
        
        参数:
            method: 优化方法（如果为None则使用配置文件中的方法）
            initial_values: 初始参数值（如果为None则使用配置文件中的初始值）
            allow_auto_resume: 是否允许自动断点恢复
        
        返回:
            scipy.optimize.OptimizeResult对象
        """
        print("\n" + "="*80)
        print("开始SMM校准")
        print("="*80)
        print(f"配置文件: {self.config_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查是否有断点可以恢复（无交互自动恢复）
        if (allow_auto_resume and self.checkpoint_enabled and
                self.config['checkpoint'].get('auto_resume', False)):
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is not None:
                print(f"\n发现断点文件: {checkpoint_path}")
                print("检测到 auto_resume=true，自动从断点恢复。")
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

        # 确定完整参数初始值
        if initial_values is None:
            initial_values = self.param_utils.get_initial_values('baseline')
        full_initial = np.asarray(initial_values, dtype=float)
        full_initial = self.param_utils.clip_to_bounds(full_initial)

        print(f"\n优化方法: {method}")
        print(f"完整初始参数: {full_initial}")

        # 创建“完整参数求解器”
        print("\n初始化MFG求解器...")
        base_population, base_initial_T = self._create_base_population_sample()
        full_solver = self._create_mfg_solver(base_population, base_initial_T)

        # Step0: Jacobian敏感性分析
        step0_enabled = bool(
            self.strategy_config.get('step0_jacobian', {}).get('enabled', True)
        )
        if step0_enabled:
            jacobian_summary = self._run_jacobian_analysis(full_solver, full_initial)
        else:
            jacobian_summary = pd.DataFrame({
                'param_name': self.param_names,
                'max_abs_elasticity': np.zeros(len(self.param_names)),
                'mean_abs_elasticity': np.zeros(len(self.param_names))
            })

        # Step1: 参数分类
        step1_enabled = bool(
            self.strategy_config.get('step1_classification', {}).get('enabled', True)
        )
        if step1_enabled:
            self.parameter_partition = self._classify_parameters(jacobian_summary)
        else:
            self.parameter_partition = {
                'internal': self.param_names.copy(),
                'external': []
            }

        self._active_param_indices = [
            self.param_names.index(name)
            for name in self.parameter_partition['internal']
        ]
        self._fixed_external_values = {
            name: float(full_initial[self.param_names.index(name)])
            for name in self.parameter_partition['external']
        }

        if len(self._active_param_indices) == 0:
            raise ValueError("内部校准参数为空，请检查Step1配置")

        internal_initial = full_initial[self._active_param_indices]
        internal_bounds = self._build_internal_bounds()
        internal_solver = self._wrap_internal_solver(full_solver)

        # Step4: 鲁棒阶段（bootstrap逆方差）
        options = self.config['optimization']['options'].copy()
        stage4_cfg = self.two_stage_config.get('stage4', {})
        stage4_weight_type = stage4_cfg.get(
            'weight_type',
            self.config['target_moments'].get('weight_type', 'inverse_variance_bootstrap')
        )
        weight_stage4 = self._build_weight_matrix(
            weight_type=stage4_weight_type,
            strict_bootstrap_se=bool(stage4_cfg.get('strict_bootstrap_se', False))
        )
        self.weight_matrix = weight_stage4

        result_stage4 = self._optimize_stage(
            stage_name=stage4_cfg.get('name', 'Step4_鲁棒阶段'),
            method=method,
            options=options,
            internal_solver=internal_solver,
            weight_matrix=weight_stage4,
            initial_internal=internal_initial,
            internal_bounds=internal_bounds
        )

        final_internal_result = result_stage4
        final_weight_matrix = weight_stage4

        # Step5: 高效阶段（协方差逆）
        stage5_cfg = self.two_stage_config.get('stage5', {})
        stage5_enabled = bool(self.two_stage_config.get('enabled', False)) and bool(
            stage5_cfg.get('enabled', False)
        )
        if stage5_enabled:
            covariance = self._estimate_moment_covariance(
                internal_solver=internal_solver,
                internal_params=np.asarray(result_stage4.x, dtype=float),
                n_replicates=int(stage5_cfg.get('covariance_replicates', 8))
            )
            weight_stage5 = self._build_weight_matrix(
                weight_type=stage5_cfg.get('weight_type', 'efficient_from_covariance'),
                covariance_matrix=covariance,
                regularization=float(stage5_cfg.get('covariance_regularization', 1.0e-8))
            )
            self.weight_matrix = weight_stage5

            stage5_method = stage5_cfg.get('method', method)
            stage5_options = options.copy()
            stage5_options.update(stage5_cfg.get('options', {}))

            result_stage5 = self._optimize_stage(
                stage_name=stage5_cfg.get('name', 'Step5_高效阶段'),
                method=stage5_method,
                options=stage5_options,
                internal_solver=internal_solver,
                weight_matrix=weight_stage5,
                initial_internal=np.asarray(result_stage4.x, dtype=float),
                internal_bounds=internal_bounds
            )
            final_internal_result = result_stage5
            final_weight_matrix = weight_stage5

        result = self._expand_result_to_full(final_internal_result)
        self.result = result
        self.weight_matrix = final_weight_matrix
        self._save_stage_summary(result_stage4, final_internal_result)
        
        # 保存最终结果
        print("\n" + "="*80)
        print("优化完成")
        print("="*80)
        self._print_optimization_result(result)
        self._save_final_results(result)
        
        # 打印最优评估
        self.obj_function.print_best_evaluation()
        
        return result

    def _save_stage_summary(
        self,
        stage4_result: OptimizeResult,
        final_internal_result: OptimizeResult
    ) -> None:
        """
        保存Step0-Step5的摘要信息，便于长期维护追踪。
        """
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameter_partition': self.parameter_partition,
            'active_param_indices': self._active_param_indices,
            'fixed_external_values': self._fixed_external_values,
            'stage4': {
                'success': bool(stage4_result.success),
                'fun': float(stage4_result.fun),
                'nfev': int(getattr(stage4_result, 'nfev', 0))
            },
            'final_internal_stage': {
                'success': bool(final_internal_result.success),
                'fun': float(final_internal_result.fun),
                'nfev': int(getattr(final_internal_result, 'nfev', 0))
            }
        }

        summary_file = self.output_dir / 'calibration_stage_summary.yaml'
        with open(summary_file, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
    
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
        best_obj_value = checkpoint.get('best_obj_value')
        if best_obj_value is None:
            print("  最优SMM距离: 暂无")
        else:
            print(f"  最优SMM距离: {best_obj_value:.6f}")
        
        # 重新创建MFG求解器和目标函数
        base_population, base_initial_T = self._create_base_population_sample()
        mfg_solver = self._create_mfg_solver(base_population, base_initial_T)
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
        
        return self.calibrate(
            initial_values=initial_values,
            allow_auto_resume=False
        )
    
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
        if (self._expand_params_for_history is not None and
                len(current_params) != len(self.param_names)):
            current_params = self._expand_params_for_history(
                np.asarray(current_params, dtype=float)
            )

        best_eval = None
        if self.obj_function is not None:
            best_eval = self.obj_function.get_best_evaluation()

        if best_eval is not None:
            best_params_arr = np.asarray(best_eval['params'], dtype=float)
            if (self._expand_params_for_history is not None and
                    len(best_params_arr) != len(self.param_names)):
                best_params_arr = self._expand_params_for_history(best_params_arr)
            best_params = best_params_arr.tolist()
            best_obj_value = float(best_eval['smm_distance'])
            current_obj_value = best_obj_value
        elif result is not None and hasattr(result, 'x') and hasattr(result, 'fun'):
            best_params = np.asarray(result.x, dtype=float).tolist()
            best_obj_value = float(result.fun)
            current_obj_value = best_obj_value
        else:
            best_params = np.asarray(current_params, dtype=float).tolist()
            best_obj_value = None
            current_obj_value = None

        n_evaluations = (
            self.obj_function.get_evaluation_count()
            if self.obj_function is not None
            else 0
        )

        checkpoint = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_evaluations': n_evaluations,
            'current_params': np.asarray(current_params, dtype=float).tolist(),
            'current_obj_value': current_obj_value,
            'best_params': best_params,
            'best_obj_value': best_obj_value,
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
        if len(result.x) == len(param_names):
            for name, value in zip(param_names, result.x):
                print(f"  {name}: {value:.6f}")
        else:
            for idx, value in enumerate(result.x):
                print(f"  param_{idx}: {value:.6f}")
        
        print(f"\n最优SMM距离: {result.fun:.6f}")
        print(f"内部参数: {self.parameter_partition.get('internal', [])}")
        print(f"外部参数: {self.parameter_partition.get('external', [])}")
    
    def _save_final_results(self, result: OptimizeResult) -> None:
        """
        保存最终校准结果
        
        参数:
            result: scipy.optimize.OptimizeResult对象
        """
        # 保存校准后的参数到YAML文件
        params_dict = self.param_utils.vector_to_dict(
            np.asarray(result.x, dtype=float)
        )
        
        calibrated_params_file = self.output_dir / 'calibrated_parameters.yaml'
        
        output_data = {
            'calibration_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': self.config['optimization']['method'],
                'success': bool(result.success),
                'n_evaluations': result.nfev,
                'smm_distance': float(result.fun)
            },
            'parameter_partition': self.parameter_partition,
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

