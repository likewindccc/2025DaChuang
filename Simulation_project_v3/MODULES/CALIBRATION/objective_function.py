import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable

from .target_moments import TargetMoments


class ObjectiveFunction:
    """
    SMM目标函数类
    
    功能：
    1. 计算SMM距离：J(θ) = [m_sim - m_target]' * W * [m_sim - m_target]
    2. 管理权重矩阵W
    3. 记录每次函数评估的结果
    4. 提供回调接口（用于监控优化过程）
    
    属性：
        target_moments: 目标矩管理器
        weight_matrix: 权重矩阵（n_moments × n_moments）
        mfg_solver_func: MFG求解函数
        evaluation_history: 评估历史记录列表
        n_evaluations: 函数评估次数
    """
    
    def __init__(
        self,
        target_moments: TargetMoments,
        weight_matrix: np.ndarray,
        mfg_solver_func: Callable,
        output_dir: Path
    ):
        """
        初始化SMM目标函数
        
        参数:
            target_moments: TargetMoments实例
            weight_matrix: 权重矩阵，形状为(n_moments, n_moments)
            mfg_solver_func: MFG求解函数，签名为func(params_dict) -> (individuals, eq_info)
            output_dir: 输出目录（用于保存评估历史）
        """
        self.target_moments = target_moments
        self.weight_matrix = weight_matrix
        self.mfg_solver_func = mfg_solver_func
        self.output_dir = output_dir
        
        # 初始化评估历史
        self.evaluation_history = []
        self.n_evaluations = 0
        
        # 验证权重矩阵维度
        n_moments = target_moments.get_n_moments()
        expected_shape = (n_moments, n_moments)
        
        if weight_matrix.shape != expected_shape:
            raise ValueError(
                f"权重矩阵形状不匹配：期望 {expected_shape}, "
                f"实际 {weight_matrix.shape}"
            )
    
    def __call__(self, params_vector: np.ndarray) -> float:
        """
        计算SMM目标函数值
        
        这是优化器调用的主接口
        
        参数:
            params_vector: 参数向量，形状为(n_params,)
        
        返回:
            SMM距离（标量）
        
        流程：
        1. 将参数向量转换为字典
        2. 调用MFG求解函数
        3. 计算模拟矩
        4. 计算SMM距离
        5. 记录评估结果
        """
        self.n_evaluations += 1
        
        print(f"\n{'='*80}")
        print(f"目标函数评估 #{self.n_evaluations}")
        print(f"{'='*80}")
        print(f"参数向量: {params_vector}")
        
        # 步骤1: 运行MFG求解
        individuals, eq_info = self.mfg_solver_func(params_vector)
        
        # 步骤2: 计算矩差异向量
        moment_diff = self.target_moments.compute_moment_difference(
            individuals, 
            eq_info
        )
        
        # 步骤3: 计算SMM距离
        # J = (m_sim - m_target)' * W * (m_sim - m_target)
        smm_distance = float(
            moment_diff.T @ self.weight_matrix @ moment_diff
        )
        
        # 步骤4: 记录评估结果
        self._record_evaluation(
            params_vector, 
            moment_diff, 
            smm_distance,
            individuals,
            eq_info
        )
        
        # 步骤5: 打印评估摘要
        print(f"\n结果:")
        print(f"  SMM距离: {smm_distance:.6f}")
        print(f"  矩差异范数: {np.linalg.norm(moment_diff):.6f}")
        
        # 打印矩对比
        self.target_moments.print_moment_comparison(individuals, eq_info)
        
        return smm_distance
    
    def _record_evaluation(
        self,
        params_vector: np.ndarray,
        moment_diff: np.ndarray,
        smm_distance: float,
        individuals: pd.DataFrame,
        eq_info: Dict
    ) -> None:
        """
        记录单次函数评估结果
        
        参数:
            params_vector: 参数向量
            moment_diff: 矩差异向量
            smm_distance: SMM距离
            individuals: 个体均衡状态
            eq_info: 均衡信息
        """
        # 计算模拟矩
        simulated_moments = self.target_moments.compute_simulated_moments(
            individuals, 
            eq_info
        )
        
        # 构建评估记录
        record = {
            'evaluation_id': self.n_evaluations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': params_vector.tolist(),
            'smm_distance': smm_distance,
            'moment_diff_norm': float(np.linalg.norm(moment_diff)),
            'simulated_moments': simulated_moments,
            'converged': eq_info.get('converged', False),
            'iterations': eq_info.get('iterations', 0)
        }
        
        # 添加到历史记录
        self.evaluation_history.append(record)
        
        # 每10次评估保存一次历史（使用进程ID避免并发写入冲突）
        if self.n_evaluations % 10 == 0:
            self._save_history(use_pid=True)
    
    def _save_history(self, use_pid: bool = False) -> None:
        """
        保存评估历史到CSV文件
        
        参数:
            use_pid: 是否在文件名中使用进程ID（并行模式下避免冲突）
        """
        import os
        
        if use_pid:
            # 并行模式：每个进程写入自己的文件
            history_file = self.output_dir / f'calibration_history_{os.getpid()}.csv'
        else:
            # 串行模式：写入同一个文件
            history_file = self.output_dir / 'calibration_history.csv'
        
        # 将历史记录转换为DataFrame
        history_data = []
        
        for record in self.evaluation_history:
            row = {
                'evaluation_id': record['evaluation_id'],
                'timestamp': record['timestamp'],
                'smm_distance': record['smm_distance'],
                'moment_diff_norm': record['moment_diff_norm'],
                'converged': record['converged'],
                'iterations': record['iterations']
            }
            
            # 添加参数
            for i, param_val in enumerate(record['params']):
                row[f'param_{i}'] = param_val
            
            # 添加模拟矩
            for moment_name, moment_val in record['simulated_moments'].items():
                row[f'sim_{moment_name}'] = moment_val
            
            history_data.append(row)
        
        df = pd.DataFrame(history_data)
        df.to_csv(history_file, index=False, encoding='utf-8-sig')
        
        if not use_pid:
            print(f"\n评估历史已保存至: {history_file}")
    
    def get_moment_difference(
        self, 
        individuals: pd.DataFrame, 
        eq_info: Dict
    ) -> np.ndarray:
        """
        获取矩差异向量（不触发函数评估计数）
        
        参数:
            individuals: 个体均衡状态
            eq_info: 均衡信息
        
        返回:
            矩差异向量
        """
        return self.target_moments.compute_moment_difference(
            individuals, 
            eq_info
        )
    
    def get_evaluation_count(self) -> int:
        """
        获取函数评估次数
        
        返回:
            评估次数
        """
        return self.n_evaluations
    
    def get_best_evaluation(self) -> Optional[Dict]:
        """
        获取历史最优评估结果
        
        返回:
            最优评估记录字典，如果历史为空则返回None
        """
        if not self.evaluation_history:
            return None
        
        # 找到SMM距离最小的评估
        best_record = min(
            self.evaluation_history,
            key=lambda r: r['smm_distance']
        )
        
        return best_record
    
    def print_best_evaluation(self) -> None:
        """
        打印历史最优评估结果
        """
        best_record = self.get_best_evaluation()
        
        if best_record is None:
            print("尚无评估历史")
            return
        
        print("\n" + "="*80)
        print("历史最优评估")
        print("="*80)
        print(f"评估ID: {best_record['evaluation_id']}")
        print(f"时间: {best_record['timestamp']}")
        print(f"SMM距离: {best_record['smm_distance']:.6f}")
        print(f"矩差异范数: {best_record['moment_diff_norm']:.6f}")
        print(f"\n参数:")
        for i, param_val in enumerate(best_record['params']):
            print(f"  param_{i}: {param_val:.6f}")
        print(f"\n模拟矩:")
        for moment_name, moment_val in best_record['simulated_moments'].items():
            target_val = self.target_moments.get_target_moments()[moment_name]
            diff = moment_val - target_val
            print(f"  {moment_name}: {moment_val:.4f} "
                  f"(目标: {target_val:.4f}, 差异: {diff:.4f})")
        print("="*80)
    
    def reset_evaluation_count(self) -> None:
        """
        重置评估计数（用于断点续跑）
        """
        self.n_evaluations = 0
        self.evaluation_history = []
    
    def load_history(self, history_file: Path) -> None:
        """
        从文件加载评估历史（用于断点续跑）
        
        参数:
            history_file: 历史文件路径
        """
        df = pd.read_csv(history_file)
        
        # 重建评估历史
        self.evaluation_history = []
        
        for _, row in df.iterrows():
            # 提取参数
            param_cols = [col for col in df.columns if col.startswith('param_')]
            params = [row[col] for col in sorted(param_cols)]
            
            # 提取模拟矩
            sim_cols = [col for col in df.columns if col.startswith('sim_')]
            simulated_moments = {}
            for col in sim_cols:
                moment_name = col[4:]  # 去掉'sim_'前缀
                simulated_moments[moment_name] = row[col]
            
            # 构建记录
            record = {
                'evaluation_id': int(row['evaluation_id']),
                'timestamp': row['timestamp'],
                'params': params,
                'smm_distance': row['smm_distance'],
                'moment_diff_norm': row['moment_diff_norm'],
                'simulated_moments': simulated_moments,
                'converged': bool(row['converged']),
                'iterations': int(row['iterations'])
            }
            
            self.evaluation_history.append(record)
        
        # 更新评估计数
        self.n_evaluations = len(self.evaluation_history)
        
        print(f"已加载 {self.n_evaluations} 条评估历史")


def create_weight_matrix(
    n_moments: int, 
    weight_type: str = 'identity'
) -> np.ndarray:
    """
    创建权重矩阵
    
    参数:
        n_moments: 矩的数量
        weight_type: 权重类型
            - 'identity': 单位矩阵（等权重）
            - 'diagonal': 对角矩阵（可自定义权重）
            - 'optimal': 最优权重矩阵（需要矩的协方差矩阵，暂不实现）
    
    返回:
        权重矩阵，形状为(n_moments, n_moments)
    """
    if weight_type == 'identity':
        return np.eye(n_moments)
    
    elif weight_type == 'diagonal':
        # 暂时返回单位矩阵，后续可从配置文件读取自定义权重
        return np.eye(n_moments)
    
    else:
        raise ValueError(f"不支持的权重类型: {weight_type}")

