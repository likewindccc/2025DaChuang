"""
MFG模拟器（主控制器）

整合贝尔曼求解器和KFE求解器，实现完整的平均场博弈均衡求解。

核心功能：
1. 初始化稀疏网格、贝尔曼求解器、KFE求解器
2. 交替迭代求解贝尔曼方程和KFE
3. 检查三重收敛标准（V, a, u）
4. 保存和可视化结果

算法流程：
1. 初始化: 设置网格、价值函数、人口分布
2. 主循环（直到收敛或达到最大迭代次数）:
   a. 固定人口分布，求解贝尔曼方程 → 得到V*, a*
   b. 固定策略a*，演化KFE → 更新人口分布
   c. 计算失业率和市场紧张度θ
   d. 检查收敛
3. 输出均衡结果

Author: AI Assistant
Date: 2025-10-03
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from pathlib import Path
import time

from .sparse_grid import create_mfg_sparse_grid, SparseGrid
from .bellman_solver import BellmanSolver
from .kfe_solver import KFESolver
from .state_space import StateSpace

logger = logging.getLogger(__name__)


class MFGSimulator:
    """
    MFG模拟器（主控制器）
    
    整合所有MFG求解组件，实现完整的均衡求解流程。
    
    Attributes:
        config: 完整配置字典
        state_space: 状态空间管理器
        sparse_grid: 稀疏网格
        bellman_solver: 贝尔曼求解器
        kfe_solver: KFE求解器
        match_func_params: 匹配函数参数
    """
    
    def __init__(
        self,
        config: Dict,
        match_func_params: Optional[np.ndarray] = None
    ):
        """
        初始化MFG模拟器
        
        Args:
            config: 完整配置字典（包含所有模块配置）
            match_func_params: 匹配函数参数（来自Module 3估计）
        """
        self.config = config
        self.match_func_params = match_func_params
        
        logger.info("=" * 60)
        logger.info("初始化MFG模拟器...")
        logger.info("=" * 60)
        
        # 1. 创建状态空间
        self.state_space = StateSpace(
            T_range=tuple(config['state_space']['ranges']['T']),
            S_range=tuple(config['state_space']['ranges']['S']),
            D_range=tuple(config['state_space']['ranges']['D']),
            W_range=tuple(config['state_space']['ranges']['W'])
        )
        logger.info(f"✅ 状态空间: {self.state_space}")
        
        # 2. 生成稀疏网格
        bounds = self.state_space.get_bounds_for_grid()
        level = config['sparse_grid']['level']
        
        self.sparse_grid = create_mfg_sparse_grid(bounds, level=level)
        logger.info(
            f"✅ 稀疏网格: {self.sparse_grid.n_points}个点 "
            f"(level={level}, 效率={self.sparse_grid.efficiency:.2%})"
        )
        
        # 3. 初始化贝尔曼求解器
        self.bellman_solver = BellmanSolver(
            grid_nodes=self.sparse_grid.get_nodes(),
            config=config,
            match_func_params=match_func_params
        )
        logger.info(f"✅ 贝尔曼求解器: {self.bellman_solver}")
        
        # 4. 初始化KFE求解器
        initial_u_rate = config['initialization']['unemployment_rate']
        self.kfe_solver = KFESolver(
            grid_nodes=self.sparse_grid.get_nodes(),
            config=config,
            match_func_params=match_func_params,
            initial_unemployment_rate=initial_u_rate
        )
        logger.info(f"✅ KFE求解器: {self.kfe_solver}")
        
        # 收敛参数
        self.epsilon_V = config['convergence']['epsilon_V']
        self.epsilon_a = config['convergence']['epsilon_a']
        self.epsilon_u = config['convergence']['epsilon_u']
        self.max_iterations = config['convergence']['max_iterations']
        
        # 历史记录
        self.history = {
            'V_U': [],
            'V_E': [],
            'a_star': [],
            'unemployment_rate': [],
            'iterations': 0,
            'convergence_metrics': []
        }
        
        logger.info("=" * 60)
        logger.info("✅ MFG模拟器初始化完成！")
        logger.info("=" * 60)
    
    def solve(self, verbose: bool = True) -> Dict:
        """
        求解MFG均衡
        
        主循环：交替求解贝尔曼方程和KFE，直到收敛。
        
        Args:
            verbose: 是否输出详细日志
        
        Returns:
            包含均衡结果的字典：
            - 'V_U': 失业价值函数
            - 'V_E': 就业价值函数
            - 'a_star': 最优努力策略
            - 'm_U': 失业者分布
            - 'm_E': 就业者分布
            - 'unemployment_rate': 均衡失业率
            - 'converged': 是否收敛
            - 'n_iterations': 迭代次数
            - 'history': 历史记录
        """
        logger.info("\n" + "=" * 60)
        logger.info("开始求解MFG均衡...")
        logger.info("=" * 60)
        
        start_time = time.time()
        converged = False
        
        # 初始值
        V_U_old = self.bellman_solver.V_U.copy()
        V_E_old = self.bellman_solver.V_E.copy()
        a_star_old = self.bellman_solver.a_star.copy()
        u_rate_old = self.kfe_solver.get_unemployment_rate()
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # ===== 步骤1: 求解贝尔曼方程 =====
            bellman_result = self.bellman_solver.solve(verbose=False)
            V_U_new = bellman_result['V_U']
            V_E_new = bellman_result['V_E']
            a_star_new = bellman_result['a_star']
            
            # ===== 步骤2: 演化KFE =====
            kfe_n_steps = self.config['kfe']['n_evolution_steps']
            kfe_result = self.kfe_solver.step(a_star_new, n_steps=kfe_n_steps)
            u_rate_new = kfe_result['unemployment_rate']
            
            # ===== 步骤3: 检查收敛 =====
            # 1) 价值函数收敛
            diff_V_U = np.max(np.abs(V_U_new - V_U_old))
            diff_V_E = np.max(np.abs(V_E_new - V_E_old))
            diff_V = max(diff_V_U, diff_V_E)
            
            # 2) 策略收敛
            diff_a = np.max(np.abs(a_star_new - a_star_old))
            
            # 3) 失业率收敛
            diff_u = abs(u_rate_new - u_rate_old)
            
            # 记录历史
            self.history['V_U'].append(V_U_new.copy())
            self.history['V_E'].append(V_E_new.copy())
            self.history['a_star'].append(a_star_new.copy())
            self.history['unemployment_rate'].append(u_rate_new)
            self.history['convergence_metrics'].append({
                'diff_V': diff_V,
                'diff_a': diff_a,
                'diff_u': diff_u
            })
            
            iter_time = time.time() - iter_start
            
            # 输出进度
            if verbose and (iteration % 10 == 0 or iteration < 5):
                logger.info(
                    f"迭代 {iteration:3d}: "
                    f"diff_V={diff_V:.2e}, diff_a={diff_a:.2e}, diff_u={diff_u:.2e} | "
                    f"u_rate={u_rate_new:.4f}, "
                    f"V_U_mean={np.mean(V_U_new):.1f}, "
                    f"a_mean={np.mean(a_star_new):.3f} | "
                    f"{iter_time:.2f}s"
                )
            
            # 检查三重收敛标准
            if (diff_V < self.epsilon_V and 
                diff_a < self.epsilon_a and 
                diff_u < self.epsilon_u):
                converged = True
                break
            
            # 更新旧值
            V_U_old = V_U_new
            V_E_old = V_E_new
            a_star_old = a_star_new
            u_rate_old = u_rate_new
        
        total_time = time.time() - start_time
        self.history['iterations'] = iteration + 1
        
        # 输出结果
        logger.info("=" * 60)
        if converged:
            logger.info(f"✅ MFG均衡在 {iteration+1} 次迭代后收敛！")
        else:
            logger.warning(f"⚠️ MFG未在 {self.max_iterations} 次迭代内收敛！")
        
        logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        logger.info(f"平均每次迭代: {total_time/(iteration+1):.2f}秒")
        logger.info(f"最终失业率: {u_rate_new:.4f}")
        logger.info(f"最终就业率: {1-u_rate_new:.4f}")
        logger.info(f"最终收敛指标:")
        logger.info(f"  - diff_V: {diff_V:.2e} (容差: {self.epsilon_V:.2e})")
        logger.info(f"  - diff_a: {diff_a:.2e} (容差: {self.epsilon_a:.2e})")
        logger.info(f"  - diff_u: {diff_u:.2e} (容差: {self.epsilon_u:.2e})")
        logger.info("=" * 60)
        
        # 返回结果
        return {
            'V_U': V_U_new,
            'V_E': V_E_new,
            'a_star': a_star_new,
            'm_U': self.kfe_solver.m_U.copy(),
            'm_E': self.kfe_solver.m_E.copy(),
            'unemployment_rate': u_rate_new,
            'employment_rate': 1.0 - u_rate_new,
            'converged': converged,
            'n_iterations': iteration + 1,
            'total_time': total_time,
            'final_metrics': {
                'diff_V': diff_V,
                'diff_a': diff_a,
                'diff_u': diff_u
            },
            'history': self.history,
            'grid_nodes': self.sparse_grid.get_nodes()
        }
    
    def save_results(self, result: Dict, output_dir: str):
        """
        保存均衡结果
        
        Args:
            result: solve()返回的结果字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存核心数组
        np.savez(
            output_path / 'mfg_equilibrium.npz',
            V_U=result['V_U'],
            V_E=result['V_E'],
            a_star=result['a_star'],
            m_U=result['m_U'],
            m_E=result['m_E'],
            grid_nodes=result['grid_nodes']
        )
        
        # 保存历史记录
        np.savez(
            output_path / 'mfg_history.npz',
            unemployment_rate=np.array(result['history']['unemployment_rate']),
            iterations=result['n_iterations']
        )
        
        # 保存元数据
        import json
        metadata = {
            'converged': bool(result['converged']),
            'n_iterations': int(result['n_iterations']),
            'unemployment_rate': float(result['unemployment_rate']),
            'employment_rate': float(result['employment_rate']),
            'total_time': float(result['total_time']),
            'final_metrics': {
                k: float(v) for k, v in result['final_metrics'].items()
            }
        }
        
        with open(output_path / 'mfg_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 结果已保存到: {output_path}")
    
    def plot_convergence(self, result: Dict, output_dir: Optional[str] = None):
        """
        绘制收敛曲线
        
        Args:
            result: solve()返回的结果
            output_dir: 输出目录（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            history = result['history']
            metrics = history['convergence_metrics']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('MFG Convergence', fontsize=16)
            
            # 失业率
            axes[0, 0].plot(history['unemployment_rate'])
            axes[0, 0].set_title('Unemployment Rate')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Rate')
            axes[0, 0].grid(True)
            
            # 价值函数差异
            diff_V = [m['diff_V'] for m in metrics]
            axes[0, 1].semilogy(diff_V)
            axes[0, 1].axhline(self.epsilon_V, color='r', linestyle='--', label='Tolerance')
            axes[0, 1].set_title('Value Function Convergence')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('max|V_new - V_old|')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 策略差异
            diff_a = [m['diff_a'] for m in metrics]
            axes[1, 0].semilogy(diff_a)
            axes[1, 0].axhline(self.epsilon_a, color='r', linestyle='--', label='Tolerance')
            axes[1, 0].set_title('Policy Convergence')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('max|a_new - a_old|')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 失业率差异
            diff_u = [m['diff_u'] for m in metrics]
            axes[1, 1].semilogy(diff_u)
            axes[1, 1].axhline(self.epsilon_u, color='r', linestyle='--', label='Tolerance')
            axes[1, 1].set_title('Unemployment Rate Convergence')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('|u_new - u_old|')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path / 'mfg_convergence.png', dpi=300, bbox_inches='tight')
                logger.info(f"✅ 收敛图已保存到: {output_path / 'mfg_convergence.png'}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过绘图")
    
    def __repr__(self) -> str:
        return (
            f"MFGSimulator(\n"
            f"  state_space={self.state_space},\n"
            f"  sparse_grid={self.sparse_grid},\n"
            f"  max_iterations={self.max_iterations}\n"
            f")"
        )


# 简单测试示例
if __name__ == "__main__":
    print("=" * 60)
    print("MFG模拟器 - 快速演示")
    print("=" * 60)
    print("\n注意：这只是一个架构演示，实际运行需要完整的配置文件。")
    print("请参考 examples/ 目录中的完整示例。")
    print("\n" + "=" * 60)

