"""
NumBa Optimization Module

提供高性能的数值计算函数，使用JIT编译加速关键算法。
"""

import numpy as np
import numba as nb
from numba import jit, prange
from typing import Tuple, Optional
import warnings

# 配置numba警告
warnings.filterwarnings('ignore', category=nb.NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaWarning)


# ===================== Copula相关优化函数 =====================

@jit(nopython=True, parallel=True, cache=True)
def fast_uniform_to_marginal(uniform_samples: np.ndarray, 
                           cdf_params: np.ndarray,
                           distribution_type: int) -> np.ndarray:
    """
    快速将均匀分布样本转换为边际分布样本
    
    Args:
        uniform_samples: 均匀分布样本 [0,1]
        cdf_params: 分布参数
        distribution_type: 分布类型编码 (0:norm, 1:gamma, 2:beta, 3:lognorm)
    
    Returns:
        转换后的边际分布样本
    """
    n_samples = uniform_samples.shape[0]
    result = np.empty(n_samples, dtype=np.float64)
    
    for i in prange(n_samples):
        u = uniform_samples[i]
        
        if distribution_type == 0:  # 正态分布
            # 使用Box-Muller变换的逆函数近似
            if u <= 0.5:
                t = np.sqrt(-2.0 * np.log(2.0 * u))
                result[i] = cdf_params[0] + cdf_params[1] * (-t + 
                    (2.515517 + 0.802853*t + 0.010328*t*t) / 
                    (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t))
            else:
                t = np.sqrt(-2.0 * np.log(2.0 * (1.0 - u)))
                result[i] = cdf_params[0] + cdf_params[1] * (t - 
                    (2.515517 + 0.802853*t + 0.010328*t*t) / 
                    (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t))
        
        elif distribution_type == 1:  # Gamma分布 (简化版本)
            # 使用近似逆CDF
            if u < 0.01:
                result[i] = cdf_params[2]  # scale参数作为下界
            elif u > 0.99:
                result[i] = cdf_params[2] * cdf_params[0] * 10  # 近似上界
            else:
                # 简化的gamma逆变换
                result[i] = cdf_params[2] * (-np.log(1.0 - u))
        
        elif distribution_type == 2:  # Beta分布
            # Beta分布的简化逆变换
            a, b = cdf_params[0], cdf_params[1]
            if a == 1.0 and b == 1.0:
                result[i] = u  # 均匀分布
            else:
                # 使用Newton-Raphson迭代近似
                x = u
                for _ in range(5):  # 限制迭代次数
                    if x <= 0.0:
                        x = 1e-10
                    elif x >= 1.0:
                        x = 1.0 - 1e-10
                    
                    # 简化的Beta PDF和CDF计算
                    pdf = x**(a-1) * (1-x)**(b-1)
                    cdf_est = x**a * (1-x)**b  # 简化估计
                    
                    if pdf > 1e-10:
                        x = x - (cdf_est - u) / pdf
                    else:
                        break
                
                result[i] = max(0.0, min(1.0, x))
        
        else:  # 默认情况：对数正态分布
            if u <= 0.0 or u >= 1.0:
                result[i] = cdf_params[0]
            else:
                # 对数正态逆变换
                log_u = np.log(u)
                result[i] = np.exp(cdf_params[0] + cdf_params[1] * log_u)
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_correlation_to_covariance(correlation_matrix: np.ndarray, 
                                  variances: np.ndarray) -> np.ndarray:
    """
    快速将相关矩阵转换为协方差矩阵
    
    Args:
        correlation_matrix: 相关矩阵
        variances: 方差向量
        
    Returns:
        协方差矩阵
    """
    n = correlation_matrix.shape[0]
    covariance_matrix = np.empty((n, n), dtype=np.float64)
    
    # 计算标准差
    std_devs = np.sqrt(variances)
    
    for i in prange(n):
        for j in range(n):
            covariance_matrix[i, j] = (correlation_matrix[i, j] * 
                                     std_devs[i] * std_devs[j])
    
    return covariance_matrix


@jit(nopython=True, cache=True)
def fast_cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
    """
    快速Cholesky分解（针对小矩阵优化）
    
    Args:
        matrix: 正定矩阵
        
    Returns:
        下三角Cholesky因子
    """
    n = matrix.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # 对角线元素
                sum_squares = 0.0
                for k in range(j):
                    sum_squares += L[i, k] * L[i, k]
                L[i, j] = np.sqrt(matrix[i, i] - sum_squares)
            else:  # 下三角元素
                sum_products = 0.0
                for k in range(j):
                    sum_products += L[i, k] * L[j, k]
                L[i, j] = (matrix[i, j] - sum_products) / L[j, j]
    
    return L


# ===================== 多元正态分布优化函数 =====================

@jit(nopython=True, parallel=True, cache=True)
def fast_multivariate_normal_sample(n_samples: int,
                                   mean: np.ndarray,
                                   cholesky_factor: np.ndarray,
                                   random_state: int = 42) -> np.ndarray:
    """
    快速多元正态分布采样
    
    Args:
        n_samples: 样本数量
        mean: 均值向量
        cholesky_factor: 协方差矩阵的Cholesky分解
        random_state: 随机种子
        
    Returns:
        多元正态分布样本 [n_samples, n_features]
    """
    np.random.seed(random_state)
    n_features = mean.shape[0]
    
    # 生成标准正态分布样本
    standard_samples = np.random.standard_normal((n_samples, n_features))
    
    # 变换为目标分布
    samples = np.empty((n_samples, n_features), dtype=np.float64)
    
    for i in prange(n_samples):
        # 应用Cholesky变换
        transformed = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            for k in range(j + 1):
                transformed[j] += cholesky_factor[j, k] * standard_samples[i, k]
        
        # 添加均值
        for j in range(n_features):
            samples[i, j] = transformed[j] + mean[j]
    
    return samples


@jit(nopython=True, cache=True)
def fast_multivariate_normal_logpdf(samples: np.ndarray,
                                   mean: np.ndarray,
                                   precision_matrix: np.ndarray,
                                   log_det_cov: float) -> np.ndarray:
    """
    快速计算多元正态分布的对数概率密度
    
    Args:
        samples: 样本矩阵 [n_samples, n_features]
        mean: 均值向量
        precision_matrix: 精度矩阵（协方差矩阵的逆）
        log_det_cov: 协方差矩阵的对数行列式
        
    Returns:
        对数概率密度数组
    """
    n_samples, n_features = samples.shape
    log_probs = np.empty(n_samples, dtype=np.float64)
    
    # 常数项
    log_2pi = 1.8378770664093453  # np.log(2 * np.pi)
    norm_const = -0.5 * (n_features * log_2pi + log_det_cov)
    
    for i in range(n_samples):
        # 计算 (x - μ)
        diff = samples[i] - mean
        
        # 计算 (x - μ)ᵀ Σ⁻¹ (x - μ)
        quadratic_form = 0.0
        for j in range(n_features):
            for k in range(n_features):
                quadratic_form += diff[j] * precision_matrix[j, k] * diff[k]
        
        log_probs[i] = norm_const - 0.5 * quadratic_form
    
    return log_probs


# ===================== 数据验证优化函数 =====================

@jit(nopython=True, parallel=True, cache=True)
def fast_data_bounds_check(data: np.ndarray,
                          lower_bounds: np.ndarray,
                          upper_bounds: np.ndarray) -> np.ndarray:
    """
    快速检查数据是否在指定边界内
    
    Args:
        data: 数据矩阵 [n_samples, n_features]
        lower_bounds: 下界向量
        upper_bounds: 上界向量
        
    Returns:
        布尔数组，True表示该样本所有特征都在边界内
    """
    n_samples, n_features = data.shape
    valid_samples = np.empty(n_samples, dtype=nb.boolean)
    
    for i in prange(n_samples):
        is_valid = True
        for j in range(n_features):
            if data[i, j] < lower_bounds[j] or data[i, j] > upper_bounds[j]:
                is_valid = False
                break
        valid_samples[i] = is_valid
    
    return valid_samples


@jit(nopython=True, parallel=True, cache=True)
def fast_descriptive_statistics(data: np.ndarray) -> np.ndarray:
    """
    快速计算描述性统计量
    
    Args:
        data: 数据矩阵 [n_samples, n_features]
        
    Returns:
        统计量矩阵 [n_features, 6]，包含：均值、标准差、最小值、25%分位数、中位数、最大值
    """
    n_samples, n_features = data.shape
    stats = np.empty((n_features, 6), dtype=np.float64)
    
    for j in prange(n_features):
        # 提取列数据
        column_data = data[:, j].copy()
        
        # 排序用于计算分位数
        column_data.sort()
        
        # 均值
        mean_val = np.mean(column_data)
        stats[j, 0] = mean_val
        
        # 标准差
        variance = 0.0
        for i in range(n_samples):
            variance += (column_data[i] - mean_val) ** 2
        std_val = np.sqrt(variance / (n_samples - 1))
        stats[j, 1] = std_val
        
        # 最小值
        stats[j, 2] = column_data[0]
        
        # 25%分位数
        q25_idx = int(0.25 * (n_samples - 1))
        stats[j, 3] = column_data[q25_idx]
        
        # 中位数
        median_idx = n_samples // 2
        if n_samples % 2 == 0:
            stats[j, 4] = (column_data[median_idx - 1] + column_data[median_idx]) / 2.0
        else:
            stats[j, 4] = column_data[median_idx]
        
        # 最大值
        stats[j, 5] = column_data[n_samples - 1]
    
    return stats


# ===================== 批量处理优化函数 =====================

@jit(nopython=True, cache=True)
def fast_batch_size_optimizer(total_samples: int,
                             available_memory_mb: float,
                             sample_size_bytes: int) -> int:
    """
    优化批次大小以最大化内存利用率
    
    Args:
        total_samples: 总样本数
        available_memory_mb: 可用内存 (MB)
        sample_size_bytes: 单个样本的字节大小
        
    Returns:
        优化的批次大小
    """
    available_bytes = available_memory_mb * 1024 * 1024
    
    # 保留30%内存用于其他操作
    usable_bytes = available_bytes * 0.7
    
    # 计算理论最大批次大小
    max_batch_size = int(usable_bytes / sample_size_bytes)
    
    # 确保批次大小合理
    min_batch_size = 100
    max_reasonable_batch_size = 10000
    
    optimal_batch_size = max(min_batch_size, 
                           min(max_batch_size, max_reasonable_batch_size))
    
    # 确保不超过总样本数
    return min(optimal_batch_size, total_samples)


# ===================== 性能监控函数 =====================

class PerformanceMonitor:
    """性能监控器（支持numba编译环境）"""
    
    def __init__(self):
        self.execution_times = {}
        self.memory_usage = {}
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _time_function_execution(iterations: int) -> float:
        """测量函数执行时间"""
        import time
        start_time = time.time()
        
        # 执行一些计算密集型操作作为基准
        result = 0.0
        for i in range(iterations):
            result += np.sqrt(i + 1) * np.log(i + 2)
        
        end_time = time.time()
        return end_time - start_time
    
    def benchmark_environment(self) -> dict:
        """对当前环境进行基准测试"""
        # 测试计算性能
        compute_time = self._time_function_execution(100000)
        
        # 测试内存分配性能
        test_array = np.random.random((1000, 1000))
        memory_time = self._time_function_execution(1000)
        
        return {
            'compute_performance': 1.0 / compute_time,  # 相对性能得分
            'memory_performance': 1.0 / memory_time,
            'numba_enabled': True,
            'array_size_test': test_array.shape
        }


# ===================== 工具函数 =====================

def check_numba_availability() -> bool:
    """检查numba是否可用并正常工作"""
    try:
        @jit(nopython=True)
        def test_function(x):
            return x ** 2 + 1
        
        result = test_function(5.0)
        return abs(result - 26.0) < 1e-10
    except:
        return False


def get_optimization_info() -> dict:
    """获取优化环境信息"""
    info = {
        'numba_version': nb.__version__,
        'numba_available': check_numba_availability(),
        'parallel_enabled': True,
        'cache_enabled': True
    }
    
    # 检查并行支持
    try:
        @jit(nopython=True, parallel=True)
        def test_parallel(arr):
            for i in prange(len(arr)):
                arr[i] = arr[i] ** 2
            return arr
        
        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        test_parallel(test_array)
        info['parallel_test_passed'] = True
    except:
        info['parallel_test_passed'] = False
        info['parallel_enabled'] = False
    
    return info
