import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.CALIBRATION.smm_calibrator import SMMCalibrator


def run_test() -> None:
    calibrator = SMMCalibrator('CONFIG/calibration_config.yaml')

    # 关闭checkpoint，避免测试过程写入冗余断点
    calibrator.checkpoint_enabled = False

    def fake_full_solver(params_vector: np.ndarray):
        rho, kappa, alpha_t, gamma_t, gamma_s, gamma_d, gamma_w = params_vector
        unemployment = np.clip(0.02 + 0.03 * (rho - 0.4), 0.0, 1.0)
        job_find = np.clip(0.03 + 0.05 * (gamma_s - 0.45), 0.0, 1.0)
        separation = np.clip(0.005 + 0.03 * (alpha_t - 0.30), 0.0, 1.0)

        employed_wage = 3500.0 + 0.1 * (kappa - 2000.0)
        wage_std = 1200.0 + 800.0 * abs(gamma_w - 0.15)
        hours_mean = 46.0 + 20.0 * (gamma_t - 0.30)
        hours_std = 18.0 + 10.0 * abs(gamma_d - 0.45)

        individuals = pd.DataFrame(
            {
                'employment_status': ['employed'] * 40 + ['unemployed'] * 10,
                'current_wage': [employed_wage] * 40 + [0.0] * 10,
                'T': [hours_mean] * 40 + [0.0] * 10,
            }
        )

        eq_info = {
            'converged': True,
            'iterations': 1,
            'final_statistics': {
                'unemployment_rate': unemployment,
                'job_finding_rate': job_find,
                'separation_rate': separation,
                'mean_wage_employed': employed_wage,
                'std_wage': wage_std,
                'mean_weekly_hours': hours_mean,
                'std_weekly_hours': hours_std,
            },
        }
        return individuals, eq_info

    calibrator._create_mfg_solver = lambda: fake_full_solver

    def fake_optimize_stage(
        stage_name,
        method,
        options,
        internal_solver,
        weight_matrix,
        initial_internal,
        internal_bounds,
    ):
        _ = (stage_name, method, options, internal_solver, weight_matrix, internal_bounds)
        class _ObjStub:
            @staticmethod
            def print_best_evaluation():
                return None

        calibrator.obj_function = _ObjStub()
        return OptimizeResult(
            {
                'x': np.asarray(initial_internal, dtype=float),
                'success': True,
                'fun': 0.0,
                'nfev': 1,
                'nit': 1,
                'message': 'mock optimize',
            }
        )

    calibrator._optimize_stage = fake_optimize_stage
    calibrator._save_final_results = lambda result: None

    result = calibrator.calibrate(method='Powell', allow_auto_resume=False)

    assert len(result.x) == calibrator.param_utils.get_n_params()
    assert (calibrator.output_dir / 'parameter_partition.yaml').exists()
    assert (calibrator.output_dir / 'calibration_stage_summary.yaml').exists()

    print('test_calibration_pipeline_clds: PASS')


if __name__ == '__main__':
    run_test()
