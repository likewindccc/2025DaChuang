import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.CALIBRATION.smm_calibrator import SMMCalibrator


def run_test() -> None:
    """校验校准实例只加载 calibration_config.yaml 中声明的目标矩。"""
    calibrator = SMMCalibrator('CONFIG/calibration_config.yaml')

    expected_names = [
        'unemployment_rate',
        'mean_wage',
        'log_std_wage',
        'mean_weekly_hours',
        'wage_iqr_ratio',
        'std_weekly_hours',
    ]

    assert calibrator.target_moments.get_moment_names() == expected_names
    assert calibrator.target_moments.get_n_moments() == 6

    print('test_calibration_active_moments: PASS')


if __name__ == '__main__':
    run_test()
