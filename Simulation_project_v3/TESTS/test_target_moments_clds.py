import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.CALIBRATION.target_moments import TargetMoments


def run_test() -> None:
    target = TargetMoments('CONFIG/target_moments.yaml')

    assert target.get_n_moments() == 8, 'CLDS口径应为M1-M8共8个矩'

    selected_target = TargetMoments(
        'CONFIG/target_moments.yaml',
        selected_moments=[
            'unemployment_rate',
            'mean_wage',
            'log_std_wage',
            'mean_weekly_hours',
            'wage_iqr_ratio',
            'std_weekly_hours',
        ],
    )
    assert selected_target.get_moment_names() == [
        'unemployment_rate',
        'mean_wage',
        'log_std_wage',
        'mean_weekly_hours',
        'wage_iqr_ratio',
        'std_weekly_hours',
    ]
    assert selected_target.get_n_moments() == 6

    individuals = pd.DataFrame(
        {
            'employment_status': ['employed', 'employed', 'unemployed', 'employed'],
            'current_wage': [3000.0, 5000.0, 0.0, 4000.0],
            'T': [42.0, 55.0, 0.0, 48.0],
        }
    )
    eq_info = {
        'final_statistics': {
            'unemployment_rate': 0.25,
            'job_finding_rate': 0.03,
            'separation_rate': 0.01,
        }
    }

    sim = target.compute_simulated_moments(individuals, eq_info)

    assert np.isclose(sim['unemployment_rate'], 0.25)
    assert np.isclose(sim['mean_wage'], 4000.0)
    assert np.isclose(sim['mean_weekly_hours'], (42.0 + 55.0 + 48.0) / 3)
    assert np.isclose(sim['job_finding_rate'], 0.03)
    assert np.isclose(sim['separation_rate'], 0.01)
    assert sim['wage_iqr_ratio'] > 0

    selected_sim = selected_target.compute_simulated_moments(individuals, eq_info)
    assert 'job_finding_rate' not in selected_sim
    assert 'separation_rate' not in selected_sim
    assert np.isclose(selected_sim['mean_wage'], 4000.0)

    se_vec = target.get_bootstrap_se_vector(strict=False)
    assert se_vec.shape[0] == 8
    assert np.all(se_vec > 0)

    selected_se_vec = selected_target.get_bootstrap_se_vector(strict=False)
    assert selected_se_vec.shape[0] == 6
    assert np.all(selected_se_vec > 0)

    print('test_target_moments_clds: PASS')


if __name__ == '__main__':
    run_test()
