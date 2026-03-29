import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MODULES.CALIBRATION.objective_function import create_weight_matrix
from MODULES.CALIBRATION.target_moments import TargetMoments


def run_test() -> None:
    target = TargetMoments('CONFIG/target_moments.yaml')
    n_moments = target.get_n_moments()

    w_inv = create_weight_matrix(target, 'inverse_variance_bootstrap')
    assert w_inv.shape == (n_moments, n_moments)
    assert np.all(np.diag(w_inv) > 0)

    cov = np.eye(n_moments) * 0.2
    cov[0, 1] = cov[1, 0] = 0.05
    w_eff = create_weight_matrix(
        target,
        'efficient_from_covariance',
        covariance_matrix=cov,
        regularization=1.0e-8,
    )
    assert w_eff.shape == (n_moments, n_moments)
    assert np.allclose(w_eff, w_eff.T, atol=1.0e-10)

    custom = {name: 1.0 + idx for idx, name in enumerate(target.get_moment_names())}
    w_diag = create_weight_matrix(target, 'diagonal', custom_weights=custom)
    assert np.isclose(np.diag(w_diag)[0], 1.0)
    assert np.isclose(np.diag(w_diag)[-1], float(n_moments))

    print('test_weight_matrix_clds: PASS')


if __name__ == '__main__':
    run_test()
