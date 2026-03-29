#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结构截距校准自检脚本。

本测试覆盖三件事：
1. 匹配概率统一截距平移后，平均 job-finding rate 能贴近目标值；
2. eta0 反解后，平均 separation rate 能贴近目标值；
3. Bellman 与 KFE 对失业者的匹配概率口径保持一致。

运行方式：
    python TESTS/test_transition_intercepts.py
"""

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MODULES.MFG.equilibrium_solver import EquilibriumSolver


def build_temp_config(project_root: Path) -> Path:
    """
    构造一份仅用于快速测试的临时配置。
    """
    base_config_path = project_root / "CONFIG" / "mfg_config.yaml"
    with open(base_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["population"]["n_individuals"] = 300
    config["equilibrium"]["max_outer_iter"] = 2
    config["paths"]["output_dir"] = str(project_root / "OUTPUT" / "_test_mfg")
    # 正式配置默认已关闭 M5/M6 对齐；本测试单独打开开关，验证结构校准
    # 逻辑本身仍然可用。
    config["economics"]["separation_rate"]["auto_calibrate_eta0"] = True
    config["market"]["match_probability"]["auto_calibrate_intercept"] = True

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        delete=False,
    ) as file:
        yaml.dump(config, file, allow_unicode=True, default_flow_style=False)
        return Path(file.name)


def main() -> None:
    """
    执行结构截距校准自检。
    """
    project_root = PROJECT_ROOT
    temp_config = build_temp_config(project_root)

    try:
        solver = EquilibriumSolver(str(temp_config), save_results=False)
        base_population, initial_t = solver.create_base_population_sample(
            verbose=False
        )

        individuals = base_population.copy(deep=True)
        individuals["employment_status"] = "unemployed"
        individuals["current_wage"] = 0.0
        solver.initial_T = initial_t.copy()

        zero_effort = pd.Series(
            np.zeros(len(individuals), dtype=float),
            index=individuals.index,
        )

        solver._calibrate_transition_intercepts(
            individuals,
            theta=solver.target_theta,
            effort=zero_effort,
            calibrate_lambda=True,
            calibrate_eta0=False,
            verbose=False,
        )
        lambda_probs = solver.kfe_solver.compute_match_probabilities(
            individuals,
            zero_effort,
            solver.target_theta,
        )
        job_finding_mean = float(lambda_probs.mean())
        target_job_finding = solver.kfe_solver.target_job_finding_rate
        assert abs(job_finding_mean - target_job_finding) < 1.0e-6

        matched_mask = np.random.random(len(individuals)) < lambda_probs
        individuals.loc[matched_mask, "employment_status"] = "employed"
        individuals.loc[matched_mask, "current_wage"] = individuals.loc[
            matched_mask, "W"
        ]

        solver._calibrate_transition_intercepts(
            individuals,
            theta=solver.target_theta,
            effort=zero_effort,
            calibrate_lambda=False,
            calibrate_eta0=True,
            verbose=False,
        )
        mu_probs = solver.kfe_solver.compute_separation_rates(individuals)
        employed_mask = individuals["employment_status"] == "employed"
        separation_mean = float(mu_probs[employed_mask].mean())
        target_separation = solver.kfe_solver.target_separation_rate
        assert abs(separation_mean - target_separation) < 1.0e-6

        kfe_lambda = solver.kfe_solver.compute_match_probabilities(
            individuals,
            zero_effort,
            solver.target_theta,
        )
        bellman_lambda = solver.bellman_solver.compute_match_probabilities_batch(
            individuals,
            np.array([0.0]),
            solver.target_theta,
        )[:, 0]
        unemployed_mask = individuals["employment_status"] == "unemployed"
        max_diff = float(
            np.max(
                np.abs(
                    kfe_lambda[unemployed_mask] -
                    bellman_lambda[unemployed_mask]
                )
            )
        )
        assert max_diff < 1.0e-12

    finally:
        temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
