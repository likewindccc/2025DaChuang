import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger(__name__)


class TargetMoments:
    """目标矩管理类。"""

    RATE_MOMENTS = {"unemployment_rate", "job_finding_rate", "separation_rate"}

    def __init__(
        self,
        config_path: str,
        selected_moments: Optional[Sequence[str]] = None,
    ):
        self.config_path = Path(config_path)
        self.target_moments: Dict[str, float] = {}
        self.moment_names: List[str] = []
        self.moment_metadata: Dict[str, Dict] = {}
        # 允许校准配置按需裁剪目标矩，但默认仍加载全部矩定义。
        self.selected_moments = (
            list(selected_moments) if selected_moments is not None else None
        )
        self._load_config()

    def _load_config(self) -> None:
        """从 YAML 加载目标矩与元数据。"""
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        moments_config = config["moments"]
        if self.selected_moments is None:
            selected_names = list(moments_config.keys())
        else:
            selected_names = []
            for moment_name in self.selected_moments:
                if moment_name not in moments_config:
                    raise KeyError(f"目标矩配置中不存在 moment={moment_name}")
                selected_names.append(moment_name)

        for moment_name in selected_names:
            moment_info = moments_config[moment_name]
            self.target_moments[moment_name] = float(moment_info["value"])
            self.moment_names.append(moment_name)
            self.moment_metadata[moment_name] = {
                "tag": moment_info.get("tag"),
                "unit": moment_info.get("unit", ""),
                "source": moment_info.get("source", ""),
                "confidence_interval": moment_info.get("confidence_interval"),
                "description": moment_info.get("description", ""),
                "confidence": moment_info.get("confidence"),
                "n": moment_info.get("n"),
                "bootstrap_se": moment_info.get("bootstrap_se"),
            }

    def get_target_moments(self) -> Dict[str, float]:
        return self.target_moments.copy()

    def get_target_vector(self) -> np.ndarray:
        return np.array([self.target_moments[name] for name in self.moment_names], dtype=float)

    def get_moment_names(self) -> List[str]:
        return self.moment_names.copy()

    def get_n_moments(self) -> int:
        return len(self.moment_names)

    def get_moment_metadata(self, moment_name: str) -> Dict:
        if moment_name not in self.moment_metadata:
            raise KeyError(f"未知目标矩: {moment_name}")
        return self.moment_metadata[moment_name].copy()

    def _extract_employed_sample(self, individuals: pd.DataFrame) -> pd.DataFrame:
        """统一提取就业者样本。"""
        if "employment_status" in individuals.columns:
            return individuals[individuals["employment_status"] == "employed"]
        if "employed" in individuals.columns:
            return individuals[individuals["employed"] == 1]
        raise KeyError("个体数据缺少 employment_status / employed 列，无法识别就业状态")

    @staticmethod
    def _safe_mean(series: pd.Series) -> float:
        cleaned = series.dropna()
        return float(cleaned.mean()) if len(cleaned) > 0 else 0.0

    @staticmethod
    def _safe_std(series: pd.Series) -> float:
        cleaned = series.dropna()
        return float(cleaned.std()) if len(cleaned) > 1 else 0.0

    def _pick_existing_key(self, source: Dict, keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in source and source[key] is not None:
                return float(source[key])
        return None

    def compute_simulated_moments(
        self,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> Dict[str, float]:
        """根据 MFG 均衡结果计算模拟矩（支持 CLDS M1-M8）。"""
        simulated: Dict[str, float] = {}
        stats = eq_info.get("final_statistics", {}) if isinstance(eq_info, dict) else {}

        employed = self._extract_employed_sample(individuals)

        wage_col = "current_wage" if "current_wage" in individuals.columns else "W"
        if wage_col not in employed.columns:
            raise KeyError("缺少工资列 current_wage / W，无法计算工资相关矩")
        wages = employed[wage_col]

        hours_col = "weekly_hours" if "weekly_hours" in employed.columns else "T"
        if hours_col not in employed.columns:
            raise KeyError("缺少工时列 weekly_hours / T，无法计算工时相关矩")
        hours = employed[hours_col]

        for moment_name in self.moment_names:
            if moment_name == "unemployment_rate":
                value = self._pick_existing_key(
                    stats,
                    ["unemployment_rate", "final_unemployment_rate"],
                )
                if value is None:
                    if "employment_status" in individuals.columns:
                        value = float((individuals["employment_status"] == "unemployed").mean())
                    elif "employed" in individuals.columns:
                        value = float(1.0 - individuals["employed"].mean())
                    else:
                        raise KeyError("无法从均衡结果推导失业率")
                simulated[moment_name] = value
                continue

            if moment_name == "mean_wage":
                simulated[moment_name] = self._safe_mean(wages)
                continue

            if moment_name == "std_wage":
                simulated[moment_name] = self._safe_std(wages)
                continue

            if moment_name == "log_std_wage":
                # 计算对数工资标准差（就业者工资取自然对数后的标准差）
                # clip 至 1 元防止 log(0) 或对数无穷，农村就业工资极低情况极少见
                log_wages = np.log(wages.clip(lower=1.0))
                simulated[moment_name] = self._safe_std(log_wages)
                continue

            if moment_name == "mean_weekly_hours":
                simulated[moment_name] = self._safe_mean(hours)
                continue

            if moment_name == "std_weekly_hours":
                simulated[moment_name] = self._safe_std(hours)
                continue

            if moment_name == "wage_iqr_ratio":
                cleaned_wages = wages.dropna()
                if len(cleaned_wages) == 0:
                    simulated[moment_name] = 0.0
                else:
                    p25 = float(cleaned_wages.quantile(0.25))
                    p75 = float(cleaned_wages.quantile(0.75))
                    simulated[moment_name] = (p75 / p25) if p25 > 1e-10 else 0.0
                continue

            if moment_name == "job_finding_rate":
                value = self._pick_existing_key(
                    stats,
                    [
                        "job_finding_rate",
                        "job_finding_rate_expected",
                        "lambda_mean_unemployed",
                    ],
                )
                if value is None:
                    raise KeyError("final_statistics 缺少 job_finding_rate/lambda_mean_unemployed")
                simulated[moment_name] = value
                continue

            if moment_name == "separation_rate":
                value = self._pick_existing_key(
                    stats,
                    [
                        "separation_rate",
                        "separation_rate_expected",
                        "mu_mean_employed",
                    ],
                )
                if value is None:
                    raise KeyError("final_statistics 缺少 separation_rate/mu_mean_employed")
                simulated[moment_name] = value
                continue

            if moment_name in stats:
                simulated[moment_name] = float(stats[moment_name])
                continue

            raise KeyError(f"未知模拟矩或缺少计算逻辑: {moment_name}")

        return simulated

    def get_simulated_vector(
        self,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> np.ndarray:
        simulated_moments = self.compute_simulated_moments(individuals, eq_info)
        return np.array([simulated_moments[name] for name in self.moment_names], dtype=float)

    def compute_moment_difference(
        self,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> np.ndarray:
        target_vec = self.get_target_vector()
        simulated_vec = self.get_simulated_vector(individuals, eq_info)
        return simulated_vec - target_vec

    def get_moment_comparison(
        self,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> pd.DataFrame:
        target = self.get_target_moments()
        simulated = self.compute_simulated_moments(individuals, eq_info)

        rows = []
        for name in self.moment_names:
            target_val = target[name]
            sim_val = simulated[name]
            diff = sim_val - target_val
            rel_error = (diff / target_val * 100.0) if abs(target_val) > 1e-10 else np.nan

            rows.append(
                {
                    "moment_name": name,
                    "target_value": target_val,
                    "simulated_value": sim_val,
                    "difference": diff,
                    "relative_error": rel_error,
                    "unit": self.moment_metadata[name].get("unit", ""),
                }
            )

        return pd.DataFrame(rows)

    def print_moment_comparison(
        self,
        individuals: pd.DataFrame,
        eq_info: Dict,
    ) -> None:
        comparison_df = self.get_moment_comparison(individuals, eq_info)

        logger.info("%s", "=" * 80)
        logger.info("矩对比分析")
        logger.info("%s", "=" * 80)

        for _, row in comparison_df.iterrows():
            logger.info("%s:", row["moment_name"])
            logger.info("  目标值: %.6f %s", row["target_value"], row["unit"])
            logger.info("  模拟值: %.6f %s", row["simulated_value"], row["unit"])
            logger.info("  差异: %.6f", row["difference"])
            if not np.isnan(row["relative_error"]):
                logger.info("  相对误差: %.2f%%", row["relative_error"])

        logger.info("%s", "=" * 80)

    def _estimate_standard_error(self, moment_name: str) -> float:
        """当缺少 bootstrap_se 时的兜底估计。"""
        metadata = self.moment_metadata[moment_name]
        value = float(self.target_moments[moment_name])
        n_value = metadata.get("n")

        try:
            n_obs = int(n_value) if n_value is not None else 0
        except (TypeError, ValueError):
            n_obs = 0

        confidence_interval = metadata.get("confidence_interval")
        if (
            isinstance(confidence_interval, (list, tuple))
            and len(confidence_interval) == 2
        ):
            lower = float(confidence_interval[0])
            upper = float(confidence_interval[1])
            ci_se = (upper - lower) / (2.0 * 1.96)
            if ci_se > 0:
                return ci_se

        if n_obs > 1 and moment_name in self.RATE_MOMENTS:
            bounded_value = min(max(value, 1e-8), 1.0 - 1e-8)
            return math.sqrt(bounded_value * (1.0 - bounded_value) / n_obs)

        if n_obs > 1 and moment_name == "mean_wage" and "std_wage" in self.target_moments:
            return float(self.target_moments["std_wage"]) / math.sqrt(n_obs)

        if n_obs > 2 and moment_name == "std_wage":
            return value / math.sqrt(2.0 * (n_obs - 1))

        if n_obs > 1 and moment_name == "mean_weekly_hours" and "std_weekly_hours" in self.target_moments:
            return float(self.target_moments["std_weekly_hours"]) / math.sqrt(n_obs)

        if n_obs > 2 and moment_name == "std_weekly_hours":
            return value / math.sqrt(2.0 * (n_obs - 1))

        if n_obs > 1 and moment_name == "wage_iqr_ratio":
            return max(0.08 * abs(value), 1e-6)

        return max(0.1 * abs(value), 1e-6)

    def get_bootstrap_se(self, moment_name: str, strict: bool = False) -> float:
        """
        获取单个目标矩的标准误。

        strict=True 时，若缺少 bootstrap_se 会直接报错。
        """
        metadata = self.get_moment_metadata(moment_name)
        bootstrap_se = metadata.get("bootstrap_se")

        if bootstrap_se is not None:
            se_value = float(bootstrap_se)
            if se_value <= 0:
                raise ValueError(f"moment={moment_name} 的 bootstrap_se 必须为正数")
            return se_value

        if strict:
            raise ValueError(
                f"moment={moment_name} 缺少 bootstrap_se。"
                "请先基于500次Bootstrap生成标准误。"
            )

        return self._estimate_standard_error(moment_name)

    def get_bootstrap_se_vector(self, strict: bool = False) -> np.ndarray:
        """按 moment_names 顺序返回标准误向量。"""
        return np.array(
            [self.get_bootstrap_se(name, strict=strict) for name in self.moment_names],
            dtype=float,
        )
