from __future__ import annotations

import numpy as np
from scipy.stats import norm


def prediction_interval_coverage_probability(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    interval: float = 0.95,
) -> float:
    z_value = norm.ppf((1.0 + interval) / 2.0)
    safe_std = np.clip(std, 1.0e-8, None)
    lower = mean - z_value * safe_std
    upper = mean + z_value * safe_std
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def regression_calibration_error(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    nominal_levels: np.ndarray | None = None,
) -> float:
    levels = nominal_levels if nominal_levels is not None else np.linspace(0.1, 0.9, 9)
    errors = []
    for level in levels:
        empirical = prediction_interval_coverage_probability(y_true, mean, std, interval=float(level))
        errors.append(abs(empirical - float(level)))
    return float(np.mean(errors))


def gaussian_negative_log_likelihood(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    safe_std = np.clip(std, 1.0e-8, None)
    value = 0.5 * np.log(2.0 * np.pi * safe_std**2) + ((y_true - mean) ** 2) / (2.0 * safe_std**2)
    return float(np.mean(value))

