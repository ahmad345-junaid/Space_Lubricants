from __future__ import annotations

import numpy as np


def interval_coverage(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z_value: float = 1.96) -> float:
    lower = mean - z_value * std
    upper = mean + z_value * std
    within = (y_true >= lower) & (y_true <= upper)
    return float(within.mean())


def negative_log_likelihood(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    safe_std = np.clip(std, 1.0e-6, None)
    value = 0.5 * np.log(2.0 * np.pi * safe_std**2) + ((y_true - mean) ** 2) / (2.0 * safe_std**2)
    return float(np.mean(value))

