from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(target_names):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index])))
        mae = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        r2 = float(r2_score(y_true[:, index], y_pred[:, index]))
        spearman = float(spearmanr(y_true[:, index], y_pred[:, index]).statistic)
        metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}
    return metrics

