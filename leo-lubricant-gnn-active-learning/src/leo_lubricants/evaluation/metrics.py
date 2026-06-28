from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _masked_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _masked_arrays(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if truth.size == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(truth, pred)))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _masked_arrays(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if truth.size == 0:
        return float("nan")
    return float(mean_absolute_error(truth, pred))


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _masked_arrays(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if truth.size < 2:
        return float("nan")
    return float(r2_score(truth, pred))


def per_target_metric_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    rows = []
    for index, target_name in enumerate(target_names):
        rows.append(
            {
                "target": target_name,
                "rmse": RMSE(y_true[:, index], y_pred[:, index]),
                "mae": MAE(y_true[:, index], y_pred[:, index]),
                "r2": R2(y_true[:, index], y_pred[:, index]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metric_table(metric_table: pd.DataFrame) -> pd.DataFrame:
    aggregate_values = {
        "target": "aggregate",
        "rmse": float(metric_table["rmse"].mean()),
        "mae": float(metric_table["mae"].mean()),
        "r2": float(metric_table["r2"].mean()),
    }
    return pd.DataFrame([aggregate_values])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, dict[str, float]]:
    table = per_target_metric_table(y_true, y_pred, target_names)
    return {
        row["target"]: {"rmse": float(row["rmse"]), "mae": float(row["mae"]), "r2": float(row["r2"])}
        for _, row in table.iterrows()
    }

