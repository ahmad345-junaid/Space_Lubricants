import numpy as np

from leo_lubricants.evaluation.metrics import MAE, R2, RMSE, per_target_metric_table


def test_metrics_produce_finite_values() -> None:
    y_true = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [1.9, 3.1], [2.8, 4.2]])
    assert np.isfinite(RMSE(y_true[:, 0], y_pred[:, 0]))
    assert np.isfinite(MAE(y_true[:, 0], y_pred[:, 0]))
    assert np.isfinite(R2(y_true[:, 0], y_pred[:, 0]))
    table = per_target_metric_table(y_true, y_pred, ["a", "b"])
    assert np.isfinite(table[["rmse", "mae", "r2"]].to_numpy()).all()
