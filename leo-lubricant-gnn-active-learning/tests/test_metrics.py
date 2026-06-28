import numpy as np

from leo_lubricants.evaluation.metrics import regression_metrics


def test_regression_metrics_returns_expected_keys() -> None:
    y_true = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [1.9, 3.1], [2.8, 4.2]])
    metrics = regression_metrics(y_true, y_pred, ["a", "b"])
    assert set(metrics["a"].keys()) == {"rmse", "mae", "r2", "spearman"}

