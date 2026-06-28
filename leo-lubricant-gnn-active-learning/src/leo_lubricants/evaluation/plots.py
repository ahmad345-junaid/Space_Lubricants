from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, target_name: str, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.scatter(y_true, y_pred, alpha=0.8)
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    axis.plot([lower, upper], [lower, upper], linestyle="--")
    axis.set_xlabel("Observed")
    axis.set_ylabel("Predicted")
    axis.set_title(f"Parity: {target_name}")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def save_uncertainty_plot(mean: np.ndarray, std: np.ndarray, target_name: str, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(mean)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(mean[order], label="mean")
    axis.fill_between(
        np.arange(len(order)),
        mean[order] - 1.96 * std[order],
        mean[order] + 1.96 * std[order],
        alpha=0.3,
        label="95% interval",
    )
    axis.set_title(f"Predictive uncertainty: {target_name}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)

