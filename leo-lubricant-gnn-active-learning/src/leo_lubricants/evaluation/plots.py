from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leo_lubricants.evaluation.calibration import prediction_interval_coverage_probability


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history["epoch"], history["train_loss"], label="train_loss")
    axis.plot(history["epoch"], history["val_loss"], label="val_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_predicted_vs_observed(
    prediction_frame: pd.DataFrame,
    target_name: str,
    output_path: str | Path,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    observed = prediction_frame[f"observed_{target_name}"].to_numpy(dtype=float)
    predicted = prediction_frame[f"predicted_{target_name}"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.scatter(observed, predicted)
    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    axis.plot([lower, upper], [lower, upper], linestyle="--")
    axis.set_xlabel("Observed")
    axis.set_ylabel("Predicted")
    axis.set_title(target_name)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_uncertainty_calibration_curve(
    prediction_frame: pd.DataFrame,
    target_name: str,
    output_path: str | Path,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    observed = prediction_frame[f"observed_{target_name}"].to_numpy(dtype=float)
    predicted = prediction_frame[f"predicted_{target_name}"].to_numpy(dtype=float)
    std = prediction_frame[f"std_{target_name}"].to_numpy(dtype=float)
    nominal = np.linspace(0.1, 0.9, 9)
    empirical = [
        prediction_interval_coverage_probability(observed, predicted, std, interval=float(level))
        for level in nominal
    ]
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(nominal, nominal, linestyle="--")
    axis.plot(nominal, empirical)
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_title(target_name)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_target_distribution_histograms(
    frame: pd.DataFrame,
    target_names: list[str],
    output_path: str | Path,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(target_names), 1, figsize=(6, 3 * len(target_names)))
    axes_array = np.atleast_1d(axes)
    for axis, target_name in zip(axes_array, target_names):
        axis.hist(frame[target_name].dropna().to_numpy(dtype=float), bins=min(10, max(3, len(frame[target_name].dropna()))))
        axis.set_title(target_name)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def plot_candidate_desirability_ranking(
    ranking_frame: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    top_frame = ranking_frame.head(top_n)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(top_frame["molecule_id"].astype(str), top_frame["desirability_score"].to_numpy(dtype=float))
    axis.set_xlabel("Molecule")
    axis.set_ylabel("Desirability")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
