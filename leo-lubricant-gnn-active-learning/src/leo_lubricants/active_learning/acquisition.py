from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leo_lubricants.active_learning.desirability import DesirabilityConfig, compute_integrated_desirability


@dataclass(frozen=True)
class AcquisitionConfig:
    desirability_weight: float = 0.6
    uncertainty_weight: float = 0.4
    diversity_weight: float = 0.0
    vapor_pressure_cutoff: float = -4.0
    desirability_cutoff: float = 0.05
    uncertainty_cutoff: float = 2.0
    selected_descriptor_matrix: np.ndarray | None = None


def compute_uncertainty_score(stds) -> pd.Series:
    if isinstance(stds, pd.DataFrame):
        values = stds.to_numpy(dtype=float)
        index = stds.index
    else:
        values = np.asarray(stds, dtype=float)
        index = None
    mean_uncertainty = values.mean(axis=1)
    min_value = float(mean_uncertainty.min())
    max_value = float(mean_uncertainty.max())
    span = max(max_value - min_value, 1.0e-12)
    normalized = (mean_uncertainty - min_value) / span
    return pd.Series(normalized, index=index, name="uncertainty_score")


def compute_performance_score(predictions, desirability_config: DesirabilityConfig) -> pd.Series:
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    return compute_integrated_desirability(predictions, desirability_config)


def _diversity_penalty(descriptors: np.ndarray, selected_descriptor_matrix: np.ndarray | None) -> np.ndarray:
    if selected_descriptor_matrix is None or selected_descriptor_matrix.size == 0:
        return np.zeros(descriptors.shape[0], dtype=float)
    penalties = np.zeros(descriptors.shape[0], dtype=float)
    for row_index, descriptor_row in enumerate(descriptors):
        distances = np.linalg.norm(selected_descriptor_matrix - descriptor_row, axis=1)
        penalties[row_index] = float(np.exp(-np.min(distances)))
    return penalties


def compute_acquisition_scores(
    predictions,
    stds,
    acquisition_config: AcquisitionConfig,
    desirability_config: DesirabilityConfig,
    descriptors: np.ndarray | None = None,
) -> pd.DataFrame:
    prediction_frame = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    std_frame = stds if isinstance(stds, pd.DataFrame) else pd.DataFrame(stds, index=prediction_frame.index)
    desirability_score = compute_performance_score(prediction_frame, desirability_config)
    uncertainty_score = compute_uncertainty_score(std_frame)
    mean_uncertainty = std_frame.mean(axis=1)
    eligible = (
        (prediction_frame["log_vapor_pressure_pa"] <= acquisition_config.vapor_pressure_cutoff)
        & (desirability_score >= acquisition_config.desirability_cutoff)
        & (mean_uncertainty <= acquisition_config.uncertainty_cutoff)
    )
    descriptor_array = np.asarray(descriptors, dtype=float) if descriptors is not None else np.zeros((len(prediction_frame), 1), dtype=float)
    diversity_penalty = _diversity_penalty(descriptor_array, acquisition_config.selected_descriptor_matrix)
    acquisition_score = (
        acquisition_config.desirability_weight * desirability_score.to_numpy(dtype=float)
        + acquisition_config.uncertainty_weight * uncertainty_score.to_numpy(dtype=float)
        - acquisition_config.diversity_weight * diversity_penalty
    )
    acquisition_score = np.where(eligible.to_numpy(dtype=bool), acquisition_score, 0.0)
    return pd.DataFrame(
        {
            "desirability_score": desirability_score,
            "uncertainty_score": uncertainty_score,
            "mean_uncertainty": mean_uncertainty,
            "eligible": eligible,
            "diversity_penalty": diversity_penalty,
            "acquisition_score": acquisition_score,
        },
        index=prediction_frame.index,
    )
