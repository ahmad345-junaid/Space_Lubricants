from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leo_lubricants.data.schema import TARGET_COLUMNS


@dataclass(frozen=True)
class ObjectiveDirection:
    minimize: tuple[str, ...] = (
        "log_vapor_pressure_pa",
        "friction_coefficient",
        "wear_rate",
        "mass_loss_percent",
        "viscosity_change_percent",
        "degradation_index",
    )
    maximize: tuple[str, ...] = ("viscosity_cst",)


def compute_desirability(predictions: np.ndarray, target_names: list[str] | None = None) -> np.ndarray:
    names = target_names or TARGET_COLUMNS
    directions = ObjectiveDirection()
    desirability_components = []
    for index, name in enumerate(names):
        values = predictions[:, index]
        lower = values.min()
        upper = values.max()
        span = max(upper - lower, 1.0e-8)
        normalized = (values - lower) / span
        if name in directions.minimize:
            normalized = 1.0 - normalized
        desirability_components.append(normalized)
    return np.mean(np.stack(desirability_components, axis=1), axis=1)


def summarize_uncertainty(stds: np.ndarray) -> np.ndarray:
    return stds.mean(axis=1)
