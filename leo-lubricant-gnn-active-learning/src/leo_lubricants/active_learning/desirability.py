from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesirabilityTarget:
    name: str
    mode: str
    weight: float
    lower: float | None = None
    target: float | None = None
    target_low: float | None = None
    target_high: float | None = None
    upper: float | None = None
    curvature: float = 1.0


@dataclass(frozen=True)
class DesirabilityConfig:
    targets: tuple[DesirabilityTarget, ...] = field(default_factory=tuple)

    @staticmethod
    def default() -> "DesirabilityConfig":
        return DesirabilityConfig(
            targets=(
                DesirabilityTarget("log_vapor_pressure_pa", "lower", 1.0, target=-5.0, upper=-3.5, curvature=1.2),
                DesirabilityTarget("viscosity_cst", "range", 1.0, lower=20.0, target_low=35.0, target_high=65.0, upper=90.0, curvature=1.0),
                DesirabilityTarget("friction_coefficient", "lower", 1.0, target=0.08, upper=0.16, curvature=1.0),
                DesirabilityTarget("wear_rate", "lower", 1.0, target=2.5, upper=5.0, curvature=1.0),
                DesirabilityTarget("mass_loss_percent", "lower", 1.0, target=0.1, upper=0.4, curvature=1.0),
                DesirabilityTarget("viscosity_change_percent", "lower", 1.0, target=1.0, upper=5.0, curvature=1.0),
                DesirabilityTarget("degradation_index", "lower", 1.0, target=0.15, upper=0.6, curvature=1.0),
            )
        )


def lower_is_better_desirability(values, target, upper, curvature):
    clipped = np.asarray(values, dtype=float)
    desirability = np.ones_like(clipped, dtype=float)
    desirability = np.where(clipped <= target, 1.0, desirability)
    middle = (clipped > target) & (clipped < upper)
    scaled = (upper - clipped[middle]) / max(upper - target, 1.0e-12)
    desirability[middle] = np.clip(scaled, 0.0, 1.0) ** curvature
    desirability = np.where(clipped >= upper, 0.0, desirability)
    return np.clip(desirability, 0.0, 1.0)


def higher_is_better_desirability(values, target, lower, curvature):
    clipped = np.asarray(values, dtype=float)
    desirability = np.ones_like(clipped, dtype=float)
    desirability = np.where(clipped >= target, 1.0, desirability)
    middle = (clipped > lower) & (clipped < target)
    scaled = (clipped[middle] - lower) / max(target - lower, 1.0e-12)
    desirability[middle] = np.clip(scaled, 0.0, 1.0) ** curvature
    desirability = np.where(clipped <= lower, 0.0, desirability)
    return np.clip(desirability, 0.0, 1.0)


def range_desirability(values, lower, target_low, target_high, upper, curvature):
    clipped = np.asarray(values, dtype=float)
    desirability = np.zeros_like(clipped, dtype=float)
    rising = (clipped > lower) & (clipped < target_low)
    if np.any(rising):
        scaled = (clipped[rising] - lower) / max(target_low - lower, 1.0e-12)
        desirability[rising] = np.clip(scaled, 0.0, 1.0) ** curvature
    plateau = (clipped >= target_low) & (clipped <= target_high)
    desirability[plateau] = 1.0
    falling = (clipped > target_high) & (clipped < upper)
    if np.any(falling):
        scaled = (upper - clipped[falling]) / max(upper - target_high, 1.0e-12)
        desirability[falling] = np.clip(scaled, 0.0, 1.0) ** curvature
    return np.clip(desirability, 0.0, 1.0)


def compute_integrated_desirability(predictions: pd.DataFrame, config: DesirabilityConfig) -> pd.Series:
    desirability_columns = []
    weights = np.array([target.weight for target in config.targets], dtype=float)
    normalized_weights = weights / weights.sum()
    for target in config.targets:
        values = predictions[target.name].to_numpy(dtype=float)
        if target.mode == "lower":
            desirability = lower_is_better_desirability(values, target.target, target.upper, target.curvature)
        elif target.mode == "higher":
            desirability = higher_is_better_desirability(values, target.target, target.lower, target.curvature)
        elif target.mode == "range":
            desirability = range_desirability(
                values,
                target.lower,
                target.target_low,
                target.target_high,
                target.upper,
                target.curvature,
            )
        else:
            raise ValueError(f"Unsupported desirability mode: {target.mode}")
        desirability_columns.append(desirability)
    desirability_matrix = np.column_stack(desirability_columns)
    zero_mask = np.any(desirability_matrix <= 0.0, axis=1)
    clipped = np.clip(desirability_matrix, 1.0e-12, 1.0)
    integrated = np.exp(np.sum(normalized_weights * np.log(clipped), axis=1))
    integrated[zero_mask] = 0.0
    return pd.Series(integrated, index=predictions.index, name="desirability_score")
