from __future__ import annotations

import numpy as np
import pandas as pd

from leo_lubricants.data.schema import ENVIRONMENT_COLUMNS, validate_columns


def environment_columns() -> list[str]:
    return list(ENVIRONMENT_COLUMNS)


def compute_degradation_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df, ENVIRONMENT_COLUMNS)
    enriched = df.copy()
    enriched["log1p_atomic_oxygen_fluence"] = np.log1p(
        enriched["atomic_oxygen_fluence"].astype(float).clip(lower=0.0)
    )
    enriched["log1p_uv_dose"] = np.log1p(enriched["uv_dose"].astype(float).clip(lower=0.0))
    enriched["thermal_range_k"] = (
        enriched["t_max_k"].astype(float) - enriched["t_min_k"].astype(float)
    )
    enriched["log1p_test_duration_h"] = np.log1p(
        enriched["test_duration_h"].astype(float).clip(lower=0.0)
    )
    enriched["mean_test_temperature_k"] = (
        enriched["t_min_k"].astype(float) + enriched["t_max_k"].astype(float)
    ) / 2.0
    enriched["ao_temperature_interaction"] = (
        enriched["log1p_atomic_oxygen_fluence"] * enriched["mean_test_temperature_k"]
    )
    enriched["uv_duration_interaction"] = (
        enriched["log1p_uv_dose"] * enriched["log1p_test_duration_h"]
    )
    enriched["mechanical_severity"] = (
        enriched["normal_load_n"].astype(float) * enriched["sliding_speed_m_s"].astype(float)
    )
    return enriched


def build_environment_matrix(df: pd.DataFrame) -> np.ndarray:
    enriched = compute_degradation_memory_features(df)
    feature_columns = environment_columns() + [
        "log1p_atomic_oxygen_fluence",
        "log1p_uv_dose",
        "thermal_range_k",
        "log1p_test_duration_h",
        "mean_test_temperature_k",
        "ao_temperature_interaction",
        "uv_duration_interaction",
        "mechanical_severity",
    ]
    return enriched[feature_columns].to_numpy(dtype=float)


def add_environment_features(frame: pd.DataFrame, environment_columns: list[str]) -> pd.DataFrame:
    validate_columns(frame, environment_columns)
    return compute_degradation_memory_features(frame)

