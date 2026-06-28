import numpy as np
import pandas as pd

from leo_lubricants.features.environment import build_environment_matrix, compute_degradation_memory_features


def test_environment_feature_generation_uses_requested_transforms() -> None:
    frame = pd.DataFrame(
        {
            "t_min_k": [270.0],
            "t_max_k": [330.0],
            "atomic_oxygen_fluence": [99.0],
            "uv_dose": [49.0],
            "test_duration_h": [23.0],
            "vacuum_pressure_pa": [0.001],
            "normal_load_n": [10.0],
            "sliding_speed_m_s": [0.2],
        }
    )
    enriched = compute_degradation_memory_features(frame)
    assert np.isclose(enriched.loc[0, "thermal_range_k"], 60.0)
    assert np.isclose(enriched.loc[0, "log1p_atomic_oxygen_fluence"], np.log1p(99.0))
    matrix = build_environment_matrix(frame)
    assert matrix.shape == (1, 16)

