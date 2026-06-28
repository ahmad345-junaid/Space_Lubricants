import pandas as pd

from leo_lubricants.active_learning.desirability import DesirabilityConfig, compute_integrated_desirability


def test_desirability_monotonicity() -> None:
    config = DesirabilityConfig.default()
    predictions = pd.DataFrame(
        [
            {
                "log_vapor_pressure_pa": -3.7,
                "viscosity_cst": 25.0,
                "friction_coefficient": 0.14,
                "wear_rate": 4.5,
                "mass_loss_percent": 0.35,
                "viscosity_change_percent": 4.0,
                "degradation_index": 0.55,
            },
            {
                "log_vapor_pressure_pa": -4.8,
                "viscosity_cst": 50.0,
                "friction_coefficient": 0.08,
                "wear_rate": 2.8,
                "mass_loss_percent": 0.18,
                "viscosity_change_percent": 1.8,
                "degradation_index": 0.22,
            },
        ]
    )
    desirability = compute_integrated_desirability(predictions, config)
    assert desirability.iloc[1] > desirability.iloc[0]
