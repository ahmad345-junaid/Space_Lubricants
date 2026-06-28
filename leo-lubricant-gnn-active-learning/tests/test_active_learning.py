import numpy as np
import pandas as pd

from leo_lubricants.active_learning.acquisition import AcquisitionConfig, compute_acquisition_scores
from leo_lubricants.active_learning.desirability import DesirabilityConfig
from leo_lubricants.active_learning.select_candidates import CandidateSelectionConfig, rank_candidates


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "log_vapor_pressure_pa": -4.8,
                "viscosity_cst": 50.0,
                "friction_coefficient": 0.08,
                "wear_rate": 2.8,
                "mass_loss_percent": 0.18,
                "viscosity_change_percent": 1.8,
                "degradation_index": 0.22,
            },
            {
                "log_vapor_pressure_pa": -3.7,
                "viscosity_cst": 55.0,
                "friction_coefficient": 0.12,
                "wear_rate": 3.8,
                "mass_loss_percent": 0.25,
                "viscosity_change_percent": 2.8,
                "degradation_index": 0.35,
            },
        ]
    )


def test_acquisition_ranking_prefers_better_candidate() -> None:
    predictions = _prediction_frame()
    stds = pd.DataFrame(np.full((2, 7), 0.2), columns=predictions.columns)
    scores = compute_acquisition_scores(
        predictions=predictions,
        stds=stds,
        acquisition_config=AcquisitionConfig(vapor_pressure_cutoff=-4.0, desirability_cutoff=0.0, uncertainty_cutoff=1.0),
        desirability_config=DesirabilityConfig.default(),
        descriptors=np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert scores.loc[0, "acquisition_score"] > scores.loc[1, "acquisition_score"]


def test_candidate_eligibility_filter() -> None:
    candidate_df = pd.DataFrame({"molecule_id": ["A", "B"], "smiles": ["CCO", "CCN"]})
    predictions = _prediction_frame()
    stds = pd.DataFrame(np.full((2, 7), 0.2), columns=predictions.columns)
    ranked = rank_candidates(
        candidate_df,
        predictions,
        stds,
        descriptors=np.array([[0.0, 0.0], [1.0, 1.0]]),
        config=CandidateSelectionConfig(
            desirability_config=DesirabilityConfig.default(),
            acquisition_config=AcquisitionConfig(vapor_pressure_cutoff=-4.0, desirability_cutoff=0.0, uncertainty_cutoff=1.0),
        ),
    )
    assert bool(ranked.loc[0, "eligible"]) is True
    assert bool(ranked.loc[1, "eligible"]) is False
