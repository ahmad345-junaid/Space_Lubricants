from __future__ import annotations

import pandas as pd

from leo_lubricants.active_learning.acquisition import upper_confidence_desirability
from leo_lubricants.active_learning.desirability import compute_desirability, summarize_uncertainty


def rank_candidates(
    candidate_frame: pd.DataFrame,
    prediction_means,
    prediction_stds,
    exploitation_weight: float,
    exploration_weight: float,
) -> pd.DataFrame:
    ranked = candidate_frame.copy()
    desirability = compute_desirability(prediction_means)
    uncertainty = summarize_uncertainty(prediction_stds)
    acquisition = upper_confidence_desirability(
        desirability=desirability,
        uncertainty=uncertainty,
        exploitation_weight=exploitation_weight,
        exploration_weight=exploration_weight,
    )
    ranked["desirability"] = desirability
    ranked["uncertainty"] = uncertainty
    ranked["acquisition_score"] = acquisition
    return ranked.sort_values("acquisition_score", ascending=False).reset_index(drop=True)

