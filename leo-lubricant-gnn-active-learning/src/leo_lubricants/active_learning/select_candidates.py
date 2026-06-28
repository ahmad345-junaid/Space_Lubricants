from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leo_lubricants.active_learning.acquisition import AcquisitionConfig, compute_acquisition_scores
from leo_lubricants.active_learning.desirability import DesirabilityConfig


@dataclass(frozen=True)
class CandidateSelectionConfig:
    desirability_config: DesirabilityConfig = field(default_factory=DesirabilityConfig.default)
    acquisition_config: AcquisitionConfig = field(default_factory=AcquisitionConfig)


def rank_candidates(
    candidate_df: pd.DataFrame,
    predictions: pd.DataFrame,
    stds: pd.DataFrame,
    descriptors: np.ndarray,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    acquisition_frame = compute_acquisition_scores(
        predictions=predictions,
        stds=stds,
        acquisition_config=config.acquisition_config,
        desirability_config=config.desirability_config,
        descriptors=descriptors,
    )
    ranked = candidate_df[[column for column in ["molecule_id", "smiles"] if column in candidate_df.columns]].copy()
    ranked = pd.concat([ranked, acquisition_frame], axis=1)
    for column in predictions.columns:
        ranked[f"predicted_{column}"] = predictions[column].to_numpy(dtype=float)
    for column in stds.columns:
        ranked[f"uncertainty_{column}"] = stds[column].to_numpy(dtype=float)
    return ranked.sort_values(["eligible", "acquisition_score"], ascending=[False, False]).reset_index(drop=True)


def select_top_batch(ranked_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    eligible = ranked_df[ranked_df["eligible"]].copy()
    return eligible.head(batch_size).reset_index(drop=True)

