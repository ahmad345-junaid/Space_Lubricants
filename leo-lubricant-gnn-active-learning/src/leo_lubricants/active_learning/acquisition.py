from __future__ import annotations

import numpy as np


def upper_confidence_desirability(
    desirability: np.ndarray,
    uncertainty: np.ndarray,
    exploitation_weight: float,
    exploration_weight: float,
) -> np.ndarray:
    return exploitation_weight * desirability + exploration_weight * uncertainty

