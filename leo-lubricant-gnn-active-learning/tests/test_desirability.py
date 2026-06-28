import numpy as np

from leo_lubricants.active_learning.desirability import compute_desirability, summarize_uncertainty


def test_desirability_prefers_better_tradeoff() -> None:
    predictions = np.array(
        [
            [0.02, 40.0, 0.12, 4.0, 0.30, 3.0, 0.45],
            [0.01, 55.0, 0.08, 2.5, 0.15, 1.8, 0.20],
        ]
    )
    values = compute_desirability(predictions)
    assert values[1] > values[0]
    assert summarize_uncertainty(np.array([[0.1] * 7]))[0] == 0.1

