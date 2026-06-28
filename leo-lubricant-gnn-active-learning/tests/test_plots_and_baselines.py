from pathlib import Path

import pandas as pd

from leo_lubricants.evaluation.plots import plot_candidate_desirability_ranking, plot_training_history
from leo_lubricants.models.baselines import build_baselines


def test_plotting_creates_files(tmp_path: Path) -> None:
    history = pd.DataFrame({"epoch": [1, 2], "train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]})
    ranking = pd.DataFrame({"molecule_id": ["A", "B"], "desirability_score": [0.6, 0.4]})
    history_path = tmp_path / "history.png"
    ranking_path = tmp_path / "ranking.png"
    plot_training_history(history, history_path)
    plot_candidate_desirability_ranking(ranking, ranking_path)
    assert history_path.exists()
    assert ranking_path.exists()


def test_baseline_training_with_sample_data() -> None:
    features = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 1.5, 2.0, 2.5],
        }
    )
    targets = pd.DataFrame(
        {
            "t1": [0.0, 1.0, 2.0, 3.0],
            "t2": [1.0, 1.2, 1.4, 1.6],
        }
    )
    baselines = build_baselines(random_state=3)
    model = baselines["random_forest"].fit(features, targets)
    predictions = model.predict(features)
    assert predictions.shape == targets.shape
