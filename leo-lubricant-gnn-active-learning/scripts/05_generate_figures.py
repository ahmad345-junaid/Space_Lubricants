from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate parity and uncertainty plots from evaluation outputs.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.schema import TARGET_COLUMNS
    from leo_lubricants.evaluation.plots import save_parity_plot, save_uncertainty_plot
    from leo_lubricants.utils.config import load_config

    experiment_config = load_config(args.experiment_config)
    predictions = pd.read_csv(experiment_config["evaluation_predictions_path"])
    figure_dir = Path(experiment_config["figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGET_COLUMNS:
        save_parity_plot(
            predictions[f"observed_{target}"].to_numpy(),
            predictions[f"predicted_{target}"].to_numpy(),
            target,
            figure_dir / f"parity_{target}.png",
        )
        save_uncertainty_plot(
            predictions[f"predicted_{target}"].to_numpy(),
            predictions[f"std_{target}"].to_numpy(),
            target,
            figure_dir / f"uncertainty_{target}.png",
        )


if __name__ == "__main__":
    main()
