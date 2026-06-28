from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate available figures from training, evaluation, and ranking outputs.")
    parser.add_argument("--input-root", default="results", help="Root directory containing metrics, models, and rankings.")
    parser.add_argument("--output-dir", default="results/figures", help="Directory for generated figures.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.schema import TARGET_COLUMNS
    from leo_lubricants.evaluation.plots import (
        plot_candidate_desirability_ranking,
        plot_predicted_vs_observed,
        plot_target_distribution_histograms,
        plot_training_history,
        plot_uncertainty_calibration_curve,
    )
    from leo_lubricants.utils.logging import get_logger

    logger = get_logger("figures")
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []
    history_path = input_root / "trained_models" / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        output_path = output_dir / "training_history.png"
        plot_training_history(history, output_path)
        generated_files.append(str(output_path))
    prediction_candidates = [
        input_root / "metrics" / "predictions.csv",
        input_root / "trained_models" / "validation_predictions.csv",
    ]
    prediction_path = next((path for path in prediction_candidates if path.exists()), None)
    if prediction_path is not None:
        prediction_frame = pd.read_csv(prediction_path)
        available_targets = [
            target_name
            for target_name in TARGET_COLUMNS
            if f"observed_{target_name}" in prediction_frame.columns and f"predicted_{target_name}" in prediction_frame.columns
        ]
        if available_targets:
            hist_source = pd.DataFrame(
                {
                    target_name: prediction_frame[f"observed_{target_name}"].to_numpy(dtype=float)
                    for target_name in available_targets
                }
            )
            histogram_path = output_dir / "target_histograms.png"
            plot_target_distribution_histograms(hist_source, available_targets, histogram_path)
            generated_files.append(str(histogram_path))
        for target_name in available_targets:
            parity_path = output_dir / f"parity_{target_name}.png"
            plot_predicted_vs_observed(prediction_frame, target_name, parity_path)
            generated_files.append(str(parity_path))
            if f"std_{target_name}" in prediction_frame.columns:
                calibration_path = output_dir / f"calibration_{target_name}.png"
                plot_uncertainty_calibration_curve(prediction_frame, target_name, calibration_path)
                generated_files.append(str(calibration_path))
    ranking_path = input_root / "candidate_rankings" / "full_ranking.csv"
    if ranking_path.exists():
        ranking_frame = pd.read_csv(ranking_path)
        ranking_output = output_dir / "candidate_desirability_ranking.png"
        plot_candidate_desirability_ranking(ranking_frame, ranking_output)
        generated_files.append(str(ranking_output))
    manifest = {"generated_files": generated_files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("generated %s figure files", len(generated_files))


if __name__ == "__main__":
    main()

