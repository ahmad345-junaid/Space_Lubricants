from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN checkpoint on a processed test split.")
    parser.add_argument("--config", default="configs/model_gnn_gpr.yaml", help="Path to the model configuration file.")
    parser.add_argument("--test", default="data/processed/test.csv", help="Path to the processed test CSV file.")
    parser.add_argument("--checkpoint", default="results/trained_models/best_model.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--gpr", default=None, help="Optional path to a fitted GPR artifact.")
    parser.add_argument("--output-dir", default="results/metrics", help="Directory for evaluation outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import read_table, write_table
    from leo_lubricants.data.schema import TARGET_COLUMNS
    from leo_lubricants.evaluation.calibration import (
        gaussian_negative_log_likelihood,
        prediction_interval_coverage_probability,
        regression_calibration_error,
    )
    from leo_lubricants.evaluation.metrics import aggregate_metric_table, per_target_metric_table
    from leo_lubricants.models.gpr_uncertainty import MultiOutputGPRUncertainty
    from leo_lubricants.utils.config import load_config
    from leo_lubricants.utils.logging import get_logger
    from leo_lubricants.workflows import denormalize_prediction_frame, load_checkpoint, prediction_dict_to_tensor, prepare_batch

    logger = get_logger("evaluate")
    load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    test_frame = read_table(args.test)
    test_batch = prepare_batch(test_frame, device=device, include_targets=True)
    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        prediction_tensor = prediction_dict_to_tensor(model(test_batch.batch)).detach().cpu().numpy()
        embeddings = model.encode(test_batch.batch).detach().cpu().numpy()
    predictions = pd.DataFrame({"molecule_id": test_frame["molecule_id"].astype(str), "smiles": test_frame["smiles"].astype(str)})
    for index, target_name in enumerate(TARGET_COLUMNS):
        predictions[f"observed_{target_name}"] = test_batch.raw_targets[target_name].to_numpy(dtype=float)
        predictions[f"predicted_{target_name}"] = prediction_tensor[:, index]
    uncertainty_summary: dict[str, float] = {}
    if args.gpr:
        gpr_model = MultiOutputGPRUncertainty.load(args.gpr)
        gpr_mean_frame, gpr_std_frame = gpr_model.predict(embeddings)
        for target_name in TARGET_COLUMNS:
            predictions[f"gpr_mean_{target_name}"] = gpr_mean_frame[target_name].to_numpy(dtype=float)
            predictions[f"std_{target_name}"] = gpr_std_frame[target_name].to_numpy(dtype=float)
        predictions = denormalize_prediction_frame(
            predictions,
            target_means=payload["target_means"],
            target_stds=payload["target_stds"],
            prefixes=("predicted_", "gpr_mean_", "std_"),
        )
        observed_raw = np.column_stack([predictions[f"observed_{name}"].to_numpy(dtype=float) for name in TARGET_COLUMNS])
        gpr_mean_raw = np.column_stack([predictions[f"gpr_mean_{name}_raw"].to_numpy(dtype=float) for name in TARGET_COLUMNS])
        gpr_std_raw = np.column_stack([predictions[f"std_{name}_raw"].to_numpy(dtype=float) for name in TARGET_COLUMNS])
        uncertainty_summary = {
            "prediction_interval_coverage_probability": prediction_interval_coverage_probability(observed_raw, gpr_mean_raw, gpr_std_raw),
            "regression_calibration_error": regression_calibration_error(observed_raw, gpr_mean_raw, gpr_std_raw),
            "gaussian_negative_log_likelihood": gaussian_negative_log_likelihood(observed_raw, gpr_mean_raw, gpr_std_raw),
        }
    else:
        predictions = denormalize_prediction_frame(
            predictions,
            target_means=payload["target_means"],
            target_stds=payload["target_stds"],
            prefixes=("predicted_",),
        )
    y_true = np.column_stack([predictions[f"observed_{name}"].to_numpy(dtype=float) for name in TARGET_COLUMNS])
    y_pred = np.column_stack([predictions[f"predicted_{name}_raw"].to_numpy(dtype=float) for name in TARGET_COLUMNS])
    metric_table = per_target_metric_table(y_true, y_pred, list(TARGET_COLUMNS))
    aggregate_table = aggregate_metric_table(metric_table)
    write_table(predictions, output_dir / "predictions.csv")
    write_table(metric_table, output_dir / "metrics_per_target.csv")
    write_table(aggregate_table, output_dir / "metrics_aggregate.csv")
    metrics_payload = {
        "per_target": metric_table.set_index("target").to_dict(orient="index"),
        "aggregate": aggregate_table.iloc[0].to_dict(),
    }
    if uncertainty_summary:
        metrics_payload["uncertainty"] = uncertainty_summary
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("saved evaluation outputs to %s", output_dir)


if __name__ == "__main__":
    main()

