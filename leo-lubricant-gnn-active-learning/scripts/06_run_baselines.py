from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate descriptor-environment baseline regressors.")
    parser.add_argument("--train", default="data/processed/train.csv", help="Path to processed training CSV.")
    parser.add_argument("--eval", default="data/processed/val.csv", help="Path to processed evaluation CSV.")
    parser.add_argument("--output-dir", default="results/metrics/baselines", help="Directory for baseline outputs.")
    parser.add_argument("--seed", default=7, type=int, help="Random seed for baseline models.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import read_table, write_table
    from leo_lubricants.data.schema import ENVIRONMENT_COLUMNS, TARGET_COLUMNS
    from leo_lubricants.evaluation.metrics import per_target_metric_table
    from leo_lubricants.features.descriptors import DESCRIPTOR_COLUMNS
    from leo_lubricants.features.environment import DEGRADATION_MEMORY_COLUMNS
    from leo_lubricants.models.baselines import build_baselines

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame = read_table(args.train)
    eval_frame = read_table(args.eval)
    feature_columns = list(DESCRIPTOR_COLUMNS) + list(ENVIRONMENT_COLUMNS) + list(DEGRADATION_MEMORY_COLUMNS)
    train_targets = train_frame[[f"raw_{target}" if f"raw_{target}" in train_frame.columns else target for target in TARGET_COLUMNS]].copy()
    train_targets.columns = list(TARGET_COLUMNS)
    eval_targets = eval_frame[[f"raw_{target}" if f"raw_{target}" in eval_frame.columns else target for target in TARGET_COLUMNS]].copy()
    eval_targets.columns = list(TARGET_COLUMNS)
    metrics_rows: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for name, baseline in build_baselines(args.seed).items():
        fitted = baseline.fit(train_frame[feature_columns], train_targets)
        predictions = fitted.predict(eval_frame[feature_columns])
        prediction_output = pd.DataFrame({"model": name, "molecule_id": eval_frame["molecule_id"].astype(str)})
        for target_name in TARGET_COLUMNS:
            prediction_output[f"observed_{target_name}"] = eval_targets[target_name].to_numpy(dtype=float)
            prediction_output[f"predicted_{target_name}"] = predictions[target_name].to_numpy(dtype=float)
        prediction_tables.append(prediction_output)
        metric_table = per_target_metric_table(eval_targets.to_numpy(dtype=float), predictions.to_numpy(dtype=float), list(TARGET_COLUMNS))
        metric_table.insert(0, "model", name)
        metrics_rows.append(metric_table)
    metrics_frame = pd.concat(metrics_rows, ignore_index=True)
    predictions_frame = pd.concat(prediction_tables, ignore_index=True)
    write_table(metrics_frame, output_dir / "baseline_metrics.csv")
    write_table(predictions_frame, output_dir / "baseline_predictions.csv")
    summary = metrics_frame.groupby("model")[["rmse", "mae", "r2"]].mean().reset_index()
    write_table(summary, output_dir / "baseline_metrics_summary.csv")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(summary.set_index("model").to_dict(orient="index"), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
