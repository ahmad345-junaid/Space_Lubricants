from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the processed test split.")
    parser.add_argument("--model-config", default="configs/model_gnn_gpr.yaml", help="Path to model configuration.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment configuration.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import read_csv, write_csv
    from leo_lubricants.pipeline import evaluate_split, load_model_bundle
    from leo_lubricants.utils.config import load_config

    model_config = load_config(args.model_config)
    experiment_config = load_config(args.experiment_config)
    model, bundle = load_model_bundle(experiment_config["model_bundle_path"])
    split_frame = read_csv(Path(experiment_config["processed_dir"]) / f"{args.split}.csv")
    predictions, metrics = evaluate_split(
        split_frame=split_frame,
        model=model,
        bundle=bundle,
        max_nodes=int(model_config["max_nodes"]),
    )
    write_csv(predictions, experiment_config["evaluation_predictions_path"])
    output_path = Path(experiment_config["evaluation_metrics_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
