from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the environment-conditioned GNN and Gaussian-process uncertainty model.")
    parser.add_argument("--model-config", default="configs/model_gnn_gpr.yaml", help="Path to model configuration.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.pipeline import train_from_processed
    from leo_lubricants.utils.config import load_config

    model_config = load_config(args.model_config)
    experiment_config = load_config(args.experiment_config)
    train_from_processed(
        processed_dir=experiment_config["processed_dir"],
        model_config=model_config,
        model_bundle_path=experiment_config["model_bundle_path"],
    )


if __name__ == "__main__":
    main()
