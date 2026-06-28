from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidate lubricant molecules using uncertainty-aware active learning.")
    parser.add_argument("--model-config", default="configs/model_gnn_gpr.yaml", help="Path to model configuration.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment configuration.")
    parser.add_argument("--active-learning-config", default="configs/active_learning.yaml", help="Path to active learning configuration.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import write_csv
    from leo_lubricants.pipeline import load_model_bundle, rank_candidate_pool
    from leo_lubricants.utils.config import load_config

    model_config = load_config(args.model_config)
    experiment_config = load_config(args.experiment_config)
    active_learning_config = load_config(args.active_learning_config)
    model, bundle = load_model_bundle(experiment_config["model_bundle_path"])
    ranked = rank_candidate_pool(
        candidate_path=active_learning_config["candidate_path"],
        model=model,
        bundle=bundle,
        max_nodes=int(model_config["max_nodes"]),
        exploitation_weight=float(active_learning_config["exploitation_weight"]),
        exploration_weight=float(active_learning_config["exploration_weight"]),
    )
    top_k = int(active_learning_config["top_k"])
    write_csv(ranked.head(top_k), active_learning_config["output_path"])


if __name__ == "__main__":
    main()
