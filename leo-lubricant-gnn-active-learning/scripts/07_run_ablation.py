from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate ablation variants of the GNN model.")
    parser.add_argument("--config", default="configs/model_gnn_gpr.yaml", help="Path to the model configuration file.")
    parser.add_argument("--train", default="data/processed/train.csv", help="Path to processed training CSV.")
    parser.add_argument("--val", default="data/processed/val.csv", help="Path to processed validation CSV.")
    parser.add_argument("--output-dir", default="results/ablations", help="Directory for ablation outputs.")
    parser.add_argument("--no-environment-gate", action="store_true")
    parser.add_argument("--no-degradation-memory", action="store_true")
    parser.add_argument("--no-descriptors", action="store_true")
    parser.add_argument("--no-gpr-uncertainty", action="store_true")
    parser.add_argument("--no-physics-loss", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import read_table, write_table
    from leo_lubricants.data.schema import TARGET_COLUMNS
    from leo_lubricants.evaluation.metrics import aggregate_metric_table, per_target_metric_table
    from leo_lubricants.models.gpr_uncertainty import GPRConfig, MultiOutputGPRUncertainty
    from leo_lubricants.models.losses import total_loss
    from leo_lubricants.utils.config import load_config
    from leo_lubricants.utils.seed import set_global_seed
    from leo_lubricants.workflows import (
        build_model_from_batch,
        extract_target_statistics,
        load_checkpoint,
        prediction_dict_to_tensor,
        prepare_batch,
        save_checkpoint,
    )

    config = load_config(args.config)
    config["no_environment_gate"] = bool(args.no_environment_gate)
    config["no_degradation_memory"] = bool(args.no_degradation_memory)
    config["no_descriptors"] = bool(args.no_descriptors)
    config["no_gpr_uncertainty"] = bool(args.no_gpr_uncertainty)
    config["no_physics_loss"] = bool(args.no_physics_loss)
    variant_tokens = [
        token
        for token, active in {
            "no_environment_gate": args.no_environment_gate,
            "no_degradation_memory": args.no_degradation_memory,
            "no_descriptors": args.no_descriptors,
            "no_gpr_uncertainty": args.no_gpr_uncertainty,
            "no_physics_loss": args.no_physics_loss,
        }.items()
        if active
    ]
    variant_name = "baseline" if not variant_tokens else "__".join(variant_tokens)
    output_dir = Path(args.output_dir) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(config.get("seed", 7)))
    train_frame = read_table(args.train)
    val_frame = read_table(args.val)
    train_batch = prepare_batch(train_frame, include_targets=True)
    val_batch = prepare_batch(val_frame, include_targets=True)
    model = build_model_from_batch(config, train_batch)
    optimizer = optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    target_means, target_stds = extract_target_statistics(train_frame, list(TARGET_COLUMNS))
    best_val_loss = float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    for _ in range(int(config["epochs"])):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_batch.batch)
        train_loss = total_loss(
            train_predictions,
            train_batch.targets,
            train_batch.mask,
            train_batch.batch,
            config.get("task_weights"),
            None if bool(config.get("no_physics_loss", False)) else config.get("physics_weights"),
            float(config.get("l2_penalty", 0.0)),
            model.parameters(),
        )
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_batch.batch)
            val_loss = total_loss(
                val_predictions,
                val_batch.targets,
                val_batch.mask,
                val_batch.batch,
                config.get("task_weights"),
                None if bool(config.get("no_physics_loss", False)) else config.get("physics_weights"),
                float(config.get("l2_penalty", 0.0)),
                model.parameters(),
            )
        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())
            save_checkpoint(model, checkpoint_path, config, target_means, target_stds)
    model, payload = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        prediction_tensor = prediction_dict_to_tensor(model(val_batch.batch)).detach().cpu().numpy()
        embeddings = model.encode(val_batch.batch).detach().cpu().numpy()
    metrics_frame = per_target_metric_table(
        val_batch.raw_targets.to_numpy(dtype=float),
        np.column_stack(
            [
                prediction_tensor[:, index] * payload["target_stds"][target_name] + payload["target_means"][target_name]
                for index, target_name in enumerate(TARGET_COLUMNS)
            ]
        ),
        list(TARGET_COLUMNS),
    )
    write_table(metrics_frame, output_dir / "metrics_per_target.csv")
    write_table(aggregate_metric_table(metrics_frame), output_dir / "metrics_aggregate.csv")
    if not bool(config.get("no_gpr_uncertainty", False)):
        gpr_model = MultiOutputGPRUncertainty(
            GPRConfig(
                noise_level=float(config.get("gp_noise", 1.0e-4)),
                normalize_y=bool(config.get("gp_normalize_y", True)),
                max_fit_points=int(config.get("gp_max_fit_points", len(train_frame))),
                random_state=int(config.get("seed", 7)),
            )
        ).fit(embeddings, val_batch.targets.detach().cpu().numpy(), list(TARGET_COLUMNS))
        gpr_model.save(output_dir / "gpr_uncertainty.joblib")
    shutil.copyfile(args.config, output_dir / "config_snapshot.yaml")
    (output_dir / "summary.json").write_text(
        json.dumps({"variant": variant_name, "best_val_loss": best_val_loss}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
