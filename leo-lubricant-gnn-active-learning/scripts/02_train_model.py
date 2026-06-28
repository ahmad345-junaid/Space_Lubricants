from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from torch import optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the environment-conditioned GNN and fit GPR uncertainty models.")
    parser.add_argument("--config", default="configs/model_gnn_gpr.yaml", help="Path to the model configuration file.")
    parser.add_argument("--train", default="data/processed/train.csv", help="Path to the processed training CSV file.")
    parser.add_argument("--val", default="data/processed/val.csv", help="Path to the processed validation CSV file.")
    parser.add_argument("--output-dir", default="results/trained_models", help="Directory for checkpoints and training artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.io import read_table, write_table
    from leo_lubricants.data.schema import TARGET_COLUMNS
    from leo_lubricants.models.gpr_uncertainty import GPRConfig, MultiOutputGPRUncertainty
    from leo_lubricants.models.losses import masked_mse_loss, physics_guided_penalty, total_loss
    from leo_lubricants.utils.config import load_config
    from leo_lubricants.utils.logging import get_logger
    from leo_lubricants.utils.seed import set_global_seed
    from leo_lubricants.workflows import (
        build_model_from_batch,
        denormalize_prediction_frame,
        extract_target_statistics,
        prediction_dict_to_tensor,
        prepare_batch,
        save_checkpoint,
    )

    logger = get_logger("train")
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(config.get("seed", 7)))
    device = torch.device("cuda" if torch.cuda.is_available() and bool(config.get("use_cuda", False)) else "cpu")
    train_frame = read_table(args.train)
    val_frame = read_table(args.val)
    train_batch = prepare_batch(train_frame, device=device, include_targets=True)
    val_batch = prepare_batch(val_frame, device=device, include_targets=True)
    model = build_model_from_batch(config, train_batch).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    target_means, target_stds = extract_target_statistics(train_frame, list(TARGET_COLUMNS))
    history_rows: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"
    task_weights = config.get("task_weights")
    physics_weights = config.get("physics_weights")
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_batch.batch)
        train_loss = total_loss(
            predictions=train_predictions,
            targets=train_batch.targets,
            mask=train_batch.mask,
            batch=train_batch.batch,
            task_weights=task_weights,
            physics_weights=None if bool(config.get("no_physics_loss", False)) else physics_weights,
            l2_penalty=float(config.get("l2_penalty", 0.0)),
            parameters=model.parameters(),
        )
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_batch.batch)
            val_loss = total_loss(
                predictions=val_predictions,
                targets=val_batch.targets,
                mask=val_batch.mask,
                batch=val_batch.batch,
                task_weights=task_weights,
                physics_weights=None if bool(config.get("no_physics_loss", False)) else physics_weights,
                l2_penalty=float(config.get("l2_penalty", 0.0)),
                parameters=model.parameters(),
            )
            train_mse = masked_mse_loss(train_predictions, train_batch.targets, train_batch.mask, task_weights)
            val_mse = masked_mse_loss(val_predictions, val_batch.targets, val_batch.mask, task_weights)
            train_physics = physics_guided_penalty(train_predictions, train_batch.batch, None if bool(config.get("no_physics_loss", False)) else physics_weights)
            val_physics = physics_guided_penalty(val_predictions, val_batch.batch, None if bool(config.get("no_physics_loss", False)) else physics_weights)
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss.item()),
                "train_mse": float(train_mse.item()),
                "val_mse": float(val_mse.item()),
                "train_physics": float(train_physics.item()),
                "val_physics": float(val_physics.item()),
            }
        )
        logger.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, float(train_loss.item()), float(val_loss.item()))
        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())
            save_checkpoint(model, best_checkpoint_path, config, target_means, target_stds)
    history_frame = pd.DataFrame(history_rows)
    write_table(history_frame, output_dir / "training_history.csv")
    shutil.copyfile(args.config, output_dir / "config_snapshot.yaml")
    checkpoint_model, checkpoint_payload_dict = None, None
    from leo_lubricants.workflows import load_checkpoint

    checkpoint_model, checkpoint_payload_dict = load_checkpoint(best_checkpoint_path, map_location=device)
    checkpoint_model = checkpoint_model.to(device)
    checkpoint_model.eval()
    with torch.no_grad():
        train_embeddings = checkpoint_model.encode(train_batch.batch).detach().cpu().numpy()
        val_embeddings = checkpoint_model.encode(val_batch.batch).detach().cpu().numpy()
        val_prediction_tensor = prediction_dict_to_tensor(checkpoint_model(val_batch.batch)).detach().cpu().numpy()
    gpr_model = MultiOutputGPRUncertainty(
        GPRConfig(
            noise_level=float(config.get("gp_noise", 1.0e-4)),
            normalize_y=bool(config.get("gp_normalize_y", True)),
            max_fit_points=int(config.get("gp_max_fit_points", len(train_frame))),
            random_state=int(config.get("seed", 7)),
        )
    ).fit(
        embeddings=train_embeddings,
        targets=train_batch.targets.detach().cpu().numpy(),
        target_names=list(TARGET_COLUMNS),
    )
    if not bool(config.get("no_gpr_uncertainty", False)):
        gpr_model.save(output_dir / "gpr_uncertainty.joblib")
        gpr_mean_frame, gpr_std_frame = gpr_model.predict(val_embeddings)
    else:
        gpr_mean_frame = pd.DataFrame(val_prediction_tensor, columns=list(TARGET_COLUMNS))
        gpr_std_frame = pd.DataFrame(np.zeros_like(val_prediction_tensor), columns=list(TARGET_COLUMNS))
    validation_predictions = pd.DataFrame({"molecule_id": val_frame["molecule_id"].astype(str), "smiles": val_frame["smiles"].astype(str)})
    for index, target_name in enumerate(TARGET_COLUMNS):
        validation_predictions[f"observed_{target_name}"] = val_batch.raw_targets[target_name].to_numpy(dtype=float)
        validation_predictions[f"predicted_{target_name}"] = val_prediction_tensor[:, index]
        validation_predictions[f"gpr_mean_{target_name}"] = gpr_mean_frame[target_name].to_numpy(dtype=float)
        validation_predictions[f"std_{target_name}"] = gpr_std_frame[target_name].to_numpy(dtype=float)
    validation_predictions = denormalize_prediction_frame(
        validation_predictions,
        target_means=checkpoint_payload_dict["target_means"],
        target_stds=checkpoint_payload_dict["target_stds"],
        prefixes=("predicted_", "gpr_mean_", "std_"),
    )
    write_table(validation_predictions, output_dir / "validation_predictions.csv")


if __name__ == "__main__":
    main()
