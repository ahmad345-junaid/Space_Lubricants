from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from leo_lubricants.data.molecular_graphs import batch_graphs, smiles_to_graph
from leo_lubricants.data.schema import (
    CANONICAL_SMILES_COLUMN,
    ENVIRONMENT_COLUMNS,
    INCHIKEY_COLUMN,
    MOLECULE_ID_COLUMN,
    SMILES_COLUMN,
    TARGET_COLUMNS,
)
from leo_lubricants.features.descriptors import DESCRIPTOR_COLUMNS
from leo_lubricants.features.environment import DEGRADATION_MEMORY_COLUMNS
from leo_lubricants.models.gnn import EnvironmentConditionedGNN, GNNConfig


@dataclass
class PreparedBatch:
    frame: pd.DataFrame
    batch: dict[str, Any]
    targets: torch.Tensor | None
    mask: torch.Tensor | None
    raw_targets: pd.DataFrame | None


def raw_target_columns(target_names: list[str] | None = None) -> list[str]:
    active_targets = target_names or list(TARGET_COLUMNS)
    return [f"raw_{target}" for target in active_targets]


def extract_target_statistics(frame: pd.DataFrame, target_names: list[str] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    active_targets = target_names or list(TARGET_COLUMNS)
    raw_columns = raw_target_columns(active_targets)
    if all(column in frame.columns for column in raw_columns):
        raw_frame = frame[raw_columns].copy()
        raw_frame.columns = active_targets
    else:
        raw_frame = frame[active_targets].copy()
    means = {column: float(raw_frame[column].mean()) for column in active_targets}
    stds = {column: float(raw_frame[column].std(ddof=0)) if float(raw_frame[column].std(ddof=0)) > 0.0 else 1.0 for column in active_targets}
    return means, stds


def denormalize_prediction_frame(
    prediction_frame: pd.DataFrame,
    target_means: dict[str, float],
    target_stds: dict[str, float],
    prefixes: tuple[str, ...],
) -> pd.DataFrame:
    denormalized = prediction_frame.copy()
    for prefix in prefixes:
        for target_name, mean_value in target_means.items():
            source = f"{prefix}{target_name}"
            if source in denormalized.columns:
                scale = target_stds[target_name]
                denormalized[f"{source}_raw"] = denormalized[source] * scale + mean_value
    return denormalized


def prediction_dict_to_tensor(
    predictions: dict[str, torch.Tensor],
    target_names: list[str] | None = None,
) -> torch.Tensor:
    active_targets = target_names or list(TARGET_COLUMNS)
    return torch.stack([predictions[target] for target in active_targets], dim=1)


def _group_values(frame: pd.DataFrame) -> pd.Series:
    if INCHIKEY_COLUMN in frame.columns:
        return frame[INCHIKEY_COLUMN].astype(str)
    if CANONICAL_SMILES_COLUMN in frame.columns:
        return frame[CANONICAL_SMILES_COLUMN].astype(str)
    return frame[SMILES_COLUMN].astype(str)


def prepare_batch(
    frame: pd.DataFrame,
    device: str | torch.device = "cpu",
    include_targets: bool = True,
    target_names: list[str] | None = None,
) -> PreparedBatch:
    active_targets = target_names or list(TARGET_COLUMNS)
    graph_objects = [
        smiles_to_graph(smiles)
        for smiles in frame[CANONICAL_SMILES_COLUMN if CANONICAL_SMILES_COLUMN in frame.columns else SMILES_COLUMN].astype(str)
    ]
    graph_batch = batch_graphs(graph_objects)
    descriptor_matrix = torch.tensor(frame[DESCRIPTOR_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    environment_matrix = torch.tensor(frame[ENVIRONMENT_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    degradation_matrix = torch.tensor(frame[DEGRADATION_MEMORY_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    group_series = _group_values(frame)
    group_lookup = {value: index for index, value in enumerate(group_series.unique())}
    graph_batch["descriptor_matrix"] = descriptor_matrix
    graph_batch["environment_matrix"] = environment_matrix
    graph_batch["degradation_memory_matrix"] = degradation_matrix
    graph_batch["molecule_ids"] = frame[MOLECULE_ID_COLUMN].astype(str).tolist()
    graph_batch["smiles"] = frame[SMILES_COLUMN].astype(str).tolist()
    graph_batch["pair_group_ids"] = torch.tensor(group_series.map(group_lookup).to_numpy(), dtype=torch.long)
    graph_batch["temperature_k"] = torch.tensor(
        ((frame["t_min_k"].to_numpy(dtype=np.float32) + frame["t_max_k"].to_numpy(dtype=np.float32)) / 2.0),
        dtype=torch.float32,
    )
    graph_batch["atomic_oxygen_fluence"] = torch.tensor(
        frame["atomic_oxygen_fluence"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    graph_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph_batch.items()
    }
    targets = None
    mask = None
    raw_targets = None
    if include_targets:
        target_frame = frame[active_targets].copy()
        mask_array = target_frame.notna().to_numpy(dtype=np.float32)
        target_array = target_frame.fillna(0.0).to_numpy(dtype=np.float32)
        targets = torch.tensor(target_array, dtype=torch.float32, device=device)
        mask = torch.tensor(mask_array, dtype=torch.float32, device=device)
        raw_columns = raw_target_columns(active_targets)
        if all(column in frame.columns for column in raw_columns):
            raw_targets = frame[raw_columns].copy()
            raw_targets.columns = active_targets
        else:
            raw_targets = frame[active_targets].copy()
    return PreparedBatch(
        frame=frame.reset_index(drop=True),
        batch=graph_batch,
        targets=targets,
        mask=mask,
        raw_targets=raw_targets,
    )


def build_model_from_batch(config_values: dict[str, Any], batch: PreparedBatch) -> EnvironmentConditionedGNN:
    gnn_config = GNNConfig(
        atom_feature_dim=int(batch.batch["node_features"].shape[1]),
        bond_feature_dim=int(batch.batch["edge_features"].shape[1]) if batch.batch["edge_features"].numel() > 0 else 10,
        descriptor_dim=int(batch.batch["descriptor_matrix"].shape[1]),
        environment_dim=int(batch.batch["environment_matrix"].shape[1]),
        degradation_memory_dim=int(batch.batch["degradation_memory_matrix"].shape[1]),
        hidden_dim=int(config_values["hidden_dim"]),
        message_passing_steps=int(config_values["message_passing_steps"]),
        dropout=float(config_values["dropout"]),
        head_hidden_dim=int(config_values.get("head_hidden_dim", config_values["hidden_dim"])),
        target_names=list(TARGET_COLUMNS),
        use_environment_gate=not bool(config_values.get("no_environment_gate", False)),
        use_descriptors=not bool(config_values.get("no_descriptors", False)),
        use_degradation_memory=not bool(config_values.get("no_degradation_memory", False)),
    )
    return EnvironmentConditionedGNN(gnn_config)


def checkpoint_payload(
    model: EnvironmentConditionedGNN,
    model_config: dict[str, Any],
    target_means: dict[str, float],
    target_stds: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": dict(model_config),
        "target_means": target_means,
        "target_stds": target_stds,
        "target_names": list(TARGET_COLUMNS),
        "descriptor_columns": list(DESCRIPTOR_COLUMNS),
        "environment_columns": list(ENVIRONMENT_COLUMNS),
        "degradation_memory_columns": list(DEGRADATION_MEMORY_COLUMNS),
    }


def save_checkpoint(
    model: EnvironmentConditionedGNN,
    path: str | Path,
    model_config: dict[str, Any],
    target_means: dict[str, float],
    target_stds: dict[str, float],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, model_config, target_means, target_stds), target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[EnvironmentConditionedGNN, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    model = EnvironmentConditionedGNN(GNNConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
