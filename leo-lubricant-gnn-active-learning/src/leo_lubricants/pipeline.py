from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import optim

from leo_lubricants.active_learning.select_candidates import rank_candidates
from leo_lubricants.data.io import read_csv, read_joblib, write_csv, write_joblib
from leo_lubricants.data.molecular_graphs import GraphBatch, build_graph_batch, iterate_minibatches
from leo_lubricants.data.preprocess import PreprocessingArtifacts, preprocess_candidates, preprocess_dataframe
from leo_lubricants.data.schema import DatasetSchema, TARGET_COLUMNS
from leo_lubricants.data.split import SplitConfig, split_dataframe
from leo_lubricants.evaluation.calibration import interval_coverage, negative_log_likelihood
from leo_lubricants.evaluation.metrics import regression_metrics
from leo_lubricants.models.gnn import EnvironmentConditionedGNN
from leo_lubricants.models.gpr_uncertainty import MultiTargetGaussianProcessRegressor
from leo_lubricants.models.losses import multitask_mse_loss
from leo_lubricants.utils.seed import set_global_seed


@dataclass
class ModelBundle:
    model_state: dict[str, Any]
    feature_columns: list[str]
    target_columns: list[str]
    scaler_artifacts: PreprocessingArtifacts
    model_config: dict[str, Any]
    gp_model: MultiTargetGaussianProcessRegressor


def preprocess_and_split(
    input_path: str | Path,
    output_dir: str | Path,
    random_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, pd.DataFrame], PreprocessingArtifacts]:
    schema = DatasetSchema()
    frame = read_csv(input_path)
    processed, artifacts = preprocess_dataframe(frame, schema=schema, fit_scaler=True)
    splits = split_dataframe(
        processed,
        SplitConfig(
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
        ),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, split_frame in splits.items():
        write_csv(split_frame, output_root / f"{split_name}.csv")
    write_joblib(artifacts, output_root / "preprocessing_artifacts.joblib")
    return splits, artifacts


def _build_model(config: dict[str, Any], feature_dim: int, output_dim: int, node_feature_dim: int) -> EnvironmentConditionedGNN:
    return EnvironmentConditionedGNN(
        node_feature_dim=node_feature_dim,
        global_feature_dim=feature_dim,
        hidden_dim=int(config["hidden_dim"]),
        message_passing_steps=int(config["message_passing_steps"]),
        output_dim=output_dim,
        dropout=float(config["dropout"]),
    )


def _predict(model: EnvironmentConditionedGNN, batch: GraphBatch) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        predictions, embeddings = model(
            batch.node_features,
            batch.adjacency,
            batch.node_mask,
            batch.global_features,
        )
    return predictions.cpu().numpy(), embeddings.cpu().numpy()


def train_from_processed(
    processed_dir: str | Path,
    model_config: dict[str, Any],
    model_bundle_path: str | Path,
) -> ModelBundle:
    set_global_seed(int(model_config["random_seed"]))
    train_frame = read_csv(Path(processed_dir) / "train.csv")
    feature_columns = [
        column
        for column in train_frame.columns
        if column.startswith("descriptor_") or column.startswith("env_")
    ]
    target_columns = list(TARGET_COLUMNS)
    batch = build_graph_batch(
        train_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        max_nodes=int(model_config["max_nodes"]),
    )
    node_feature_dim = int(batch.node_features.shape[-1])
    model = _build_model(model_config, len(feature_columns), len(target_columns), node_feature_dim)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    for _ in range(int(model_config["epochs"])):
        model.train()
        for minibatch in iterate_minibatches(batch, int(model_config["batch_size"])):
            optimizer.zero_grad()
            predictions, _ = model(
                minibatch.node_features,
                minibatch.adjacency,
                minibatch.node_mask,
                minibatch.global_features,
            )
            loss = multitask_mse_loss(predictions, minibatch.targets)
            loss.backward()
            optimizer.step()

    train_predictions, embeddings = _predict(model, batch)
    gp_model = MultiTargetGaussianProcessRegressor(
        max_points=int(model_config["gp_max_points"]),
        noise=float(model_config["gp_noise"]),
    ).fit(embeddings, batch.targets.cpu().numpy())
    artifacts = read_joblib(Path(processed_dir) / "preprocessing_artifacts.joblib")
    bundle = ModelBundle(
        model_state=model.state_dict(),
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler_artifacts=artifacts,
        model_config=model_config,
        gp_model=gp_model,
    )
    write_joblib(bundle, model_bundle_path)
    training_predictions = train_frame[["molecule_id"]].copy()
    for index, target in enumerate(target_columns):
        training_predictions[f"pred_{target}"] = train_predictions[:, index]
    write_csv(training_predictions, Path(model_bundle_path).parent.parent / "metrics" / "training_predictions.csv")
    return bundle


def load_model_bundle(path: str | Path) -> tuple[EnvironmentConditionedGNN, ModelBundle]:
    bundle: ModelBundle = read_joblib(path)
    dummy_batch = build_graph_batch(
        pd.DataFrame(
            {
                "molecule_id": ["tmp"],
                "smiles": ["CCO"],
                **{column: [0.0] for column in bundle.feature_columns},
                **{column: [0.0] for column in bundle.target_columns},
            }
        ),
        feature_columns=bundle.feature_columns,
        target_columns=bundle.target_columns,
        max_nodes=int(bundle.model_config["max_nodes"]),
    )
    node_feature_dim = int(dummy_batch.node_features.shape[-1])
    model = _build_model(
        bundle.model_config,
        feature_dim=len(bundle.feature_columns),
        output_dim=len(bundle.target_columns),
        node_feature_dim=node_feature_dim,
    )
    model.load_state_dict(bundle.model_state)
    model.eval()
    return model, bundle


def evaluate_split(
    split_frame: pd.DataFrame,
    model: EnvironmentConditionedGNN,
    bundle: ModelBundle,
    max_nodes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    batch = build_graph_batch(
        split_frame,
        feature_columns=bundle.feature_columns,
        target_columns=bundle.target_columns,
        max_nodes=max_nodes,
    )
    nn_predictions, embeddings = _predict(model, batch)
    gp_means, gp_stds = bundle.gp_model.predict(embeddings)
    combined_predictions = 0.5 * nn_predictions + 0.5 * gp_means
    observed = batch.targets.cpu().numpy()
    metrics = regression_metrics(observed, combined_predictions, bundle.target_columns)
    metrics["calibration"] = {
        "interval_coverage": interval_coverage(observed, gp_means, gp_stds),
        "negative_log_likelihood": negative_log_likelihood(observed, gp_means, gp_stds),
    }
    predictions = split_frame[["molecule_id"]].copy()
    for index, target in enumerate(bundle.target_columns):
        predictions[f"observed_{target}"] = observed[:, index]
        predictions[f"predicted_{target}"] = combined_predictions[:, index]
        predictions[f"std_{target}"] = gp_stds[:, index]
    return predictions, metrics


def rank_candidate_pool(
    candidate_path: str | Path,
    model: EnvironmentConditionedGNN,
    bundle: ModelBundle,
    max_nodes: int,
    exploitation_weight: float,
    exploration_weight: float,
) -> pd.DataFrame:
    schema = DatasetSchema()
    candidate_frame = read_csv(candidate_path)
    processed = preprocess_candidates(candidate_frame, schema=schema, artifacts=bundle.scaler_artifacts)
    batch = build_graph_batch(
        processed,
        feature_columns=bundle.feature_columns,
        target_columns=None,
        max_nodes=max_nodes,
    )
    nn_predictions, embeddings = _predict(model, batch)
    gp_means, gp_stds = bundle.gp_model.predict(embeddings)
    combined_predictions = 0.5 * nn_predictions + 0.5 * gp_means
    scored = processed.copy()
    for index, target in enumerate(bundle.target_columns):
        scored[f"pred_{target}"] = combined_predictions[:, index]
        scored[f"std_{target}"] = gp_stds[:, index]
    ranked = rank_candidates(
        candidate_frame=scored,
        prediction_means=combined_predictions,
        prediction_stds=gp_stds,
        exploitation_weight=exploitation_weight,
        exploration_weight=exploration_weight,
    )
    return ranked
