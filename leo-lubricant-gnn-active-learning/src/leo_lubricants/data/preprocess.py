from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leo_lubricants.data.io import read_table, write_table
from leo_lubricants.data.schema import (
    CANONICAL_SMILES_COLUMN,
    DatasetSchema,
    INCHIKEY_COLUMN,
    TARGET_COLUMNS,
    validate_columns,
)
from leo_lubricants.data.split import molecule_disjoint_split, scaffold_disjoint_split
from leo_lubricants.features.descriptors import add_descriptors, canonicalize_smiles
from leo_lubricants.features.environment import compute_degradation_memory_features


@dataclass
class PreprocessingArtifacts:
    target_columns: list[str]
    target_means: dict[str, float]
    target_stds: dict[str, float]
    retained_columns: list[str]
    split_strategy: str


def _canonicalize_frame(
    df: pd.DataFrame,
    smiles_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = row.to_dict()
        smiles = str(record.get(smiles_column, ""))
        try:
            record[CANONICAL_SMILES_COLUMN] = canonicalize_smiles(smiles)
            valid_rows.append(record)
        except ValueError as error:
            record["rejection_reason"] = str(error)
            rejected_rows.append(record)
    valid_frame = pd.DataFrame(valid_rows)
    rejected_frame = pd.DataFrame(rejected_rows)
    return valid_frame, rejected_frame


def _normalize_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, float]]:
    means = train_df[target_columns].mean(axis=0).to_dict()
    stds_series = train_df[target_columns].std(axis=0, ddof=0).replace(0.0, 1.0)
    stds = stds_series.to_dict()
    normalized_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        normalized = split_df.copy()
        for target in target_columns:
            normalized[f"raw_{target}"] = normalized[target]
            normalized[target] = (normalized[target] - means[target]) / stds[target]
        normalized_splits[split_name] = normalized
    return normalized_splits, {key: float(value) for key, value in means.items()}, {key: float(value) for key, value in stds.items()}


def _split_dataset(
    df: pd.DataFrame,
    split_strategy: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: str,
    smiles_column: str,
) -> tuple[dict[str, pd.DataFrame], str]:
    if split_strategy == "scaffold":
        try:
            return (
                scaffold_disjoint_split(
                    df=df,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    seed=seed,
                    smiles_column=smiles_column,
                ),
                "scaffold",
            )
        except ValueError:
            return (
                molecule_disjoint_split(
                    df=df,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size,
                    seed=seed,
                    group_column=group_column,
                ),
                "molecule_disjoint",
            )
    if split_strategy == "molecule_disjoint":
        return (
            molecule_disjoint_split(
                df=df,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                group_column=group_column,
            ),
            "molecule_disjoint",
        )
    raise ValueError("split_strategy must be either 'scaffold' or 'molecule_disjoint'.")


def preprocess_dataframe(
    frame: pd.DataFrame,
    schema: DatasetSchema,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(frame, schema.required_supervised_columns)
    canonicalized, rejected = _canonicalize_frame(frame, schema.smiles_column)
    if canonicalized.empty:
        raise ValueError("All rows were rejected during SMILES canonicalization.")
    with_descriptors = add_descriptors(canonicalized, smiles_column=CANONICAL_SMILES_COLUMN)
    enriched = compute_degradation_memory_features(with_descriptors)
    if INCHIKEY_COLUMN not in enriched.columns:
        raise ValueError("Descriptor generation did not produce InChIKey values.")
    return enriched, rejected


def preprocess_candidates(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    validate_columns(frame, schema.required_inference_columns)
    canonicalized, rejected = _canonicalize_frame(frame, schema.smiles_column)
    if not rejected.empty:
        canonicalized = canonicalized.reset_index(drop=True)
    if canonicalized.empty:
        raise ValueError("All candidate rows were rejected during SMILES canonicalization.")
    with_descriptors = add_descriptors(canonicalized, smiles_column=CANONICAL_SMILES_COLUMN)
    return compute_degradation_memory_features(with_descriptors)


def run_preprocessing(
    input_path: str | Path,
    output_dir: str | Path,
    split_strategy: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: str,
) -> PreprocessingArtifacts:
    schema = DatasetSchema()
    raw_frame = read_table(input_path)
    processed, rejected = preprocess_dataframe(raw_frame, schema)
    splits, resolved_strategy = _split_dataset(
        df=processed,
        split_strategy=split_strategy,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        group_column=group_column,
        smiles_column=schema.smiles_column,
    )
    normalized_splits, means, stds = _normalize_targets(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        target_columns=list(TARGET_COLUMNS),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in normalized_splits.items():
        write_table(split_df, output_root / f"{split_name}.csv")
    write_table(rejected, output_root / "rejected_rows.csv")
    retained_columns = list(normalized_splits["train"].columns)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "requested_split_strategy": split_strategy,
        "split_strategy": resolved_strategy,
        "group_column": group_column,
        "target_means": means,
        "target_stds": stds,
        "retained_columns": retained_columns,
        "dropped_rows": int(len(rejected)),
        "train_rows": int(len(normalized_splits["train"])),
        "val_rows": int(len(normalized_splits["val"])),
        "test_rows": int(len(normalized_splits["test"])),
    }
    (output_root / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return PreprocessingArtifacts(
        target_columns=list(TARGET_COLUMNS),
        target_means=means,
        target_stds=stds,
        retained_columns=retained_columns,
        split_strategy=resolved_strategy,
    )

