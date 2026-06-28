from __future__ import annotations

from pathlib import Path

import pandas as pd

from leo_lubricants.data.preprocess import run_preprocessing
from leo_lubricants.data.schema import DatasetSchema
from leo_lubricants.workflows import load_checkpoint, prepare_batch


def preprocess_and_split(
    input_path: str | Path,
    output_dir: str | Path,
    random_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    run_preprocessing(
        input_path=input_path,
        output_dir=output_dir,
        split_strategy="scaffold",
        train_size=train_fraction,
        val_size=val_fraction,
        test_size=test_fraction,
        seed=random_seed,
        group_column="inchikey",
    )


def load_model_bundle(path: str | Path):
    return load_checkpoint(path)


def encode_frame(frame: pd.DataFrame, checkpoint_path: str | Path):
    model, _ = load_checkpoint(checkpoint_path)
    batch = prepare_batch(frame, include_targets=False)
    return model.encode(batch.batch).detach().cpu().numpy()


def preprocess_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from leo_lubricants.data.preprocess import preprocess_candidates

    return preprocess_candidates(frame, DatasetSchema())

