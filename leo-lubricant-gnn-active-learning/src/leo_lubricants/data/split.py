from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from leo_lubricants.data.schema import CANONICAL_SMILES_COLUMN, INCHIKEY_COLUMN, SMILES_COLUMN
from leo_lubricants.features.descriptors import canonicalize_smiles


def _normalize_sizes(train_size: float, val_size: float, test_size: float) -> tuple[float, float, float]:
    total = train_size + val_size + test_size
    if total <= 0:
        raise ValueError("Split sizes must sum to a positive value.")
    return train_size / total, val_size / total, test_size / total


def _group_allocation(count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.array(fractions, dtype=float) * count
    base = np.floor(raw).astype(int)
    remainder = count - int(base.sum())
    order = np.argsort(-(raw - base))
    for index in order[:remainder]:
        base[index] += 1
    if count >= 3:
        positive = [index for index, fraction in enumerate(fractions) if fraction > 0]
        for index in positive:
            if base[index] == 0:
                donor = int(np.argmax(base))
                if base[donor] > 1:
                    base[donor] -= 1
                    base[index] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _resolve_group_values(df: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column in df.columns:
        series = df[group_column].astype(str)
    elif INCHIKEY_COLUMN in df.columns:
        series = df[INCHIKEY_COLUMN].astype(str)
    elif CANONICAL_SMILES_COLUMN in df.columns:
        series = df[CANONICAL_SMILES_COLUMN].astype(str)
    else:
        series = df[SMILES_COLUMN].astype(str).map(canonicalize_smiles)
    return series.replace("", np.nan).fillna(df[SMILES_COLUMN].astype(str))


def molecule_disjoint_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: str,
) -> dict[str, pd.DataFrame]:
    fractions = _normalize_sizes(train_size, val_size, test_size)
    working = df.copy()
    working["_split_group"] = _resolve_group_values(working, group_column)
    unique_groups = pd.Index(working["_split_group"].dropna().unique())
    if len(unique_groups) < 3:
        raise ValueError("At least three unique groups are required for a molecule-disjoint split.")
    generator = np.random.default_rng(seed)
    shuffled_groups = unique_groups.to_numpy(copy=True)
    generator.shuffle(shuffled_groups)
    n_train, n_val, n_test = _group_allocation(len(shuffled_groups), fractions)
    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:n_train + n_val + n_test])
    splits = {
        "train": working[working["_split_group"].isin(train_groups)].drop(columns="_split_group"),
        "val": working[working["_split_group"].isin(val_groups)].drop(columns="_split_group"),
        "test": working[working["_split_group"].isin(test_groups)].drop(columns="_split_group"),
    }
    if sum(len(split) for split in splits.values()) != len(df):
        raise RuntimeError("Split assignment did not preserve all rows.")
    for split_name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(f"The {split_name} split is empty.")
    return {name: split.reset_index(drop=True) for name, split in splits.items()}


def _require_murcko() -> object:
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:
        raise ImportError(
            "RDKit is required for scaffold splitting. Install it with "
            "`conda install -c conda-forge rdkit` or `pip install rdkit`."
        ) from error
    return MurckoScaffold


def _scaffold_labels(df: pd.DataFrame, smiles_column: str) -> pd.Series:
    MurckoScaffold = _require_murcko()
    source_smiles = (
        df[CANONICAL_SMILES_COLUMN].astype(str)
        if CANONICAL_SMILES_COLUMN in df.columns
        else df[smiles_column].astype(str).map(canonicalize_smiles)
    )
    scaffolds = source_smiles.map(lambda value: MurckoScaffold.MurckoScaffoldSmiles(smiles=value))
    return scaffolds.where(scaffolds.str.len() > 0, source_smiles)


def scaffold_disjoint_split(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    smiles_column: str,
) -> dict[str, pd.DataFrame]:
    scaffold_series = _scaffold_labels(df, smiles_column=smiles_column)
    unique_scaffolds = pd.Index(scaffold_series.unique())
    if len(unique_scaffolds) < 3:
        warnings.warn(
            "Scaffold-disjoint splitting is not feasible for this dataset size. Falling back to molecule-disjoint splitting.",
            stacklevel=2,
        )
        group_column = INCHIKEY_COLUMN if INCHIKEY_COLUMN in df.columns else CANONICAL_SMILES_COLUMN
        return molecule_disjoint_split(
            df=df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            group_column=group_column,
        )
    working = df.copy()
    working["_scaffold"] = scaffold_series
    try:
        splits = molecule_disjoint_split(
            df=working.rename(columns={"_scaffold": "scaffold_group"}),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            group_column="scaffold_group",
        )
    except ValueError:
        warnings.warn(
            "Scaffold-disjoint splitting produced an empty split. Falling back to molecule-disjoint splitting.",
            stacklevel=2,
        )
        group_column = INCHIKEY_COLUMN if INCHIKEY_COLUMN in df.columns else CANONICAL_SMILES_COLUMN
        return molecule_disjoint_split(
            df=df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            group_column=group_column,
        )
    return {
        split_name: split.drop(columns=["scaffold_group"], errors="ignore").reset_index(drop=True)
        for split_name, split in splits.items()
    }

