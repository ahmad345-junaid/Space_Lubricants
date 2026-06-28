from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


def _suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def read_table(path: str | Path) -> pd.DataFrame:
    suffix = _suffix(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ImportError(
                "Reading Parquet files requires a parquet engine such as pyarrow or fastparquet."
            ) from error
    raise ValueError(f"Unsupported table format for {path}. Supported formats are CSV and Parquet.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(target)
    if suffix == ".csv":
        df.to_csv(target, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(target, index=False)
        except ImportError as error:
            raise ImportError(
                "Writing Parquet files requires a parquet engine such as pyarrow or fastparquet."
            ) from error
        return
    raise ValueError(f"Unsupported table format for {path}. Supported formats are CSV and Parquet.")


def read_csv(path: str | Path) -> pd.DataFrame:
    return read_table(path)


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    write_table(frame, path)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return dict(data or {})


def write_joblib(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target)


def read_joblib(path: str | Path) -> Any:
    return joblib.load(path)

