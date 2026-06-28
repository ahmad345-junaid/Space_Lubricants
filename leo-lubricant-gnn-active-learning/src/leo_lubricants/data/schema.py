from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MOLECULE_ID_COLUMN = "molecule_id"
SMILES_COLUMN = "smiles"
CANONICAL_SMILES_COLUMN = "canonical_smiles"
INCHIKEY_COLUMN = "inchikey"

MOLECULAR_COLUMNS = [
    MOLECULE_ID_COLUMN,
    SMILES_COLUMN,
    CANONICAL_SMILES_COLUMN,
    INCHIKEY_COLUMN,
]

ENVIRONMENT_COLUMNS = [
    "t_min_k",
    "t_max_k",
    "atomic_oxygen_fluence",
    "uv_dose",
    "test_duration_h",
    "vacuum_pressure_pa",
    "normal_load_n",
    "sliding_speed_m_s",
]

TARGET_COLUMNS = [
    "log_vapor_pressure_pa",
    "viscosity_cst",
    "friction_coefficient",
    "wear_rate",
    "mass_loss_percent",
    "viscosity_change_percent",
    "degradation_index",
]


@dataclass(frozen=True)
class ColumnGroups:
    molecular: tuple[str, ...] = tuple(MOLECULAR_COLUMNS)
    environment: tuple[str, ...] = tuple(ENVIRONMENT_COLUMNS)
    targets: tuple[str, ...] = tuple(TARGET_COLUMNS)


@dataclass(frozen=True)
class DatasetSchema:
    column_groups: ColumnGroups = field(default_factory=ColumnGroups)
    molecule_id_column: str = MOLECULE_ID_COLUMN
    smiles_column: str = SMILES_COLUMN
    canonical_smiles_column: str = CANONICAL_SMILES_COLUMN
    inchikey_column: str = INCHIKEY_COLUMN

    @property
    def environment_columns(self) -> tuple[str, ...]:
        return self.column_groups.environment

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self.column_groups.targets

    @property
    def required_supervised_columns(self) -> tuple[str, ...]:
        return (
            self.molecule_id_column,
            self.smiles_column,
            *self.environment_columns,
            *self.target_columns,
        )

    @property
    def required_inference_columns(self) -> tuple[str, ...]:
        return (self.molecule_id_column, self.smiles_column, *self.environment_columns)


def validate_columns(df: pd.DataFrame, required: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def available_targets(df: pd.DataFrame) -> list[str]:
    return [column for column in TARGET_COLUMNS if column in df.columns]

