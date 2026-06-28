import json
from pathlib import Path

import pandas as pd

from leo_lubricants.data.preprocess import preprocess_dataframe, run_preprocessing
from leo_lubricants.data.schema import DatasetSchema


def _frame_with_invalid_smiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": ["A", "B", "C", "D", "E"],
            "smiles": ["CCO", "not_a_smiles", "c1ccccc1", "CCN", "CC(=O)O"],
            "t_min_k": [270.0, 280.0, 290.0, 295.0, 300.0],
            "t_max_k": [320.0, 330.0, 340.0, 350.0, 360.0],
            "atomic_oxygen_fluence": [1000.0, 1200.0, 1400.0, 1600.0, 1800.0],
            "uv_dose": [80.0, 90.0, 110.0, 130.0, 150.0],
            "test_duration_h": [20.0, 24.0, 30.0, 36.0, 42.0],
            "vacuum_pressure_pa": [0.001, 0.001, 0.0008, 0.0007, 0.0006],
            "normal_load_n": [10.0, 11.0, 12.0, 13.0, 14.0],
            "sliding_speed_m_s": [0.2, 0.25, 0.3, 0.35, 0.4],
            "log_vapor_pressure_pa": [-4.5, -4.3, -4.1, -4.0, -3.9],
            "viscosity_cst": [45.0, 46.0, 47.0, 48.0, 49.0],
            "friction_coefficient": [0.09, 0.095, 0.1, 0.105, 0.11],
            "wear_rate": [3.0, 3.2, 3.4, 3.6, 3.8],
            "mass_loss_percent": [0.2, 0.21, 0.22, 0.23, 0.24],
            "viscosity_change_percent": [2.0, 2.2, 2.4, 2.6, 2.8],
            "degradation_index": [0.3, 0.32, 0.34, 0.36, 0.38],
        }
    )


def test_invalid_smiles_are_rejected() -> None:
    processed, rejected = preprocess_dataframe(_frame_with_invalid_smiles(), DatasetSchema())
    assert len(processed) == 4
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_reason"].startswith("Invalid SMILES string")


def test_run_preprocessing_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    _frame_with_invalid_smiles().to_csv(input_path, index=False)
    run_preprocessing(
        input_path=input_path,
        output_dir=output_dir,
        split_strategy="molecule_disjoint",
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=3,
        group_column="molecule_id",
    )
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "val.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert (output_dir / "rejected_rows.csv").exists()
    metadata = json.loads((output_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dropped_rows"] == 1
    assert metadata["split_strategy"] == "molecule_disjoint"
