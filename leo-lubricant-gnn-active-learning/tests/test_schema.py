import pandas as pd
import pytest

from leo_lubricants.data.schema import DatasetSchema, available_targets, validate_columns


def test_validate_columns_detects_missing_fields() -> None:
    schema = DatasetSchema()
    frame = pd.DataFrame({"molecule_id": ["A"], "smiles": ["CCO"]})
    with pytest.raises(ValueError):
        validate_columns(frame, schema.required_supervised_columns)


def test_available_targets_preserves_target_order() -> None:
    frame = pd.DataFrame(
        {
            "friction_coefficient": [0.1],
            "degradation_index": [0.2],
            "viscosity_cst": [45.0],
        }
    )
    assert available_targets(frame) == [
        "viscosity_cst",
        "friction_coefficient",
        "degradation_index",
    ]

