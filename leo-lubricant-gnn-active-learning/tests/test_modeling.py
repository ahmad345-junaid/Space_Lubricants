import numpy as np
import pandas as pd
import torch

from leo_lubricants.data.preprocess import preprocess_dataframe
from leo_lubricants.data.schema import DatasetSchema, TARGET_COLUMNS
from leo_lubricants.models.gpr_uncertainty import GPRConfig, MultiOutputGPRUncertainty
from leo_lubricants.models.losses import masked_mse_loss
from leo_lubricants.workflows import build_model_from_batch, prepare_batch


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_id": ["A", "B", "C"],
            "smiles": ["CCO", "CCN", "c1ccccc1"],
            "t_min_k": [270.0, 280.0, 290.0],
            "t_max_k": [320.0, 340.0, 360.0],
            "atomic_oxygen_fluence": [1000.0, 1500.0, 2000.0],
            "uv_dose": [80.0, 100.0, 120.0],
            "test_duration_h": [20.0, 30.0, 40.0],
            "vacuum_pressure_pa": [0.001, 0.0008, 0.0006],
            "normal_load_n": [10.0, 12.0, 14.0],
            "sliding_speed_m_s": [0.2, 0.3, 0.4],
            "log_vapor_pressure_pa": [-4.5, -4.3, -4.1],
            "viscosity_cst": [45.0, 50.0, 55.0],
            "friction_coefficient": [0.09, 0.1, 0.11],
            "wear_rate": [3.0, 3.2, 3.4],
            "mass_loss_percent": [0.2, 0.22, 0.24],
            "viscosity_change_percent": [2.0, 2.2, 2.4],
            "degradation_index": [0.3, 0.35, 0.4],
        }
    )


def _prepared_batch():
    processed, _ = preprocess_dataframe(_sample_frame(), DatasetSchema())
    return prepare_batch(processed, include_targets=True)


def test_model_forward_pass_on_sample_molecules() -> None:
    prepared = _prepared_batch()
    model = build_model_from_batch(
        {
            "hidden_dim": 32,
            "head_hidden_dim": 32,
            "message_passing_steps": 2,
            "dropout": 0.1,
        },
        prepared,
    )
    outputs = model(prepared.batch)
    assert set(outputs.keys()) == set(TARGET_COLUMNS)
    assert all(output.shape == (3,) for output in outputs.values())


def test_encode_returns_2d_embeddings() -> None:
    prepared = _prepared_batch()
    model = build_model_from_batch(
        {
            "hidden_dim": 32,
            "head_hidden_dim": 32,
            "message_passing_steps": 2,
            "dropout": 0.1,
        },
        prepared,
    )
    embeddings = model.encode(prepared.batch)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == 3


def test_masked_loss_ignores_missing_targets() -> None:
    prepared = _prepared_batch()
    model = build_model_from_batch(
        {
            "hidden_dim": 16,
            "head_hidden_dim": 16,
            "message_passing_steps": 1,
            "dropout": 0.0,
        },
        prepared,
    )
    predictions = model(prepared.batch)
    targets = prepared.targets.clone()
    mask = prepared.mask.clone()
    mask[0, 0] = 0.0
    targets[0, 0] = 500.0
    loss_with_mask = masked_mse_loss(predictions, targets, mask, None)
    targets[0, 0] = prepared.targets[0, 0]
    loss_without_change = masked_mse_loss(predictions, targets, mask, None)
    assert torch.isclose(loss_with_mask, loss_without_change)


def test_gpr_fit_and_predict_return_correct_shapes() -> None:
    embeddings = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 1.5]], dtype=float)
    targets = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], dtype=float)
    gpr = MultiOutputGPRUncertainty(GPRConfig(noise_level=1.0e-5)).fit(embeddings, targets, ["a", "b"])
    means, stds = gpr.predict(embeddings)
    assert means.shape == (3, 2)
    assert stds.shape == (3, 2)
