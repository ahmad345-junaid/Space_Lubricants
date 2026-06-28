from __future__ import annotations

from typing import Iterable

import torch

from leo_lubricants.data.schema import TARGET_COLUMNS
from leo_lubricants.workflows import prediction_dict_to_tensor


def _resolve_task_weights(
    task_weights: dict[str, float] | torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if task_weights is None:
        return torch.ones(len(TARGET_COLUMNS), device=device, dtype=dtype)
    if isinstance(task_weights, torch.Tensor):
        return task_weights.to(device=device, dtype=dtype)
    return torch.tensor(
        [float(task_weights.get(target_name, 1.0)) for target_name in TARGET_COLUMNS],
        device=device,
        dtype=dtype,
    )


def masked_mse_loss(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    task_weights: dict[str, float] | torch.Tensor | None,
) -> torch.Tensor:
    prediction_tensor = prediction_dict_to_tensor(predictions)
    weights = _resolve_task_weights(task_weights, prediction_tensor.device, prediction_tensor.dtype)
    squared_error = (prediction_tensor - targets) ** 2
    weighted_error = squared_error * mask * weights.unsqueeze(0)
    normalization = (mask * weights.unsqueeze(0)).sum().clamp_min(1.0)
    return weighted_error.sum() / normalization


def _group_indices(group_ids: torch.Tensor) -> Iterable[torch.Tensor]:
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        if int(mask.sum()) > 1:
            yield torch.nonzero(mask, as_tuple=False).squeeze(-1)


def physics_guided_penalty(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None,
) -> torch.Tensor:
    device = next(iter(predictions.values())).device
    penalty = torch.zeros(1, device=device, dtype=next(iter(predictions.values())).dtype)
    pair_group_ids = batch.get("pair_group_ids")
    temperature_values = batch.get("temperature_k")
    atomic_oxygen = batch.get("atomic_oxygen_fluence")
    if pair_group_ids is None or temperature_values is None or atomic_oxygen is None:
        return penalty.squeeze(0)
    weight_map = weights or {}
    pair_count = 0
    for group_indices in _group_indices(pair_group_ids):
        temperatures = temperature_values[group_indices]
        vapor_predictions = predictions["log_vapor_pressure_pa"][group_indices]
        viscosity_predictions = predictions["viscosity_cst"][group_indices]
        ao_predictions = predictions["degradation_index"][group_indices]
        ao_values = atomic_oxygen[group_indices]
        temp_order = torch.argsort(temperatures)
        sorted_temps = temperatures[temp_order]
        sorted_vapor = vapor_predictions[temp_order]
        sorted_viscosity = viscosity_predictions[temp_order]
        temp_deltas = sorted_temps[1:] - sorted_temps[:-1]
        if temp_deltas.numel() > 0 and bool(torch.any(temp_deltas > 0)):
            vapor_deltas = sorted_vapor[1:] - sorted_vapor[:-1]
            viscosity_deltas = sorted_viscosity[1:] - sorted_viscosity[:-1]
            penalty = penalty + float(weight_map.get("vapor_pressure_temperature", 1.0)) * torch.relu(-vapor_deltas[temp_deltas > 0]).mean()
            penalty = penalty + float(weight_map.get("viscosity_temperature", 1.0)) * torch.relu(viscosity_deltas[temp_deltas > 0]).mean()
            pair_count += 1
        ao_order = torch.argsort(ao_values)
        sorted_ao = ao_values[ao_order]
        sorted_deg = ao_predictions[ao_order]
        ao_deltas = sorted_ao[1:] - sorted_ao[:-1]
        if ao_deltas.numel() > 0 and bool(torch.any(ao_deltas > 0)):
            degradation_deltas = sorted_deg[1:] - sorted_deg[:-1]
            penalty = penalty + float(weight_map.get("degradation_atomic_oxygen", 1.0)) * torch.relu(-degradation_deltas[ao_deltas > 0]).mean()
            pair_count += 1
    if pair_count == 0:
        return torch.zeros((), device=device, dtype=next(iter(predictions.values())).dtype)
    return penalty.squeeze(0) / float(pair_count)


def total_loss(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    task_weights: dict[str, float] | torch.Tensor | None,
    physics_weights: dict[str, float] | None,
    l2_penalty: float,
    parameters,
) -> torch.Tensor:
    mse_value = masked_mse_loss(predictions, targets, mask, task_weights)
    physics_value = physics_guided_penalty(predictions, batch, physics_weights)
    l2_value = torch.zeros((), device=targets.device, dtype=targets.dtype)
    for parameter in parameters:
        l2_value = l2_value + torch.sum(parameter**2)
    return mse_value + physics_value + float(l2_penalty) * l2_value

