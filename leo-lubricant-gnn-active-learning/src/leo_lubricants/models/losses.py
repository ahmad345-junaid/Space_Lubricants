from __future__ import annotations

import torch


def multitask_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    errors = (predictions - targets) ** 2
    if weights is not None:
        errors = errors * weights
    return errors.mean()

