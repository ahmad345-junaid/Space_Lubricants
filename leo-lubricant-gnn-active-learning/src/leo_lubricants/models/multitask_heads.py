from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiTaskHeadConfig:
    input_dim: int
    hidden_dim: int
    dropout: float
    target_names: tuple[str, ...]


class MultiTaskPredictionHeads(nn.Module):
    def __init__(self, config: MultiTaskHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict(
            {
                target_name: nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
                )
                for target_name in config.target_names
            }
        )

    def forward(self, shared_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            target_name: head(shared_embedding).squeeze(-1)
            for target_name, head in self.heads.items()
        }

