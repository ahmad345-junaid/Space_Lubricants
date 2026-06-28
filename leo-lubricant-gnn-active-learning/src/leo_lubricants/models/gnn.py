from __future__ import annotations

import torch
from torch import nn

from leo_lubricants.models.multitask_heads import MultiTaskHead


class EnvironmentConditionedGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        message_passing_steps: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.update_layer = nn.GRUCell(hidden_dim, hidden_dim)
        self.steps = message_passing_steps
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(hidden_dim * 2, hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.node_encoder(node_features))
        for _ in range(self.steps):
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            messages = torch.matmul(adjacency, hidden) / degree
            messages = torch.relu(self.message_layer(messages))
            hidden = self.update_layer(
                messages.reshape(-1, messages.shape[-1]),
                hidden.reshape(-1, hidden.shape[-1]),
            ).reshape_as(hidden)
            hidden = self.dropout(hidden)

        mask = node_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        encoded_global = self.global_encoder(global_features)
        embedding = torch.cat([pooled, encoded_global], dim=-1)
        predictions = self.head(embedding)
        return predictions, embedding

