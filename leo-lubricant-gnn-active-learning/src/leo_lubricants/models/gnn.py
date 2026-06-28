from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from leo_lubricants.models.multitask_heads import MultiTaskHeadConfig, MultiTaskPredictionHeads


@dataclass(frozen=True)
class GNNConfig:
    atom_feature_dim: int
    bond_feature_dim: int
    descriptor_dim: int
    environment_dim: int
    degradation_memory_dim: int
    hidden_dim: int
    message_passing_steps: int
    dropout: float
    head_hidden_dim: int
    target_names: list[str]
    use_environment_gate: bool = True
    use_descriptors: bool = True
    use_degradation_memory: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "GNNConfig":
        return cls(**values)


class BondMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bond_feature_dim: int,
        environment_dim: int,
        dropout: float,
        use_environment_gate: bool,
    ) -> None:
        super().__init__()
        self.use_environment_gate = use_environment_gate
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + bond_feature_dim + environment_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_layer = nn.Linear(environment_dim, hidden_dim)
        self.update_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        bond_features: torch.Tensor,
        environment_embedding: torch.Tensor,
        graph_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_states
        source_index = edge_index[0]
        target_index = edge_index[1]
        source_states = node_states[source_index]
        target_states = node_states[target_index]
        edge_environment = environment_embedding[graph_index[source_index]]
        messages = self.message_mlp(
            torch.cat([source_states, target_states, bond_features, edge_environment], dim=1)
        )
        if self.use_environment_gate:
            messages = messages * torch.sigmoid(self.gate_layer(edge_environment))
        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, target_index, messages)
        updated = self.update_layer(torch.cat([node_states, aggregated], dim=1))
        return self.norm(node_states + updated)


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_states: torch.Tensor, graph_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        scores = self.score_network(node_states).squeeze(-1)
        pooled = torch.zeros((num_graphs, node_states.shape[1]), dtype=node_states.dtype, device=node_states.device)
        for graph_id in range(num_graphs):
            mask = graph_index == graph_id
            active_nodes = node_states[mask]
            active_scores = scores[mask]
            if active_nodes.numel() == 0:
                continue
            weights = torch.softmax(active_scores, dim=0)
            pooled[graph_id] = torch.sum(active_nodes * weights.unsqueeze(-1), dim=0)
        return pooled


class EnvironmentConditionedGNN(nn.Module):
    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_projection = nn.Linear(config.atom_feature_dim, config.hidden_dim)
        self.descriptor_projection = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.environment_projection = nn.Sequential(
            nn.Linear(config.environment_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.degradation_projection = nn.Sequential(
            nn.Linear(config.degradation_memory_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.message_layers = nn.ModuleList(
            [
                BondMessageLayer(
                    hidden_dim=config.hidden_dim,
                    bond_feature_dim=config.bond_feature_dim,
                    environment_dim=config.hidden_dim,
                    dropout=config.dropout,
                    use_environment_gate=config.use_environment_gate,
                )
                for _ in range(config.message_passing_steps)
            ]
        )
        self.node_norm = nn.LayerNorm(config.hidden_dim)
        self.readout = AttentionReadout(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.prediction_heads = MultiTaskPredictionHeads(
            MultiTaskHeadConfig(
                input_dim=config.hidden_dim * 4,
                hidden_dim=config.head_hidden_dim,
                dropout=config.dropout,
                target_names=tuple(config.target_names),
            )
        )

    def _project_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptor_embedding = self.descriptor_projection(batch["descriptor_matrix"])
        environment_embedding = self.environment_projection(batch["environment_matrix"])
        degradation_embedding = self.degradation_projection(batch["degradation_memory_matrix"])
        if not self.config.use_descriptors:
            descriptor_embedding = torch.zeros_like(descriptor_embedding)
        if not self.config.use_degradation_memory:
            degradation_embedding = torch.zeros_like(degradation_embedding)
        return descriptor_embedding, environment_embedding, degradation_embedding

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = batch["node_features"]
        edge_index = batch["edge_index"]
        edge_features = batch["edge_features"]
        graph_index = batch["graph_index"]
        num_graphs = int(batch["descriptor_matrix"].shape[0])
        descriptor_embedding, environment_embedding, degradation_embedding = self._project_context(batch)
        node_states = self.atom_projection(node_features)
        node_states = node_states + descriptor_embedding[graph_index] + environment_embedding[graph_index]
        if self.config.use_degradation_memory:
            node_states = node_states + degradation_embedding[graph_index]
        node_states = self.node_norm(node_states)
        for layer in self.message_layers:
            node_states = layer(
                node_states=node_states,
                edge_index=edge_index,
                bond_features=edge_features,
                environment_embedding=environment_embedding,
                graph_index=graph_index,
            )
            node_states = self.dropout(node_states)
        molecular_embedding = self.readout(node_states, graph_index, num_graphs)
        return torch.cat(
            [
                molecular_embedding,
                descriptor_embedding,
                environment_embedding,
                degradation_embedding,
            ],
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.prediction_heads(self.encode(batch))

