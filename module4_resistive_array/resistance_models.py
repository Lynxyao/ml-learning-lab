from __future__ import annotations

import torch
from torch import nn


class LinearInverseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPInverseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GridGNNInverseModel(nn.Module):
    """Small dependency-free GNN for square resistive grids.

    The model broadcasts the measurement vector to every cell, adds a learned
    cell-position embedding, performs local message passing on the grid, and
    predicts one value per cell.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        message_steps: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        grid_size = int(round(output_dim**0.5))
        if grid_size * grid_size != output_dim:
            raise ValueError("GridGNNInverseModel output_dim must be a square number")

        self.output_dim = output_dim
        self.grid_size = grid_size
        self.message_steps = message_steps
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position = nn.Parameter(torch.randn(output_dim, hidden_size) * 0.02)
        self.self_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(message_steps)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(message_steps)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(message_steps)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        adjacency = self._build_grid_adjacency(grid_size)
        self.register_buffer("adjacency", adjacency)

    @staticmethod
    def _build_grid_adjacency(grid_size: int) -> torch.Tensor:
        n_cells = grid_size * grid_size
        adjacency = torch.zeros(n_cells, n_cells, dtype=torch.float32)
        for row in range(grid_size):
            for col in range(grid_size):
                idx = row * grid_size + col
                for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    n_row, n_col = row + d_row, col + d_col
                    if 0 <= n_row < grid_size and 0 <= n_col < grid_size:
                        adjacency[idx, n_row * grid_size + n_col] = 1.0
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        return adjacency / degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_state = self.input_proj(x).unsqueeze(1)
        node_state = global_state + self.position.unsqueeze(0)
        for self_layer, neighbor_layer, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            neighbor_state = torch.einsum("ij,bjh->bih", self.adjacency, node_state)
            update = self_layer(node_state) + neighbor_layer(neighbor_state)
            node_state = norm(node_state + self.dropout(torch.nn.functional.gelu(update)))
        return self.head(node_state).squeeze(-1)


def build_model(model_type: str, input_dim: int, output_dim: int, hidden_size: int = 96) -> nn.Module:
    if model_type == "linear":
        return LinearInverseModel(input_dim=input_dim, output_dim=output_dim)
    if model_type == "mlp":
        return MLPInverseModel(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    if model_type == "grid_gnn":
        return GridGNNInverseModel(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    raise ValueError(f"Unknown model_type: {model_type}")
