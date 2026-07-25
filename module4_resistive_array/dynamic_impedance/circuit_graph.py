from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CircuitGraph:
    """Oriented component-edge graph used by modified nodal analysis."""

    n_nodes: int
    edge_index: torch.Tensor
    edge_labels: tuple[str, ...] = ()
    grid_size: int | None = None

    def __post_init__(self) -> None:
        if self.edge_index.dtype != torch.long or self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a long tensor with shape [2, n_edges]")
        if torch.any(self.edge_index < 0) or torch.any(self.edge_index >= self.n_nodes):
            raise ValueError("edge_index contains an invalid node")
        if torch.any(self.edge_index[0] == self.edge_index[1]):
            raise ValueError("self-loop circuit components are not supported")
        if self.edge_labels and len(self.edge_labels) != self.n_edges:
            raise ValueError("edge_labels must match n_edges")

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @classmethod
    def row_column_array(cls, grid_size: int) -> "CircuitGraph":
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        sources = []
        targets = []
        labels = []
        for row in range(grid_size):
            for column in range(grid_size):
                sources.append(row)
                targets.append(grid_size + column)
                labels.append(f"r{row + 1}c{column + 1}")
        return cls(
            n_nodes=2 * grid_size,
            edge_index=torch.tensor([sources, targets], dtype=torch.long),
            edge_labels=tuple(labels),
            grid_size=grid_size,
        )

    def incidence_matrix(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        incidence = torch.zeros(self.n_nodes, self.n_edges, dtype=dtype, device=device)
        edge = self.edge_index.to(device=device)
        component = torch.arange(self.n_edges, device=device)
        incidence[edge[0], component] = 1
        incidence[edge[1], component] = -1
        return incidence

    def edge_adjacency(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        incidence = self.incidence_matrix(dtype=dtype, device=device).abs()
        adjacency = incidence.transpose(0, 1) @ incidence
        adjacency.fill_diagonal_(0)
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1)


def pairwise_row_column_protocol(grid_size: int) -> torch.Tensor:
    """Return the n^2 row-to-column voltage-drive pairs in cell order."""
    return torch.tensor(
        [(row, grid_size + column) for row in range(grid_size) for column in range(grid_size)],
        dtype=torch.long,
    )
