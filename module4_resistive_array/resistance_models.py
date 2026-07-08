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


def build_model(model_type: str, input_dim: int, output_dim: int, hidden_size: int = 96) -> nn.Module:
    if model_type == "linear":
        return LinearInverseModel(input_dim=input_dim, output_dim=output_dim)
    if model_type == "mlp":
        return MLPInverseModel(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    raise ValueError(f"Unknown model_type: {model_type}")
