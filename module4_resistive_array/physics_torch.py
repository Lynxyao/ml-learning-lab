from __future__ import annotations

import numpy as np
import torch

from resistance_physics import make_measurement_spec


def measurement_matrix(grid_size: int, device: torch.device) -> torch.Tensor:
    spec = make_measurement_spec(grid_size)
    return torch.tensor(spec.matrix, dtype=torch.float32, device=device)


def forward_measurements_from_maps(
    maps: torch.Tensor,
    measurement_matrix: torch.Tensor,
    voltage: float = 1.0,
    min_resistance: float = 1e-3,
) -> torch.Tensor:
    """Differentiable version of the synthetic forward measurement model.

    maps can be [batch, n*n] or [batch, n, n]. The output is [batch, n_measurements].
    """
    flat_maps = maps.reshape(maps.shape[0], -1)
    positive_maps = torch.clamp(flat_maps, min=min_resistance)
    conductance = 1.0 / positive_maps
    return voltage * conductance @ measurement_matrix.T


def numpy_forward_measurements(maps: np.ndarray, grid_size: int, voltage: float = 1.0) -> np.ndarray:
    matrix = make_measurement_spec(grid_size).matrix
    conductance = 1.0 / np.maximum(maps.reshape(len(maps), -1), 1e-6)
    return (voltage * conductance @ matrix.T).astype(np.float32)
