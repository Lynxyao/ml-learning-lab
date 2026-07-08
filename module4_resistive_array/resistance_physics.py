from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementSpec:
    names: list[str]
    matrix: np.ndarray


def make_measurement_spec(grid_size: int) -> MeasurementSpec:
    """Build a measurement mask for row, column, and path currents.

    Each row in the matrix indicates which local resistors contribute to one
    current measurement. Simon's real circuit solver can replace this matrix
    later while keeping the inverse-learning scripts unchanged.
    """
    n_cells = grid_size * grid_size
    rows: list[np.ndarray] = []
    names: list[str] = []

    for r in range(grid_size):
        weights = np.zeros(n_cells, dtype=np.float32)
        weights[r * grid_size : (r + 1) * grid_size] = 1.0
        rows.append(weights)
        names.append(f"row_{r + 1}_current")

    for c in range(grid_size):
        weights = np.zeros(n_cells, dtype=np.float32)
        weights[c::grid_size] = 1.0
        rows.append(weights)
        names.append(f"column_{c + 1}_current")

    diag = np.zeros(n_cells, dtype=np.float32)
    anti_diag = np.zeros(n_cells, dtype=np.float32)
    for i in range(grid_size):
        diag[i * grid_size + i] = 1.0
        anti_diag[i * grid_size + (grid_size - 1 - i)] = 1.0
    rows.extend([diag, anti_diag])
    names.extend(["main_diagonal_current", "anti_diagonal_current"])

    total = np.ones(n_cells, dtype=np.float32)
    rows.append(total)
    names.append("whole_array_current")

    return MeasurementSpec(names=names, matrix=np.stack(rows, axis=0))


def generate_resistance_maps(
    n_samples: int,
    grid_size: int = 3,
    seed: int = 7,
    low_reference: float = 1.0,
    high_reference: float = 100.0,
    low_probability: float = 0.25,
    jitter: float = 0.03,
) -> np.ndarray:
    """Generate local resistance maps with low/high reference states.

    This follows Simon's current setup more closely: each local region is treated
    as a resistor with either a low or high reference value, with small physical
    jitter. Later, this can be extended to continuous cell-growth states.
    """
    rng = np.random.default_rng(seed)
    maps = np.empty((n_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(n_samples):
        low_mask = rng.random((grid_size, grid_size)) < low_probability
        if not low_mask.any():
            low_mask[rng.integers(0, grid_size), rng.integers(0, grid_size)] = True
        local = np.where(low_mask, low_reference, high_reference).astype(np.float32)
        local *= rng.normal(1.0, jitter, size=(grid_size, grid_size)).astype(np.float32)
        maps[i] = np.clip(local, min(low_reference, high_reference) * 0.2, max(low_reference, high_reference) * 2.0)

    return maps


def measure_maps(
    maps: np.ndarray,
    measurement_noise: float = 0.02,
    seed: int = 7,
    voltage: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Convert hidden local resistances into current measurements.

    Under a fixed voltage, each local branch contributes conductance 1/R. The
    measured current along a row/column/path is approximately V * sum(1/R_i).
    This is why low-resistance regions dominate the signal and high-resistance
    regions are harder to reconstruct.
    """
    if maps.ndim != 3 or maps.shape[1] != maps.shape[2]:
        raise ValueError("maps must have shape [samples, grid_size, grid_size]")

    grid_size = maps.shape[1]
    spec = make_measurement_spec(grid_size)
    conductance = 1.0 / np.maximum(maps.reshape(len(maps), -1), 1e-6)
    measurements = voltage * (conductance @ spec.matrix.T)

    if measurement_noise > 0:
        rng = np.random.default_rng(seed)
        scale = np.maximum(np.abs(measurements), 1.0) * measurement_noise
        measurements = measurements + rng.normal(0.0, scale).astype(np.float32)

    return measurements.astype(np.float32), spec.names


def make_dataset(
    n_samples: int,
    grid_size: int = 3,
    measurement_noise: float = 0.02,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    maps = generate_resistance_maps(n_samples=n_samples, grid_size=grid_size, seed=seed)
    measurements, names = measure_maps(maps, measurement_noise=measurement_noise, seed=seed + 11)
    return measurements, maps.astype(np.float32), names
