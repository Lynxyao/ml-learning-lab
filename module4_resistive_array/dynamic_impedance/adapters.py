from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .circuit_graph import pairwise_row_column_protocol
from .data_schema import MeasurementBatch, PhysicalParameterBatch


def load_numeric_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def load_legacy_dc_csv(
    current_csv: Path,
    resistance_csv: Path | None = None,
    voltage: float = 5.0,
) -> tuple[MeasurementBatch, PhysicalParameterBatch | None]:
    current = load_numeric_csv(current_csv)
    n_measurements = current.shape[1]
    grid_size = int(round(n_measurements**0.5))
    if grid_size * grid_size != n_measurements:
        raise ValueError("legacy pairwise current count must be a perfect square")
    measurement = MeasurementBatch.from_dc_currents(
        torch.from_numpy(current),
        pairwise_row_column_protocol(grid_size),
        voltage=voltage,
    )
    if resistance_csv is None:
        return measurement, None
    resistance = load_numeric_csv(resistance_csv)
    if resistance.shape != current.shape:
        raise ValueError("current and resistance CSV files must have matching shapes")
    target = PhysicalParameterBatch(
        resistance=torch.from_numpy(resistance),
        resistance_mask=torch.ones_like(torch.from_numpy(resistance), dtype=torch.bool),
    )
    return measurement, target


def select_batch(batch: MeasurementBatch, index: torch.Tensor) -> MeasurementBatch:
    return MeasurementBatch(
        current_real=batch.current_real[index],
        current_imag=batch.current_imag[index],
        frequency_hz=batch.frequency_hz[index],
        voltage_real=batch.voltage_real[index],
        voltage_imag=batch.voltage_imag[index],
        drive_pairs=batch.drive_pairs,
        mask=batch.mask[index],
        mode=batch.mode,
    )
