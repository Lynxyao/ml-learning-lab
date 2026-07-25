from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics_torch import forward_measurements_from_maps, measurement_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "resistance_results" / "jacobian"


def make_reference_map(grid_size: int, low_ohm: float, high_ohm: float, low_probability: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = np.where(
        rng.random((grid_size, grid_size)) < low_probability,
        low_ohm,
        high_ohm,
    ).astype(np.float32)
    return torch.tensor(values.reshape(1, -1), dtype=torch.float32, requires_grad=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze forward-model Jacobian sensitivity.")
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--low_ohm", type=float, default=1.0)
    parser.add_argument("--high_ohm", type=float, default=100.0)
    parser.add_argument("--low_probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device("cpu")
    matrix = measurement_matrix(args.grid_size, device)
    reference = make_reference_map(
        args.grid_size,
        low_ohm=args.low_ohm,
        high_ohm=args.high_ohm,
        low_probability=args.low_probability,
        seed=args.seed,
    )

    def forward(flat_map: torch.Tensor) -> torch.Tensor:
        return forward_measurements_from_maps(flat_map.reshape(1, -1), matrix).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(forward, reference.reshape(-1)).detach().numpy()
    abs_jacobian = np.abs(jacobian)
    cell_sensitivity = abs_jacobian.sum(axis=0).reshape(args.grid_size, args.grid_size)
    measurement_sensitivity = abs_jacobian.sum(axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"jacobian_{args.grid_size}x{args.grid_size}.npz",
        reference_map=reference.detach().numpy().reshape(args.grid_size, args.grid_size),
        jacobian=jacobian,
        cell_sensitivity=cell_sensitivity,
        measurement_sensitivity=measurement_sensitivity,
    )
    summary = {
        "grid_size": args.grid_size,
        "n_cells": args.grid_size * args.grid_size,
        "n_measurements": int(matrix.shape[0]),
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "condition_number": float(np.linalg.cond(jacobian @ jacobian.T)),
        "min_cell_sensitivity": float(cell_sensitivity.min()),
        "max_cell_sensitivity": float(cell_sensitivity.max()),
    }
    (args.output_dir / f"jacobian_summary_{args.grid_size}x{args.grid_size}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
