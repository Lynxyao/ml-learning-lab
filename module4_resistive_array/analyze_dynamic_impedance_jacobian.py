from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamic_impedance import CircuitGraph, pairwise_row_column_protocol
from dynamic_impedance.adapters import load_numeric_csv
from dynamic_impedance.jacobian import measurement_parameter_jacobian


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DC or AC impedance-parameter identifiability.")
    parser.add_argument("--mode", choices=("dc", "ac"), default="dc")
    parser.add_argument("--resistance_csv", type=Path, default=PROJECT_ROOT / "data" / "resistance_8x8_simon_v2" / "R_test.csv")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--frequencies", default="10,100,1000,10000")
    parser.add_argument("--inductance_h", type=float, default=1e-3)
    parser.add_argument("--capacitance_f", type=float, default=1e-6)
    parser.add_argument("--parameter", choices=("resistance", "inductance", "capacitance"), default="resistance")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "resistance_results" / "dynamic_impedance_jacobian")
    args = parser.parse_args()

    maps = load_numeric_csv(args.resistance_csv)
    resistance = torch.from_numpy(maps[args.sample])
    grid_size = int(round(resistance.numel() ** 0.5))
    graph = CircuitGraph.row_column_array(grid_size)
    protocol = pairwise_row_column_protocol(grid_size)
    kwargs = {}
    if args.mode == "ac":
        kwargs = {
            "frequency_hz": torch.tensor([float(value) for value in args.frequencies.split(",")]),
            "inductance": torch.full_like(resistance, args.inductance_h),
            "capacitance": torch.full_like(resistance, args.capacitance_f),
        }
    elif args.parameter != "resistance":
        raise ValueError("DC mode only supports resistance sensitivity")
    jacobian = measurement_parameter_jacobian(
        graph,
        resistance,
        protocol,
        voltage=args.voltage,
        parameter=args.parameter,
        **kwargs,
    )
    reduction_axes = tuple(range(jacobian.ndim - 1))
    sensitivity = torch.sqrt(jacobian.abs().square().sum(dim=reduction_axes)).detach().cpu().numpy()
    relative = sensitivity / max(float(sensitivity.max()), 1e-20)
    matrix = jacobian.detach().cpu().numpy()
    flattened = matrix.reshape(-1, graph.n_edges)
    singular_values = np.linalg.svd(flattened, compute_uv=False)
    summary = {
        "mode": args.mode,
        "parameter": args.parameter,
        "sample": args.sample,
        "grid_size": grid_size,
        "jacobian_shape": list(matrix.shape),
        "numerical_rank": int(np.linalg.matrix_rank(flattened)),
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / max(singular_values[-1], 1e-20)),
        "ac_parameters_are_synthetic_placeholders": args.mode == "ac",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / f"{args.mode}_{args.parameter}_jacobian.npz",
        jacobian=matrix,
        cell_sensitivity=sensitivity,
        relative_cell_sensitivity=relative,
        singular_values=singular_values,
    )
    (args.output_dir / f"{args.mode}_{args.parameter}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    image = ax.imshow(relative.reshape(grid_size, grid_size), vmin=0, vmax=1, cmap="magma")
    ax.set_title(f"{args.mode.upper()} relative sensitivity: {args.parameter}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, label="Relative Jacobian norm")
    fig.savefig(args.output_dir / f"{args.mode}_{args.parameter}_sensitivity.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
