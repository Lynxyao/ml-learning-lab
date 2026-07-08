from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from resistance_physics import make_measurement_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "resistance_results/sensitivity"


def current_sensitivity(resistance: np.ndarray, voltage: float, measurement_matrix: np.ndarray) -> np.ndarray:
    """Return dI/dR for each measurement with respect to each local resistor."""
    flat_r = resistance.reshape(-1)
    local_derivative = -voltage / np.maximum(flat_r, 1e-9) ** 2
    return measurement_matrix * local_derivative[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze current sensitivity for low/high resistance references.")
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--low_reference", type=float, default=1.0)
    parser.add_argument("--high_reference", type=float, default=100.0)
    parser.add_argument("--voltage", type=float, default=1.0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    spec = make_measurement_spec(args.grid_size)
    low_map = np.full((args.grid_size, args.grid_size), args.low_reference, dtype=np.float32)
    high_map = np.full((args.grid_size, args.grid_size), args.high_reference, dtype=np.float32)
    low_sens = np.abs(current_sensitivity(low_map, args.voltage, spec.matrix))
    high_sens = np.abs(current_sensitivity(high_map, args.voltage, spec.matrix))
    ratio = np.divide(low_sens, np.maximum(high_sens, 1e-12))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "low_vs_high_current_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["measurement", "mean_abs_dI_dR_at_low", "mean_abs_dI_dR_at_high", "low_to_high_ratio"])
        for name, low_row, high_row, ratio_row in zip(spec.names, low_sens, high_sens, ratio):
            active = spec.matrix[spec.names.index(name)] > 0
            writer.writerow(
                [
                    name,
                    float(low_row[active].mean()),
                    float(high_row[active].mean()),
                    float(ratio_row[active].mean()),
                ]
            )

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = np.arange(len(spec.names))
    ax.bar(x - 0.18, [row[row > 0].mean() for row in low_sens], width=0.36, label=f"R={args.low_reference:g}")
    ax.bar(x + 0.18, [row[row > 0].mean() for row in high_sens], width=0.36, label=f"R={args.high_reference:g}")
    ax.set_yscale("log")
    ax.set_ylabel("|dI/dR| under fixed voltage")
    ax.set_title("Current measurements are much more sensitive to low resistance")
    ax.set_xticks(x)
    ax.set_xticklabels(spec.names, rotation=35, ha="right")
    ax.legend()
    fig.savefig(args.output_dir / "low_vs_high_current_sensitivity.png", dpi=180)
    plt.close(fig)

    print(f"Saved sensitivity CSV to {csv_path}")
    print(f"Saved sensitivity plot to {args.output_dir / 'low_vs_high_current_sensitivity.png'}")
    print(f"Low/high sensitivity ratio: {(args.high_reference / args.low_reference) ** 2:.1f}x")


if __name__ == "__main__":
    main()
