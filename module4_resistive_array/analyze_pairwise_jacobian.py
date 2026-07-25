from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from circuit_physics import ideal_pairwise_terminal_currents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "resistance_8x8_simon_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resistance_results" / "pairwise_jacobian_8x8"
RELATIVE_SVD_THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-6)


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.float64)


def forward_data_agreement(
    resistance: np.ndarray,
    measured_current: np.ndarray,
    voltage: float,
    batch_size: int = 256,
) -> dict[str, float | int]:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(resistance), batch_size):
            batch = torch.tensor(resistance[start : start + batch_size], dtype=torch.float64)
            predictions.append(ideal_pairwise_terminal_currents(batch, voltage=voltage).numpy())
    predicted_current = np.concatenate(predictions, axis=0)
    error = predicted_current - measured_current
    return {
        "n_samples": int(len(resistance)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "correlation": float(np.corrcoef(predicted_current.reshape(-1), measured_current.reshape(-1))[0, 1]),
    }


def make_cases(
    r_train: np.ndarray,
    r_test: np.ndarray,
    low_ohm: float,
    high_ohm: float,
    seed: int,
) -> dict[str, np.ndarray]:
    n_cells = r_train.shape[1]
    rng = np.random.default_rng(seed)
    cases = {
        "all_low": np.full(n_cells, low_ohm, dtype=np.float64),
        "all_high": np.full(n_cells, high_ohm, dtype=np.float64),
        "train_cellwise_median": np.median(r_train, axis=0),
        "train_sample_0": r_train[0].copy(),
        "test_sample_0": r_test[0].copy(),
    }
    for fraction in (0.10, 0.25, 0.50):
        values = np.full(n_cells, low_ohm, dtype=np.float64)
        n_high = max(1, int(round(fraction * n_cells)))
        values[rng.choice(n_cells, size=n_high, replace=False)] = high_ohm
        cases[f"synthetic_high_{int(fraction * 100):02d}pct"] = values
    return cases


def cosine_ambiguity(jacobian: np.ndarray) -> tuple[float, int]:
    column_norm = np.linalg.norm(jacobian, axis=0, keepdims=True)
    normalized = jacobian / np.maximum(column_norm, 1e-15)
    cosine = np.abs(normalized.T @ normalized)
    np.fill_diagonal(cosine, 0.0)
    return float(cosine.max()), int(np.sum(np.triu(cosine > 0.99, k=1)))


def analyze_case(
    name: str,
    resistance: np.ndarray,
    voltage: float,
    output_dir: Path,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    log_resistance = torch.tensor(
        np.log(np.maximum(resistance, 1e-9)),
        dtype=torch.float64,
        requires_grad=True,
    )

    def forward(log_r: torch.Tensor) -> torch.Tensor:
        r = torch.exp(log_r).reshape(1, -1)
        return ideal_pairwise_terminal_currents(r, voltage=voltage).reshape(-1)

    currents_t = forward(log_resistance)
    jacobian_t = torch.autograd.functional.jacobian(forward, log_resistance, vectorize=True)
    currents = currents_t.detach().numpy()
    jacobian_log_r = jacobian_t.detach().numpy()

    # Fractional current change per unit fractional resistance change makes
    # sensitivities comparable across different current and resistance scales.
    current_scale = np.maximum(np.abs(currents), max(float(np.max(np.abs(currents))) * 1e-12, 1e-15))
    relative_jacobian = jacobian_log_r / current_scale[:, None]
    singular_values = np.linalg.svd(relative_jacobian, compute_uv=False)
    s_max = float(singular_values[0])
    s_min = float(singular_values[-1])
    numerical_rank = int(np.linalg.matrix_rank(relative_jacobian))
    effective_ranks = {
        f"effective_rank_rel_{threshold:g}": int(np.sum(singular_values >= s_max * threshold))
        for threshold in RELATIVE_SVD_THRESHOLDS
    }
    max_cosine, ambiguous_pairs = cosine_ambiguity(relative_jacobian)
    grid_size = int(round(resistance.size**0.5))
    cell_sensitivity = np.linalg.norm(relative_jacobian, axis=0).reshape(grid_size, grid_size)

    summary: dict[str, float | int | str] = {
        "case": name,
        "n_cells": int(resistance.size),
        "n_measurements": int(currents.size),
        "numerical_rank": numerical_rank,
        **effective_ranks,
        "largest_singular_value": s_max,
        "smallest_singular_value": s_min,
        "condition_number": float(s_max / s_min) if s_min > 0 else float("inf"),
        "max_cell_sensitivity": float(cell_sensitivity.max()),
        "min_cell_sensitivity": float(cell_sensitivity.min()),
        "max_column_cosine_similarity": max_cosine,
        "cell_pairs_with_cosine_over_0p99": ambiguous_pairs,
        "min_current": float(currents.min()),
        "max_current": float(currents.max()),
    }

    np.savez_compressed(
        output_dir / f"{name}.npz",
        resistance_map=resistance.reshape(grid_size, grid_size),
        currents=currents,
        jacobian_log_resistance=jacobian_log_r,
        relative_jacobian=relative_jacobian,
        singular_values=singular_values,
        cell_sensitivity=cell_sensitivity,
    )
    return summary, singular_values, cell_sensitivity


def save_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(
    output_dir: Path,
    singular_values_by_case: dict[str, np.ndarray],
    sensitivity_by_case: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for name, singular_values in singular_values_by_case.items():
        normalized = singular_values / max(float(singular_values[0]), 1e-15)
        ax.semilogy(np.arange(1, len(normalized) + 1), normalized, marker=".", linewidth=1.2, label=name)
    for threshold in RELATIVE_SVD_THRESHOLDS:
        ax.axhline(threshold, color="0.75", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title("8x8 pairwise-terminal Jacobian spectrum")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "singular_value_spectra.png", dpi=180)
    plt.close(fig)

    n_cases = len(sensitivity_by_case)
    n_cols = 4
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, sensitivity) in zip(axes, sensitivity_by_case.items()):
        image = ax.imshow(sensitivity, cmap="magma")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n_cases:]:
        ax.axis("off")
    fig.savefig(output_dir / "cell_relative_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze identifiability of Simon-style n x n pairwise terminal-current measurements."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--low_ohm", type=float, default=1.0)
    parser.add_argument("--high_ohm", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    r_train = load_csv(args.data_dir / "R_train.csv")
    r_test = load_csv(args.data_dir / "R_test.csv")
    i_train = load_csv(args.data_dir / "I_train.csv")
    i_test = load_csv(args.data_dir / "I_test.csv")
    if r_train.shape[1] != r_test.shape[1]:
        raise ValueError("R_train and R_test must have the same number of cells")
    grid_size = int(round(r_train.shape[1] ** 0.5))
    if grid_size * grid_size != r_train.shape[1]:
        raise ValueError("resistance rows must describe square maps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement = {
        "interpretation": (
            "Agreement here validates the Python implementation against Simon's generated CSV data; "
            "it does not validate the ideal circuit assumptions against physical hardware."
        ),
        "train": forward_data_agreement(r_train, i_train, args.voltage),
        "test": forward_data_agreement(r_test, i_test, args.voltage),
    }
    (args.output_dir / "forward_model_data_agreement.json").write_text(
        json.dumps(agreement, indent=2),
        encoding="utf-8",
    )
    summaries = []
    singular_values_by_case = {}
    sensitivity_by_case = {}
    for name, resistance in make_cases(r_train, r_test, args.low_ohm, args.high_ohm, args.seed).items():
        print(f"Analyzing {name} ...")
        summary, singular_values, sensitivity = analyze_case(name, resistance, args.voltage, args.output_dir)
        summaries.append(summary)
        singular_values_by_case[name] = singular_values
        sensitivity_by_case[name] = sensitivity

    save_summary(args.output_dir / "jacobian_summary.csv", summaries)
    (args.output_dir / "jacobian_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    save_figures(args.output_dir, singular_values_by_case, sensitivity_by_case)
    print(json.dumps(agreement, indent=2))
    print(json.dumps(summaries, indent=2, allow_nan=True))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
