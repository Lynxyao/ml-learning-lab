from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "resistance_results" / "hybrid_presentation"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    label: str,
    source: Path,
    curriculum: bool,
    objective: str,
    nested_key: str | None = None,
) -> dict[str, str | float | bool]:
    data = read_json(source)
    metrics = data[nested_key] if nested_key else data
    return {
        "model": label,
        "curriculum": curriculum,
        "objective": objective,
        "mae_ohm": metrics.get("mae_ohm", float("nan")),
        "rmse_ohm": metrics.get("rmse_ohm", float("nan")),
        "cell_accuracy": metrics["cell_low_high_accuracy"],
        "high_recall": metrics["high_recall"],
        "high_precision": metrics["high_precision"],
        "exact_map_accuracy": metrics["exact_map_accuracy"],
    }


def collect_rows() -> list[dict[str, str | float | bool]]:
    results = PROJECT_ROOT / "resistance_results"
    return [
        metric_row(
            "MLP regression (original)",
            results / "real_csv_8x8_v2" / "regression_conductance_mlp_metrics.json",
            False,
            "regression",
        ),
        metric_row(
            "Grid GNN regression (original)",
            results / "real_csv_8x8_v2" / "regression_conductance_grid_gnn_metrics.json",
            False,
            "regression",
        ),
        metric_row(
            "Hybrid regression (no curriculum)",
            results / "hybrid_gnn_8x8_v2_no_curriculum" / "hybrid_test_metrics.json",
            False,
            "multitask hybrid",
            "hybrid_regression",
        ),
        metric_row(
            "MLP regression + curriculum",
            results / "fair_curriculum_mlp" / "regression_conductance_mlp_metrics.json",
            True,
            "regression",
        ),
        metric_row(
            "MLP classification + curriculum",
            results / "fair_curriculum_mlp_classification" / "classification_high_mlp_metrics.json",
            True,
            "classification",
        ),
        metric_row(
            "Grid GNN classification + curriculum",
            results / "fair_curriculum_grid_gnn_classification" / "classification_high_grid_gnn_metrics.json",
            True,
            "classification",
        ),
        metric_row(
            "Hybrid physics-correction GNN + curriculum",
            results / "hybrid_gnn_8x8_v2" / "hybrid_test_metrics.json",
            True,
            "multitask hybrid",
            "hybrid_regression",
        ),
    ]


def save_csv(path: Path, rows: list[dict[str, str | float | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(path: Path, rows: list[dict[str, str | float | bool]]) -> None:
    selected = [rows[0], rows[2], rows[4], rows[5], rows[6]]
    labels = [
        "MLP reg.\noriginal",
        "Hybrid\nno curriculum",
        "MLP class.\n+ curriculum",
        "Grid GNN class.\n+ curriculum",
        "Hybrid\n+ curriculum",
    ]
    colors = ["#667085", "#4C78A8", "#59A14F", "#F28E2B", "#B33C86"]
    x = np.arange(len(selected))
    high_recall = [float(row["high_recall"]) for row in selected]
    exact_map = [float(row["exact_map_accuracy"]) for row in selected]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, high_recall, width, label="High-resistance recall", color=colors)
    axes[0].bar(x + width / 2, exact_map, width, label="Exact-map accuracy", color=colors, alpha=0.5)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Sparse high-resistance reconstruction")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    regression_rows = [row for row in rows if np.isfinite(float(row["mae_ohm"]))]
    reg_labels = [
        "MLP\noriginal",
        "Grid GNN\noriginal",
        "Hybrid\nno curriculum",
        "MLP\n+ curriculum",
        "Hybrid\n+ curriculum",
    ]
    mae = [float(row["mae_ohm"]) for row in regression_rows]
    axes[1].bar(np.arange(len(mae)), mae, color=["#667085", "#9C755F", "#4C78A8", "#59A14F", "#B33C86"])
    axes[1].set_xticks(np.arange(len(mae)), reg_labels, fontsize=8)
    axes[1].set_ylabel("MAE (ohm)")
    axes[1].set_title("Resistance-map regression error")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(mae):
        axes[1].text(index, value + max(mae) * 0.025, f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle("8x8 inverse sensing: preliminary simulator-only comparison", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_markdown(path: Path, rows: list[dict[str, str | float | bool]]) -> None:
    hybrid = rows[-1]
    mlp_classification = rows[4]
    gnn_classification = rows[5]
    no_curriculum = rows[2]
    text = f"""# Module 4 Hybrid GNN: Preliminary 8x8 Results

## Experimental question

Can an inverse model recover sparse high-resistance cells from 64 pairwise terminal-current measurements when those cells have very low Jacobian sensitivity?

## Model design

- Physics proxy: convert each terminal-pair current to an equivalent-resistance estimate using `R_eq = V / I` and the uniform-network correction factor.
- Circuit graph: connect cells that share a row terminal or column terminal.
- Learned output: predict a bounded correction to `log(R_proxy)` rather than predicting the resistance map blindly.
- Physical consistency: reconstructed resistance maps are passed through the differentiable KCL/Laplacian forward solver.
- Calibration: learn small regularized global, row, column, and offset corrections for future experimental fine-tuning.
- Uncertainty: predict per-cell log-resistance uncertainty.

## Main preliminary results

| Experiment | High recall | Exact-map accuracy | MAE (ohm) |
|---|---:|---:|---:|
| Hybrid, no sparse curriculum | {float(no_curriculum['high_recall']):.3f} | {float(no_curriculum['exact_map_accuracy']):.3f} | {float(no_curriculum['mae_ohm']):.3f} |
| MLP classification + curriculum | {float(mlp_classification['high_recall']):.3f} | {float(mlp_classification['exact_map_accuracy']):.3f} | N/A |
| Grid GNN classification + curriculum | {float(gnn_classification['high_recall']):.3f} | {float(gnn_classification['exact_map_accuracy']):.3f} | N/A |
| Hybrid physics-correction GNN + curriculum | {float(hybrid['high_recall']):.3f} | {float(hybrid['exact_map_accuracy']):.3f} | {float(hybrid['mae_ohm']):.3f} |

## Interpretation

1. The original training distribution is insufficient for the sparse binary test condition. Hybrid structure alone improves MAE but does not recover high-resistance cells.
2. Sparse-state curriculum data is necessary, but curriculum alone does not make MLP regression recover the high-resistance cells.
3. A class-weighted MLP can detect most high-resistance cells, showing that the measurements contain weak but usable information under the ideal simulator.
4. The hybrid model performs best in this preliminary test, combining a physics proxy, row/column circuit topology, multitask supervision, and forward consistency.
5. These results validate the computational pipeline against Simon's simulator only. They do not validate the forward model against physical hardware.

## Limitations to state in the meeting

- The test set contains only 50 maps and is generated by the same ideal simulator used for curriculum augmentation.
- Exact test-map patterns are excluded from curriculum generation, but the simulator family and resistance levels are still shared.
- The hybrid result should not be described as experimental validation or proof that the local physical resistances can be recovered in hardware.
- Architecture, curriculum, and multitask loss changed together. Additional single-factor ablations and multiple random seeds are still needed.
- Noise, contact resistance, parasitic resistance, and model mismatch have not yet been included.

## Next validation

- Repeat all models over at least five random seeds and report mean plus standard deviation.
- Add controlled current noise and forward-model mismatch.
- Simulate row/column gain errors, contact resistance, and offsets, then test calibration recovery.
- Fine-tune only the calibration layer when a small real experimental dataset becomes available.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    save_csv(OUTPUT_DIR / "model_comparison.csv", rows)
    save_comparison_plot(OUTPUT_DIR / "model_comparison.png", rows)
    save_markdown(OUTPUT_DIR / "meeting_results_notes.md", rows)
    print(f"Saved presentation outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
