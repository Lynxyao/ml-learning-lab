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

from dynamic_impedance import CircuitGraph, DynamicImpedanceGNN, MeasurementBatch, pairwise_row_column_protocol
from dynamic_impedance.adapters import load_numeric_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_model(checkpoint: dict, device: torch.device) -> DynamicImpedanceGNN:
    graph = CircuitGraph.row_column_array(int(checkpoint["grid_size"]))
    model = DynamicImpedanceGNN(
        graph,
        hidden_size=int(checkpoint["hidden_size"]),
        message_steps=int(checkpoint["message_steps"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frequency-ready impedance GNN in DC mode.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "resistance_8x8_simon_v2")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "resistance_results" / "dynamic_impedance_dc" / "dynamic_impedance_dc_best.pt")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "resistance_results" / "dynamic_impedance_dc")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current = torch.from_numpy(load_numeric_csv(args.data_dir / "I_test.csv"))
    true_r = load_numeric_csv(args.data_dir / "R_test.csv")
    model = build_model(checkpoint, device)
    protocol = pairwise_row_column_protocol(int(checkpoint["grid_size"])).to(device)
    predicted = []
    probability = []
    uncertainty = []
    with torch.no_grad():
        for start in range(0, len(current), args.batch_size):
            batch = MeasurementBatch.from_dc_currents(
                current[start : start + args.batch_size].to(device),
                protocol,
                voltage=float(checkpoint["voltage"]),
            )
            output = model(batch)
            predicted.append(output["resistance"].cpu().numpy())
            probability.append(torch.sigmoid(output["high_state_logit"]).cpu().numpy())
            uncertainty.append(output["log_parameter_uncertainty"][..., 0].cpu().numpy())
    pred_r = np.concatenate(predicted)
    probability = np.concatenate(probability)
    uncertainty = np.concatenate(uncertainty)
    threshold = float(checkpoint["threshold"])
    true_high = true_r > threshold
    pred_high = probability >= 0.5
    tp = float((true_high & pred_high).sum())
    fp = float((~true_high & pred_high).sum())
    fn = float((true_high & ~pred_high).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "mae_ohm": float(np.abs(pred_r - true_r).mean()),
        "rmse_ohm": float(np.sqrt(np.square(pred_r - true_r).mean())),
        "log_rmse": float(np.sqrt(np.square(np.log(pred_r) - np.log(true_r)).mean())),
        "high_precision": precision,
        "high_recall": recall,
        "high_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "cell_accuracy": float((true_high == pred_high).mean()),
        "trained_mode": "dc",
        "rlc_trained": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dynamic_dc_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "dynamic_dc_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        n_cells = true_r.shape[1]
        writer.writerow(
            ["sample_id"]
            + [f"true_r_{i + 1}" for i in range(n_cells)]
            + [f"pred_r_{i + 1}" for i in range(n_cells)]
            + [f"uncertainty_{i + 1}" for i in range(n_cells)]
            + [f"high_probability_{i + 1}" for i in range(n_cells)]
        )
        for index in range(len(true_r)):
            writer.writerow([index, *true_r[index], *pred_r[index], *uncertainty[index], *probability[index]])
    n = min(args.examples, len(true_r))
    grid_size = int(checkpoint["grid_size"])
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.5 * n), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    for sample in range(n):
        panels = [
            (true_r[sample], "True R", "viridis"),
            (pred_r[sample], "DC-GNN prediction", "viridis"),
            (np.abs(pred_r[sample] - true_r[sample]), "Absolute error", "magma"),
            (uncertainty[sample], "Log-R uncertainty", "magma"),
        ]
        for column, (values, title, cmap) in enumerate(panels):
            image = axes[sample, column].imshow(values.reshape(grid_size, grid_size), cmap=cmap)
            axes[sample, column].set_title(f"{title}\nSample {sample}", fontsize=9)
            axes[sample, column].set_xticks([])
            axes[sample, column].set_yticks([])
            fig.colorbar(image, ax=axes[sample, column], fraction=0.046, pad=0.04)
    fig.savefig(args.output_dir / "dynamic_dc_test_examples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
