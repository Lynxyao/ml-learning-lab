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

from resistance_dataset import load_npz, split_and_scale, unscale_maps
from resistance_models import build_model
from train_inverse_torch import predict, regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "resistance_results"


def save_example_csv(path: Path, measurements: np.ndarray, true_maps: np.ndarray, pred_maps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_size = true_maps.shape[1]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_id", "mae_ohm"]
        header += [f"true_r{r + 1}c{c + 1}" for r in range(grid_size) for c in range(grid_size)]
        header += [f"pred_r{r + 1}c{c + 1}" for r in range(grid_size) for c in range(grid_size)]
        header += [f"measurement_{i + 1}" for i in range(measurements.shape[1])]
        writer.writerow(header)
        for i in range(len(true_maps)):
            mae = float(np.mean(np.abs(pred_maps[i] - true_maps[i])))
            writer.writerow([i, mae, *true_maps[i].ravel(), *pred_maps[i].ravel(), *measurements[i]])


def save_heatmap_examples(path: Path, true_maps: np.ndarray, pred_maps: np.ndarray, n_examples: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n_examples, len(true_maps))
    fig, axes = plt.subplots(n, 3, figsize=(8.5, 2.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(1, -1)

    vmin = float(min(true_maps[:n].min(), pred_maps[:n].min()))
    vmax = float(max(true_maps[:n].max(), pred_maps[:n].max()))
    for i in range(n):
        error = np.abs(pred_maps[i] - true_maps[i])
        panels = [(true_maps[i], "True local map"), (pred_maps[i], "Predicted map"), (error, "Absolute error")]
        for j, (arr, title) in enumerate(panels):
            im = axes[i, j].imshow(arr, cmap="viridis" if j < 2 else "magma", vmin=vmin if j < 2 else None, vmax=vmax if j < 2 else None)
            axes[i, j].set_title(f"{title}\nSample {i}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved resistive-array inverse model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_npz", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_npz = args.data_npz or Path(checkpoint["data_npz"])
    x, maps, measurement_names, grid_size = load_npz(data_npz)
    data = split_and_scale(x, maps, measurement_names, grid_size, seed=args.seed)

    model = build_model(
        checkpoint["model_type"],
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_size=checkpoint["hidden_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    pred_scaled = predict(model, data.x_test, args.batch_size, device)
    pred_maps = unscale_maps(pred_scaled, checkpoint["y_mean"], checkpoint["y_std"], grid_size)
    true_maps = unscale_maps(data.y_test, checkpoint["y_mean"], checkpoint["y_std"], grid_size)
    measurements = data.x_test * checkpoint["x_std"] + checkpoint["x_mean"]
    metrics = regression_metrics(true_maps, pred_maps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    save_heatmap_examples(figures_dir / "resistance_inverse_examples.png", true_maps, pred_maps)
    save_example_csv(args.output_dir / "test_sample_predictions.csv", measurements, true_maps, pred_maps)
    (args.output_dir / "test_metrics_rerun.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved heatmap examples to {figures_dir / 'resistance_inverse_examples.png'}")
    print(f"Saved sample predictions to {args.output_dir / 'test_sample_predictions.csv'}")


if __name__ == "__main__":
    main()
