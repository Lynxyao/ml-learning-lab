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
from hybrid_physics_gnn import HybridPhysicsGNN
from train_hybrid_physics_gnn import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, predict_outputs
from train_real_csv_torch import load_csv, low_high_metrics, regression_metrics


def build_model(checkpoint: dict, device: torch.device) -> HybridPhysicsGNN:
    normalization = checkpoint["normalization"]
    model = HybridPhysicsGNN(
        grid_size=int(checkpoint["grid_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        message_steps=int(checkpoint["message_steps"]),
        dropout=float(checkpoint["dropout"]),
        voltage=float(checkpoint["voltage"]),
        prediction_min=float(checkpoint["prediction_min"]),
        prediction_max=float(checkpoint["prediction_max"]),
        max_log_correction=float(checkpoint["max_log_correction"]),
        log_current_mean=torch.tensor(normalization["log_current_mean"]),
        log_current_std=torch.tensor(normalization["log_current_std"]),
        log_proxy_mean=torch.tensor(normalization["log_proxy_mean"]),
        log_proxy_std=torch.tensor(normalization["log_proxy_std"]),
        current_scale=float(normalization["current_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def reconstruct_currents(
    model: HybridPhysicsGNN,
    predicted_r: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(predicted_r), batch_size):
        resistance = torch.tensor(predicted_r[start : start + batch_size], dtype=torch.float32, device=device)
        ideal = ideal_pairwise_terminal_currents(resistance, voltage=model.voltage)
        chunks.append(model.calibrate_forward_current(ideal).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def save_predictions(
    path: Path,
    true_r: np.ndarray,
    output: dict[str, np.ndarray],
    probability: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        n_cells = true_r.shape[1]
        header = ["sample_id"]
        for prefix in ("true_r", "proxy_r", "pred_r", "uncertainty", "high_probability"):
            header.extend(f"{prefix}_{i + 1}" for i in range(n_cells))
        writer.writerow(header)
        for index in range(len(true_r)):
            writer.writerow(
                [
                    index,
                    *true_r[index],
                    *output["proxy_resistance"][index],
                    *output["resistance"][index],
                    *output["log_resistance_uncertainty"][index],
                    *probability[index],
                ]
            )


def save_examples(
    path: Path,
    true_r: np.ndarray,
    output: dict[str, np.ndarray],
    n_examples: int = 8,
) -> None:
    n_examples = min(n_examples, len(true_r))
    grid_size = int(round(true_r.shape[1] ** 0.5))
    fig, axes = plt.subplots(n_examples, 5, figsize=(14, 2.45 * n_examples), constrained_layout=True)
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    pred_r = output["resistance"]
    uncertainty = output["log_resistance_uncertainty"]
    for sample in range(n_examples):
        panels = [
            (true_r[sample], "True R", "viridis"),
            (output["proxy_resistance"][sample], "Physics proxy", "viridis"),
            (pred_r[sample], "Hybrid prediction", "viridis"),
            (np.abs(pred_r[sample] - true_r[sample]), "Absolute error", "magma"),
            (uncertainty[sample], "Predicted uncertainty", "magma"),
        ]
        for column, (values, title, cmap) in enumerate(panels):
            image = axes[sample, column].imshow(values.reshape(grid_size, grid_size), cmap=cmap)
            axes[sample, column].set_title(f"{title}\nSample {sample}", fontsize=9)
            axes[sample, column].set_xticks([])
            axes[sample, column].set_yticks([])
            fig.colorbar(image, ax=axes[sample, column], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained hybrid physics-correction GNN.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "hybrid_physics_gnn_best.pt")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_dir = args.data_dir or Path(checkpoint.get("data_dir", DEFAULT_DATA_DIR))
    i_test = load_csv(data_dir / "I_test.csv")
    r_test = load_csv(data_dir / "R_test.csv")
    model = build_model(checkpoint, device)

    output = predict_outputs(model, i_test, args.batch_size, device)
    pred_r = output["resistance"]
    proxy_r = output["proxy_resistance"]
    probability = 1.0 / (1.0 + np.exp(-output["high_state_logit"]))
    threshold = float(checkpoint["threshold"])
    probability_threshold = float(checkpoint["probability_threshold"])
    state_pred_r = np.where(probability >= probability_threshold, 100.0, 1.0).astype(np.float32)

    reconstructed_current = reconstruct_currents(model, pred_r, args.batch_size, device)
    log_current_error = np.log(np.maximum(reconstructed_current, 1e-8)) - np.log(np.maximum(i_test, 1e-8))
    absolute_log_r_error = np.abs(np.log(np.maximum(pred_r, 1e-8)) - np.log(np.maximum(r_test, 1e-8)))
    uncertainty = output["log_resistance_uncertainty"]
    uncertainty_error_correlation = float(
        np.corrcoef(uncertainty.reshape(-1), absolute_log_r_error.reshape(-1))[0, 1]
    )

    metrics = {
        "hybrid_regression": regression_metrics(r_test, pred_r, threshold),
        "physics_proxy": regression_metrics(r_test, proxy_r, threshold),
        "hybrid_high_state_classifier": {
            **low_high_metrics(r_test, state_pred_r, threshold),
            "probability_threshold_chosen_on_validation": probability_threshold,
        },
        "forward_current_reconstruction": {
            "log_current_rmse": float(np.sqrt(np.mean(log_current_error**2))),
            "current_mae": float(np.mean(np.abs(reconstructed_current - i_test))),
        },
        "uncertainty": {
            "mean_predicted_log_r_uncertainty": float(uncertainty.mean()),
            "uncertainty_absolute_error_correlation": uncertainty_error_correlation,
        },
        "calibration": model.calibration_summary(),
        "forward_model": checkpoint["forward_model"],
        "forward_model_experimentally_validated": checkpoint["forward_model_experimentally_validated"],
        "best_epoch": int(checkpoint["best_epoch"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hybrid_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_predictions(args.output_dir / "hybrid_test_predictions.csv", r_test, output, probability)
    save_examples(args.output_dir / "hybrid_test_examples.png", r_test, output)

    presentation = {
        "model": "Hybrid physics-proxy correction GNN",
        "test_mae_ohm": metrics["hybrid_regression"]["mae_ohm"],
        "test_rmse_ohm": metrics["hybrid_regression"]["rmse_ohm"],
        "regression_high_recall": metrics["hybrid_regression"]["high_recall"],
        "classifier_high_recall": metrics["hybrid_high_state_classifier"]["high_recall"],
        "classifier_high_precision": metrics["hybrid_high_state_classifier"]["high_precision"],
        "exact_map_accuracy": metrics["hybrid_high_state_classifier"]["exact_map_accuracy"],
        "log_current_reconstruction_rmse": metrics["forward_current_reconstruction"]["log_current_rmse"],
        "uncertainty_error_correlation": uncertainty_error_correlation,
        "interpretation": (
            "Simulator consistency is reported separately from experimental validation. "
            "High-state recall is the primary diagnostic because the test set is dominated by low-resistance cells."
        ),
    }
    (args.output_dir / "presentation_summary.json").write_text(
        json.dumps(presentation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved hybrid test outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
