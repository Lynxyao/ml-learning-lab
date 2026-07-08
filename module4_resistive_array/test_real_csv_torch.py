from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from resistance_models import MLPInverseModel
from train_real_csv_torch import (
    DEFAULT_OUTPUT,
    choose_probability_threshold,
    inverse_regression_target,
    load_csv,
    low_high_metrics,
    predict,
    probability_threshold_scan,
    real_csv_mode_name,
    regression_metrics,
    save_prediction_csv,
    save_prediction_figure,
    save_threshold_scan,
    split_train_val,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Module 4 real-CSV inverse model.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.checkpoint is None:
        default_checkpoint = args.output_dir / "checkpoints" / "regression_conductance_physics_w0p1_model.pt"
        args.checkpoint = default_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_dir = args.data_dir or Path(checkpoint["data_dir"])

    i_train = load_csv(data_dir / "I_train.csv")
    r_train = load_csv(data_dir / "R_train.csv")
    i_test = load_csv(data_dir / "I_test.csv")
    r_test = load_csv(data_dir / "R_test.csv")

    seed = int(args.seed if args.seed is not None else checkpoint["seed"])
    _, _, x_val_raw, y_val_raw = split_train_val(i_train, r_train, seed)
    x_val = (x_val_raw - checkpoint["x_mean"]) / checkpoint["x_std"]
    x_test = (i_test - checkpoint["x_mean"]) / checkpoint["x_std"]

    model = MLPInverseModel(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    raw_pred = predict(model, x_test, args.batch_size, device)
    task = checkpoint["task"]
    threshold = float(checkpoint["threshold"])

    if task == "classification":
        positive_state = checkpoint["positive_state"]
        probability_threshold = checkpoint.get("probability_threshold")
        val_scan_rows = []
        if probability_threshold is None:
            val_prob_positive = 1.0 / (1.0 + np.exp(-predict(model, x_val, args.batch_size, device)))
            probability_threshold, val_scan_rows = choose_probability_threshold(
                y_val_raw,
                val_prob_positive,
                threshold,
                positive_state,
            )

        prob_positive = 1.0 / (1.0 + np.exp(-raw_pred))
        if positive_state == "high":
            pred_r = np.where(prob_positive >= probability_threshold, 100.0, 1.0).astype(np.float32)
        else:
            pred_r = np.where(prob_positive >= probability_threshold, 1.0, 100.0).astype(np.float32)
        metrics = low_high_metrics(r_test, pred_r, threshold)
        metrics[f"mean_{positive_state}_probability"] = float(prob_positive.mean())
        metrics["positive_state"] = positive_state
        metrics["probability_threshold_chosen_on_validation"] = float(probability_threshold)
        scan_rows = probability_threshold_scan(r_test, prob_positive, threshold, positive_state)
    else:
        target_transform = checkpoint["target_transform"]
        pred_trans = raw_pred * checkpoint["y_std"] + checkpoint["y_mean"]
        pred_r = inverse_regression_target(pred_trans, target_transform)
        pred_r = np.clip(
            pred_r,
            float(checkpoint["prediction_min"]),
            float(checkpoint["prediction_max"]),
        )
        metrics = regression_metrics(r_test, pred_r, threshold)
        metrics["target_transform"] = target_transform
        metrics["physics_loss_weight"] = float(checkpoint["physics_loss_weight"])
        metrics["prediction_min"] = float(checkpoint["prediction_min"])
        metrics["prediction_max"] = float(checkpoint["prediction_max"])
        metrics["voltage"] = float(checkpoint["voltage"])
        if float(checkpoint["physics_loss_weight"]) > 0:
            metrics["physics_forward_model"] = "simon_laplacian_schur"
        scan_rows = []
        val_scan_rows = []

    mode_args = argparse.Namespace(**{key: checkpoint[key] for key in checkpoint if key in {
        "task",
        "positive_state",
        "target_transform",
        "physics_loss_weight",
    }})
    mode_name = real_csv_mode_name(mode_args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{mode_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_csv(args.output_dir / f"{mode_name}_test_predictions.csv", r_test, pred_r, i_test, threshold)
    save_prediction_figure(args.output_dir / f"{mode_name}_examples.png", r_test, pred_r)
    save_threshold_scan(args.output_dir / f"{mode_name}_threshold_scan.csv", scan_rows)
    if val_scan_rows:
        save_threshold_scan(args.output_dir / f"{mode_name}_validation_threshold_scan.csv", val_scan_rows)

    print("\nData summary")
    print(f"I_test shape={i_test.shape}, range=({i_test.min():.4f}, {i_test.max():.4f})")
    print(f"R_test unique={np.unique(r_test)}")
    print("\nTest metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved test outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
