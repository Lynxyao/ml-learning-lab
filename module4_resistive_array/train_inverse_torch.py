from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resistance_dataset import load_npz, split_and_scale, unscale_maps
from resistance_models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/resistance/resistance_inverse_synthetic.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "resistance_results"


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = make_loader(x, np.zeros((len(x), 1), dtype=np.float32), batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        outputs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def regression_metrics(true_maps: np.ndarray, pred_maps: np.ndarray) -> dict[str, float]:
    error = pred_maps - true_maps
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    per_sample_true = true_maps.reshape(len(true_maps), -1)
    per_sample_pred = pred_maps.reshape(len(pred_maps), -1)
    correlations = []
    for truth, pred in zip(per_sample_true, per_sample_pred):
        if np.std(truth) < 1e-6 or np.std(pred) < 1e-6:
            continue
        correlations.append(float(np.corrcoef(truth, pred)[0, 1]))
    return {
        "mae_ohm": mae,
        "rmse_ohm": rmse,
        "mean_pattern_correlation": float(np.mean(correlations)) if correlations else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an inverse model for resistive-array reconstruction.")
    parser.add_argument("--data_npz", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, maps, measurement_names, grid_size = load_npz(args.data_npz)
    data = split_and_scale(x, maps, measurement_names, grid_size, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        input_dim=data.x_train.shape[1],
        output_dim=data.y_train.shape[1],
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(data.x_train, data.y_train, args.batch_size, shuffle=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_inverse_model.pt"

    history: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_pred_scaled = predict(model, data.x_val, args.batch_size, device)
        val_pred = unscale_maps(val_pred_scaled, data.y_mean, data.y_std, grid_size)
        val_true = unscale_maps(data.y_val, data.y_mean, data.y_std, grid_size)
        val_metrics = regression_metrics(val_true, val_pred)
        row = {
            "epoch": float(epoch),
            "train_mse_scaled": float(np.mean(losses)),
            "val_mae_ohm": val_metrics["mae_ohm"],
            "val_rmse_ohm": val_metrics["rmse_ohm"],
            "val_pattern_correlation": val_metrics["mean_pattern_correlation"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_mse_scaled']:.4f} "
            f"val_mae={row['val_mae_ohm']:.2f}ohm "
            f"val_corr={row['val_pattern_correlation']:.3f}"
        )

        if row["val_mae_ohm"] < best_val:
            best_val = row["val_mae_ohm"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model,
                    "hidden_size": args.hidden_size,
                    "input_dim": data.x_train.shape[1],
                    "output_dim": data.y_train.shape[1],
                    "grid_size": grid_size,
                    "measurement_names": measurement_names,
                    "x_mean": data.x_mean,
                    "x_std": data.x_std,
                    "y_mean": data.y_mean,
                    "y_std": data.y_std,
                    "data_npz": str(args.data_npz),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred_scaled = predict(model, data.x_test, args.batch_size, device)
    test_pred = unscale_maps(test_pred_scaled, data.y_mean, data.y_std, grid_size)
    test_true = unscale_maps(data.y_test, data.y_mean, data.y_std, grid_size)
    metrics = regression_metrics(test_true, test_pred)

    (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        true_maps=test_true,
        predicted_maps=test_pred,
        measurements=data.x_test * data.x_std + data.x_mean,
        measurement_names=np.array(measurement_names, dtype=object),
    )

    print("\nTest metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
