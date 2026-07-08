from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fall_dataset import classification_metrics, load_npz, split_and_scale
from fall_models import FallRNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/fall/unimib_fall_windows.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "fall_results"


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    loader = make_loader(x, np.zeros(len(x)), batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    return p, (p >= 0.5).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RNN fall prediction model.")
    parser.add_argument("--data_npz", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, groups = load_npz(args.data_npz)
    data = split_and_scale(x, y, groups=groups, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallRNN(hidden_size=args.hidden_size, model=args.model).to(device)

    pos = max(float((data.y_train == 1).sum()), 1.0)
    neg = max(float((data.y_train == 0).sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_loader(data.x_train, data.y_train, args.batch_size, shuffle=True)
    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        _, val_pred = predict(model, data.x_val, args.batch_size, device)
        val_metrics = classification_metrics(data.y_val, val_pred)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_recall_fall": float(val_metrics["recall_fall"]),
            "val_f1_fall": float(val_metrics["f1_fall"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_recall={row['val_recall_fall']:.4f} "
            f"val_f1={row['val_f1_fall']:.4f}"
        )

        if row["val_f1_fall"] > best_val_f1:
            best_val_f1 = row["val_f1_fall"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model,
                    "hidden_size": args.hidden_size,
                    "mean": data.mean,
                    "std": data.std,
                    "data_npz": str(args.data_npz),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_probs, test_pred = predict(model, data.x_test, args.batch_size, device)
    test_metrics = classification_metrics(data.y_test, test_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "test_predictions.npz", probs=test_probs, pred=test_pred, y_true=data.y_test)

    print("\nTest metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
