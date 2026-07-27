# -*- coding: utf-8 -*-
"""Export held-out metrics for a fixed set of ECG teaching checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ecg_dataset import ECGBeatDataset, load_split
from ecg_network_torch import build_ecg_model
from ecg_test_torch import confusion_matrix_np, metrics_from_confusion


def resolve(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def evaluate(checkpoint, data_npz, model_name, split_mode, seed, batch_size):
    base_dataset = ECGBeatDataset(data_npz)
    _, _, test_idx, _ = load_split(data_npz, split_mode=split_mode, seed=seed)
    test_set = ECGBeatDataset(data_npz, indices=test_idx)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    num_classes = int(base_dataset.labels.max().item()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ecg_model(
        model_name,
        in_channels=test_set.signals.shape[1],
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, labels in loader:
            predictions = model(signals.to(device)).argmax(dim=1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())

    matrix = confusion_matrix_np(y_true, y_pred, num_classes)
    metrics = metrics_from_confusion(matrix)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["label_names"] = [str(value) for value in base_dataset.label_names[:num_classes]]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_npz", default="data/ecg/demo_beat_segments.npz")
    parser.add_argument("--checkpoint_dir", default="ecg_results/teaching_checkpoints/checkpoints")
    parser.add_argument("--history", default="ecg_results/teaching_checkpoints/train_history.json")
    parser.add_argument("--output_dir", default="website/assets/module2")
    parser.add_argument("--epochs", default="1,3,4,5,6,10")
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--split_mode", default="random", choices=["record", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    data_npz = resolve(args.data_npz)
    checkpoint_dir = resolve(args.checkpoint_dir)
    output_dir = resolve(args.output_dir)
    history_path = resolve(args.history)
    epochs = [int(value.strip()) for value in args.epochs.split(",") if value.strip()]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history_by_epoch = {int(row["epoch"]): row for row in history}
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "Synthetic ECG teaching data (not clinical data)",
        "model": "1D CNN",
        "class_weighting": "inverse frequency",
        "split": "fixed random held-out split",
        "seed": args.seed,
        "epochs": epochs,
    }

    for epoch in epochs:
        checkpoint = checkpoint_dir / f"epoch_{epoch}.pth"
        metrics = evaluate(
            checkpoint,
            data_npz,
            args.model,
            args.split_mode,
            args.seed,
            args.batch_size,
        )
        metrics.update(
            {
                "epoch": epoch,
                "dataset": manifest["dataset"],
                "model": manifest["model"],
                "class_weighting": manifest["class_weighting"],
                "split": manifest["split"],
                "train_loss": history_by_epoch[epoch]["train_loss"],
                "train_acc": history_by_epoch[epoch]["train_acc"],
                "val_loss": history_by_epoch[epoch]["val_loss"],
                "val_acc": history_by_epoch[epoch]["val_acc"],
            }
        )
        (output_dir / f"checkpoint-{epoch}.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(epochs)} checkpoint summaries to {output_dir}")


if __name__ == "__main__":
    main()
