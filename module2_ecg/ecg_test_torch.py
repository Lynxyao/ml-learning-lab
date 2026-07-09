# -*- coding: utf-8 -*-
"""Evaluate an ECG classifier and save teaching-friendly metrics."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ecg_dataset import ECGBeatDataset, load_split
from ecg_network_torch import build_ecg_model


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def confusion_matrix_np(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def metrics_from_confusion(matrix):
    total = matrix.sum()
    correct = np.trace(matrix)
    accuracy = float(correct / total) if total else 0.0
    per_class = []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[i, :].sum())})
    supported = [row for row in per_class if row["support"] > 0]
    macro_precision = float(np.mean([row["precision"] for row in supported])) if supported else 0.0
    macro_recall = float(np.mean([row["recall"] for row in supported])) if supported else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in supported])) if supported else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Module 2 ECG classifier.")
    parser.add_argument("--data_npz", default=project_path("data", "ecg", "mitdb_beat_segments.npz"))
    parser.add_argument("--checkpoint", default=project_path("ecg_results", "checkpoints", "best_model.pth"))
    parser.add_argument("--model", default="cnn", choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--save_root", default=project_path("ecg_results"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_mode", default="record", choices=["record", "random"])
    args = parser.parse_args()
    args.data_npz = resolve_project_path(args.data_npz)
    args.checkpoint = resolve_project_path(args.checkpoint)
    args.save_root = resolve_project_path(args.save_root)

    base_dataset = ECGBeatDataset(args.data_npz)
    _, _, test_idx, split_meta = load_split(
        args.data_npz,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    test_set = ECGBeatDataset(args.data_npz, indices=test_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_channels = test_set.signals.shape[1]
    num_classes = int(base_dataset.labels.max().item()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ecg_model(args.model, in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            logits = model(signals.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    matrix = confusion_matrix_np(y_true, y_pred, num_classes)
    metrics = metrics_from_confusion(matrix)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["label_names"] = [str(x) for x in base_dataset.label_names[:num_classes]]
    metrics["split_mode"] = args.split_mode
    metrics.update(split_meta)

    os.makedirs(args.save_root, exist_ok=True)
    out_path = os.path.join(args.save_root, "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
