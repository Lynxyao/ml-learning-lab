# -*- coding: utf-8 -*-
"""Train an ECG time-series classifier."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ecg_dataset import ECGBeatDataset, class_counts, inverse_frequency_weights, load_split
from ecg_network_torch import build_ecg_model


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses, accs = [], []

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(signals)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy(logits.detach(), labels))

    return float(np.mean(losses)), float(np.mean(accs))


def main():
    parser = argparse.ArgumentParser(description="Train Module 2 ECG classifier.")
    parser.add_argument("--data_npz", default=project_path("data", "ecg", "mitdb_beat_segments.npz"))
    parser.add_argument("--model", default="cnn", choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--save_root", default=project_path("ecg_results"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_mode", default="record", choices=["record", "random"])
    parser.add_argument("--class_weighting", default="inverse", choices=["none", "inverse"])
    args = parser.parse_args()
    args.data_npz = resolve_project_path(args.data_npz)
    args.save_root = resolve_project_path(args.save_root)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dataset = ECGBeatDataset(args.data_npz)
    train_idx, val_idx, test_idx, split_meta = load_split(
        args.data_npz,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    train_set = ECGBeatDataset(args.data_npz, indices=train_idx)
    val_set = ECGBeatDataset(args.data_npz, indices=val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    in_channels = train_set.signals.shape[1]
    num_classes = int(base_dataset.labels.max().item()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ecg_model(args.model, in_channels=in_channels, num_classes=num_classes).to(device)

    if args.class_weighting == "inverse":
        weights = inverse_frequency_weights(base_dataset.labels.numpy()[train_idx], num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        weights = None
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_root, exist_ok=True)
    ckpt_dir = os.path.join(args.save_root, "checkpoints")
    split_dir = os.path.join(args.save_root, "splits")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    np.savetxt(os.path.join(split_dir, f"seed_{args.seed}_train.txt"), train_idx, fmt="%d")
    np.savetxt(os.path.join(split_dir, f"seed_{args.seed}_val.txt"), val_idx, fmt="%d")
    np.savetxt(os.path.join(split_dir, f"seed_{args.seed}_test.txt"), test_idx, fmt="%d")
    split_manifest = {
        "data_npz": args.data_npz,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "num_classes": num_classes,
        "class_weighting": args.class_weighting,
        "class_weights": weights.tolist() if weights is not None else None,
        "train_counts": class_counts(base_dataset.labels.numpy()[train_idx], num_classes).tolist(),
        "val_counts": class_counts(base_dataset.labels.numpy()[val_idx], num_classes).tolist(),
        "test_counts": class_counts(base_dataset.labels.numpy()[test_idx], num_classes).tolist(),
    }
    split_manifest.update(split_meta)
    with open(os.path.join(split_dir, f"seed_{args.seed}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    best_val_acc = -1.0
    history = []
    print(f"Using device: {device}")
    print(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_idx)}")
    print(f"Split mode: {args.split_mode}")
    print(f"Train class counts: {split_manifest['train_counts']}")
    print(f"Val class counts:   {split_manifest['val_counts']}")
    print(f"Test class counts:  {split_manifest['test_counts']}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "latest_model.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

    with open(os.path.join(args.save_root, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training done. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {os.path.join(ckpt_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
