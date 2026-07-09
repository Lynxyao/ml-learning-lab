# -*- coding: utf-8 -*-
"""Save ECG signal segment plots for the teaching interface."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ecg_dataset import ECGBeatDataset


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description="Visualize prepared ECG beat segments.")
    parser.add_argument("--data_npz", default=project_path("data", "ecg", "mitdb_beat_segments.npz"))
    parser.add_argument("--out_dir", default=project_path("ecg_results", "sample_plots"))
    parser.add_argument("--max_samples", type=int, default=12)
    parser.add_argument("--balanced", action="store_true", help="Sample evenly across classes when possible")
    args = parser.parse_args()
    args.data_npz = resolve_project_path(args.data_npz)
    args.out_dir = resolve_project_path(args.out_dir)

    dataset = ECGBeatDataset(args.data_npz)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.balanced:
        labels = dataset.labels.numpy()
        chosen = []
        per_class = max(1, args.max_samples // len(dataset.label_names))
        for label in sorted(np.unique(labels)):
            class_indices = np.flatnonzero(labels == label)[:per_class]
            chosen.extend(class_indices.tolist())
        indices = chosen[:args.max_samples]
    else:
        indices = list(range(min(args.max_samples, len(dataset))))

    for idx in indices:
        signal, label = dataset[idx]
        label_name = str(dataset.label_names[int(label)])
        y = signal[0].numpy()

        plt.figure(figsize=(8, 2.5))
        plt.plot(y, linewidth=1.25)
        plt.title(f"ECG beat segment #{idx} | class: {label_name}")
        plt.xlabel("Sample")
        plt.ylabel("Normalized amplitude")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"ecg_sample_{idx:03d}_{label_name}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
