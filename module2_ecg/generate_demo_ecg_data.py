# -*- coding: utf-8 -*-
"""Generate a tiny synthetic ECG-like dataset for pipeline smoke tests.

This is not a clinical dataset. It is only for checking that the Module 2
training, evaluation, and visualization scripts work before downloading MIT-BIH.
"""

import argparse
import os

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))


LABEL_NAMES = np.array(["N", "SVEB", "VEB", "F", "Q"])


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def gaussian(x, center, width, amp):
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def synthetic_beat(label, length, rng):
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    beat = np.zeros_like(x)

    # P, QRS, and T wave components with class-specific morphology.
    beat += gaussian(x, 0.22, 0.035, 0.12)
    beat += gaussian(x, 0.44, 0.012, -0.25)
    beat += gaussian(x, 0.50, 0.018, 1.0)
    beat += gaussian(x, 0.56, 0.014, -0.35)
    beat += gaussian(x, 0.74, 0.055, 0.28)

    if label == 1:  # SVEB: early, narrower beat
        beat = np.roll(beat, -int(length * 0.08))
        beat += gaussian(x, 0.46, 0.010, 0.25)
    elif label == 2:  # VEB: wider QRS and inverted T tendency
        beat += gaussian(x, 0.50, 0.055, 0.70)
        beat -= gaussian(x, 0.73, 0.060, 0.45)
    elif label == 3:  # Fusion: mixed morphology
        beat += gaussian(x, 0.52, 0.040, 0.35)
        beat += gaussian(x, 0.69, 0.045, -0.12)
    elif label == 4:  # Unknown/noisy
        beat *= 0.45
        beat += gaussian(x, 0.62, 0.030, rng.uniform(-0.4, 0.4))

    baseline = 0.05 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * x + rng.uniform(0, np.pi))
    noise = rng.normal(0.0, 0.035, size=length).astype(np.float32)
    scale = rng.uniform(0.85, 1.15)
    shift = rng.uniform(-0.06, 0.06)
    return (scale * beat + baseline + noise + shift).astype(np.float32)


def generate_dataset(samples_per_class, length, seed):
    rng = np.random.default_rng(seed)
    signals, labels = [], []
    for label in range(len(LABEL_NAMES)):
        for _ in range(samples_per_class):
            signals.append(synthetic_beat(label, length, rng))
            labels.append(label)

    signals = np.asarray(signals, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    indices = rng.permutation(len(labels))
    return signals[indices], labels[indices]


def main():
    parser = argparse.ArgumentParser(description="Create synthetic ECG-like data for Module 2 smoke tests.")
    parser.add_argument("--out_npz", default=project_path("data", "ecg", "demo_beat_segments.npz"))
    parser.add_argument("--samples_per_class", type=int, default=160)
    parser.add_argument("--length", type=int, default=234)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_npz = resolve_project_path(args.out_npz)

    signals, labels = generate_dataset(args.samples_per_class, args.length, args.seed)
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, signals=signals, labels=labels, label_names=LABEL_NAMES)
    print(f"Saved demo ECG dataset: {args.out_npz}")
    print(f"signals shape: {signals.shape}, labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
