# -*- coding: utf-8 -*-
"""Dataset utilities and optional MIT-BIH beat segmentation helper."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_LABEL_NAMES = np.array(["N", "SVEB", "VEB", "F", "Q"])
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_project_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)

AAMI_LABEL_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4, "?": 4,
}


class ECGBeatDataset(Dataset):
    """Prepared ECG beat dataset loaded from an `.npz` file."""

    def __init__(self, data_npz, indices=None, normalize=True):
        pack = np.load(data_npz, allow_pickle=True)
        signals = pack["signals"].astype(np.float32)
        labels = pack["labels"].astype(np.int64)

        if signals.ndim == 2:
            signals = signals[:, None, :]
        if signals.ndim != 3:
            raise ValueError(f"Expected signals shape [N,L] or [N,C,L], got {signals.shape}")

        if indices is not None:
            signals = signals[indices]
            labels = labels[indices]

        if normalize:
            mean = signals.mean(axis=-1, keepdims=True)
            std = signals.std(axis=-1, keepdims=True) + 1e-6
            signals = (signals - mean) / std

        self.signals = torch.from_numpy(signals)
        self.labels = torch.from_numpy(labels)
        self.label_names = pack["label_names"] if "label_names" in pack else DEFAULT_LABEL_NAMES
        self.records = pack["records"] if "records" in pack else None
        self.samples = pack["samples"] if "samples" in pack else None
        self.symbols = pack["symbols"] if "symbols" in pack else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def split_indices(num_samples, val_fraction=0.15, test_fraction=0.15, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    test_n = int(round(num_samples * test_fraction))
    val_n = int(round(num_samples * val_fraction))
    test_idx = indices[:test_n]
    val_idx = indices[test_n:test_n + val_n]
    train_idx = indices[test_n + val_n:]
    return train_idx, val_idx, test_idx


def split_indices_by_record(records, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Split by record id so beats from one patient/record do not leak across sets."""
    records = np.asarray(records)
    unique_records = np.unique(records)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_records)

    test_n = max(1, int(round(len(shuffled) * test_fraction)))
    val_n = max(1, int(round(len(shuffled) * val_fraction)))
    if test_n + val_n >= len(shuffled):
        raise ValueError("Record-wise split needs more records or smaller val/test fractions.")

    test_records = shuffled[:test_n]
    val_records = shuffled[test_n:test_n + val_n]
    train_records = shuffled[test_n + val_n:]

    train_idx = np.flatnonzero(np.isin(records, train_records))
    val_idx = np.flatnonzero(np.isin(records, val_records))
    test_idx = np.flatnonzero(np.isin(records, test_records))
    return train_idx, val_idx, test_idx, train_records, val_records, test_records


def load_split(data_npz, split_mode="random", val_fraction=0.15, test_fraction=0.15, seed=42):
    pack = np.load(data_npz, allow_pickle=True)
    labels = pack["labels"]
    if split_mode == "random":
        train_idx, val_idx, test_idx = split_indices(
            len(labels),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        return train_idx, val_idx, test_idx, {}

    if split_mode == "record":
        if "records" not in pack:
            raise ValueError("Record-wise split requires a data npz prepared with the refined ecg_dataset.py.")
        train_idx, val_idx, test_idx, train_records, val_records, test_records = split_indices_by_record(
            pack["records"],
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        meta = {
            "train_records": [str(x) for x in train_records],
            "val_records": [str(x) for x in val_records],
            "test_records": [str(x) for x in test_records],
        }
        return train_idx, val_idx, test_idx, meta

    raise ValueError(f"Unknown split_mode: {split_mode}")


def class_counts(labels, num_classes):
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)


def inverse_frequency_weights(labels, num_classes):
    counts = class_counts(labels, num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (num_classes * counts[nonzero])
    return weights


def make_beat_window(signal, center, before=90, after=144):
    start = center - before
    end = center + after
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def prepare_mitdb_npz(mitdb_dir, out_npz, lead=0, before=90, after=144, max_records=0):
    """Convert local MIT-BIH records to fixed-length beat segments."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb first: pip install -r module2_ecg/requirements_ecg.txt") from exc

    mitdb_path = Path(mitdb_dir)
    record_names = sorted(p.stem for p in mitdb_path.glob("*.hea"))
    if max_records and max_records > 0:
        record_names = record_names[:max_records]
    if not record_names:
        raise FileNotFoundError(f"No MIT-BIH .hea files found in {mitdb_dir}")

    signals, labels, records, samples, symbols = [], [], [], [], []
    for record_name in record_names:
        record_path = str(mitdb_path / record_name)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")
        ecg = record.p_signal[:, lead].astype(np.float32)

        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in AAMI_LABEL_MAP:
                continue
            segment = make_beat_window(ecg, int(sample), before=before, after=after)
            if segment is None:
                continue
            signals.append(segment)
            labels.append(AAMI_LABEL_MAP[symbol])
            records.append(record_name)
            samples.append(int(sample))
            symbols.append(symbol)

    if not signals:
        raise RuntimeError("No usable beat segments were extracted.")

    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        signals=np.asarray(signals, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        records=np.asarray(records),
        samples=np.asarray(samples, dtype=np.int64),
        symbols=np.asarray(symbols),
        label_names=DEFAULT_LABEL_NAMES,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH beat segments for Module 2 ECG.")
    parser.add_argument("--mitdb_dir", required=True, help="Directory containing MIT-BIH .hea/.dat/.atr files")
    parser.add_argument("--out_npz", default=project_path("data", "ecg", "mitdb_beat_segments.npz"))
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--before", type=int, default=90)
    parser.add_argument("--after", type=int, default=144)
    parser.add_argument("--max_records", type=int, default=0, help="0 = use all records")
    args = parser.parse_args()
    args.mitdb_dir = resolve_project_path(args.mitdb_dir)
    args.out_npz = resolve_project_path(args.out_npz)

    out_path = prepare_mitdb_npz(
        args.mitdb_dir,
        args.out_npz,
        lead=args.lead,
        before=args.before,
        after=args.after,
        max_records=args.max_records,
    )
    print(f"Saved prepared ECG dataset to: {out_path}")


if __name__ == "__main__":
    main()
