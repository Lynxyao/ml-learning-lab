from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class FallData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["subject_id"].astype(np.int64) if "subject_id" in data.files else None
    return x, y, groups


def split_indices(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 7,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(x))
    if groups is not None:
        first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(first.split(x, y, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
        train_rel, val_rel = next(second.split(x[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]))
        return train_val_idx[train_rel], train_val_idx[val_rel], test_idx

    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, random_state=seed, stratify=y[train_val_idx]
    )
    return train_idx, val_idx, test_idx


def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 7,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> FallData:
    train_idx, val_idx, test_idx = split_indices(x, y, groups, seed, test_size, val_size)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return FallData(
        x_train=(x_train - mean) / std,
        y_train=y_train,
        x_val=(x_val - mean) / std,
        y_val=y_val,
        x_test=(x_test - mean) / std,
        y_test=y_test,
        mean=mean,
        std=std,
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fall": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fall": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_adl_fall": cm.tolist(),
    }
