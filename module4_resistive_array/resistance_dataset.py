from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResistanceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    measurement_names: list[str]
    grid_size: int


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    data = np.load(path, allow_pickle=True)
    x = data["measurements"].astype(np.float32)
    maps = data["resistance_maps"].astype(np.float32)
    names = [str(v) for v in data["measurement_names"].tolist()]
    grid_size = int(data["grid_size"])
    return x, maps, names, grid_size


def split_and_scale(
    x: np.ndarray,
    maps: np.ndarray,
    measurement_names: list[str],
    grid_size: int,
    seed: int = 7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> ResistanceData:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = int(round(len(x) * test_fraction))
    n_val = int(round(len(x) * val_fraction))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train = maps[train_idx].reshape(len(train_idx), -1)
    y_val = maps[val_idx].reshape(len(val_idx), -1)
    y_test = maps[test_idx].reshape(len(test_idx), -1)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6

    return ResistanceData(
        x_train=(x_train - x_mean) / x_std,
        y_train=(y_train - y_mean) / y_std,
        x_val=(x_val - x_mean) / x_std,
        y_val=(y_val - y_mean) / y_std,
        x_test=(x_test - x_mean) / x_std,
        y_test=(y_test - y_mean) / y_std,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        measurement_names=measurement_names,
        grid_size=grid_size,
    )


def unscale_maps(y_scaled: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, grid_size: int) -> np.ndarray:
    y = y_scaled * y_std + y_mean
    return y.reshape(len(y), grid_size, grid_size)
