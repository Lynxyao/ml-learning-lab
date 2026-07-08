from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "external/fall_detection_rnn/UniMiB-SHAR/data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fall/unimib_fall_windows.npz"


def _load_mat(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download or place the UniMiB-SHAR data first.")
    return scipy.io.loadmat(path)[key]


def reshape_windows(flat_data: np.ndarray) -> np.ndarray:
    """Convert UniMiB rows from [x151, y151, z151] to [151, 3]."""
    if flat_data.shape[1] != 453:
        raise ValueError(f"Expected 453 columns, got {flat_data.shape[1]}")
    x = flat_data[:, 0:151]
    y = flat_data[:, 151:302]
    z = flat_data[:, 302:453]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def prepare(source_dir: Path, output_path: Path) -> Path:
    fall_data = _load_mat(source_dir / "fall_data.mat", "fall_data")
    fall_labels = _load_mat(source_dir / "fall_labels.mat", "fall_labels")
    adl_data = _load_mat(source_dir / "adl_data.mat", "adl_data")
    adl_labels = _load_mat(source_dir / "adl_labels.mat", "adl_labels")

    x = np.concatenate([reshape_windows(adl_data), reshape_windows(fall_data)], axis=0)
    y = np.concatenate(
        [np.zeros(len(adl_data), dtype=np.int64), np.ones(len(fall_data), dtype=np.int64)],
        axis=0,
    )
    subject_id = np.concatenate(
        [adl_labels[:, 1].astype(np.int64), fall_labels[:, 1].astype(np.int64)],
        axis=0,
    )
    activity_id = np.concatenate(
        [adl_labels[:, 0].astype(np.int64), fall_labels[:, 0].astype(np.int64)],
        axis=0,
    )
    trial_id = np.concatenate(
        [adl_labels[:, 2].astype(np.int64), fall_labels[:, 2].astype(np.int64)],
        axis=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=x,
        y=y,
        subject_id=subject_id,
        activity_id=activity_id,
        trial_id=trial_id,
        label_names=np.array(["ADL", "Fall"]),
        source="UniMiB-SHAR",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UniMiB-SHAR fall prediction data.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = prepare(args.source_dir, args.output)
    data = np.load(output)
    x, y = data["X"], data["y"]
    print(f"Wrote {output}")
    print(f"X shape: {x.shape}  y shape: {y.shape}")
    print(f"ADL samples: {(y == 0).sum()}  Fall samples: {(y == 1).sum()}")


if __name__ == "__main__":
    main()
