from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RESERVED_COLUMNS = {"sample_id", "frame", "label", "subject_id"}


def parse_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "fall", "risk", "high", "positive"}:
        return 1
    if text in {"0", "adl", "nonfall", "non-fall", "low", "negative"}:
        return 0
    raise ValueError(f"Unsupported label {value!r}; use 0/1 or ADL/fall.")


def load_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON must be an array of row objects or {'rows': [...]}.")
        return rows
    raise ValueError("Input must be CSV or JSON.")


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == length:
        return sequence
    old_axis = np.linspace(0.0, 1.0, len(sequence))
    new_axis = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(new_axis, old_axis, sequence[:, column]) for column in range(sequence.shape[1])],
        axis=1,
    ).astype(np.float32)


def convert(rows: list[dict[str, object]], sequence_length: int) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("No rows found.")
    required = {"sample_id", "frame", "label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_names = [name for name in rows[0] if name not in RESERVED_COLUMNS]
    if not feature_names:
        raise ValueError("At least one numeric feature column is required.")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    sequences, labels, subjects, sample_ids = [], [], [], []
    for sample_id, sample_rows in grouped.items():
        sample_rows.sort(key=lambda row: float(row["frame"]))
        if len(sample_rows) < 2:
            raise ValueError(f"Sample {sample_id!r} has fewer than two frames.")
        sample_labels = {parse_label(row["label"]) for row in sample_rows}
        if len(sample_labels) != 1:
            raise ValueError(f"Sample {sample_id!r} has inconsistent labels.")
        try:
            sequence = np.asarray(
                [[float(row[name]) for name in feature_names] for row in sample_rows],
                dtype=np.float32,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Sample {sample_id!r} contains missing or nonnumeric features.") from exc
        if not np.isfinite(sequence).all():
            raise ValueError(f"Sample {sample_id!r} contains NaN or infinite values.")

        sequences.append(resample(sequence, sequence_length))
        labels.append(sample_labels.pop())
        subjects.append(str(sample_rows[0].get("subject_id") or sample_id))
        sample_ids.append(sample_id)

    subject_lookup = {value: index for index, value in enumerate(sorted(set(subjects)))}
    return {
        "X": np.stack(sequences).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "subject_id": np.asarray([subject_lookup[value] for value in subjects], dtype=np.int64),
        "sample_id": np.asarray(sample_ids),
        "feature_names": np.asarray(feature_names),
        "source": np.asarray(["Holomotion-compatible labeled sequence import"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert labeled Holomotion-style CSV/JSON time-series rows to Module 3 NPZ."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence_length", type=int, default=151)
    args = parser.parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    arrays = convert(load_rows(args.input), args.sequence_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {len(arrays['y'])} sequences to {args.output}")
    print(f"Shape: {arrays['X'].shape}; features: {arrays['feature_names'].tolist()}")


if __name__ == "__main__":
    main()
