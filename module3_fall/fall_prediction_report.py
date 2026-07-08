from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fall_dataset import load_npz, split_indices
from fall_models import FallRNN
from fall_train_torch import predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/fall/unimib_fall_windows.npz"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "fall_results/checkpoints/best_model.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "fall_results/prediction_report"
LABELS = np.array(["ADL", "Fall"])


def _scale_with_checkpoint(x: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    return (x - mean) / (std + 1e-6)


def save_prediction_csv(
    output_path: Path,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    subject_id: np.ndarray | None,
    activity_id: np.ndarray | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "test_order",
                "original_sample_index",
                "subject_id",
                "activity_id",
                "ground_truth",
                "predicted",
                "probability_fall",
                "correct",
            ]
        )
        for row_num, original_idx in enumerate(test_idx):
            writer.writerow(
                [
                    row_num,
                    int(original_idx),
                    "" if subject_id is None else int(subject_id[original_idx]),
                    "" if activity_id is None else int(activity_id[original_idx]),
                    LABELS[y_true[row_num]],
                    LABELS[preds[row_num]],
                    f"{float(probs[row_num]):.6f}",
                    bool(y_true[row_num] == preds[row_num]),
                ]
            )


def plot_confusion_matrix(y_true: np.ndarray, preds: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Fall Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_probability_histogram(y_true: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(probs[y_true == 0], bins=30, alpha=0.7, label="Ground truth ADL")
    plt.hist(probs[y_true == 1], bins=30, alpha=0.7, label="Ground truth Fall")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision threshold")
    plt.xlabel("Predicted probability of Fall")
    plt.ylabel("Number of test samples")
    plt.title("Model Confidence on Test Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_example_windows(
    x_test_raw: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
    max_examples: int,
) -> None:
    wrong = np.where(y_true != preds)[0]
    right_fall = np.where((y_true == 1) & (preds == 1))[0]
    right_adl = np.where((y_true == 0) & (preds == 0))[0]
    selected = np.concatenate([wrong[:4], right_fall[:4], right_adl[:4]])[:max_examples]
    if len(selected) == 0:
        selected = np.arange(min(max_examples, len(y_true)))

    cols = 2
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.2 * rows, 4)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    channels = ["x", "y", "z"]
    colors = ["#0B7285", "#C92A2A", "#5C940D"]

    for ax, idx in zip(axes, selected):
        for ch, name in enumerate(channels):
            ax.plot(x_test_raw[idx, :, ch], label=name, linewidth=1, color=colors[ch])
        status = "correct" if y_true[idx] == preds[idx] else "wrong"
        ax.set_title(
            f"{status}: truth={LABELS[y_true[idx]]}, pred={LABELS[preds[idx]]}, "
            f"P(Fall)={probs[idx]:.2f}"
        )
        ax.set_ylabel("Acceleration")
        ax.grid(alpha=0.25)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    axes[0].legend(loc="upper right")
    fig.supxlabel("Time step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_prediction_report(
    output_dir: Path,
    x_test_raw: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    test_idx: np.ndarray,
    subject_id: np.ndarray | None = None,
    activity_id: np.ndarray | None = None,
    max_examples: int = 12,
) -> list[Path]:
    """Save per-sample predictions and visual summaries for one test run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_sample_predictions.csv"
    confusion_path = output_dir / "confusion_matrix.png"
    histogram_path = output_dir / "fall_probability_histogram.png"
    examples_path = output_dir / "example_test_predictions.png"

    save_prediction_csv(csv_path, test_idx, y_true, probs, preds, subject_id, activity_id)
    plot_confusion_matrix(y_true, preds, confusion_path)
    plot_probability_histogram(y_true, probs, histogram_path)
    plot_example_windows(x_test_raw, y_true, probs, preds, examples_path, max_examples)
    return [csv_path, confusion_path, histogram_path, examples_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual and CSV prediction reports for Module 3.")
    parser.add_argument("--data_npz", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_examples", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["subject_id"].astype(np.int64) if "subject_id" in data.files else None
    subject_id = data["subject_id"].astype(np.int64) if "subject_id" in data.files else None
    activity_id = data["activity_id"].astype(np.int64) if "activity_id" in data.files else None

    _, _, test_idx = split_indices(x, y, groups)
    x_test_raw = x[test_idx]
    y_test = y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    x_test_scaled = _scale_with_checkpoint(x_test_raw, checkpoint)

    model = FallRNN(model=checkpoint["model_type"], hidden_size=int(checkpoint["hidden_size"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    probs, preds = predict(model, x_test_scaled, args.batch_size, device)
    outputs = generate_prediction_report(
        args.output_dir,
        x_test_raw,
        y_test,
        probs,
        preds,
        test_idx,
        subject_id,
        activity_id,
        args.max_examples,
    )

    for output in outputs:
        print(f"Saved report output: {output}")


if __name__ == "__main__":
    main()
