from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "fall_results/checkpoint_series"
DEFAULT_WEBSITE_ASSETS = PROJECT_ROOT / "website/assets/module3"


def plot_confusion_matrix(ax: plt.Axes, matrix: np.ndarray) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Held-out confusion matrix")
    ax.set_xticks([0, 1], labels=["ADL", "Fall"])
    ax.set_yticks([0, 1], labels=["ADL", "Fall"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    threshold = float(matrix.max()) / 2.0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "#17324d",
                fontweight="bold",
            )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def render_epoch(
    result: dict[str, object],
    predictions_path: Path,
    output_path: Path,
) -> None:
    arrays = np.load(predictions_path)
    probabilities = arrays["test_probs"]
    labels = arrays["y_test"]
    metrics = result["test"]
    matrix = np.asarray(metrics["confusion_matrix_adl_fall"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    plot_confusion_matrix(axes[0], matrix)

    axes[1].hist(
        probabilities[labels == 0],
        bins=np.linspace(0, 1, 26),
        alpha=0.72,
        color="#2a7f9e",
        label="Ground-truth ADL",
    )
    axes[1].hist(
        probabilities[labels == 1],
        bins=np.linspace(0, 1, 26),
        alpha=0.65,
        color="#d95f59",
        label="Ground-truth fall",
    )
    axes[1].axvline(0.5, color="#222222", linestyle="--", linewidth=1.2)
    axes[1].set_title("Held-out fall probabilities")
    axes[1].set_xlabel("Predicted probability of fall")
    axes[1].set_ylabel("Number of windows")
    axes[1].legend(fontsize=8)

    metric_names = ["Accuracy", "Fall recall", "Fall F1"]
    metric_values = [
        float(metrics["accuracy"]),
        float(metrics["recall_fall"]),
        float(metrics["f1_fall"]),
    ]
    bars = axes[2].bar(
        metric_names,
        metric_values,
        color=["#256d85", "#d08b30", "#3c8d70"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Epoch {int(result['epoch'])} held-out metrics")
    axes[2].set_ylabel("Score")
    axes[2].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, metric_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )

    fig.suptitle(
        "Bidirectional GRU checkpoint: real predictions on a fixed subject-held-out split",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Module 3 epoch checkpoint figures.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--website_assets", type=Path, default=DEFAULT_WEBSITE_ASSETS)
    args = parser.parse_args()

    results_path = args.results_dir / "checkpoint_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    args.website_assets.mkdir(parents=True, exist_ok=True)

    for result in payload["checkpoints"]:
        epoch = int(result["epoch"])
        render_epoch(
            result,
            args.results_dir / f"epoch_{epoch:03d}_predictions.npz",
            args.website_assets / f"checkpoint-epoch-{epoch:02d}.png",
        )

    shutil.copy2(results_path, args.website_assets / "checkpoint-results.json")
    print(f"Rendered {len(payload['checkpoints'])} checkpoint figures to {args.website_assets}")


if __name__ == "__main__":
    main()
