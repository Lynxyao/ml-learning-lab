from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/fall/unimib_fall_windows.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "fall_results/figures/example_windows.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fall and ADL accelerometer windows.")
    parser.add_argument("--data_npz", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    names = ["x", "y", "z"]

    adl_idx = int(np.where(y == 0)[0][0])
    fall_idx = int(np.where(y == 1)[0][0])
    examples = [("ADL example", adl_idx), ("Fall example", fall_idx)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (title, idx) in zip(axes, examples):
        for channel in range(3):
            ax.plot(x[idx, :, channel], label=names[channel], linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel("Acceleration")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time step")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
