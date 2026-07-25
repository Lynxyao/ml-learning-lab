from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a plain 64-to-64 MLP with no physics-informed loss."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--curriculum_samples", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    suffix = "original" if args.curriculum_samples == 0 else f"curriculum{args.curriculum_samples}"
    data_dir = PROJECT_ROOT / "data" / "resistance_8x8_simon_v2"
    output_dir = PROJECT_ROOT / "resistance_results" / f"mlp_baseline_8x8_{suffix}"
    mode_name = "regression_conductance_mlp"
    if args.curriculum_samples > 0:
        mode_name += f"_curriculum{args.curriculum_samples}"
    checkpoint = output_dir / "checkpoints" / f"{mode_name}_model.pt"

    train_command = [
        sys.executable,
        str(Path(__file__).with_name("train_real_csv_torch.py")),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--model",
        "mlp",
        "--task",
        "regression",
        "--target_transform",
        "conductance",
        "--forward_consistency_weight",
        "0",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hidden_size",
        str(args.hidden_size),
        "--seed",
        str(args.seed),
        "--synthetic_sparse_samples",
        str(args.curriculum_samples),
    ]
    if args.max_train_samples is not None:
        train_command.extend(["--max_train_samples", str(args.max_train_samples)])

    print("Plain MLP baseline: 64 currents -> 64 conductance targets -> 64 resistances")
    print("Physics-informed loss: disabled (weight = 0)")
    print(f"Sparse curriculum samples: {args.curriculum_samples}")
    subprocess.run(train_command, check=True)

    test_command = [
        sys.executable,
        str(Path(__file__).with_name("test_real_csv_torch.py")),
        "--checkpoint",
        str(checkpoint),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.batch_size),
    ]
    subprocess.run(test_command, check=True)
    print(f"\nComplete. Results: {output_dir}")


if __name__ == "__main__":
    main()
