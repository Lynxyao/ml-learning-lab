from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_hybrid_physics_gnn import generate_sparse_curriculum
from train_real_csv_torch import load_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "resistance_8x8_simon_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "resistance_8x8_simon_v2_curriculum"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fair shared sparse-curriculum CSV dataset.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--original_samples", type=int, default=3000)
    parser.add_argument("--synthetic_samples", type=int, default=3000)
    parser.add_argument("--max_high_cells", type=int, default=8)
    parser.add_argument("--low_ohm", type=float, default=1.0)
    parser.add_argument("--high_ohm", type=float, default=100.0)
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    i_train = load_csv(args.source_dir / "I_train.csv")
    r_train = load_csv(args.source_dir / "R_train.csv")
    i_test = load_csv(args.source_dir / "I_test.csv")
    r_test = load_csv(args.source_dir / "R_test.csv")
    rng = np.random.default_rng(args.seed)
    if args.original_samples < len(i_train):
        chosen = rng.choice(len(i_train), size=args.original_samples, replace=False)
        i_train, r_train = i_train[chosen], r_train[chosen]

    synthetic_i, synthetic_r = generate_sparse_curriculum(
        args.synthetic_samples,
        r_train.shape[1],
        args.low_ohm,
        args.high_ohm,
        args.max_high_cells,
        args.voltage,
        args.seed + 101,
        excluded_maps=r_test,
    )
    combined_i = np.concatenate([i_train, synthetic_i], axis=0)
    combined_r = np.concatenate([r_train, synthetic_r], axis=0)
    order = rng.permutation(len(combined_i))
    combined_i, combined_r = combined_i[order], combined_r[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "I_train.csv", combined_i, delimiter=",", fmt="%.9g")
    np.savetxt(args.output_dir / "R_train.csv", combined_r, delimiter=",", fmt="%.9g")
    np.savetxt(args.output_dir / "I_test.csv", i_test, delimiter=",", fmt="%.9g")
    np.savetxt(args.output_dir / "R_test.csv", r_test, delimiter=",", fmt="%.9g")
    metadata = {
        "source_dir": str(args.source_dir),
        "original_samples": int(len(i_train)),
        "synthetic_sparse_samples": int(len(synthetic_i)),
        "total_train_samples": int(len(combined_i)),
        "max_high_cells_per_synthetic_map": args.max_high_cells,
        "train_high_cell_fraction": float((combined_r > 50.0).mean()),
        "test_high_cell_fraction": float((r_test > 50.0).mean()),
        "curriculum_exact_test_maps_excluded": True,
        "forward_model": "ideal_pairwise_terminal_kcl_laplacian",
        "experimentally_validated": False,
        "seed": args.seed,
    }
    (args.output_dir / "curriculum_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
