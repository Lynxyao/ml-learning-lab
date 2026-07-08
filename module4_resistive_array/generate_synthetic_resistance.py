from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from resistance_physics import make_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/resistance/resistance_inverse_synthetic.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic resistive-array inverse-sensing data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--grid_size", type=int, choices=[2, 3, 4, 5, 6], default=3)
    parser.add_argument("--measurement_noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    measurements, maps, names = make_dataset(
        n_samples=args.samples,
        grid_size=args.grid_size,
        measurement_noise=args.measurement_noise,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        measurements=measurements,
        resistance_maps=maps,
        measurement_names=np.array(names, dtype=object),
        grid_size=np.array(args.grid_size, dtype=np.int64),
        measurement_noise=np.array(args.measurement_noise, dtype=np.float32),
    )

    print(f"Saved {len(measurements)} synthetic samples to {args.output}")
    print(f"Measurement input shape: {measurements.shape} fixed-voltage current features")
    print(f"Hidden output map shape: {maps.shape} local resistance values")
    print("Measurements:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
