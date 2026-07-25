from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from circuit_physics import ideal_pairwise_terminal_currents
from hybrid_physics_gnn import HybridPhysicsGNN
from train_real_csv_torch import choose_probability_threshold, load_csv, split_train_val


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "resistance_8x8_simon_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resistance_results" / "hybrid_gnn_8x8_v2"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def currents_from_maps(resistance: np.ndarray, voltage: float, batch_size: int = 256) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(resistance), batch_size):
            batch = torch.tensor(resistance[start : start + batch_size], dtype=torch.float32)
            chunks.append(ideal_pairwise_terminal_currents(batch, voltage=voltage).numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def generate_sparse_curriculum(
    n_samples: int,
    n_cells: int,
    low_ohm: float,
    high_ohm: float,
    max_high_cells: int,
    voltage: float,
    seed: int,
    excluded_maps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0:
        return np.empty((0, n_cells), dtype=np.float32), np.empty((0, n_cells), dtype=np.float32)
    rng = np.random.default_rng(seed)
    max_high_cells = min(max(max_high_cells, 1), n_cells)
    excluded_keys = set()
    if excluded_maps is not None:
        excluded_binary = np.where(excluded_maps > (low_ohm + high_ohm) / 2, high_ohm, low_ohm).astype(np.float32)
        excluded_keys = {row.tobytes() for row in excluded_binary}

    accepted = []
    while len(accepted) < n_samples:
        values = np.full(n_cells, low_ohm, dtype=np.float32)
        n_high = int(rng.integers(1, max_high_cells + 1))
        values[rng.choice(n_cells, size=n_high, replace=False)] = high_ohm
        if values.tobytes() not in excluded_keys:
            accepted.append(values)
    resistance = np.stack(accepted, axis=0)
    current = currents_from_maps(resistance, voltage=voltage)
    return current, resistance


@torch.no_grad()
def predict_outputs(
    model: HybridPhysicsGNN,
    current: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(current), batch_size):
        batch = torch.tensor(current[start : start + batch_size], dtype=torch.float32, device=device)
        output = model(batch)
        for key, value in output.items():
            chunks.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def high_f1(true_r: np.ndarray, probability: np.ndarray, threshold: float, probability_threshold: float) -> float:
    true_high = true_r > threshold
    pred_high = probability >= probability_threshold
    tp = float((true_high & pred_high).sum())
    fp = float((~true_high & pred_high).sum())
    fn = float((true_high & ~pred_high).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def training_loss(
    model: HybridPhysicsGNN,
    current: torch.Tensor,
    true_r: torch.Tensor,
    args: argparse.Namespace,
    positive_weight: torch.Tensor,
    compute_forward: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(current)
    true_log_r = torch.log(true_r.clamp_min(args.prediction_min))
    error = output["log_resistance"] - true_log_r
    sigma = output["log_resistance_uncertainty"]
    cell_weight = torch.where(true_r > args.threshold, args.high_cell_weight, 1.0)
    supervised = (cell_weight * (0.5 * (error / sigma).square() + torch.log(sigma))).sum() / cell_weight.sum()

    high_target = (true_r > args.threshold).float()
    classification = nn.functional.binary_cross_entropy_with_logits(
        output["high_state_logit"],
        high_target,
        pos_weight=positive_weight,
    )
    correction = output["delta_log_resistance"].square().mean()
    calibration = model.calibration_penalty()
    forward = torch.zeros((), dtype=current.dtype, device=current.device)
    if compute_forward and args.forward_consistency_weight > 0:
        n_forward = min(args.forward_batch_size, len(current))
        chosen = torch.randperm(len(current), device=current.device)[:n_forward]
        ideal_current = ideal_pairwise_terminal_currents(
            output["resistance"][chosen],
            voltage=args.voltage,
        )
        reconstructed = model.calibrate_forward_current(ideal_current)
        forward = nn.functional.mse_loss(
            torch.log(reconstructed),
            torch.log(current[chosen].clamp_min(1e-8)),
        )

    total = (
        supervised
        + args.classification_weight * classification
        + args.forward_consistency_weight * forward
        + args.correction_weight * correction
        + args.calibration_weight * calibration
    )
    parts = {
        "total": float(total.detach()),
        "supervised": float(supervised.detach()),
        "classification": float(classification.detach()),
        "forward_consistency": float(forward.detach()),
        "correction": float(correction.detach()),
        "calibration": float(calibration.detach()),
    }
    return total, parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Module 4 hybrid physics-correction GNN.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--message_steps", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--prediction_min", type=float, default=1.0)
    parser.add_argument("--prediction_max", type=float, default=110.0)
    parser.add_argument("--max_log_correction", type=float, default=5.0)
    parser.add_argument("--high_cell_weight", type=float, default=4.0)
    parser.add_argument("--classification_weight", type=float, default=0.25)
    parser.add_argument("--forward_consistency_weight", type=float, default=0.05)
    parser.add_argument("--correction_weight", type=float, default=0.002)
    parser.add_argument("--calibration_weight", type=float, default=0.1)
    parser.add_argument("--forward_batch_size", type=int, default=32)
    parser.add_argument("--synthetic_sparse_samples", type=int, default=4000)
    parser.add_argument("--synthetic_max_high_cells", type=int, default=8)
    parser.add_argument("--synthetic_low_ohm", type=float, default=1.0)
    parser.add_argument("--synthetic_high_ohm", type=float, default=100.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    i_all = load_csv(args.data_dir / "I_train.csv")
    r_all = load_csv(args.data_dir / "R_train.csv")
    r_holdout = load_csv(args.data_dir / "R_test.csv")
    if args.max_train_samples is not None and args.max_train_samples < len(i_all):
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(i_all), size=args.max_train_samples, replace=False)
        i_all, r_all = i_all[chosen], r_all[chosen]
    i_train, r_train, i_val, r_val = split_train_val(i_all, r_all, args.seed)

    synthetic_i, synthetic_r = generate_sparse_curriculum(
        args.synthetic_sparse_samples,
        r_all.shape[1],
        args.synthetic_low_ohm,
        args.synthetic_high_ohm,
        args.synthetic_max_high_cells,
        args.voltage,
        args.seed + 101,
        excluded_maps=r_holdout,
    )
    if len(synthetic_i):
        synthetic_train_i, synthetic_train_r, synthetic_val_i, synthetic_val_r = split_train_val(
            synthetic_i,
            synthetic_r,
            args.seed + 102,
        )
        i_train = np.concatenate([i_train, synthetic_train_i], axis=0)
        r_train = np.concatenate([r_train, synthetic_train_r], axis=0)
        i_val = np.concatenate([i_val, synthetic_val_i], axis=0)
        r_val = np.concatenate([r_val, synthetic_val_r], axis=0)

    n_cells = r_train.shape[1]
    grid_size = int(round(n_cells**0.5))
    if grid_size * grid_size != n_cells:
        raise ValueError("HybridPhysicsGNN requires square resistance maps")
    uniform_factor = n_cells / (2 * grid_size - 1)
    train_proxy = np.clip(args.voltage / np.maximum(i_train, 1e-8) * uniform_factor, args.prediction_min, args.prediction_max)
    log_current = np.log(np.maximum(i_train, 1e-8))
    log_proxy = np.log(train_proxy)
    normalization = {
        "log_current_mean": log_current.mean(axis=0).astype(np.float32),
        "log_current_std": (log_current.std(axis=0) + 1e-6).astype(np.float32),
        "log_proxy_mean": log_proxy.mean(axis=0).astype(np.float32),
        "log_proxy_std": (log_proxy.std(axis=0) + 1e-6).astype(np.float32),
        "current_scale": float(np.median(i_train)),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPhysicsGNN(
        grid_size=grid_size,
        hidden_size=args.hidden_size,
        message_steps=args.message_steps,
        dropout=args.dropout,
        voltage=args.voltage,
        prediction_min=args.prediction_min,
        prediction_max=args.prediction_max,
        max_log_correction=args.max_log_correction,
        **{key: torch.tensor(value) if key != "current_scale" else value for key, value in normalization.items()},
    ).to(device)

    high_count = float((r_train > args.threshold).sum())
    low_count = float(r_train.size - high_count)
    positive_weight = torch.tensor(low_count / max(high_count, 1.0), dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(torch.tensor(i_train), torch.tensor(r_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_score = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_parts: dict[str, list[float]] = {}
        for current, true_r in loader:
            current = current.to(device=device, dtype=torch.float32)
            true_r = true_r.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            loss, parts = training_loss(model, current, true_r, args, positive_weight, compute_forward=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in parts.items():
                epoch_parts.setdefault(key, []).append(value)

        val_output = predict_outputs(model, i_val, args.batch_size, device)
        val_log_rmse = float(
            np.sqrt(np.mean((val_output["log_resistance"] - np.log(np.maximum(r_val, args.prediction_min))) ** 2))
        )
        val_probability = 1.0 / (1.0 + np.exp(-val_output["high_state_logit"]))
        val_high_f1 = max(high_f1(r_val, val_probability, args.threshold, p) for p in np.linspace(0.1, 0.9, 17))
        selection_score = val_log_rmse + 0.25 * (1.0 - val_high_f1)
        row = {
            "epoch": epoch,
            **{key: float(np.mean(values)) for key, values in epoch_parts.items()},
            "val_log_rmse": val_log_rmse,
            "val_high_f1": val_high_f1,
            "selection_score": selection_score,
        }
        history.append(row)
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} total={row['total']:.4f} "
                f"map={row['supervised']:.4f} forward={row['forward_consistency']:.4f} "
                f"val_log_rmse={val_log_rmse:.4f} val_high_f1={val_high_f1:.4f}"
            )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_output = predict_outputs(model, i_val, args.batch_size, device)
    val_probability = 1.0 / (1.0 + np.exp(-val_output["high_state_logit"]))
    probability_threshold, threshold_rows = choose_probability_threshold(
        r_val,
        val_probability,
        args.threshold,
        "high",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "hybrid_physics_gnn_best.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": "hybrid_physics_gnn",
        "grid_size": grid_size,
        "hidden_size": args.hidden_size,
        "message_steps": args.message_steps,
        "dropout": args.dropout,
        "voltage": args.voltage,
        "prediction_min": args.prediction_min,
        "prediction_max": args.prediction_max,
        "max_log_correction": args.max_log_correction,
        "threshold": args.threshold,
        "probability_threshold": probability_threshold,
        "normalization": normalization,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "training_arguments": vars(args),
        "calibration": model.calibration_summary(),
        "forward_model": "ideal_pairwise_terminal_kcl_laplacian",
        "forward_model_experimentally_validated": False,
    }
    torch.save(checkpoint, checkpoint_path)
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "validation_threshold_scan.json").write_text(
        json.dumps(threshold_rows, indent=2),
        encoding="utf-8",
    )
    data_summary = {
        "original_samples_used": int(len(i_all)),
        "synthetic_sparse_samples": int(len(synthetic_i)),
        "combined_train_samples": int(len(i_train)),
        "combined_validation_samples": int(len(i_val)),
        "train_high_cell_fraction": float((r_train > args.threshold).mean()),
        "validation_high_cell_fraction": float((r_val > args.threshold).mean()),
        "curriculum_exact_test_maps_excluded": True,
    }
    (args.output_dir / "training_data_summary.json").write_text(
        json.dumps(data_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(data_summary, indent=2))
    print(f"Best epoch: {best_epoch}; validation probability threshold: {probability_threshold:.3f}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
