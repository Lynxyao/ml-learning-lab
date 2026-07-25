from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dynamic_impedance import CircuitGraph, DynamicImpedanceGNN, MeasurementBatch, pairwise_row_column_protocol
from dynamic_impedance.adapters import load_numeric_csv
from dynamic_impedance.data_schema import PhysicalParameterBatch
from dynamic_impedance.mna import dc_pair_currents
from dynamic_impedance.physics_losses import PhysicsLossWeights, total_inverse_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_sparse_curriculum(
    n_samples: int,
    n_cells: int,
    graph: CircuitGraph,
    protocol: torch.Tensor,
    voltage: float,
    low_ohm: float,
    high_ohm: float,
    max_high_cells: int,
    seed: int,
    excluded_maps: np.ndarray | None,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_samples <= 0:
        empty = torch.empty(0, n_cells, dtype=torch.float32)
        return empty, empty
    rng = np.random.default_rng(seed)
    excluded = set()
    if excluded_maps is not None:
        binary = np.where(excluded_maps > (low_ohm + high_ohm) / 2, high_ohm, low_ohm).astype(np.float32)
        excluded = {row.tobytes() for row in binary}
    accepted: list[np.ndarray] = []
    accepted_keys: set[bytes] = set()
    while len(accepted) < n_samples:
        values = np.full(n_cells, low_ohm, dtype=np.float32)
        n_high = int(rng.integers(1, min(max_high_cells, n_cells) + 1))
        values[rng.choice(n_cells, n_high, replace=False)] = high_ohm
        key = values.tobytes()
        if key not in excluded and key not in accepted_keys:
            accepted.append(values)
            accepted_keys.add(key)
    resistance = torch.from_numpy(np.stack(accepted))
    current_chunks = []
    protocol_device = protocol.to(device)
    for start in range(0, n_samples, batch_size):
        maps = resistance[start : start + batch_size].to(device)
        current_chunks.append(dc_pair_currents(graph, maps, protocol_device, voltage=voltage).cpu())
    return torch.cat(current_chunks), resistance


@torch.no_grad()
def evaluate(
    model: DynamicImpedanceGNN,
    currents: torch.Tensor,
    resistance: torch.Tensor,
    protocol: torch.Tensor,
    batch_size: int,
    device: torch.device,
    voltage: float,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    prediction = []
    probability = []
    for start in range(0, len(currents), batch_size):
        current = currents[start : start + batch_size].to(device)
        batch = MeasurementBatch.from_dc_currents(current, protocol.to(device), voltage=voltage)
        output = model(batch)
        prediction.append(output["resistance"].cpu())
        probability.append(torch.sigmoid(output["high_state_logit"]).cpu())
    predicted = torch.cat(prediction)
    probability = torch.cat(probability)
    true_high = resistance > threshold
    pred_high = probability >= 0.5
    tp = (true_high & pred_high).sum().item()
    fp = (~true_high & pred_high).sum().item()
    fn = (true_high & ~pred_high).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "mae_ohm": float((predicted - resistance).abs().mean()),
        "rmse_ohm": float(torch.sqrt((predicted - resistance).square().mean())),
        "log_rmse": float(torch.sqrt((predicted.clamp_min(1e-12).log() - resistance.log()).square().mean())),
        "high_precision": precision,
        "high_recall": recall,
        "high_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "cell_accuracy": float((true_high == pred_high).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frequency-ready impedance GNN in DC-only mode.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "resistance_8x8_simon_v2")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "resistance_results" / "dynamic_impedance_dc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--message_steps", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--forward_batch_size", type=int, default=16)
    parser.add_argument("--synthetic_sparse_samples", type=int, default=4000)
    parser.add_argument("--synthetic_max_high_cells", type=int, default=8)
    parser.add_argument("--synthetic_low_ohm", type=float, default=1.0)
    parser.add_argument("--synthetic_high_ohm", type=float, default=100.0)
    parser.add_argument("--classification_weight", type=float, default=0.25)
    parser.add_argument("--boundary_current_weight", type=float, default=0.05)
    parser.add_argument("--calibration_weight", type=float, default=0.1)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current = load_numeric_csv(args.data_dir / "I_train.csv")
    resistance = load_numeric_csv(args.data_dir / "R_train.csv")
    if current.shape != resistance.shape:
        raise ValueError("I_train.csv and R_train.csv must have matching shapes")
    rng = np.random.default_rng(args.seed)
    if args.max_train_samples is not None and args.max_train_samples < len(current):
        selected = rng.choice(len(current), args.max_train_samples, replace=False)
        current, resistance = current[selected], resistance[selected]
    order = rng.permutation(len(current))
    n_validation = max(1, int(round(len(order) * args.validation_fraction)))
    validation_index = order[:n_validation]
    train_index = order[n_validation:]
    i_train = torch.from_numpy(current[train_index])
    r_train = torch.from_numpy(resistance[train_index])
    i_validation = torch.from_numpy(current[validation_index])
    r_validation = torch.from_numpy(resistance[validation_index])

    grid_size = int(round(current.shape[1] ** 0.5))
    if grid_size * grid_size != current.shape[1]:
        raise ValueError("measurement count must be a square for the row-column protocol")
    graph = CircuitGraph.row_column_array(grid_size)
    protocol = pairwise_row_column_protocol(grid_size)
    r_test_path = args.data_dir / "R_test.csv"
    excluded_maps = load_numeric_csv(r_test_path) if r_test_path.exists() else None
    sparse_i, sparse_r = generate_sparse_curriculum(
        args.synthetic_sparse_samples,
        current.shape[1],
        graph,
        protocol,
        args.voltage,
        args.synthetic_low_ohm,
        args.synthetic_high_ohm,
        args.synthetic_max_high_cells,
        args.seed + 101,
        excluded_maps,
        device,
    )
    if len(sparse_i):
        sparse_order = torch.randperm(len(sparse_i), generator=torch.Generator().manual_seed(args.seed))
        n_sparse_validation = max(1, int(round(len(sparse_i) * args.validation_fraction)))
        sparse_validation = sparse_order[:n_sparse_validation]
        sparse_train = sparse_order[n_sparse_validation:]
        i_train = torch.cat([i_train, sparse_i[sparse_train]], dim=0)
        r_train = torch.cat([r_train, sparse_r[sparse_train]], dim=0)
        i_validation = torch.cat([i_validation, sparse_i[sparse_validation]], dim=0)
        r_validation = torch.cat([r_validation, sparse_r[sparse_validation]], dim=0)
    model = DynamicImpedanceGNN(
        graph,
        hidden_size=args.hidden_size,
        message_steps=args.message_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    weights = PhysicsLossWeights(
        classification=args.classification_weight,
        boundary_current=args.boundary_current_weight,
        calibration=args.calibration_weight,
    )
    loader = DataLoader(TensorDataset(i_train, r_train), batch_size=args.batch_size, shuffle=True)
    history = []
    best_score = float("inf")
    best_state = None
    print(f"Device: {device}; grid: {grid_size}x{grid_size}; train/val: {len(i_train)}/{len(i_validation)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, float] = {}
        for current_batch, resistance_batch in loader:
            current_batch = current_batch.to(device)
            resistance_batch = resistance_batch.to(device)
            measurement = MeasurementBatch.from_dc_currents(current_batch, protocol.to(device), voltage=args.voltage)
            target = PhysicalParameterBatch(resistance_batch)
            optimizer.zero_grad()
            loss, parts = total_inverse_loss(
                model,
                graph,
                measurement,
                target,
                weights=weights,
                high_threshold=args.threshold,
                boundary_sample_count=args.forward_batch_size,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value.detach())
        for key in running:
            running[key] /= len(loader)
        metrics = evaluate(
            model,
            i_validation,
            r_validation,
            protocol,
            args.batch_size,
            device,
            args.voltage,
            args.threshold,
        )
        row = {"epoch": epoch, **running, **{f"val_{key}": value for key, value in metrics.items()}}
        history.append(row)
        selection_score = metrics["log_rmse"] - 0.05 * metrics["high_f1"]
        if selection_score < best_score:
            best_score = selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} loss={running['total']:.4f} boundary={running['boundary_current']:.4f} "
            f"val_log_rmse={metrics['log_rmse']:.4f} val_high_f1={metrics['high_f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "grid_size": grid_size,
        "hidden_size": args.hidden_size,
        "message_steps": args.message_steps,
        "dropout": args.dropout,
        "voltage": args.voltage,
        "threshold": args.threshold,
        "mode": "dc",
        "frequency_ready": True,
        "rlc_trained": False,
        "loss_weights": asdict(weights),
        "training_arguments": vars(args),
        "best_selection_score": best_score,
    }
    torch.save(checkpoint, args.output_dir / "dynamic_impedance_dc_best.pt")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "model_status.json").write_text(
        json.dumps(
            {
                "trained_mode": "dc",
                "trained_parameters": ["resistance"],
                "inactive_untrained_parameters": ["inductance", "capacitance"],
                "physics_backend": "independent_incidence_matrix_dc_mna",
                "experimental_validation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "training_data_summary.json").write_text(
        json.dumps(
            {
                "original_samples_used": int(len(current)),
                "synthetic_sparse_samples": int(len(sparse_i)),
                "combined_train_samples": int(len(i_train)),
                "combined_validation_samples": int(len(i_validation)),
                "train_high_cell_fraction": float((r_train > args.threshold).float().mean()),
                "validation_high_cell_fraction": float((r_validation > args.threshold).float().mean()),
                "exact_test_maps_excluded_from_curriculum": excluded_maps is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.output_dir / 'dynamic_impedance_dc_best.pt'}")


if __name__ == "__main__":
    main()
