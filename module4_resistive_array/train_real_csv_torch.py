from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resistance_models import MLPInverseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(r"C:\Users\10131\MATLAB Drive\module_4")
DEFAULT_OUTPUT = PROJECT_ROOT / "resistance_results/real_csv"


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.float32)


def standardize_train_test(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def split_train_val(x: np.ndarray, y: np.ndarray, seed: int, val_fraction: float = 0.15) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = int(round(len(x) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def transform_regression_target(r: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return r
    if transform == "log":
        return np.log(np.maximum(r, 1e-6))
    if transform == "conductance":
        return 1.0 / np.maximum(r, 1e-6)
    raise ValueError(transform)


def inverse_regression_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return y
    if transform == "log":
        return np.exp(y)
    if transform == "conductance":
        return 1.0 / np.maximum(y, 1e-6)
    raise ValueError(transform)


def inverse_regression_target_torch(
    y: torch.Tensor,
    transform: str,
    pred_min: float,
    pred_max: float,
) -> torch.Tensor:
    if transform == "raw":
        r = y
    elif transform == "log":
        r = torch.exp(y)
    elif transform == "conductance":
        r = 1.0 / torch.clamp(y, min=1e-6)
    else:
        raise ValueError(transform)
    return torch.clamp(r, min=pred_min, max=pred_max)


def fit_conductance_forward_model(r_train: np.ndarray, i_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a calibrated physics-inspired map from local conductance to currents.

    The idealized circuit relation is that fixed-voltage current is driven by
    conductance, 1/R. The real CSV data may include a simulator-specific scale
    or path definition, so we fit only a small linear calibration from the known
    training pairs. This gives the inverse model a forward-consistency check
    without hard-coding an incorrect circuit equation.
    """
    conductance = 1.0 / np.maximum(r_train, 1e-6)
    design = np.concatenate([conductance, np.ones((len(conductance), 1), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(design.astype(np.float64), i_train.astype(np.float64), rcond=None)
    weights = coef[:-1].astype(np.float32)
    bias = coef[-1].astype(np.float32)
    return weights, bias


def simon_forward_current_torch(
    r_flat: torch.Tensor,
    voltage: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable PyTorch version of Simon's MATLAB forward model.

    The resistor array is represented as a bipartite network between row
    terminals and column terminals. For each row-column terminal pair, the
    model applies [V, 0] on that pair while leaving all other terminals
    floating, then uses Schur reduction of the Laplacian to compute the source
    current. This mirrors main.m: build_L -> L_manipulate -> I_a(1).
    """
    if r_flat.ndim != 2:
        raise ValueError("r_flat must have shape [batch, n_cells]")
    batch_size, n_cells = r_flat.shape
    grid_size = int(round(n_cells**0.5))
    if grid_size * grid_size != n_cells:
        raise ValueError("number of cells must be a square")

    r = r_flat.reshape(batch_size, grid_size, grid_size)
    g = 1.0 / torch.clamp(r, min=eps)
    n_nodes = 2 * grid_size
    laplacian = torch.zeros(
        batch_size,
        n_nodes,
        n_nodes,
        dtype=r_flat.dtype,
        device=r_flat.device,
    )

    row_sum = g.sum(dim=2)
    col_sum = g.sum(dim=1)
    row_idx = torch.arange(grid_size, device=r_flat.device)
    col_idx = torch.arange(grid_size, device=r_flat.device)
    laplacian[:, row_idx, row_idx] = row_sum
    laplacian[:, grid_size + col_idx, grid_size + col_idx] = col_sum
    laplacian[:, :grid_size, grid_size:] = -g
    laplacian[:, grid_size:, :grid_size] = -g.transpose(1, 2)

    currents = []
    va = torch.tensor([voltage, 0.0], dtype=r_flat.dtype, device=r_flat.device)
    all_idx = torch.arange(n_nodes, device=r_flat.device)
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = torch.tensor([i, grid_size + j], dtype=torch.long, device=r_flat.device)
            beta = all_idx[(all_idx != alpha[0]) & (all_idx != alpha[1])]

            l_aa = laplacian[:, alpha][:, :, alpha]
            l_ab = laplacian[:, alpha][:, :, beta]
            l_ba = laplacian[:, beta][:, :, alpha]
            l_bb = laplacian[:, beta][:, :, beta]
            solved = torch.linalg.solve(l_bb, l_ba)
            l_red = l_aa - torch.bmm(l_ab, solved)
            ia = torch.matmul(l_red, va)
            currents.append(ia[:, 0])

    return torch.stack(currents, dim=1)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    loader = make_loader(x, np.zeros((len(x), 1), dtype=np.float32), batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        chunks.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def low_high_metrics(true_r: np.ndarray, pred_r: np.ndarray, threshold: float) -> dict[str, float]:
    true_low = true_r < threshold
    pred_low = pred_r < threshold
    correct = true_low == pred_low
    low_mask = true_low
    high_mask = ~true_low
    exact_map = correct.all(axis=1)
    return {
        "cell_low_high_accuracy": float(correct.mean()),
        "low_recall": float((pred_low[low_mask]).mean()) if low_mask.any() else 0.0,
        "high_recall": float((~pred_low[high_mask]).mean()) if high_mask.any() else 0.0,
        "high_precision": float((high_mask[~pred_low]).mean()) if (~pred_low).any() else 0.0,
        "exact_3x3_map_accuracy": float(exact_map.mean()),
    }


def probability_threshold_scan(
    true_r: np.ndarray,
    prob_positive: np.ndarray,
    resistance_threshold: float,
    positive_state: str,
) -> list[dict[str, float]]:
    rows = []
    for prob_threshold in np.linspace(0.05, 0.95, 19):
        if positive_state == "high":
            pred_r = np.where(prob_positive >= prob_threshold, 100.0, 1.0).astype(np.float32)
        else:
            pred_r = np.where(prob_positive >= prob_threshold, 1.0, 100.0).astype(np.float32)
        metrics = low_high_metrics(true_r, pred_r, resistance_threshold)
        rows.append({"probability_threshold": float(prob_threshold), **metrics})
    return rows


def save_threshold_scan(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def high_f1_from_maps(true_r: np.ndarray, pred_r: np.ndarray, threshold: float) -> float:
    true_high = true_r > threshold
    pred_high = pred_r > threshold
    tp = float((true_high & pred_high).sum())
    fp = float((~true_high & pred_high).sum())
    fn = float((true_high & ~pred_high).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def choose_probability_threshold(
    true_r: np.ndarray,
    prob_positive: np.ndarray,
    resistance_threshold: float,
    positive_state: str,
) -> tuple[float, list[dict[str, float]]]:
    rows = probability_threshold_scan(true_r, prob_positive, resistance_threshold, positive_state)
    best_threshold = 0.5
    best_score = -1.0
    for row in rows:
        prob_threshold = row["probability_threshold"]
        if positive_state == "high":
            pred_r = np.where(prob_positive >= prob_threshold, 100.0, 1.0).astype(np.float32)
        else:
            pred_r = np.where(prob_positive >= prob_threshold, 1.0, 100.0).astype(np.float32)
        score = high_f1_from_maps(true_r, pred_r, resistance_threshold)
        if score > best_score:
            best_score = score
            best_threshold = prob_threshold
    return best_threshold, rows


def regression_metrics(true_r: np.ndarray, pred_r: np.ndarray, threshold: float) -> dict[str, float]:
    out = {
        "mae_ohm": float(np.mean(np.abs(pred_r - true_r))),
        "rmse_ohm": float(np.sqrt(np.mean((pred_r - true_r) ** 2))),
    }
    out.update(low_high_metrics(true_r, pred_r, threshold))
    return out


def save_prediction_csv(path: Path, true_r: np.ndarray, pred_r: np.ndarray, currents: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_id", "cell_accuracy", "mae_ohm"]
        header += [f"true_r{i + 1}" for i in range(true_r.shape[1])]
        header += [f"pred_r{i + 1}" for i in range(pred_r.shape[1])]
        header += [f"current_{i + 1}" for i in range(currents.shape[1])]
        writer.writerow(header)
        for i in range(len(true_r)):
            true_low = true_r[i] < threshold
            pred_low = pred_r[i] < threshold
            cell_acc = float((true_low == pred_low).mean())
            mae = float(np.mean(np.abs(pred_r[i] - true_r[i])))
            writer.writerow([i, cell_acc, mae, *true_r[i], *pred_r[i], *currents[i]])


def save_prediction_figure(path: Path, true_r: np.ndarray, pred_r: np.ndarray, n_examples: int = 8) -> None:
    n = min(n_examples, len(true_r))
    fig, axes = plt.subplots(n, 3, figsize=(8.2, 2.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    value_vmin = float(min(true_r[:n].min(), pred_r[:n].min()))
    value_vmax = float(max(true_r[:n].max(), pred_r[:n].max()))
    error_vmax = float(np.abs(pred_r[:n] - true_r[:n]).max())
    for i in range(n):
        true_map = true_r[i].reshape(3, 3)
        pred_map = pred_r[i].reshape(3, 3)
        err_map = np.abs(pred_map - true_map)
        panels = [(true_map, "True R"), (pred_map, "Predicted R"), (err_map, "Absolute error")]
        for j, (arr, title) in enumerate(panels):
            if j < 2:
                im = axes[i, j].imshow(arr, cmap="viridis", vmin=value_vmin, vmax=value_vmax)
            else:
                im = axes[i, j].imshow(arr, cmap="magma", vmin=0.0, vmax=error_vmax)
            axes[i, j].set_title(f"{title}\nSample {i}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for row in range(3):
                for col in range(3):
                    axes[i, j].text(
                        col,
                        row,
                        f"{arr[row, col]:.0f}",
                        ha="center",
                        va="center",
                        color="white" if arr[row, col] > (arr.max() + arr.min()) / 2 else "black",
                        fontsize=7,
                    )
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    loss_fn: nn.Module,
    y_mean: np.ndarray | None = None,
    y_std: np.ndarray | None = None,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device) if y_mean is not None else None
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device) if y_std is not None else None
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device) if x_mean is not None else None
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device) if x_std is not None else None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        target_losses = []
        current_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            target_loss = loss_fn(pred, yb)
            current_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if (
                args.task == "regression"
                and args.physics_loss_weight > 0
                and y_mean_t is not None
                and y_std_t is not None
                and x_mean_t is not None
                and x_std_t is not None
            ):
                pred_trans = pred * y_std_t + y_mean_t
                pred_r = inverse_regression_target_torch(
                    pred_trans,
                    args.target_transform,
                    pred_min=args.prediction_min,
                    pred_max=args.prediction_max,
                )
                pred_current_raw = simon_forward_current_torch(pred_r, voltage=args.voltage)
                pred_current_scaled = (pred_current_raw - x_mean_t) / x_std_t
                current_loss = loss_fn(pred_current_scaled, xb)
            loss = target_loss + args.physics_loss_weight * current_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            target_losses.append(float(target_loss.item()))
            current_losses.append(float(current_loss.item()))
        val_pred = predict(model, x_val, args.batch_size, device)
        val_loss = float(loss_fn(torch.tensor(val_pred, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)).item())
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "target_loss": float(np.mean(target_losses)),
            "current_loss": float(np.mean(current_losses)),
            "val_loss": val_loss,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={row['train_loss']:.4f} "
                f"target={row['target_loss']:.4f} "
                f"physics={row['current_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f}"
            )
    return history


def real_csv_mode_name(args: argparse.Namespace) -> str:
    if args.task == "classification":
        return f"{args.task}_{args.positive_state}"
    if args.physics_loss_weight > 0:
        weight_tag = str(args.physics_loss_weight).replace(".", "p")
        return f"regression_{args.target_transform}_physics_w{weight_tag}"
    return f"regression_{args.target_transform}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Module 4 inverse models on Simon's real CSV training files.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--positive_state", choices=["high", "low"], default="high")
    parser.add_argument("--target_transform", choices=["raw", "log", "conductance"], default="conductance")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--physics_loss_weight",
        type=float,
        default=0.1,
        help="Add Simon forward-current consistency loss for regression. Use 0 to disable.",
    )
    parser.add_argument("--prediction_min", type=float, default=1.0)
    parser.add_argument("--prediction_max", type=float, default=110.0)
    parser.add_argument(
        "--voltage",
        type=float,
        default=5.0,
        help="Fixed voltage used by Simon's forward circuit model.",
    )
    args = parser.parse_args()

    i_train = load_csv(args.data_dir / "I_train.csv")
    r_train = load_csv(args.data_dir / "R_train.csv")
    x_train_raw, y_train_raw, x_val_raw, y_val_raw = split_train_val(i_train, r_train, args.seed)
    x_train, x_val, _, x_mean, x_std = standardize_train_test(x_train_raw, x_val_raw, x_val_raw)

    y_mean = None
    y_std = None

    if args.task == "classification":
        if args.positive_state == "high":
            y_train = (y_train_raw > args.threshold).astype(np.float32)
            y_val = (y_val_raw > args.threshold).astype(np.float32)
        else:
            y_train = (y_train_raw < args.threshold).astype(np.float32)
            y_val = (y_val_raw < args.threshold).astype(np.float32)
        positive = max(float(y_train.sum()), 1.0)
        negative = max(float(y_train.size - y_train.sum()), 1.0)
        pos_weight = torch.full((r_train.shape[1],), negative / positive)
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        y_train_trans = transform_regression_target(y_train_raw, args.target_transform)
        y_val_trans = transform_regression_target(y_val_raw, args.target_transform)
        y_mean = y_train_trans.mean(axis=0, keepdims=True)
        y_std = y_train_trans.std(axis=0, keepdims=True) + 1e-6
        y_train = (y_train_trans - y_mean) / y_std
        y_val = (y_val_trans - y_mean) / y_std
        loss_fn = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPInverseModel(input_dim=x_train.shape[1], output_dim=r_train.shape[1], hidden_size=args.hidden_size).to(device)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    history = train_model(
        model,
        train_loader,
        x_val,
        y_val,
        args,
        device,
        loss_fn,
        y_mean=y_mean,
        y_std=y_std,
        x_mean=x_mean,
        x_std=x_std,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_name = real_csv_mode_name(args)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{mode_name}_model.pt"

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": x_train.shape[1],
        "output_dim": r_train.shape[1],
        "hidden_size": args.hidden_size,
        "task": args.task,
        "positive_state": args.positive_state,
        "target_transform": args.target_transform,
        "threshold": args.threshold,
        "physics_loss_weight": args.physics_loss_weight,
        "prediction_min": args.prediction_min,
        "prediction_max": args.prediction_max,
        "voltage": args.voltage,
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    if args.task == "classification":
        val_prob_positive = 1.0 / (1.0 + np.exp(-predict(model, x_val, args.batch_size, device)))
        threshold_to_use, val_scan_rows = choose_probability_threshold(
            y_val_raw,
            val_prob_positive,
            args.threshold,
            args.positive_state,
        )
        checkpoint["probability_threshold"] = threshold_to_use
        save_threshold_scan(args.output_dir / f"{mode_name}_validation_threshold_scan.csv", val_scan_rows)

    torch.save(checkpoint, checkpoint_path)
    (args.output_dir / f"{mode_name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nData summary")
    print(f"I_train shape={i_train.shape}, range=({i_train.min():.4f}, {i_train.max():.4f})")
    print(f"R_train shape={r_train.shape}, range=({r_train.min():.4f}, {r_train.max():.4f})")
    print(f"\nSaved training history to {args.output_dir / f'{mode_name}_history.json'}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
