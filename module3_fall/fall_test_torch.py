from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fall_dataset import classification_metrics, load_npz, split_and_scale, split_indices
from fall_models import FallRNN
from fall_prediction_report import generate_prediction_report
from fall_train_torch import predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "fall_results/checkpoints/best_model.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "fall_results/prediction_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved fall prediction RNN.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_npz", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max_report_examples", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_npz = args.data_npz or Path(checkpoint["data_npz"])
    if not data_npz.is_absolute():
        data_npz = PROJECT_ROOT / data_npz

    x, y, groups = load_npz(data_npz)
    data = split_and_scale(x, y, groups=groups)
    _, _, test_idx = split_indices(x, y, groups)

    model = FallRNN(model=checkpoint["model_type"], hidden_size=int(checkpoint["hidden_size"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    probs, pred = predict(model, data.x_test, args.batch_size, device)
    metrics = classification_metrics(data.y_test, pred)

    out_path = args.checkpoint.parent.parent / "test_metrics_rerun.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(args.checkpoint.parent.parent / "test_predictions_rerun.npz", probs=probs, pred=pred, y_true=data.y_test)

    raw = np.load(data_npz, allow_pickle=True)
    subject_id = raw["subject_id"].astype(np.int64) if "subject_id" in raw.files else None
    activity_id = raw["activity_id"].astype(np.int64) if "activity_id" in raw.files else None
    report_outputs = generate_prediction_report(
        args.report_dir,
        x[test_idx],
        data.y_test,
        probs,
        pred,
        test_idx,
        subject_id,
        activity_id,
        args.max_report_examples,
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved rerun metrics to {out_path}")
    for output in report_outputs:
        print(f"Saved report output to {output}")


if __name__ == "__main__":
    main()
