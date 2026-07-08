"""Local teaching server for the interactive ML website.

This server intentionally uses only the Python standard library. It serves the
static website and exposes one Server-Sent Events endpoint for realtime ECG
training logs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
WEBSITE_DIR = ROOT / "website"
RUNS_DIR = ROOT / "runs"

PYTHON_EXE = Path(
    r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\python3.10.exe"
)
PROJECT5_DIR = Path(r"C:\Users\10131\PycharmProjects\PythonProject5")
PROJECT5_SITE_PACKAGES = PROJECT5_DIR / ".venv310" / "Lib" / "site-packages"
ECG_TRAIN_SCRIPT = PROJECT5_DIR / "module2_ecg" / "ecg_train_torch.py"
ECG_TEST_SCRIPT = PROJECT5_DIR / "module2_ecg" / "ecg_test_torch.py"
ECG_DATASET = PROJECT5_DIR / "data" / "ecg" / "mitdb_beat_segments.npz"

PROJECT2_DIR = Path(r"C:\Users\10131\PycharmProjects\PythonProject2")
PROJECT2_SITE_PACKAGES = PROJECT2_DIR / ".venv" / "Lib" / "site-packages"
WFM_TRAIN_SCRIPT = PROJECT2_DIR / "train_torch.py"
WFM_TEST_SCRIPT = PROJECT2_DIR / "test_torch.py"
GENERATED_ASSETS_DIR = WEBSITE_DIR / "assets" / "generated"


def sse_payload(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def safe_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback


def safe_int(value: str, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return max(minimum, min(maximum, parsed))


class LearningLabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBSITE_DIR), **kwargs)

    def do_GET(self):  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if parsed.path == "/api/ecg/train":
            self.handle_ecg_train(parsed.query)
            return
        if parsed.path == "/api/wfm/train":
            self.handle_wfm_train(parsed.query)
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        super().do_GET()

    def send_json(self, data: dict[str, object]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def send_event(self, event: str, data: dict[str, object]) -> None:
        self.wfile.write(sse_payload(event, data))
        self.wfile.flush()

    def handle_ecg_train(self, query: str) -> None:
        params = parse_qs(query)
        model = safe_choice(params.get("model", ["cnn"])[0], {"cnn", "lstm", "cnn_lstm"}, "cnn")
        split_mode = safe_choice(params.get("split_mode", ["record"])[0], {"record", "random"}, "record")
        class_weighting = safe_choice(
            params.get("class_weighting", ["inverse"])[0], {"none", "inverse"}, "inverse"
        )
        epochs = safe_int(params.get("epochs", ["3"])[0], 1, 30, 3)

        run_id = datetime.now().strftime("ecg_%Y%m%d_%H%M%S")
        save_root = RUNS_DIR / run_id
        save_root.mkdir(parents=True, exist_ok=True)

        self.send_sse_headers()
        self.send_event(
            "status",
            {
                "message": "Starting realtime ECG training run.",
                "run_id": run_id,
                "model": model,
                "epochs": epochs,
                "split_mode": split_mode,
                "class_weighting": class_weighting,
            },
        )

        train_cmd = [
            str(PYTHON_EXE),
            str(ECG_TRAIN_SCRIPT),
            "--data_npz",
            str(ECG_DATASET),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch_size",
            "256",
            "--split_mode",
            split_mode,
            "--class_weighting",
            class_weighting,
            "--save_root",
            str(save_root),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT5_SITE_PACKAGES)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONWARNINGS"] = "ignore::FutureWarning"

        try:
            self.stream_process(train_cmd, cwd=PROJECT5_DIR, env=env)
            best_checkpoint = save_root / "checkpoints" / "best_model.pth"
            if best_checkpoint.exists():
                test_cmd = [
                    str(PYTHON_EXE),
                    str(ECG_TEST_SCRIPT),
                    "--data_npz",
                    str(ECG_DATASET),
                    "--checkpoint",
                    str(best_checkpoint),
                    "--model",
                    model,
                    "--split_mode",
                    split_mode,
                    "--save_root",
                    str(save_root),
                ]
                self.send_event("status", {"message": "Training complete. Running held-out test evaluation."})
                self.stream_process(test_cmd, cwd=PROJECT5_DIR, env=env)

            metrics_path = save_root / "test_metrics.json"
            metrics = None
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.send_event(
                "complete",
                {
                    "message": "Realtime ECG run complete.",
                    "run_id": run_id,
                    "save_root": str(save_root),
                    "metrics": metrics,
                },
            )
        except Exception as exc:  # keep browser stream readable for students
            self.send_event("error", {"message": str(exc)})

    def handle_wfm_train(self, query: str) -> None:
        params = parse_qs(query)
        epochs = safe_int(params.get("epochs", ["20"])[0], 1, 200, 20)
        l1_lambda = safe_int(params.get("l1_lambda", ["100"])[0], 0, 200, 100)
        test_count = safe_int(params.get("test_count", ["10"])[0], 1, 50, 10)

        run_id = datetime.now().strftime("wfm_%Y%m%d_%H%M%S")
        save_root_name = f"realtime_{run_id}"
        output_root = PROJECT2_DIR / f"w2f_1_{save_root_name}"
        GENERATED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        self.send_sse_headers()
        self.send_event(
            "status",
            {
                "message": "Starting realtime WFM GAN training run.",
                "run_id": run_id,
                "epochs": epochs,
                "l1_lambda": l1_lambda,
                "test_count": test_count,
            },
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT2_SITE_PACKAGES)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONWARNINGS"] = "ignore::FutureWarning"

        train_cmd = [
            str(PYTHON_EXE),
            str(WFM_TRAIN_SCRIPT),
            "--dataset",
            "w2f_1",
            "--all_subfolder",
            "train",
            "--train_epoch",
            str(epochs),
            "--batch_size",
            "1",
            "--L1_lambda",
            str(l1_lambda),
            "--test_count",
            str(test_count),
            "--split_seed",
            "42",
            "--save_root",
            save_root_name,
        ]

        try:
            self.stream_process(train_cmd, cwd=PROJECT2_DIR, env=env)
            test_cmd = [
                str(PYTHON_EXE),
                str(WFM_TEST_SCRIPT),
                "--dataset",
                "w2f_1",
                "--all_subfolder",
                "train",
                "--split_seed",
                "42",
                "--save_root",
                save_root_name,
                "--max_vis",
                "3",
            ]
            self.send_event("status", {"message": "Training complete. Generating held-out test images."})
            self.stream_process(test_cmd, cwd=PROJECT2_DIR, env=env)

            fixed_image = output_root / "Fixed_results" / f"w2f_1_{epochs}.png"
            test_dir = output_root / "Test_results_pairs"
            test_images = sorted(test_dir.glob("*.png")) if test_dir.exists() else []

            generated_fixed = GENERATED_ASSETS_DIR / f"{run_id}_fixed.png"
            generated_test = GENERATED_ASSETS_DIR / f"{run_id}_test.png"
            fixed_url = None
            test_url = None
            if fixed_image.exists():
                generated_fixed.write_bytes(fixed_image.read_bytes())
                fixed_url = f"assets/generated/{generated_fixed.name}"
            if test_images:
                generated_test.write_bytes(test_images[0].read_bytes())
                test_url = f"assets/generated/{generated_test.name}"

            self.send_event(
                "complete",
                {
                    "message": "Realtime WFM run complete.",
                    "run_id": run_id,
                    "output_root": str(output_root),
                    "fixed_image_url": fixed_url,
                    "test_image_url": test_url,
                },
            )
        except Exception as exc:
            self.send_event("error", {"message": str(exc)})

    def stream_process(self, command: list[str], cwd: Path, env: dict[str, str]) -> None:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.strip()
            if clean:
                self.send_event("log", {"line": clean, "time": time.time()})
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Training command failed with exit code {return_code}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local ML learning lab server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), LearningLabHandler)
    print(f"Serving ML Learning Lab at http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
