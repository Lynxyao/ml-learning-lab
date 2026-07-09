# -*- coding: utf-8 -*-
"""
Test script for paired image-to-image translation (instructional version)

Goal:
  - Use the held-out test set (e.g., 10 images) that was not used during training.
  - For each test sample, generate a 3-panel comparison image:
      [Input x] [Generated G(x)] [Ground truth y]
  - This matches the visualization format used in training Fixed_results.

Assumptions:
  - Each dataset image is a concatenation of two 256x256 halves (width=512):
      * left  half: target y (force grayscale image encoded as RGB)
      * right half: input  x (microscope/wrinkle image)
    when inverse_order=True (default in your training script).

Outputs:
  - results/<dataset>_<save_root>/Test_results_pairs/<filename>.png
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import util
from network_torch import GeneratorUNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='w2f_1')
parser.add_argument('--save_root', default='results')
parser.add_argument('--all_subfolder', default='train',
                    help='Folder containing ALL concatenated paired images (e.g., 252)')
parser.add_argument('--split_seed', type=int, default=42,
                    help='Seed used in training to create the split files')
parser.add_argument('--inverse_order', type=bool, default=True,
                    help='True: right half is input x, left half is target y')
parser.add_argument('--max_vis', type=int, default=0,
                    help='0 = visualize all test samples, otherwise visualize first N only')
opt = parser.parse_args()
print(opt)

root = f"{opt.dataset}_{opt.save_root}"

# Where training wrote the held-out file list
split_dir = os.path.join(root, "splits")
split_test_path = os.path.join(split_dir, f"split_seed_{opt.split_seed}_test.txt")

if not os.path.exists(split_test_path):
    raise FileNotFoundError(
        f"Cannot find test split file: {split_test_path}\n"
        f"Run train_torch.py first with the same --split_seed, so it writes splits/."
    )

# Output directory for 3-panel comparisons
out_dir = os.path.join(root, "Test_results_pairs")
os.makedirs(out_dir, exist_ok=True)

# Load test file list
with open(split_test_path, "r", encoding="utf-8") as f:
    test_files = [line.strip() for line in f if line.strip()]

if opt.max_vis and opt.max_vis > 0:
    test_files = test_files[:opt.max_vis]

print(f"Test samples to visualize: {len(test_files)}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = GeneratorUNet(in_channels=3, out_channels=3).to(device)

ckpt_path = os.path.join(root, "checkpoints", "G_latest.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Generator checkpoint not found: {ckpt_path}\n"
        f"Train first, or check your dataset/save_root."
    )

G.load_state_dict(torch.load(ckpt_path, map_location=device))
G.eval()
print("Loaded checkpoint:", ckpt_path)

# Helper: NHWC numpy -> NCHW torch
def to_tensor(x: np.ndarray) -> torch.Tensor:
    x = x.astype(np.float32)
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x)

# Optional: enforce 256x256
def resize_to_256(x_t: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x_t.shape
    if h != 256 or w != 256:
        x_t = F.interpolate(x_t, size=(256, 256), mode='bilinear', align_corners=False)
    return x_t

data_root = os.path.join("data", opt.dataset, opt.all_subfolder)

for i, fname in enumerate(test_files, start=1):
    img_path = os.path.join(data_root, fname)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"[WARN] Cannot read: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = img_rgb[None, ...]  # [1,H,W,3]

    H = img_rgb.shape[1]
    W = img_rgb.shape[2]

    # Determine the half width based on height (expected H=256, W=512)
    half = H
    if W < 2 * half:
        raise ValueError(
            f"Image width is too small for a concatenated pair: {fname} has W={W}, expected >= {2*half}."
        )

    # Split into x (input) and y (target)
    if opt.inverse_order:
        x_np = img_rgb[:, :, half:, :]     # right half
        y_np = img_rgb[:, :, 0:half, :]    # left half
    else:
        x_np = img_rgb[:, :, 0:half, :]
        y_np = img_rgb[:, :, half:, :]

    # Normalize to [-1,1] for network
    x_norm = util.norm(x_np)
    y_norm = util.norm(y_np)

    # Inference
    x_t = resize_to_256(to_tensor(x_norm).to(device))
    with torch.no_grad():
        pred = G(x_t).cpu().numpy()  # [1,3,256,256]

    pred = np.transpose(pred, (0, 2, 3, 1))  # back to NHWC
    pred = util.denorm(pred)                 # back to [0,255] for display

    # Save a 3-panel figure using the same visualization utility as training
    # util.show_result expects (x, generated, gt) in NHWC, and x/gt are usually normalized already.
    # To keep consistent with training Fixed_results, we pass normalized x and y, and raw pred in [-1,1] or [0,255]?
    # Here we pass x_norm and y_norm (normalized), and pred_norm (normalized) for consistent rendering.

    pred_norm = util.norm(pred)  # convert [0,255] back to [-1,1] so all 3 are in the same range

    out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_test.png")
    util.show_result(
        x_norm, pred_norm, y_norm,
        num_epoch=f"test #{i}",        # just an index label; not actual epoch
        save=True,
        path=out_path
    )

    print(f"[{i}/{len(test_files)}] saved:", out_path)

print("Done. Saved all 3-panel test comparisons to:", os.path.abspath(out_dir))
