# -*- coding: utf-8 -*-
"""
PyTorch training script (instructional version)

Purpose:
  - Read concatenated images from data/<dataset>/<all_subfolder>
  - Right half is input x (microscope/wrinkle image)
  - Left half is target y (force grayscale image)
  - Train a pix2pix-style GAN (U-Net + PatchGAN)

Instructional modification:
  - Deterministically split the full dataset into:
      * train set (~240 images)
      * test set (~10–12 images)
  - This script trains ONLY on the training subset.
  - The split is reproducible via --split_seed.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import util
from network_torch import GeneratorUNet, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='w2f_1', help='Dataset subdirectory name')
parser.add_argument('--all_subfolder', default='train', help='Folder that contains ALL images (e.g., 252)')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--lrD', type=float, default=0.0002)
parser.add_argument('--lrG', type=float, default=0.0002)
parser.add_argument('--L1_lambda', type=float, default=100.0)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--save_root', default='results', help='Output directory')
parser.add_argument('--inverse_order', type=bool, default=True,
                    help='True: right half is input, left half is target')
# --- split controls ---
parser.add_argument('--test_count', type=int, default=10, help='Number of test images to hold out')
parser.add_argument('--split_seed', type=int, default=42, help='Seed for reproducible train/test split')
opt = parser.parse_args()
print(opt)

# Output folders
root = f"{opt.dataset}_{opt.save_root}"
model_prefix = opt.dataset + '_'
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, 'Fixed_results'), exist_ok=True)
os.makedirs(os.path.join(root, 'checkpoints'), exist_ok=True)

# Load ALL images from data/<dataset>/<all_subfolder>
all_root = os.path.join('data', opt.dataset, opt.all_subfolder)
train_loader = util.data_loader(all_root, batch_size=opt.batch_size, shuffle=False)
print("Loaded all images. loader shape:", train_loader.shape)

img_h = train_loader.shape[1]  # height (should be 256)
img_w = train_loader.shape[2]  # width  (should be 512 for concatenated pairs)

# -----------------------------
# Deterministic train/test split
# -----------------------------
rng = np.random.default_rng(opt.split_seed)

all_files = list(train_loader.file_list)
all_files = sorted(all_files)  # stable ordering before permutation
N = len(all_files)

if opt.test_count >= N:
    raise ValueError(f"--test_count must be < total images. Got test_count={opt.test_count}, total={N}.")

perm = rng.permutation(N)
test_idx = perm[:opt.test_count]
train_idx = perm[opt.test_count:]

train_files = [all_files[i] for i in train_idx]
test_files = [all_files[i] for i in test_idx]  # for reference/logging only

# Overwrite loader to use ONLY training files
train_loader.file_list = train_files
train_loader.shape = (len(train_files),) + tuple(train_loader.shape[1:])
train_loader.flag = 0

print(f"Total images: {N}")
print(f"Train images: {len(train_files)}")
print(f"Test images:  {len(test_files)}")
print("Example test files:", test_files[:5])

# Save the split lists for transparency/reproducibility (optional but useful for teaching)
split_dir = os.path.join(root, "splits")
os.makedirs(split_dir, exist_ok=True)
with open(os.path.join(split_dir, f"split_seed_{opt.split_seed}_test.txt"), "w", encoding="utf-8") as f:
    for p in test_files:
        f.write(p + "\n")
with open(os.path.join(split_dir, f"split_seed_{opt.split_seed}_train.txt"), "w", encoding="utf-8") as f:
    for p in train_files:
        f.write(p + "\n")

# Prepare a fixed batch for consistent visualization across epochs
fixed_batch = train_loader.next_batch()
if opt.inverse_order:
    fixed_x_ = fixed_batch[:, :, img_h:, :]      # right half
    fixed_y_ = fixed_batch[:, :, 0:img_h, :]     # left half
else:
    fixed_x_ = fixed_batch[:, :, 0:img_h, :]
    fixed_y_ = fixed_batch[:, :, img_h:, :]

fixed_x_ = util.norm(fixed_x_)  # [-1, 1]
fixed_y_ = util.norm(fixed_y_)

# Convert NHWC (numpy) -> NCHW (torch)
def to_tensor(x: np.ndarray) -> torch.Tensor:
    x = x.astype(np.float32)
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

G = GeneratorUNet(in_channels=3, out_channels=3).to(device)
D = Discriminator(in_channels=3).to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
optimizer_D = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

print("Start training...")
total_start = time.time()

for epoch in range(opt.train_epoch):
    epoch_start = time.time()
    D_losses, G_losses = [], []

    # Reproducible shuffling per epoch (teaching-friendly)
    epoch_rng = np.random.default_rng(opt.split_seed + epoch + 1)
    epoch_rng.shuffle(train_loader.file_list)
    train_loader.flag = 0  # restart iteration from beginning after shuffle

    num_iter = train_loader.shape[0] // opt.batch_size
    for it in range(num_iter):
        train_img = train_loader.next_batch()  # numpy, [B,H,W,C]

        # Split concatenated image into (x, y)
        if opt.inverse_order:
            x_ = train_img[:, :, img_h:, :]
            y_ = train_img[:, :, 0:img_h, :]
        else:
            x_ = train_img[:, :, 0:img_h, :]
            y_ = train_img[:, :, img_h:, :]

        x_ = util.norm(x_)
        y_ = util.norm(y_)

        x_t = to_tensor(x_).to(device)
        y_t = to_tensor(y_).to(device)

        # -------------------------
        # 1) Update Discriminator D
        # -------------------------
        optimizer_D.zero_grad()

        fake_y = G(x_t).detach()
        pred_real = D(x_t, y_t)
        pred_fake = D(x_t, fake_y)

        valid = torch.ones_like(pred_real, device=device)
        fake = torch.zeros_like(pred_fake, device=device)

        loss_D_real = criterion_GAN(pred_real, valid)
        loss_D_fake = criterion_GAN(pred_fake, fake)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        optimizer_D.step()

        # -------------------------
        # 2) Update Generator G
        # -------------------------
        optimizer_G.zero_grad()

        fake_y = G(x_t)
        pred_fake_for_G = D(x_t, fake_y)

        loss_G_GAN = criterion_GAN(pred_fake_for_G, valid)
        loss_G_L1 = criterion_L1(fake_y, y_t) * opt.L1_lambda
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())

        if (it + 1) % 10 == 0 or it == num_iter - 1:
            print(f"[Epoch {epoch+1}/{opt.train_epoch}] "
                  f"[Iter {it+1}/{num_iter}] "
                  f"D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}")

    epoch_time = time.time() - epoch_start
    print(f"==> Epoch {epoch+1} finished, time: {epoch_time:.2f}s, "
          f"D_loss: {np.mean(D_losses):.4f}, G_loss: {np.mean(G_losses):.4f}")

    # Save visualization results (Fixed_results)
    G.eval()
    with torch.no_grad():
        fx_t = to_tensor(fixed_x_).to(device)
        out_fixed = G(fx_t).cpu().numpy()  # [B,3,H,W]
        out_fixed = np.transpose(out_fixed, (0, 2, 3, 1))  # back to NHWC

        util.show_result(
            fixed_x_, out_fixed, fixed_y_,
            num_epoch=epoch + 1,
            save=True,
            path=os.path.join(root, 'Fixed_results', f"{model_prefix}{epoch+1}.png")
        )
    G.train()

    # Save checkpoints
    torch.save(G.state_dict(), os.path.join(root, 'checkpoints', 'G_latest.pth'))
    torch.save(D.state_dict(), os.path.join(root, 'checkpoints', 'D_latest.pth'))

total_time = time.time() - total_start
print(f"Training done! total time: {total_time/60:.1f} min")
