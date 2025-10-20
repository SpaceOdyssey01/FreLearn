#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coder_planC.py — Autoencoder pretraining on read.py NPZ shards (Plan C)
- Reads shards exported by read.py: segments [N,C,H,W], labels [N]
- Per-sample, per-channel min-max normalization to [0,1]
- Loss: (1 - alpha)*MSE + alpha*(1 - SSIM)
- Saves: ae_best.pth, ae_last.pth, encoder.pth
"""

import os
import glob
import math
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


# -------------------- Utils --------------------
def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def minmax_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-channel min-max normalization to [0,1].
    x: [B, C, H, W]
    """
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    denom = (x_max - x_min).clamp_min(eps)
    x01 = (x - x_min) / denom
    return x01.clamp_(0.0, 1.0)


# -------------------- SSIM --------------------
def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)],
        dtype=torch.float32
    )
    gauss = gauss / gauss.sum()
    return gauss.unsqueeze(1)  # [W,1]


def create_ssim_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    _1D = _gaussian_window(window_size, sigma=1.5).to(device)
    _2D = _1D @ _1D.t()  # [W,W]
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index as a loss component (returns 1 - SSIM).
    Assumes inputs in [0,1], so data_range=1.0.
    """
    def __init__(self, window_size: int = 11, channel: int = 1, data_range: float = 1.0):
        super().__init__()
        assert window_size % 2 == 1, "window_size should be odd."
        self.window_size = window_size
        self.channel = channel
        self.data_range = float(data_range)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y: [B,C,H,W], assumed in [0,1]
        B, C, H, W = x.shape
        window = create_ssim_window(self.window_size, C, x.device)
        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, groups=C, padding=padding)
        mu_y = F.conv2d(y, window, groups=C, padding=padding)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, groups=C, padding=padding) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, groups=C, padding=padding) - mu_y2
        sigma_xy = F.conv2d(x * y, window, groups=C, padding=padding) - mu_xy

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
                   (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return 1.0 - ssim_map.mean()  # 1 - SSIM


class MSESSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIMLoss(window_size=window_size, data_range=1.0)

    def forward(self, pred01: torch.Tensor, target01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          total_loss: (1 - alpha) * MSE + alpha * (1 - SSIM)
          mse:        MSE
          ssim_index: SSIM index itself (for logging)
        """
        mse = F.mse_loss(pred01, target01)
        ssim_loss = self.ssim(pred01, target01)   # 这里是 1 - SSIM
        ssim_index = 1.0 - ssim_loss              # 还原成 SSIM 指数
        total = (1 - self.alpha) * mse + self.alpha * ssim_loss
        return total, mse, ssim_index


# -------------------- Model --------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.se = SEBlock(ch)

    def forward(self, x):
        out = self.block(x) + x
        return F.relu(self.se(out), inplace=True)


class Autoencoder2D(nn.Module):
    """
    No downsampling encoder; final activation is sigmoid (outputs in [0,1]).
    """
    def __init__(self, in_ch=1, latent_ch=128):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            ResBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            ResBlock(128),
            nn.Conv2d(128, latent_ch, 3, padding=1),
            nn.BatchNorm2d(latent_ch), nn.GELU(),
            ResBlock(latent_ch),
        )
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(latent_ch + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            ResBlock(128),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            ResBlock(64),
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, in_ch, 3, padding=1)
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        z  = self.enc3(e2)
        return z, e1, e2

    def forward(self, x01):
        """
        Expects x01 in [0,1]; returns reconstruction in [0,1] (sigmoid).
        """
        z, e1, e2 = self.encode(x01)
        d1 = torch.cat([z, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = torch.cat([d1, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return torch.sigmoid(out)


# -------------------- NPZ Shards Dataset (compatible with read.py output) --------------------
class GFCCNPZIterableDataset(IterableDataset):
    """
    Iterate over *.npz shards exported by read.py.
      - 'segments': [N,C,H,W]  (float32)
      - 'labels':   [N]        (int64)  # AE 不用标签，但可以读出来
    Yields (x, y) per batch.
    """
    def __init__(self, shards_dir: str, batch_size: int = 8):
        super().__init__()
        self.shards = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
        if not self.shards:
            raise FileNotFoundError(f"No .npz shards under: {shards_dir}")
        self.batch_size = int(batch_size)

    def __iter__(self):
        for f in self.shards:
            with np.load(f) as npz:
                segs = npz["segments"]   # [N,C,H,W], float32
                labs = npz["labels"]     # [N], int64
            n = segs.shape[0]
            for st in range(0, n, self.batch_size):
                ed = min(st + self.batch_size, n)
                x = torch.from_numpy(segs[st:ed])        # float32
                y = torch.from_numpy(labs[st:ed]).long() # int64 (unused)
                yield x, y


def build_loader_for_dir(shards_dir: str, batch_size: int, num_workers: int = 0):
    ds = GFCCNPZIterableDataset(shards_dir, batch_size=batch_size)
    return DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True)


# -------------------- Training --------------------
def train_one_epoch(model, loader, opt, device, loss_fn, max_norm: float | None = 1.0):
    model.train()
    sum_total = sum_mse = sum_ssim = 0.0
    n = 0
    for x, _ in loader:
        x = to_device(x, device).float()
        x01 = minmax_unit(x)          # normalize inputs to [0,1]
        pred = model(x01)             # pred in [0,1]

        total, mse, ssim = loss_fn(pred, x01)

        opt.zero_grad(set_to_none=True)
        total.backward()
        if max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()

        bs = x.size(0)
        sum_total += total.item() * bs
        sum_mse   += mse.item()   * bs
        sum_ssim  += ssim.item()  * bs
        n += bs
    return (sum_total / max(1, n),
            sum_mse   / max(1, n),
            sum_ssim  / max(1, n))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    sum_total = sum_mse = sum_ssim = 0.0
    n = 0
    for x, _ in loader:
        x = to_device(x, device).float()
        x01 = minmax_unit(x)
        pred = model(x01)

        total, mse, ssim = loss_fn(pred, x01)

        bs = x.size(0)
        sum_total += total.item() * bs
        sum_mse   += mse.item()   * bs
        sum_ssim  += ssim.item()  * bs
        n += bs
    return (sum_total / max(1, n),
            sum_mse   / max(1, n),
            sum_ssim  / max(1, n))


# -------------------- Main --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # shards root exported by read.py, contains foldK_overlap/foldK_nonoverlap/shards/*.npz
    ap.add_argument("--slices_root", type=str, default="./slices")
    ap.add_argument("--fold", type=int, default=5, help="1..5 used as validation fold")
    ap.add_argument("--mode_train", type=str, default="overlap", choices=["overlap", "nonoverlap"])
    ap.add_argument("--mode_val", type=str, default="nonoverlap", choices=["overlap", "nonoverlap"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    # model & optim
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--latent_ch", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of SSIM in MSE+SSIM")
    ap.add_argument("--compute_device", type=str, default="cuda", choices=["cpu", "cuda"])

    # saves
    ap.add_argument("--save_dir", type=str, default="./ae_ckpts")
    ap.add_argument("--encoder_out", type=str, default="encoder.pth")
    ap.add_argument("--ae_best_out", type=str, default="ae_best.pth")
    ap.add_argument("--ae_last_out", type=str, default="ae_last.pth")
    return ap.parse_args()


def _dir_of(slices_root: str, k: int, mode: str) -> str:
    return os.path.join(slices_root, f"fold{k}_{mode}", "shards")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        args.compute_device if torch.cuda.is_available() and args.compute_device == "cuda" else "cpu"
    )

    # ===== build train/val loaders from read.py shards =====
    # train = union of folds except the chosen val fold, using args.mode_train
    train_dirs: List[str] = []
    for k in range(1, 6):
        if k == args.fold:
            continue
        d = _dir_of(args.slices_root, k, args.mode_train)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npz")):
            train_dirs.append(d)
    if not train_dirs:
        raise FileNotFoundError(f"No train shards found under {args.slices_root} with mode={args.mode_train}")

    train_loaders = [build_loader_for_dir(d, batch_size=args.batch_size, num_workers=args.num_workers)
                     for d in train_dirs]

    # val = chosen fold with args.mode_val
    val_dir = _dir_of(args.slices_root, args.fold, args.mode_val)
    if not (os.path.isdir(val_dir) and glob.glob(os.path.join(val_dir, "*.npz"))):
        raise FileNotFoundError(f"No val shards found: {val_dir}")
    val_loader = build_loader_for_dir(val_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # ===== Model & Opt =====
    model = Autoencoder2D(in_ch=args.in_ch, latent_ch=args.latent_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = MSESSIMLoss(alpha=args.alpha, window_size=11)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # train over all train_loaders (each is a directory of shards)
        tr_total = tr_mse = tr_ssim = 0.0
        tr_n = 0
        for loader in train_loaders:
            a, b, c = train_one_epoch(model, loader, opt, device, loss_fn)
            tr_total += a; tr_mse += b; tr_ssim += c; tr_n += 1
        if tr_n > 0:
            tr_total /= tr_n; tr_mse /= tr_n; tr_ssim /= tr_n

        va_total, va_mse, va_ssim = evaluate(model, val_loader, device, loss_fn)

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"Train — Total: {tr_total:.6f}, MSE: {tr_mse:.6f}, SSIM: {tr_ssim:.6f} | "
              f"Val — Total: {va_total:.6f}, MSE: {va_mse:.6f}, SSIM: {va_ssim:.6f}")

        # Save last
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch},
                   os.path.join(args.save_dir, args.ae_last_out))

        # Save best & encoder weights
        if va_total < best_val:
            best_val = va_total
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_total": va_total,
                        "val_mse": va_mse,
                        "val_ssim": va_ssim},
                       os.path.join(args.save_dir, args.ae_best_out))
            enc_state = {
                "enc1": model.enc1.state_dict(),
                "enc2": model.enc2.state_dict(),
                "enc3": model.enc3.state_dict(),
            }
            enc_path = os.path.join(args.save_dir, args.encoder_out)
            torch.save(enc_state, enc_path)
            print(f"  -> [BEST] Val Total improved to {best_val:.6f}. Saved encoder to {enc_path}")

    print("[DONE] Pretraining complete.")
    print(f"Best Val Total: {best_val:.6f}")
    print(f"Artifacts in: {args.save_dir}")


if __name__ == "__main__":
    main() 