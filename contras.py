# minimal_train.py — 三阶段：Stage-1（WT→投影→对比→融合→调制→iDWT→残差；按 STAGE1_SCALE 缩放）
#                    → Stage-M（同流程；按 STAGEM_SCALE 缩放）
#                    → Stage-2（再WT→分类）
import os, math, random, csv, io, sys, time, bisect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torch.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pytorch_metric_learning.losses import NTXentLoss
import argparse
from pytorch_wavelets import DWTForward, DWTInverse

# ================== 超参 ==================
FOLD = 5
DROPOUT = 0.5
NOISE = 0
EPOCHS = 150
HEAD_LR = 6e-4
BACKBONE_LR = 1.5e-4
WEIGHT_DECAY = 1e-5
BALANCE_LOSS_W = 1e-3
PATIENCE = 80
EMA_DECAY = 0.997
STAGE1_SCALE = 0.9  # Stage-1 分数缩放（1.0 表示不缩）
STAGEM_SCALE = 0.8  # Stage-M 分数缩放

PROB_BAL_W = 1e-2
GATE_REG_W = 1e-3
MIXUP_ALPHA = 0.4
MIXUP_STOP_FR = 0.4
SUBBAND_DROP_P = 0.05

HIGH = 0.35
LOW = 0.25
TARGET_HIGH = 1.1
TARGET_LOW = 1.02
SMOOTH = 0.5


# ================== AE utils & model ==================
def minmax_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    denom = (x_max - x_min).clamp_min(eps)
    return ((x - x_min) / denom).clamp_(0.0, 1.0)


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
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
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
    def __init__(self, in_ch=3, latent_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), ResBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), ResBlock(128),
            nn.Conv2d(128, latent_ch, 3, padding=1), nn.BatchNorm2d(latent_ch), nn.GELU(), ResBlock(latent_ch),
        )
        # 解码器保留但本脚本不使用
        self.dec1 = nn.Sequential(
            nn.Conv2d(latent_ch + 64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), ResBlock(128),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), ResBlock(64),
        )
        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, in_ch, 3, padding=1))

    @torch.no_grad()
    def encode(self, x01: torch.Tensor):
        e1 = self.enc1(x01);
        e2 = self.enc2(e1);
        z = self.enc3(e2)
        return z, e1, e2


class Resize2D(nn.Module):
    """任意比例/目标尺寸缩放；下采样建议使用 area（等价于平均池化）。"""

    def __init__(self, mode: str = "area", antialias: bool = True):
        super().__init__()
        self.mode = mode
        self.antialias = antialias

    def forward(self, x: torch.Tensor, scale: float | None = None, size: tuple[int, int] | None = None):
        assert (scale is None) ^ (size is None), "Provide exactly one of scale or size."
        if scale is not None:
            H, W = x.shape[-2:]
            size = (max(1, int(H * scale)), max(1, int(W * scale)))
        if self.mode == "area":
            return F.interpolate(x, size=size, mode="area")
        else:
            return F.interpolate(x, size=size, mode=self.mode, align_corners=False, antialias=self.antialias)


# ==== 小波变换 ====
class WaveletTransform(nn.Module):
    def __init__(self, wave: str = 'coif1', device: str = 'cpu'):
        super().__init__()
        self.wave_name = wave
        self._device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.dwt1 = DWTForward(J=1, wave=wave).to(self._device).eval()
        self.idwt1 = DWTInverse(wave=wave).to(self._device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = x.to(self._device).float()
        yl1, yh1_list = self.dwt1(x)  # yl1: [B,C,H/2,W/2], yh1_list[0]: [B,C,3,H/2,W/2]
        yh1 = yh1_list[0]
        return yl1, yh1[:, :, 0], yh1[:, :, 1], yh1[:, :, 2]

    @torch.no_grad()
    def inverse(self, low: torch.Tensor, hh: torch.Tensor, hv: torch.Tensor, hd: torch.Tensor):
        yl = low.to(self._device).float()
        yh = torch.stack([hh, hv, hd], dim=2).to(self._device).float()  # [B,C,3,H/2,W/2]
        x = self.idwt1((yl, [yh]))
        return x


@torch.no_grad()
def _pad_to_even(x: torch.Tensor):
    _, _, H, W = x.shape
    ph = 1 if (H % 2 == 1) else 0
    pw = 1 if (W % 2 == 1) else 0
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, ph, pw


# ================== 工具：AUG/阈值相关 ==================
def set_aug_strength_for_train_dataset(ds, s: float):
    from torch.utils.data import ConcatDataset as _Concat
    if isinstance(ds, _Concat):
        for sub in ds.datasets:
            if hasattr(sub, "set_strength"): sub.set_strength(s)
    else:
        if hasattr(ds, "set_strength"): ds.set_strength(s)


def set_class_strength_for_train_dataset(ds, mapping: dict):
    from torch.utils.data import ConcatDataset as _Concat
    if isinstance(ds, _Concat):
        for sub in ds.datasets:
            if hasattr(sub, "set_class_strength"): sub.set_class_strength(mapping)
    else:
        if hasattr(ds, "set_class_strength"): ds.set_class_strength(mapping)


def _batch_fnr(probs_pos: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    with torch.no_grad():
        pred = (probs_pos >= thr).long();
        y = labels.long()
        tp = ((pred == 1) & (y == 1)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        den = tp + fn
        if den == 0: return float('nan')
        return fn / float(den)


def _map_fnr_to_press(fnr_ema: float, low: float, high: float, tl: float, th: float) -> float:
    if not (fnr_ema == fnr_ema): return tl
    if high <= low: return tl
    t = (fnr_ema - low) / (high - low);
    t = min(1.0, max(0.0, t))
    return tl * (1.0 - t) + th * t


def aug_strength_schedule(epoch: int, max_epochs: int, s_max=0.8, s_min=0.3):
    t = epoch / max(1, max_epochs - 1)
    return s_min + 0.5 * (s_max - s_min) * (1.0 + math.cos(math.pi * t))


# ================== 数据集 ==================
class EEGFeatureDataset(Dataset):
    """
    源：
      1) 目录 .pt 分片（旧） -> 直接读取 (low/high*/labels)
      2) 单个 .pt 文件（旧）
      3) 目录 .npz 分片（新） -> AE.encode 得到 z，再在线 DWT 得到 (low/high*)；并返回 pad 后 raw
    """

    def __init__(self, feature_path, augment=False,
                 noise_factor=0.1, channel_mask_pct=0, spatial_mask_pct=0,
                 cache_shards=2, dtype=None,
                 wave_name: str = "coif1", wt_device: str = "cpu", npz_cache_files: int = 2,
                 use_ae_encoder: bool = True,
                 ae_ckpt: str = "./ae_ckpts/ae_best.pth",
                 ae_in_ch: int = 3,
                 ae_latent_ch: int = 3,
                 ae_device: str = "cpu"):
        super().__init__()
        self.augment = augment
        self.class_strength = {0: 1.0, 1: 1.0}
        self.noise_factor = noise_factor
        self.channel_mask_pct = channel_mask_pct
        self.spatial_mask_pct = spatial_mask_pct
        self.cache_shards = max(1, int(cache_shards))
        self._cache = OrderedDict()
        self.dtype = dtype
        self.strength = 0.6

        self.mode = None  # 'pt_dir' | 'pt_single' | 'npz_dir'
        self._npz_files = []
        self._npz_index = []
        self._npz_cache = OrderedDict()
        self._npz_cache_files = max(1, int(npz_cache_files))
        self._wav = None

        # AE
        self.use_ae_encoder = bool(use_ae_encoder)
        self.ae = None
        self._ae_device = torch.device(ae_device if (ae_device == "cuda" and torch.cuda.is_available()) else "cpu")
        self._ae_in_ch = int(ae_in_ch)
        self._ae_latent_ch = int(ae_latent_ch)
        self._ae_ckpt = ae_ckpt

        if os.path.isdir(feature_path):
            has_pt = any(f.endswith(".pt") for f in os.listdir(feature_path))
            has_npz = any(f.endswith(".npz") for f in os.listdir(feature_path))
            if has_pt and not has_npz:
                # .pt 分片
                self.mode = 'pt_dir'
                cand = sorted([os.path.join(feature_path, f) for f in os.listdir(feature_path) if f.endswith(".pt")])
                shards, shard_sizes, labels_list = [], [], []
                for p in cand:
                    try:
                        pack = torch.load(p, map_location="cpu")
                    except Exception as e:
                        print(f"[WARN] skip {p} ({e})"); continue
                    if "labels" not in pack: print(f"[WARN] skip shard(no labels): {p}"); continue
                    n = int(pack["labels"].shape[0])
                    shard_sizes.append(n);
                    labels_list.append(pack["labels"].to(torch.long).cpu());
                    shards.append(p)
                    del pack
                if not shards: raise FileNotFoundError(f"No readable .pt under: {feature_path}")
                self.shards, self.shard_sizes = shards, shard_sizes
                self.cum_sizes = [];
                s = 0
                for n in self.shard_sizes: s += n; self.cum_sizes.append(s)
                self.total = self.cum_sizes[-1];
                self.labels = torch.cat(labels_list, dim=0)
                fp = torch.load(self.shards[0], map_location="cpu")
                self._feat_shape = tuple(fp["low"].shape[1:]);
                del fp

            elif has_npz:
                # .npz -> AE.encode(z) -> WT
                self.mode = 'npz_dir'
                self._npz_files = sorted(
                    [os.path.join(feature_path, f) for f in os.listdir(feature_path) if f.endswith(".npz")])
                if not self._npz_files:
                    raise FileNotFoundError(f"No .npz under: {feature_path}")
                total = 0
                labels_cat = []
                for fi, f in enumerate(self._npz_files):
                    with np.load(f) as npz:
                        n = int(npz["segments"].shape[0])
                        labs = torch.from_numpy(npz["labels"]).long()
                    for li in range(n):
                        self._npz_index.append((fi, li))
                    labels_cat.append(labs)
                    total += n
                self.labels = torch.cat(labels_cat, dim=0)
                self.total = total

                # WT（CPU/GPU 按 wt_device）
                self._wav = WaveletTransform(wave=wave_name, device=wt_device)

                # AE 编码器
                if self.use_ae_encoder:
                    self.ae = Autoencoder2D(in_ch=self._ae_in_ch, latent_ch=self._ae_latent_ch).to(self._ae_device)
                    try:
                        ckpt = torch.load(self._ae_ckpt, map_location=self._ae_device)
                        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                        self.ae.load_state_dict(state, strict=True)
                        self.ae.eval()
                        for p in self.ae.parameters(): p.requires_grad_(False)
                        print(
                            f"[EEGFeatureDataset] AE loaded from {self._ae_ckpt} (in_ch={self._ae_in_ch}, latent={self._ae_latent_ch})")
                    except Exception as e:
                        raise RuntimeError(f"Load AE ckpt failed: {self._ae_ckpt} ({e})")

                # 估计形状
                with np.load(self._npz_files[0]) as npz:
                    ex = torch.from_numpy(npz["segments"][0:1]).float()
                if self.use_ae_encoder and (self.ae is not None):
                    ex01 = minmax_unit(ex)
                    z, _, _ = self.ae.encode(ex01.to(self._ae_device))
                    ex_enc = z.to("cpu")
                else:
                    ex_enc = ex
                ex_enc, _, _ = _pad_to_even(ex_enc)
                low, h_h, h_v, h_d = self._wav(ex_enc)
                self._feat_shape = tuple(low.shape[1:])

            else:
                raise FileNotFoundError(f"No .pt or .npz under: {feature_path}")

        else:
            # 单个 .pt
            if not os.path.isfile(feature_path) or not feature_path.endswith(".pt"):
                raise FileNotFoundError(feature_path)
            self.mode = 'pt_single'
            data = torch.load(feature_path, map_location="cpu")
            self.low, self.high_h, self.high_v, self.high_d = data['low'], data['high_h'], data['high_v'], data[
                'high_d']
            self.labels = data['labels'].to(torch.long).cpu();
            self.total = int(self.labels.shape[0])
            if self.dtype is not None:
                self.low = self.low.to(self.dtype);
                self.high_h = self.high_h.to(self.dtype)
                self.high_v = self.high_v.to(self.dtype);
                self.high_d = self.high_d.to(self.dtype)
            self._feat_shape = tuple(self.low.shape[1:])

    def __len__(self):
        return self.total

    def set_class_strength(self, mapping: dict):
        self.class_strength = {int(k): float(v) for k, v in mapping.items()}

    def set_strength(self, s: float):
        self.strength = max(0.0, float(s))

    @property
    def feature_shape(self):
        return self._feat_shape

    # ====== 简化增广 ======
    def add_noise(self, x):
        s = float(self.strength);
        sigma = min(self.noise_factor * s, self.noise_factor * 2.0)
        return x if sigma <= 0 else x + torch.randn_like(x) * sigma

    def channel_dropout(self, x):
        return x

    def spatial_dropout(self, x):
        s = float(self.strength);
        p = min(1.0, 0.3 * s)
        if random.random() < p:
            c, h, w = x.shape;
            scale = (0.5 + 0.5 * s)
            mh = max(1, int(h * self.spatial_mask_pct * scale));
            mw = max(1, int(w * self.spatial_mask_pct * scale))
            mh = min(mh, int(0.4 * h));
            mw = min(mw, int(0.4 * w))
            top = random.randint(0, max(0, h - mh));
            left = random.randint(0, max(0, w - mw))
            mask = torch.ones_like(x);
            mask[:, top:top + mh, left:left + mw] = 0;
            x = x * mask
        return x

    def time_mask(self, x, p=0.25, frac=0.08):
        s = float(self.strength);
        p = min(1.0, p * s)
        if random.random() < p:
            c, h, w = x.shape;
            frac = min(0.5, frac * (0.5 + 1.0 * s));
            L = max(1, int(h * frac))
            st = random.randint(0, max(0, h - L));
            x = x.clone();
            x[:, st:st + L, :] = 0
        return x

    def amplitude_jitter(self, x, scale=0.1):
        if random.random() < 0.5 * self.strength:
            g = (1.0 + scale * self.strength * torch.randn(x.size(0), 1, 1, device=x.device, dtype=x.dtype)).clamp(0.7,
                                                                                                                   1.3)
            return x * g
        return x

    def baseline_wander(self, x):
        if random.random() < 0.5 * self.strength:
            c, h, w = x.shape
            t = torch.linspace(0, 2 * math.pi, steps=h, device=x.device, dtype=x.dtype)
            amp = 0.05 * self.strength * torch.randn(c, 1, 1, device=x.device, dtype=x.dtype)
            phase = 2 * math.pi * torch.rand(c, 1, 1, device=x.device, dtype=x.dtype)
            drift = amp * torch.sin(t.view(1, h, 1) + phase);
            return x + drift
        return x

    def freq_mask(self, x, p=0.25, frac=0.15):
        s = float(self.strength);
        p = min(1.0, p * s)
        if random.random() < p:
            c, h, w = x.shape;
            frac = min(0.5, frac * (0.5 + 1.0 * s));
            L = max(1, int(w * frac))
            st = random.randint(0, max(0, w - L));
            x = x.clone();
            x[:, :, st:st + L] = 0
        return x

    # ====== .pt 读取 ======
    def _find_shard(self, idx):
        shard_idx = bisect.bisect_right(self.cum_sizes, idx)
        start = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        return shard_idx, idx - start

    def _get_shard_pack(self, shard_idx):
        if shard_idx in self._cache:
            pack = self._cache.pop(shard_idx);
            self._cache[shard_idx] = pack;
            return pack
        path = self.shards[shard_idx];
        pack = torch.load(path, map_location="cpu")
        if self.dtype is not None:
            for k in ("low", "high_h", "high_v", "high_d"): pack[k] = pack[k].to(self.dtype)
        self._cache[shard_idx] = pack
        if len(self._cache) > self.cache_shards: self._cache.popitem(last=False)
        return pack

    def _get_item_from_shard(self, idx):
        shard_idx, off = self._find_shard(idx);
        pack = self._get_shard_pack(shard_idx)
        low = pack["low"][off].clone();
        hh = pack["high_h"][off].clone();
        hv = pack["high_v"][off].clone();
        hd = pack["high_d"][off].clone()
        label = pack["labels"][off].long().item()
        return low, hh, hv, hd, label, None  # pt 模式无 raw

    def _get_item_from_single(self, idx):
        low = self.low[idx].clone();
        hh = self.high_h[idx].clone();
        hv = self.high_v[idx].clone();
        hd = self.high_d[idx].clone()
        return low, hh, hv, hd, int(self.labels[idx].item()), None

    # ====== .npz：先 AE.encode → 再 WT ======
    def _load_npz_file(self, file_idx):
        if file_idx in self._npz_cache:
            arr = self._npz_cache.pop(file_idx)
            self._npz_cache[file_idx] = arr
            return arr
        f = self._npz_files[file_idx]
        with np.load(f) as npz:
            segs = torch.from_numpy(npz["segments"]).float()  # [N,C,H,W]
            labs = torch.from_numpy(npz["labels"]).long()
        self._npz_cache[file_idx] = (segs, labs)
        if len(self._npz_cache) > self._npz_cache_files:
            self._npz_cache.popitem(last=False)
        return segs, labs

    def _get_item_from_npz(self, idx):
        file_idx, local_idx = self._npz_index[idx]
        segs, labs = self._load_npz_file(file_idx)
        x = segs[local_idx].unsqueeze(0).float()  # [1,C,H,W]
        y = int(labs[local_idx].item())
        x_raw_pad, _, _ = _pad_to_even(x)  # raw：pad 后原图

        # AE.encode 得到 z
        if self.use_ae_encoder and (self.ae is not None):
            x01 = minmax_unit(x)
            z, _, _ = self.ae.encode(x01.to(self._ae_device))
            x_enc = z.to("cpu")
        else:
            x_enc = x

        x_enc_pad, _, _ = _pad_to_even(x_enc)
        low, hh, hv, hd = self._wav(x_enc_pad)
        return low[0].clone(), hh[0].clone(), hv[0].clone(), hd[0].clone(), y, x_raw_pad[0].clone()

    def __getitem__(self, idx):
        if self.mode == 'pt_dir':
            low, hh, hv, hd, label, raw = self._get_item_from_shard(idx)
        elif self.mode == 'pt_single':
            low, hh, hv, hd, label, raw = self._get_item_from_single(idx)
        elif self.mode == 'npz_dir':
            low, hh, hv, hd, label, raw = self._get_item_from_npz(idx)
        else:
            raise RuntimeError("Unknown dataset mode")

        if self.augment:
            def aug(x, y):
                s_eff = float(self.strength) * float(self.class_strength.get(int(y), 1.0))
                old = self.strength;
                self.strength = s_eff
                try:
                    x = self.amplitude_jitter(x);
                    x = self.add_noise(x);
                    x = self.baseline_wander(x)
                    x = self.channel_dropout(x);
                    x = self.spatial_dropout(x);
                    x = self.time_mask(x);
                    x = self.freq_mask(x)
                finally:
                    self.strength = old
                return x

            low, hh, hv, hd = aug(low, label), aug(hh, label), aug(hv, label), aug(hd, label)

        sample = {"low": low, "high_h": hh, "high_v": hv, "high_d": hd, "label": torch.tensor(label, dtype=torch.long)}
        if raw is not None:
            sample["raw"] = raw
        return sample


# ================== 日志 ==================
class MetricsLogger:
    def __init__(self, save_dir="result", csv_name="train_log.csv"):
        os.makedirs(save_dir, exist_ok=True);
        self.csv_path = os.path.join(save_dir, csv_name);
        self.rows = []
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "train_acc", "val_acc", "bal_acc", "f1", "auc",
                                        "train_con_loss", "train_cls_loss", "val_con_loss", "val_cls_loss",
                                        "tau", "lr_min", "lr_max", "thr", "pos_rate_val"])

    def log_epoch(self, **kw):
        keys = ["epoch", "train_acc", "val_acc", "bal_acc", "f1", "auc", "train_con_loss", "train_cls_loss",
                "val_con_loss", "val_cls_loss", "tau", "lr_min", "lr_max", "thr", "pos_rate_val"]
        row = [kw.get(k, "") for k in keys];
        self.rows.append(row)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f: csv.writer(f).writerow(row)

    def plot_curves(self, save_dir="result"):
        if not self.rows and os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                data = list(csv.reader(f));
                header, data = data[0], data[1:]
        else:
            header = ["epoch", "train_acc", "val_acc", "bal_acc", "f1", "auc", "train_con_loss", "train_cls_loss",
                      "val_con_loss", "val_cls_loss", "tau", "lr_min", "lr_max", "thr", "pos_rate_val"]
            data = self.rows
        idx = {k: i for i, k in enumerate(header)}
        epochs = [int(r[idx["epoch"]]) for r in data]
        tr = [float(r[idx["train_acc"]]) for r in data]
        va = [float(r[idx["val_acc"]]) for r in data]
        vb = [float(r[idx["bal_acc"]]) * 100.0 for r in data]
        plt.figure();
        plt.plot(epochs, tr, label="Train Acc");
        plt.plot(epochs, va, label="Val Acc")
        plt.xlabel("Epoch");
        plt.ylabel("Accuracy (%)");
        plt.title("Train/Val Accuracy");
        plt.legend();
        plt.grid(True)
        ap = os.path.join(save_dir, "accuracy_curve.png");
        plt.savefig(ap, dpi=150, bbox_inches="tight");
        plt.close()
        plt.figure();
        plt.plot(epochs, vb, label="Val Balanced Acc")
        plt.xlabel("Epoch");
        plt.ylabel("Balanced Acc (%)");
        plt.title("Val Balanced Acc");
        plt.legend();
        plt.grid(True)
        bp = os.path.join(save_dir, "balanced_accuracy_curve.png");
        plt.savefig(bp, dpi=150, bbox_inches="tight");
        plt.close()
        print(f"[LOG] 曲线已保存：{ap} | {bp}")


class tee_stdout:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True); self.path = path; self.file = None; self._stdout = None

    def __enter__(self):
        self.file = open(self.path, "w", encoding="utf-8"); self._stdout = sys.stdout; sys.stdout = self; return self

    def write(self, s):
        if self._stdout and not getattr(self._stdout, "closed", False): self._stdout.write(s); self._stdout.flush()
        if self.file and not self.file.closed: self.file.write(s); self.file.flush(); return len(s)

    def flush(self):
        try:
            if self._stdout and not getattr(self._stdout, "closed", False): self._stdout.flush()
        except:
            pass
        try:
            if self.file and not self.file.closed: self.file.flush()
        except:
            pass

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout if self._stdout is not None else sys.__stdout__
        if self.file and not self.file.closed: self.file.close()
        return False


# ================== 训练核心组件 ==================
class EMA:
    def __init__(self, modules: dict, decay=0.999):
        self.decay = float(decay);
        self.shadow = {}
        with torch.no_grad():
            for name, m in modules.items():
                for k, v in m.state_dict().items(): self.shadow[f"{name}.{k}"] = v.detach().clone()

    @torch.no_grad()
    def update(self, modules: dict):
        d = self.decay
        for name, m in modules.items():
            for k, v in m.state_dict().items():
                key = f"{name}.{k}"
                if key not in self.shadow: self.shadow[key] = v.detach().clone(); continue
                if torch.is_floating_point(v):
                    self.shadow[key].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self.shadow[key].copy_(v)

    def _load_shadow_into(self, modules: dict):
        with torch.no_grad():
            for name, m in modules.items():
                sd = m.state_dict()
                for k in sd.keys(): sd[k].copy_(self.shadow[f"{name}.{k}"])
                m.load_state_dict(sd, strict=True)

    def apply(self, modules: dict):
        class _Ctx:
            def __init__(self, ema, modules): self.ema, self.modules = ema, modules; self.backup = None

            def __enter__(self):
                self.backup = {n: {k: v.detach().clone() for k, v in m.state_dict().items()} for n, m in
                               self.modules.items()}
                self.ema._load_shadow_into(self.modules)

            def __exit__(self, exc_type, exc, tb):
                with torch.no_grad():
                    for n, m in self.modules.items(): m.load_state_dict(self.backup[n], strict=True)
                return False

        return _Ctx(self, modules)


def prob_balance_loss_from_logits(logits, target_prior=None):
    p = F.softmax(logits, dim=1).mean(dim=0).clamp_min(1e-8)
    target = torch.full_like(p, 1.0 / p.numel()) if target_prior is None else target_prior.to(p)
    return F.kl_div(p.log(), target, reduction='batchmean')


class _RelationProjector(nn.Module):
    def __init__(self, in_dim, hid=256, out_dim=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hid, out_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def _relation_vector(A: torch.Tensor, B: torch.Tensor, pool_hw=(4, 4)):
    assert A.shape == B.shape and A.dim() == 4
    Bsz, C, Hp, Wp = A.shape
    Ach = A.flatten(2);
    Bch = B.flatten(2)
    num = (Ach * Bch).sum(dim=2);
    den = (Ach.norm(dim=2) * Bch.norm(dim=2)).clamp_min(1e-8)
    rel_ch = (num / den)
    Ab = F.normalize(A, dim=1);
    Bb = F.normalize(B, dim=1)
    rel_sp = (Ab * Bb).sum(dim=1, keepdim=True)
    ph, pw = pool_hw
    rel_sp = F.adaptive_avg_pool2d(rel_sp, output_size=(ph, pw)).flatten(1)
    r = torch.cat([rel_ch, rel_sp], dim=1)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r


class HLF_RelationLoss_BP(nn.Module):
    def __init__(self, temperature=0.07, pool_hw=(4, 4),
                 w_hh_low=1.0, w_hv_low=1.0, w_hd_low=1.0,
                 proj_hidden=256, proj_out=128, p_drop=0.2):
        super().__init__()
        self.tau = float(temperature)
        self.pool_hw = pool_hw
        self.proj = None
        self.w = {'hh_low': w_hh_low, 'hv_low': w_hv_low, 'hd_low': w_hd_low}
        self.ntxent = NTXentLoss(temperature=self.tau)
        self.proj_hidden = proj_hidden
        self.proj_out = proj_out
        self.p_drop = p_drop

    def _ensure_projector(self, in_dim, device):
        if self.proj is None:
            self.proj = _RelationProjector(in_dim, self.proj_hidden, self.proj_out, self.p_drop).to(device)

    def _pair_nce(self, A, B):
        r = _relation_vector(A, B, pool_hw=self.pool_hw)
        self._ensure_projector(r.size(1), r.device)
        # 视图扰动：给 r 一个极小的高斯抖动，避免完全一致导致退化
        if self.training:
            r = r + 1e-4 * torch.randn_like(r)

        z1 = self.proj(r)
        z2 = self.proj(r)  # 仍保留 Dropout 视图差

        # 再次净化 & 退化检测
        z1 = torch.nan_to_num(z1)
        z2 = torch.nan_to_num(z2)
        # 若 batch 太小/退化，直接返回 0-loss
        if z1.size(0) < 2:
            return torch.zeros([], device=z1.device, dtype=z1.dtype)
        return self.ntxent(torch.cat([z1, z2], dim=0),
                           torch.cat([torch.arange(z1.size(0), device=z1.device)] * 2, dim=0))

    def forward(self, hh, hv, hd, low):
        loss = 0.0
        if self.w['hh_low'] > 0: loss = loss + self.w['hh_low'] * self._pair_nce(hh, low)
        if self.w['hv_low'] > 0: loss = loss + self.w['hv_low'] * self._pair_nce(hv, low)
        if self.w['hd_low'] > 0: loss = loss + self.w['hd_low'] * self._pair_nce(hd, low)
        return loss


# ======= CloFormer 风格模块（简化） =======
class _AttnMap2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x): return self.net(x)


class _CloLocal2D(nn.Module):
    def __init__(self, dim, dw_kernel=3, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.dw_q = nn.Conv2d(dim, dim, kernel_size=dw_kernel, padding=dw_kernel // 2, groups=dim, bias=True)
        self.dw_k = nn.Conv2d(dim, dim, kernel_size=dw_kernel, padding=dw_kernel // 2, groups=dim, bias=True)
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=dw_kernel, padding=dw_kernel // 2, groups=dim, bias=True)
        self.map = _AttnMap2D(dim)
        self.scale = (dim // heads) ** -0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.dw_q(q);
        k = self.dw_k(k);
        v = self.dw_v(v)
        attn = torch.tanh(self.map(q * k) * self.scale)
        return attn * v


class _CloGlobal2D(nn.Module):
    def __init__(self, dim, heads=4, window=7):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.to_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.pool = nn.AvgPool2d(kernel_size=window, stride=window) if window > 1 else nn.Identity()
        self.scale = (dim // heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape;
        h = self.heads
        q = self.to_q(x)
        kv = self.pool(x)
        k, v = self.to_kv(kv).chunk(2, dim=1)

        def _reshape(t, HH, WW): return t.view(B, h, C // h, HH * WW)

        q = _reshape(q, H, W);
        k = _reshape(k, *kv.shape[-2:]);
        v = _reshape(v, *kv.shape[-2:])
        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-1, -2)).transpose(-1, -2).contiguous().view(B, C, H, W)
        return out


class CloBlock2D(nn.Module):
    def __init__(self, dim, heads=4, dw_kernel=3, window=7, drop=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.local = _CloLocal2D(dim, dw_kernel=dw_kernel, heads=heads)
        self.global_ = _CloGlobal2D(dim, heads=heads, window=window)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        xl = self.local(x)
        xg = self.global_(x)
        y = self.fuse(torch.cat([xl, xg], dim=1))
        y = self.drop(y)
        return shortcut + y


class SimpleProjector(nn.Module):
    def __init__(self, in_channels, out_dim=128, patch=(16, 3), dropout=0.0,
                 clo_depth=1, clo_heads=4, clo_window=7, clo_dw_kernel=3):
        super().__init__()
        self.out_dim = int(out_dim)
        self.patch = tuple(patch)
        self.in_channels = int(in_channels)
        self.inorm = nn.InstanceNorm2d(in_channels, affine=True)

        self.use_fuse = (self.in_channels > 1)
        if self.use_fuse:
            self.fuse_fc = nn.Linear(in_channels, in_channels, bias=False)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels if not self.use_fuse else 1, self.out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.out_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        self.clo = nn.Sequential(*[
            CloBlock2D(self.out_dim, heads=clo_heads, dw_kernel=clo_dw_kernel,
                       window=clo_window, drop=dropout)
            for _ in range(max(1, int(clo_depth)))
        ])

        ph, pw = self.patch
        self.patch_embed = nn.Conv2d(self.out_dim, self.out_dim,
                                     kernel_size=(ph, pw), stride=(ph, pw),
                                     padding=(ph // 2, pw // 2), bias=False)

        self.vec_ln = nn.LayerNorm(self.out_dim)
        self.vec_dp = nn.Dropout(dropout)

    def _fuse_channels(self, x):
        if not self.use_fuse: return x
        B, C, H, W = x.shape
        gap = x.mean(dim=(2, 3))
        w = F.softmax(self.fuse_fc(gap), dim=1).view(B, C, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def forward(self, x, return_vec=True, return_feat=False):
        x = self.inorm(x)
        x = self._fuse_channels(x) if self.use_fuse else x
        x = self.pre(x)
        x = self.clo(x)
        feat = self.patch_embed(x)
        vec = feat.mean(dim=(2, 3))
        vec = self.vec_dp(self.vec_ln(vec))
        if return_feat and return_vec:
            return vec, feat
        elif return_vec:
            return vec
        else:
            return feat


class GatedMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=DROPOUT):
        super().__init__();
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden * 2);
        self.fc2 = nn.Linear(hidden, dim);
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, g = self.fc1(x).chunk(2, dim=-1);
        x = F.gelu(g) * a;
        x = self.fc2(x);
        return self.drop(x)


class FuseHL(nn.Module):

    def __init__(self, dim=128, dropout=DROPOUT, use_residual=False):
        super().__init__()
        # 注意：这里不再是 dim*2，而是 dim
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, low_vec: torch.Tensor, high_vec: torch.Tensor, return_alpha: bool = False):

        s = low_vec + high_vec  # [B, D]
        alpha = self.alpha_head(s)  # [B, 1] in (0,1)

        mix = (1.0 - alpha) * low_vec + alpha * high_vec
        fused = self.post(mix)
        if self.use_residual:
            fused = fused + self.residual(s)
        return (fused, alpha.squeeze(-1)) if return_alpha else fused


class BandModulator(nn.Module):
    def __init__(self, d_vec: int, c_in: int, scale: float = 0.25):
        super().__init__()
        self.to_low = nn.Linear(d_vec, c_in)
        self.to_hh = nn.Linear(d_vec, c_in)
        self.to_hv = nn.Linear(d_vec, c_in)
        self.to_hd = nn.Linear(d_vec, c_in)
        self.scale = float(scale)

    def forward(self, fused: torch.Tensor, yl: torch.Tensor, hh: torch.Tensor, hv: torch.Tensor, hd: torch.Tensor):
        B, C, _, _ = yl.shape

        def _gain(linear):
            g = linear(fused).view(B, C, 1, 1)
            return 1.0 + self.scale * torch.tanh(g)

        yl_m = yl * _gain(self.to_low)
        hh_m = hh * _gain(self.to_hh)
        hv_m = hv * _gain(self.to_hv)
        hd_m = hd * _gain(self.to_hd)
        return yl_m, hh_m, hv_m, hd_m


class ResidualFuseMap(nn.Module):
    def __init__(self, d_vec: int):
        super().__init__()
        self.beta = nn.Sequential(nn.LayerNorm(d_vec), nn.Linear(d_vec, 1))

    def forward(self, fused: torch.Tensor, x_in: torch.Tensor, recon: torch.Tensor):
        b = torch.sigmoid(self.beta(fused)).view(-1, 1, 1, 1)
        return (1.0 - b) * x_in + b * recon


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_factor=0.08, last_epoch=-1):
        self.warmup_epochs = warmup_epochs;
        self.max_epochs = max_epochs;
        self.min_factor = float(min_factor)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            w = epoch / max(1, self.warmup_epochs);
            return [base_lr * w for base_lr in self.base_lrs]
        t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cosf = 0.5 * (1 + math.cos(math.pi * t))
        return [base_lr * (self.min_factor + (1 - self.min_factor) * cosf) for base_lr in self.base_lrs]


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=DROPOUT):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(256, num_classes))

    def forward(self, x): return self.classifier(x)


def param_groups(module):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad: continue
        (no_decay if (p.ndim == 1 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}]


def set_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False


def stage_proj_contrast_mod_resfuse(
        x_in: torch.Tensor,
        wt: WaveletTransform,
        conv_low: nn.Module, conv_hh: nn.Module, conv_hv: nn.Module, conv_hd: nn.Module,
        high_gate: nn.Module, hl_fuse: nn.Module,
        modulator: nn.Module, resfuser: nn.Module,
        hlf_seq_loss: nn.Module,
        use_amp: bool,
        scale: float | None = None,
        resizer: nn.Module | None = None,
):
    # 1) 记住原始尺寸；pad 到偶数以便 DWT
    H0, W0 = x_in.shape[-2], x_in.shape[-1]
    x_b, ph, pw = _pad_to_even(x_in)
    # 2) WT
    low, hh, hv, hd = wt(x_b)

    # 3) 四个 projector + 关系对比
    vec_low, feat_low = conv_low(low.float(), return_vec=True, return_feat=True)
    vec_hh, feat_hh = conv_hh(hh.float(), return_vec=True, return_feat=True)
    vec_hv, feat_hv = conv_hv(hv.float(), return_vec=True, return_feat=True)
    vec_hd, feat_hd = conv_hd(hd.float(), return_vec=True, return_feat=True)
    cl_loss = hlf_seq_loss(feat_hh.float(), feat_hv.float(), feat_hd.float(), feat_low.float())

    # 4) 高频加权 + 低高融合
    three = torch.stack([vec_hh, vec_hv, vec_hd], dim=1)  # (B,3,D)
    gate3 = torch.softmax(high_gate(three), dim=1)  # (B,3,1)
    high_vec = (three * gate3).sum(dim=1)  # (B,D)
    fused, _ = hl_fuse(vec_low, high_vec, return_alpha=True)  # (B,D)

    # 5) 子带调制 + iDWT
    low_m, hh_m, hv_m, hd_m = modulator(fused, low, hh, hv, hd)
    x_recon = wt.inverse(low_m, hh_m, hv_m, hd_m).to(x_in.device)

    if (ph or pw):
        x_recon = x_recon[..., :H0, :W0]

    if (scale is not None) and (scale < 1.0):
        if resizer is None:
            resizer = Resize2D(mode="area")
        Ht = max(1, int(H0 * scale))
        Wt = max(1, int(W0 * scale))
        x_in_s = resizer(x_in, size=(Ht, Wt))
        x_rec_s = resizer(x_recon, size=(Ht, Wt))
        x_out = resfuser(fused, x_in_s, x_rec_s)
    else:
        # 不缩放时两者现在已同尺寸
        x_out = resfuser(fused, x_in, x_recon)

    return fused, x_out, cl_loss


def train(train_feature_path, val_feature_path, num_classes=2, epochs=100, batch_size=64, result_dir="result",
          use_ae_encoder=True, ae_ckpt="./ae_ckpts/ae_best.pth", ae_in_ch=3, ae_latent_ch=3, ae_device="cpu"):
    os.makedirs(result_dir, exist_ok=True)

    if isinstance(train_feature_path, (list, tuple)):
        train_datasets = [EEGFeatureDataset(p, augment=True, noise_factor=NOISE,
                                            wave_name="coif1", wt_device="cpu",
                                            use_ae_encoder=use_ae_encoder, ae_ckpt=ae_ckpt,
                                            ae_in_ch=ae_in_ch, ae_latent_ch=ae_latent_ch, ae_device=ae_device)
                          for p in train_feature_path]
        train_set = ConcatDataset(train_datasets);
        y_train = torch.cat([ds.labels.clone().long() for ds in train_datasets], dim=0)
        C, H, W = train_datasets[0].feature_shape
    else:
        train_set = EEGFeatureDataset(train_feature_path, augment=True, noise_factor=NOISE,
                                      wave_name="coif1", wt_device="cpu",
                                      use_ae_encoder=use_ae_encoder, ae_ckpt=ae_ckpt,
                                      ae_in_ch=ae_in_ch, ae_latent_ch=ae_latent_ch, ae_device=ae_device)
        y_train = train_set.labels.clone().long();
        C, H, W = train_set.feature_shape

    val_set = EEGFeatureDataset(val_feature_path, augment=False,
                                wave_name="coif1", wt_device="cpu",
                                use_ae_encoder=use_ae_encoder, ae_ckpt=ae_ckpt,
                                ae_in_ch=ae_in_ch, ae_latent_ch=ae_latent_ch, ae_device=ae_device)
    vC, vH, vW = val_set.feature_shape
    if (vC, vH, vW) != (C, H, W): raise ValueError(f"Train feature {(C, H, W)} != Val feature {(vC, vH, vW)}")
    print(f"[INFO] Feature shape: C={C}, H={H}, W={W}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = MetricsLogger(save_dir=result_dir, csv_name="train_log.csv")
    print(f"使用设备: {device}")

    num_classes = int(y_train.max().item() + 1)
    class_count = torch.bincount(y_train, minlength=num_classes).float()
    sample_weights = (1.0 / (class_count + 1e-6))[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    train_prior = (torch.bincount(y_train, minlength=num_classes).float());
    train_prior = (train_prior / train_prior.sum()).to('cpu')
    set_class_strength_for_train_dataset(train_set, {0: 1.0, 1: 1.2})

    # ----- 模型 -----
    conv_low = SimpleProjector(in_channels=C, out_dim=128, patch=(16, 3), dropout=DROPOUT).to(device)
    conv_high_h = SimpleProjector(in_channels=C, out_dim=128, patch=(16, 3), dropout=DROPOUT).to(device)
    conv_high_v = SimpleProjector(in_channels=C, out_dim=128, patch=(16, 3), dropout=DROPOUT).to(device)
    conv_high_d = SimpleProjector(in_channels=C, out_dim=128, patch=(16, 3), dropout=DROPOUT).to(device)

    high_gate = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    hl_fuse = FuseHL(dim=128, dropout=DROPOUT).to(device)
    classifier = ClassifierHead(in_dim=128, num_classes=num_classes, dropout=DROPOUT).to(device)

    hlf_seq_loss = HLF_RelationLoss_BP(
        temperature=0.15, pool_hw=(4, 4),
        w_hh_low=1.0, w_hv_low=1.0, w_hd_low=1.0,
        proj_hidden=256, proj_out=128, p_drop=0.2
    ).to(device)

    wt_runtime = WaveletTransform(wave="coif1", device=device.type)  # 和模型同设备
    modulator = BandModulator(d_vec=128, c_in=C, scale=0.25).to(device)
    resfuser = ResidualFuseMap(d_vec=128).to(device)
    # === 新增：全局复用的缩放器 ===
    resizer = Resize2D(mode="area")

    with torch.no_grad():
        pos = float((y_train == 1).sum().item());
        neg = float((y_train == 0).sum().item())
        p = max(1e-6, pos / max(1.0, (pos + neg)));
        prior_bias = math.log(p / (1.0 - p))
        final_linear = classifier.classifier[-1]
        # 预建 projector（关系向量维：128 + 4*4）
        hlf_seq_loss._ensure_projector(in_dim=128 + 4 * 4, device=device)
        _ = hlf_seq_loss.proj(torch.randn(2, 128 + 16, device=device))
        if isinstance(final_linear, nn.Linear) and final_linear.bias is not None and final_linear.out_features == 2:
            final_linear.bias.data[1] += prior_bias / 2.0;
            final_linear.bias.data[0] -= prior_bias / 2.0

    ema_modules = {
        "low": conv_low, "hh": conv_high_h, "hv": conv_high_v, "hd": conv_high_d,
        "hlfuse": hl_fuse, "cls": classifier, "high_gate": high_gate,
        "mod": modulator, "resf": resfuser
    }
    ema = EMA(ema_modules, decay=EMA_DECAY)

    # ----- 优化器 -----
    optim_params = []

    def add_module(m, lr):
        for g in param_groups(m):
            g["lr"] = lr
            optim_params.append(g)

    for m in [conv_low, conv_high_h, conv_high_v, conv_high_d]:
        add_module(m, BACKBONE_LR)
    for m in [high_gate, hl_fuse, classifier, hlf_seq_loss, modulator, resfuser]:
        add_module(m, HEAD_LR)
    optimizer = torch.optim.AdamW(optim_params)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, max_epochs=epochs, min_factor=0.08)

    # ----- 训练循环 -----
    use_amp = (device.type == 'cuda');
    scaler = GradScaler('cuda', enabled=use_amp)
    best_val_metric = 0.0;
    best_epoch = -1;
    best_thr_to_save = 0.5
    best_pack = {"val_acc": 0.0, "bal_acc": 0.0, "f1": 0.0, "auc": float('nan')}
    bad_epochs = 0;
    fnr_ema = None;
    press_ema = TARGET_LOW;
    train_thr = 0.5

    print("开始训练...");
    print("=" * 100)
    for epoch in range(epochs):
        s_epoch = aug_strength_schedule(epoch, epochs, s_max=1.0, s_min=0.6)
        set_aug_strength_for_train_dataset(train_loader.dataset, s_epoch)
        print(f"[AUG] strength = {s_epoch:.2f}")
        warm_ratio, decay_ratio, T = 0.40, 0.15, epochs
        if epoch < 5:
            W_CLS, W_CL = 1.5, 0.0
        else:
            t = min(1.0, max(0.0, (epoch - 5) / max(1, warm_ratio * T - 5)))
            wcl = 1.0 * 0.6 * (1 - math.cos(math.pi * t)) * 0.5
            if epoch >= (1.0 - decay_ratio) * T:
                s = (epoch - (1.0 - decay_ratio) * T) / (decay_ratio * T);
                wcl = wcl * (1 - 0.5 * s)
            W_CLS, W_CL = 1.5, wcl

        for m in [conv_low, conv_high_h, conv_high_v, conv_high_d, high_gate, hl_fuse, classifier, hlf_seq_loss,
                  modulator, resfuser]:
            m.train()
        total_cl, total_cls, total_bal = 0.0, 0.0, 0.0;
        correct = 0;
        total = 0;
        pos_cnt = 0

        for batch in train_loader:
            low = batch['low'].to(device, non_blocking=True)
            hh = batch['high_h'].to(device, non_blocking=True)
            hv = batch['high_v'].to(device, non_blocking=True)
            hd = batch['high_d'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            raw = batch.get('raw', None)
            if raw is not None: raw = raw.to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                # ---------- Stage-1：统一用封装函数（按 STAGE1_SCALE 缩放） ----------
                # 如果有原始图 raw（npz 模式），直接用 raw；否则用当前子带的 iDWT 重建作为输入图
                x0 = raw if raw is not None else wt_runtime.inverse(low, hh, hv, hd).to(device)
                fused1, y0, cl_loss1 = stage_proj_contrast_mod_resfuse(
                    x_in=x0, wt=wt_runtime,
                    conv_low=conv_low, conv_hh=conv_high_h, conv_hv=conv_high_v, conv_hd=conv_high_d,
                    high_gate=high_gate, hl_fuse=hl_fuse,
                    modulator=modulator, resfuser=resfuser,
                    hlf_seq_loss=hlf_seq_loss, use_amp=use_amp,
                    scale=STAGE1_SCALE, resizer=resizer
                )

                # ---------- Stage-M：再次 projector→对比→融合→调制→iDWT→残差（按 STAGEM_SCALE） ----------
                fusedM, y1, cl_lossM = stage_proj_contrast_mod_resfuse(
                    x_in=y0, wt=wt_runtime,
                    conv_low=conv_low, conv_hh=conv_high_h, conv_hv=conv_high_v, conv_hd=conv_high_d,
                    high_gate=high_gate, hl_fuse=hl_fuse,
                    modulator=modulator, resfuser=resfuser,
                    hlf_seq_loss=hlf_seq_loss, use_amp=use_amp,
                    scale=STAGEM_SCALE, resizer=resizer
                )

                # ---------- Stage-2：对 y1 再做一轮 WT→投影→对比→融合→分类 ----------
                y1_b, _, _ = _pad_to_even(y1)
                low2, hh2, hv2, hd2 = wt_runtime(y1_b)

                vec_low2, feat_low2 = conv_low(low2.float(), return_vec=True, return_feat=True)
                vec_hh2, feat_hh2 = conv_high_h(hh2.float(), return_vec=True, return_feat=True)
                vec_hv2, feat_hv2 = conv_high_v(hv2.float(), return_vec=True, return_feat=True)
                vec_hd2, feat_hd2 = conv_high_d(hd2.float(), return_vec=True, return_feat=True)

                cl_loss2 = hlf_seq_loss(feat_hh2.float(), feat_hv2.float(), feat_hd2.float(), feat_low2.float())

                three2 = torch.stack([vec_hh2, vec_hv2, vec_hd2], dim=1)
                gate3_2 = torch.softmax(high_gate(three2), dim=1)
                high_vec2 = (three2 * gate3_2).sum(dim=1)
                fused2, alpha_hl2 = hl_fuse(vec_low2, high_vec2, return_alpha=True)

                # --- mixup 与分类 ---
                press_t = torch.tensor([1.0, press_ema], device=fused2.device).detach()
                MIXUP_STOP = int(MIXUP_STOP_FR * epochs)
                if MIXUP_ALPHA > 0 and epoch < MIXUP_STOP:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    perm = torch.randperm(y.size(0), device=y.device)
                    fused_mix = lam * fused2 + (1 - lam) * fused2[perm]
                    logits_train = classifier(fused_mix)
                    y_one = F.one_hot(y, num_classes=logits_train.size(1)).float()
                    soft = lam * y_one + (1 - lam) * y_one[perm]
                    soft = soft * press_t;
                    soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    cls_loss = F.kl_div(F.log_softmax(logits_train, dim=1), soft, reduction='batchmean')
                    with torch.no_grad():
                        was = classifier.training;
                        classifier.eval()
                        logits_clean = classifier(fused2);
                        if was: classifier.train()
                else:
                    logits_train = classifier(fused2)
                    ce = F.cross_entropy(logits_train, y, reduction='none', label_smoothing=0.02)
                    w = torch.where(y == 1, torch.full_like(ce, press_ema), torch.ones_like(ce))
                    cls_loss = (ce * w).mean()
                    logits_clean = logits_train

                # --- 其它正则 ---
                with torch.no_grad():
                    probs_clean = F.softmax(logits_clean, dim=1)[:, 1]
                    fnr_cur = _batch_fnr(probs_clean, y, thr=train_thr)
                    if fnr_cur == fnr_cur:
                        fnr_ema = fnr_cur if (fnr_ema is None) else (SMOOTH * fnr_ema + (1.0 - SMOOTH) * fnr_cur)
                    press_cur = _map_fnr_to_press(fnr_ema if fnr_ema is not None else 0.0, LOW, HIGH, TARGET_LOW,
                                                  TARGET_HIGH)
                    press_ema = SMOOTH * press_ema + (1.0 - SMOOTH) * press_cur

                prob_bal = PROB_BAL_W * prob_balance_loss_from_logits(logits_train,
                                                                      target_prior=train_prior.to(logits_train.device))
                u3 = torch.full_like(gate3_2.mean(0), 1 / 3);
                gate_reg = GATE_REG_W * F.kl_div(gate3_2.mean(0).clamp_min(1e-8).log(), u3, reduction='sum')
                bal_loss = BALANCE_LOSS_W * (logits_train.mean(dim=0) ** 2).sum()

                cl_loss = cl_loss1 + cl_lossM + cl_loss2
                loss = W_CL * cl_loss + W_CLS * cls_loss + bal_loss + prob_bal + gate_reg
                if not torch.isfinite(loss):
                    # 跳过这个 batch，避免污染权重
                    optimizer.zero_grad(set_to_none=True)
                    continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(conv_low.parameters()) + list(conv_high_h.parameters()) +
                list(conv_high_v.parameters()) + list(conv_high_d.parameters()) +
                list(high_gate.parameters()) + list(hl_fuse.parameters()) +
                list(classifier.parameters()) + list(hlf_seq_loss.parameters()) +
                list(modulator.parameters()) + list(resfuser.parameters()),
                max_norm=0.5
            )
            scaler.step(optimizer);
            scaler.update();
            ema.update(ema_modules)

            bs = y.size(0);
            total_cl += cl_loss.item() * bs;
            total_cls += cls_loss.item() * bs;
            total_bal += bal_loss.item() * bs
            pred = logits_clean.argmax(dim=1);
            correct += (pred == y).sum().item();
            pos_cnt += (pred == 1).sum().item();
            total += bs

        for m in [conv_low, conv_high_h, conv_high_v, conv_high_d, high_gate, hl_fuse, classifier, hlf_seq_loss,
                  modulator, resfuser]:
            m.eval()
        scheduler.step()
        train_acc = 100.0 * correct / max(1, total);
        train_pos_rate = pos_cnt / max(1, total)
        lrs = [pg['lr'] for pg in optimizer.param_groups];
        lr_min, lr_max = min(lrs), max(lrs)

        # ----- 验证 -----
        with ema.apply(ema_modules):
            with torch.no_grad():
                val_cl = val_cls = 0.0;
                val_total = 0;
                val_logits_list = [];
                val_labels_list = []
                for batch in val_loader:
                    low = batch['low'].to(device);
                    hh = batch['high_h'].to(device);
                    hv = batch['high_v'].to(device);
                    hd = batch['high_d'].to(device)
                    y = batch['label'].to(device);
                    raw = batch.get('raw', None)
                    if raw is not None: raw = raw.to(device, non_blocking=True)
                    with autocast('cuda', enabled=use_amp):
                        # Stage-1（验证同样按 STAGE1_SCALE）
                        x0 = raw if raw is not None else wt_runtime.inverse(low, hh, hv, hd).to(device)
                        fused1, y0, cl_loss1 = stage_proj_contrast_mod_resfuse(
                            x_in=x0, wt=wt_runtime,
                            conv_low=conv_low, conv_hh=conv_high_h, conv_hv=conv_high_v, conv_hd=conv_high_d,
                            high_gate=high_gate, hl_fuse=hl_fuse,
                            modulator=modulator, resfuser=resfuser,
                            hlf_seq_loss=hlf_seq_loss, use_amp=use_amp,
                            scale=STAGE1_SCALE, resizer=resizer
                        )

                        # Stage-M（验证同样按 STAGEM_SCALE）
                        fusedM, y1, cl_lossM = stage_proj_contrast_mod_resfuse(
                            x_in=y0, wt=wt_runtime,
                            conv_low=conv_low, conv_hh=conv_high_h, conv_hv=conv_high_v, conv_hd=conv_high_d,
                            high_gate=high_gate, hl_fuse=hl_fuse,
                            modulator=modulator, resfuser=resfuser,
                            hlf_seq_loss=hlf_seq_loss, use_amp=use_amp,
                            scale=STAGEM_SCALE, resizer=resizer
                        )

                        # Stage-2
                        y1_b, _, _ = _pad_to_even(y1)
                        low2, hh2, hv2, hd2 = wt_runtime(y1_b)

                        vec_low2, feat_low2 = conv_low(low2, return_vec=True, return_feat=True)
                        vec_hh2, feat_hh2 = conv_high_h(hh2, return_vec=True, return_feat=True)
                        vec_hv2, feat_hv2 = conv_high_v(hv2, return_vec=True, return_feat=True)
                        vec_hd2, feat_hd2 = conv_high_d(hd2, return_vec=True, return_feat=True)

                        cl_loss2 = hlf_seq_loss(feat_hh2, feat_hv2, feat_hd2, feat_low2)
                        three2 = torch.stack([vec_hh2, vec_hv2, vec_hd2], dim=1)
                        gate3_2 = torch.softmax(high_gate(three2), dim=1)
                        high_vec2 = (three2 * gate3_2).sum(dim=1)
                        fused2, _ = hl_fuse(vec_low2, high_vec2, return_alpha=True)

                        logits = classifier(fused2);
                        cls_loss = F.cross_entropy(logits, y)
                        cl_loss = cl_loss1 + cl_lossM + cl_loss2
                    bs = y.size(0);
                    val_cl += cl_loss.item() * bs;
                    val_cls += cls_loss.item() * bs;
                    val_total += bs
                    val_logits_list.append(logits.detach().cpu());
                    val_labels_list.append(y.detach().cpu())

        val_logits = torch.cat(val_logits_list, dim=0);
        val_labels = torch.cat(val_labels_list, dim=0)
        probs = F.softmax(val_logits, dim=1)[:, 1].numpy();
        y_true = val_labels.numpy()
        grid = np.linspace(0.05, 0.95, 19);
        q = np.quantile(probs, np.linspace(0.05, 0.95, 19))
        ths = np.unique(np.clip(np.concatenate([grid, q]), 1e-4, 1 - 1e-4))
        best_bal, best_thr = 0.0, 0.5
        for thr in ths:
            pred = (probs >= thr).astype(int);
            pos_rate = pred.mean()
            if pos_rate < 0.10 or pos_rate > 0.90: continue
            bal = balanced_accuracy_score(y_true, pred)
            if bal > best_bal: best_bal, best_thr = bal, thr
        pred_best = (probs >= best_thr).astype(int)
        val_acc = (pred_best == y_true).mean() * 100.0;
        bal_acc = best_bal;
        f1 = f1_score(y_true, pred_best)
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float('nan')
        cm = confusion_matrix(y_true, pred_best);
        pos_rate_at_thr = float(pred_best.mean())

        metric = bal_acc * 100.0;
        improved = metric > best_val_metric
        if improved:
            best_val_metric = metric;
            best_epoch = epoch + 1;
            best_thr_to_save = float(best_thr)
            best_pack = {"val_acc": float(val_acc), "bal_acc": float(bal_acc * 100.0), "f1": float(f1),
                         "auc": float(auc)}
            torch.save({
                'conv_low': conv_low.state_dict(), 'conv_high_h': conv_high_h.state_dict(),
                'conv_high_v': conv_high_v.state_dict(), 'conv_high_d': conv_high_d.state_dict(),
                'high_gate': high_gate.state_dict(), 'hl_fuse': hl_fuse.state_dict(),
                'classifier': classifier.state_dict(), 'hlf_seq_loss': hlf_seq_loss.state_dict(),
                'modulator': modulator.state_dict(), 'resfuser': resfuser.state_dict(),
                'epoch': epoch, 'best_metric_bal_acc': best_val_metric, 'best_thr': best_thr_to_save,
                'val_acc_at_best': float(val_acc), 'val_f1_at_best': float(f1), 'val_auc_at_best': float(auc),
                'ema_shadow': ema.shadow, 'ema_decay': EMA_DECAY,
            }, os.path.join(result_dir, "best_model.pt"))
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch + 1:3d}/{epochs}")
        print(f"  Train: ConLoss={total_cl / max(1, total):.4f}, ClasLoss={total_cls / max(1, total):.4f}  "
              f"Acc={train_acc:.2f}%  pos_rate={train_pos_rate:.2f}  (lr=[{lr_min:.2e}, {lr_max:.2e}])")
        print(f"  Val:   ConLoss={val_cl / max(1, val_total):.4f}, ClasLoss={val_cls / max(1, val_total):.4f}, "
              f"Acc={val_acc:.2f}%  BalAcc={bal_acc * 100:.2f}%  F1={f1:.3f}  AUC={auc:.3f}  thr*={best_thr:.2f}  pos_rate@thr*={pos_rate_at_thr:.2f}")
        print(f"  CM:\n{cm}")

        logger.log_epoch(epoch=epoch + 1, train_acc=train_acc, val_acc=val_acc, bal_acc=bal_acc, f1=f1,
                         auc=auc if not (isinstance(auc, float) and (auc != auc)) else "",
                         train_con_loss=total_cl / max(1, total), train_cls_loss=total_cls / max(1, total),
                         val_con_loss=val_cl / max(1, val_total), val_cls_loss=val_cls / max(1, val_total),
                         lr_min=lr_min, lr_max=lr_max, thr=best_thr, pos_rate_val=pos_rate_at_thr)

        if bad_epochs >= PATIENCE:
            print(
                f"Early stop at epoch {epoch + 1}. Best BalAcc={best_val_metric:.2f}% @ epoch {best_epoch}, thr*={best_thr_to_save:.2f}")
            break

    print(f"[DONE] Best BalAcc={best_val_metric:.2f}% @ epoch {best_epoch}, thr*={best_thr_to_save:.2f}")
    logger.plot_curves(save_dir=result_dir)
    return {"best_bal_acc": best_val_metric, "best_epoch": best_epoch, "best_thr": best_thr_to_save, **best_pack}


def run_single_fold(slices_root="./slices", val_fold=1, folds=5, epochs=EPOCHS, batch_size=64,
                    use_ae_encoder=True, ae_ckpt="./ae_ckpts/ae_best.pth", ae_in_ch=3, ae_latent_ch=3, ae_device="cpu"):
    def has_npz(dir_):
        return os.path.isdir(dir_) and any(f.endswith(".npz") for f in os.listdir(dir_))

    if not (1 <= int(val_fold) <= int(folds)): raise ValueError(f"--val_fold 必须在 1..{folds}，当前 {val_fold}")
    train_dirs = []
    for j in range(1, folds + 1):
        if j == val_fold: continue
        d = os.path.join(slices_root, f"fold{j}_overlap", "shards")
        if has_npz(d):
            train_dirs.append(d)
        else:
            print(f"[警告] 跳过不存在/无 .npz 的目录：{d}")
    val_dir = os.path.join(slices_root, f"fold{val_fold}_nonoverlap", "shards")
    if not train_dirs: raise RuntimeError("没有可用训练分片（未发现 .npz）")
    if not has_npz(val_dir): raise RuntimeError(f"验证目录无 .npz：{val_dir}")
    result_dir = os.path.join("result", f"fold{val_fold}")
    stamp = time.strftime('%Y%m%d_%H%M%S');
    log_path = os.path.join(result_dir, f"console_{stamp}.log")
    print(f"\n========== Single Run（验证=Fold {val_fold} nonoverlap；训练=其余折 overlap）==========")
    for td in train_dirs: print("  -", td)
    print("Val:", val_dir);
    print(f"Result dir: {result_dir}")
    with tee_stdout(log_path):
        metrics = train(train_feature_path=train_dirs, val_feature_path=val_dir, num_classes=2, epochs=epochs,
                        batch_size=batch_size, result_dir=result_dir,
                        use_ae_encoder=use_ae_encoder, ae_ckpt=ae_ckpt, ae_in_ch=ae_in_ch, ae_latent_ch=ae_latent_ch,
                        ae_device=ae_device)
    print(f"[DONE] Fold {val_fold} 最优 BalAcc={metrics['best_bal_acc']:.2f}% (epoch {metrics['best_epoch']})，"
          f"Acc={metrics['val_acc']:.2f}%  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  thr*={metrics['best_thr']:.2f}")
    print(f"控制台日志保存在：{log_path}")
    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices_root", type=str, default="./slices", help="包含 .npz 的根目录（fold*_*/shards/*.npz）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=64)
    # AE 参数
    ap.add_argument("--use_ae_encoder", action="store_true", help="开启后：首次WT前先过AE.encode(z)")
    ap.add_argument("--ae_ckpt", type=str, default="./ae_ckpts/ae_best.pth")
    ap.add_argument("--ae_in_ch", type=int, default=3)
    ap.add_argument("--ae_latent_ch", type=int, default=3, help="z 的通道数；也决定首次WT的 C")
    ap.add_argument("--ae_device", type=str, default="cpu", choices=["cpu", "cuda"])
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    run_single_fold(slices_root=args.slices_root, val_fold=FOLD, folds=args.folds, epochs=args.epochs,
                    batch_size=args.batch_size,
                    use_ae_encoder=args.use_ae_encoder, ae_ckpt=args.ae_ckpt, ae_in_ch=args.ae_in_ch,
                    ae_latent_ch=args.ae_latent_ch, ae_device=args.ae_device)