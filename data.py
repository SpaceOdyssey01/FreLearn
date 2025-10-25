from config import *
from wt_utils import *
from ae import*
from torch.utils.data import ConcatDataset as Concat
# ================== 工具：AUG/阈值相关 ==================
def set_aug_strength_for_train_dataset(ds, s: float):
    if isinstance(ds, Concat):
        for sub in ds.datasets:
            if hasattr(sub, "set_strength"): sub.set_strength(s)
    else:
        if hasattr(ds, "set_strength"): ds.set_strength(s)


def set_class_strength_for_train_dataset(ds, mapping: dict):
    
    if isinstance(ds, Concat):
        for sub in ds.datasets:
            if hasattr(sub, "set_class_strength"): sub.set_class_strength(mapping)
    else:
        if hasattr(ds, "set_class_strength"): ds.set_class_strength(mapping)


def batch_fnr(probs_pos: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    with torch.no_grad():
        pred = (probs_pos >= thr).long();
        y = labels.long()
        tp = ((pred == 1) & (y == 1)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        den = tp + fn
        if den == 0: return float('nan')
        return fn / float(den)


def map_fnr_to_press(fnr_ema: float, low: float, high: float, tl: float, th: float) -> float:
    if not (fnr_ema == fnr_ema): return tl
    if high <= low: return tl
    t = (fnr_ema - low) / (high - low);
    t = min(1.0, max(0.0, t))
    return tl * (1.0 - t) + th * t


def aug_strength_schedule(epoch: int, max_epochs: int, s_max=0.8, s_min=0.3):
    t = epoch / max(1, max_epochs - 1)
    return s_min + 0.5 * (s_max - s_min) * (1.0 + math.cos(math.pi * t))

import os, random
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGFeatureDataset(Dataset):
    def __init__(self, feature_path: str, augment: bool = False):
        super().__init__()
        self.root = feature_path
        self.augment = bool(augment)

        self._npz_files = sorted(
            [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith(".npz")]
        )
        if not self._npz_files:
            raise RuntimeError(f"No .npz found under: {self.root}")

        self._npz_index = []
        self._lengths = []
        for fi, f in enumerate(self._npz_files):
            with np.load(f, allow_pickle=False, mmap_mode="r") as z:
                if not all(k in z.files for k in ("segments", "labels")):
                    raise KeyError(f"{f} must contain 'segments' and 'labels'. has: {z.files}")
                N = z["segments"].shape[0]
                self._lengths.append(N)
                self._npz_index.extend([(fi, i) for i in range(N)])

        self._npz_cache = {}        # file_idx -> (segments, labels, subjects)
        self._npz_cache_files = 2

        self.noise_sigma = 0.02
        self.freq_mask_frac = 0.15
        self.time_mask_frac = 0.08
        self.spatial_mask_pct = 0.20

    def __len__(self):
        return len(self._npz_index)

    def _load_npz_file(self, file_idx):
        if file_idx in self._npz_cache:
            segs, labs, subs = self._npz_cache.pop(file_idx)
            self._npz_cache[file_idx] = (segs, labs, subs)
            return segs, labs, subs

        f = self._npz_files[file_idx]
        with np.load(f, allow_pickle=False, mmap_mode="r") as npz:
            # 关键改动：不要 np.asarray，直接 from_numpy 保留 memmap 视图
            segs_np = npz["segments"]        # memmap (N, C, H, W) or (N, H, W)
            labs_np = npz["labels"]
            subs_np = npz["subjects"] if "subjects" in npz.files else None

        segs = torch.from_numpy(segs_np).float()
        labs = torch.from_numpy(labs_np).long()
        subs = (torch.from_numpy(subs_np).long()
                if subs_np is not None
                else torch.full((segs.shape[0],), -1, dtype=torch.long))

        self._npz_cache[file_idx] = (segs, labs, subs)
        if len(self._npz_cache) > self._npz_cache_files:
            # 驱逐最旧
            oldest_key = next(iter(self._npz_cache))
            self._npz_cache.pop(oldest_key)
        return segs, labs, subs

    @staticmethod
    def _to_chw3(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)                # [1,H,W]
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"unexpected x.dim={x.dim()}, shape={tuple(x.shape)}")

        C, H, W = x.shape
        if C == 3:
            return x.contiguous()
        if C == 1:
            return x.repeat(3, 1, 1).contiguous()
        x1 = x.mean(dim=0, keepdim=True)
        return x1.repeat(3, 1, 1).contiguous()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return x
        # 关键改动：进入增广前先 clone，避免污染缓存
        x = x.clone()

        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])

        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * self.noise_sigma

        if torch.rand(1).item() < 0.25:
            H = x.shape[1]
            L = max(1, int(H * self.time_mask_frac))
            st = random.randint(0, max(0, H - L))
            x[:, st:st+L, :] = 0

        if torch.rand(1).item() < 0.25:
            W = x.shape[2]
            L = max(1, int(W * self.freq_mask_frac))
            st = random.randint(0, max(0, W - L))
            x[:, :, st:st+L] = 0

        if torch.rand(1).item() < 0.3:
            H, W = x.shape[1], x.shape[2]
            mh = max(1, int(H * self.spatial_mask_pct))
            mw = max(1, int(W * self.spatial_mask_pct))
            top = random.randint(0, max(0, H - mh))
            left = random.randint(0, max(0, W - mw))
            x[:, top:top+mh, left:left+mw] = 0

        return x

    def __getitem__(self, idx):
        file_idx, local_idx = self._npz_index[idx]
        segs, labs, subs = self._load_npz_file(file_idx)

        x = segs[local_idx]                 # 可能仍指向缓存 —— 记得增广先 clone
        y = labs[local_idx].item()
        sid = subs[local_idx].item()

        x = self._to_chw3(x).float()
        x = self._augment(x)                # 这里面会 clone

        return {
            "raw": x,                                  # FloatTensor [3,H,W]
            "label": torch.tensor(y, dtype=torch.long),
            "sid": torch.tensor(sid, dtype=torch.long) # 建议 tensor，更稳
        }
