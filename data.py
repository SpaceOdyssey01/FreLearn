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
                        for p in self.ae.parameters():
                            p.requires_grad_(False)

                        ckpt_abs = os.path.abspath(self._ae_ckpt)
                        meta = []
                        if isinstance(ckpt, dict):
                            for k in ["epoch", "step", "best", "iter", "global_step"]:
                                if k in ckpt:
                                    meta.append(f"{k}={ckpt[k]}")
                        n_params = sum(p.numel() for p in self.ae.parameters())
                        meta_str = (" [" + ", ".join(meta) + "]") if meta else ""
                        print(f"[AE] Loaded: {ckpt_abs}  "
                            f"(in_ch={self._ae_in_ch}, latent={self._ae_latent_ch}, params={n_params/1e6:.2f}M){meta_str}")

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
                ex_enc, _, _ = pad_to_even(ex_enc)
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
        x_raw_pad, _, _ = pad_to_even(x)  # raw：pad 后原图

        # AE.encode 得到 z
        if self.use_ae_encoder and (self.ae is not None):
            x01 = minmax_unit(x)
            z, _, _ = self.ae.encode(x01.to(self._ae_device))
            x_enc = z.to("cpu")
        else:
            x_enc = x

        x_enc_pad, _, _ = pad_to_even(x_enc)
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
