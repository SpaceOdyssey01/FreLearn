from config import *
from ae import *
from wt_utils import *
from data import *
from contrast import HLF_RelationLoss_BP
import json
from collections import defaultdict
from typing import List
import time
from contextlib import contextmanager

class StepProfiler:
    def __init__(self, device, name="train", max_batches=100):
        self.device = device
        self.name = name
        self.max_batches = max_batches
        self.reset()

    def reset(self):
        self.n = 0
        self.t_fetch = 0.0
        self.t_to = 0.0
        self.t_fwd = 0.0
        self.t_bwd = 0.0
        self.t_opt = 0.0
        self.use_cuda_evt = (hasattr(self.device, "type") and self.device.type == "cuda")

    @contextmanager
    def _evt(self):
        if self.use_cuda_evt:
            import torch
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.perf_counter()
            yield lambda: (None, None, time.perf_counter() - t0)

    def add_fetch(self, dt):
        if self.n < self.max_batches:
            self.t_fetch += dt

    def add_to(self, dt):
        if self.n < self.max_batches:
            self.t_to += dt

    def add_fwd(self, dt):
        if self.n < self.max_batches:
            self.t_fwd += dt

    def add_bwd(self, dt):
        if self.n < self.max_batches:
            self.t_bwd += dt

    def add_opt(self, dt):
        if self.n < self.max_batches:
            self.t_opt += dt

    def step_done(self):
        self.n += 1

    def summary(self):
        n = max(1, min(self.n, self.max_batches))
        total = (self.t_fetch + self.t_to + self.t_fwd + self.t_bwd + self.t_opt) / n
        lines = [
            f"[Profile:{self.name}] per-step avg over {n} batches (sec):",
            f"  fetch(DataLoader): {self.t_fetch/n:.4f}",
            f"  to(device)      : {self.t_to/n:.4f}",
            f"  forward+loss    : {self.t_fwd/n:.4f}",
            f"  backward        : {self.t_bwd/n:.4f}",
            f"  optimizer step  : {self.t_opt/n:.4f}",
            f"  ---- total(step): {total:.4f} sec"
        ]
        print("\n".join(lines))


def timed_fetch(it):
    t0 = time.perf_counter()
    batch = next(it)
    dt = time.perf_counter() - t0
    return batch, dt

class Resize2D(nn.Module):

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

class _AttnMap2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )
    def forward(self, x):
        return self.net(x)

class _LocalBranch(nn.Module):
   
    def __init__(self, dim, dim_head, group_heads: int, kernel_size: int, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.group_heads = group_heads
        self.m = group_heads * dim_head
        self.to_qkv = nn.Conv2d(dim, 3 * self.m, kernel_size=1, bias=qkv_bias)

        # 对拼接后的 QKV 一次性做 DW 卷积（和 blocks.py 对齐）
        self.mixer = nn.Conv2d(3 * self.m, 3 * self.m, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, groups=3 * self.m, bias=True)

        self.attn_map = _AttnMap2D(self.m)   # 对 q*k 做门控映射
        self.scale = (self.dim_head) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)                               # [B, 3m, H, W]
        qkv = self.mixer(qkv)                              # DWConv 混合
        q, k, v = qkv.chunk(3, dim=1)                      # 各 [B, m, H, W]
        attn = self.attn_map(q * k) * self.scale
        attn = torch.tanh(attn)
        attn = self.attn_drop(attn)
        out = attn * v                                     # [B, m, H, W]
        return out

class _GlobalBranch(nn.Module):
    
    def __init__(self, dim, dim_head, group_heads: int, window: int, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.group_heads = group_heads
        self.m = group_heads * dim_head
        self.to_q  = nn.Conv2d(dim, self.m, kernel_size=1, bias=qkv_bias)
        self.to_kv = nn.Conv2d(dim, 2 * self.m, kernel_size=1, bias=qkv_bias)
        self.pool  = nn.AvgPool2d(kernel_size=window, stride=window) if window > 1 else nn.Identity()
        self.scale = (self.dim_head) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_heads
        dh = self.dim_head

        q = self.to_q(x)                                   # [B, m, H, W]
        kv_in = self.pool(x)                               # [B, C, H', W']
        k, v = self.to_kv(kv_in).chunk(2, dim=1)           # [B, m, H', W'], [B, m, H', W']

        # reshape 到多头
        def to_heads(t, HH, WW):                           # t: [B, m, HH, WW]
            return t.view(B, h, dh, HH * WW)               # [B, h, dh, L]
        q = to_heads(q,  H,  W)                            # [B, h, dh, HW]
        k = to_heads(k, *kv_in.shape[-2:])                 # [B, h, dh, H'W']
        v = to_heads(v, *kv_in.shape[-2:])                 # [B, h, dh, H'W']

        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale   # [B, h, HW, H'W']
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v.transpose(-1, -2))              # [B, h, HW, dh]
        out = out.transpose(-1, -2).contiguous().view(B, self.m, H, W)  # [B, m, H, W]
        return out

class CloBlock2D_MS(nn.Module):
   
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 group_split: List[int],
                 kernel_sizes: List[int],
                 window: int = 7,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 qkv_bias: bool = True):
        super().__init__()
        assert sum(group_split) == num_heads, "sum(group_split) must equal num_heads"
        assert len(kernel_sizes) + 1 == len(group_split), "len(kernel_sizes)+1 must equal len(group_split)"
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.group_split = group_split
        self.kernel_sizes = kernel_sizes
        self.window = window

        self.norm = nn.GroupNorm(1, dim)

        # 构建局部多尺度分支
        locals_ = []
        for g_heads, ksz in zip(group_split[:-1], kernel_sizes):
            if g_heads > 0:
                locals_.append(
                    _LocalBranch(dim, self.dim_head, g_heads, ksz, qkv_bias=qkv_bias, attn_drop=attn_drop)
                )
        self.local_branches = nn.ModuleList(locals_)

        # 构建全局分支（最后一个 split）
        self.global_branch = None
        g_heads_global = group_split[-1]
        if g_heads_global > 0:
            self.global_branch = _GlobalBranch(dim, self.dim_head, g_heads_global,
                                               window=window, qkv_bias=qkv_bias, attn_drop=attn_drop)

        # 融合投影：cat(各分支) → 1x1 → [B, dim, H, W]
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout2d(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)

        outs = []
        for lb in self.local_branches:
            outs.append(lb(x))                 # 每个 [B, g_heads*dh, H, W]

        if self.global_branch is not None:
            outs.append(self.global_branch(x)) # [B, g_heads_global*dh, H, W]


        if len(outs) == 0:
            return shortcut

        y = torch.cat(outs, dim=1)            # [B, sum_heads*dh, H, W] == [B, dim, H, W]
        y = self.proj(y)                      # [B, dim, H, W]
        y = self.proj_drop(y)
        return shortcut + y                   # 残差


class SimpleProjector(nn.Module):
    def __init__(self, in_channels, out_dim=128, patch=(16, 3), dropout=0.0,
                 clo_depth=1, clo_heads=3, clo_window=7, clo_dw_kernel=3,
                 spatial_only=False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.spatial_only = bool(spatial_only)
        self.inorm = nn.InstanceNorm2d(in_channels, affine=True)

        
        self.use_fuse = (in_channels > 1) and (not self.spatial_only)

        if self.spatial_only:
            
            self.pre = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, in_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
            )
            self.clo = nn.Sequential(*[
                CloBlock2D_MS(
                    
                    dim=in_channels,
                    num_heads=clo_heads,
                    group_split=[max(1, clo_heads//3), max(1, clo_heads//3), clo_heads - 2*max(1, clo_heads//3)],
                    kernel_sizes=[clo_dw_kernel, max(3, clo_dw_kernel-2)],
                    window=clo_window,
                    
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qkv_bias=True
                )
                for _ in range(max(1, int(clo_depth)))
            ])
            self.down = nn.Identity()
        else:

            self.out_dim = int(out_dim)
            self.patch = tuple(patch)
            if self.use_fuse:
                self.fuse_fc = nn.Linear(in_channels, in_channels, bias=False)
            self.pre = nn.Sequential(
                nn.Conv2d(in_channels if not self.use_fuse else 1, self.out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, self.out_dim), nn.GELU(), nn.Dropout2d(dropout),
            )
            self.clo = nn.Sequential(*[
                CloBlock2D_MS(
                    dim=self.out_dim,
                    num_heads=clo_heads,
                    group_split=[max(1, clo_heads//3), max(1, clo_heads//3), clo_heads - 2*max(1, clo_heads//3)],
                    kernel_sizes=[clo_dw_kernel, max(3, clo_dw_kernel-2)],
                    window=clo_window,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qkv_bias=True
                )
                for _ in range(max(1, int(clo_depth)))
            ])
            ph, pw = self.patch
            self.patch_embed = nn.Conv2d(self.out_dim, self.out_dim,
                                         kernel_size=(ph, pw), stride=(ph, pw),
                                         padding=(ph // 2, pw // 2), bias=False)
            self.vec_ln = nn.LayerNorm(self.out_dim)
            self.vec_dp = nn.Dropout(dropout)

    def _fuse_channels(self, x):

        B, C, H, W = x.shape
        gap = x.mean(dim=(2, 3))
        w = F.softmax(self.fuse_fc(gap), dim=1).view(B, C, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)

    def forward(self, x, return_vec=True, return_feat=False, subband_size=None):
        x = self.inorm(x)

        if self.spatial_only:
           
            x = self.pre(x)
            x = self.clo(x)
            x = self.down(x)                         # [B,3,H',W']

            if (subband_size is not None) and (x.shape[-2:] != subband_size):
                x = F.interpolate(x, size=subband_size, mode="bilinear", align_corners=False)
            # spatial_only 返回“子带图”（供 iDWT 和 HLF）
            return (x, x) if return_feat else x

        if self.use_fuse:
            x = self._fuse_channels(x)
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



class HighGateMap(nn.Module):
    """从三张高频子带图（hh/hv/hd）计算 (B,3,1,1) 的样本级权重"""
    def __init__(self, c_in=3, hidden=16):
        super().__init__()
        # 用全局均值（也可拼通道均值）做一个极轻的门控
        self.mlp = nn.Sequential(
            nn.Linear(3 * c_in, hidden), nn.GELU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, hh_map, hv_map, hd_map):
        B, C, _, _ = hh_map.shape
        s = torch.cat([hh_map.mean((2,3)), hv_map.mean((2,3)), hd_map.mean((2,3))], dim=1)  # [B, 3C]
        w = F.softmax(self.mlp(s), dim=1).view(B, 3, 1, 1)                                   # [B,3,1,1]
        return w

class AlphaHL(nn.Module):
    """从低/高融合候选图得到像素无关的 α"""
    def __init__(self, c_in=3, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * c_in, hidden), nn.GELU(),
            nn.Linear(hidden, c_in), nn.Sigmoid()
        )
    def forward(self, low_map, high_map):
        s = torch.cat([low_map.mean((2,3)), high_map.mean((2,3))], dim=1)  # [B, 2C]
        alpha = self.mlp(s).view(-1, low_map.size(1), 1, 1)                # [B,C,1,1] in (0,1)
        return alpha

def stage_spatial_fusion_iwt(
        x_in: torch.Tensor, wt: WaveletTransform,
        conv_low: nn.Module, conv_hh: nn.Module, conv_hv: nn.Module, conv_hd: nn.Module,
        high_gate_map: nn.Module, alpha_hl: nn.Module,
        hlf_seq_loss: nn.Module,
        resfuser: nn.Module,
        resizer: nn.Module|None = None,
        scale: float|None = None
):
    H0, W0 = x_in.shape[-2], x_in.shape[-1]
    x_b, ph, pw = pad_to_even(x_in)

    # WT
    low, hh, hv, hd = wt(x_b)  # [B,3,H/2,W/2] ×4


    low_map, feat_low = conv_low(low, return_feat=True, subband_size=low.shape[-2:])
    hh_map,  feat_hh  = conv_hh(hh,  return_feat=True, subband_size=hh.shape[-2:])
    hv_map,  feat_hv  = conv_hv(hv,  return_feat=True, subband_size=hv.shape[-2:])
    hd_map,  feat_hd  = conv_hd(hd,  return_feat=True, subband_size=hd.shape[-2:])

    cl_loss = hlf_seq_loss(feat_hh.float(), feat_hv.float(), feat_hd.float(), feat_low.float())


    w3 = high_gate_map(hh_map, hv_map, hd_map)                   # [B,3,1,1]
    high_stack = torch.stack([hh_map, hv_map, hd_map], dim=1)    # [B,3,3,H/2,W/2]
    high_map = (high_stack * w3.unsqueeze(2)).sum(dim=1)         # [B,3,H/2,W/2]


    alpha = alpha_hl(low_map, high_map)                          # [B,3,1,1]
    low_f = (1.0 - alpha) * low_map + alpha * high_map           # [B,3,H/2,W/2]
    hh_f, hv_f, hd_f = hh_map, hv_map, hd_map              

    x_rec = wt.inverse(low_f, hh_f, hv_f, hd_f)
    if (ph or pw): x_rec = x_rec[..., :H0, :W0]

    if (scale is not None) and (scale < 1.0):
        if resizer is None: resizer = Resize2D(mode="area")
        Ht, Wt = max(1, int(H0 * scale)), max(1, int(W0 * scale))
        x_in_s  = resizer(x_in,  size=(Ht, Wt))
        x_rec_s = resizer(x_rec, size=(Ht, Wt))
        x_out = resfuser(low_map.mean((2,3)), x_in_s, x_rec_s)  # 残差比例也可由 low_map 均值代替 fused 向量
    else:
        x_out = resfuser(low_map.mean((2,3)), x_in, x_rec)      # 仍用 ResidualFuseMap；其 beta 输入换成低频统计

    return x_out, cl_loss



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

class ClassifierFrom3(nn.Module):
    def __init__(self, num_classes, hidden=32, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

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
    x_b, ph, pw = pad_to_even(x_in)
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
        
        x_out = resfuser(fused, x_in, x_recon)

    return fused, x_out, cl_loss
def _build_ae_if_needed(use_ae_encoder: bool, device, ae_ckpt: str, ae_in_ch: int, ae_latent_ch: int):
    """
    在 GPU 上构建 AE，并把参数冻住，仅用于 encode()。
    若未启用或加载失败，返回 None，等价于不做 AE 编码。
    """
    if not use_ae_encoder:
        return None

    # 你自己的 AE 类名（按需改成你项目里的类/构造函数）
    AE_CANDIDATE_NAMES = ["AE", "AutoEncoder", "AEEncoder", "EEGAE"]

    ae_cls = None
    for name in AE_CANDIDATE_NAMES:
        ae_cls = globals().get(name, None)
        if ae_cls is not None:
            break

    if ae_cls is None:
        print("[WARN] 未找到 AE 类，跳过 AE 编码。")
        return None

    try:
        ae = ae_cls(in_ch=ae_in_ch, latent_ch=ae_latent_ch)  # 你的 AE 构造签名如不同，请在此调整
    except TypeError:
        try:
            ae = ae_cls(ae_in_ch, ae_latent_ch)
        except Exception as e:
            print(f"[WARN] AE 构造失败：{e}；跳过 AE 编码。")
            return None

    try:
        ckpt = torch.load(ae_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ae.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            ae.load_state_dict(ckpt, strict=False)
        print(f"[INFO] AE 权重已加载：{ae_ckpt}")
    except Exception as e:
        print(f"[WARN] AE 权重加载失败：{e}；继续无 AE。")
        return None

    ae.to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def train(train_loader, val_loader, device, val_fold,
          use_ae_encoder=False, ae_ckpt="./ae_ckpts/ae_best.pth", ae_in_ch=3, ae_latent_ch=3):
    """
    把 AE.encode 与 WT 都放到 GPU；Dataset 仅返回原始张量。
    """

    # —— cuDNN 性能设置（需要可复现则关掉 benchmark）
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 1) WT 放到 GPU（你原本就这么做）
    wt_runtime = WaveletTransform(wave="coif1", device=device.type).to(device)

    # 2) AE（只在循环里做 encode）
    ae = _build_ae_if_needed(use_ae_encoder, device, ae_ckpt, ae_in_ch, ae_latent_ch)

    # 3) 其余模块定义（与你原来的保持一致）
    def _mk_proj():
        return SimpleProjector(
            in_channels=3, spatial_only=True,
            clo_depth=1, clo_heads=3, clo_dw_kernel=3, clo_window=7
        ).to(device)

    # 四个子带 × 三个 stage
    conv_low_1, conv_hh_1, conv_hv_1, conv_hd_1 = _mk_proj(), _mk_proj(), _mk_proj(), _mk_proj()
    conv_low_M, conv_hh_M, conv_hv_M, conv_hd_M = _mk_proj(), _mk_proj(), _mk_proj(), _mk_proj()
    conv_low_2, conv_hh_2, conv_hv_2, conv_hd_2 = _mk_proj(), _mk_proj(), _mk_proj(), _mk_proj()

    high_gate_map = HighGateMap(c_in=3).to(device)
    alpha_hl      = AlphaHL(c_in=3).to(device)
    resfuser      = ResidualFuseMap(d_vec=3).to(device)
    classifier    = ClassifierFrom3(num_classes=NUM_CLASSES, hidden=32, dropout=DROPOUT).to(device)

    hlf_seq_loss1 = HLF_RelationLoss_BP(temperature=0.15, pool_hw=(4,4), proj_hidden=64, proj_out=32, p_drop=0.2).to(device)
    hlf_seq_lossM = HLF_RelationLoss_BP(temperature=0.15, pool_hw=(4,4), proj_hidden=64, proj_out=32, p_drop=0.2).to(device)
    hlf_seq_loss2 = HLF_RelationLoss_BP(temperature=0.15, pool_hw=(4,4), proj_hidden=64, proj_out=32, p_drop=0.2).to(device)

    def _pg(m):
        return [{"params": [p for p in m.parameters() if p.requires_grad], "weight_decay": WEIGHT_DECAY}]
    optim_params = []
    def _add(m, lr):
        for g in _pg(m):
            g["lr"] = lr; optim_params.append(g)

    for m in [conv_low_1, conv_hh_1, conv_hv_1, conv_hd_1,
              conv_low_M, conv_hh_M, conv_hv_M, conv_hd_M,
              conv_low_2, conv_hh_2, conv_hv_2, conv_hd_2]:
        _add(m, BACKBONE_LR)
    for m in [high_gate_map, alpha_hl, resfuser, classifier,
              hlf_seq_loss1, hlf_seq_lossM, hlf_seq_loss2]:
        _add(m, HEAD_LR)

    optimizer  = torch.optim.AdamW(optim_params)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    ema_modules = {
        "low1": conv_low_1, "hh1": conv_hh_1, "hv1": conv_hv_1, "hd1": conv_hd_1,
        "lowM": conv_low_M, "hhM": conv_hh_M, "hvM": conv_hv_M, "hdM": conv_hd_M,
        "low2": conv_low_2, "hh2": conv_hh_2, "hv2": conv_hv_2, "hd2": conv_hd_2,
        "hlf1": hlf_seq_loss1, "hlfM": hlf_seq_lossM, "hlf2": hlf_seq_loss2,
        "gate": high_gate_map, "alpha": alpha_hl, "cls": classifier, "resf": resfuser
    }
    ema = EMA(ema_modules, decay=EMA_DECAY)

    criterion = nn.CrossEntropyLoss().to(device)
    def _prep_y(y): return y.long()

    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

    def _search_best_thr(y_true_np, y_prob_np):
        best_thr, best_bal, best_f1, best_acc, best_pos = 0.5, 0.0, 0.0, 0.0, 0.0
        for thr in np.linspace(0.2, 0.8, 101):
            pred = (y_prob_np >= thr).astype(int)
            bal  = balanced_accuracy_score(y_true_np, pred)
            f1   = f1_score(y_true_np, pred, zero_division=0)
            acc  = (pred == y_true_np).mean()
            posr = pred.mean()
            if bal > best_bal:
                best_bal, best_thr, best_f1, best_acc, best_pos = bal, thr, f1, acc, posr
        return best_thr, best_bal, best_f1, best_acc, best_pos

    best_val = -1.0
    best_pack = {"balacc": 0.0, "f1": 0.0, "auc": 0.5, "epoch": 0, "thr": 0.5}

    # -----------------------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        # ============== Train ==============
        for m in [conv_low_1, conv_hh_1, conv_hv_1, conv_hd_1,
                  conv_low_M, conv_hh_M, conv_hv_M, conv_hd_M,
                  conv_low_2, conv_hh_2, conv_hv_2, conv_hd_2,
                  high_gate_map, alpha_hl, resfuser, classifier,
                  hlf_seq_loss1, hlf_seq_lossM, hlf_seq_loss2]:
            m.train()

        total, total_cl, total_cls = 0, 0.0, 0.0
        y_prob_all, y_true_all = [], []
        train_loss, n_train = 0.0, 0

        prof = StepProfiler(device, name="train", max_batches=100)
        it = iter(train_loader)
        for step in range(len(train_loader)):
            try:
                batch, dt_fetch = timed_fetch(it)
            except StopIteration:
                break
            prof.add_fetch(dt_fetch)

            with prof._evt() as start_end:
                # 1) H2D 复制（pin_memory + non_blocking 能加速）
                x = batch["raw"].to(device, non_blocking=True).float()
                y = batch["label"].to(device, non_blocking=True).long()

                # 2) （可选）AE 编码：在 GPU 上做
                if ae is not None and hasattr(ae, "encode"):
                    with torch.no_grad():
                        x = ae.encode(x)  # x 仍在 GPU 上

                _, _, dt_to = start_end()
            prof.add_to(dt_to)

            optimizer.zero_grad(set_to_none=True)

            with prof._evt() as start_end:
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    # -------- Stage-1 ----------
                    y0, cl1 = stage_spatial_fusion_iwt(
                        x_in=x, wt=wt_runtime,
                        conv_low=conv_low_1, conv_hh=conv_hh_1, conv_hv=conv_hv_1, conv_hd=conv_hd_1,
                        high_gate_map=high_gate_map, alpha_hl=alpha_hl,
                        hlf_seq_loss=hlf_seq_loss1, resfuser=resfuser,
                        resizer=None, scale=STAGE1_SCALE
                    )
                    # -------- Stage-M ----------
                    y1, clM = stage_spatial_fusion_iwt(
                        x_in=y0, wt=wt_runtime,
                        conv_low=conv_low_M, conv_hh=conv_hh_M, conv_hv=conv_hv_M, conv_hd=conv_hd_M,
                        high_gate_map=high_gate_map, alpha_hl=alpha_hl,
                        hlf_seq_loss=hlf_seq_lossM, resfuser=resfuser,
                        resizer=None, scale=STAGEM_SCALE
                    )
                    # -------- Stage-2 ----------
                    y1_b, _, _ = pad_to_even(y1)
                    low2, hh2, hv2, hd2 = wt_runtime(y1_b)
                    low2_map, feat_low2 = conv_low_2(low2, return_feat=True, subband_size=low2.shape[-2:])
                    hh2_map,  feat_hh2  = conv_hh_2(hh2,  return_feat=True, subband_size=hh2.shape[-2:])
                    hv2_map,  feat_hv2  = conv_hv_2(hv2,  return_feat=True, subband_size=hv2.shape[-2:])
                    hd2_map,  feat_hd2  = conv_hd_2(hd2,  return_feat=True, subband_size=hd2.shape[-2:])
                    cl2 = hlf_seq_loss2(feat_hh2.float(), feat_hv2.float(), feat_hd2.float(), feat_low2.float())

                    w3_2   = high_gate_map(hh2_map, hv2_map, hd2_map)
                    high2  = (torch.stack([hh2_map, hv2_map, hd2_map], dim=1) * w3_2.unsqueeze(2)).sum(dim=1)
                    alpha2 = alpha_hl(low2_map, high2)
                    low2_f = (1.0 - alpha2) * low2_map + alpha2 * high2
                    vec2   = low2_f.mean(dim=(2,3))
                    logits = classifier(vec2)
                    cls_loss = criterion(logits, y)

                    cl_loss_total = W_CL1 * cl1 + W_CLM * clM + W_CL2 * cl2
                    cl_loss_total = CL_SCALE * cl_loss_total
                    loss = W_CL * cl_loss_total + W_CLS * cls_loss
                _, _, dt_fwd = start_end()
            prof.add_fwd(dt_fwd)

            with prof._evt() as start_end:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                ema.update(ema_modules)
                _, _, dt_bwd = start_end()
            prof.add_bwd(dt_bwd)
            prof.step_done()
            # 把优化器步进时间并到 backward 段上了；如需分开，可再加一段 profile

            bs = x.size(0)
            total      += bs
            total_cl   += float(cl_loss_total.detach()) * bs
            total_cls  += float(cls_loss.detach()) * bs
            train_loss += float(loss.detach()) * bs; n_train += bs

            with torch.no_grad():
                prob1 = F.softmax(logits, dim=1)[:, 1]
                y_prob_all.append(prob1.detach().cpu())      # ← 统一放 CPU，避免 cat 报设备不一致
                y_true_all.append(y.detach().cpu())

        prof.summary()

        train_loss /= max(1, n_train)
        y_prob_all = torch.cat(y_prob_all).numpy()
        y_true_all = torch.cat(y_true_all).numpy()
        pred_tr = (y_prob_all >= 0.5).astype(int)
        train_acc = float((pred_tr == y_true_all).mean() * 100.0)
        train_pos_rate = float(pred_tr.mean() * 100.0)

        # ============== Val ==============
        for m in [conv_low_1, conv_hh_1, conv_hv_1, conv_hd_1,
                  conv_low_M, conv_hh_M, conv_hv_M, conv_hd_M,
                  conv_low_2, conv_hh_2, conv_hv_2, conv_hd_2,
                  high_gate_map, alpha_hl, resfuser, classifier,
                  hlf_seq_loss1, hlf_seq_lossM, hlf_seq_loss2]:
            m.eval()

        val_total, val_cl, val_cls = 0, 0.0, 0.0
        y_prob_v, y_true_v = [], []
        val_sids, val_labels_epoch = [], []

        with ema.apply(ema_modules):
            with torch.no_grad():
                prof_val = StepProfiler(device, name="val", max_batches=50)
                it_val = iter(val_loader)
                for step in range(len(val_loader)):
                    try:
                        batch, dt_fetch = timed_fetch(it_val)
                    except StopIteration:
                        break
                    prof_val.add_fetch(dt_fetch)

                    with prof_val._evt() as start_end:
                        # 1) H2D 复制（pin_memory + non_blocking 能加速）
                        x = batch["raw"].to(device, non_blocking=True).float()
                        y = batch["label"].to(device, non_blocking=True).long()

                        # 2) （可选）AE 编码：在 GPU 上做
                        if ae is not None and hasattr(ae, "encode"):
                            with torch.no_grad():
                                x = ae.encode(x)  # x 仍在 GPU 上

                        _, _, dt_to = start_end()
                    prof_val.add_to(dt_to)

                    with prof_val._evt() as start_end:
                        # --- 与训练相同的前向 ---
                        y0, cl1 = stage_spatial_fusion_iwt(
                            x_in=x, wt=wt_runtime,
                            conv_low=conv_low_1, conv_hh=conv_hh_1, conv_hv=conv_hv_1, conv_hd=conv_hd_1,
                            high_gate_map=high_gate_map, alpha_hl=alpha_hl,
                            hlf_seq_loss=hlf_seq_loss1, resfuser=resfuser,
                            resizer=None, scale=STAGE1_SCALE
                        )
                        y1, clM = stage_spatial_fusion_iwt(
                            x_in=y0, wt=wt_runtime,
                            conv_low=conv_low_M, conv_hh=conv_hh_M, conv_hv=conv_hv_M, conv_hd=conv_hd_M,
                            high_gate_map=high_gate_map, alpha_hl=alpha_hl,
                            hlf_seq_loss=hlf_seq_lossM, resfuser=resfuser,
                            resizer=None, scale=STAGEM_SCALE
                        )
                        y1_b, _, _ = pad_to_even(y1)
                        low2, hh2, hv2, hd2 = wt_runtime(y1_b)
                        low2_map, feat_low2 = conv_low_2(low2, return_feat=True, subband_size=low2.shape[-2:])
                        hh2_map,  feat_hh2  = conv_hh_2(hh2,  return_feat=True, subband_size=hh2.shape[-2:])
                        hv2_map,  feat_hv2  = conv_hv_2(hv2,  return_feat=True, subband_size=hv2.shape[-2:])
                        hd2_map,  feat_hd2  = conv_hd_2(hd2,  return_feat=True, subband_size=hd2.shape[-2:])
                        cl2 = hlf_seq_loss2(feat_hh2.float(), feat_hv2.float(), feat_hd2.float(), feat_low2.float())

                        w3_2   = high_gate_map(hh2_map, hv2_map, hd2_map)
                        high2  = (torch.stack([hh2_map, hv2_map, hd2_map], dim=1) * w3_2.unsqueeze(2)).sum(dim=1)
                        alpha2 = alpha_hl(low2_map, high2)
                        low2_f = (1.0 - alpha2) * low2_map + alpha2 * high2
                        vec2   = low2_f.mean(dim=(2,3))
                        logits = classifier(vec2)
                        cls_loss = criterion(logits, y)

                        cl_loss_total = W_CL1 * cl1 + W_CLM * clM + W_CL2 * cl2
                        cl_loss_total = CL_SCALE * cl_loss_total
                        loss = W_CL * cl_loss_total + W_CLS * cls_loss
                        _, _, dt_fwd = start_end()
                    prof_val.add_fwd(dt_fwd)
                    prof_val.step_done()

                    bs = x.size(0)
                    val_total += bs
                    val_cl    += float(cl_loss_total) * bs
                    val_cls   += float(cls_loss) * bs

                    prob1 = F.softmax(logits, dim=1)[:, 1]
                    y_prob_v.append(prob1.detach().cpu())
                    y_true_v.append(y.detach().cpu())

                    sid_b = batch.get("sid", None)
                    if sid_b is None:
                        sid_b = torch.full_like(y, fill_value=-1)
                    val_sids.append(sid_b.detach().cpu())
                    val_labels_epoch.append(y.detach().cpu())

                prof_val.summary()

        # 验证指标
        y_prob_v = torch.cat(y_prob_v).numpy()
        y_true_v = torch.cat(y_true_v).numpy()
        try:
            auc_v = roc_auc_score(y_true_v, y_prob_v)
        except Exception:
            auc_v = 0.5
        best_thr, balacc_v, f1_v, val_acc_frac, pos_rate_at_thr = _search_best_thr(y_true_v, y_prob_v)
        val_acc = float(val_acc_frac * 100.0)

        # 学习率区间（打印信息用）
        lr_list = [pg["lr"] for pg in optimizer.param_groups]
        lr_min, lr_max = (min(lr_list), max(lr_list)) if lr_list else (0.0, 0.0)

        cm = confusion_matrix(y_true_v, (y_prob_v >= best_thr).astype(int))

        # 写出 per-subject 命中/失误
        sid_np = torch.cat(val_sids).numpy()
        y_np   = torch.cat(val_labels_epoch).numpy()
        pred_np = (y_prob_v >= best_thr).astype(int)
        per_subject = defaultdict(lambda: {"label": None, "correct": 0, "wrong": 0})
        for s, y_true_i, y_pred_i in zip(sid_np, y_np, pred_np):
            s = int(s)
            lab = "MDD" if int(y_true_i) == 1 else "HC"
            if per_subject[s]["label"] is None:
                per_subject[s]["label"] = lab
            if int(y_pred_i) == int(y_true_i):
                per_subject[s]["correct"] += 1
            else:
                per_subject[s]["wrong"] += 1
        result_dir = os.path.join("result", f"fold{val_fold}")
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, f"val_subject_hitmiss_e{epoch:03d}.json")
        items = [{"subject_id": int(s), **v}
                 for s, v in sorted(per_subject.items(), key=lambda kv: (1_000_000_000 if kv[0] < 0 else kv[0]))]
        payload = {
            "epoch": int(epoch),
            "best_thr": float(best_thr),
            "val_bal_acc": float(balacc_v * 100.0),
            "val_acc": float(val_acc),
            "f1": float(f1_v),
            "auc": float(auc_v),
            "fold": int(val_fold),
            "subjects": items
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[VAL] 每受试者命中/失误统计已写入：{out_path}")

        # 训练/验证摘要
        print(f"Epoch {epoch:3d}/{EPOCHS}")
        print(f"  Train: ConLoss={total_cl/max(1,total):.4f}, ClasLoss={total_cls/max(1,total):.4f}  "
              f"Acc={train_acc:.2f}%  pos_rate={train_pos_rate:.2f}  (lr=[{lr_min:.2e}, {lr_max:.2e}])")
        print(f"  Val:   ConLoss={val_cl/max(1,val_total):.4f}, ClasLoss={val_cls/max(1,val_total):.4f}, "
              f"Acc={val_acc:.2f}%  BalAcc={balacc_v*100:.2f}%  F1={f1_v:.3f}  AUC={auc_v:.3f}  "
              f"thr*={best_thr:.2f}  pos_rate@thr*={pos_rate_at_thr*100:.2f}")
        print(f"  CM:\n{cm}")

        # 调度 & 保存 best
        scheduler.step(balacc_v)
        if balacc_v > best_val:
            best_val = balacc_v
            best_pack = {"balacc": balacc_v, "f1": f1_v, "auc": auc_v, "epoch": epoch, "thr": float(best_thr)}
            torch.save({
                "conv_low_1": conv_low_1.state_dict(), "conv_hh_1": conv_hh_1.state_dict(),
                "conv_hv_1": conv_hv_1.state_dict(),   "conv_hd_1": conv_hd_1.state_dict(),
                "conv_low_M": conv_low_M.state_dict(), "conv_hh_M": conv_hh_M.state_dict(),
                "conv_hv_M": conv_hv_M.state_dict(),   "conv_hd_M": conv_hd_M.state_dict(),
                "conv_low_2": conv_low_2.state_dict(), "conv_hh_2": conv_hh_2.state_dict(),
                "conv_hv_2": conv_hv_2.state_dict(),   "conv_hd_2": conv_hd_2.state_dict(),
                "high_gate_map": high_gate_map.state_dict(),
                "alpha_hl": alpha_hl.state_dict(),
                "resfuser": resfuser.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.shadow
            }, "best_ckpt_spatial_only.pth")

    # 训练结束返回
    return {
        "best_epoch": int(best_pack["epoch"]),
        "best_bal_acc": float(best_pack["balacc"] * 100.0),
        "val_acc": float(val_acc),
        "f1": float(best_pack["f1"]),
        "auc": float(best_pack["auc"]),
        "best_thr": float(best_pack["thr"]),
    }


def run_single_fold(slices_root="./slices", val_fold=1, folds=5, epochs=EPOCHS, batch_size=64,
                    use_ae_encoder=False, ae_ckpt="./ae_ckpts/ae_best.pth",
                    ae_in_ch=3, ae_latent_ch=3, ae_device="cuda"):
    """
    overlap 折作为训练集，nonoverlap 折作为验证集。
    ★ 本函数里 Dataset 不做 AE/WT 重活；AE/WT 都在 train() 内的 GPU 上做。
    """

    def has_npz(dir_):
        return os.path.isdir(dir_) and any(f.endswith(".npz") for f in os.listdir(dir_))

    if not (1 <= int(val_fold) <= int(folds)):
        raise ValueError(f"--val_fold 必须在 1..{folds}，当前 {val_fold}")

    # 训练折（overlap）
    train_dirs = []
    for j in range(1, folds + 1):
        if j == val_fold: continue
        d = os.path.join(slices_root, f"fold{j}_overlap", "shards")
        if has_npz(d):
            train_dirs.append(d)
        else:
            print(f"[警告] 跳过不存在/无 .npz 的目录：{d}")

    # 验证折（nonoverlap）
    val_dir = os.path.join(slices_root, f"fold{val_fold}_nonoverlap", "shards")
    if not train_dirs:
        raise RuntimeError("没有可用训练分片（未发现 .npz）")
    if not has_npz(val_dir):
        raise RuntimeError(f"验证目录无 .npz：{val_dir}")

    result_dir = os.path.join("result", f"fold{val_fold}")
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(result_dir, f"console_{stamp}.log")
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n========== Single Run（验证=Fold {val_fold} nonoverlap；训练=其余折 overlap）==========")
    for td in train_dirs: print("  -", td)
    print("Val:", val_dir)
    print(f"Result dir: {result_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    train_datasets = [
        EEGFeatureDataset(
            feature_path=td,
            augment=True
        ) for td in train_dirs
    ]
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    # 验证集（nonoverlap）
    val_ds = EEGFeatureDataset(
        feature_path=val_dir,
        augment=False
    )


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=max(4, (os.cpu_count() or 8)//2),
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(2, (os.cpu_count() or 8)//4),
        pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2
    )

    # 交给 train()：在 GPU 里创建 WT & AE
    metrics = train(
        train_loader=train_loader, val_loader=val_loader, device=device, val_fold=val_fold,
        use_ae_encoder=use_ae_encoder,  # 若想启用AE，传 True（但Dataset仍然不做）
        ae_ckpt=ae_ckpt, ae_in_ch=ae_in_ch, ae_latent_ch=ae_latent_ch
    )

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