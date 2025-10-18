# contrast.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

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
    Ach = A.flatten(2); Bch = B.flatten(2)
    num = (Ach * Bch).sum(dim=2)
    den = (Ach.norm(dim=2) * Bch.norm(dim=2)).clamp_min(1e-8)
    rel_ch = (num / den)
    Ab = F.normalize(A, dim=1); Bb = F.normalize(B, dim=1)
    rel_sp = (Ab * Bb).sum(dim=1, keepdim=True)
    ph, pw = pool_hw
    rel_sp = F.adaptive_avg_pool2d(rel_sp, output_size=(ph, pw)).flatten(1)
    r = torch.cat([rel_ch, rel_sp], dim=1)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r

class HLF_RelationLoss_BP(nn.Module):
    """
    与你现有 contras.py 中的实现保持完全一致的接口：
    - ._ensure_projector(in_dim, device)
    - forward(hh, hv, hd, low) -> loss
    - 成员：self.proj (nn.Module), self.ntxent
    """
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
        if self.training:
            r = r + 1e-4 * torch.randn_like(r)  # 轻微抖动避免退化
        z1 = self.proj(r); z2 = self.proj(r)
        z1 = torch.nan_to_num(z1); z2 = torch.nan_to_num(z2)
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
