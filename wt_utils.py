
from config import *
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
def pad_to_even(x: torch.Tensor):
    _, _, H, W = x.shape
    ph = 1 if (H % 2 == 1) else 0
    pw = 1 if (W % 2 == 1) else 0
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, ph, pw