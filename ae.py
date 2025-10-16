from config import *
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