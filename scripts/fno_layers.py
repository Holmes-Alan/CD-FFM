"""
fno_layers.py
=============
FNO building blocks: spectral convolution, the flat FNOEncoder (original
single-resolution design), and the shared SpectralConv2d used by
fno_unet.py's U-Net variant. Split out from model.py so fno_unet.py can
import SpectralConv2d without a circular import (model.py -> fno_unet.py
-> model.py).
"""
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weight1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B,C,H,W//2+1) complex
        Wf = x_ft.shape[-1]
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, Wf)
        out_ft = torch.zeros(B, self.out_ch, H, Wf, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weight2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOLayer(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spectral(x) + self.bypass(x)
        return self.act(self.norm(y))


class FNOEncoder(nn.Module):
    """Flat (non-U-Net) FNO encoder -- the original design, kept available
    as backbone="fno_flat" for reference/ablation now that the primary
    "fno" backbone is FNOUNet (fno_unet.py), matched structurally to the
    real Flower U-Net. in_ch -> feat_dim over a FIXED small grid (ERA5's
    Nordic crop). modes1/modes2 chosen relative to that fixed grid size."""

    def __init__(self, in_ch, width=64, feat_dim=128, depth=4, modes1=12, modes2=16):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.layers = nn.ModuleList([FNOLayer(width, modes1, modes2) for _ in range(depth)])
        self.proj = nn.Conv2d(width, feat_dim, 1)

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers:
            x = x + layer(x)  # residual
        return self.proj(x)
