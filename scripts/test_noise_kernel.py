"""Check of fast_data.noise_field (GPU if available, else CPU):
1. rbf path reproduces the pre-change implementation exactly (same generator seed);
2. unit marginal variance for rbf and powerlaw;
3. powerlaw: radially averaged spectrum of a field on a 512x512 km, 1 km grid has slope ~ -beta;
4. chunked evaluation = unchunked."""
import math

import numpy as np
import torch

from fast_data import KM_PER_DEG, noise_field, pick_device

dev = pick_device("auto")


def old_noise_field(lat, lon, n_out, n_features=64, length_scale_deg=0.5, generator=None):
    B, N = lat.shape
    freq = torch.randn(B, n_features, 2, device=dev, generator=generator) / length_scale_deg
    phase = torch.rand(B, n_features, n_out, device=dev, generator=generator) * (2 * math.pi)
    coeff = torch.randn(B, n_features, n_out, device=dev, generator=generator)
    coords = torch.stack([lat, lon * math.cos(math.radians(64.5))], dim=-1)
    ang = 2 * math.pi * torch.bmm(coords, freq.transpose(1, 2))
    a, b = coeff * torch.cos(phase), coeff * torch.sin(phase)
    return (torch.bmm(torch.cos(ang), a) - torch.bmm(torch.sin(ang), b)) * math.sqrt(2.0 / n_features)


# 512x512 km grid at 1 km around (64.5N, 20E), in degrees (lon scaled back by cos 64.5)
L = 512
y = (torch.arange(L, device=dev, dtype=torch.float64) - L / 2) / KM_PER_DEG
x = y / math.cos(math.radians(64.5))
lat = (64.5 + y[:, None].expand(L, L)).reshape(1, -1).float()
lon = (20.0 + x[None, :].expand(L, L)).reshape(1, -1).float()

g1, g2 = torch.Generator(device=dev).manual_seed(0), torch.Generator(device=dev).manual_seed(0)
new = noise_field(lat, lon, 3, generator=g1)
old = old_noise_field(lat, lon, 3, generator=g2)
print(f"1. rbf new vs old max |diff| = {float((new - old).abs().max()):.2e} (expect ~0)")


def radial_psd(f):
    f = f - f.mean()
    w = torch.hann_window(L, device=dev, dtype=torch.float64)
    P = torch.fft.fftshift(torch.abs(torch.fft.fft2(f * w[:, None] * w[None, :])) ** 2)
    k = torch.arange(L, device=dev) - L // 2
    kr = torch.sqrt(k[:, None].double() ** 2 + k[None, :].double() ** 2)
    nb = L // 2
    b = torch.clamp((kr / (L // 2) * nb).long(), max=nb)
    s = torch.zeros(nb + 1, device=dev, dtype=torch.float64).index_add_(0, b.flatten(), P.flatten())
    n = torch.zeros(nb + 1, device=dev, dtype=torch.float64).index_add_(0, b.flatten(), torch.ones_like(P).flatten())
    return (s / n.clamp(min=1))[:nb].cpu().numpy()


kk = np.arange(L // 2) / L  # cycles/km
for kern, kw in [("rbf F=64 ls=0.5", dict(kernel="rbf")),
                 ("powerlaw beta=2.4 F=1024", dict(kernel="powerlaw", beta=2.4, n_features=1024)),
                 ("powerlaw beta=3.0 F=1024", dict(kernel="powerlaw", beta=3.0, n_features=1024))]:
    vals, psd = [], 0
    for r in range(16):
        f = noise_field(lat, lon, 1, **kw)[0, :, 0].double()
        vals.append(float(f.var()))
        psd = psd + radial_psd(f.reshape(L, L))
    m = (kk >= 1 / 100) & (kk <= 1 / 5)
    slope = np.polyfit(np.log(kk[m]), np.log(psd[m]), 1)[0]
    print(f"2/3. {kern}: variance over 512 km box = {np.mean(vals):.2f} (field var 1, box var <= 1) | "
          f"PSD slope 5-100 km = {slope:.2f}")

g1, g2 = torch.Generator(device=dev).manual_seed(1), torch.Generator(device=dev).manual_seed(1)
a = noise_field(lat, lon, 3, kernel="powerlaw", n_features=1024, generator=g1)
b = noise_field(lat, lon, 3, kernel="powerlaw", n_features=1024, generator=g2, chunk=10000)
print(f"4. chunked vs unchunked max |diff| = {float((a - b).abs().max()):.2e} (expect ~0)")
