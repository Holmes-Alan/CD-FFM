"""
diffusion_baseline.py
======================
CorrDiff-style generative baseline (Mardani et al.): a grid-based, two-stage model on the 1 km NGCD grid.
  stage "reg": UNet regression  mean = R(c)                       (masked MSE in z-space)
  stage "edm": EDM diffusion on the residual r = x - mean, conditioned on [c, mean]
               (Karras et al. 2022 preconditioning, log-normal sigma, Heun sampler)
Same inputs as the functional model, but as image channels at 1 km: ERA5 window (4 ch, bilinear),
the 17 static climatology covariates, day of year (cos, sin), normalized lat/lon, land mask. Same
training years, validation year (checkpoint selection on a fixed validation-patch loss) and test year.

Training on random 128x128 km patches cut on the GPU from GPUData (no DataLoader); inference on the full
2020x1550 domain at once (fully convolutional UNet). Evaluation goes through fast_eval.py, which calls
sample_grid_day() for runs with objective "unet_reg" / "edm".

  python3 diffusion_baseline.py --name w6_unet_reg --stage reg ...
  python3 diffusion_baseline.py --name w6_corrdiff --stage edm --reg-run w6_unet_reg ...
"""
import argparse
import copy
import json
import math
import os
import re
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, IterableDataset

from fast_data import DATA_DIR, GPUData, doy_features, pick_device, static_features
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

RUNS_ROOT = CDFFM_ROOT + "/runs"
GRID_OBJECTIVES = ("unet_reg", "edm")
COND_CH = 4 + 17 + 2 + 2 + 1  # era5, static, doy, lat/lon, mask
SIGMA_DATA = 0.5
warnings.filterwarnings("ignore", message=".*does not have many workers.*")


# --------------------------------------------------------------------------- UNet

class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm, self.qkv, self.out, self.heads = nn.GroupNorm(32, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).permute(1, 0, 2, 4, 3)
        o = F.scaled_dot_product_attention(q, k, v)  # (B,heads,HW,C/heads)
        return x + self.out(o.permute(0, 1, 3, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    """Compact ADM/EDM-style UNet: channels ch*mult per level, n_res ResBlocks per level with
    FiLM noise embedding, optional self-attention at the coarsest level. Input sides must be divisible by
    2**(len(mult)-1)."""

    def __init__(self, in_ch, out_ch, ch=64, mult=(1, 2, 4, 4), n_res=2, emb_dim=256, attn=False):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(64, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        chans = [ch * m for m in mult]
        self.down, self.ds = nn.ModuleList(), nn.ModuleList()
        skips, c = [ch], ch
        for li, co in enumerate(chans):
            for _ in range(n_res):
                self.down.append(ResBlock(c, co, emb_dim))
                c = co
                skips.append(c)
            self.ds.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if li < len(chans) - 1 else nn.Identity())
            if li < len(chans) - 1:
                skips.append(c)
        # attention off by default: a global attention trained on 16x16 bottleneck tokens would see ~50k tokens
        # on the full domain (train/test mismatch, O(n^2) memory); the attention-free UNet is fully convolutional
        self.mid1, self.mid2 = ResBlock(c, c, emb_dim), ResBlock(c, c, emb_dim)
        self.mid_attn = Attn(c) if attn else nn.Identity()
        self.up, self.us = nn.ModuleList(), nn.ModuleList()
        for li, co in reversed(list(enumerate(chans))):
            for _ in range(n_res + 1):
                self.up.append(ResBlock(c + skips.pop(), co, emb_dim))
                c = co
            self.us.append(nn.Conv2d(c, c, 3, padding=1) if li > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.n_res, self.n_levels = n_res, len(chans)

    @staticmethod
    def _fourier(c_noise):  # (B,) -> (B,64)
        f = torch.exp(-math.log(10000) * torch.arange(32, device=c_noise.device) / 32)
        a = c_noise[:, None] * f[None] * 100.0
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, x, c_noise):
        emb = self.emb(self._fourier(c_noise))
        h = self.inp(x)
        hs = [h]
        k = 0
        for li in range(self.n_levels):
            for _ in range(self.n_res):
                h = self.down[k](h, emb)
                k += 1
                hs.append(h)
            if li < self.n_levels - 1:
                h = self.ds[li](h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        k = 0
        for ui in range(self.n_levels):
            for _ in range(self.n_res + 1):
                h = self.up[k](torch.cat([h, hs.pop()], dim=1), emb)
                k += 1
            if ui < self.n_levels - 1:
                h = self.us[ui](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


def edm_denoise(net, x_noisy, sigma, cond):
    """Karras et al. preconditioning. x_noisy: (B,3,H,W) noisy residual; sigma: (B,)."""
    s = sigma[:, None, None, None]
    c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    c_out = s * SIGMA_DATA / torch.sqrt(s ** 2 + SIGMA_DATA ** 2)
    c_in = 1.0 / torch.sqrt(s ** 2 + SIGMA_DATA ** 2)
    f = net(torch.cat([c_in * x_noisy, cond], dim=1), torch.log(sigma) / 4.0)
    return c_skip * x_noisy + c_out * f


@torch.no_grad()
def patched(fn, inp, mask, P=128, stride=112, batch=32):
    """Apply fn to overlapping P x P tiles of inp (1, C, H, W) and blend the outputs (1, 3, H, W) with
    linear ramps over the overlaps -- the inference counterpart of patch training. The UNet's GroupNorm
    normalizes over the whole input, so applying it to the full domain (40 % empty sea) instead of to
    training-sized patches changes its statistics; tiling keeps inference identical to training.
    Tiles without land are skipped."""
    _, C, H, W = inp.shape
    starts = lambda L: sorted(set(list(range(0, L - P + 1, stride)) + [L - P]))
    ov = P - stride
    k = torch.arange(P, device=inp.device, dtype=inp.dtype)
    w1 = torch.clamp(torch.minimum((k + 1) / ov, (P - k) / ov), max=1.0)
    w2 = (w1[:, None] * w1[None, :])[None, None]
    tiles = [(r, c) for r in starts(H) for c in starts(W) if mask[0, 0, r:r + P, c:c + P].any()]
    out = torch.zeros(1, 3, H, W, device=inp.device, dtype=inp.dtype)
    wsum = torch.zeros(1, 1, H, W, device=inp.device, dtype=inp.dtype)
    for b0 in range(0, len(tiles), batch):
        tb = tiles[b0:b0 + batch]
        y = fn(torch.cat([inp[:, :, r:r + P, c:c + P] for r, c in tb]))
        for i, (r, c) in enumerate(tb):
            out[:, :, r:r + P, c:c + P] += y[i:i + 1] * w2
            wsum[:, :, r:r + P, c:c + P] += w2
    return out / wsum.clamp(min=1e-8)


def edm_sample_patched(net, cond, mask, shape, steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0, generator=None,
                       P=128, stride=112):
    """EDM Heun sampler on the full domain, with every denoiser call applied tile-wise (patched)."""
    dev = cond.device
    i = torch.arange(steps, device=dev, dtype=torch.float64)
    sig = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev, dtype=torch.float64)])
    x = torch.randn(shape, device=dev, generator=generator) * sig[0].float()

    def den(xx, s):
        f = lambda t: edm_denoise(net, t[:, :3], s.expand(t.shape[0]), t[:, 3:])
        return patched(f, torch.cat([xx, cond], dim=1), mask, P, stride)

    for k in range(steps):
        s0, s1 = sig[k].float(), sig[k + 1].float()
        d0 = (x - den(x, s0)) / s0
        x1 = x + (s1 - s0) * d0
        if s1 > 0:
            d1 = (x1 - den(x1, s1)) / s1
            x1 = x + (s1 - s0) * 0.5 * (d0 + d1)
        x = x1
    return x


def edm_sample(net, cond, shape, steps=18, sigma_min=0.002, sigma_max=80.0, rho=7.0, generator=None):
    """Deterministic EDM Heun sampler (2*steps-1 network evaluations)."""
    dev = cond.device
    i = torch.arange(steps, device=dev, dtype=torch.float64)
    sig = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev, dtype=torch.float64)])
    x = torch.randn(shape, device=dev, generator=generator) * sig[0].float()
    for k in range(steps):
        s0, s1 = sig[k].float(), sig[k + 1].float()
        d0 = (x - edm_denoise(net, x, s0.expand(shape[0]), cond)) / s0
        x1 = x + (s1 - s0) * d0
        if s1 > 0:
            d1 = (x1 - edm_denoise(net, x1, s1.expand(shape[0]), cond)) / s1
            x1 = x + (s1 - s0) * 0.5 * (d0 + d1)
        x = x1
    return x


# --------------------------------------------------------------------------- grid data

class GridData:
    """Patches (B, C, P, P) cut on the GPU from a GPUData (temperature, static clim)."""

    def __init__(self, data, patch=128, min_valid=0.3, n_val_patches=8, seed=0):
        self.d, self.P, dev = data, patch, data.lat.device
        st = np.load(f"{DATA_DIR}/static.npz")
        rows, cols = st["valid_rows"], st["valid_cols"]
        self.H, self.W = [int(v) for v in st["grid_shape"]]
        idx = np.full((self.H, self.W), -1, dtype=np.int64)
        idx[rows, cols] = np.arange(len(rows))
        self.idx_map = torch.from_numpy(idx).to(dev)
        from fast_eval import bilinear_setup  # lazy import (fast_eval imports fast_train)
        self.bl = bilinear_setup(data.era5_lat, data.era5_lon, data.lat, data.lon, dev)
        self.latn = (data.lat - data.lat.mean()) / data.lat.std()
        self.lonn = (data.lon - data.lon.mean()) / data.lon.std()
        valid = (idx >= 0).astype(np.float32)
        cs = np.pad(valid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        org = [(r, c) for r in range(0, self.H - patch + 1, 8) for c in range(0, self.W - patch + 1, 8)
               if (cs[r + patch, c + patch] - cs[r, c + patch] - cs[r + patch, c] + cs[r, c]) / patch ** 2 >= min_valid]
        self.origins = torch.tensor(org, device=dev)
        off = torch.arange(patch, device=dev)
        self.off_r, self.off_c = off[:, None].expand(patch, patch), off[None, :].expand(patch, patch)
        # fixed validation patches: 12 validation dates x n_val_patches
        g = np.random.default_rng(seed)
        vm = np.load(f"{DATA_DIR}/year_{data.val_year}_meta.npz")
        vt = np.load(f"{DATA_DIR}/year_{data.val_year}_targets.npy", mmap_mode="r")
        vdays = np.linspace(14, len(vm["dates"]) - 16, 12).round().astype(int)
        self.val_targets = torch.from_numpy(np.stack([np.array(vt[i]) for i in vdays])).to(dev)  # (12,N,3) fp16
        self.val_era5 = torch.from_numpy(vm["era5"][vdays]).to(dev)
        self.val_doy = torch.from_numpy(vm["doy"][vdays].astype(np.int64)).to(dev)
        self.val_org = self.origins[torch.from_numpy(g.choice(len(org), size=12 * n_val_patches)).to(dev)]
        print(f"[GridData] {len(org)} patch origins (>= {min_valid:.0%} land), {12 * n_val_patches} val patches", flush=True)

    def cond_at(self, pix, era5, doy):
        """pix: (B,P,P) pixel indices (-1 = invalid); era5: (B,4,h,w); doy: (B,) -> cond (B,26,P,P), mask (B,1,P,P)."""
        mask = pix >= 0
        p = pix.clamp(min=0)
        i0, j0, wi, wj = (t[p] for t in self.bl)  # (B,P,P)
        b = torch.arange(pix.shape[0], device=pix.device)[:, None, None]
        e = lambda di, dj: era5[b, :, i0 + di, j0 + dj]  # (B,P,P,4)
        e5 = ((1 - wi) * (1 - wj))[..., None] * e(0, 0) + ((1 - wi) * wj)[..., None] * e(0, 1) \
            + (wi * (1 - wj))[..., None] * e(1, 0) + (wi * wj)[..., None] * e(1, 1)
        stat = self.d.static[p]  # (B,P,P,17)
        dy = doy_features(doy)[:, None, None, :].expand(-1, pix.shape[1], pix.shape[2], -1)
        ll = torch.stack([self.latn[p], self.lonn[p]], dim=-1)
        m = mask[..., None].float()
        c = torch.cat([e5 * m, stat * m, dy, ll * m, m], dim=-1)
        return c.permute(0, 3, 1, 2).contiguous(), mask[:, None].float()

    def patch_pix(self, org):
        return self.idx_map[org[:, 0, None, None] + self.off_r, org[:, 1, None, None] + self.off_c]

    def batch(self, B):
        d = self.d
        day = torch.randint(0, d.n_days, (B,), device=d.lat.device)
        org = self.origins[torch.randint(0, len(self.origins), (B,), device=d.lat.device)]
        pix = self.patch_pix(org)
        cond, mask = self.cond_at(pix, d.era5[day], d.doy[day])
        x = d.gather_targets(day, pix.clamp(min=0)).float()  # (B,P,P,3)
        ok = torch.isfinite(x).all(-1, keepdim=True) & (pix >= 0)[..., None]
        x = torch.where(ok, x, torch.zeros_like(x)).permute(0, 3, 1, 2)
        return x, cond, ok.permute(0, 3, 1, 2).float()

    def val_batches(self):
        n = self.val_org.shape[0] // 12
        for k in range(12):
            org = self.val_org[k * n:(k + 1) * n]
            pix = self.patch_pix(org)
            day = torch.full((n,), k, device=pix.device)
            cond, _ = self.cond_at(pix, self.val_era5[day], self.val_doy[day])
            x = self.val_targets[k][pix.clamp(min=0)].float()
            ok = torch.isfinite(x).all(-1, keepdim=True) & (pix >= 0)[..., None]
            yield torch.where(ok, x, torch.zeros_like(x)).permute(0, 3, 1, 2), cond, ok.permute(0, 3, 1, 2).float()


# --------------------------------------------------------------------------- models


def saved_steps(ckdir):
    """{step: path} of the periodic checkpoints in ckdir. Lightning may add a '-v1' suffix when it saves
    the same step twice (e.g. at the final step); the plain name is preferred."""
    out = {}
    for f_ in sorted(os.listdir(ckdir), key=len, reverse=True):  # longer (-vN) names first, plain overwrites
        m = re.match(r"stepstep=(\d+)(-v\d+)?\.ckpt$", f_)
        if m:
            out[int(m.group(1))] = f"{ckdir}/{f_}"
    return out

def build_net(args):
    in_ch = COND_CH if args.stage == "reg" else 3 + COND_CH + 3
    return UNet(in_ch, 3, ch=args.ch, mult=tuple(int(m) for m in args.mult.split(",")), n_res=args.n_res,
                attn=getattr(args, "attn", False))


def regress(net, cond):
    return net(cond, torch.zeros(cond.shape[0], device=cond.device))


def load_grid_run(name, device, which="best"):
    with open(f"{RUNS_ROOT}/{name}/args.json") as f:
        args = argparse.Namespace(**json.load(f))
    ckdir = f"{RUNS_ROOT}/{name}/ckpt"
    if which == "best":
        rows = [json.loads(l) for l in open(f"{RUNS_ROOT}/{name}/metrics.jsonl") if l.strip()]
        saved = saved_steps(ckdir)
        rows = [r for r in rows if r["step"] in saved]
        step = min(rows, key=lambda r: r["val_loss"])["step"]
        path = saved[step]
    else:
        path = f"{ckdir}/last.ckpt"
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_net(args).to(device)
    net.load_state_dict({k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")})
    net.eval()
    reg = None
    if args.stage == "edm":
        _, reg, _, _ = load_grid_run(args.reg_run, device, "best")
    return args, net, reg, int(ck.get("global_step", -1))


class FullDomain:
    """Precomputed full-domain inputs for sampling (padded to a multiple of the UNet stride)."""

    def __init__(self, data_like, lat, lon, static, era5_lat, era5_lon, stride=8):
        from fast_eval import bilinear_setup
        st = np.load(f"{DATA_DIR}/static.npz")
        H, W = [int(v) for v in st["grid_shape"]]
        self.H, self.W = H, W
        self.Hp, self.Wp = -(-H // stride) * stride, -(-W // stride) * stride
        dev = lat.device
        idx = np.full((self.Hp, self.Wp), -1, dtype=np.int64)
        idx[st["valid_rows"], st["valid_cols"]] = np.arange(len(st["valid_rows"]))
        self.pix = torch.from_numpy(idx).to(dev)[None]  # (1,Hp,Wp)
        self.rows = torch.from_numpy(st["valid_rows"].astype(np.int64)).to(dev)
        self.cols = torch.from_numpy(st["valid_cols"].astype(np.int64)).to(dev)
        self.gd = argparse.Namespace(bl=bilinear_setup(era5_lat, era5_lon, lat, lon, dev),
                                     latn=(lat - lat.mean()) / lat.std(), lonn=(lon - lon.mean()) / lon.std(),
                                     d=argparse.Namespace(static=static))

    def cond(self, era5, doy):
        """-> cond (1,26,Hp,Wp), mask (1,1,Hp,Wp)."""
        return GridData.cond_at(self.gd, self.pix, era5[None], doy.view(1))

    def gather(self, grid):  # (M,3,Hp,Wp) -> (M,N,3)
        return grid[:, :, self.rows, self.cols].permute(0, 2, 1)


@torch.no_grad()
def sample_grid_day(args, net, reg, fd, era5, doy, M, steps=18, generator=None):
    """-> z-space ensemble (M,N,3) at the valid pixels (M=1 for the regression stage)."""
    cond, mask = fd.cond(era5, doy)
    P = getattr(args, "patch", 128)
    stride = P - 16
    rnet = net if args.stage == "reg" else reg
    mean = patched(lambda t: regress(rnet, t), cond, mask, P, stride)  # tile-wise, as in training
    if args.stage == "reg":
        return fd.gather(mean)
    out = []
    for _ in range(M):
        r = edm_sample_patched(net, torch.cat([cond, mean], dim=1), mask, mean.shape, steps=steps,
                               generator=generator, P=P, stride=stride)
        out.append(mean + r)
    return fd.gather(torch.cat(out))


# --------------------------------------------------------------------------- training

class _Placeholder(IterableDataset):
    def __init__(self, n=None):
        self.n = n

    def __iter__(self):
        i = 0
        while self.n is None or i < self.n:
            yield torch.zeros(1)
            i += 1


class GridLit(pl.LightningModule):
    def __init__(self, args, gd):
        super().__init__()
        self.args, self.gd = args, gd
        self.net = build_net(args)
        self.ema = copy.deepcopy(self.net)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.reg = None
        if args.stage == "edm":
            _, reg, _, step = load_grid_run(args.reg_run, gd.d.device)
            for p in reg.parameters():
                p.requires_grad_(False)
            self.reg = [reg]  # list: keep out of the checkpoint
            print(f"[edm] regression stage {args.reg_run} (step {step})", flush=True)
        self.run_dir = f"{RUNS_ROOT}/{args.name}"
        self.t0, self._t_last = time.time(), None
        dev = gd.d.device
        g = torch.Generator(device=dev).manual_seed(1234)
        nv = gd.val_org.shape[0] // 12
        self.val_sigma = torch.exp(torch.linspace(-3, 2.5, nv, device=dev))  # fixed sigma grid for val loss
        self.val_noise = [torch.randn(nv, 3, gd.P, gd.P, device=dev, generator=g) for _ in range(12)]

    def loss(self, net, x, cond, ok, sigma=None, noise=None):
        if self.args.stage == "reg":
            pred = regress(net, cond)
            return (((pred - x) ** 2) * ok).sum() / (3 * ok.sum()).clamp(min=1)
        with torch.no_grad():
            mean = regress(self.reg[0], cond)
        y = (x - mean) * ok
        if sigma is None:
            sigma = torch.exp(torch.randn(x.shape[0], device=x.device) * 1.2 - 1.2)
        n = (torch.randn_like(y) if noise is None else noise) * sigma[:, None, None, None]
        D = edm_denoise(net, y + n, sigma, torch.cat([cond, mean], dim=1))
        w = ((sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2)[:, None, None, None]
        return (w * (D - y) ** 2 * ok).sum() / (3 * ok.sum()).clamp(min=1)

    def training_step(self, batch, batch_idx):
        x, cond, ok = self.gd.batch(self.args.batch_size)
        loss = self.loss(self.net, x, cond, ok)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            for pe, pm in zip(self.ema.parameters(), self.net.parameters()):
                pe.mul_(self.args.ema).add_(pm.detach(), alpha=1 - self.args.ema)
        if self._t_last is None:
            self._t_last = (self.global_step, time.time())
        elif self.global_step % 200 == 0:
            s0, t0 = self._t_last
            now = time.time()
            if self.global_step % 1000 == 0 or self.global_step <= 1000:
                print(f"[step {self.global_step}] {(now - t0) / max(self.global_step - s0, 1) * 1000:.0f} ms/step "
                      f"(incl. validation), elapsed {(now - self.t0) / 60:.1f} min", flush=True)
            self._t_last = (self.global_step, now)

    def validation_step(self, batch, batch_idx):
        self._run_val()

    def on_train_end(self):
        if self.global_step % self.args.val_every != 0:
            self._run_val()

    @torch.no_grad()
    def _run_val(self):
        tot = 0.0
        for k, (x, cond, ok) in enumerate(self.gd.val_batches()):
            tot += float(self.loss(self.ema.eval(), x, cond, ok, sigma=self.val_sigma, noise=self.val_noise[k]))
        res = {"step": int(self.global_step), "val_loss": tot / 12, "elapsed_min": (time.time() - self.t0) / 60}
        with open(f"{self.run_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[val step {res['step']}] val_loss={res['val_loss']:.4f}", flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=0.0)
        a = self.args
        f = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.max_steps))))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, f), "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(_Placeholder(), batch_size=None)

    def val_dataloader(self):
        return DataLoader(_Placeholder(1), batch_size=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", choices=["reg", "edm"], required=True)
    ap.add_argument("--reg-run", default=None, help="edm stage: trained regression run")
    ap.add_argument("--train-years", default="2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015")
    ap.add_argument("--val-year", type=int, default=2017)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mult", default="1,2,4,4")
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--attn", action="store_true", help="bottleneck self-attention (off: fully convolutional)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    ap.add_argument("--data-store", choices=["auto", "device", "cpu", "disk"], default="auto",
                    help="where the training targets live (see fast_train.py)")
    args = ap.parse_args()
    if args.stage == "edm" and not args.reg_run:
        raise SystemExit("--stage edm needs --reg-run")
    args.objective = "unet_reg" if args.stage == "reg" else "edm"
    args.static, args.doy_cond, args.target = "clim", True, "temp"  # fixed inputs, same as the final FFM
    pl.seed_everything(args.seed)
    run_dir = f"{RUNS_ROOT}/{args.name}"
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    with open(f"{run_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=1)
    device = pick_device(args.device)
    data = GPUData([int(y) for y in args.train_years.split(",")], args.val_year, device,
                   static="clim", seed=0, store=args.data_store)
    data.val_year = args.val_year
    gd = GridData(data, patch=args.patch)
    module = GridLit(args, gd)
    print(f"[model] {args.stage} UNet params={sum(p.numel() for p in module.net.parameters()):,}", flush=True)
    ckpt_cb = ModelCheckpoint(dirpath=f"{run_dir}/ckpt", every_n_train_steps=args.ckpt_every,
                              save_top_k=-1, save_last=True, filename="step{step:06d}")
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu", devices=[device.index or 0] if device.type == "cuda" else 1,
        plugins=[LightningEnvironment()], max_steps=args.max_steps,
        max_time=timedelta(minutes=args.time_budget_min) if args.time_budget_min else None,
        val_check_interval=args.val_every, check_val_every_n_epoch=None, limit_val_batches=1,
        num_sanity_val_steps=0, gradient_clip_val=args.grad_clip,
        logger=TensorBoardLogger(save_dir=run_dir, name="tb"), log_every_n_steps=100,
        callbacks=[ckpt_cb], enable_progress_bar=False)
    last = f"{run_dir}/ckpt/last.ckpt"
    trainer.fit(module, ckpt_path=last if (args.resume and os.path.exists(last)) else None)
    print(f"[done] {trainer.global_step} steps in {(time.time() - module.t0) / 60:.1f} min -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
