"""
model.py
========
FNO encoder (over the small, fixed ERA5 Nordic grid -- never patched, see
functional_flow_downscaling_plan.md §2) + LIIF-style continuous query +
flow-matching velocity decoder + correlated noise field.

Output channels (fixed order, always all 4 present -- train.py decides
which are supervised per curriculum stage, not this module):
    0: temp_mean (NGCD TG, Celsius, harmonic-normalized)
    1: temp_max  (NGCD TX)
    2: temp_min  (NGCD TN)
    3: precip    (NGCD RR, p_dry-jump + GPD-normalized, see transforms.py)

FNO mode count here is a FIXED hyperparameter, not dynamically scaled to
input size -- that scaling concern (§2.2's discussion) only applies to a
whole-field generator operating at patch vs. full-domain scale, which this
design avoids entirely (the encoder always sees the same small, fixed
ERA5 grid; the decoder is pointwise). No patch/mode-scaling risk here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flower_layers import Flower
from fno_layers import FNOEncoder
from fno_unet import FNOUNet

N_OUT = 4  # temp_mean, temp_max, temp_min, precip


# ---------------------------------------------------------------------------
# FNO encoders -- FNOEncoder (flat, backbone="fno_flat") imported from
# fno_layers.py; FNOUNetAdapter (backbone="fno", the primary comparison arm)
# below, mirroring FlowerEncoderAdapter's padding/cropping treatment since
# FNOUNet has the same strided-conv divisibility requirement as Flower's U-Net.
# ---------------------------------------------------------------------------

class FNOUNetAdapter(nn.Module):
    """Same padding/cropping treatment as FlowerEncoderAdapter, built eagerly
    at __init__ for the same reason (lazy construction would register
    FNOUNet's parameters too late for the optimizer to see them)."""

    def __init__(self, in_ch, H, W, feat_dim=128, lifting_dim=64, n_levels=4, modes1=12, modes2=16, groups=8):
        super().__init__()
        min_divisor = 2 ** (n_levels - 1)
        H_pad = -(-H // min_divisor) * min_divisor
        W_pad = -(-W // min_divisor) * min_divisor
        self.H, self.W = H, W
        self.pad_h, self.pad_w = H_pad - H, W_pad - W
        self.fno_unet = FNOUNet(
            dim_in=in_ch, dim_out=feat_dim, spatial_resolution=[H_pad, W_pad],
            lifting_dim=lifting_dim, n_levels=n_levels, modes1=modes1, modes2=modes2, groups=groups,
        )

    def forward(self, x):
        x_padded = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="reflect")
        out = self.fno_unet(x_padded)
        return out[:, :, :self.H, :self.W]


# ---------------------------------------------------------------------------
# Flower encoder -- thin adapter around the REAL Flower U-Net, copied
# verbatim (2D-only) into flower_layers.py from
# Extreme_weather/flowers/flower_standalone.py per explicit instruction:
# an earlier version of this file improvised a single-head, single-
# resolution "SelfWarp-like" block that got the real mechanism wrong in
# several ways (SelfWarp is multi-head -- 32 by default -- each warping
# its own value-channel slice; the real model is a full U-Net with
# n_levels downsample/bottleneck/upsample stages, not a flat residual
# stack; and it uses a custom periodic-boundary-aware grid_sample). That
# version is gone; this wraps the actual copied classes instead.
# ---------------------------------------------------------------------------

class FlowerEncoderAdapter(nn.Module):
    """Flower's U-Net requires spatial dims divisible by 2**(n_levels-1);
    our ERA5 grid (71x113) isn't, so this pads (reflect) up to the nearest
    valid size ONCE at construction time (H,W are fixed -- always the same
    Nordic crop -- so there's no lazy/first-forward construction here,
    which would otherwise register Flower's parameters too late for the
    optimizer to see them) and crops the output back down in forward().
    FNOEncoder needs no such handling (FFT works at any size) -- this
    asymmetry is inherent to using the real, validated U-Net architecture,
    not something to "fix" on the FNO side.

    Hyperparameters (lifting_dim=64, num_heads=8, groups=8) are scaled down
    from Flower's production config (lifting_dim=160, num_heads=40,
    groups=40, tuned for the 720x1440 global grid) to match this project's
    much smaller regional domain and FNOEncoder's width=64 -- chosen so
    channel counts at every U-Net level stay evenly divisible by num_heads
    and groups (verified: 64/128/256/512 all divide cleanly by 8)."""

    def __init__(self, in_ch, H, W, feat_dim=128, lifting_dim=64, n_levels=4, num_heads=8, groups=8):
        super().__init__()
        min_divisor = 2 ** (n_levels - 1)
        H_pad = -(-H // min_divisor) * min_divisor  # round up to the nearest multiple
        W_pad = -(-W // min_divisor) * min_divisor
        self.H, self.W = H, W
        self.pad_h, self.pad_w = H_pad - H, W_pad - W
        self.flower = Flower(
            dim_in=in_ch, dim_out=feat_dim, n_spatial_dims=2, spatial_resolution=[H_pad, W_pad],
            lifting_dim=lifting_dim, n_levels=n_levels, num_heads=num_heads, groups=groups,
            boundary_condition_types=["NONE", "NONE"],  # non-periodic -- regional crop, not global
            dim_meta=0,
        )

    def forward(self, x):
        x_padded = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="reflect")
        out = self.flower(x_padded)
        return out[:, :, :self.H, :self.W]


# ---------------------------------------------------------------------------
# LIIF-style continuous query (MLP does the interpolation, not bilinear
# blending -- same philosophy as pm_psd's LIIFStationQuery)
# ---------------------------------------------------------------------------

def _make_coord_grid(H, W, device):
    """Normalized cell-centre coordinates in [-1,1], (H,W,2), [...,0]=row(y),
    [...,1]=col(x) -- same convention as pm_psd/model_v2.py's make_coord."""
    ys = -1.0 + 2.0 * (torch.arange(H, device=device).float() + 0.5) / H
    xs = -1.0 + 2.0 * (torch.arange(W, device=device).float() + 0.5) / W
    g = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # (H,W,2)
    return g


class LIIFQuery(nn.Module):
    """Faithful adoption of pm_psd/model_v2.py's LIIFStationQuery design
    (feat_unfold + local_ensemble) -- an earlier version of this class used
    only a single nearest-cell lookup with no neighborhood context and no
    boundary blending, missing the two things that most directly address
    this project's own spatial-coherence concern (plan §2.2): feat_unfold
    gives the MLP a 3x3 neighborhood instead of one cell; local_ensemble
    queries 4 diagonal-corner positions and blends them by (opposite-
    corner) area weight, so there's no discontinuous jump as a query point
    crosses a cell boundary -- the standard bilinear-via-area-weighting
    construction, not something to simplify away.

    static_feat (elevation/land-sea/coast-distance, once sourced) is this
    project's own addition on top of the original design -- concatenated
    identically at each of the 4 ensemble positions, since static
    covariates don't need the sub-cell-offset treatment the way the
    queried feature does."""

    def __init__(self, feat_dim, out_dim, lat_grid, lon_grid, static_dim=0, hidden=128,
                 feat_unfold=True, local_ensemble=True):
        super().__init__()
        self.register_buffer("lat0", torch.tensor(float(lat_grid[0])))
        self.register_buffer("lat1", torch.tensor(float(lat_grid[-1])))
        self.register_buffer("lon0", torch.tensor(float(lon_grid[0])))
        self.register_buffer("lon1", torch.tensor(float(lon_grid[-1])))
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.static_dim = static_dim
        in_dim = feat_dim * (9 if feat_unfold else 1) + 2 + static_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _normalize_coord(self, lat, lon):
        y = 2 * (lat - self.lat0) / (self.lat1 - self.lat0) - 1
        x = 2 * (lon - self.lon0) / (self.lon1 - self.lon0) - 1
        return torch.stack([y, x], dim=-1)  # (B,N,2), row-major (y,x)

    def forward(self, feat_map, lat, lon, static_feat=None):
        """feat_map: (B,C,H,W) -- may be a cropped (non-contiguous) slice
        from FNOUNetAdapter/FlowerEncoderAdapter's pad-then-crop treatment,
        hence .reshape() below, not .view() (confirmed necessary by an
        actual RuntimeError from the smoke test). lat,lon: (B,N) degrees.
        static_feat: (B,N,static_dim) or None. Returns (B,N,out_dim)."""
        B, Cf, H, W = feat_map.shape
        coord = self._normalize_coord(lat, lon)  # (B,N,2)

        feat = feat_map
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).reshape(B, Cf * 9, H, W)

        if self.local_ensemble:
            vx_lst, vy_lst, eps_shift = [-1, 1], [-1, 1], 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0.0

        rx = 2 / H / 2
        ry = 2 / W / 2
        feat_coord = _make_coord_grid(H, W, feat_map.device).permute(2, 0, 1).unsqueeze(0).expand(B, 2, H, W)

        static_part = static_feat
        if static_part is None and self.static_dim > 0:
            static_part = torch.zeros(B, lat.shape[1], self.static_dim, device=lat.device)

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                c = coord.clone()
                c[:, :, 0] = c[:, :, 0] + vx * rx + eps_shift
                c[:, :, 1] = c[:, :, 1] + vy * ry + eps_shift
                c = c.clamp(-1 + 1e-6, 1 - 1e-6)
                grid = c.flip(-1).unsqueeze(1)  # grid_sample expects (x,y)
                q_feat = F.grid_sample(feat, grid, mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, grid, mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel = coord - q_coord
                rel_scaled = torch.stack([rel[:, :, 0] * H, rel[:, :, 1] * W], dim=-1)

                parts = [q_feat, rel_scaled]
                if static_part is not None:
                    parts.append(static_part)
                inp = torch.cat(parts, dim=-1)
                bs, q = coord.shape[:2]
                preds.append(self.mlp(inp.reshape(bs * q, -1)).reshape(bs, q, -1))
                areas.append(torch.abs(rel_scaled[:, :, 0] * rel_scaled[:, :, 1]) + 1e-9)

        tot = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot).unsqueeze(-1)
        return ret


# ---------------------------------------------------------------------------
# Correlated noise field (random Fourier features) -- NOT i.i.d. per point,
# see functional_flow_downscaling_plan.md §2.2 for the Bochner's-theorem
# justification for why this is a legitimate stochastic-process source.
# ---------------------------------------------------------------------------

class NoiseField:
    """Not an nn.Module (no learned parameters) -- a reproducible-per-seed
    random basis. length_scale_deg controls spatial correlation length; a
    fixed starting hyperparameter, not yet tuned/learned (plan §9)."""

    def __init__(self, out_dim, n_features=64, length_scale_deg=0.5):
        self.out_dim = out_dim
        self.n_features = n_features
        self.length_scale = length_scale_deg

    def sample(self, lat, lon, seed, device):
        """lat,lon: (N,) degrees (single sample's query points -- one seed
        per generated scenario, shared across all its query points).
        Returns (N, out_dim)."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        freq = torch.randn(self.n_features, 2, generator=g) / self.length_scale
        phase = torch.rand(self.n_features, self.out_dim, generator=g) * 2 * math.pi
        coeff = torch.randn(self.n_features, self.out_dim, generator=g)
        freq, phase, coeff = freq.to(device), phase.to(device), coeff.to(device)

        coords = torch.stack([lat, lon], dim=-1)  # (N,2)
        ang = 2 * math.pi * (coords @ freq.T)  # (N, n_features)
        basis = torch.cos(ang.unsqueeze(-1) + phase.unsqueeze(0))  # (N,n_features,out_dim)
        field = (basis * coeff.unsqueeze(0)).sum(dim=1) / math.sqrt(self.n_features)
        return field


# ---------------------------------------------------------------------------
# Flow-matching velocity decoder
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,N) or (B,) in [0,1]."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowVelocityNet(nn.Module):
    """Plain concatenation-conditioned MLP -- a deliberately simple starting
    point (not full AdaLN-style modulation); upgrade later only if needed."""

    def __init__(self, x_dim, cond_dim, time_dim=64, hidden=256, depth=4):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        in_dim = x_dim + cond_dim + time_dim
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, cond, t):
        """x_t: (B,N,x_dim). cond: (B,N,cond_dim). t: (B,N) or (B,1) broadcastable."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(x_t.shape[0], x_t.shape[1])
        te = self.time_embed(t)  # (B,N,time_dim)
        inp = torch.cat([x_t, cond, te], dim=-1)
        return self.net(inp)


class TwoTimeVelocityNet(nn.Module):
    """Flow map (few-step sampler): u(x_t, t, s) = average velocity from t to s, so
    Psi(x_t, t, s) = x_t + (s - t) * u(x_t, t, s) jumps along the learned ODE in one call.
    Same pointwise MLP as FlowVelocityNet plus an embedding of the step size d = s - t;
    with s = None (d = 0) it is the instantaneous velocity, so Euler sampling and the exact
    log score keep working. from_teacher() copies a trained FlowVelocityNet and zero-inits the
    new step-size inputs, so before distillation u(x, t, s) = v(x, t) (one Euler step)."""

    def __init__(self, x_dim, cond_dim, time_dim=64, hidden=256, depth=4):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_dim = time_dim
        d, layers = x_dim + cond_dim + 2 * time_dim, []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, x_dim)]
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_teacher(cls, teacher):
        first = teacher.net[0]
        x_cond_time = first.in_features
        hidden = first.out_features
        depth = sum(isinstance(m, nn.Linear) for m in teacher.net) - 1
        x_dim = teacher.net[-1].out_features
        tdim = teacher.time_embed.dim
        new = cls(x_dim, x_cond_time - x_dim - tdim, time_dim=tdim, hidden=hidden, depth=depth)
        with torch.no_grad():
            for a, b in zip(new.net, teacher.net):
                if isinstance(a, nn.Linear) and a is not new.net[0]:
                    a.weight.copy_(b.weight)
                    a.bias.copy_(b.bias)
            new.net[0].weight.zero_()
            new.net[0].weight[:, :x_cond_time].copy_(first.weight)
            new.net[0].bias.copy_(first.bias)
        return new

    def forward(self, x_t, cond, t, s=None):
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(x_t.shape[0], x_t.shape[1])
        d = torch.zeros_like(t) if s is None else (s - t)
        inp = torch.cat([x_t, cond, self.time_embed(t), self.time_embed(d)], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class DownscalingFlowModel(nn.Module):
    def __init__(self, era5_in_ch, lat_grid, lon_grid, backbone="fno", feat_dim=128, cond_mlp_dim=64,
                 static_dim=0, encoder_kwargs=None, decoder_kwargs=None, n_out=N_OUT, extra_cond_dim=0):
        """n_out: output channels (default 4 = TG,TX,TN,RR; the fast pipeline's
        temperature-only stage uses 3). extra_cond_dim: width of an optional
        per-point conditioning vector concatenated to the LIIF condition
        before the velocity net (e.g. 2 for day-of-year cos/sin) -- the
        caller concatenates it; this only sizes the velocity net's input."""
        super().__init__()
        self.n_out = n_out
        self.extra_cond_dim = extra_cond_dim
        if backbone not in ("fno", "fno_flat", "flower"):
            raise ValueError(f"backbone must be 'fno', 'fno_flat', or 'flower', got {backbone!r}")
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        self.backbone = backbone
        if backbone == "fno":
            self.encoder = FNOUNetAdapter(
                era5_in_ch, H=len(lat_grid), W=len(lon_grid), feat_dim=feat_dim, **encoder_kwargs)
        elif backbone == "fno_flat":
            self.encoder = FNOEncoder(era5_in_ch, feat_dim=feat_dim, **encoder_kwargs)
        else:
            self.encoder = FlowerEncoderAdapter(
                era5_in_ch, H=len(lat_grid), W=len(lon_grid), feat_dim=feat_dim, **encoder_kwargs)
        self.query = LIIFQuery(feat_dim, cond_mlp_dim, lat_grid, lon_grid, static_dim=static_dim)
        self.velocity_net = FlowVelocityNet(n_out, cond_mlp_dim + extra_cond_dim, **decoder_kwargs)
        self.noise_field = NoiseField(n_out)

    def encode(self, era5_window):
        """era5_window: (B,C,H,W) normalized ERA5 fields -> feature map."""
        return self.encoder(era5_window)

    def condition(self, feat_map, lat, lon, static_feat=None):
        return self.query(feat_map, lat, lon, static_feat)

    def velocity(self, x_t, cond, t, s=None):
        """Instantaneous velocity; with s (flow-map models only) the average velocity from t to s."""
        if s is None:
            return self.velocity_net(x_t, cond, t)
        return self.velocity_net(x_t, cond, t, s)

    def sample_noise(self, lat, lon, seed, device):
        """lat,lon: (N,) for ONE sample (seed shared across its query batch)."""
        return self.noise_field.sample(lat, lon, seed, device)
