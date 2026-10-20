"""
fast_data.py
=============
GPU-resident data + sampling for the fast pipeline (fast_train.py). By default all
training targets (z-scored NGCD TG/TX/TN at valid pixels, fp16; ~44 GB for 15 years)
live on the GPU and a training batch is a handful of tensor index ops -- no DataLoader,
no decompression. On a smaller GPU they stay in host memory or on disk instead
(GPUData(store=...), fast_train.py --data-store) and only each batch is moved to the GPU.

Also holds the shared GPU-side math used by training AND evaluation:
correlated noise field (batched), harmonic denormalization, day-of-year
features, and the ODE sampler with an optional, vectorized exact
mass-conservation constraint.
"""
import math
import os

import numpy as np
import torch
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

DATA_DIR = CDFFM_ROOT + "/fast_data"

# Named spatial hold-out regions (lat/lon boxes, degrees). "lapland": the
# northern-Fennoscandia box used to test whether the continuous query
# generalizes to locations never sampled during training (plan §8).
HOLDOUT_REGIONS = {
    "lapland": dict(lat_min=66.5, lon_min=24.0),
}


def pick_device(name="auto"):
    """'auto' -> the first CUDA/ROCm GPU if there is one, else the CPU; otherwise torch.device(name)."""
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device):
    """Wait for queued GPU work (for timing); no-op on the CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free_host_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # not available (e.g. macOS): assume it fits
        return float("inf")


def _np_to(device, a, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)


def harmonic_basis(doy):
    """doy: (...) tensor -> (..., 5) basis matching climstats.basis/eval_harmonic."""
    ang = 2.0 * math.pi * doy.double() / 365.0
    return torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang),
                        torch.cos(2 * ang), torch.sin(2 * ang)], dim=-1)


def doy_features(doy):
    """(cos, sin) of the annual angle -- same convention as climstats.doy_cos_sin."""
    ang = 2.0 * math.pi * doy.float() / 365.0
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def mu_sigma(coeffs_mean, coeffs_var, doy, sigma_floor):
    """coeffs_*: (B,N,C,5) gathered at the query points; doy: (B,). Returns
    mu, sigma (B,N,C) -- identical math to climstats.eval_harmonic."""
    b = harmonic_basis(doy).to(coeffs_mean.dtype)[:, None, None, :]  # (B,1,1,5)
    mu = (coeffs_mean * b).sum(-1)
    var = (coeffs_var * b).sum(-1)
    return mu, torch.sqrt(torch.clamp(var, min=sigma_floor ** 2))


KM_PER_DEG = 111.2


def _rff_frequencies(B, n_features, kernel, length_scale_deg, beta, lmax_km, lmin_km, device, generator):
    """(B,F,2) random frequencies in cycles/deg, drawn from the kernel's spectral density.
    rbf:      Gaussian spectral density (squared-exponential kernel, scale length_scale_deg).
    powerlaw: isotropic per-mode power spectrum S(k) ~ k^-beta, band-limited to wavelengths
              [lmin_km, lmax_km] -- 'spectrum-matched' noise: beta is measured from NGCD's
              residual spectrum (~2.4 for TG/TX/TN at 5-100 km). The radial frequency has density
              ~ S(k) * k = k^(1-beta) and is drawn by inverting its CDF."""
    if kernel == "rbf":
        return torch.randn(B, n_features, 2, device=device, generator=generator) / length_scale_deg
    if kernel != "powerlaw":
        raise ValueError(f"unknown noise kernel {kernel}")
    k0, k1 = KM_PER_DEG / lmax_km, KM_PER_DEG / lmin_km  # cycles/deg
    u = torch.rand(B, n_features, device=device, generator=generator, dtype=torch.float64)
    e = 2.0 - beta
    if abs(e) < 1e-6:
        k = k0 * (k1 / k0) ** u
    else:
        k = (k0 ** e + u * (k1 ** e - k0 ** e)) ** (1.0 / e)
    th = torch.rand(B, n_features, device=device, generator=generator, dtype=torch.float64) * (2 * math.pi)
    return torch.stack([k * torch.cos(th), k * torch.sin(th)], dim=-1).float()


STATIC_DIMS = {"none": 0, "clim": 17, "clim_rr": 19}
TARGET_VARS = {"temp": ["TG", "TX", "TN"], "precip": ["RR"]}


def _interp(x, xp, fp):
    """np.interp for torch (xp ascending 1D; clamps outside like numpy)."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    w = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return f0 + w * (f1 - f0)


class PrecipTransform:
    """GPU version of transforms.denormalize_ngcd_precip: z -> mm/day. p_dry per pixel; bulk quantile
    table and single global GPD tail shared (compute_norm_stats.py). z <= Phi^-1(p_dry) -> dry (0)."""

    def __init__(self, data_dir, device):
        r = np.load(f"{data_dir}/static_rr.npz")
        self.p_dry = torch.from_numpy(r["p_dry"].astype(np.float32)).to(device)  # (N,), may contain NaN
        self.frac = torch.from_numpy(r["bulk_levels"] / 100.0).to(device)
        self.knots = torch.from_numpy(r["bulk_knots"]).to(device)
        self.thr, self.sig, self.xi = float(r["gpd_threshold"]), float(r["gpd_sigma"]), float(r["gpd_xi"])
        self.zeta_hi = 1.0 - float(r["bulk_levels"][-1]) / 100.0

    def to_mm(self, z, pix, eps=1e-4):
        """z: (..., P) normalized; pix: (P,) pixel indices -> mm, same shape (NaN stays NaN)."""
        zd = z.double()
        pd = self.p_dry[pix].double()
        F = 0.5 * (1.0 + torch.erf(zd / math.sqrt(2.0)))
        G = ((F - pd) / (1.0 - pd).clamp(min=1e-9)).clamp(0.0, 1.0)
        bulk = _interp(G, self.frac, self.knots)
        surv = ((1.0 - G) / self.zeta_hi).clamp(min=1e-12)
        e = -self.sig * torch.log(surv) if abs(self.xi) < 1e-6 else (self.sig / self.xi) * (surv ** (-self.xi) - 1.0)
        x = torch.where(G > self.frac[-1], self.thr + e, bulk)
        # Dry/wet decision at the MIDDLE of the gap in the training data: dry days sit exactly at G = 0
        # (z0 = Phi^-1(p_dry)), the lightest wet days at G = frac[0] (0.1-0.2 mm all map to the first
        # bulk level). Continuous model samples fall inside the gap; the old rule (G ~ 0 only) turned
        # all of them into 0.2 mm "wet" days (wet-day frequency bias ~1.7). Truth is unaffected.
        x = torch.where((F <= pd + eps) | (G < 0.5 * self.frac[0]), torch.zeros_like(x), x)
        x = torch.where(torch.isfinite(zd) & torch.isfinite(pd), x, torch.full_like(x, float("nan")))
        return x.to(z.dtype)

    def to_z(self, x, pix, wet_mm=0.1):
        """Forward transform (transforms.normalize_ngcd_precip): mm/day -> z. x: (..., P)."""
        xd = x.double().clamp(min=0.0)
        pd = self.p_dry[pix].double()
        G = _interp(xd, self.knots, self.frac)
        e = (xd - self.thr).clamp(min=0.0)
        fe = e / self.sig if abs(self.xi) < 1e-6 else torch.log1p((self.xi * e / self.sig).clamp(min=-1 + 1e-9)) / self.xi
        G = torch.where(xd > self.thr, 1.0 - self.zeta_hi * torch.exp(-fe), G)
        F = torch.where(xd <= wet_mm, pd, pd + (1.0 - pd) * G).clamp(1e-9, 1 - 1e-9)
        return (math.sqrt(2.0) * torch.erfinv(2.0 * F - 1.0)).to(x.dtype)


def to_phys(data_like, z, pix, doy):
    """Normalized model space -> physical units, for either target. data_like: object with
    target, coeffs_mean, coeffs_var, sigma_floor (+ precip transform rr). z: (B,P,C); pix: (P,);
    doy: (1,) or (B,). Temperature: z*sigma+mu (identical to the old inline code); precip: mm/day."""
    if data_like.target == "precip":
        return data_like.rr.to_mm(z[..., 0], pix)[..., None]
    mu, sig = mu_sigma(data_like.coeffs_mean[pix][None], data_like.coeffs_var[pix][None], doy, data_like.sigma_floor)
    return z * sig + mu


def static_features(coeffs_mean, coeffs_var, cell_id, n_cells, sigma_floor, p_dry=None):
    """Per-pixel static covariates from the NGCD 1981-2015 climatology (a terrain proxy, no DEM):
    for each of TG/TX/TN -- annual mean mu0, mu0 minus its ERA5-cell mean (sub-grid warm/cold
    offset: valley floors, ridges, coast), annual-cycle amplitude (coast vs inland), January and
    July sigma (cold-air pooling, variability) -- plus the TX-TN diurnal range and its sub-grid
    offset. coeffs_*: (N,3,5); cell_id: (N,) long. Returns (N,17) float32, standardized."""
    cm, cv = coeffs_mean.double(), coeffs_var.double()
    dev = cm.device
    count = torch.zeros(n_cells, dtype=torch.float64, device=dev).index_add_(
        0, cell_id, torch.ones(cell_id.shape[0], dtype=torch.float64, device=dev))

    def rel(v):
        m = torch.zeros(n_cells, dtype=torch.float64, device=dev).index_add_(0, cell_id, v) / count.clamp(min=1)
        return v - m[cell_id]

    def sigma_at(doy):
        b = harmonic_basis(torch.tensor([doy], device=dev))[0].to(cv.dtype)  # (5,)
        return torch.sqrt(torch.clamp((cv * b).sum(-1), min=sigma_floor ** 2))  # (N,3)

    s_jan, s_jul = sigma_at(15), sigma_at(196)
    feats = []
    for c in range(3):
        mu0 = cm[:, c, 0]
        feats += [mu0, rel(mu0), torch.sqrt(cm[:, c, 1] ** 2 + cm[:, c, 2] ** 2), s_jan[:, c], s_jul[:, c]]
    dtr = cm[:, 1, 0] - cm[:, 2, 0]
    feats += [dtr, rel(dtr)]
    if p_dry is not None:  # precipitation model: dry-day probability and its sub-grid offset
        pd = torch.nan_to_num(p_dry.double(), nan=float(torch.nanmean(p_dry.double())))
        feats += [pd, rel(pd)]
    f = torch.stack(feats, dim=-1)
    f = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
    return f.float()


def noise_kwargs(args):
    """noise_field keyword arguments from a training config (old runs: RBF defaults)."""
    return dict(n_features=getattr(args, "noise_features", 64), length_scale_deg=args.noise_ls,
                kernel=getattr(args, "noise_kernel", "rbf"), beta=getattr(args, "noise_beta", 2.4),
                lmax_km=getattr(args, "noise_lmax_km", 500.0), lmin_km=getattr(args, "noise_lmin_km", 2.0))


def noise_field(lat, lon, n_out, n_features=64, length_scale_deg=0.5, generator=None,
                kernel="rbf", beta=2.4, lmax_km=500.0, lmin_km=2.0, chunk=None):
    """Batched random-Fourier-feature Gaussian field, one independent draw
    per batch row. lat, lon: (B,N) degrees -> (B,N,n_out), unit marginal
    variance (the old NoiseField had variance 0.5). Longitude is scaled by
    cos(64.5 deg) so the correlation length is roughly isotropic in km over
    the Nordic domain. Computed as cos(a)@A - sin(a)@B, so memory is
    O(N*F), not O(N*F*n_out); chunk (points) bounds it further for ~1.3M
    points per day in eval -- same random field, evaluated piecewise.
    kernel: 'rbf' or 'powerlaw' (see _rff_frequencies)."""
    B, N = lat.shape
    dev = lat.device
    freq = _rff_frequencies(B, n_features, kernel, length_scale_deg, beta, lmax_km, lmin_km, dev, generator)
    phase = torch.rand(B, n_features, n_out, device=dev, generator=generator) * (2 * math.pi)
    coeff = torch.randn(B, n_features, n_out, device=dev, generator=generator)
    a = coeff * torch.cos(phase)
    b = coeff * torch.sin(phase)
    step = N if chunk is None else chunk
    field = torch.empty(B, N, n_out, device=dev)
    for s in range(0, N, step):
        e = min(N, s + step)
        coords = torch.stack([lat[:, s:e], lon[:, s:e] * math.cos(math.radians(64.5))], dim=-1)  # (B,n,2)
        ang = 2 * math.pi * torch.bmm(coords, freq.transpose(1, 2))  # (B,n,F)
        field[:, s:e] = torch.bmm(torch.cos(ang), a) - torch.bmm(torch.sin(ang), b)
    return field * math.sqrt(2.0 / n_features)


class GPUData:
    def __init__(self, train_years, val_year, device, n_val_dates=12, n_val_pixels=20000,
                 holdout=None, cluster_min_pixels=64, seed=0, data_dir=DATA_DIR, static="none", target="temp",
                 store="auto"):
        """store: where the training targets live -- 'device' (fastest), 'cpu' (host RAM) or 'disk' (memory-mapped
        .npy files, slowest, needs no RAM); 'auto' picks the first that fits. Batches are identical in all cases."""
        self.device = device
        st = np.load(f"{data_dir}/static.npz")
        self.era5_lat = st["era5_lat"]
        self.era5_lon = st["era5_lon"]
        self.sigma_floor = float(st["sigma_floor"])
        self.lat = _np_to(device, st["valid_lat"])
        self.lon = _np_to(device, st["valid_lon"])
        self.cell_id = _np_to(device, st["era5_cell_id"], torch.long)
        self.coeffs_mean = _np_to(device, st["coeffs_mean"])  # (N,3,5)
        self.coeffs_var = _np_to(device, st["coeffs_var"])
        self.n_valid = self.lat.shape[0]
        self.target, self.vars = target, TARGET_VARS[target]
        self.n_out = len(self.vars)
        self.rr = PrecipTransform(data_dir, device) if target == "precip" else None
        suffix = "rr" if target == "precip" else "targets"
        self.static = None  # (N,S) per-pixel static covariates, gathered per query point
        if static != "none":
            self.static = static_features(self.coeffs_mean, self.coeffs_var, self.cell_id,
                                          len(self.era5_lat) * len(self.era5_lon), self.sigma_floor,
                                          p_dry=self.rr.p_dry if static == "clim_rr" else None)

        # ---- training years -> one tensor (D, N, 3) fp16 on the GPU, in host RAM, or memory-mapped from disk
        metas = [np.load(f"{data_dir}/year_{y}_meta.npz") for y in train_years]
        tgt = [np.load(f"{data_dir}/year_{y}_{suffix}.npy", mmap_mode="r") for y in train_years]
        D = sum(t.shape[0] for t in tgt)
        need = D * self.n_valid * self.n_out * 2
        if store == "auto":
            gpu_free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0
            store = ("device" if device.type == "cuda" and need < gpu_free - 6e9 else
                     "cpu" if need < 0.8 * _free_host_bytes() else "disk")
        if device.type == "cpu" and store == "device":
            store = "cpu"
        self.store = store
        self._mm, self._mm_start = tgt, np.cumsum([0] + [t.shape[0] for t in tgt])
        self.targets = None
        if store in ("device", "cpu"):
            where = device if store == "device" else torch.device("cpu")
            self.targets = torch.empty((D, self.n_valid, self.n_out), dtype=torch.float16, device=where)
            o = 0
            for t in tgt:
                for s in range(0, t.shape[0], 8):  # chunked copy keeps host RSS small
                    chunk = np.array(t[s:s + 8])  # real (writable) copy of the memmap slice
                    self.targets[o + s:o + s + chunk.shape[0]] = torch.from_numpy(chunk).to(where)
                o += t.shape[0]
        self.era5 = _np_to(device, np.concatenate([m["era5"] for m in metas]))  # (D,4,H,W)
        self.era5_t2m_phys = _np_to(device, np.concatenate([m["era5_t2m_phys"] for m in metas]))  # (D,H,W)
        self.doy = _np_to(device, np.concatenate([m["doy"] for m in metas]).astype(np.int64))
        self.n_days = D

        # ---- spatial hold-out: training never samples these pixels
        self.holdout_mask = torch.zeros(self.n_valid, dtype=torch.bool, device=device)
        if holdout:
            r = HOLDOUT_REGIONS[holdout]
            self.holdout_mask = (self.lat >= r["lat_min"]) & (self.lon >= r["lon_min"])
        self.allowed_idx = torch.nonzero(~self.holdout_mask).squeeze(1)

        # ---- cluster table over allowed pixels: cells with >= cluster_min_pixels members
        cid = self.cell_id[self.allowed_idx]
        order = torch.argsort(cid, stable=True)
        uniq, counts = torch.unique_consecutive(cid[order], return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        keep = counts >= cluster_min_pixels
        self._cl_order = order
        self.cl_cell = uniq[keep]
        self.cl_start = starts[keep]
        self.cl_count = counts[keep]

        # ---- fixed validation set: n_val_dates spread over val_year, fixed pixel subset
        g = np.random.default_rng(seed)
        vm = np.load(f"{data_dir}/year_{val_year}_meta.npz")
        vt = np.load(f"{data_dir}/year_{val_year}_{suffix}.npy", mmap_mode="r")
        n_vd = len(vm["dates"])
        vdays = np.linspace(14, n_vd - 16, n_val_dates).round().astype(int)
        vpix = np.sort(g.choice(self.n_valid, size=min(n_val_pixels, self.n_valid), replace=False))
        self.val_dates = [str(vm["dates"][i]) for i in vdays]
        self.val_era5 = _np_to(device, vm["era5"][vdays])
        self.val_doy = _np_to(device, vm["doy"][vdays].astype(np.int64))
        self.val_pix = _np_to(device, vpix, torch.long)
        self.val_targets = _np_to(device, np.stack([vt[i][vpix] for i in vdays]).astype(np.float32))  # (Vd,P,3)

        print(f"[GPUData] target={target} {self.vars}, train days={D} ({train_years}), valid pixels={self.n_valid}, "
              f"held-out pixels={int(self.holdout_mask.sum())}, cluster cells={len(self.cl_cell)}, "
              f"val: {len(vdays)} dates x {len(vpix)} pixels, "
              f"targets: {need / 1e9:.1f} GB on {self.store}"
              + (f", GPU mem={torch.cuda.memory_allocated(device) / 1e9:.1f} GB" if device.type == "cuda" else ""),
              flush=True)

    def gather_targets(self, day, idx):
        """targets[day, idx] as fp16 on self.device, for day (B,) and pixel indices idx (B, ...)."""
        bshape = (-1,) + (1,) * (idx.dim() - 1)
        if self.store == "device":
            return self.targets[day.view(bshape), idx]
        d, ix = day.cpu(), idx.cpu()
        if self.store == "cpu":
            out = self.targets[d.view(bshape), ix]
        else:  # disk: one day row (N, C) per sample from the memory-mapped year files
            rows = []
            for k, i in zip(d.tolist(), ix):
                y = int(np.searchsorted(self._mm_start, k, side="right") - 1)
                rows.append(np.asarray(self._mm[y][k - self._mm_start[y]])[i.numpy()])
            out = torch.from_numpy(np.stack(rows))
        return out.to(self.device)

    def gather_coeffs(self, idx):
        """idx: (B,N) pixel indices -> coeffs_mean, coeffs_var (B,N,3,5)."""
        return self.coeffs_mean[idx], self.coeffs_var[idx]

    def sample_batch(self, B, n_scattered, n_clusters, points_per_cluster, generator=None):
        """Returns a dict of GPU tensors. Point layout per sample:
        [n_scattered uniform points | n_clusters * points_per_cluster points
        from n_clusters random ERA5 cells] -- the cluster block reshapes to
        (B, n_clusters, points_per_cluster), so per-cell means need no loop."""
        dev = self.device
        day = torch.randint(0, self.n_days, (B,), device=dev, generator=generator)
        s_idx = self.allowed_idx[torch.randint(0, len(self.allowed_idx), (B, n_scattered), device=dev, generator=generator)]
        cells = torch.randint(0, len(self.cl_cell), (B, n_clusters), device=dev, generator=generator)
        r = (torch.rand(B, n_clusters, points_per_cluster, device=dev, generator=generator)
             * self.cl_count[cells][..., None]).long()
        c_idx = self.allowed_idx[self._cl_order[self.cl_start[cells][..., None] + r]]
        idx = torch.cat([s_idx, c_idx.reshape(B, -1)], dim=1)  # (B,N)

        x1 = self.gather_targets(day, idx).float()  # (B,N,3)
        valid = torch.isfinite(x1)
        x1 = torch.nan_to_num(x1, nan=0.0)
        era5_cell_phys = self.era5_t2m_phys[day].reshape(B, -1).gather(1, self.cl_cell[cells])  # (B,C)
        return dict(day=day, idx=idx, lat=self.lat[idx], lon=self.lon[idx], x1=x1, valid=valid,
                    era5=self.era5[day], doy=self.doy[day], era5_cell_phys=era5_cell_phys,
                    static=self.static[idx] if self.static is not None else None)


@torch.no_grad()
def ode_sample(model, cond, lat, lon, n_steps, noise="field", noise_ls=0.5, generator=None, noise_features=64,
               constraint=None, noise_kw=None, flowmap=False):
    """Euler integration of the learned velocity from t=0 to 1. cond:
    (B,N,cond_dim) precomputed condition (incl. any extra conditioning).
    constraint: optional callable x -> x applied after every step (e.g. the
    exact mass-conservation projection, see make_mean_constraint)."""
    B, N, _ = cond.shape
    n_out = model.n_out
    if noise == "field":
        kw = noise_kw or dict(n_features=noise_features, length_scale_deg=noise_ls)
        x = noise_field(lat, lon, n_out, generator=generator, **kw)
    else:
        x = torch.randn(B, N, n_out, device=cond.device, generator=generator)
    dt = 1.0 / n_steps
    for s in range(n_steps):
        t = torch.full((B, N), s * dt, device=cond.device)
        if flowmap:  # few-step flow map: jump t -> t + dt with the average velocity
            x = x + model.velocity(x, cond, t, t + dt) * dt
        else:
            x = x + model.velocity(x, cond, t) * dt
        if constraint is not None:
            x = constraint(x)
    return x


def make_mean_constraint(cell_id, mu0, sigma0, target_phys):
    """Exact (hard) mass conservation for channel 0 (TG), vectorized: shift
    each ERA5 cell's group of points so its PHYSICAL mean equals the ERA5
    value. cell_id: (N,) long cell index per point (same cells for every
    batch row); mu0, sigma0: (B,N) channel-0 climatology at the points;
    target_phys: (B, n_cells_total) ERA5 t2m_mean in degC per cell.
    Replaces infer.py's per-cell Python loop with index_add segment means."""
    n_cells = target_phys.shape[1]
    ones = torch.ones_like(cell_id, dtype=mu0.dtype)
    count = torch.zeros(n_cells, device=mu0.device, dtype=mu0.dtype).index_add_(0, cell_id, ones)
    has = count[cell_id] >= 2  # (N,) -- singleton groups left alone, as in infer.py

    def project(x):
        phys = x[..., 0] * sigma0 + mu0  # (B,N)
        sums = torch.zeros(phys.shape[0], n_cells, device=phys.device, dtype=phys.dtype).index_add_(1, cell_id, phys)
        mean = sums / count.clamp(min=1)
        shift = (target_phys - mean)[:, cell_id]  # (B,N)
        phys = torch.where(has, phys + shift, phys)
        x = x.clone()
        x[..., 0] = (phys - mu0) / sigma0
        return x

    return project
