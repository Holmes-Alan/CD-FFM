"""
fast_eval.py
=============
Full GPU evaluation of a fast_train.py run on a whole test year: every
(sub-sampled) day, ALL ~1.33M valid NGCD pixels, ensemble of --members
samples. Inference-time options are evaluated WITHOUT retraining:

  --constraint none|exact|exact_debiased
      exact: shift each ERA5 cell's group of pixels after every ODE step so
      its mean TG equals the ERA5 value (vectorized version of infer.py's
      loop; cells with < --constraint-min-pixels NGCD pixels are left
      alone -- mostly-sea coastal cells aren't comparable to their land
      pixels). exact_debiased: same, but the target is ERA5 + the
      training-period monthly mean (NGCD cell mean - ERA5) offset, since
      ERA5 is not a coarsening of NGCD (smoothed orography -> lapse-rate
      bias); the offset is computed once from the training years and cached.
  --ode-steps N    Euler steps (speed/quality tradeoff)

Reports, per variable and per season + whole year (and held-out vs inside
for --holdout runs): RMSE of one member and of the ensemble mean, MAE,
bias, CRPS, spread and spread/skill, rank histogram, hot/cold extremes
(|z| > 1.645 vs local climatology: CSI of one sample, Brier score of the
ensemble probability), the TG cell-mean residual vs ERA5 for predictions AND
for the truth itself (the size of the truth's residual is the error an
'exact' constraint to raw ERA5 necessarily injects), and the radially
averaged power spectrum (truth vs one sample vs ensemble mean) over the
largest fully-valid 512x512 km box. Writes runs/<run>/eval_<year>_<tag>/.
"""
import argparse
import datetime
import json
import re
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from fast_data import (pick_device, DATA_DIR, HOLDOUT_REGIONS, TARGET_VARS, PrecipTransform, doy_features, mu_sigma, noise_field,
                       noise_kwargs, static_features)
from fast_train import VARS, build_model
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

RUNS_ROOT = CDFFM_ROOT + "/runs"
SEASONS = ["year", "DJF", "MAM", "JJA", "SON"]
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
EXTREME_Z = 1.645  # ~95th percentile of the local climatology
RR_THRESHOLDS_MM = (0.1, 1.0, 10.0, 20.0)  # precipitation: wet day, and 1 / 10 / 20 mm per day


def saved_steps(ckdir):
    """{step: path} of the periodic checkpoints in ckdir. Lightning may add a '-v1' suffix when it saves
    the same step twice (e.g. at the final step); the plain name is preferred."""
    out = {}
    for f_ in sorted(os.listdir(ckdir), key=len, reverse=True):  # longer (-vN) names first, plain overwrites
        m = re.match(r"stepstep=(\d+)(-v\d+)?\.ckpt$", f_)
        if m:
            out[int(m.group(1))] = f"{ckdir}/{f_}"
    return out


def best_val_step(name):
    """Step of the saved checkpoint with the lowest validation CRPS (sum over
    TG/TX/TN, val year) -- early stopping on the validation year; the test
    year never enters the choice."""
    rows = [json.loads(l) for l in open(f"{RUNS_ROOT}/{name}/metrics.jsonl") if l.strip()]
    saved = saved_steps(f"{RUNS_ROOT}/{name}/ckpt")
    rows = [r for r in rows if r["step"] in saved]
    with open(f"{RUNS_ROOT}/{name}/args.json") as f:
        vv = TARGET_VARS[json.load(f).get("target", "temp")]
    return min(rows, key=lambda r: sum(r[f"all/{v}/crps"] for v in vv))["step"]


def ckpt_path(name, which):
    """which: 'last', 'best' (best_val_step) or a step number."""
    if which == "last":
        return f"{RUNS_ROOT}/{name}/ckpt/last.ckpt"
    step = best_val_step(name) if which == "best" else int(which)
    return saved_steps(f"{RUNS_ROOT}/{name}/ckpt")[step]


def load_run(name, device, era5_lat, era5_lon, which="last"):
    with open(f"{RUNS_ROOT}/{name}/args.json") as f:
        args = SimpleNamespace(**json.load(f))
    ck = torch.load(ckpt_path(name, which), map_location=device, weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model(args, era5_lat, era5_lon).to(device)
    model.load_state_dict(sd)
    model.eval()
    return args, model, int(ck.get("global_step", -1))


def psd_box(valid_grid, size=512, stride=32):
    """Top-left corner of the size x size box with the highest valid fraction."""
    ii = np.pad(valid_grid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_rc = -1.0, (0, 0)
    H, W = valid_grid.shape
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            s = ii[r + size, c + size] - ii[r, c + size] - ii[r + size, c] + ii[r, c]
            if s > best:
                best, best_rc = s, (r, c)
    return best_rc, best / size ** 2


def radial_psd(field, mask, nbins):
    """field, mask: (L,L) GPU tensors. Mean-fills invalid pixels, Hann window,
    returns radially averaged power (nbins,) over wavenumber in cycles/km."""
    L = field.shape[0]
    f = torch.where(mask, field, field[mask].mean())
    f = f - f.mean()
    w = torch.hann_window(L, device=f.device)
    P = torch.fft.fftshift(torch.abs(torch.fft.fft2(f * w[:, None] * w[None, :])) ** 2)
    k = torch.arange(L, device=f.device) - L // 2
    kr = torch.sqrt(k[:, None].float() ** 2 + k[None, :].float() ** 2)
    b = torch.clamp((kr / (L // 2) * nbins).long(), max=nbins)
    s = torch.zeros(nbins + 1, device=f.device).index_add_(0, b.flatten(), P.flatten())
    n = torch.zeros(nbins + 1, device=f.device).index_add_(0, b.flatten(), torch.ones_like(P).flatten())
    return (s / n.clamp(min=1))[:nbins]


@torch.no_grad()
def condition_all(model, era5, lat, lon, doy, doy_cond, chunk, static=None):
    feat = model.encode(era5[None])
    parts = [model.condition(feat, lat[None, a:a + chunk], lon[None, a:a + chunk],
                             static[None, a:a + chunk] if static is not None else None)
             for a in range(0, lat.shape[0], chunk)]
    cond = torch.cat(parts, dim=1)
    if doy_cond:
        cond = torch.cat([cond, doy_features(doy.view(1))[:, None, :].expand(-1, cond.shape[1], -1)], dim=-1)
    return cond


def make_constraint(cell_id, n_cells, mu0, sigma0, target_cell, min_pixels):
    """Single-sample version of the exact cell-mean projection on channel 0."""
    count = torch.zeros(n_cells, device=mu0.device).index_add_(0, cell_id, torch.ones_like(mu0))
    active = count[cell_id] >= min_pixels

    def project(x):
        phys = x[:, 0] * sigma0 + mu0
        mean = torch.zeros(n_cells, device=x.device).index_add_(0, cell_id, phys) / count.clamp(min=1)
        phys = torch.where(active, phys + (target_cell - mean)[cell_id], phys)
        x = x.clone()
        x[:, 0] = (phys - mu0) / sigma0
        return x

    return project, count, active


@torch.no_grad()
def sample_day(model, cond, lat, lon, args, M, n_steps, chunk, constraint, noise_scale=1.0):
    N = cond.shape[1]
    members = []
    for _ in range(M):
        if args.noise == "field":
            x = noise_field(lat[None], lon[None], model.n_out, chunk=chunk, **noise_kwargs(args))[0]
        else:
            x = torch.randn(N, model.n_out, device=lat.device)
        x = x * noise_scale  # > 1 widens the ensemble (sampling-time spread fix, no retraining)
        for s in range(n_steps):
            v = torch.empty_like(x)
            for a in range(0, N, chunk):
                b = min(N, a + chunk)
                tt = torch.full((1, b - a), s / n_steps, device=x.device)
                if getattr(args, "objective", "fm") == "flowmap":  # jump t -> t + 1/n_steps
                    v[a:b] = model.velocity(x[None, a:b], cond[:, a:b], tt, tt + 1.0 / n_steps)[0]
                else:
                    v[a:b] = model.velocity(x[None, a:b], cond[:, a:b], tt)[0]
            x = x + v / n_steps
            if constraint is not None:
                x = constraint(x)
        members.append(x)
    return torch.stack(members)  # (M,N,3)


@torch.no_grad()
def regress_day(model, cond, chunk):
    N = cond.shape[1]
    C = model.n_out
    out = torch.empty(N, C, device=cond.device)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        out[a:b] = model.velocity(torch.zeros(1, b - a, C, device=cond.device), cond[:, a:b],
                                  torch.zeros(1, b - a, device=cond.device))[0]
    return out[None]


def bilinear_setup(era5_lat, era5_lon, lat, lon, device):
    """Fractional ERA5 grid indices of every NGCD pixel (regular grid, either axis order) ->
    corner indices + weights for bilinear interpolation of a (C, n_lat, n_lon) field."""
    def frac(axis, x):
        order = np.argsort(axis)
        f = np.interp(x, axis[order], order.astype(np.float64))
        i0 = np.clip(np.floor(f), 0, len(axis) - 2).astype(np.int64)
        return i0, np.clip(f - i0, 0.0, 1.0)
    i0, wi = frac(np.asarray(era5_lat, dtype=np.float64), lat.cpu().numpy().astype(np.float64))
    j0, wj = frac(np.asarray(era5_lon, dtype=np.float64), lon.cpu().numpy().astype(np.float64))
    t = lambda v, dt=torch.float32: torch.from_numpy(v).to(device=device, dtype=dt)
    return t(i0, torch.long), t(j0, torch.long), t(wi), t(wj)


def baseline_day(era5, bl):
    """ERA5 standardized anomaly (window channels 0-2 = t2m mean/max/min z-scores w.r.t. ERA5's own
    harmonic climatology), bilinearly interpolated to the NGCD pixels; used directly as the NGCD
    z-score, i.e. pred = mu_NGCD + sigma_NGCD * z_ERA5 -- a BCSD-style bias-corrected interpolation
    (deterministic, no training). era5: (4,H,W) -> (1,N,3)."""
    i0, j0, wi, wj = bl
    e = era5[:3]
    z = ((1 - wi) * (1 - wj) * e[:, i0, j0] + (1 - wi) * wj * e[:, i0, j0 + 1]
         + wi * (1 - wj) * e[:, i0 + 1, j0] + wi * wj * e[:, i0 + 1, j0 + 1])  # (3,N)
    return z.T[None]


def era5_precip_bilinear(era5, bl, lm, ls):
    """ERA5 total precipitation (window channel 3 = standardized log1p mm), bilinear -> mm/day (N,).
    Interpolated in log1p space, as the model sees it."""
    i0, j0, wi, wj = bl
    e = era5[3]
    z = ((1 - wi) * (1 - wj) * e[i0, j0] + (1 - wi) * wj * e[i0, j0 + 1]
         + wi * (1 - wj) * e[i0 + 1, j0] + wi * wj * e[i0 + 1, j0 + 1])
    return torch.expm1(z * ls + lm).clamp(min=0.0)


def cell_offsets(train_years, st, device, n_cells, cell_id, cache):
    """Monthly mean (NGCD TG cell mean - ERA5 t2m_mean) per ERA5 cell over the
    training years, degC, shape (12, n_cells). Cached to `cache`."""
    if os.path.exists(cache):
        return torch.from_numpy(np.load(cache)["offset"]).to(device)
    cm = torch.from_numpy(st["coeffs_mean"][:, 0:1]).to(device)[None]  # (1,N,1,5)
    cv = torch.from_numpy(st["coeffs_var"][:, 0:1]).to(device)[None]
    count = torch.zeros(n_cells, device=device).index_add_(0, cell_id, torch.ones(len(cell_id), device=device))
    s = torch.zeros(12, n_cells, device=device)
    n = torch.zeros(12, 1, device=device)
    for y in train_years:
        meta = dict(np.load(f"{DATA_DIR}/year_{y}_meta.npz"))  # materialize once (npz re-reads per access)
        tgt = np.load(f"{DATA_DIR}/year_{y}_targets.npy", mmap_mode="r")
        for i, ds in enumerate(meta["dates"]):
            m = int(str(ds)[5:7]) - 1
            z = torch.from_numpy(np.array(tgt[i, :, 0])).to(device).float()
            mu, sig = mu_sigma(cm, cv, torch.tensor([int(meta["doy"][i])], device=device), float(st["sigma_floor"]))
            phys = torch.nan_to_num(z * sig[0, :, 0] + mu[0, :, 0])
            mean = torch.zeros(n_cells, device=device).index_add_(0, cell_id, phys) / count.clamp(min=1)
            era5 = torch.from_numpy(meta["era5_t2m_phys"][i].reshape(-1)).to(device)
            s[m] += mean - era5
            n[m] += 1
        print(f"  offsets: year {y} done", flush=True)
    off = (s / n.clamp(min=1)).cpu().numpy()
    np.savez(cache, offset=off, train_years=np.array(train_years))
    return torch.from_numpy(off).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--constraint", choices=["none", "exact", "exact_debiased"], default="none")
    ap.add_argument("--constraint-min-pixels", type=int, default=64)
    ap.add_argument("--day-stride", type=int, default=1)
    ap.add_argument("--psd-every", type=int, default=7, help="PSD on every k-th evaluated day")
    ap.add_argument("--map-dates", default="2019-01-15,2019-07-28,2019-11-06")
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt", default="last",
                    help="'last', 'best' (lowest val-year CRPS among saved checkpoints) or a step number")
    ap.add_argument("--target", choices=["temp", "precip"], default="temp", help="for --baseline only (runs know theirs)")
    ap.add_argument("--edm-steps", type=int, default=18, help="CorrDiff-style runs: Heun steps (2*steps-1 UNet calls)")
    ap.add_argument("--baseline", choices=["bcsd", "era5_bilinear"], default=None,
                    help="evaluate a no-training baseline instead of a run (bcsd: ERA5 anomaly, bilinear, "
                         "rescaled by the NGCD climatology); --run then just names the output dir")
    ap.add_argument("--region", choices=sorted(HOLDOUT_REGIONS), default=None,
                    help="also report metrics inside/outside this region (automatic for hold-out runs)")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="multiply the initial noise; > 1 widens an under-dispersed ensemble")
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    a = ap.parse_args()

    dev = pick_device(a.device)
    t0 = time.time()
    st = np.load(f"{DATA_DIR}/static.npz")
    lat = torch.from_numpy(st["valid_lat"]).to(dev)
    lon = torch.from_numpy(st["valid_lon"]).to(dev)
    rows, cols = st["valid_rows"], st["valid_cols"]
    cell_id = torch.from_numpy(st["era5_cell_id"]).to(dev).long()
    n_cells = len(st["era5_lat"]) * len(st["era5_lon"])
    coeffs_mean = torch.from_numpy(st["coeffs_mean"]).to(dev)[None]  # (1,N,3,5)
    coeffs_var = torch.from_numpy(st["coeffs_var"]).to(dev)[None]
    floor = float(st["sigma_floor"])
    H, W = [int(v) for v in st["grid_shape"]]
    N = lat.shape[0]

    is_grid = False
    if not a.baseline and os.path.exists(f"{RUNS_ROOT}/{a.run}/args.json"):
        with open(f"{RUNS_ROOT}/{a.run}/args.json") as f:
            is_grid = json.load(f).get("objective") in ("unet_reg", "edm")
    if is_grid:  # CorrDiff-style grid baseline (diffusion_baseline.py)
        from diffusion_baseline import FullDomain, load_grid_run, sample_grid_day
        args, gnet, greg, step = load_grid_run(a.run, dev, a.ckpt)
        model = None
    elif a.baseline:  # no model: --run only names the output directory
        args = SimpleNamespace(objective="baseline", baseline=a.baseline, holdout=None, doy_cond=False,
                               train_years="", noise="none", noise_ls=0.0, target=a.target)
        if (a.baseline == "bcsd") != (a.target == "temp"):
            raise SystemExit("--baseline bcsd is for --target temp, era5_bilinear for --target precip")
        model, step = None, 0
        os.makedirs(f"{RUNS_ROOT}/{a.run}", exist_ok=True)
        with open(f"{RUNS_ROOT}/{a.run}/args.json", "w") as f:
            json.dump(vars(args), f)
        bl = bilinear_setup(st["era5_lat"], st["era5_lon"], lat, lon, dev)
    else:
        args, model, step = load_run(a.run, dev, st["era5_lat"], st["era5_lon"], a.ckpt)
    regression = args.objective in ("regression", "baseline", "unet_reg")
    target = getattr(args, "target", "temp")
    precip = target == "precip"
    VV = TARGET_VARS[target]
    C = len(VV)
    rr = PrecipTransform(DATA_DIR, dev) if precip else None
    allpix = torch.arange(N, device=dev)
    if precip and a.constraint != "none":
        raise SystemExit("ERA5 constraints are for TG; use --constraint none with precipitation")
    if a.baseline == "era5_bilinear":
        rs = np.load(CDFFM_ROOT + "/norm_stats.npz")
        e5p = (float(rs["era5_precip_log1p_mean"]), float(rs["era5_precip_log1p_std"]))
    static = None
    if getattr(args, "static", "none") != "none":
        static = static_features(coeffs_mean[0], coeffs_var[0], cell_id, n_cells, floor,
                                 p_dry=rr.p_dry if args.static == "clim_rr" else None)
    if is_grid:
        fd = FullDomain(None, lat, lon, static, st["era5_lat"], st["era5_lon"],
                        stride=2 ** (len(args.mult.split(",")) - 1))
    M = 1 if regression else a.members
    tag = a.tag or (f"M{M}_s{a.ode_steps}_{a.constraint}" if not regression else args.objective)
    out_dir = f"{RUNS_ROOT}/{a.run}/eval_{a.year}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] run={a.run} step={step} objective={args.objective} M={M} ode_steps={a.ode_steps} "
          f"constraint={a.constraint} ckpt={a.ckpt} noise_scale={a.noise_scale} -> {out_dir}", flush=True)

    held = torch.zeros(N, dtype=torch.bool, device=dev)
    region = getattr(args, "holdout", None) or a.region  # --region: same split for a model trained on all pixels
    if region:
        r = HOLDOUT_REGIONS[region]
        held = (lat >= r["lat_min"]) & (lon >= r["lon_min"])
    groups = {"all": torch.ones(N, dtype=torch.bool, device=dev)}
    if held.any():
        groups["holdout"], groups["inside"] = held, ~held
    G, S = len(groups), len(SEASONS)

    meta = dict(np.load(f"{DATA_DIR}/year_{a.year}_meta.npz"))  # materialize once (npz re-reads per access)
    tgt = np.load(f"{DATA_DIR}/year_{a.year}_{'rr' if precip else 'targets'}.npy", mmap_mode="r")
    dates = [str(d) for d in meta["dates"]]
    days = list(range(0, len(dates), a.day_stride))
    map_dates = set(a.map_dates.split(",")) if a.map_dates else set()
    for md in map_dates:
        if md in dates and dates.index(md) not in days:
            days.append(dates.index(md))
    days.sort()

    offsets = None
    if a.constraint == "exact_debiased":
        tyears = [int(y) for y in args.train_years.split(",")]
        offsets = cell_offsets(tyears, st, dev, n_cells, cell_id,
                               f"{DATA_DIR}/offset_tg_month_{'_'.join(map(str, tyears))}.npz")

    z = lambda: torch.zeros(G, S, C, device=dev)
    acc_keys = ["n", "se1", "seM", "ae", "bias", "crps", "crps_fair", "spread",
                "hot_hit", "hot_miss", "hot_fa", "hot_brier", "hot_n",
                "cold_hit", "cold_miss", "cold_fa", "cold_brier"]
    if precip:  # absolute-threshold metrics: obs / forecast (member 0) counts, hits, Brier of the ensemble prob.
        acc_keys += [f"t{t}_{k}" for t in RR_THRESHOLDS_MM for k in ("obs", "fc", "hit", "fa", "brier")]
    acc = {k: z() for k in acc_keys}
    rank_hist = torch.zeros(G, S, C, M + 1, device=dev)
    resid = {k: [] for k in ("pred_vs_truth", "pred_raw", "truth_raw", "pred_debiased", "truth_debiased")}

    valid_grid = np.zeros((H, W), dtype=bool)
    valid_grid[rows, cols] = True
    (br, bc), frac = psd_box(valid_grid)
    L, NB = 512, 128
    in_box = (rows >= br) & (rows < br + L) & (cols >= bc) & (cols < bc + L)
    box_idx = torch.from_numpy(np.where(in_box)[0]).to(dev)
    box_r = torch.from_numpy(rows[in_box] - br).to(dev).long()
    box_c = torch.from_numpy(cols[in_box] - bc).to(dev).long()
    box_mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
    box_mask[box_r, box_c] = True
    psd = {k: torch.zeros(C, NB, device=dev) for k in ("truth", "member", "ensmean")}
    n_psd = 0
    daily = []
    print(f"[eval] PSD box rows {br}:{br + L} cols {bc}:{bc + L}, valid fraction {frac:.3f}", flush=True)

    for it, d in enumerate(days):
        torch.manual_seed(a.seed * 100000 + d)
        ds = dates[d]
        month = int(ds[5:7])
        s_idx = SEASONS.index(SEASON_OF_MONTH[month])
        doy = torch.tensor([int(meta["doy"][d])], device=dev)
        era5 = torch.from_numpy(meta["era5"][d]).to(dev)
        truth_z = torch.from_numpy(np.array(tgt[d])).to(dev).float()  # (N,C)
        ok = torch.isfinite(truth_z)
        truth_z = torch.nan_to_num(truth_z)
        if precip:
            truth = rr.to_mm(truth_z[:, 0], allpix)[:, None]  # mm/day; NaN where p_dry is missing
            ok = ok & torch.isfinite(truth)
            truth = torch.nan_to_num(truth)
            project = None
        else:
            mu, sig = mu_sigma(coeffs_mean, coeffs_var, doy, floor)
            mu, sig = mu[0], sig[0]  # (N,3)
            truth = truth_z * sig + mu
            era5_cell = torch.from_numpy(meta["era5_t2m_phys"][d].reshape(-1)).to(dev)
            target_cell = era5_cell + (offsets[month - 1] if offsets is not None else 0.0)
            project, count, active = make_constraint(cell_id, n_cells, mu[:, 0], sig[:, 0], target_cell,
                                                     a.constraint_min_pixels)
        if a.baseline == "era5_bilinear":
            ens_mm = era5_precip_bilinear(era5, bl, *e5p)
            ens_z = torch.nan_to_num(rr.to_z(ens_mm, allpix))[None, :, None]
        elif a.baseline:
            ens_z = baseline_day(era5, bl)
        elif is_grid:
            ens_z = sample_grid_day(args, gnet, greg, fd, era5, doy, M, steps=a.edm_steps)
        elif regression:
            cond = condition_all(model, era5, lat, lon, doy, args.doy_cond, a.chunk, static)
            ens_z = regress_day(model, cond, a.chunk)
        else:
            cond = condition_all(model, era5, lat, lon, doy, args.doy_cond, a.chunk, static)
            ens_z = sample_day(model, cond, lat, lon, args, M, a.ode_steps, a.chunk,
                               project if a.constraint != "none" else None, a.noise_scale)
        if precip:
            ens = torch.nan_to_num(rr.to_mm(ens_z[..., 0], allpix))[..., None]  # (M,N,1) mm/day
            if a.baseline == "era5_bilinear":
                ens = ens_mm[None, :, None]  # raw interpolated amounts, not round-tripped
        else:
            ens = ens_z * sig[None] + mu[None]  # (M,N,3) physical
        mean = ens.mean(0)

        e1, eM = ens[0] - truth, mean - truth
        crps = (ens - truth[None]).abs().mean(0)
        crps_fair = crps
        if M > 1:
            pair = torch.zeros_like(truth)
            for i in range(M):
                pair += (ens[i][None] - ens).abs().sum(0)
            # crps: standard ensemble estimator (biased high for small M, by ~E|X-X'|/(2M));
            # crps_fair: unbiased for the CRPS of the underlying distribution (Ferro 2014) --
            # comparable across ensemble sizes and with deterministic models.
            crps_fair = crps - 0.5 * pair / (M * (M - 1))
            crps = crps - 0.5 * pair / (M * M)
        spread = ens.std(0, unbiased=False) if M > 1 else torch.zeros_like(truth)
        rank = (ens < truth[None]).sum(0)  # (N,3) in 0..M
        hot_o, cold_o = truth_z > EXTREME_Z, truth_z < -EXTREME_Z
        hot_f, cold_f = ens_z[0] > EXTREME_Z, ens_z[0] < -EXTREME_Z
        hot_p = (ens_z > EXTREME_Z).float().mean(0)
        cold_p = (ens_z < -EXTREME_Z).float().mean(0)

        for gi, gm in enumerate(groups.values()):
            m = (ok & gm[:, None]).float()
            for si in (0, s_idx):
                acc["n"][gi, si] += m.sum(0)
                acc["se1"][gi, si] += (e1 ** 2 * m).sum(0)
                acc["seM"][gi, si] += (eM ** 2 * m).sum(0)
                acc["ae"][gi, si] += (eM.abs() * m).sum(0)
                acc["bias"][gi, si] += (eM * m).sum(0)
                acc["crps"][gi, si] += (crps * m).sum(0)
                acc["crps_fair"][gi, si] += (crps_fair * m).sum(0)
                acc["spread"][gi, si] += (spread * m).sum(0)
                acc["hot_hit"][gi, si] += ((hot_f & hot_o).float() * m).sum(0)
                acc["hot_miss"][gi, si] += ((~hot_f & hot_o).float() * m).sum(0)
                acc["hot_fa"][gi, si] += ((hot_f & ~hot_o).float() * m).sum(0)
                acc["hot_brier"][gi, si] += (((hot_p - hot_o.float()) ** 2) * m).sum(0)
                acc["cold_hit"][gi, si] += ((cold_f & cold_o).float() * m).sum(0)
                acc["cold_miss"][gi, si] += ((~cold_f & cold_o).float() * m).sum(0)
                acc["cold_fa"][gi, si] += ((cold_f & ~cold_o).float() * m).sum(0)
                acc["cold_brier"][gi, si] += (((cold_p - cold_o.float()) ** 2) * m).sum(0)
                for c in range(C):
                    sel = m[:, c] > 0
                    rank_hist[gi, si, c] += torch.bincount(rank[sel, c], minlength=M + 1).float()
                if precip:
                    for t in RR_THRESHOLDS_MM:
                        o_, f_ = truth > t, ens[0] > t
                        acc[f"t{t}_obs"][gi, si] += (o_.float() * m).sum(0)
                        acc[f"t{t}_fc"][gi, si] += (f_.float() * m).sum(0)
                        acc[f"t{t}_hit"][gi, si] += ((o_ & f_).float() * m).sum(0)
                        acc[f"t{t}_fa"][gi, si] += ((~o_ & f_).float() * m).sum(0)
                        pr = (ens > t).float().mean(0)
                        acc[f"t{t}_brier"][gi, si] += (((pr - o_.float()) ** 2) * m).sum(0)

        if not precip:  # TG cell-mean residual vs ERA5 (raw and debiased), predictions vs truth
            act_cells = torch.unique(cell_id[active])
            def cell_mean(v):
                return (torch.zeros(n_cells, device=dev).index_add_(0, cell_id, v) / count.clamp(min=1))[act_cells]
            pm, tm = cell_mean(ens[0, :, 0]), cell_mean(torch.where(ok[:, 0], truth[:, 0], mu[:, 0]))
            resid["pred_vs_truth"].append(float((pm - tm).abs().mean()))  # the cell-scale error that matters
            resid["pred_raw"].append(float((pm - era5_cell[act_cells]).abs().mean()))
            resid["truth_raw"].append(float((tm - era5_cell[act_cells]).abs().mean()))
            if offsets is not None:
                tc = target_cell[act_cells]
                resid["pred_debiased"].append(float((pm - tc).abs().mean()))
                resid["truth_debiased"].append(float((tm - tc).abs().mean()))

        if it % a.psd_every == 0:
            for c in range(C):
                for k, fld in (("truth", truth), ("member", ens[0]), ("ensmean", mean)):
                    grid = torch.zeros(L, L, device=dev)
                    grid[box_r, box_c] = fld[box_idx, c]
                    psd[k][c] += radial_psd(grid, box_mask, NB)
            n_psd += 1

        if ds in map_dates:
            save_maps(f"{out_dir}/maps_{ds}.png", ds, rows, cols, H, W, truth, ens, ok, VV, precip)
            # Raw fields at the valid pixels (degC, fp16, ~40 MB) for cross-run comparison figures
            # (compare_eval.py); pixel order = static.npz valid_rows/valid_cols.
            def _h(v):
                return v.detach().cpu().numpy().astype(np.float16)
            np.savez(f"{out_dir}/fields_{ds}.npz", truth=_h(torch.where(ok, truth, torch.full_like(truth, float("nan")))),
                     member0=_h(ens[0]), member1=_h(ens[min(1, M - 1)]), ensmean=_h(mean), ensstd=_h(ens.std(0) if M > 1 else torch.zeros_like(mean)))
        # per-day domain means (all pixels) for bootstrap confidence intervals (bootstrap_ci.py)
        okf = ok.float()
        nd = okf.sum(0).clamp(min=1)
        daily.append({"date": ds, "n": nd.tolist(),
                      "crps_fair": ((crps_fair * okf).sum(0) / nd).tolist(),
                      "se_ensmean": ((eM ** 2 * okf).sum(0) / nd).tolist(),
                      "ae_ensmean": ((eM.abs() * okf).sum(0) / nd).tolist()})
        if it % 10 == 0:
            print(f"  [{it + 1}/{len(days)}] {ds} {VV[0]} crps(day)={float((crps[:, 0] * ok[:, 0]).sum() / ok[:, 0].sum()):.2f} "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    metrics = {"run": a.run, "step": step, "year": a.year, "n_days": len(days), "members": M,
               "ode_steps": a.ode_steps, "constraint": a.constraint, "ckpt": a.ckpt, "noise_scale": a.noise_scale,
               "args": vars(args), "groups": {}}
    for gi, g in enumerate(groups):
        metrics["groups"][g] = {}
        for si, sname in enumerate(SEASONS):
            metrics["groups"][g][sname] = {}
            for c, v in enumerate(VV):
                n = float(acc["n"][gi, si, c].clamp(min=1))
                rmseM = math.sqrt(float(acc["seM"][gi, si, c]) / n)
                sp = float(acc["spread"][gi, si, c]) / n
                csi = lambda h, mi, fa: float(h / max(float(h + mi + fa), 1.0))
                metrics["groups"][g][sname][v] = {
                    "n": n,
                    "rmse_1member": math.sqrt(float(acc["se1"][gi, si, c]) / n),
                    "rmse_ensmean": rmseM, "mae": float(acc["ae"][gi, si, c]) / n,
                    "bias": float(acc["bias"][gi, si, c]) / n, "crps": float(acc["crps"][gi, si, c]) / n,
                    "crps_fair": float(acc["crps_fair"][gi, si, c]) / n,
                    "spread": sp, "spread_skill": sp / max(rmseM, 1e-6),
                    "hot_csi": csi(acc["hot_hit"][gi, si, c], acc["hot_miss"][gi, si, c], acc["hot_fa"][gi, si, c]),
                    "hot_brier": float(acc["hot_brier"][gi, si, c]) / n,
                    "cold_csi": csi(acc["cold_hit"][gi, si, c], acc["cold_miss"][gi, si, c], acc["cold_fa"][gi, si, c]),
                    "cold_brier": float(acc["cold_brier"][gi, si, c]) / n,
                    "rank_hist": (rank_hist[gi, si, c] / rank_hist[gi, si, c].sum().clamp(min=1)).tolist(),
                }
                if precip:
                    for t in RR_THRESHOLDS_MM:
                        g_ = lambda k: float(acc[f"t{t}_{k}"][gi, si, c])
                        metrics["groups"][g][sname][v][f">{t}mm"] = {
                            "obs_freq": g_("obs") / n, "fc_freq": g_("fc") / n,
                            "freq_bias": g_("fc") / max(g_("obs"), 1.0),
                            "csi": g_("hit") / max(g_("obs") + g_("fa"), 1.0),  # hit/(hit+miss+fa)
                            "brier": g_("brier") / n}
    metrics["tg_cell_mean_abs_residual_degC"] = {k: (float(np.mean(v)) if v else None) for k, v in resid.items()}
    metrics["daily"] = daily
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    k_cpkm = (np.arange(NB) + 0.5) / NB * 0.5  # bin centres, cycles/km (Nyquist 0.5 at 1 km)
    np.savez(f"{out_dir}/psd.npz", k_cycles_per_km=k_cpkm, vars=np.array(VV),
             **{k: (v / max(n_psd, 1)).cpu().numpy() for k, v in psd.items()})
    save_psd_plot(f"{out_dir}/psd.png", k_cpkm, {k: (v / max(n_psd, 1)).cpu().numpy() for k, v in psd.items()}, VV)

    print(f"\n=== {a.run} | {a.year} | {len(days)} days | {tag} ===", flush=True)
    for g in groups:
        y = metrics["groups"][g]["year"]
        print(f"[{g}] " + " | ".join(
            f"{v}: crps={y[v]['crps']:.2f} fair={y[v]['crps_fair']:.2f} rmseM={y[v]['rmse_ensmean']:.2f} rmse1={y[v]['rmse_1member']:.2f} "
            f"bias={y[v]['bias']:+.2f} ss={y[v]['spread_skill']:.2f} hotCSI={y[v]['hot_csi']:.2f} "
            f"coldCSI={y[v]['cold_csi']:.2f}" for v in VV), flush=True)
        if precip:
            print("      " + " | ".join(
                f">{t}mm: obs {y['RR'][f'>{t}mm']['obs_freq']:.3f} freqbias {y['RR'][f'>{t}mm']['freq_bias']:.2f} "
                f"CSI {y['RR'][f'>{t}mm']['csi']:.2f} Brier {y['RR'][f'>{t}mm']['brier']:.4f}"
                for t in RR_THRESHOLDS_MM), flush=True)
    if not precip:
        print("TG cell-mean |residual| vs ERA5 (degC): " + json.dumps(metrics["tg_cell_mean_abs_residual_degC"]), flush=True)
    print(f"[done] {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


def save_maps(path, ds, rows, cols, H, W, truth, ens, ok, vv=VARS, precip=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid(v):
        g = np.full((H, W), np.nan, dtype=np.float32)
        g[rows, cols] = v.detach().cpu().numpy()
        return g

    M = ens.shape[0]
    panels = [("truth", truth), ("sample 1", ens[0])]
    if M > 1:
        panels += [("sample 2", ens[1]), ("ensemble mean", ens.mean(0)), ("ensemble std", ens.std(0))]
    panels += [("error (ens mean - truth)", ens.mean(0) - truth)]
    fig, axes = plt.subplots(len(vv), len(panels), figsize=(3.2 * len(panels), 3.4 * len(vv)), squeeze=False)
    for c, v in enumerate(vv):
        t = truth[:, c][ok[:, c]]
        lo, hi = float(torch.quantile(t, 0.01)), float(torch.quantile(t, 0.99))
        if precip:
            lo, hi = 0.0, max(float(torch.quantile(t, 0.995)), 1.0)
        for j, (title, fld) in enumerate(panels):
            ax = axes[c][j]
            g = grid(fld[:, c])
            if title.startswith("error"):
                e = float(np.nanpercentile(np.abs(g), 99))
                im = ax.imshow(g, cmap="RdBu_r", vmin=-e, vmax=e)
            elif title == "ensemble std":
                im = ax.imshow(g, cmap="viridis")
            else:
                im = ax.imshow(g, cmap="YlGnBu" if precip else "RdYlBu_r", vmin=lo, vmax=hi)
            ax.set_title(f"{v} {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(ds)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def save_psd_plot(path, k, psd, vv=VARS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(vv), figsize=(5 * len(vv), 4), squeeze=False)
    axes = axes[0]
    for c, v in enumerate(vv):
        for name in ("truth", "member", "ensmean"):
            axes[c].loglog(k[1:], psd[name][c][1:], label=name)
        axes[c].set_title(f"{v} power spectrum")
        axes[c].set_xlabel("wavenumber (cycles/km)")
        axes[c].grid(alpha=0.3, which="both")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
