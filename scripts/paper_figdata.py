"""
paper_figdata.py
=================
GPU data for the paper's teaser figure and Figure 1 (plots: paper_figs.py, CPU).

  teaser    For a few 2019 dates: 16-member ensembles of the final temperature model (distilled flow
            map, 2 steps) and of the precipitation model, on all 1.33M pixels, + NGCD truth, ERA5 inputs
            in physical units, and per-pixel exact log-likelihood of the observed TG/TX/TN.
            -> figures/paper_data/teaser_<date>.npz
  multires  Resolution-free sampling: ONE sample (fixed noise realisation) of the final model queried on
            nested point grids of 8, 4, 2, 1, 0.5 and 0.25 km over a 64 x 64 km box in western Norway.
            Locations, static descriptors and climatology are bilinear in grid-index space; the noise field
            and the model are evaluated at the points themselves. Checks that coincident points agree.
            -> figures/paper_data/multires_<date>.npz

  python3 paper_figdata.py teaser --dates 2019-07-28,2019-01-15 --auto-wet 1
  python3 paper_figdata.py multires --date 2019-07-28
"""
import argparse
import math
import os

import numpy as np
import torch

from fast_data import (pick_device, sync, DATA_DIR, PrecipTransform, doy_features, mu_sigma, noise_field, noise_kwargs,
                       ode_sample, static_features)
from fast_eval import condition_all, load_run, sample_day
from fast_logscore import log_density_z
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
OUT = f"{ROOT}/figures/paper_data"
TEMP_RUN, TEMP_MAP_RUN, RR_RUN = "w2e_static_pl24_l8", "w5_flowmap", "w4_rr_final"


def common(dev):
    st = np.load(f"{DATA_DIR}/static.npz")
    g = dict(st=st, lat=torch.from_numpy(st["valid_lat"]).to(dev), lon=torch.from_numpy(st["valid_lon"]).to(dev),
             cell_id=torch.from_numpy(st["era5_cell_id"]).to(dev).long(),
             n_cells=len(st["era5_lat"]) * len(st["era5_lon"]),
             cm=torch.from_numpy(st["coeffs_mean"]).to(dev), cv=torch.from_numpy(st["coeffs_var"]).to(dev),
             floor=float(st["sigma_floor"]))
    return g


def era5_phys(meta, d):
    """ERA5 TG/TX/TN (degC) and total precipitation (mm) on the 71x113 grid for day index d."""
    import climstats
    ns = np.load(f"{ROOT}/norm_stats.npz")
    doy = int(meta["doy"][d])
    t = []
    for c, ch in enumerate(["t2m_mean_degC", "t2m_max_degC", "t2m_min_degC"]):
        mu, sg = climstats.eval_harmonic(ns[f"era5_{ch}_coeffs_mean"], ns[f"era5_{ch}_coeffs_var"], doy)
        t.append(meta["era5"][d, c] * sg + mu)
    tp = np.expm1(meta["era5"][d, 3] * float(ns["era5_precip_log1p_std"]) + float(ns["era5_precip_log1p_mean"]))
    return np.stack(t).astype(np.float32), np.clip(tp, 0, None).astype(np.float32)


@torch.no_grad()
def teaser(a, dev):
    g = common(dev)
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    dates = [str(x) for x in meta["dates"]]
    tt = np.load(f"{DATA_DIR}/year_2019_targets.npy", mmap_mode="r")
    tr = np.load(f"{DATA_DIR}/year_2019_rr.npy", mmap_mode="r")
    rr = PrecipTransform(DATA_DIR, dev)
    allpix = torch.arange(g["lat"].shape[0], device=dev)
    want = [d for d in a.dates.split(",") if d]
    if a.auto_wet:  # add the 2019 day(s) with the largest area of heavy (> 20 mm) precipitation
        frac = []
        for d in range(len(dates)):
            mm = rr.to_mm(torch.from_numpy(np.array(tr[d, :, 0])).to(dev).float(), allpix)
            frac.append(float((mm > 20).float().nanmean()))
        for d in np.argsort(frac)[::-1][:a.auto_wet]:
            print(f"[teaser] heavy-precip day {dates[d]}: {100 * frac[d]:.1f}% of pixels > 20 mm", flush=True)
            want.append(dates[d])
    targs, tmodel, _ = load_run(TEMP_MAP_RUN, dev, g["st"]["era5_lat"], g["st"]["era5_lon"], "best")
    largs, lmodel, _ = load_run(TEMP_RUN, dev, g["st"]["era5_lat"], g["st"]["era5_lon"], "best")  # for log-lik
    for p in lmodel.parameters():
        p.requires_grad_(False)
    rargs, rmodel, _ = load_run(RR_RUN, dev, g["st"]["era5_lat"], g["st"]["era5_lon"], "best")
    stat_t = static_features(g["cm"], g["cv"], g["cell_id"], g["n_cells"], g["floor"])
    stat_r = static_features(g["cm"], g["cv"], g["cell_id"], g["n_cells"], g["floor"], p_dry=rr.p_dry)
    os.makedirs(OUT, exist_ok=True)
    for ds in want:
        d = dates.index(ds)
        torch.manual_seed(1000 + d)
        doy = torch.tensor([int(meta["doy"][d])], device=dev)
        era5 = torch.from_numpy(meta["era5"][d]).to(dev)
        mu, sig = mu_sigma(g["cm"][None], g["cv"][None], doy, g["floor"])
        mu, sig = mu[0], sig[0]
        z_true = torch.from_numpy(np.array(tt[d])).to(dev).float()
        ok = torch.isfinite(z_true).all(-1)
        # temperature: flow map, 2 steps, 16 members
        cond = condition_all(tmodel, era5, g["lat"], g["lon"], doy, targs.doy_cond, a.chunk, stat_t)
        ens_z = sample_day(tmodel, cond, g["lat"], g["lon"], targs, a.members, 2, a.chunk, None)
        ens_t = ens_z * sig[None] + mu[None]
        # exact per-pixel log-likelihood of the observation (z-space, 20 RK4 steps) -> degC units
        condl = condition_all(lmodel, era5, g["lat"], g["lon"], doy, largs.doy_cond, a.chunk, stat_t)[0]
        ll = torch.full((g["lat"].shape[0],), float("nan"), device=dev)
        zt = torch.nan_to_num(z_true)
        for s0 in range(0, zt.shape[0], a.chunk):
            s1 = min(zt.shape[0], s0 + a.chunk)
            with torch.enable_grad():
                ll[s0:s1] = log_density_z(lmodel, zt[s0:s1], condl[s0:s1], 20)
        ll = torch.where(ok, ll - torch.log(sig).sum(-1), torch.full_like(ll, float("nan")))
        # precipitation: Euler 10 steps, 16 members
        condr = condition_all(rmodel, era5, g["lat"], g["lon"], doy, rargs.doy_cond, a.chunk, stat_r)
        ens_rz = sample_day(rmodel, condr, g["lat"], g["lon"], rargs, a.members, 10, a.chunk, None)
        ens_r = rr.to_mm(ens_rz[..., 0], allpix)
        true_r = rr.to_mm(torch.from_numpy(np.array(tr[d, :, 0])).to(dev).float(), allpix)
        e5t, e5p = era5_phys(meta, d)
        h = lambda x: x.detach().cpu().numpy().astype(np.float16)
        np.savez(f"{OUT}/teaser_{ds}.npz", date=ds, truth_t=h(torch.where(ok[:, None], z_true * sig + mu, torch.full_like(mu, float("nan")))),
                 ens_t=h(ens_t), loglik_t=h(ll), truth_r=h(true_r), ens_r=h(ens_r), era5_t=e5t, era5_tp=e5p)
        print(f"[teaser] {ds}: TG ens-mean MAE {float((ens_t.mean(0) - (z_true * sig + mu))[ok][:, 0].abs().mean()):.2f} degC, "
              f"mean log-lik {float(ll[ok].mean()):.2f} nats, RR truth mean {float(true_r.nanmean()):.2f} mm", flush=True)


def bilinear_grid(arr, r, c):
    """arr: (H,W,...) tensor; r, c: (n,) fractional indices -> (n,...); NaN if any corner is NaN."""
    H, W = arr.shape[:2]
    r0 = r.floor().long().clamp(0, H - 2)
    c0 = c.floor().long().clamp(0, W - 2)
    wr, wc = (r - r0).clamp(0, 1), (c - c0).clamp(0, 1)
    ex = (slice(None),) + (None,) * (arr.dim() - 2)
    return ((1 - wr)[ex] * (1 - wc)[ex] * arr[r0, c0] + (1 - wr)[ex] * wc[ex] * arr[r0, c0 + 1]
            + wr[ex] * (1 - wc)[ex] * arr[r0 + 1, c0] + wr[ex] * wc[ex] * arr[r0 + 1, c0 + 1])


@torch.no_grad()
def multires(a, dev):
    g = common(dev)
    st = g["st"]
    H, W = [int(v) for v in st["grid_shape"]]
    rows, cols = torch.from_numpy(st["valid_rows"].astype(np.int64)).to(dev), torch.from_numpy(st["valid_cols"].astype(np.int64)).to(dev)
    ns = np.load(f"{ROOT}/norm_stats.npz")
    glat, glon = (torch.from_numpy(ns[k].astype(np.float64)).to(dev) for k in ("ngcd_lat", "ngcd_lon"))

    def full(v):  # (N,...) at valid pixels -> (H,W,...) grid with NaN elsewhere
        out = torch.full((H, W) + tuple(v.shape[1:]), float("nan"), dtype=torch.float64, device=dev)
        out[rows, cols] = v.double()
        return out

    stat = static_features(g["cm"], g["cv"], g["cell_id"], g["n_cells"], g["floor"])
    stat_g, cm_g, cv_g = full(stat), full(g["cm"]), full(g["cv"])
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    dates = [str(x) for x in meta["dates"]]
    d = dates.index(a.date)
    doy = torch.tensor([int(meta["doy"][d])], device=dev)
    era5 = torch.from_numpy(meta["era5"][d]).to(dev)
    targs, model, _ = load_run(TEMP_MAP_RUN, dev, st["era5_lat"], st["era5_lon"], "best")
    feat = model.encode(era5[None])
    # box: 64 x 64 km around the target location (grid index space, 1 km pixels)
    i = int(np.argmin((st["valid_lat"] - a.lat) ** 2 + ((st["valid_lon"] - a.lon) * np.cos(np.radians(a.lat))) ** 2))
    r_lo, c_lo = float(int(st["valid_rows"][i] - a.box // 2)), float(int(st["valid_cols"][i] - a.box // 2))
    tt = np.load(f"{DATA_DIR}/year_2019_targets.npy", mmap_mode="r")
    mu1, sig1 = mu_sigma(g["cm"][None], g["cv"][None], doy, g["floor"])
    truth_full = full((torch.from_numpy(np.array(tt[d])).to(dev).float() * sig1[0] + mu1[0]))
    out = {"date": a.date, "box_rows": np.array([r_lo, r_lo + a.box]), "box_cols": np.array([c_lo, c_lo + a.box])}
    # corner-aligned, exactly nested grids: every point of a coarser grid is also a point of all finer ones,
    # and the 1 km grid coincides with the NGCD pixel centres
    for res in [8, 4, 2, 1, 0.5, 0.25]:
        n = int(round(a.box / res))
        centres = torch.arange(n, device=dev, dtype=torch.float64) * res
        rr_, cc_ = torch.meshgrid(r_lo + centres, c_lo + centres, indexing="ij")
        r, c = rr_.reshape(-1), cc_.reshape(-1)
        lat, lon = bilinear_grid(glat, r, c).float(), bilinear_grid(glon, r, c).float()
        sv, cmv, cvv = bilinear_grid(stat_g, r, c).float(), bilinear_grid(cm_g, r, c).float(), bilinear_grid(cv_g, r, c).float()
        land = torch.isfinite(sv).all(-1) & torch.isfinite(cmv).flatten(1).all(-1)
        cond = model.condition(feat, lat[None], lon[None], torch.nan_to_num(sv)[None])
        cond = torch.cat([cond, doy_features(doy)[:, None, :].expand(-1, cond.shape[1], -1)], dim=-1)
        gen = torch.Generator(device=dev).manual_seed(a.seed)  # same noise realisation at every resolution
        z0 = noise_field(lat[None], lon[None], 3, generator=gen, **noise_kwargs(targs))
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        x = ode_sample(model, cond, lat[None], lon[None], 2, noise=targs.noise, noise_kw=noise_kwargs(targs),
                       generator=gen, flowmap=True)[0]
        mu, sig = mu_sigma(torch.nan_to_num(cmv)[None], torch.nan_to_num(cvv, nan=1.0)[None], doy, g["floor"])
        phys = torch.where(land[:, None], x * sig[0] + mu[0], torch.full_like(x, float("nan")))
        key = f"{res:g}km".replace(".", "p")
        out[f"sample_{key}"] = phys.reshape(n, n, 3).cpu().numpy().astype(np.float32)
        out[f"noise_{key}"] = z0[0, :, 0].reshape(n, n).cpu().numpy().astype(np.float32)
        out[f"lat_{key}"], out[f"lon_{key}"] = lat.reshape(n, n).cpu().numpy(), lon.reshape(n, n).cpu().numpy()
        if res == 1:
            tr_ = truth_full[(rr_.round().long().clamp(0, H - 1)), (cc_.round().long().clamp(0, W - 1))]
            out["truth_1km"] = tr_.cpu().numpy().astype(np.float32)
        print(f"[multires] {res:g} km: {n}x{n} points, land {float(land.float().mean()):.2f}", flush=True)
    # Kolmogorov consistency, empirically: a location gets the same value whatever other points are queried
    keys = ["8km", "4km", "2km", "1km", "0p5km", "0p25km"]
    res_of = dict(zip(keys, [8, 4, 2, 1, 0.5, 0.25]))
    fine = out["sample_0p25km"]
    for k in keys[:-1]:
        step = int(round(res_of[k] / 0.25))
        dmax = np.nanmax(np.abs(out[f"sample_{k}"] - fine[::step, ::step]))
        print(f"[multires] max |{k} sample - 0.25 km sample at the same points| = {dmax:.2e} degC", flush=True)
        out[f"consistency_max_abs_{k}"] = np.float32(dmax)
    np.savez(f"{OUT}/multires_{a.date}.npz", **out)
    print(f"[multires] wrote {OUT}/multires_{a.date}.npz", flush=True)


@torch.no_grad()
def multires_full(a, dev):
    """The whole Nordic domain at 16, 8, 4, 2 and 1 km, ONE sample (fixed noise realisation), each resolution
    queried separately. Grids are corner-aligned on the NGCD grid, so the points of a k-km grid are the valid 1 km
    pixels with row % k == 0 and col % k == 0 (all on the training grid). Times each query."""
    import time
    g = common(dev)
    st = g["st"]
    rows_all, cols_all = st["valid_rows"], st["valid_cols"]
    stat = static_features(g["cm"], g["cv"], g["cell_id"], g["n_cells"], g["floor"])
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    d = [str(x) for x in meta["dates"]].index(a.date)
    doy = torch.tensor([int(meta["doy"][d])], device=dev)
    era5 = torch.from_numpy(meta["era5"][d]).to(dev)
    targs, model, _ = load_run(TEMP_MAP_RUN, dev, st["era5_lat"], st["era5_lon"], "best")
    mu_all, sig_all = mu_sigma(g["cm"][None], g["cv"][None], doy, g["floor"])
    out = {"date": a.date, "grid_shape": st["grid_shape"]}
    # untimed warm-up: the first GPU call includes one-off kernel compilation / MIOpen tuning (~25 s)
    wsel = torch.from_numpy(np.where((rows_all % 16 == 0) & (cols_all % 16 == 0))[0]).to(dev)
    wc = condition_all(model, era5, g["lat"][wsel], g["lon"][wsel], doy, targs.doy_cond, a.chunk, stat[wsel])
    wx = noise_field(g["lat"][wsel][None], g["lon"][wsel][None], 3, chunk=a.chunk, **noise_kwargs(targs))[0]
    t0w = torch.zeros((1, wx.shape[0]), device=dev)
    model.velocity(wx[None], wc, t0w, t0w + 0.5)
    sync(dev)
    for res in [16, 8, 4, 2, 1]:
        sel = np.where((rows_all % res == 0) & (cols_all % res == 0))[0]
        idx = torch.from_numpy(sel).to(dev)
        sync(dev)
        t0 = time.time()
        lat, lon = g["lat"][idx], g["lon"][idx]
        cond = condition_all(model, era5, lat, lon, doy, targs.doy_cond, a.chunk, stat[idx])  # (1,n,D)
        gen = torch.Generator(device=dev).manual_seed(a.seed)  # same noise realisation at every resolution
        x = noise_field(lat[None], lon[None], 3, generator=gen, chunk=a.chunk, **noise_kwargs(targs))[0]
        for k in range(2):  # flow map, 2 steps
            t = torch.full((1, a.chunk), k / 2, device=dev)
            for s0 in range(0, x.shape[0], a.chunk):
                s1 = min(x.shape[0], s0 + a.chunk)
                x[s0:s1] = x[s0:s1] + 0.5 * model.velocity(x[None, s0:s1], cond[:, s0:s1], t[:, :s1 - s0], t[:, :s1 - s0] + 0.5)[0]
        phys = x * sig_all[0, idx] + mu_all[0, idx]
        sync(dev)
        dt = time.time() - t0
        key = f"{res}km"
        out[f"val_{key}"] = phys.cpu().numpy().astype(np.float32)
        out[f"rows_{key}"], out[f"cols_{key}"] = rows_all[sel].astype(np.int32), cols_all[sel].astype(np.int32)
        out[f"seconds_{key}"] = np.float32(dt)
        print(f"[multires_full] {res:2d} km: {len(sel):8d} points, {dt:.2f} s (encode + query + 2 flow-map steps)", flush=True)
    # consistency with the 1 km field at the shared points
    v1 = {(r, c): i for i, (r, c) in enumerate(zip(out["rows_1km"], out["cols_1km"]))}
    for res in [16, 8, 4, 2]:
        k = f"{res}km"
        j = np.array([v1[(r, c)] for r, c in zip(out[f"rows_{k}"], out[f"cols_{k}"])])
        print(f"[multires_full] max |{k} - 1 km at shared points| = {np.abs(out[f'val_{k}'] - out['val_1km'][j]).max():.2e} degC", flush=True)
    np.savez_compressed(f"{OUT}/multires_full_{a.date}.npz", **out)
    print(f"[multires_full] wrote {OUT}/multires_full_{a.date}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teaser", "multires", "multires_full"])
    ap.add_argument("--dates", default="2019-07-28,2019-01-15")
    ap.add_argument("--auto-wet", type=int, default=1, help="teaser: also the N 2019 days with most heavy precip")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--date", default="2019-07-28")
    ap.add_argument("--lat", type=float, default=61.5)
    ap.add_argument("--lon", type=float, default=7.0)
    ap.add_argument("--box", type=float, default=64.0, help="multires box side (km)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    a = ap.parse_args()
    dev = pick_device(a.device)
    os.makedirs(OUT, exist_ok=True)  # both modes write here (multires runs first and must create it)
    {"teaser": teaser, "multires": multires, "multires_full": multires_full}[a.mode](a, dev)


if __name__ == "__main__":
    main()
