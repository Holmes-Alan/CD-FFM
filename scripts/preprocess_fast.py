"""
preprocess_fast.py
===================
One-time preprocessing for the fast (GPU-resident) pipeline -- fast_data.py /
fast_train.py. Replaces per-sample zip+HDF5 decompression (0.74 s/sample,
98% of the old step time) with arrays that are loaded onto the GPU once.

  --static      static.npz: valid-pixel coordinates, nearest ERA5 cell per
                pixel, per-pixel climatology coefficients (to denormalize on
                the GPU), ERA5 grid.
  --year YYYY   year_YYYY_targets.npy  (n_days, n_valid, 3) float16, NGCD
                TG/TX/TN z-scores (NaN where NGCD is missing that day)
                year_YYYY_meta.npz     era5 (n_days,4,H,W) normalized window
                (same channels/order as dataset.py's _era5_window),
                era5_t2m_phys (n_days,H,W) degC, doy, dates.
  --precip-static  static_rr.npz: per-valid-pixel p_dry + the global bulk/GPD
                parameters of the NGCD precipitation transform (transforms.py).
  --precip-year YYYY  year_YYYY_rr.npy (n_days, n_valid, 1) float16: NGCD RR
                Gaussianized, z = Phi^-1(F), dry days at z0 = Phi^-1(p_dry)
                (NaN where NGCD or p_dry is missing). Same pixel order as --static.

Uses the residual-variance climatology (norm_stats.npz with *_coeffs_var
keys) -- refuses to run on the old E[x^2]-based file. Plain system python3
(numpy/h5py only; Python 3.6-compatible on purpose).
"""
import argparse
import datetime
import os

import numpy as np

import climstats
import ngcd_io
import transforms
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

STATS_PATH = CDFFM_ROOT + "/norm_stats.npz"
ERA5_DIR = CDFFM_ROOT + "/era5_nordic_daily"
OUT_DIR = CDFFM_ROOT + "/fast_data"
TEMP_VARS = ["TG", "TX", "TN"]
ERA5_TEMP = ["t2m_mean_degC", "t2m_max_degC", "t2m_min_degC"]


def load_stats():
    with np.load(STATS_PATH) as f:
        stats = {k: f[k] for k in f.files}
    if "ngcd_TG_coeffs_var" not in stats:
        raise SystemExit(f"{STATS_PATH} has no *_coeffs_var keys -- it is the old (broken-variance) "
                         f"file; run compute_norm_stats.py first.")
    return stats


def valid_mask(stats):
    """Pixels with finite climatology for all three temperature variables.
    Shared by --static and --year so both use the identical pixel ordering
    (row-major np.where over this mask)."""
    m = np.ones(stats["ngcd_lat"].shape, dtype=bool)
    for v in TEMP_VARS:
        m &= np.isfinite(stats[f"ngcd_{v}_coeffs_mean"]).all(-1)
        m &= np.isfinite(stats[f"ngcd_{v}_coeffs_var"]).all(-1)
    return m


def doy_of(date_str):
    d = datetime.datetime.strptime(str(date_str), "%Y-%m-%d").date()
    return d, (d - datetime.date(d.year, 1, 1)).days


def nearest_index(grid, values):
    """Nearest cell centre on a regular 1D grid (ascending or descending)."""
    step = grid[1] - grid[0]
    assert np.allclose(np.diff(grid), step, atol=1e-4), "ERA5 grid is not regular"
    idx = np.rint((values - grid[0]) / step).astype(np.int64)
    return np.clip(idx, 0, len(grid) - 1)


def build_static(stats):
    mask = valid_mask(stats)
    rows, cols = np.where(mask)
    with np.load(f"{ERA5_DIR}/2015.npz") as f:
        era5_lat = f["lat"].astype(np.float64)
        era5_lon = f["lon"].astype(np.float64)
    lat = stats["ngcd_lat"][rows, cols].astype(np.float64)
    lon = stats["ngcd_lon"][rows, cols].astype(np.float64)
    cell = nearest_index(era5_lat, lat) * len(era5_lon) + nearest_index(era5_lon, lon)
    out = {
        "valid_rows": rows.astype(np.int32),
        "valid_cols": cols.astype(np.int32),
        "valid_lat": lat.astype(np.float32),
        "valid_lon": lon.astype(np.float32),
        "era5_cell_id": cell.astype(np.int32),
        "era5_lat": era5_lat.astype(np.float32),
        "era5_lon": era5_lon.astype(np.float32),
        # (n_valid, 3, 5): TG,TX,TN harmonic coefficients for GPU-side denormalization
        "coeffs_mean": np.stack([stats[f"ngcd_{v}_coeffs_mean"][rows, cols] for v in TEMP_VARS], 1).astype(np.float32),
        "coeffs_var": np.stack([stats[f"ngcd_{v}_coeffs_var"][rows, cols] for v in TEMP_VARS], 1).astype(np.float32),
        "sigma_floor": np.float32(climstats.SIGMA_FLOOR),
        "grid_shape": np.array(stats["ngcd_lat"].shape, dtype=np.int32),
    }
    path = f"{OUT_DIR}/static.npz"
    np.savez(path, **out)
    print(f"wrote {path}: n_valid={len(rows)}, ERA5 grid {len(era5_lat)}x{len(era5_lon)}, "
          f"distinct ERA5 cells hit={len(np.unique(cell))}", flush=True)


def build_year(stats, year):
    mask = valid_mask(stats)
    rows, cols = np.where(mask)
    cm = {v: stats[f"ngcd_{v}_coeffs_mean"][rows, cols] for v in TEMP_VARS}
    cv = {v: stats[f"ngcd_{v}_coeffs_var"][rows, cols] for v in TEMP_VARS}

    with np.load(f"{ERA5_DIR}/{year}.npz") as f:
        era5 = {k: f[k] for k in f.files}
    dates = [str(s) for s in era5["dates"]]
    n_days, n_valid = len(dates), len(rows)

    targets = np.empty((n_days, n_valid, 3), dtype=np.float16)
    era5_win = np.empty((n_days, 4) + era5["t2m_mean_degC"].shape[1:], dtype=np.float32)
    doys = np.empty(n_days, dtype=np.int16)
    zf_cache = {}
    nan_frac, zmax, z2 = [], 0.0, []
    for i, ds in enumerate(dates):
        d, doy = doy_of(ds)
        doys[i] = doy
        for c, v in enumerate(TEMP_VARS):
            x = ngcd_io.load_ngcd_day(v, d, zf_cache=zf_cache)[rows, cols]
            mu, sigma = climstats.eval_harmonic(cm[v], cv[v], doy)
            z = (x - mu) / sigma
            targets[i, :, c] = z.astype(np.float16)
            finite = np.isfinite(z)
            nan_frac.append(1.0 - finite.mean())
            if finite.any():
                zmax = max(zmax, float(np.abs(z[finite]).max()))
                z2.append(float(np.mean(z[finite] ** 2)))
        for c, ch in enumerate(ERA5_TEMP):
            mu, sigma = climstats.eval_harmonic(stats[f"era5_{ch}_coeffs_mean"], stats[f"era5_{ch}_coeffs_var"], doy)
            era5_win[i, c] = (era5[ch][i] - mu) / sigma
        era5_win[i, 3] = ((np.log1p(era5["tp_sum_mm"][i]) - float(stats["era5_precip_log1p_mean"]))
                          / float(stats["era5_precip_log1p_std"]))
        if i % 60 == 0:
            print(f"  {ds}: done", flush=True)
    for zf in zf_cache.values():
        zf.close()

    np.save(f"{OUT_DIR}/year_{year}_targets.npy", targets)
    np.savez(f"{OUT_DIR}/year_{year}_meta.npz", era5=era5_win,
             era5_t2m_phys=era5["t2m_mean_degC"].astype(np.float32),
             doy=doys, dates=np.array(dates))
    print(f"wrote year {year}: {n_days} days x {n_valid} pixels | NGCD NaN frac mean={np.mean(nan_frac):.2e} "
          f"| z: max|z|={zmax:.2f}, mean E[z^2] per day/var={np.mean(z2):.3f} (expect ~1) "
          f"| ERA5 window max|z|={np.nanmax(np.abs(era5_win[:, :3])):.2f}", flush=True)


RR_KEYS = ["p_dry", "bulk_levels", "bulk_knots", "gpd_threshold", "gpd_sigma", "gpd_xi", "wet_threshold_mm"]


def rr_params(stats, rows, cols):
    pr = {k: stats[f"ngcd_precip_{k}"] for k in RR_KEYS}
    pr["p_dry"] = pr["p_dry"][rows, cols].astype(np.float64)
    return pr


def build_precip_static(stats):
    rows, cols = np.where(valid_mask(stats))
    pr = rr_params(stats, rows, cols)
    path = f"{OUT_DIR}/static_rr.npz"
    np.savez(path, **pr)
    print(f"wrote {path}: p_dry mean={np.nanmean(pr['p_dry']):.3f}, NaN p_dry at "
          f"{int(np.isnan(pr['p_dry']).sum())} of {len(rows)} valid pixels", flush=True)


def build_precip_year(stats, year):
    rows, cols = np.where(valid_mask(stats))
    pr = rr_params(stats, rows, cols)
    dates = ngcd_io.year_dates(year)
    out = np.empty((len(dates), len(rows), 1), dtype=np.float16)
    zf_cache = {}
    wet, z2 = [], []
    for i, d in enumerate(dates):
        x = ngcd_io.load_ngcd_day("RR", d, zf_cache=zf_cache)[rows, cols].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(pr["p_dry"])
        z = np.full(len(rows), np.nan, dtype=np.float32)
        z[ok] = transforms.normalize_ngcd_precip(
            x[ok], pr["p_dry"][ok], pr["bulk_levels"], pr["bulk_knots"], float(pr["gpd_threshold"]),
            float(pr["gpd_sigma"]), float(pr["gpd_xi"]), wet_threshold_mm=float(pr["wet_threshold_mm"]))
        out[i, :, 0] = z.astype(np.float16)
        wet.append(float(np.mean(x[ok] > float(pr["wet_threshold_mm"]))))
        z2.append(float(np.mean(z[ok] ** 2)))
        if i % 60 == 0:
            print(f"  {d}: wet fraction {wet[-1]:.2f}", flush=True)
    for zf in zf_cache.values():
        zf.close()
    np.save(f"{OUT_DIR}/year_{year}_rr.npy", out)
    print(f"wrote year {year} RR: {len(dates)} days | wet-day fraction {np.mean(wet):.3f} "
          f"(1 - mean p_dry = {1 - np.nanmean(pr['p_dry']):.3f}) | mean E[z^2] {np.mean(z2):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--precip-static", action="store_true")
    ap.add_argument("--precip-year", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = load_stats()
    if args.static:
        build_static(stats)
    if args.year is not None:
        build_year(stats, args.year)
    if args.precip_static:
        build_precip_static(stats)
    if args.precip_year is not None:
        build_precip_year(stats, args.precip_year)


if __name__ == "__main__":
    main()
