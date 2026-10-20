"""
compute_norm_stats.py
======================
Computes all normalization statistics needed before training:
  - ERA5 temperature: per-cell harmonic mu(doy)/sigma(doy) (ERA5's own grid)
  - NGCD temperature: per-cell harmonic mu(doy)/sigma(doy) (NGCD's own grid)
  - ERA5 precip: global log1p(mm) mean/std (lighter treatment -- input feature only)
  - NGCD precip: per-cell p_dry + ONE global bulk quantile table + ONE global
    GPD tail fit (sigma, xi) -- the simplified scope chosen over the full
    per-cell/harmonic/regionally-pooled recipe, see
    memory/feedback_no_gpd_downscaling_target.md

Fit ONLY on the training split (1981-2015) per
functional_flow_downscaling_plan.md §8 -- validation/test years must never
leak into these statistics.

Memory: NGCD streaming reads and the harmonic regression's (N,2020,1550)
arrays need tens of GB of RAM.
"""
import argparse
import datetime
import os

import numpy as np

import climstats
import ngcd_io
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ERA5_DIR = CDFFM_ROOT + "/era5_nordic_daily"
OUT_PATH = CDFFM_ROOT + "/norm_stats.npz"

TRAIN_YEARS = list(range(1981, 2016))  # matches plan §8's temporal split
WET_THRESHOLD_MM = 0.1  # standard meteorological "wet day" convention
BULK_LEVELS = np.arange(5, 96, 5, dtype=np.float64)
GPD_THRESHOLD_Q = 0.95
BULK_POOL_CAP = 2_000_000


def _report_floor(coeffs_var, label):
    """Fraction of cells where the residual-variance fit falls under
    climstats.SIGMA_FLOOR, per sample doy -- should be ~0 everywhere; the
    old E[x^2]-mean^2 fit hit its clip on 19-28% of NGCD pixels in July."""
    cv = coeffs_var.reshape(-1, climstats.N_BASIS).astype(np.float64)
    cv = cv[~np.isnan(cv[:, 0])]
    parts = []
    for doy in (15, 75, 135, 196, 257, 318):
        var = cv @ climstats.basis(np.array([doy]))[0]
        parts.append(f"doy{doy}={np.mean(var < climstats.SIGMA_FLOOR ** 2):.1e}")
    print(f"    {label} sigma-floor-binding frac: " + " ".join(parts), flush=True)


def era5_temp_harmonic():
    print("=== ERA5 temperature harmonic fit ===", flush=True)
    out = {}
    for ch in ["t2m_mean_degC", "t2m_max_degC", "t2m_min_degC"]:
        vals, doys = [], []
        for y in TRAIN_YEARS:
            d = np.load(f"{ERA5_DIR}/{y}.npz")
            dates = d["dates"]
            doy = np.array([
                (datetime.datetime.strptime(str(s), "%Y-%m-%d").date() - datetime.date(int(str(s)[:4]), 1, 1)).days
                for s in dates
            ])
            vals.append(d[ch])
            doys.append(doy)
        vals = np.concatenate(vals, axis=0)
        doys = np.concatenate(doys, axis=0)
        coeffs_mean, coeffs_var = climstats.fit_harmonic(vals, doys)
        out[f"era5_{ch}_coeffs_mean"] = coeffs_mean
        out[f"era5_{ch}_coeffs_var"] = coeffs_var
        print(f"  {ch}: fit over {len(doys)} samples, grid {coeffs_mean.shape[:-1]}", flush=True)
        _report_floor(coeffs_var, ch)
    return out


def era5_precip_lognormal():
    print("=== ERA5 precip log1p mean/std ===", flush=True)
    pooled = []
    for y in TRAIN_YEARS:
        d = np.load(f"{ERA5_DIR}/{y}.npz")
        pooled.append(np.log1p(d["tp_sum_mm"]).ravel())
    pooled = np.concatenate(pooled)
    mean, std = float(pooled.mean()), float(pooled.std())
    print(f"  log1p(precip) mean={mean:.4f} std={std:.4f} (n={len(pooled)})", flush=True)
    return {"era5_precip_log1p_mean": mean, "era5_precip_log1p_std": std}


def ngcd_temp_harmonic(n_samples, seed):
    print("=== NGCD temperature harmonic fit ===", flush=True)
    rng = np.random.default_rng(seed)
    dates = ngcd_io.sample_dates(TRAIN_YEARS, n_samples, rng)
    doys = np.array([(d - datetime.date(d.year, 1, 1)).days for d in dates])

    out = {}
    for var in ["TG", "TX", "TN"]:
        vals = np.empty((len(dates),) + ngcd_io.GRID_SHAPE, dtype=np.float32)
        zf_cache = {}
        for i, d in enumerate(dates):
            vals[i] = ngcd_io.load_ngcd_day(var, d, zf_cache=zf_cache)
        for zf in zf_cache.values():
            zf.close()
        coeffs_mean, coeffs_var = climstats.fit_harmonic(vals, doys)
        out[f"ngcd_{var}_coeffs_mean"] = coeffs_mean
        out[f"ngcd_{var}_coeffs_var"] = coeffs_var
        print(f"  {var}: fit over {len(dates)} sampled dates", flush=True)
        _report_floor(coeffs_var, var)
        del vals
    return out


def ngcd_precip_gpd(n_samples, seed):
    print("=== NGCD precip: p_dry + bulk table + global GPD tail ===", flush=True)
    rng = np.random.default_rng(seed)
    dates = ngcd_io.sample_dates(TRAIN_YEARS, n_samples, rng)

    wet_count = None
    valid_count = None
    wet_pool = []

    zf_cache = {}
    for d in dates:
        arr = ngcd_io.load_ngcd_day("RR", d, zf_cache=zf_cache)
        valid = ~np.isnan(arr)
        wet = valid & (arr > WET_THRESHOLD_MM)
        if wet_count is None:
            wet_count = np.zeros(arr.shape, dtype=np.int64)
            valid_count = np.zeros(arr.shape, dtype=np.int64)
        wet_count += wet.astype(np.int64)
        valid_count += valid.astype(np.int64)
        wet_vals = arr[wet]
        if len(wet_vals) > 0:
            # subsample this day's contribution before pooling, keeps the
            # running pool bounded without biasing toward later days
            if len(wet_vals) > 50_000:
                wet_vals = rng.choice(wet_vals, size=50_000, replace=False)
            wet_pool.append(wet_vals.astype(np.float32))
    for zf in zf_cache.values():
        zf.close()

    p_dry = np.where(valid_count > 0, 1.0 - wet_count / np.maximum(valid_count, 1), np.nan).astype(np.float32)

    pooled = np.concatenate(wet_pool)
    if len(pooled) > BULK_POOL_CAP:
        pooled = rng.choice(pooled, size=BULK_POOL_CAP, replace=False)
    pooled = pooled.astype(np.float64)

    bulk_knots = climstats.bulk_quantile_table(pooled, BULK_LEVELS)
    threshold = float(np.percentile(pooled, GPD_THRESHOLD_Q * 100))
    exceedances = pooled[pooled > threshold] - threshold
    sigma, xi = climstats.lmoment_gpd_fit(exceedances, threshold)

    print(f"  n sampled dates={len(dates)}, pooled wet values={len(pooled)}", flush=True)
    print(f"  threshold(mm)={threshold:.3f} sigma={sigma:.3f} xi={xi:.3f}", flush=True)
    print(f"  p_dry: mean={np.nanmean(p_dry):.3f} over valid cells", flush=True)

    return {
        "ngcd_precip_p_dry": p_dry,
        "ngcd_precip_bulk_levels": BULK_LEVELS,
        "ngcd_precip_bulk_knots": bulk_knots,
        "ngcd_precip_gpd_threshold": threshold,
        "ngcd_precip_gpd_sigma": sigma,
        "ngcd_precip_gpd_xi": xi,
        "ngcd_precip_wet_threshold_mm": WET_THRESHOLD_MM,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngcd-temp-samples", type=int, default=500)
    ap.add_argument("--ngcd-precip-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-era5", action="store_true",
                     help="NGCD-only run (needs 1981-2015 fully valid in ngcd_raw/)")
    ap.add_argument("--skip-ngcd", action="store_true",
                     help="ERA5-only run -- usable as soon as era5_nordic_daily/ is ready, "
                          "independent of NGCD download status")
    args = ap.parse_args()
    if args.skip_era5 and args.skip_ngcd:
        raise SystemExit("--skip-era5 and --skip-ngcd together would compute nothing")

    # Merge into whatever's already there, so an ERA5-only run now and an
    # NGCD-only run later (once 2014/2015 land) combine into one complete
    # file without either clobbering the other's keys.
    stats = {}
    if os.path.exists(OUT_PATH):
        with np.load(OUT_PATH) as f:
            # drop the old E[x^2] fits -- superseded by *_coeffs_var (see
            # climstats.fit_harmonic); keeping them would just invite misuse
            stats = {k: f[k] for k in f.files if not k.endswith("_coeffs_sq")}
        print(f"merging into existing {OUT_PATH} ({len(stats)} keys already present)", flush=True)

    if not args.skip_era5:
        stats.update(era5_temp_harmonic())
        stats.update(era5_precip_lognormal())
    if not args.skip_ngcd:
        stats.update(ngcd_temp_harmonic(args.ngcd_temp_samples, args.seed))
        stats.update(ngcd_precip_gpd(args.ngcd_precip_samples, args.seed + 1))
        lat, lon = ngcd_io.get_static_latlon()
        stats["ngcd_lat"] = lat
        stats["ngcd_lon"] = lon

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, **stats)
    print(f"\nwrote {OUT_PATH} ({len(stats)} keys total)")


if __name__ == "__main__":
    main()
