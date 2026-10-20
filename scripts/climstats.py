"""
climstats.py
============
Shared math for the normalization statistics: harmonic (seasonal)
mean/std climatology, and a SIMPLIFIED (single global fit, not per-cell/
harmonic/regionally-pooled) GPD tail for precipitation.

The harmonic part reuses only the Pass-1 idea from
Extreme_weather/fcn_finetune/compute_gpd_stats.py (plain OLS regression for
seasonal mean/variance) -- not its EVT tail machinery, which stays out of
scope per feedback_no_gpd_downscaling_target.md. The GPD part here is a
deliberately scoped-down reimplementation (one global sigma/xi, not
per-cell with lat-band shape pooling) -- see that same memory file for why.
"""
import numpy as np

HARMONIC_K = 2
N_BASIS = 2 * HARMONIC_K + 1


def basis(doy):
    """doy: (N,) day-of-year in [0,365). Returns (N, N_BASIS)."""
    ang = 2.0 * np.pi * doy.astype(np.float64) / 365.0
    cols = [np.ones_like(ang)]
    for k in range(1, HARMONIC_K + 1):
        cols.append(np.cos(k * ang))
        cols.append(np.sin(k * ang))
    return np.stack(cols, axis=1)


SIGMA_FLOOR = 0.5  # degC -- real daily anomaly std is well above this; a safety net, not a regular clamp


def fit_harmonic(values, doys):
    """values: (N, ...spatial) -- one sample per day-of-year in doys: (N,).
    NaN-valued cells (e.g. NGCD's out-of-domain FillValue mask, constant
    across days) propagate to NaN coefficients for that cell, which is the
    desired behavior -- no separate masking needed.
    Returns coeffs_mean, coeffs_var, each (...spatial, N_BASIS) float32:
    harmonic fits of the mean and of the RESIDUAL variance.

    The variance is fitted to the residuals (x - mean_fit(doy))^2 directly,
    NOT as E[x^2]_fit - mean_fit^2: mean_fit^2 contains harmonics up to
    order 2*HARMONIC_K that the E[x^2] fit cannot represent, so that
    difference went negative wherever the true variance is small (confirmed:
    19-28% of NGCD temperature pixels in mid-July clipped to sigma=0.03,
    giving z-targets up to ~300). Fitting the residual variance directly has
    no such cancellation."""
    # Stay in float32 for the big (N,M) arrays (a float64 copy OOM-killed a
    # 16GB job once); only the tiny (5x5) normal-equations matrix needs
    # float64 for a stable inverse.
    b32 = basis(doys).astype(np.float32)  # (N, N_BASIS)
    b64 = b32.astype(np.float64)
    BtB_inv = np.linalg.inv(b64.T @ b64).astype(np.float32)
    spatial_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)  # view, no copy
    coeffs_mean = BtB_inv @ (b32.T @ flat)  # (N_BASIS, M)
    resid = flat - b32 @ coeffs_mean  # (N, M) float32
    np.square(resid, out=resid)  # in place: peak memory = values + one residual array
    coeffs_var = BtB_inv @ (b32.T @ resid)
    del resid
    coeffs_mean = coeffs_mean.T.reshape(spatial_shape + (N_BASIS,))
    coeffs_var = coeffs_var.T.reshape(spatial_shape + (N_BASIS,))
    return coeffs_mean.astype(np.float32), coeffs_var.astype(np.float32)


def eval_harmonic(coeffs_mean, coeffs_var, doy, sigma_floor=SIGMA_FLOOR):
    """doy: scalar (whole-grid case, coeffs_mean shape (...,5)) OR (N,) array
    (per-point case, coeffs_mean shape (N,5), one doy per gathered point) --
    broadcasting handles both. coeffs_var: residual-variance fit from
    fit_harmonic (NOT the old E[x^2] fit -- see fit_harmonic's docstring).
    Returns mu, sigma with doy's batch shape; sigma floored at sigma_floor."""
    doy = np.asarray(doy, dtype=np.float64)
    ang = 2.0 * np.pi * doy / 365.0
    b = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang),
                  np.cos(2 * ang), np.sin(2 * ang)], axis=-1)  # (...,5)
    mu = np.sum(coeffs_mean.astype(np.float64) * b, axis=-1)
    var = np.sum(coeffs_var.astype(np.float64) * b, axis=-1)
    sigma = np.sqrt(np.maximum(var, sigma_floor ** 2))
    return mu.astype(np.float32), sigma.astype(np.float32)


def doy_cos_sin(doy):
    """First-harmonic (cos,sin) encoding of day-of-year -- continuous
    across the Dec31->Jan1 boundary, unlike raw doy. Same angle convention
    as basis()/eval_harmonic() above, reused here as the canonical
    "calendar time" feature for model INPUT conditioning (distinct from
    those functions' own use of doy, which is for fitting/evaluating the
    seasonal normalization climatology, not for feeding the network
    directly). Returns (cos, sin) as float32 scalars or arrays matching
    doy's shape."""
    doy = np.asarray(doy, dtype=np.float64)
    ang = 2.0 * np.pi * doy / 365.0
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def lmoment_gpd_fit(exceedances, threshold):
    """Single GLOBAL fit (not per-cell) via L-moments (Hosking & Wallis 1987).
    exceedances: 1D array, values ABOVE threshold (already subtracted:
    pass the excess e = x - threshold, not x itself). Returns (sigma, xi)."""
    e = np.sort(exceedances.astype(np.float64))  # ascending
    k = len(e)
    i = np.arange(1, k + 1, dtype=np.float64)
    b0 = e.mean()
    b1 = np.mean(((i - 1) / (k - 1)) * e)
    b2 = np.mean((((i - 1) * (i - 2)) / ((k - 1) * (k - 2))) * e)
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    tau3 = l3 / l2 if l2 != 0 else 0.0
    k_hat = (1 - 3 * tau3) / (1 + tau3)
    xi = -k_hat
    sigma = (1 + k_hat) * (2 + k_hat) * l2
    return float(sigma), float(xi)


def gpd_forward_excess(e, sigma, xi):
    """e >= 0 excess over threshold -> -log(survival). Scalar sigma/xi (global fit)."""
    if abs(xi) < 1e-6:
        return e / sigma
    ratio = np.clip(xi * e / sigma, -1 + 1e-9, None)
    return np.log1p(ratio) / xi


def gpd_inverse_excess(surv, sigma, xi):
    """surv = (1-F)/zeta in (0,1] -> excess e >= 0. Scalar sigma/xi."""
    surv = np.clip(surv, 1e-12, None)
    if abs(xi) < 1e-6:
        return -sigma * np.log(surv)
    return (sigma / xi) * (np.power(surv, -xi) - 1.0)


def bulk_quantile_table(values, levels):
    """values: 1D pooled array (e.g. all wet-day amounts across sampled
    dates/cells). levels: e.g. np.arange(5,96,5). Returns (len(levels),)
    knot array -- ONE global table, not per-cell/per-region."""
    return np.percentile(values, levels).astype(np.float64)
