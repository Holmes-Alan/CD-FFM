"""
transforms.py
==============
Forward (physical -> training space) and inverse (training space ->
physical) transforms, consuming norm_stats.npz (compute_norm_stats.py).

Pure elementwise math on already-gathered per-point parameter arrays --
this module doesn't know about grids or coordinates. The caller (dataset /
inference code) is responsible for looking up each query point's
coeffs_mean/coeffs_var/p_dry etc. from the stats grids (e.g. via direct
NGCD row/col indexing during training, nearest/bilinear lookup for
arbitrary continuous queries at inference).

Temperature: per-point harmonic z-score, z = (x - mu(doy)) / sigma(doy).
Precipitation: p_dry-jump + global bulk-quantile-table + global GPD tail,
all folded into ONE continuous transform -- see
functional_flow_downscaling_plan.md §3 for the derivation. No separate
wet/dry classifier: z <= z0 = Phi^-1(p_dry) IS "dry," by construction of
the inverse.
"""
import numpy as np
from scipy.special import erf, erfinv

import climstats


# ---------------------------------------------------------------------------
# Temperature (same functions for ERA5 input and NGCD target -- only the
# coeffs passed in differ, each dataset fit independently)
# ---------------------------------------------------------------------------

def normalize_temp(x, coeffs_mean, coeffs_var, doy):
    """x, coeffs_mean, coeffs_var: (N,) / (N,5) per-point arrays already
    gathered at the query locations. doy: scalar or (N,)."""
    mu, sigma = climstats.eval_harmonic(coeffs_mean, coeffs_var, doy)
    return (x - mu) / sigma


def denormalize_temp(z, coeffs_mean, coeffs_var, doy):
    mu, sigma = climstats.eval_harmonic(coeffs_mean, coeffs_var, doy)
    return z * sigma + mu


# ---------------------------------------------------------------------------
# ERA5 precip (input feature only -- lighter treatment, no GPD)
# ---------------------------------------------------------------------------

def normalize_era5_precip(x_mm, log1p_mean, log1p_std):
    return (np.log1p(x_mm) - log1p_mean) / log1p_std


def denormalize_era5_precip(z, log1p_mean, log1p_std):
    return np.expm1(z * log1p_std + log1p_mean)


# ---------------------------------------------------------------------------
# NGCD precip (generative target -- p_dry jump + global bulk/GPD)
# ---------------------------------------------------------------------------

def _phi_inv(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def normalize_ngcd_precip(x_mm, p_dry, bulk_levels, bulk_knots, gpd_threshold, gpd_sigma, gpd_xi,
                           wet_threshold_mm=0.1):
    """x_mm, p_dry: (N,) per-point arrays (p_dry gathered at each point's
    NGCD cell; bulk/gpd params are GLOBAL scalars, same for every point).
    wet_threshold_mm MUST match compute_norm_stats.py's WET_THRESHOLD_MM --
    that's the threshold p_dry/the bulk table were fit against, so "dry"
    here has to mean the same thing or trace amounts get forced through a
    bulk table that was never fit on values that small. Returns z: (N,)."""
    x_mm = np.asarray(x_mm, dtype=np.float64)
    p_dry = np.asarray(p_dry, dtype=np.float64)
    zeta_hi = 1.0 - bulk_levels[-1] / 100.0
    bulk_frac = bulk_levels / 100.0  # in [0,1], the G-space fraction at each knot

    is_dry = x_mm <= wet_threshold_mm
    is_tail = x_mm > gpd_threshold
    is_bulk = ~is_dry & ~is_tail

    G = np.zeros_like(x_mm)
    if np.any(is_bulk):
        G[is_bulk] = np.interp(x_mm[is_bulk], bulk_knots, bulk_frac)
    if np.any(is_tail):
        e = x_mm[is_tail] - gpd_threshold
        surv = np.exp(-climstats.gpd_forward_excess(e, gpd_sigma, gpd_xi))
        G[is_tail] = 1.0 - zeta_hi * surv

    F = np.where(is_dry, p_dry, p_dry + (1.0 - p_dry) * G)
    return _phi_inv(F).astype(np.float32)


def denormalize_ngcd_precip(z, p_dry, bulk_levels, bulk_knots, gpd_threshold, gpd_sigma, gpd_xi,
                             boundary_eps=1e-4):
    """Inverse: z <= z0=Phi^-1(p_dry) snaps to exactly 0 (the "dry" decision
    is implicit in this threshold, not a separate classifier).

    boundary_eps: z always arrives as float32 in practice (model output),
    so F=Phi(z) recomputed here in float64 is NOT bit-exact to the p_dry
    used when z0 was originally derived -- an exact F<=p_dry check flips
    the wrong way right at the boundary for every "should be dry" point
    (confirmed: 100% of them, via a round-trip test). A small tolerance
    is also the practically-correct behavior for genuine model samples
    landing near (not exactly at) z0, not just a float-precision patch."""
    z = np.asarray(z, dtype=np.float64)
    p_dry = np.asarray(p_dry, dtype=np.float64)
    F = _phi(z)
    zeta_hi = 1.0 - bulk_levels[-1] / 100.0
    bulk_frac = bulk_levels / 100.0

    is_dry = F <= p_dry + boundary_eps
    G = np.clip((F - p_dry) / np.clip(1.0 - p_dry, 1e-9, None), 0.0, 1.0)
    is_tail = G > bulk_frac[-1]
    is_bulk = ~is_dry & ~is_tail

    x_mm = np.zeros_like(z)
    if np.any(is_bulk):
        x_mm[is_bulk] = np.interp(G[is_bulk], bulk_frac, bulk_knots)
    if np.any(is_tail):
        surv = np.clip((1.0 - G[is_tail]) / zeta_hi, 1e-12, None)
        e = climstats.gpd_inverse_excess(surv, gpd_sigma, gpd_xi)
        x_mm[is_tail] = gpd_threshold + e
    # is_dry rows stay 0, matching the forward map's x<=0 -> p_dry jump
    return x_mm.astype(np.float32)
