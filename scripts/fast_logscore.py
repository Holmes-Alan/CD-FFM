"""
fast_logscore.py
=================
Exact per-pixel logarithmic score (negative log predictive density at the observation) for the
flow-matching models -- a second proper scoring rule next to CRPS, no sampling needed.

Why it is exact and cheap here: the velocity net is pointwise (MLP on [x_t, cond(s), t] at each
location), and at every pixel the noise field has a standard-normal marginal in each of the C output
channels (independent channels). So the predictive density of (TG,TX,TN) at a pixel is the push-forward
of N(0, I_C) through a C-dimensional ODE, and by the instantaneous change of variables

    log p_1(x_1) = log N(x_0; 0, I) - int_0^1 div_x v(x_t, t) dt,

with x_t obtained by integrating the ODE BACKWARD from the observation x_1 (z-space). div_x v is the
trace of a C x C Jacobian: C exact backward passes (no Hutchinson estimator). Physical units:
log p(x_phys) = log p(z) - sum_c log sigma_c(pixel, doy).

This scores the continuous-time model (RK4, --steps), not the 10-step Euler sampler used for CRPS.

Deterministic runs (regression, --objective regression) get the standard comparison: a Gaussian with
the residual covariance (3x3, z-space) fitted on the evaluated days themselves -- an *oracle* spread,
i.e. the most favourable homoscedastic Gaussian for the baseline.

  python3 fast_logscore.py --run w2e_static_pl24_l8 --ckpt best --day-stride 7
Output: runs/<run>/logscore_<year>_<tag>.json (nats per pixel, joint TG/TX/TN, degC units; by group and
season), and a printed summary.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from fast_data import DATA_DIR, HOLDOUT_REGIONS, TARGET_VARS, mu_sigma, static_features, pick_device
from fast_eval import RUNS_ROOT, SEASON_OF_MONTH, SEASONS, condition_all, load_run, regress_day


def velocity_and_div(model, x, cond, t):
    """x: (n,C) z-space state, cond: (n,D), t: float -> v (n,C), exact div_x v (n,)."""
    C = x.shape[-1]
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        tt = torch.full((1, x.shape[0]), t, device=x.device)
        v = model.velocity(x[None], cond[None], tt)[0]
        div = torch.zeros(x.shape[0], device=x.device)
        for c in range(C):  # pointwise net: d(sum_i v_ic)/dx_jc = dv_jc/dx_jc
            g = torch.autograd.grad(v[:, c].sum(), x, retain_graph=c < C - 1)[0]
            div = div + g[:, c]
    return v.detach(), div.detach()


def log_density_z(model, x1, cond, steps):
    """Backward RK4 from t=1 (observation) to t=0; returns log p_1(x1) in z-space, (n,)."""
    h = 1.0 / steps
    x, integral = x1.clone(), torch.zeros(x1.shape[0], device=x1.device)
    for k in range(steps):
        t = 1.0 - k * h
        k1, d1 = velocity_and_div(model, x, cond, t)
        k2, d2 = velocity_and_div(model, x - 0.5 * h * k1, cond, t - 0.5 * h)
        k3, d3 = velocity_and_div(model, x - 0.5 * h * k2, cond, t - 0.5 * h)
        k4, d4 = velocity_and_div(model, x - h * k3, cond, t - h)
        x = x - h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        integral = integral + h / 6.0 * (d1 + 2 * d2 + 2 * d3 + d4)
    C = x.shape[-1]
    log_p0 = -0.5 * (x ** 2).sum(-1) - 0.5 * C * math.log(2 * math.pi)
    return log_p0 - integral


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--day-stride", type=int, default=7)
    ap.add_argument("--steps", type=int, default=20, help="RK4 steps of the backward ODE")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--region", choices=sorted(HOLDOUT_REGIONS), default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    a = ap.parse_args()

    dev = pick_device(a.device)
    t0 = time.time()
    st = np.load(f"{DATA_DIR}/static.npz")
    lat = torch.from_numpy(st["valid_lat"]).to(dev)
    lon = torch.from_numpy(st["valid_lon"]).to(dev)
    cell_id = torch.from_numpy(st["era5_cell_id"]).to(dev).long()
    n_cells = len(st["era5_lat"]) * len(st["era5_lon"])
    coeffs_mean = torch.from_numpy(st["coeffs_mean"]).to(dev)[None]
    coeffs_var = torch.from_numpy(st["coeffs_var"]).to(dev)[None]
    floor = float(st["sigma_floor"])
    N = lat.shape[0]

    args, model, step = load_run(a.run, dev, st["era5_lat"], st["era5_lon"], a.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(args, "target", "temp") != "temp":
        raise SystemExit("log score: temperature runs only (precipitation has a dry-day point mass)")
    C = len(TARGET_VARS["temp"])
    deterministic = args.objective == "regression"
    static = None
    if getattr(args, "static", "none") != "none":
        static = static_features(coeffs_mean[0], coeffs_var[0], cell_id, n_cells, floor)
    tag = a.tag or f"s{a.day_stride}_rk{a.steps}"

    held = torch.zeros(N, dtype=torch.bool, device=dev)
    region = getattr(args, "holdout", None) or a.region
    if region:
        r = HOLDOUT_REGIONS[region]
        held = (lat >= r["lat_min"]) & (lon >= r["lon_min"])
    groups = {"all": torch.ones(N, dtype=torch.bool, device=dev)}
    if held.any():
        groups["holdout"], groups["inside"] = held, ~held
    G, S = len(groups), len(SEASONS)

    meta = dict(np.load(f"{DATA_DIR}/year_{a.year}_meta.npz"))
    tgt = np.load(f"{DATA_DIR}/year_{a.year}_targets.npy", mmap_mode="r")
    dates = [str(d) for d in meta["dates"]]
    days = list(range(0, len(dates), a.day_stride))
    print(f"[logscore] run={a.run} step={step} objective={args.objective} days={len(days)} "
          f"rk4_steps={a.steps}", flush=True)

    acc = {k: torch.zeros(G, S, dtype=torch.float64, device=dev) for k in ("n", "nll_z", "logsig")}
    rr = torch.zeros(G, S, C, C, dtype=torch.float64, device=dev)  # deterministic: residual 2nd moments
    for it, d in enumerate(days):
        ds = dates[d]
        s_idx = SEASONS.index(SEASON_OF_MONTH[int(ds[5:7])])
        doy = torch.tensor([int(meta["doy"][d])], device=dev)
        era5 = torch.from_numpy(meta["era5"][d]).to(dev)
        _, sig = mu_sigma(coeffs_mean, coeffs_var, doy, floor)
        logsig = torch.log(sig[0]).sum(-1).double()  # (N,)
        z1 = torch.from_numpy(np.array(tgt[d])).to(dev).float()
        ok = torch.isfinite(z1).all(-1)
        z1 = torch.nan_to_num(z1)
        with torch.no_grad():
            cond = condition_all(model, era5, lat, lon, doy, args.doy_cond, a.chunk, static)[0]  # (N,D)
        if deterministic:
            with torch.no_grad():
                res = (z1 - regress_day(model, cond[None], a.chunk)[0]).double()  # (N,C)
            outer = res[:, :, None] * res[:, None, :]
        else:
            lp = torch.empty(N, dtype=torch.float64, device=dev)
            for s0 in range(0, N, a.chunk):
                s1 = min(N, s0 + a.chunk)
                lp[s0:s1] = log_density_z(model, z1[s0:s1], cond[s0:s1], a.steps).double()
        for gi, gm in enumerate(groups.values()):
            m = ok & gm
            for si in (0, s_idx):
                acc["n"][gi, si] += m.sum()
                acc["logsig"][gi, si] += logsig[m].sum()
                if deterministic:
                    rr[gi, si] += outer[m].sum(0)
                else:
                    acc["nll_z"][gi, si] += (-lp[m]).sum()
        if it % 5 == 0:
            msg = "" if deterministic else f" mean NLL_z(day)={float(-lp[ok].mean()):.3f}"
            print(f"  [{it + 1}/{len(days)}] {ds}{msg} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    out = {"run": a.run, "step": step, "year": a.year, "n_days": len(days), "rk4_steps": a.steps,
           "objective": args.objective, "units": "nats per pixel, joint TG/TX/TN, degC",
           "baseline_note": ("Gaussian with oracle residual covariance fitted on the evaluated days"
                             if deterministic else "exact flow density (continuous-time ODE)"), "groups": {}}
    for gi, g in enumerate(groups):
        out["groups"][g] = {}
        for si, sname in enumerate(SEASONS):
            n = float(acc["n"][gi, si].clamp(min=1))
            if deterministic:
                Sig = rr[gi, si] / n
                nll_z = 0.5 * float(torch.logdet(2 * math.pi * Sig)) + 0.5 * C
            else:
                nll_z = float(acc["nll_z"][gi, si]) / n
            out["groups"][g][sname] = {"nll_z": nll_z, "nll_degC": nll_z + float(acc["logsig"][gi, si]) / n, "n": n}
    path = f"{RUNS_ROOT}/{a.run}/logscore_{a.year}_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n=== {a.run} | log score {a.year} | {len(days)} days | {tag} ===")
    for g in groups:
        print(f"[{g}] " + "  ".join(f"{s}: {out['groups'][g][s]['nll_degC']:.3f}" for s in SEASONS)
              + "   (nats/pixel, joint TG/TX/TN, lower is better)", flush=True)
    print(f"[done] {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
