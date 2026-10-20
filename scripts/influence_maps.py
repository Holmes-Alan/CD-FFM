"""
influence_maps.py
==================
Where does the information for one 1 km location come from, and what spatial dependence has the model learned?

  grad   Likelihood influence maps. For a target pixel s* and a day, the exact log-likelihood of the observed
         (TG, TX, TN) at s*,  log p(y(s*) | ERA5)  (flow run backwards, exact divergence, as in fast_logscore.py,
         but differentiable: create_graph=True), is differentiated w.r.t. every ERA5 input value (4 x 71 x 113).
         Also the gradient of the zero-noise best estimate of TG at s*. Averaged over days -> effective receptive
         field. CD-FFM and, for comparison, the FNO flow.
  corr   One-point correlation maps over 2019: correlation (over days) between the standardized TG anomaly at s*
         and at every 1 km pixel, for NGCD observations, one CD-FFM sample per day (flow map, 2 steps) and the
         regression network.
Output: figures/paper_data/influence_grad.npz, influence_corr.npz (plots: influence_figs.py).
"""
import argparse
import math
import os

import numpy as np
import torch

from fast_data import DATA_DIR, doy_features, mu_sigma, static_features, pick_device
from fast_eval import condition_all, load_run, regress_day, sample_day
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
OUT = f"{ROOT}/figures/paper_data"
TARGETS = {"W Norway mountains": (61.5, 7.0), "Stockholm": (59.33, 18.07),
           "Sodankyla (Lapland)": (67.37, 26.63), "Tromso (coast)": (69.65, 18.96)}
RUNS = {"CD-FFM": "w2e_static_pl24_l8", "FNO flow (124M)": "w3_fno_final"}


def nearest(st, lat, lon):
    return int(np.argmin((st["valid_lat"] - lat) ** 2 + ((st["valid_lon"] - lon) * np.cos(np.radians(lat))) ** 2))


def loglik_graph(model, x1, cond, steps):
    """Differentiable exact log p_1(x1) for a few points: x1 (n,C), cond (n,D) -> (n,). Backward RK4; the divergence
    is the trace of the C x C Jacobian with create_graph=True so that d/d(inputs) flows through it."""
    C = x1.shape[-1]

    def vdiv(x, t):
        tt = torch.full((1, x.shape[0]), t, device=x.device)
        v = model.velocity(x[None], cond[None], tt)[0]
        div = sum(torch.autograd.grad(v[:, c].sum(), x, create_graph=True)[0][:, c] for c in range(C))
        return v, div

    h = 1.0 / steps
    x = x1.clone().requires_grad_(True)
    integral = torch.zeros(x1.shape[0], device=x1.device)
    for k in range(steps):
        t = 1.0 - k * h
        k1, d1 = vdiv(x, t)
        k2, d2 = vdiv(x - 0.5 * h * k1, t - 0.5 * h)
        k3, d3 = vdiv(x - 0.5 * h * k2, t - 0.5 * h)
        k4, d4 = vdiv(x - h * k3, t - h)
        x = x - h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        integral = integral + h / 6.0 * (d1 + 2 * d2 + 2 * d3 + d4)
    return -0.5 * (x ** 2).sum(-1) - 0.5 * C * math.log(2 * math.pi) - integral


def grad_mode(a, dev):
    st = np.load(f"{DATA_DIR}/static.npz")
    lat, lon = torch.from_numpy(st["valid_lat"]).to(dev), torch.from_numpy(st["valid_lon"]).to(dev)
    cell_id = torch.from_numpy(st["era5_cell_id"]).to(dev).long()
    n_cells = len(st["era5_lat"]) * len(st["era5_lon"])
    cm, cv = torch.from_numpy(st["coeffs_mean"]).to(dev), torch.from_numpy(st["coeffs_var"]).to(dev)
    floor = float(st["sigma_floor"])
    static = static_features(cm, cv, cell_id, n_cells, floor)
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    tgt = np.load(f"{DATA_DIR}/year_2019_targets.npy", mmap_mode="r")
    days = list(range(0, len(meta["dates"]), a.day_stride))
    tidx = {k: nearest(st, *v) for k, v in TARGETS.items()}
    out = {"targets": np.array(list(TARGETS)), "target_lat": np.array([v[0] for v in TARGETS.values()]),
           "target_lon": np.array([v[1] for v in TARGETS.values()]), "era5_lat": st["era5_lat"], "era5_lon": st["era5_lon"],
           "days": np.array([str(meta["dates"][d]) for d in days]), "models": np.array(list(RUNS))}
    for mname, run in RUNS.items():
        args, model, step = load_run(run, dev, st["era5_lat"], st["era5_lon"], "best")
        for p in model.parameters():
            p.requires_grad_(False)
        G_ll = np.zeros((len(TARGETS), len(days), 4, len(st["era5_lat"]), len(st["era5_lon"])), dtype=np.float32)
        G_mu = np.zeros_like(G_ll)
        LL = np.zeros((len(TARGETS), len(days)), dtype=np.float32)
        for di, d in enumerate(days):
            doy = torch.tensor([int(meta["doy"][d])], device=dev)
            z_obs = torch.from_numpy(np.array(tgt[d])).to(dev).float()
            for ti, (tname, p) in enumerate(tidx.items()):
                if not torch.isfinite(z_obs[p]).all():
                    continue
                era5 = torch.from_numpy(meta["era5"][d]).to(dev).requires_grad_(True)
                cond = model.condition(model.encode(era5[None]), lat[None, p:p + 1], lon[None, p:p + 1], static[None, p:p + 1])
                cond = torch.cat([cond, doy_features(doy)[:, None, :]], dim=-1)[0]  # (1,D)
                _, sig = mu_sigma(cm[None, p:p + 1], cv[None, p:p + 1], doy, floor)
                ll = loglik_graph(model, z_obs[p:p + 1], cond, a.steps)[0] - torch.log(sig).sum()
                g, = torch.autograd.grad(ll, era5, retain_graph=True)
                G_ll[ti, di] = g.cpu().numpy()
                LL[ti, di] = float(ll)
                # zero-noise best estimate of TG at s* (10 Euler steps from x = 0)
                x = torch.zeros(1, 1, 3, device=dev)
                for k in range(10):
                    x = x + model.velocity(x, cond[None], torch.full((1, 1), k / 10, device=dev)) / 10
                g2, = torch.autograd.grad(x[0, 0, 0], era5)
                G_mu[ti, di] = g2.cpu().numpy()
            print(f"[grad] {mname} day {di + 1}/{len(days)}", flush=True)
        key = "flower" if "Flower" in mname else "fno"
        out[f"{key}_grad_loglik"] = G_ll
        out[f"{key}_grad_mean"] = G_mu
        out[f"{key}_loglik"] = LL
        print(f"[grad] {mname}: mean log-lik at targets {LL.mean(1)}", flush=True)
    np.savez_compressed(f"{OUT}/influence_grad.npz", **out)
    print(f"wrote {OUT}/influence_grad.npz", flush=True)


@torch.no_grad()
def corr_mode(a, dev):
    st = np.load(f"{DATA_DIR}/static.npz")
    lat, lon = torch.from_numpy(st["valid_lat"]).to(dev), torch.from_numpy(st["valid_lon"]).to(dev)
    cell_id = torch.from_numpy(st["era5_cell_id"]).to(dev).long()
    n_cells = len(st["era5_lat"]) * len(st["era5_lon"])
    cm, cv = torch.from_numpy(st["coeffs_mean"]).to(dev), torch.from_numpy(st["coeffs_var"]).to(dev)
    static = static_features(cm, cv, cell_id, n_cells, float(st["sigma_floor"]))
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    tgt = np.load(f"{DATA_DIR}/year_2019_targets.npy", mmap_mode="r")
    tidx = torch.tensor([nearest(st, *v) for v in TARGETS.values()], device=dev)
    fargs, fmodel, _ = load_run("w5_flowmap", dev, st["era5_lat"], st["era5_lon"], "best")
    rargs, rmodel, _ = load_run("w3_reg_static", dev, st["era5_lat"], st["era5_lon"], "best")
    N, T = lat.shape[0], len(tidx)
    acc = {}
    for src in ("truth", "flowerffm", "regression"):
        acc[src] = dict(n=torch.zeros(T, N, device=dev), sx=torch.zeros(T, N, device=dev, dtype=torch.float64),
                        sy=torch.zeros(T, N, device=dev, dtype=torch.float64), sxx=torch.zeros(T, N, device=dev, dtype=torch.float64),
                        syy=torch.zeros(T, N, device=dev, dtype=torch.float64), sxy=torch.zeros(T, N, device=dev, dtype=torch.float64))
    for d in range(len(meta["dates"])):
        torch.manual_seed(5000 + d)
        doy = torch.tensor([int(meta["doy"][d])], device=dev)
        era5 = torch.from_numpy(meta["era5"][d]).to(dev)
        fields = {"truth": torch.from_numpy(np.array(tgt[d, :, 0])).to(dev).float()}
        cond = condition_all(fmodel, era5, lat, lon, doy, fargs.doy_cond, a.chunk, static)
        fields["flowerffm"] = sample_day(fmodel, cond, lat, lon, fargs, 1, 2, a.chunk, None)[0, :, 0]
        condr = condition_all(rmodel, era5, lat, lon, doy, rargs.doy_cond, a.chunk, static)
        fields["regression"] = regress_day(rmodel, condr, a.chunk)[0, :, 0]
        for src, f in fields.items():
            y = f.double()
            x = y[tidx][:, None].expand(T, N)
            ok = (torch.isfinite(x) & torch.isfinite(y)[None]).float()
            x, yy = torch.nan_to_num(x) * ok, torch.nan_to_num(y)[None] * ok
            A = acc[src]
            A["n"] += ok
            A["sx"] += x
            A["sy"] += yy
            A["sxx"] += x * x
            A["syy"] += yy * yy
            A["sxy"] += x * yy
        if d % 30 == 0:
            print(f"[corr] day {d + 1}/{len(meta['dates'])}", flush=True)
    out = {"targets": np.array(list(TARGETS)), "target_idx": tidx.cpu().numpy()}
    for src, A in acc.items():
        n = A["n"].double().clamp(min=2)
        cov = A["sxy"] / n - (A["sx"] / n) * (A["sy"] / n)
        vx = A["sxx"] / n - (A["sx"] / n) ** 2
        vy = A["syy"] / n - (A["sy"] / n) ** 2
        out[f"corr_{src}"] = (cov / torch.sqrt(vx.clamp(min=1e-12) * vy.clamp(min=1e-12))).float().cpu().numpy().astype(np.float16)
    np.savez_compressed(f"{OUT}/influence_corr.npz", **out)
    print(f"wrote {OUT}/influence_corr.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grad", "corr"])
    ap.add_argument("--day-stride", type=int, default=14)
    ap.add_argument("--steps", type=int, default=20, help="RK4 steps of the backward ODE")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = pick_device(a.device)
    grad_mode(a, dev) if a.mode == "grad" else corr_mode(a, dev)


if __name__ == "__main__":
    main()
