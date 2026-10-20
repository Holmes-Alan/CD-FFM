"""
precip_tricks.py
=================
Two no-retraining fixes for the look of downscaled precipitation (model w4_rr_final, 2019, every 7th day, 16 members).

1. Inference-time noise prior. The velocity is pointwise and the noise field has unit variance at every location
   whatever its spectrum, so swapping the noise band at sampling time leaves each pixel's predictive distribution
   unchanged (CRPS / Brier identical in expectation) and changes only the spatial texture. Trained band: 8-500 km;
   tested: 25-500 and 75-500 km (fewer fragmented wet/dry edges expected).
2. Probability-matched (PM) ensemble mean (Ebert 2001): the spatial pattern of the ensemble mean with the amount
   distribution of the pooled members -- domain-wide and tile-local (256 km tiles).

Per-pixel scores: fair CRPS, Brier and frequency bias at 1 / 10 / 20 mm. Spatial scores against NGCD: fractions
skill score (FSS) at 1 and 10 mm for 1, 5, 25, 75 km neighbourhoods; wet-edge density (fraction of wet pixels with a
dry land neighbour); power spectrum in the 512 km box. Products: one sample, ensemble mean, PM means.
Output: runs/w4_rr_final/precip_tricks_2019.json, figures/paper_data/precip_tricks_<date>.npz (maps of one day).
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from fast_data import DATA_DIR, PrecipTransform, static_features, pick_device
from fast_eval import RUNS_ROOT, condition_all, load_run, psd_box, radial_psd, sample_day
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

THR = (1.0, 10.0, 20.0)
FSS_THR = (1.0, 10.0)
FSS_N = (1, 5, 25, 75)


def pm_mean(ens, mean):
    """Domain-wide probability matching. ens (M,N), mean (N) -> (N)."""
    M, N = ens.shape
    pooled = torch.sort(ens.reshape(-1), descending=True).values[M // 2::M][:N]
    out = torch.empty_like(mean)
    out[torch.argsort(mean, descending=True)] = pooled
    return out


def pm_mean_local(ens, mean, rows, cols, tile=256):
    out = torch.empty_like(mean)
    key = (rows // tile) * 1000 + cols // tile
    for k in torch.unique(key):
        sel = torch.nonzero(key == k).squeeze(1)
        out[sel] = pm_mean(ens[:, sel], mean[sel])
    return out


def to_grid(v, rows, cols, H, W):
    g = torch.full((H, W), float("nan"), device=v.device)
    g[rows, cols] = v
    return g


def box_sum(x, n):
    """Sum over n x n windows centred on each pixel (zero outside), via an integral image. x: (H,W)."""
    H, W = x.shape
    r = n // 2
    ii = F.pad(x.double().cumsum(0).cumsum(1), (1, 0, 1, 0))
    i0 = (torch.arange(H, device=x.device) - r).clamp(0, H)
    i1 = (torch.arange(H, device=x.device) + r + 1).clamp(0, H)
    j0 = (torch.arange(W, device=x.device) - r).clamp(0, W)
    j1 = (torch.arange(W, device=x.device) + r + 1).clamp(0, W)
    return (ii[i1][:, j1] - ii[i0][:, j1] - ii[i1][:, j0] + ii[i0][:, j0]).float()


def fss(fc, ob, land, n):
    """Fractions skill score. fc, ob: (H,W) binary float; land: (H,W) bool; fractions over n x n windows of land."""
    lm = land.float()
    cnt = box_sum(lm, n).clamp(min=1)
    pf = (box_sum(fc * lm, n) / cnt)[land]
    po = (box_sum(ob * lm, n) / cnt)[land]
    den = (pf ** 2).mean() + (po ** 2).mean()
    return float(1 - ((pf - po) ** 2).mean() / den) if den > 0 else float("nan")


def edge_density(wet, land):
    """Fraction of wet land pixels with at least one dry land 4-neighbour."""
    dry = (~wet) & land
    nb = torch.zeros_like(wet)
    nb[1:] |= dry[:-1]
    nb[:-1] |= dry[1:]
    nb[:, 1:] |= dry[:, :-1]
    nb[:, :-1] |= dry[:, 1:]
    w = wet & land
    return float((w & nb).sum()) / max(float(w.sum()), 1.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="w4_rr_final")
    ap.add_argument("--day-stride", type=int, default=7)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--bands", default="8,25,75", help="lower wavelength limits (km) of the sampling noise")
    ap.add_argument("--map-date", default="2019-01-01")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    a = ap.parse_args()
    dev = pick_device(a.device)
    st = np.load(f"{DATA_DIR}/static.npz")
    lat, lon = torch.from_numpy(st["valid_lat"]).to(dev), torch.from_numpy(st["valid_lon"]).to(dev)
    rows = torch.from_numpy(st["valid_rows"].astype(np.int64)).to(dev)
    cols = torch.from_numpy(st["valid_cols"].astype(np.int64)).to(dev)
    H, W = [int(v) for v in st["grid_shape"]]
    land = torch.zeros(H, W, dtype=torch.bool, device=dev)
    land[rows, cols] = True
    cm, cv = torch.from_numpy(st["coeffs_mean"]).to(dev), torch.from_numpy(st["coeffs_var"]).to(dev)
    cell_id = torch.from_numpy(st["era5_cell_id"]).to(dev).long()
    n_cells = len(st["era5_lat"]) * len(st["era5_lon"])
    rr = PrecipTransform(DATA_DIR, dev)
    static = static_features(cm, cv, cell_id, n_cells, float(st["sigma_floor"]), p_dry=rr.p_dry)
    args, model, step = load_run(a.run, dev, st["era5_lat"], st["era5_lon"], "best")
    meta = dict(np.load(f"{DATA_DIR}/year_2019_meta.npz"))
    tgt = np.load(f"{DATA_DIR}/year_2019_rr.npy", mmap_mode="r")
    dates = [str(x) for x in meta["dates"]]
    days = list(range(0, len(dates), a.day_stride))
    allpix = torch.arange(lat.shape[0], device=dev)
    valid_np = np.zeros((H, W), dtype=bool)
    valid_np[st["valid_rows"], st["valid_cols"]] = True
    (br, bc), _ = psd_box(valid_np)
    L, NB = 512, 128
    bmask = land[br:br + L, bc:bc + L]
    bands = [float(b) for b in a.bands.split(",")]
    products = ["sample", "ensmean", "pm_global", "pm_local"]
    S = {}  # accumulators keyed by (band, product, metric)

    def add(key, v):
        if v == v:  # skip NaN
            s, n = S.get(key, (0.0, 0))
            S[key] = (s + v, n + 1)

    psd = {}
    maps = {}
    for it, d in enumerate(days):
        ds = dates[d]
        doy = torch.tensor([int(meta["doy"][d])], device=dev)
        era5 = torch.from_numpy(meta["era5"][d]).to(dev)
        truth = rr.to_mm(torch.from_numpy(np.array(tgt[d, :, 0])).to(dev).float(), allpix)
        ok = torch.isfinite(truth)
        truth = torch.nan_to_num(truth)
        cond = condition_all(model, era5, lat, lon, doy, args.doy_cond, a.chunk, static)
        tg = to_grid(truth, rows, cols, H, W)
        tg0 = torch.nan_to_num(tg)
        psd.setdefault("truth", 0)
        psd["truth"] = psd["truth"] + radial_psd(tg0[br:br + L, bc:bc + L], bmask, NB)
        for b in bands:
            bargs = copy.copy(args)
            bargs.noise_lmin_km = b
            torch.manual_seed(1000 + d)
            ens = torch.nan_to_num(rr.to_mm(sample_day(model, cond, lat, lon, bargs, a.members, 10, a.chunk, None)[..., 0], allpix))
            mean = ens.mean(0)
            M = ens.shape[0]
            # per-pixel probabilistic scores
            crps = (ens - truth[None]).abs().mean(0)
            pair = torch.zeros_like(truth)
            for i in range(M):
                pair += (ens[i][None] - ens).abs().sum(0)
            crps = crps - 0.5 * pair / (M * (M - 1))
            add((b, "ens", "crps"), float(crps[ok].mean()))
            for t in THR:
                p = (ens > t).float().mean(0)
                o = (truth > t).float()
                add((b, "ens", f"brier_{t:g}"), float(((p - o) ** 2)[ok].mean()))
            prods = {"sample": ens[0], "ensmean": mean, "pm_global": pm_mean(ens, mean),
                     "pm_local": pm_mean_local(ens, mean, rows, cols)}
            for pname, f in prods.items():
                add((b, pname, "mae"), float((f - truth).abs()[ok].mean()))
                add((b, pname, "rmse"), float(((f - truth) ** 2)[ok].mean().sqrt()))
                fg = torch.nan_to_num(to_grid(f, rows, cols, H, W))
                for t in THR:  # pooled over days below (a mean of daily ratios explodes on near-dry days)
                    add((b, pname, f"nfc_{t:g}"), float((f > t)[ok].float().sum()))
                    add((b, pname, f"nob_{t:g}"), float((truth > t)[ok].float().sum()))
                for t in FSS_THR:
                    for n in FSS_N:
                        add((b, pname, f"fss_{t:g}mm_{n}km"), fss((fg > t).float(), (tg0 > t).float(), land, n))
                add((b, pname, "edge_1mm"), edge_density(fg > 1.0, land))
                k = f"{b:g}_{pname}"
                psd[k] = psd.get(k, 0) + radial_psd(fg[br:br + L, bc:bc + L], bmask, NB)
                if ds == a.map_date:
                    maps[f"{pname}_{b:g}"] = f.cpu().numpy().astype(np.float16)
            if ds == a.map_date:
                maps["truth"] = truth.cpu().numpy().astype(np.float16)
        add(("truth", "truth", "edge_1mm"), edge_density(tg0 > 1.0, land))
        print(f"[precip_tricks] {it + 1}/{len(days)} {ds}", flush=True)
    for (b, pname, k) in list(S):
        if k.startswith("nfc_"):
            t = k[4:]
            S[(b, pname, f"freqbias_{t}")] = (S[(b, pname, k)][0] / max(S[(b, pname, f"nob_{t}")][0], 1.0), 1)
    res = {"run": a.run, "days": len(days), "members": a.members, "bands_lmin_km": bands,
           "metrics": {f"{k[0]}|{k[1]}|{k[2]}": s / n for k, (s, n) in S.items()},
           "psd": {k: (v / len(days)).cpu().numpy().tolist() for k, v in psd.items()}}
    path = f"{RUNS_ROOT}/{a.run}/precip_tricks_2019.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    if maps:
        np.savez_compressed(f"{CDFFM_ROOT}/figures/paper_data/precip_tricks_{a.map_date}.npz", **maps)
    # summary
    m = res["metrics"]
    print("\nper-pixel (should be ~unchanged across bands): band | CRPS | Brier 1/10/20 mm")
    for b in bands:
        print(f"  {b:5g} km | {m[f'{b}|ens|crps']:.3f} | " + " / ".join(f"{m[f'{b}|ens|brier_{t:g}']:.4f}" for t in THR))
    print(f"\nspatial: product | FSS 1 mm @ 5/25/75 km | FSS 10 mm @ 5/25/75 km | wet-edge density (NGCD {m['truth|truth|edge_1mm']:.3f}) | MAE | freq bias 1/10 mm")
    for b in bands:
        for p in products:
            fs1 = " ".join(f"{m[f'{b}|{p}|fss_1mm_{n}km']:.3f}" for n in (5, 25, 75))
            fs10 = " ".join(f"{m[f'{b}|{p}|fss_10mm_{n}km']:.3f}" for n in (5, 25, 75))
            print(f"  {b:4g} km {p:10s} | {fs1} | {fs10} | {m[f'{b}|{p}|edge_1mm']:.3f} | {m[f'{b}|{p}|mae']:.3f} | "
                  f"{m[f'{b}|{p}|freqbias_1']:.2f} / {m[f'{b}|{p}|freqbias_10']:.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
