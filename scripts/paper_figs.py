"""
paper_figs.py
==============
Teaser figure and Figure 1 of the paper from figures/paper_data/*.npz (paper_figdata.py) and the
evaluation metrics. CPU only (numpy + matplotlib).

  fig0_teaser.png             downscaling quality and uncertainty for temperature over the Nordic domain
                              + CRPS skill of all models (precipitation is an additional experiment: Extended Data)
  fig1_resolution_free.png    one sample of the functional model queried at 8 km ... 0.25 km with the same noise
"""
import argparse
import json
import os

import numpy as np
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
DATA = f"{ROOT}/figures/paper_data"
OUT = f"{ROOT}/figures/paper"
RUNS = f"{ROOT}/runs"


def grid(v, rows, cols, H, W):
    g = np.full((H, W), np.nan, dtype=np.float32)
    g[rows, cols] = v
    return g


def crop(g):
    """Crop to the land bounding box."""
    r = np.where(np.isfinite(g).any(1))[0]
    c = np.where(np.isfinite(g).any(0))[0]
    return g[r[0]:r[-1] + 1, c[0]:c[-1] + 1], (r[0], c[0])


def crps_of(run, tag, var):
    p = f"{RUNS}/{run}/eval_2019_{tag}/metrics.json"
    if not os.path.exists(p):
        return None
    y = json.load(open(p))["groups"]["all"]["year"]
    return y[var].get("crps_fair", y[var]["crps"]) if var in y else None


def teaser(t_date, plt):
    st = np.load(f"{ROOT}/fast_data/static.npz")
    rows, cols, cell = st["valid_rows"], st["valid_cols"], st["era5_cell_id"]
    H, W = [int(v) for v in st["grid_shape"]]
    T = np.load(f"{DATA}/teaser_{t_date}.npz")
    fig = plt.figure(figsize=(17, 7.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.9], hspace=0.25, wspace=0.28)

    def mapax(i, j, g, title, **kw):
        ax = fig.add_subplot(gs[i, j])
        gc, _ = crop(g)
        im = ax.imshow(gc, interpolation="nearest", **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        return ax

    # --- row 1: temperature (TG), exceedance for TX
    tg_true = T["truth_t"][:, 0].astype(np.float32)
    ens = T["ens_t"].astype(np.float32)
    lo, hi = np.nanpercentile(tg_true, [1, 99])
    tkw = dict(cmap="RdYlBu_r", vmin=lo, vmax=hi)
    mapax(0, 0, grid(T["era5_t"][0].reshape(-1)[cell], rows, cols, H, W), f"ERA5 input, 25 km\nTG {t_date} (°C)", **tkw)
    mapax(0, 1, grid(tg_true, rows, cols, H, W), "NGCD observations, 1 km", **tkw)
    mapax(0, 2, grid(ens[:, :, 0].mean(0), rows, cols, H, W), "CD-FFM ensemble mean", **tkw)
    mapax(0, 3, grid(ens[:, :, 0].std(0), rows, cols, H, W), "Spread, 16 members (°C)", cmap="viridis")
    thr = 25.0 if int(t_date[5:7]) in (5, 6, 7, 8, 9) else -20.0
    if thr > 0:
        p = (ens[:, :, 1] > thr).mean(0)
        ttl = f"P(TX > {thr:g} °C)"
    else:
        p = (ens[:, :, 2] < thr).mean(0)
        ttl = f"P(TN < {thr:g} °C)"
    ax = mapax(0, 4, grid(p, rows, cols, H, W), ttl, cmap="magma", vmin=0, vmax=1)
    obs = (T["truth_t"][:, 1] > thr) if thr > 0 else (T["truth_t"][:, 2] < thr)
    go, _ = crop(grid(obs.astype(np.float32), rows, cols, H, W))
    ax.contour(np.nan_to_num(go), levels=[0.5], colors="cyan", linewidths=0.5)
    mapax(0, 5, grid(T["loglik_t"].astype(np.float32), rows, cols, H, W),
          "Exact log-likelihood of the\nobservation (nats, TG/TX/TN)", cmap="cividis",
          vmin=float(np.nanpercentile(T["loglik_t"].astype(np.float32), 2)),
          vmax=float(np.nanpercentile(T["loglik_t"].astype(np.float32), 98)))

    # --- row 2: skill summary (CRPS relative to the statistical baseline) + zooms
    ax = fig.add_subplot(gs[1, :3])
    models = [("CD-FFM (ours)", "w5_flowmap", "final_map2"),
              ("CorrDiff-style (every 7th day)", "w6_corrdiff", "s7_patch"),
              ("FNO flow, 124M", "w3_fno_final", "final"),
              ("Regression", "w3_reg_static", "final")]
    vars_ = ["TG", "TX", "TN"]
    base = {v: crps_of("baseline_bcsd", "best", v) for v in vars_}
    x = np.arange(len(vars_))
    w = 0.8 / len(models)
    for k, (lab, rt, tt) in enumerate(models):
        vals = []
        for v in vars_:
            c = crps_of(rt, tt, v)
            vals.append(np.nan if (c is None or base[v] is None) else 100 * (1 - c / base[v]))
        ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(vars_)
    ax.set_ylabel("CRPS skill vs BCSD-style\nbaseline (%), 2019")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.set_title("Probabilistic skill (higher is better)", fontsize=9)

    # zooms: 128 x 128 km in western Norway, truth vs sample, TG
    i = int(np.argmin((st["valid_lat"] - 61.5) ** 2 + ((st["valid_lon"] - 7.0) * np.cos(np.radians(61.5))) ** 2))
    r0, c0 = int(rows[i]) - 64, int(cols[i]) - 64
    for j, (ttl, v) in enumerate([("NGCD, 128 km zoom", tg_true), ("CD-FFM sample", ens[0, :, 0]),
                                  ("CD-FFM sample 2", ens[1, :, 0])]):
        ax = fig.add_subplot(gs[1, 3 + j])
        g = grid(v, rows, cols, H, W)[r0:r0 + 128, c0:c0 + 128]
        zl, zh = np.nanpercentile(grid(tg_true, rows, cols, H, W)[r0:r0 + 128, c0:c0 + 128], [1, 99])
        ax.imshow(g, cmap="RdYlBu_r", vmin=zl, vmax=zh, interpolation="nearest")
        ax.set_title(ttl + " (TG)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Downscaling ERA5 to 1 km daily temperature over Fennoscandia: samples, uncertainty and skill", fontsize=12)
    fig.savefig(f"{OUT}/fig0_teaser.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig0_teaser.png", flush=True)


def resolution_free(date, plt):
    M = np.load(f"{DATA}/multires_{date}.npz")
    keys = ["8km", "4km", "2km", "1km", "0p5km", "0p25km"]
    labels = ["8 km", "4 km", "2 km", "1 km", "0.5 km", "0.25 km"]
    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(2, 7, hspace=0.32, wspace=0.08)
    tg = [M[f"sample_{k}"][..., 0] for k in keys]
    lo, hi = np.nanpercentile(np.concatenate([t.ravel() for t in tg]), [1, 99])
    ext = [0, 64, 64, 0]
    for j, (k, lab) in enumerate(zip(keys, labels)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(M[f"noise_{k}"], cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=ext, interpolation="nearest")
        ax.set_title(f"noise z(s) at {lab}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(tg[j], cmap="RdYlBu_r", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_title(f"sample x(s) at {lab}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 6])
    ax.imshow(M["truth_1km"][..., 0], cmap="RdYlBu_r", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
    ax.set_title("NGCD observations, 1 km", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, label="TG (°C)")
    ax = fig.add_subplot(gs[0, 6])
    ax.axis("off")
    cons = max(float(M[f"consistency_max_abs_{k}"]) for k in keys[:-1])
    ax.text(0.0, 0.5, "One random function,\nqueried on nested grids:\nencode ERA5 once, evaluate\nthe Gaussian field z and the\npointwise flow at any points.\n\nShared points agree to\n{:.1e} °C.".format(cons),
            fontsize=9, va="center")
    fig.suptitle(f"Resolution-free downscaling: one CD-FFM sample, 64 × 64 km in western Norway, TG {date}", fontsize=12)
    fig.savefig(f"{OUT}/fig1_resolution_free.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_resolution_free.png", flush=True)


def ensemble(date, var, plt, nsamp=6):
    """One input -> many plausible 1 km fields: ERA5, NGCD, nsamp samples, ensemble mean and spread; bottom row:
    128 km zoom (western Norway for temperature, the wettest box for precipitation) of truth and the samples."""
    from matplotlib.colors import LogNorm
    st = np.load(f"{ROOT}/fast_data/static.npz")
    rows, cols, cell = st["valid_rows"], st["valid_cols"], st["era5_cell_id"]
    H, W = [int(v) for v in st["grid_shape"]]
    T = np.load(f"{DATA}/teaser_{date}.npz")
    if var == "TG":
        truth = T["truth_t"][:, 0].astype(np.float32)
        ens = T["ens_t"][:, :, 0].astype(np.float32)
        e5 = T["era5_t"][0].reshape(-1)[cell]
        lo, hi = np.nanpercentile(truth, [1, 99])
        kw = dict(cmap="RdYlBu_r", vmin=lo, vmax=hi)
        unit, name, tr = "°C", "daily mean temperature", (lambda v: v)
        i = int(np.argmin((st["valid_lat"] - 61.5) ** 2 + ((st["valid_lon"] - 7.0) * np.cos(np.radians(61.5))) ** 2))
        r0, c0 = int(rows[i]) - 64, int(cols[i]) - 64
    else:
        truth = T["truth_r"].astype(np.float32)
        ens = T["ens_r"].astype(np.float32)
        e5 = T["era5_tp"].reshape(-1)[cell]
        vmax = max(float(np.nanpercentile(truth, 99.5)), 5.0)
        kw = dict(cmap="YlGnBu", norm=LogNorm(vmin=0.3, vmax=vmax))
        unit, name, tr = "mm", "daily precipitation", (lambda v: np.where(v > 0.3, v, np.nan))
        gt = grid(np.nan_to_num(truth), rows, cols, H, W)  # wettest 128 km box
        cs = np.pad(np.nan_to_num(gt).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        best, r0, c0 = -1, 0, 0
        for r in range(0, H - 128, 32):
            for c in range(0, W - 128, 32):
                v = cs[r + 128, c + 128] - cs[r, c + 128] - cs[r + 128, c] + cs[r, c]
                if v > best:
                    best, r0, c0 = v, r, c
    fig = plt.figure(figsize=(2.3 * (nsamp + 4), 6.6))
    gs = fig.add_gridspec(2, nsamp + 4, height_ratios=[1.25, 1], hspace=0.18, wspace=0.06)
    panels = [("ERA5 input, 25 km", e5), ("NGCD observations", truth)] + \
             [(f"sample {m + 1}", ens[m]) for m in range(nsamp)] + [("ensemble mean (16)", ens.mean(0))]
    for j, (ttl, v) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        g, _ = crop(grid(tr(v), rows, cols, H, W))
        ax.imshow(g, interpolation="nearest", **kw)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if j > 0:
            zax = fig.add_subplot(gs[1, j])
            zg = grid(tr(v), rows, cols, H, W)[r0:r0 + 128, c0:c0 + 128]
            zk = dict(kw)
            if var == "TG":
                zk["vmin"], zk["vmax"] = np.nanpercentile(grid(truth, rows, cols, H, W)[r0:r0 + 128, c0:c0 + 128], [1, 99])
            zax.imshow(zg, interpolation="nearest", **zk)
            zax.set_xticks([])
            zax.set_yticks([])
            if j == 1:
                zax.set_ylabel("128 km zoom", fontsize=9)
    ax = fig.add_subplot(gs[0, nsamp + 3])
    g, _ = crop(grid(ens.std(0), rows, cols, H, W))
    im = ax.imshow(g, cmap="viridis", interpolation="nearest", vmax=float(np.nanpercentile(ens.std(0), 99)))
    ax.set_title(f"ensemble spread ({unit})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"One input, many plausible 1 km fields: CD-FFM {name}, {date}", fontsize=12)
    path = f"{OUT}/fig_ensemble_{var}_{date}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def multires_full(date, plt):
    """One input, one sample, many resolutions: the whole domain at 16-1 km (queried separately) and a 64 km zoom
    continuing to 0.5 and 0.25 km (from multires_<date>.npz)."""
    F = np.load(f"{DATA}/multires_full_{date}.npz")
    Z = np.load(f"{DATA}/multires_{date}.npz") if os.path.exists(f"{DATA}/multires_{date}.npz") else None
    T = np.load(f"{DATA}/teaser_{date}.npz") if os.path.exists(f"{DATA}/teaser_{date}.npz") else None
    st = np.load(f"{ROOT}/fast_data/static.npz")
    H, W = [int(v) for v in F["grid_shape"]]
    lo, hi = np.nanpercentile(F["val_1km"][:, 0], [1, 99])
    kw = dict(cmap="RdYlBu_r", vmin=lo, vmax=hi, interpolation="nearest")
    res = [16, 8, 4, 2, 1]
    fig = plt.figure(figsize=(19, 8.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.2, wspace=0.06)
    if T is not None:
        ax = fig.add_subplot(gs[0, 0])
        g, _ = crop(grid(T["era5_t"][0].reshape(-1)[st["era5_cell_id"]], st["valid_rows"], st["valid_cols"], H, W))
        ax.imshow(g, **kw)
        ax.set_title("ERA5 input, 25 km", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j, r in enumerate(res):
        k = f"{r}km"
        rr, cc = F[f"rows_{k}"] // r, F[f"cols_{k}"] // r
        g = np.full((H // r + 1, W // r + 1), np.nan, dtype=np.float32)
        g[rr, cc] = F[f"val_{k}"][:, 0]
        g, _ = crop(g)
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(g, **kw)
        ax.set_title(f"{r} km: {len(rr):,} points, {float(F[f'seconds_{k}']):.2f} s", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if Z is not None:
        zk = ["4km", "2km", "1km", "0p5km", "0p25km"]
        zl = ["4 km", "2 km", "1 km", "0.5 km", "0.25 km"]
        zz = [Z[f"sample_{k}"][..., 0] for k in zk]
        zlo, zhi = np.nanpercentile(np.concatenate([v.ravel() for v in zz]), [1, 99])
        for j, (v, l) in enumerate(zip(zz, zl)):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(v, cmap="RdYlBu_r", vmin=zlo, vmax=zhi, interpolation="nearest")
            ax.set_title(f"64 km zoom, {l}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 5])
        im = ax.imshow(Z["truth_1km"][..., 0], cmap="RdYlBu_r", vmin=zlo, vmax=zhi, interpolation="nearest")
        ax.set_title("NGCD observations, 1 km", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, label="TG (°C)")
    fig.suptitle(f"One input, one sample, any resolution: CD-FFM daily mean temperature, {date} "
                 "(same noise realisation; shared points identical)", fontsize=12)
    path = f"{OUT}/fig_multires_full_{date}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-date", default="2019-07-28")
    ap.add_argument("--m-date", default="2019-07-28")
    ap.add_argument("--only", choices=["all", "extra"], default="all",
                    help="extra: only the ensemble and full-domain multi-resolution figures")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    if a.only == "extra":
        ensemble(a.t_date, "TG", plt)
        if os.path.exists(f"{DATA}/teaser_2019-01-01.npz"):
            ensemble("2019-01-01", "RR", plt)
        if os.path.exists(f"{DATA}/multires_full_{a.m_date}.npz"):
            multires_full(a.m_date, plt)
        return
    teaser(a.t_date, plt)
    if os.path.exists(f"{DATA}/multires_{a.m_date}.npz"):
        resolution_free(a.m_date, plt)


if __name__ == "__main__":
    main()
