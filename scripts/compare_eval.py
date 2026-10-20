"""
compare_eval.py
================
Cross-model comparison figures from fast_eval.py outputs (CPU only, numpy +
matplotlib; needs a few GB of RAM: each map panel is a 2020x1550 grid).

  1. maps_<date>_<VAR>.png  one row for the full domain, one for a 128x128 km zoom:
     ERA5 input (nearest cell, i.e. what the model sees) | NGCD truth | one sample
     per run (| ensemble std of the first generative run). Needs fields_<date>.npz,
     written by fast_eval.py for its --map-dates (e.g. a --tag maps run with
     --day-stride 1000: only the map dates, ~2 min per run).
  2. psd_compare.png        radially averaged power spectrum: truth vs one sample per run.
  3. metrics_compare.png    CRPS / ensemble-mean RMSE / spread-skill / hot CSI per run and variable.
     2 and 3 read runs/<run>/eval_<year>_<metrics-tag>/{psd.npz,metrics.json} (full-year eval).

  python3 compare_eval.py --runs w1_flower_reg,w1_flower_fm,w1_fno_fm --out ../figures/wave1
  python3 compare_eval.py --runs w1_flower_reg,w2_flower_y15@maps_m32 --field ensmean --out ...
     (run@tag: map fields from that eval tag; --field ensmean: ensemble mean instead of one sample)
"""
import argparse
import json
import os

import numpy as np
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
RUNS_ROOT = f"{ROOT}/runs"
DATA_DIR = f"{ROOT}/fast_data"
VARS = ["TG", "TX", "TN"]
ERA5_TEMP = ["t2m_mean_degC", "t2m_max_degC", "t2m_min_degC"]  # ERA5 window channels 0-2 <-> TG/TX/TN
ZOOM_KM = 128
NO_SPREAD = False


def is_deterministic(run):
    return json.load(open(f"{RUNS_ROOT}/{run}/args.json")).get("objective") in ("regression", "baseline")


PAPER = False  # --paper-labels: readable model names instead of run names
PAPER_LABELS = {"baseline_bcsd": "BCSD-style", "w3_reg_static": "Regression", "w6_unet_reg": "CorrDiff-style\nUNet regression",
                "w3_fno_final": "FNO flow (124M)", "w3_iid_static": "CD-FFM, i.i.d. noise", "w2d_static_rbf": "CD-FFM, RBF noise",
                "w2e_static_pl24_l8": "CD-FFM", "w5_flowmap": "CD-FFM, flow map"}


def label_of(run, tag=None):
    if PAPER:
        lab = PAPER_LABELS.get(run, run)
        return lab + "\nzero-noise ODE" if tag and tag.endswith("det0") else lab
    a = json.load(open(f"{RUNS_ROOT}/{run}/args.json"))
    if a.get("objective") in ("regression", "baseline"):
        return f"{run}\n({a['objective']})"
    return run


def era5_phys(meta, d, doy):
    """ERA5 TG/TX/TN in degC on the 71x113 grid for day index d (undo preprocess_fast's z-score)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import climstats
    st = np.load(f"{ROOT}/norm_stats.npz")
    out = []
    for c, ch in enumerate(ERA5_TEMP):
        mu, sigma = climstats.eval_harmonic(st[f"era5_{ch}_coeffs_mean"], st[f"era5_{ch}_coeffs_var"], doy)
        out.append(meta["era5"][d, c] * sigma + mu)
    return np.stack(out)  # (3,71,113)


def to_grid(v, rows, cols, H, W):
    g = np.full((H, W), np.nan, dtype=np.float32)
    g[rows, cols] = v
    return g


def zoom_box(truth_grid, size=ZOOM_KM, stride=32):
    """Top-left of the fully valid size x size box with the largest spatial std of the truth
    (= the most texture: coast / mountains) -- where downscaling quality is visible."""
    H, W = truth_grid.shape
    best, rc = -1.0, (0, 0)
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            b = truth_grid[r:r + size, c:c + size]
            if np.isfinite(b).all():
                s = float(b.std())
                if s > best:
                    best, rc = s, (r, c)
    return rc


def _spec(r, tag):
    """'run' or 'run@tag' -> (run, eval tag for its map fields)."""
    return (r.split("@")[0], r.split("@")[1]) if "@" in r else (r, tag)


def map_figures(runs, tag, year, out, field="member0"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = np.load(f"{DATA_DIR}/static.npz")
    rows, cols, cell_id = st["valid_rows"], st["valid_cols"], st["era5_cell_id"]
    H, W = [int(v) for v in st["grid_shape"]]
    meta = dict(np.load(f"{DATA_DIR}/year_{year}_meta.npz"))
    dates = [str(d) for d in meta["dates"]]

    specs = dict(_spec(r, tag) for r in runs)
    runs = list(specs)
    fdir = {r: f"{RUNS_ROOT}/{r}/eval_{year}_{specs[r]}" for r in runs}
    have = [r for r in runs if os.path.isdir(fdir[r]) and any(f.startswith("fields_") for f in os.listdir(fdir[r]))]
    if not have:
        print(f"[maps] no fields_*.npz under eval_{year}_{tag} -- run fast_eval.py --tag {tag} first")
        return
    map_dates = sorted(f[len("fields_"):-len(".npz")] for f in os.listdir(fdir[have[0]]) if f.startswith("fields_"))
    std_run = None if NO_SPREAD else next((r for r in have if not is_deterministic(r)), None)

    for ds in map_dates:
        d = dates.index(ds)
        e5 = era5_phys(meta, d, int(meta["doy"][d]))  # (3,71,113)
        fields = {r: np.load(f"{fdir[r]}/fields_{ds}.npz") for r in have if os.path.exists(f"{fdir[r]}/fields_{ds}.npz")}
        for c, v in enumerate(VARS):
            truth = fields[have[0]]["truth"][:, c].astype(np.float32)
            panels = [("ERA5 input (0.25°)", e5[c].reshape(-1)[cell_id]), ("NGCD observations (1 km)" if PAPER else "NGCD truth (1 km)", truth)]
            fname = {"member0": "sample", "ensmean": "ensemble mean"}[field]
            def ptitle(r):
                if PAPER:
                    det = is_deterministic(r) or specs[r].endswith("det0") or r == "w6_unet_reg"  # UNet stage: deterministic
                    return label_of(r, specs[r]) + ("" if det else f"\n{fname}")
                return label_of(r) + ("" if is_deterministic(r) else f"\n{fname} ({specs[r]})")
            panels += [(ptitle(r), fields[r][field][:, c].astype(np.float32)) for r in fields]
            if std_run in fields:
                panels.append((f"{label_of(std_run) if PAPER else std_run}\nensemble std", fields[std_run]["ensstd"][:, c].astype(np.float32)))
            grids = [(t, to_grid(p, rows, cols, H, W)) for t, p in panels]
            fin = truth[np.isfinite(truth)]
            lo, hi = np.percentile(fin, 1), np.percentile(fin, 99)
            zr, zc = zoom_box(grids[1][1])
            zlo, zhi = np.nanpercentile(grids[1][1][zr:zr + ZOOM_KM, zc:zc + ZOOM_KM], [1, 99])

            n = len(grids)
            fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 7.4), squeeze=False,
                                     gridspec_kw={"height_ratios": [1.3, 1]})
            for j, (title, g) in enumerate(grids):
                is_std = title.endswith("ensemble std")
                kw = dict(cmap="viridis") if is_std else dict(cmap="RdYlBu_r", vmin=lo, vmax=hi)
                im = axes[0][j].imshow(g, interpolation="nearest", **kw)
                axes[0][j].add_patch(plt.Rectangle((zc, zr), ZOOM_KM, ZOOM_KM, fill=False, ec="k", lw=1))
                axes[0][j].set_title(title, fontsize=8)
                fig.colorbar(im, ax=axes[0][j], fraction=0.04)
                zkw = dict(cmap="viridis") if is_std else dict(cmap="RdYlBu_r", vmin=zlo, vmax=zhi)
                im = axes[1][j].imshow(g[zr:zr + ZOOM_KM, zc:zc + ZOOM_KM], interpolation="nearest", **zkw)
                axes[1][j].set_title(f"zoom {ZOOM_KM}×{ZOOM_KM} km", fontsize=8)
                fig.colorbar(im, ax=axes[1][j], fraction=0.04)
                for ax in (axes[0][j], axes[1][j]):
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"{v} (°C), {ds}")
            fig.tight_layout()
            path = f"{out}/maps_{ds}_{v}{'' if field == 'member0' else '_' + field}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"wrote {path}", flush=True)


def psd_figure(runs, tag, year, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {r: np.load(f"{RUNS_ROOT}/{r}/eval_{year}_{tag}/psd.npz") for r in runs
            if os.path.exists(f"{RUNS_ROOT}/{r}/eval_{year}_{tag}/psd.npz")}
    if not data:
        print(f"[psd] no eval_{year}_{tag}/psd.npz yet")
        return
    first = next(iter(data.values()))
    k = first["k_cycles_per_km"][1:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for c, v in enumerate(VARS):
        ax = axes[c]
        ax.loglog(k, first["truth"][c][1:], "k-", lw=2.2, label="NGCD truth")
        for r, p in data.items():
            ax.loglog(k, p["member"][c][1:], lw=1.1, label=label_of(r).replace("\n", " "))
        ax.set_title(f"{v} power spectrum (one sample)")
        ax.set_xlabel("wavenumber (cycles/km)")
        ax.grid(alpha=0.3, which="both")
        sec = ax.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
        sec.set_xlabel("wavelength (km)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = f"{out}/psd_compare.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def metrics_figure(runs, tag, year, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    for r in runs:
        p = f"{RUNS_ROOT}/{r}/eval_{year}_{tag}/metrics.json"
        if os.path.exists(p):
            data[r] = json.load(open(p))["groups"]["all"]["year"]
    if not data:
        print(f"[metrics] no eval_{year}_{tag}/metrics.json yet")
        return
    keys = [("crps", "CRPS (°C) ↓"), ("rmse_ensmean", "RMSE ens. mean (°C) ↓"),
            ("spread_skill", "spread / skill (1 = calibrated)"), ("hot_csi", "hot-extreme CSI, 1 sample ↑")]
    names = list(data)
    x = np.arange(len(VARS))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2))
    for i, (key, title) in enumerate(keys):
        ax = axes[i]
        for j, r in enumerate(names):
            ax.bar(x + (j - (len(names) - 1) / 2) * w, [data[r][v][key] for v in VARS], w, label=r)
        if key == "spread_skill":
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(VARS)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{year}, all pixels, every evaluated day")
    fig.tight_layout()
    path = f"{out}/metrics_compare.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    print(f"{'run':22s} " + " | ".join(f"{v} crps rmseM  ss  hotCSI" for v in VARS))
    for r in names:
        print(f"{r:22s} " + " | ".join(
            "{v} {crps:4.2f} {rmse_ensmean:5.2f} {spread_skill:4.2f} {hot_csi:5.2f}".format(v=v, **data[r][v]) for v in VARS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names, in plotting order")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--maps-tag", default="maps", help="eval tag holding fields_<date>.npz")
    ap.add_argument("--metrics-tag", default="best", help="eval tag of the full-year evaluation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-spread", action="store_true", help="omit the ensemble-spread panel (e.g. for deterministic products)")
    ap.add_argument("--field", choices=["member0", "ensmean"], default="member0",
                    help="what to show for generative runs: one sample or the ensemble mean")
    ap.add_argument("--paper-labels", action="store_true", help="readable model names (CD-FFM, Regression, ...) in panel titles")
    a = ap.parse_args()
    global NO_SPREAD, PAPER
    NO_SPREAD = a.no_spread
    PAPER = a.paper_labels
    runs = a.runs.split(",")
    os.makedirs(a.out, exist_ok=True)
    plain = [r.split("@")[0] for r in runs]
    metrics_figure(plain, a.metrics_tag, a.year, a.out)
    psd_figure(plain, a.metrics_tag, a.year, a.out)
    map_figures(runs, a.maps_tag, a.year, a.out, a.field)


if __name__ == "__main__":
    main()
