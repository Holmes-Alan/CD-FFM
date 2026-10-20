"""
paper_summary_figs.py
======================
Summary figures for the paper from existing evaluation outputs (CPU; numpy + matplotlib, Python 3.6).
CPU only; needs a few GB of RAM (map fields are 1.33M-pixel arrays).

  fig_noise_prior.png   power spectra of one sample, TG/TX/TN: truth vs the five noise priors (+ regression),
                        and pixel-scale roughness in a flat and a mountain box vs NGCD
  fig_steps.png         CRPS and spread/skill vs network evaluations per member: Euler (2/5/10/20) and,
                        when available, the distilled flow map (1/2/4)
  fig_rank_hist.png     rank histograms (16 members): final CD-FFM vs FNO flow
  fig_data_scaling.png  CRPS vs number of training years (5/10/15)
Missing inputs are skipped.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_speckle import roughness  # noqa: E402
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
RUNS = f"{ROOT}/runs"
OUT = f"{ROOT}/figures/paper"
T3 = ["TG", "TX", "TN"]
FINAL = "w2e_static_pl24_l8"
NOISE = [("i.i.d.", "w3_iid_static"), ("RBF", "w2d_static_rbf"), ("power law 2-500 km", "w2d_static_pl24"),
         ("power law 8-500 km (final)", FINAL), ("power law 20-500 km", "w2e_static_pl24_l20")]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def year_metrics(run, tag):
    m = jload(f"{RUNS}/{run}/eval_2019_{tag}/metrics.json")
    return None if m is None else m["groups"]["all"]["year"]


def crps(y, v):
    return y[v].get("crps_fair", y[v]["crps"])


def fig_noise_prior(plt):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    truth_done = False
    for lab, run in NOISE + [("regression", "w3_reg_static")]:
        p = f"{RUNS}/{run}/eval_2019_final/psd.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p)
        k = d["k_cycles_per_km"][1:]
        for c, v in enumerate(T3):
            if not truth_done:
                axes[c].loglog(k, d["truth"][c][1:], "k-", lw=2.4, label="NGCD")
            axes[c].loglog(k, d["member"][c][1:], lw=1.2, ls="--" if run == "w3_reg_static" else "-", label=lab)
        truth_done = True
    for c, v in enumerate(T3):
        axes[c].set_title(f"{v}: power spectrum of one sample")
        axes[c].set_xlabel("wavenumber (cycles/km)")
        axes[c].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)

    # roughness (TG, mean |x - median3x3|) in a flat and a mountain box, Jan and Jul, vs NGCD
    st = np.load(f"{ROOT}/fast_data/static.npz")
    rows, cols, lat, lon = st["valid_rows"], st["valid_cols"], st["valid_lat"], st["valid_lon"]
    H, W = [int(x) for x in st["grid_shape"]]
    boxes = {}
    for name, (la, lo) in {"flat": (67.3, 26.5), "mountain": (61.5, 7.0)}.items():
        i = int(np.argmin((lat - la) ** 2 + ((lon - lo) * np.cos(np.radians(la))) ** 2))
        boxes[name] = (int(np.clip(rows[i] - 64, 0, H - 128)), int(np.clip(cols[i] - 64, 0, W - 128)))

    def grid(v):
        g = np.full((H, W), np.nan)
        g[rows, cols] = v
        return g

    labels, vals = [], []
    for lab, run in [("NGCD", None)] + NOISE:
        r_ = []
        for date in ("2019-01-15", "2019-07-28"):
            src = FINAL if run is None else run
            p = f"{RUNS}/{src}/eval_2019_final/fields_{date}.npz"
            if not os.path.exists(p):
                r_ = None
                break
            fld = np.load(p)["truth" if run is None else "member0"][:, 0].astype(np.float64)
            g = grid(fld)
            r_.append([roughness(g[r0:r0 + 128, c0:c0 + 128]) for (r0, c0) in boxes.values()])
        if r_ is not None:
            labels.append(lab)
            vals.append(np.mean(r_, axis=0))
    if vals:
        vals = np.array(vals)
        x = np.arange(len(labels))
        axes[3].bar(x - 0.2, vals[:, 0], 0.4, label="flat (N Finland)")
        axes[3].bar(x + 0.2, vals[:, 1], 0.4, label="mountain (W Norway)")
        axes[3].set_yscale("log")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        axes[3].set_title("TG pixel-scale roughness of one sample (°C)")
        axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_noise_prior.png", dpi=120)
    plt.close(fig)
    print("wrote fig_noise_prior.png", flush=True)


def fig_steps(plt):
    series = {"Euler (final model)": [(2, FINAL, "s7_ode2"), (5, FINAL, "s7_ode5"), (10, FINAL, "s7_ref"),
                                      (20, FINAL, "s7_ode20")],
              "flow map (distilled)": [(1, "w5_flowmap", "s7_map1"), (2, "w5_flowmap", "s7_map2"),
                                       (4, "w5_flowmap", "s7_map4")]}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for name, pts in series.items():
        got = [(n, year_metrics(r, t)) for n, r, t in pts]
        got = [(n, y) for n, y in got if y is not None]
        if not got:
            continue
        n = [g[0] for g in got]
        for c, v in enumerate(T3):
            axes[c].plot(n, [crps(g[1], v) for g in got], "o-", label=name)
        axes[3].plot(n, [g[1]["TG"]["spread_skill"] for g in got], "o-", label=name)
    for c, v in enumerate(T3):
        axes[c].set_title(f"{v} CRPS (°C) vs network evaluations per member")
    axes[3].set_title("TG spread/skill")
    for ax in axes:
        ax.set_xscale("log", basex=2)
        ax.set_xlabel("network evaluations per member")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_steps.png", dpi=120)
    plt.close(fig)
    print("wrote fig_steps.png", flush=True)


def fig_rank_hist(plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for lab, run in [("CD-FFM (final)", FINAL), ("FNO flow (124M)", "w3_fno_final")]:
        y = year_metrics(run, "final")
        if y is None:
            continue
        for c, v in enumerate(T3):
            h = np.array(y[v]["rank_hist"])
            axes[c].step(np.arange(len(h)), h, where="mid", label=lab)
            axes[c].axhline(1.0 / len(h), color="k", lw=0.8, ls="--")
    for c, v in enumerate(T3):
        axes[c].set_title(f"{v} rank histogram (16 members; flat = calibrated)")
        axes[c].set_xlabel("rank of the observation")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rank_hist.png", dpi=120)
    plt.close(fig)
    print("wrote fig_rank_hist.png", flush=True)


def fig_data_scaling(plt):
    pts = [(5, "w1_flower_doy"), (10, "w2_flower_y10"), (15, "w2_flower_y15")]  # same config (RBF noise, no static)

    def fair(run):  # fair CRPS only (unbiased across ensemble sizes): tag 'final', else 'best' if it has it
        for tag in ("final", "best"):
            y = year_metrics(run, tag)
            if y is not None and "crps_fair" in y["TG"]:
                return y
        return None
    got = [(n, fair(r)) for n, r in pts]
    missing = [r for (n, r), (_, y) in zip(pts, got) if y is None]
    if missing:  # never mix estimators in one curve
        print(f"fig_data_scaling skipped: no fair CRPS yet for {missing}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for v in T3:
        ax.plot([g[0] for g in got], [crps(g[1], v) for g in got], "o-", label=v)
    ax.set_xlabel("training years")
    ax.set_ylabel("fair CRPS (°C), 2019")
    ax.set_title("Data scaling (RBF noise, no static descriptors)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_data_scaling.png", dpi=120)
    plt.close(fig)
    print("wrote fig_data_scaling.png", flush=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig_noise_prior(plt)
    fig_steps(plt)
    fig_rank_hist(plt)
    fig_data_scaling(plt)


if __name__ == "__main__":
    main()
