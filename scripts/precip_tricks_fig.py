"""Maps for precip_tricks.py (CPU; numpy + matplotlib): NGCD vs samples with the trained (8 km) and smoother
inference-time noise bands, the ensemble mean and the probability-matched means, full domain + a 256 km zoom."""
import os
import sys

import numpy as np
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
date = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    D = np.load(f"{ROOT}/figures/paper_data/precip_tricks_{date}.npz")
    st = np.load(f"{ROOT}/fast_data/static.npz")
    rows, cols = st["valid_rows"], st["valid_cols"]
    H, W = [int(v) for v in st["grid_shape"]]

    def grid(v):
        g = np.full((H, W), np.nan, dtype=np.float32)
        g[rows, cols] = np.where(v > 0.3, v, np.nan)
        return g

    panels = [("NGCD observations", "truth"), ("sample, trained noise 8-500 km", "sample_8"),
              ("sample, noise 25-500 km", "sample_25"), ("sample, noise 75-500 km", "sample_75"),
              ("ensemble mean (16)", "ensmean_8"), ("PM mean, domain-wide", "pm_global_8"), ("PM mean, local", "pm_local_8")]
    panels = [(t, k) for t, k in panels if k in D.files]
    truth = D["truth"].astype(np.float32)
    vmax = max(float(np.nanpercentile(truth, 99.5)), 5.0)
    norm = LogNorm(vmin=0.3, vmax=vmax)
    tg = np.full((H, W), 0.0, dtype=np.float32)
    tg[rows, cols] = np.nan_to_num(truth)
    cs = np.pad(tg.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, r0, c0 = -1, 0, 0
    for r in range(0, H - 256, 32):
        for c in range(0, W - 256, 32):
            v = cs[r + 256, c + 256] - cs[r, c + 256] - cs[r + 256, c] + cs[r, c]
            if v > best:
                best, r0, c0 = v, r, c
    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 6.4), squeeze=False)
    for j, (ttl, k) in enumerate(panels):
        g = grid(D[k].astype(np.float32))
        rr = np.where(np.isfinite(grid(np.ones_like(truth) * 1.0)).any(1))[0]
        cc = np.where(np.isfinite(grid(np.ones_like(truth) * 1.0)).any(0))[0]
        axes[0][j].imshow(g[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1], cmap="YlGnBu", norm=norm, interpolation="nearest")
        axes[0][j].set_title(ttl, fontsize=8)
        im = axes[1][j].imshow(g[r0:r0 + 256, c0:c0 + 256], cmap="YlGnBu", norm=norm, interpolation="nearest")
        axes[1][j].set_title("256 km zoom", fontsize=8)
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, label="mm / day")
    fig.suptitle(f"Precipitation without retraining: inference-time noise band and probability matching, {date}", fontsize=11)
    path = f"{ROOT}/figures/paper/fig_precip_tricks_{date}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
