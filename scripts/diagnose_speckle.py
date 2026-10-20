"""
diagnose_speckle.py
====================
Why is the 32-member ensemble mean of the flow model speckled (pixel-to-pixel noise) in flat
terrain where NGCD and the regression output are smooth? CPU only.

For one map date: truth, regression, flow-model ensemble mean and one sample, in physical units
AND in the normalized z-space the model actually predicts (z = (x - mu_pixel) / sigma_pixel),
plus the per-pixel climatology sigma and mu. If z is smooth but x is speckled, the speckle comes
from the per-pixel normalization; if z itself is speckled, it comes from the network.

Roughness = mean |x - median3x3(x)| over the box (degC or z units): the pixel-scale noise level.

  python3 diagnose_speckle.py --run w2_flower_y15@maps_m32 --ref w1_flower_reg@maps --date 2019-01-15
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import climstats
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
DATA = f"{ROOT}/fast_data"
VARS = ["TG", "TX", "TN"]


def median3(g):
    """3x3 median filter with NaN handling (numpy only)."""
    H, W = g.shape
    p = np.pad(g, 1, mode="edge")
    stack = np.stack([p[i:i + H, j:j + W] for i in range(3) for j in range(3)])
    return np.nanmedian(stack, axis=0)


def roughness(g):
    return float(np.nanmean(np.abs(g - median3(g))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="w2_flower_y15@maps_m32")
    ap.add_argument("--ref", default="w1_flower_reg@maps")
    ap.add_argument("--date", default="2019-01-15")
    ap.add_argument("--var", default="TG")
    ap.add_argument("--out", default=f"{ROOT}/figures/speckle")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = VARS.index(a.var)

    st = np.load(f"{DATA}/static.npz")
    rows, cols, lat, lon = st["valid_rows"], st["valid_cols"], st["valid_lat"], st["valid_lon"]
    H, W = [int(v) for v in st["grid_shape"]]
    meta = np.load(f"{DATA}/year_{a.date[:4]}_meta.npz")
    d = [str(x) for x in meta["dates"]].index(a.date)
    doy = int(meta["doy"][d])
    mu, sig = climstats.eval_harmonic(st["coeffs_mean"][:, c], st["coeffs_var"][:, c], doy)

    def load(spec):
        run, tag = spec.split("@")
        return np.load(f"{ROOT}/runs/{run}/eval_{a.date[:4]}_{tag}/fields_{a.date}.npz")

    f, r = load(a.run), load(a.ref)
    prods = {
        "NGCD truth": f["truth"][:, c].astype(np.float64),
        f"{a.ref.split('@')[0]} (regression)": r["member0"][:, c].astype(np.float64),
        f"{a.run.split('@')[0]} ens. mean": f["ensmean"][:, c].astype(np.float64),
        f"{a.run.split('@')[0]} sample": f["member0"][:, c].astype(np.float64),
    }

    def grid(v):
        g = np.full((H, W), np.nan)
        g[rows, cols] = v
        return g

    # two 128x128 boxes: flat interior (Finnish Lapland) and steep terrain (western Norway)
    boxes = {}
    for name, (la, lo) in {"flat (N Finland)": (67.3, 26.5), "terrain (W Norway)": (61.5, 7.0)}.items():
        i = int(np.argmin((lat - la) ** 2 + ((lon - lo) * np.cos(np.radians(la))) ** 2))
        r0 = int(np.clip(rows[i] - 64, 0, H - 128))
        c0 = int(np.clip(cols[i] - 64, 0, W - 128))
        boxes[name] = (r0, c0)

    sig_g, mu_g = grid(sig), grid(mu)
    print(f"{a.var} {a.date} (doy {doy}); roughness = mean |x - median3x3(x)|")
    print(f"{'product':38s} " + " | ".join(f"{b:>20s} degC   z" for b in boxes))
    rough = {}
    for name, v in prods.items():
        g, gz = grid(v), grid((v - mu) / sig)
        vals = []
        for (r0, c0) in boxes.values():
            sl = (slice(r0, r0 + 128), slice(c0, c0 + 128))
            vals += [roughness(g[sl]), roughness(gz[sl])]
        rough[name] = vals
        print(f"{name:38s} " + " | ".join(f"{vals[2 * k]:20.3f} {vals[2 * k + 1]:5.3f}" for k in range(len(boxes))))
    for lbl, gg in (("sigma_pixel", sig_g), ("mu_pixel", mu_g)):
        vals = []
        for (r0, c0) in boxes.values():
            vals.append(roughness(gg[r0:r0 + 128, c0:c0 + 128]))
        print(f"{lbl:38s} " + " | ".join(f"{v:20.3f}      " for v in vals))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for bname, (r0, c0) in boxes.items():
        sl = (slice(r0, r0 + 128), slice(c0, c0 + 128))
        names = list(prods)
        fig, axes = plt.subplots(2, len(names) + 1, figsize=(3.3 * (len(names) + 1), 6.6))
        t = grid(prods["NGCD truth"])[sl]
        lo_, hi_ = np.nanpercentile(t, [1, 99])
        for j, n in enumerate(names):
            g = grid(prods[n])[sl]
            gz = grid((prods[n] - mu) / sig)[sl]
            im = axes[0][j].imshow(g, cmap="RdYlBu_r", vmin=lo_, vmax=hi_, interpolation="nearest")
            axes[0][j].set_title(f"{n}\n°C, rough {rough[n][2 * list(boxes).index(bname)]:.2f}", fontsize=8)
            fig.colorbar(im, ax=axes[0][j], fraction=0.04)
            im = axes[1][j].imshow(gz, cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
            axes[1][j].set_title(f"z-space, rough {rough[n][2 * list(boxes).index(bname) + 1]:.2f}", fontsize=8)
            fig.colorbar(im, ax=axes[1][j], fraction=0.04)
        im = axes[0][-1].imshow(sig_g[sl], cmap="viridis", interpolation="nearest")
        axes[0][-1].set_title("climatology sigma (°C)", fontsize=8)
        fig.colorbar(im, ax=axes[0][-1], fraction=0.04)
        im = axes[1][-1].imshow(mu_g[sl], cmap="RdYlBu_r", interpolation="nearest")
        axes[1][-1].set_title("climatology mu (°C)", fontsize=8)
        fig.colorbar(im, ax=axes[1][-1], fraction=0.04)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{a.var} {a.date} — {bname}, 128×128 km at 1 km")
        fig.tight_layout()
        path = f"{a.out}/speckle_{a.date}_{a.var}_{bname.split()[0]}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
