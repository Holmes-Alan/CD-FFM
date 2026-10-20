"""
influence_figs.py -- plots for influence_maps.py output (CPU; numpy + matplotlib, Python 3.6).

  fig_influence.png   rows: target locations. Columns: (a) CD-FFM likelihood influence map
                      |d log p(y(s*)) / d ERA5|, averaged over days and summed over the 4 input fields;
                      (b) the same for the FNO flow; (c-e) one-point correlation maps of the daily TG anomaly at s*
                      with every 1 km pixel, 2019: NGCD observations, CD-FFM (one sample per day), regression.
Also prints the effective receptive-field radius (distance holding 50 / 90 % of the gradient mass).
"""
import os

import numpy as np
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

ROOT = CDFFM_ROOT
DATA = f"{ROOT}/figures/paper_data"
OUT = f"{ROOT}/figures/paper"


def haversine(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def radius(mass, LAT, LON, lat0, lon0, q):
    d = haversine(lat0, lon0, LAT, LON).ravel()
    m = mass.ravel()
    o = np.argsort(d)
    c = np.cumsum(m[o]) / m.sum()
    return float(d[o][np.searchsorted(c, q)])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    os.makedirs(OUT, exist_ok=True)
    G = np.load(f"{DATA}/influence_grad.npz")
    C = np.load(f"{DATA}/influence_corr.npz") if os.path.exists(f"{DATA}/influence_corr.npz") else None
    st = np.load(f"{ROOT}/fast_data/static.npz")
    rows, cols = st["valid_rows"], st["valid_cols"]
    H, W = [int(v) for v in st["grid_shape"]]
    elat, elon = G["era5_lat"], G["era5_lon"]
    LAT, LON = np.meshgrid(elat, elon, indexing="ij")
    ext = [elon.min() - 0.125, elon.max() + 0.125, elat.min() - 0.125, elat.max() + 0.125]
    sub = slice(None, None, 97)
    vl, vo = st["valid_lat"][sub], st["valid_lon"][sub]
    names = [str(t) for t in G["targets"]]
    ncol = 5 if C is not None else 2
    fig, axes = plt.subplots(len(names), ncol, figsize=(4.0 * ncol, 3.4 * len(names)), squeeze=False)
    print("effective receptive field (km): radius holding 50 % / 90 % of |d log p / d ERA5|, averaged over days")
    for ti, name in enumerate(names):
        lat0, lon0 = float(G["target_lat"][ti]), float(G["target_lon"][ti])
        for k, (key, title) in enumerate([("flower", "CD-FFM"), ("fno", "FNO flow (124M)")]):
            g = np.abs(G[f"{key}_grad_loglik"][ti]).sum(1).mean(0)  # |grad| summed over 4 fields, mean over days
            ax = axes[ti][k]
            if elat[0] > elat[-1]:
                img, origin = g, "upper"
            else:
                img, origin = g, "lower"
            im = ax.imshow(img + 1e-12, extent=ext, origin=origin, cmap="magma", aspect="auto",
                           norm=LogNorm(vmin=g.max() * 1e-3, vmax=g.max()))
            ax.scatter(vo, vl, s=0.05, c="w", alpha=0.15, linewidths=0)
            ax.plot(lon0, lat0, marker="*", ms=12, mfc="cyan", mec="k")
            r50, r90 = radius(g, LAT, LON, lat0, lon0, 0.5), radius(g, LAT, LON, lat0, lon0, 0.9)
            ax.set_title(f"{title}: |∂ log p / ∂ ERA5|\n{name}  (50%: {r50:.0f} km, 90%: {r90:.0f} km)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"  {name:22s} {title:16s} {r50:6.0f} / {r90:6.0f}")
        if C is not None:
            for k, (src, title) in enumerate([("truth", "NGCD observations"), ("flowerffm", "CD-FFM sample"),
                                              ("regression", "regression")]):
                r = C[f"corr_{src}"][ti].astype(np.float32)
                g = np.full((H, W), np.nan, dtype=np.float32)
                g[rows, cols] = r
                rr = np.where(np.isfinite(g).any(1))[0]
                cc = np.where(np.isfinite(g).any(0))[0]
                ax = axes[ti][2 + k]
                im2 = ax.imshow(g[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                pi = int(C["target_idx"][ti])
                ax.plot(cols[pi] - cc[0], rows[pi] - rr[0], marker="*", ms=12, mfc="yellow", mec="k")
                ax.set_title(f"one-point correlation, TG anomaly\n{title}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if ti == 0 and k == 2:
                    fig.colorbar(im2, ax=ax, fraction=0.04)
    fig.suptitle("Where the information for one 1 km location comes from (left) and the learned spatial dependence (right)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_influence.png", dpi=140)
    print(f"wrote {OUT}/fig_influence.png")


if __name__ == "__main__":
    main()
