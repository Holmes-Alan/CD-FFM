"""Is NGCD TG consistent with (TX+TN)/2? (plan §4.4 open check: model TG error > TX and TN error.)
CPU, numpy only. Per day of one year: statistics of D = TG - (TX+TN)/2 over
valid pixels, plus how much of TG's day-to-day anomaly variance is NOT explained by (TX+TN)/2."""
import sys

import numpy as np

sys.path.insert(0, ".")
import climstats
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

DATA = CDFFM_ROOT + "/fast_data"
year = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
st = np.load(f"{DATA}/static.npz")
cm, cv = st["coeffs_mean"], st["coeffs_var"]  # (N,3,5)
meta = np.load(f"{DATA}/year_{year}_meta.npz")
tgt = np.load(f"{DATA}/year_{year}_targets.npy", mmap_mode="r")
doys, dates = meta["doy"], meta["dates"]
rows = []
for d in range(0, len(dates), 3):
    z = np.array(tgt[d]).astype(np.float32)  # (N,3)
    phys = np.empty_like(z)
    for c in range(3):
        mu, sig = climstats.eval_harmonic(cm[:, c], cv[:, c], int(doys[d]))
        phys[:, c] = z[:, c] * sig + mu
    ok = np.isfinite(phys).all(1)
    tg, tx, tn = phys[ok, 0], phys[ok, 1], phys[ok, 2]
    D = tg - 0.5 * (tx + tn)
    rows.append((str(dates[d]), D.mean(), D.std(), np.abs(D).mean(), np.percentile(np.abs(D), 99),
                 tg.std(), (0.5 * (tx + tn)).std(), np.corrcoef(tg, 0.5 * (tx + tn))[0, 1]))
    if d % 30 == 0:
        print("%s  mean(TG-(TX+TN)/2)=%+.2f  std=%.2f  mean|.|=%.2f  p99|.|=%.2f  spatial std TG=%.2f mid=%.2f  corr=%.3f"
              % rows[-1], flush=True)
r = np.array([x[1:] for x in rows], dtype=np.float64)
print(f"\n{year} summary over {len(rows)} days: mean D = {r[:, 0].mean():+.2f} degC, spatial std of D = "
      f"{r[:, 1].mean():.2f}, mean |D| = {r[:, 2].mean():.2f}, p99 |D| = {r[:, 3].mean():.2f}, "
      f"spatial corr(TG, mid) = {r[:, 6].mean():.3f}")
for s, months in [("DJF", (12, 1, 2)), ("JJA", (6, 7, 8))]:
    m = np.array([int(x[0][5:7]) in months for x in rows])
    print(f"  {s}: mean D = {r[m, 0].mean():+.2f}, spatial std D = {r[m, 1].mean():.2f}, mean |D| = {r[m, 2].mean():.2f}")
