"""
ngcd_io.py
==========
Read access into the downloaded NGCD yearly zips, without ever repacking
them into a second copy on disk. Each zip member is a single-day,
single-variable netCDF4/HDF5 file (~12-19MB); random access via zipfile's
central directory is fast, so reading on demand (for stats computation or
for a training __getitem__) is practical and avoids ~750GB of redundant
storage a full repack would cost at NGCD's native 2020x1550 resolution.

Grid facts (verified against the actual downloaded data, see
functional_flow_downscaling_plan.md §1): LAEA projection, 2020x1550 cells,
lat/lon are secondary 2D arrays identical across all years (checked
1981 vs 2005), out-of-domain cells use _FillValue=-999.99 (not NaN).
"""
import io
import zipfile
from datetime import date, timedelta

import h5py
import numpy as np
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

NGCD_RAW_DIR = CDFFM_ROOT + "/ngcd_raw"
VARIABLES = {"TG": "mean_temp", "TX": "max_temp", "TN": "min_temp", "RR": "precip"}
GRID_SHAPE = (2020, 1550)
FILL_VALUE = -999.99
# TG/TX/TN are stored in Kelvin (confirmed via the files' own `units` attribute);
# era5_nordic_daily is already in Celsius (see subset_aggregate_era5.py) -- convert
# here, at the I/O boundary, so every consumer gets consistent units for free
# rather than each caller needing to remember a special case.
KELVIN_VARS = {"TG", "TX", "TN"}


def _member_name(variable, d):
    return f"NGCD_{variable}_type2_version_26.03_{d.strftime('%Y%m%d')}.nc"


def _read_member(zf, member):
    """Read one zip member into an h5py.File via an in-memory buffer -- no
    temp file on disk needed (h5py accepts a file-like object directly)."""
    data = zf.read(member)
    return h5py.File(io.BytesIO(data), "r")


def get_static_latlon(raw_dir=NGCD_RAW_DIR, reference_year=1981):
    """lat/lon are identical across years -- read once from any file."""
    zpath = f"{raw_dir}/{reference_year}.zip"
    zf = zipfile.ZipFile(zpath)
    d = date(reference_year, 1, 1)
    f = _read_member(zf, _member_name("RR", d))
    lat = f["lat"][:].astype(np.float32)
    lon = f["lon"][:].astype(np.float32)
    f.close()
    zf.close()
    assert lat.shape == GRID_SHAPE, f"unexpected grid shape {lat.shape}"
    return lat, lon


def year_dates(year):
    d = date(year, 1, 1)
    out = []
    while d.year == year:
        out.append(d)
        d += timedelta(days=1)
    return out


def load_ngcd_day(variable, d, raw_dir=NGCD_RAW_DIR, zf_cache=None):
    """Single (variable, date) -> (2020,1550) float32, FillValue -> NaN.
    zf_cache: optional dict {year: ZipFile} to avoid reopening the same
    zip repeatedly across many calls (caller owns closing them)."""
    year = d.year
    if zf_cache is not None and year in zf_cache:
        zf = zf_cache[year]
    else:
        zf = zipfile.ZipFile(f"{raw_dir}/{year}.zip")
        if zf_cache is not None:
            zf_cache[year] = zf

    f = _read_member(zf, _member_name(variable, d))
    arr = f[variable][0].astype(np.float32)  # shape (1,2020,1550) -> (2020,1550)
    f.close()

    arr = np.where(arr <= FILL_VALUE + 1e-3, np.nan, arr)  # mask BEFORE unit conversion -- fill value isn't a real Kelvin reading
    if variable in KELVIN_VARS:
        arr = arr - 273.15
    return arr


def load_ngcd_year(variable, year, raw_dir=NGCD_RAW_DIR):
    """All days of one (variable, year) -> (n_days,2020,1550) float32 + dates list.
    Memory note: this is ~18GB/year at full precision for all 4 variables
    combined, but ~4.5GB for ONE variable-year -- needs several GB of RAM. Use sample_dates() below for stats computation instead of
    loading full years where possible."""
    dates = year_dates(year)
    zf = zipfile.ZipFile(f"{raw_dir}/{year}.zip")
    out = np.empty((len(dates),) + GRID_SHAPE, dtype=np.float32)
    for i, d in enumerate(dates):
        f = _read_member(zf, _member_name(variable, d))
        arr = f[variable][0].astype(np.float32)
        f.close()
        arr = np.where(arr <= FILL_VALUE + 1e-3, np.nan, arr)
        if variable in KELVIN_VARS:
            arr = arr - 273.15
        out[i] = arr
    zf.close()
    return out, dates


def sample_dates(years, n_samples, rng):
    """Uniformly sample n_samples dates spread across the given years, for
    stats computation (mirrors compute_gpd_stats.py's subsampling approach
    rather than touching every single day of the full 43-year record)."""
    all_dates = [d for y in years for d in year_dates(y)]
    idx = rng.choice(len(all_dates), size=min(n_samples, len(all_dates)), replace=False)
    return [all_dates[i] for i in sorted(idx)]
