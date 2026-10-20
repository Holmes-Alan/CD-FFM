#!/usr/bin/env python3
"""
Crop the existing global 0.25-deg ERA5 t2m/tp6h archive to the Nordic domain
(Finland/Sweden/Norway, to match NGCD coverage) and aggregate the 6-hourly
timesteps to daily, to pair with NGCD's daily 1km target grid.

Source (already downloaded, not touched/modified):
    $CDFFM_ERA5_SOURCE/t2m/<year>.nc
    $CDFFM_ERA5_SOURCE/tp6h/<year>.nc

Output:
    $CDFFM_ROOT/era5_nordic_daily/<year>.npz
        lat, lon            : 1D coord arrays (deg), Nordic subset only
        dates               : 'YYYY-MM-DD' strings, one per day
        t2m_mean_degC       : (n_days, n_lat, n_lon) float32, daily mean 2m temp
        t2m_max_degC        : (n_days, n_lat, n_lon) float32, max of the 4 6-hourly samples
                               (NOT a true diurnal max/min - only 4 obs/day; see caveat below)
        t2m_min_degC        : (n_days, n_lat, n_lon) float32
        tp_sum_mm           : (n_days, n_lat, n_lon) float32, daily precip total (mm)

Caveat: ERA5 here is 6-hourly (00/06/12/18 UTC). t2m_max/min are the max/min
of those 4 samples, not the true 24h extrema a station thermometer would record
(which is what NGCD's TX/TN represent). Daily mean and precip sum are much
closer to a fair comparison. Keep this in mind when evaluating the model on
temperature extremes.

Usage (one year per call; configs/commands/01-04_data.txt lists all years):
    python3 subset_aggregate_era5.py --year 2020
"""
import argparse
import os

import h5py
import numpy as np
from paths import CDFFM_ROOT, ERA5_SOURCE_DIR  # noqa: E402  (set CDFFM_ROOT)

SRC_DIR = ERA5_SOURCE_DIR
OUT_DIR = CDFFM_ROOT + "/era5_nordic_daily"

# Nordic bounding box (Finland + Sweden + Norway), with a small margin.
LAT_MIN, LAT_MAX = 54.0, 71.5
LON_MIN, LON_MAX = 4.0, 32.0


def contiguous_index_range(coord, lo, hi):
    """coord is sorted ascending; return (start, stop) for coord[start:stop] covering [lo, hi]."""
    idx = np.where((coord >= lo) & (coord <= hi))[0]
    if len(idx) == 0:
        raise ValueError(f"No grid points found in range [{lo}, {hi}]")
    return int(idx[0]), int(idx[-1]) + 1


def load_cropped(path, varname, lat_slice, lon_slice):
    with h5py.File(path, "r") as f:
        data = f[varname][:, lat_slice, lon_slice]  # hyperslab read, not a full-array load
        time_raw = f["time"][:]
        units = f["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode()
    assert units.startswith("hours since"), f"Unexpected time units: {units}"
    base = np.datetime64(units.replace("hours since ", "").strip(), "h")
    times = base + time_raw.astype("timedelta64[h]")
    return data, times


def daily_aggregate(data, times, how):
    """data: (n_time, n_lat, n_lon); times: datetime64[h]; groups consecutive timesteps by day."""
    days = times.astype("datetime64[D]")
    unique_days, start_idx = np.unique(days, return_index=True)
    start_idx = np.sort(start_idx)
    bounds = list(start_idx) + [len(times)]
    out = np.empty((len(unique_days),) + data.shape[1:], dtype=np.float32)
    for i in range(len(unique_days)):
        chunk = data[bounds[i]:bounds[i + 1]]
        if how == "mean":
            out[i] = chunk.mean(axis=0)
        elif how == "max":
            out[i] = chunk.max(axis=0)
        elif how == "min":
            out[i] = chunk.min(axis=0)
        elif how == "sum":
            out[i] = chunk.sum(axis=0)
        else:
            raise ValueError(how)
    return out, unique_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{year}.npz")
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping.")
        return

    t2m_path = os.path.join(SRC_DIR, "t2m", f"{year}.nc")
    tp_path = os.path.join(SRC_DIR, "tp6h", f"{year}.nc")

    with h5py.File(t2m_path, "r") as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    lat_start, lat_stop = contiguous_index_range(lat, LAT_MIN, LAT_MAX)
    lon_start, lon_stop = contiguous_index_range(lon, LON_MIN, LON_MAX)
    lat_crop = lat[lat_start:lat_stop]
    lon_crop = lon[lon_start:lon_stop]
    lat_slice = slice(lat_start, lat_stop)
    lon_slice = slice(lon_start, lon_stop)

    print(f"[{year}] cropping to lat[{lat_start}:{lat_stop}] ({lat_crop[0]:.2f}-{lat_crop[-1]:.2f}), "
          f"lon[{lon_start}:{lon_stop}] ({lon_crop[0]:.2f}-{lon_crop[-1]:.2f})")

    t2m_k, times = load_cropped(t2m_path, "2m_temperature", lat_slice, lon_slice)
    t2m_c = t2m_k - 273.15

    t2m_mean, days = daily_aggregate(t2m_c, times, "mean")
    t2m_max, _ = daily_aggregate(t2m_c, times, "max")
    t2m_min, _ = daily_aggregate(t2m_c, times, "min")

    tp_m, tp_times = load_cropped(tp_path, "total_precipitation_6hr", lat_slice, lon_slice)
    tp_mm = tp_m * 1000.0
    tp_sum, tp_days = daily_aggregate(tp_mm, tp_times, "sum")

    assert np.array_equal(days, tp_days), "t2m and tp6h day axes do not match"

    date_strs = np.array([str(d) for d in days])

    np.savez_compressed(
        out_path,
        lat=lat_crop.astype(np.float32),
        lon=lon_crop.astype(np.float32),
        dates=date_strs,
        t2m_mean_degC=t2m_mean,
        t2m_max_degC=t2m_max,
        t2m_min_degC=t2m_min,
        tp_sum_mm=tp_sum,
    )
    print(f"[{year}] wrote {out_path}  ({len(days)} days, grid {t2m_mean.shape[1:]})")


if __name__ == "__main__":
    main()
