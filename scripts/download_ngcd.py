#!/usr/bin/env python3
"""
Download NGCD (Nordic Gridded Climate Dataset) from the Copernicus CDS, one
year at a time (a single all-years request hits CDS's "cost limits exceeded"
- so the request is split by year).

Dataset: insitu-gridded-observations-nordic (1km daily, FI/SE/NO, 1961-present)

One-time setup:
  1. pip install --user cdsapi
  2. A CDS account: put your key in ~/.cdsapirc (see the CDS 'how to API' page) or in the
     environment variable CDSAPI_KEY, and accept the licence of the
     "insitu-gridded-observations-nordic" dataset once on its web page.

REQUEST fields verified against the dataset's actual process JSON schema:
https://cds.climate.copernicus.eu/api/retrieve/v1/processes/insitu-gridded-observations-nordic
(fetched 2026-10-06). No grid_resolution/time_aggregation/format fields exist
(fixed at 1km/daily; format is whatever CDS returns). If CDS changes the
schema later, re-fetch that URL to check.

Usage (one year per call; configs/commands/01-04_data.txt lists all years):
    python3 download_ngcd.py --year 2020
"""
import argparse
import os

import cdsapi
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

# Your CDS key: environment variable CDSAPI_KEY, else cdsapi reads ~/.cdsapirc.
CDS_API_KEY = os.environ.get("CDSAPI_KEY")
CDS_API_URL = "https://cds.climate.copernicus.eu/api"

DATASET = "insitu-gridded-observations-nordic"
OUT_DIR = CDFFM_ROOT + "/ngcd_raw"


def build_request(year):
    return {
        "variable": [
            "mean_temperature",
            "maximum_temperature",
            "minimum_temperature",
            "precipitation",
        ],
        "spatial_interpolation_method": ["type_2"],  # NGCD-2 (Bayesian). Add "type_1" too if you want both.
        "version": ["26_03"],  # latest release batch as of 2026-10
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()


    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.year}.zip")
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping.")
    else:
        client = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY) if CDS_API_KEY else cdsapi.Client()
        client.retrieve(DATASET, build_request(args.year), out_path)
        print(f"Downloaded to {out_path}")
