"""Where data, runs and figures live. Set CDFFM_ROOT (default: <repository>/work) and, for the ERA5
preprocessing only, CDFFM_ERA5_SOURCE (global 0.25 deg ERA5 t2m / tp6h yearly netCDF files)."""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDFFM_ROOT = os.environ.get("CDFFM_ROOT", os.path.join(_REPO, "work"))
ERA5_SOURCE_DIR = os.environ.get("CDFFM_ERA5_SOURCE", os.path.join(CDFFM_ROOT, "era5_global"))
