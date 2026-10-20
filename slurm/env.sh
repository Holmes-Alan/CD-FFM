# Sourced by slurm/array_gpu.sh and slurm/array_cpu.sh (from the repository root). Edit for your system.

# Where data, preprocessed arrays, runs and figures live (about 1.3 TB with the raw NGCD archive, 70 GB without).
export CDFFM_ROOT="${CDFFM_ROOT:-$PWD/work}"
# Global 0.25 deg ERA5 archive, only for scripts/subset_aggregate_era5.py: $CDFFM_ERA5_SOURCE/{t2m,tp6h}/<year>.nc
# export CDFFM_ERA5_SOURCE=/path/to/era5

# Python environment with requirements.txt, for example:
# source /path/to/venv/bin/activate

# Example used for the paper (LUMI, AMD Instinct MI250X, ROCm):
#   module load Local-CSC pytorch
#   source /path/to/venv/bin/activate
#   # private MIOpen cache per job (a shared /tmp cache can be owned by another user)
#   export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
#   export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
