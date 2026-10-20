#!/usr/bin/env bash
#SBATCH --job-name=cdffm_cpu
#SBATCH --account=<your_project>
#SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# CPU counterpart of array_gpu.sh, for the data preparation (configs/commands/01-04_data.txt). Submit from the
# repository root; data steps must run in order (download / ERA5 crop -> climatologies -> preprocessing), e.g.
#   grep -E 'download_ngcd' configs/commands/01-04_data.txt > /tmp/01.txt
#   sbatch --array=0-38 slurm/array_cpu.sh /tmp/01.txt
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
LINE=$(grep -vE '^\s*(#|$)' "$1" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$LINE" ]; then echo "no command line for array task $SLURM_ARRAY_TASK_ID"; exit 0; fi
echo "task $SLURM_ARRAY_TASK_ID: $LINE"
cd scripts
eval "$LINE"
