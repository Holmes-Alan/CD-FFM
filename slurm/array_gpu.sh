#!/usr/bin/env bash
#SBATCH --job-name=cdffm
#SBATCH --account=<your_project>
#SBATCH --partition=<gpu_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Runs line <array task id> of a command file (comments and blank lines skipped) in scripts/, on one GPU.
# Submit from the repository root, one array task per command line, e.g.
#   n=$(grep -cvE '^\s*(#|$)' configs/commands/08_eval.txt)
#   sbatch --array=0-$((n - 1)) slurm/array_gpu.sh configs/commands/08_eval.txt
# Time: training and most evaluations need < 4 h on one MI250X die; the CorrDiff-style diffusion stage ~11 h.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
LINE=$(grep -vE '^\s*(#|$)' "$1" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$LINE" ]; then echo "no command line for array task $SLURM_ARRAY_TASK_ID"; exit 0; fi
echo "task $SLURM_ARRAY_TASK_ID: $LINE"
cd scripts
eval "$LINE"
