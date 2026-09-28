#!/usr/bin/env bash
#SBATCH --job-name=softmaxgrpo-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=softmaxgrpo-imagenet-%j.out
set -euo pipefail
cd "${SOFTMAXGRPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the release directory}}"
if [[ -n "${SOFTMAXGRPO_ENV_SCRIPT:-}" ]]; then
    source "$SOFTMAXGRPO_ENV_SCRIPT"
fi
exec srun --ntasks=1 bash scripts/imagenet.sh "$@"
