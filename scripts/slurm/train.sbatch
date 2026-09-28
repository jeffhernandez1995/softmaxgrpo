#!/usr/bin/env bash
#SBATCH --job-name=softmaxgrpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=softmaxgrpo-%j.out
set -euo pipefail

# Submit from the release directory. Activate the same environment on every node
# before submitting, or set SOFTMAXGRPO_ENV_SCRIPT to a shared activation script.
# Supply your site's partition/account as sbatch arguments.
cd "${SOFTMAXGRPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the release directory}}"
export SOFTMAXGRPO_ROOT="$PWD"
if [[ -n "${SOFTMAXGRPO_ENV_SCRIPT:-}" ]]; then
    source "$SOFTMAXGRPO_ENV_SCRIPT"
fi
export NNODES="$SLURM_JOB_NUM_NODES"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
experiment="${1:?Usage: sbatch scripts/slurm/train.sbatch EXPERIMENT [Hydra overrides...]}"
shift

if [[ "$NNODES" -eq 1 ]]; then
    exec srun --ntasks=1 bash scripts/train.sh "$experiment" "$@"
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes[0]}"
head_ip="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_ADDRESS="${head_ip}:${RAY_PORT}"
export RAY_HEAD_IP="$head_ip"

# Each Ray process lives inside this job's srun step. Cleanup never stops unrelated jobs.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash scripts/slurm/ray_node.sh &
ray_step_pid=$!
cleanup() { kill "$ray_step_pid" 2>/dev/null || true; wait "$ray_step_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

python - <<'PY'
import os
import time
import ray

deadline = time.monotonic() + 180
while True:
    try:
        ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
        break
    except (ConnectionError, RuntimeError):
        if time.monotonic() > deadline:
            raise
        time.sleep(2)
expected_nodes = int(os.environ["NNODES"])
expected_gpus = expected_nodes * int(os.environ["GPUS_PER_NODE"])
while time.monotonic() < deadline:
    alive = sum(n["Alive"] for n in ray.nodes())
    if alive >= expected_nodes and ray.cluster_resources().get("GPU", 0) >= expected_gpus:
        break
    time.sleep(2)
else:
    raise RuntimeError("Timed out waiting for all allocated Ray nodes and GPUs")
ray.shutdown()
PY
python -m softmaxgrpo.train --config-name "$experiment" "$@"
