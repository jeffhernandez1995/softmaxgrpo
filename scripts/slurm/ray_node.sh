#!/usr/bin/env bash
set -euo pipefail
cd "${SOFTMAXGRPO_ROOT:?}"
if [[ -n "${SOFTMAXGRPO_ENV_SCRIPT:-}" ]]; then
    source "$SOFTMAXGRPO_ENV_SCRIPT"
fi
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    exec ray start --head --node-ip-address="$RAY_HEAD_IP" --port="$RAY_PORT" \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" --block
else
    # Poll the head socket; tolerate variable scheduler startup time.
    python - <<'PY'
import os
import socket
import time

deadline = time.monotonic() + 180
while True:
    try:
        with socket.create_connection((os.environ["RAY_HEAD_IP"], int(os.environ["RAY_PORT"])), timeout=2):
            break
    except OSError:
        if time.monotonic() > deadline:
            raise
        time.sleep(2)
PY
    exec ray start --address="$RAY_ADDRESS" --num-cpus="$SLURM_CPUS_PER_TASK" \
        --num-gpus="$GPUS_PER_NODE" --block
fi
