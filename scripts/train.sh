#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_dir"
experiment="${1:?Usage: bash scripts/train.sh EXPERIMENT [Hydra overrides...]}"
shift
exec python -m softmaxgrpo.train --config-name "$experiment" "$@"
