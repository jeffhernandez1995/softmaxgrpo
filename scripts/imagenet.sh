#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_dir"
: "${IMAGENET_DIR:?Set IMAGENET_DIR to the ImageNet directory containing train/ and val/}"
exec python -m softmaxgrpo.imagenet --data-dir "$IMAGENET_DIR" "$@"
