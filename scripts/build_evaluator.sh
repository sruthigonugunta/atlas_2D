#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EVAL_DIR="$ROOT_DIR/external/evaluator"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH" >&2
  exit 1
fi
if [[ ! -d "$EVAL_DIR" ]]; then
  echo "Error: evaluator directory not found at $EVAL_DIR" >&2
  exit 1
fi

PLATFORM_ARGS=()
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  PLATFORM_ARGS=(--platform=linux/amd64)
fi

docker build "${PLATFORM_ARGS[@]}" -t atlas-evaluator:latest "$EVAL_DIR"
