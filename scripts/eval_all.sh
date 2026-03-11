#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/eval_all.sh [--runs_root runs] [--gt_deriv data/train/derivatives/ATLAS] [--splits_json splits/atlas_train_dev_test.json] [--split dev]
USAGE
}

RUNS_ROOT="runs"
GT_DERIV="data/train/derivatives/ATLAS"
SPLITS_JSON="splits/atlas_train_dev_test.json"
SPLIT_NAME="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs_root)
      RUNS_ROOT="$2"; shift 2 ;;
    --gt_deriv)
      GT_DERIV="$2"; shift 2 ;;
    --splits_json)
      SPLITS_JSON="$2"; shift 2 ;;
    --split)
      SPLIT_NAME="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
 done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RUNS_ROOT="$(cd "$RUNS_ROOT" && pwd -P)"

found=0
for run_dir in "$RUNS_ROOT"/*/*; do
  if [[ -d "$run_dir" ]]; then
    found=1
    echo "Evaluating $run_dir"
    "$SCRIPT_DIR/eval_one.sh" --run_dir "$run_dir" --gt_deriv "$GT_DERIV" --splits_json "$SPLITS_JSON" --split "$SPLIT_NAME"
  fi
 done

if [[ $found -eq 0 ]]; then
  echo "No run directories found under $RUNS_ROOT" >&2
  exit 1
fi
