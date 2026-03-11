#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/eval_one.sh --run_dir <runs/base_cnn/run_0002> [--gt_deriv data/train/derivatives/ATLAS] [--splits_json splits/atlas_train_dev_test.json] [--split dev]
  ./scripts/eval_one.sh --gt_deriv <path> --preds_bids <path> --out <path> [--splits_json ...] [--split dev]
USAGE
}

RUN_DIR=""
GT_DERIV=""
PREDS_BIDS=""
OUT_DIR=""
SPLITS_JSON="splits/atlas_train_dev_test.json"
SPLIT_NAME="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    --gt_deriv)
      GT_DERIV="$2"; shift 2 ;;
    --preds_bids)
      PREDS_BIDS="$2"; shift 2 ;;
    --out)
      OUT_DIR="$2"; shift 2 ;;
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

if [[ -n "$RUN_DIR" ]]; then
  PREDS_BIDS="$RUN_DIR/preds_bids"
  OUT_DIR="$RUN_DIR/eval_out"
fi

if [[ -z "$GT_DERIV" ]]; then
  GT_DERIV="data/train/derivatives/ATLAS"
fi

if [[ -z "$PREDS_BIDS" || -z "$OUT_DIR" ]]; then
  usage
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SETTINGS_PY="$ROOT_DIR/external/evaluator/settings.py"

read -r GT_DERIV_NAME PRED_DERIV_NAME GT_ROOT_MOUNT PRED_ROOT_MOUNT METRICS_PATH <<EOF
$(python - <<'PY'
from pathlib import Path
import ast

settings_path = Path("external/evaluator/settings.py").resolve()
text = settings_path.read_text()

def get_list(key: str) -> list[str]:
    for line in text.splitlines():
        if key in line:
            start = line.find("[")
            end = line.find("]", start)
            if start != -1 and end != -1:
                return ast.literal_eval(line[start:end+1])
    raise KeyError(key)

def get_str(key: str) -> str:
    for line in text.splitlines():
        if key in line:
            line = line.split("#", 1)[0]
            parts = line.split(":", 1)
            if len(parts) == 2:
                val = parts[1].strip().strip(",").strip()
                if val.startswith(("\"", "'")) and val.endswith(("\"", "'")):
                    return val[1:-1]
    raise KeyError(key)

print(get_list("GroundTruthBIDSDerivativeName")[0],
      get_list("PredictionBIDSDerivativeName")[0],
      get_str("GroundTruthRoot").rstrip('/'),
      get_str("PredictionRoot").rstrip('/'),
      get_str("MetricsOutputPath").rstrip('/'))
PY
)
EOF

GT_DERIV="$(cd "$GT_DERIV" && pwd -P)"
PREDS_BIDS="$(cd "$PREDS_BIDS" && pwd -P)"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd -P)"
SPLITS_JSON="$(cd "$(dirname "$SPLITS_JSON")" && pwd -P)/$(basename "$SPLITS_JSON")"

if [[ ! -d "$GT_DERIV" ]]; then
  echo "Error: GT derivative not found: $GT_DERIV" >&2
  exit 1
fi

PRED_DERIV_PATH="$PREDS_BIDS/derivatives/$PRED_DERIV_NAME"
if [[ ! -d "$PRED_DERIV_PATH" ]]; then
  echo "Error: prediction derivative not found: $PRED_DERIV_PATH" >&2
  exit 1
fi

if [[ ! -f "$SPLITS_JSON" ]]; then
  echo "Error: splits JSON not found: $SPLITS_JSON" >&2
  exit 1
fi

# Staging roots for BIDS validation and subset selection
STAGING_ROOT="$ROOT_DIR/eval/staging"
GT_ROOT="$STAGING_ROOT/gt_root"
PRED_ROOT="$STAGING_ROOT/preds_root"
mkdir -p "$GT_ROOT/derivatives/$GT_DERIV_NAME" "$PRED_ROOT/derivatives/$PRED_DERIV_NAME"

# Root dataset descriptions
cat > "$GT_ROOT/dataset_description.json" <<JSON
{
  "Name": "atlas2_gt",
  "BIDSVersion": "1.6.0"
}
JSON
cat > "$PRED_ROOT/dataset_description.json" <<JSON
{
  "Name": "$PRED_DERIV_NAME",
  "BIDSVersion": "1.6.0"
}
JSON

# Derivative dataset descriptions (mounted into derivative directories)
GT_DERIV_DESC="$GT_ROOT/${GT_DERIV_NAME}_dataset_description.json"
PRED_DERIV_DESC="$PRED_ROOT/${PRED_DERIV_NAME}_dataset_description.json"
cat > "$GT_DERIV_DESC" <<JSON
{
  "Name": "$GT_DERIV_NAME",
  "BIDSVersion": "1.6.0",
  "GeneratedBy": [{"Name": "$GT_DERIV_NAME"}]
}
JSON
cat > "$PRED_DERIV_DESC" <<JSON
{
  "Name": "$PRED_DERIV_NAME",
  "BIDSVersion": "1.6.0",
  "GeneratedBy": [{"Name": "$PRED_DERIV_NAME"}]
}
JSON

# Build subset symlinks for the requested split
python - <<PY
import json
import shutil
from pathlib import Path

splits = json.loads(Path("$SPLITS_JSON").read_text())
key = f"${SPLIT_NAME}_ids"
ids = splits.get(key, [])
if not ids:
    raise SystemExit(f"Split {key} is empty or missing in {splits.keys()}")

gt_deriv = Path("$GT_DERIV")
pred_deriv = Path("$PRED_DERIV_PATH")

gt_stage = Path("$GT_ROOT") / "derivatives" / "$GT_DERIV_NAME"
pred_stage = Path("$PRED_ROOT") / "derivatives" / "$PRED_DERIV_NAME"

# Clear existing symlinks
for p in gt_stage.glob("sub-*"):
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
for p in pred_stage.glob("sub-*"):
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)

for sid in ids:
    sub = sid if sid.startswith("sub-") else f"sub-{sid}"
    gt_target = f"/gt_full/{sub}"
    pred_target = f"/pred_full/{sub}"
    (gt_stage / sub).symlink_to(gt_target)
    (pred_stage / sub).symlink_to(pred_target)
PY

PLATFORM_ARG=""
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  PLATFORM_ARG="--platform=linux/amd64"
fi

# Run evaluator
set -x
 docker run --rm \
  $PLATFORM_ARG \
  --memory="4g" \
  --init \
  --memory-swap="4g" \
  --network="none" \
  --cap-drop="ALL" \
  --security-opt="no-new-privileges" \
  --shm-size="128m" \
  --pids-limit="256" \
  -v "$GT_ROOT:$GT_ROOT_MOUNT:ro" \
  -v "$GT_DERIV:/gt_full:ro" \
  -v "$GT_DERIV_DESC:$GT_ROOT_MOUNT/derivatives/$GT_DERIV_NAME/dataset_description.json:ro" \
  -v "$PRED_ROOT:$PRED_ROOT_MOUNT:ro" \
  -v "$PRED_DERIV_PATH:/pred_full:ro" \
  -v "$PRED_DERIV_DESC:$PRED_ROOT_MOUNT/derivatives/$PRED_DERIV_NAME/dataset_description.json:ro" \
  -v "$OUT_DIR:/output" \
  atlas-evaluator
set +x

METRICS_FILE="$OUT_DIR/$(basename "$METRICS_PATH")"
if [[ ! -f "$METRICS_FILE" ]]; then
  echo "Error: metrics file not found: $METRICS_FILE" >&2
  exit 1
fi

echo "Metrics written: $METRICS_FILE"
python -m json.tool "$METRICS_FILE" | head -n 20
