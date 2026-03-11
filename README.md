# ATLAS/ISLES 2022 lesion segmentation (baseline pipeline)

This repo provides a minimal, reproducible 3D U-Net baseline for ATLAS v2.0 (MNI space) with Dice loss and
BIDS-derivative prediction outputs suitable for the ISLES 2022 evaluator. **No training notebooks are required.**

## TODO:
Make sure that each run is updating automatically (run_0001 -> run_0002 if 0001 already exists)
## Dataset layout (assumed)
```
data/
  train/derivatives/ATLAS/sub-*/ses-1/anat/
    sub-*_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
    sub-*_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
  test/derivatives/ATLAS/sub-*/ses-1/anat/
    sub-*_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
```
No extra preprocessing is applied beyond per-volume intensity normalization in the dataloader.

## Splits (Tier A/B/C)
Splits are deterministic (seed `9001`), created from **labeled** training subjects:
- **Tier C held-out** = 15% of total labeled subjects
- **Tier B dev/val** = 20% of the remaining
- **Tier A train** = rest

Saved to `splits/atlas_train_dev_test.json` with keys: `train_ids`, `dev_ids`, `heldout_ids`, `seed`, `created_at`, `notes`.

Generate or refresh splits:
```
python scripts/train_base_cnn.py --make_splits
```

## Model & training
- **Model:** MONAI 3D UNet (`in_channels=1`, `out_channels=1`)
- **Loss:** soft Dice (optionally Dice + BCE)
- **Sampling:** patch-based with lesion-biased centers (no augmentations)
- **Output:** sigmoid → threshold 0.5
- **Checkpoint:** `runs/.../checkpoints/best.pt` (best dev Dice each epoch)

### Hardware context
- **Local:** MacBook Air M4, 24GB RAM, MPS (use conservative patch size, batch 1, `num_workers=0`)
- **Lab:** Linux, 120GB RAM, 2–6× RTX 3080 (CUDA; enable AMP; optional DDP)

Set the run directory explicitly (required):
```
export RUN_DIR=runs/base_cnn/run_0003
```

### Example: Linux CUDA (30 epochs, AMP)
```
python scripts/train_base_cnn.py \
  --run_dir runs/base_cnn/run_0003 \
  --data_root data \
  --max_epochs 30 \
  --batch_size 2 \
  --patch_size 128 128 128 \
  --patches_per_volume 8 \
  --num_workers 4 \
```

## Inference (BIDS-derivatives output)
Predictions are written to:
```
runs/<model>/<run_id>/preds_bids/derivatives/atlas2_prediction/sub-*/ses-1/anat/*_mask.nii.gz
```
A `dataset_description.json` is written at both the BIDS root and derivative root.

### Predict on public test (unlabeled)
```
python scripts/infer.py \
  --run_dir $RUN_DIR \
  --split test
```

### Predict on dev split (labeled)
```
python scripts/infer.py \
  --run_dir $RUN_DIR \
  --split dev
```

## Local evaluator (Docker)
Evaluator contract (from `external/evaluator/settings.py`):
- GT root: `/opt/evaluation/ground-truth/derivatives/atlas2`
- Pred root: `/input/derivatives/atlas2_prediction`
- Output: `/output/metrics.json`

Build the evaluator image:
```
./scripts/build_evaluator.sh
```

Evaluate a run on **dev IDs only**:
```
./scripts/eval_one.sh --run_dir $RUN_DIR --split dev
```

## Formatting predictions from a flat folder
If you have raw prediction NIfTIs in a flat directory:
```
python scripts/format_predictions.py \
  --input_dir /path/to/preds \
  --out_root $RUN_DIR/preds_bids \
  --splits_json splits/atlas_train_dev_test.json \
  --split dev
```

## Smoke test
Runs quick checks: data counts, forward pass, one training step, and one test prediction written.
```
python scripts/smoke_test.py
```

## Files/folders added
- `scripts/train_base_cnn.py`, `scripts/infer.py`, `scripts/smoke_test.py`
- `splits/atlas_train_dev_test.json`
- Updated evaluator helpers in `scripts/`
- Run skeleton under `$RUN_DIR/`

## Notes
- Public test labels are hidden; evaluate locally on dev/heldout from `data/train`.
- No augmentations are used in this baseline.

## Lab Runbook (Linux CUDA)
Single copy‑paste chain for a fresh lab machine (adjust only `ATLAS_DATA_ROOT` and GPU count):

```
export ATLAS_DATA_ROOT=/path/to/ATLAS_BIDS
export RUN_DIR=runs/base_cnn/run_0002

git clone <YOUR_REPO_URL> atlas && cd atlas
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/smoke_test.py --data_root "$ATLAS_DATA_ROOT"
python scripts/train_base_cnn.py --make_splits --data_root "$ATLAS_DATA_ROOT"

# Single‑GPU training (30 epochs, AMP, patches_per_volume=8)
python scripts/train_base_cnn.py --data_root "$ATLAS_DATA_ROOT" --run_dir $RUN_DIR --max_epochs 30 --batch_size 2 --patch_size 128 128 128 --patches_per_volume 8 --num_workers 4 --amp --pin_memory

# Multi‑GPU training (2–6 GPUs). Set GPUS then run with torchrun.
export GPUS=2
torchrun --nproc_per_node=$GPUS scripts/train_base_cnn.py --data_root "$ATLAS_DATA_ROOT" --run_dir $RUN_DIR --max_epochs 30 --batch_size 1 --patch_size 128 128 128 --patches_per_volume 8 --num_workers 4 --amp --pin_memory

# Optional: dev inference + evaluator
python scripts/infer.py --data_root "$ATLAS_DATA_ROOT" --run_dir $RUN_DIR --split dev
python scripts/quick_sanity.py --data_root "$ATLAS_DATA_ROOT" --run_dir $RUN_DIR --split dev --n 5
./scripts/build_evaluator.sh
./scripts/eval_one.sh --run_dir $RUN_DIR --split dev --gt_deriv "$ATLAS_DATA_ROOT/train/derivatives/ATLAS"
```

Notes:
- If you hit OOM on 3080s, reduce `--batch_size` to 1 and/or `--patch_size` to `96 96 96`, or increase `--accum_steps` for effective batch size.
- Outputs to copy off the machine:
  - Checkpoint: `$RUN_DIR/checkpoints/best.pt`
  - Training log: `$RUN_DIR/logs/train_log.csv`
  - Predictions: `$RUN_DIR/preds_bids/`
  - Evaluator metrics: `$RUN_DIR/eval_out/metrics.json`

codex resume 019c1f83-ba14-7150-a680-8091a91caf23
