#!/usr/bin/env python3
"""Quick sanity check: voxel counts for GT vs predictions on a few dev subjects."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nb
import numpy as np


def read_eval_setting_list(key: str) -> list[str]:
    settings_path = Path(__file__).resolve().parents[1] / "external" / "evaluator" / "settings.py"
    text = settings_path.read_text()
    for line in text.splitlines():
        if key in line:
            start = line.find("[")
            end = line.find("]", start)
            if start != -1 and end != -1:
                import ast

                return ast.literal_eval(line[start : end + 1])
    raise KeyError(f"Setting {key} not found in {settings_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print voxel count sanity stats for dev predictions.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    parser.add_argument("--splits_json", default="splits/atlas_train_dev_test.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    train_deriv = data_root / "train" / "derivatives" / "ATLAS"

    splits = json.loads(Path(args.splits_json).read_text())
    ids = splits.get(f"{args.split}_ids", [])
    ids = ids[: args.n]

    pred_deriv = read_eval_setting_list("PredictionBIDSDerivativeName")[0]
    preds_root = Path(args.run_dir) / "preds_bids" / "derivatives" / pred_deriv

    if not preds_root.exists():
        raise SystemExit(f"Prediction derivative not found: {preds_root}")

    print(f"Checking {len(ids)} subjects from split '{args.split}'")
    for sid in ids:
        sub = sid if sid.startswith("sub-") else f"sub-{sid}"
        gt_mask = train_deriv / sub / "ses-1" / "anat" / f"{sub}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        pred_mask = preds_root / sub / "ses-1" / "anat" / f"{sub}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"

        if not gt_mask.exists() or not pred_mask.exists():
            print(f"{sub}: missing GT or pred (gt={gt_mask.exists()} pred={pred_mask.exists()})")
            continue

        gt = nb.load(str(gt_mask)).get_fdata() > 0.5
        pred = nb.load(str(pred_mask)).get_fdata() > 0.5
        inter = np.logical_and(gt, pred).sum()
        gt_sum = gt.sum()
        pred_sum = pred.sum()
        print(f"{sub}: gt={gt_sum} pred={pred_sum} overlap={inter}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
