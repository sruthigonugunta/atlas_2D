#!/usr/bin/env python3
"""Format predictions into evaluator-ready BIDS-derivatives structure."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nb


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


def write_dataset_description(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Name": name,
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [{"Name": name}],
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_subject_session(path: Path) -> tuple[str, str]:
    subject = ""
    session = ""
    for part in path.parts:
        if part.startswith("sub-") and not subject:
            subject = part
        if part.startswith("ses-") and not session:
            session = part
    if not subject:
        match = re.search(r"(sub-[a-zA-Z0-9]+)", path.name)
        if match:
            subject = match.group(1)
    if not session:
        match = re.search(r"(ses-[a-zA-Z0-9]+)", path.name)
        if match:
            session = match.group(1)
    if not subject:
        raise ValueError(f"Could not parse subject from {path}")
    if not session:
        session = "ses-1"
    return subject, session


def default_mask_name(t1w_name: str) -> str:
    if "_T1w" in t1w_name:
        return t1w_name.replace("_T1w", "_label-L_desc-T1lesion_mask")
    base = t1w_name.replace(".nii.gz", "").replace(".nii", "")
    return f"{base}_label-L_desc-T1lesion_mask.nii.gz"


def find_gt_masks(gt_root: Path) -> list[Path]:
    candidates = list(gt_root.glob("sub-*/ses-*/anat/*mask*.nii*"))
    if not candidates:
        candidates = [p for p in gt_root.rglob("*.nii*") if "mask" in p.name]
    return sorted(candidates)


def create_dummy_predictions(gt_root: Path, out_root: Path, derivative: str, n_subjects: int) -> None:
    mask_files = find_gt_masks(gt_root)
    if not mask_files:
        raise RuntimeError(f"No GT masks found under {gt_root}")

    groups: dict[tuple[str, str], Path] = {}
    for p in mask_files:
        subject, session = parse_subject_session(p)
        key = (subject, session)
        if key not in groups:
            groups[key] = p

    selected = sorted(groups.items())[: max(n_subjects, 0)]
    if not selected:
        raise RuntimeError("No subjects found for dummy prediction generation")

    deriv_root = out_root / "derivatives" / derivative
    deriv_root.mkdir(parents=True, exist_ok=True)
    write_dataset_description(out_root / "dataset_description.json", name=derivative)
    write_dataset_description(deriv_root / "dataset_description.json", name=derivative)

    for (subject, session), gt_path in selected:
        gt_img = nb.load(str(gt_path))
        zeros = np.zeros(gt_img.shape, dtype=np.uint8)
        header = gt_img.header.copy()
        header.set_data_dtype(np.uint8)
        pred_img = nb.Nifti1Image(zeros, gt_img.affine, header)
        out_dir = deriv_root / subject / session / "anat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / gt_path.name
        nb.save(pred_img, str(out_path))
        print(f"Wrote {out_path}")


def iter_pred_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*.nii*")):
        if p.is_file():
            yield p


def format_existing_predictions(
    input_dir: Path,
    out_root: Path,
    derivative: str,
    splits_json: Path | None,
    split: str,
    t1w_root: Path | None,
) -> None:
    deriv_root = out_root / "derivatives" / derivative
    deriv_root.mkdir(parents=True, exist_ok=True)
    write_dataset_description(out_root / "dataset_description.json", name=derivative)
    write_dataset_description(deriv_root / "dataset_description.json", name=derivative)

    allowed_ids: set[str] | None = None
    if splits_json and splits_json.exists():
        data = json.loads(splits_json.read_text())
        key = f"{split}_ids"
        if key in data:
            allowed_ids = {f"sub-{sid}" if not sid.startswith("sub-") else sid for sid in data[key]}

    for pred_path in iter_pred_files(input_dir):
        subject, session = parse_subject_session(pred_path)
        if allowed_ids and subject not in allowed_ids:
            continue

        fname = pred_path.name
        if "_mask" not in fname:
            if t1w_root:
                t1w_candidates = list((t1w_root / subject / session / "anat").glob("*_T1w.nii.gz"))
                if t1w_candidates:
                    fname = default_mask_name(t1w_candidates[0].name)
            if "_mask" not in fname:
                fname = f"{subject}_{session}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"

        out_dir = deriv_root / subject / session / "anat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        shutil.copy2(pred_path, out_path)
        print(f"Copied {pred_path} -> {out_path}")


def main() -> int:
    default_derivative = read_eval_setting_list("PredictionBIDSDerivativeName")[0]

    parser = argparse.ArgumentParser(description="Format predictions into BIDS-derivatives for evaluator.")
    parser.add_argument("--out_root", required=True, help="Output BIDS root (e.g., runs/base_cnn/run_0002/preds_bids)")
    parser.add_argument("--derivative_name", default=default_derivative, help="Prediction derivative name")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gt_root", help="GT derivative root; generate dummy predictions from masks")
    mode.add_argument("--input_dir", help="Directory with existing prediction NIfTI files")

    parser.add_argument("--n_subjects", type=int, default=3, help="Number of subjects for dummy generation")
    parser.add_argument("--splits_json", help="Optional splits JSON for filtering")
    parser.add_argument("--split", default="dev", help="Split key to use when filtering (train/dev/heldout)")
    parser.add_argument("--t1w_root", help="Optional T1w root to derive mask filenames")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    derivative = args.derivative_name

    if args.gt_root:
        create_dummy_predictions(Path(args.gt_root).resolve(), out_root, derivative, args.n_subjects)
        return 0

    input_dir = Path(args.input_dir).resolve()
    splits_json = Path(args.splits_json).resolve() if args.splits_json else None
    t1w_root = Path(args.t1w_root).resolve() if args.t1w_root else None
    if not input_dir.exists():
        raise SystemExit(f"input_dir does not exist: {input_dir}")

    format_existing_predictions(input_dir, out_root, derivative, splits_json, args.split, t1w_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
