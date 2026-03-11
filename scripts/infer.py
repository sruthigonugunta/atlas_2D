#!/usr/bin/env python3
"""Run inference and write BIDS-derivative predictions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nb
import torch
from monai.inferers import sliding_window_inference

from train_base_cnn import build_model, normalize_volume, load_nifti, list_labeled_samples


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


def mask_name_from_t1w(t1w_name: str) -> str:
    if "_T1w" in t1w_name:
        return t1w_name.replace("_T1w", "_label-L_desc-T1lesion_mask")
    base = t1w_name.replace(".nii.gz", "").replace(".nii", "")
    return f"{base}_label-L_desc-T1lesion_mask.nii.gz"


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_test_t1w(test_deriv: Path) -> List[Path]:
    return sorted(test_deriv.rglob("*_T1w.nii.gz"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference and write BIDS-derivative predictions.")
    parser.add_argument("--run_dir", required=True, help="Run directory (e.g., runs/base_cnn/run_0002)")
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: run_dir/checkpoints/best.pt)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "heldout", "test"])
    parser.add_argument("--splits_json", default="splits/atlas_train_dev_test.json")
    parser.add_argument("--train_deriv", default=None)
    parser.add_argument("--test_deriv", default=None)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    if args.train_deriv is None:
        args.train_deriv = str(data_root / "train" / "derivatives" / "ATLAS")
    if args.test_deriv is None:
        args.test_deriv = str(data_root / "test" / "derivatives" / "ATLAS")

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    print(f"Device: {device}")

    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    derivative = read_eval_setting_list("PredictionBIDSDerivativeName")[0]
    out_root = run_dir / "preds_bids"
    deriv_root = out_root / "derivatives" / derivative
    deriv_root.mkdir(parents=True, exist_ok=True)
    write_dataset_description(out_root / "dataset_description.json", name=derivative)
    write_dataset_description(deriv_root / "dataset_description.json", name=derivative)

    if args.split == "test":
        t1w_paths = list_test_t1w(Path(args.test_deriv))
    else:
        splits = load_splits(Path(args.splits_json))
        ids = splits[f"{args.split}_ids"]
        samples = list_labeled_samples(Path(args.train_deriv))
        id_set = {sid if sid.startswith("sub-") else f"sub-{sid}" for sid in ids}
        t1w_paths = [s.t1w_path for s in samples if s.subject in id_set]

    if not t1w_paths:
        raise SystemExit("No T1w images found for inference")

    print(f"Running inference on {len(t1w_paths)} volumes ({args.split})")

    for t1w_path in t1w_paths:
        data, affine, header = load_nifti(t1w_path)
        data = normalize_volume(data)
        inp = torch.from_numpy(data[None, None, ...]).float().to(device)

        with torch.no_grad():
            logits = sliding_window_inference(inp, roi_size=tuple(args.patch_size), sw_batch_size=args.sw_batch_size, predictor=model)
            probs = torch.sigmoid(logits)
            pred = (probs > args.threshold).float().cpu().numpy()[0, 0]

        header = header.copy()
        header.set_data_dtype(np.uint8)
        pred_img = nb.Nifti1Image(pred.astype(np.uint8), affine, header)

        # Build output path
        parts = t1w_path.parts
        subject = next(p for p in parts if p.startswith("sub-"))
        session = next((p for p in parts if p.startswith("ses-")), "ses-1")
        out_dir = deriv_root / subject / session / "anat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = mask_name_from_t1w(t1w_path.name)
        out_path = out_dir / out_name
        nb.save(pred_img, str(out_path))
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
