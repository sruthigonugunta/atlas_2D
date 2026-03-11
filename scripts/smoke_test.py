#!/usr/bin/env python3
"""Quick smoke test for data + model + inference plumbing."""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import numpy as np
import torch
import nibabel as nb

from train_base_cnn import build_model, normalize_volume, load_nifti, extract_patch
from monai.inferers import sliding_window_inference
from infer import mask_name_from_t1w


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick smoke test for data + model + inference plumbing.")
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/)")
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    train_deriv = data_root / "train" / "derivatives" / "ATLAS"
    test_deriv = data_root / "test" / "derivatives" / "ATLAS"

    t1w_train = sorted(train_deriv.rglob("*_T1w.nii.gz"))
    mask_train = sorted(train_deriv.rglob("*_mask.nii.gz"))
    t1w_test = sorted(test_deriv.rglob("*_T1w.nii.gz"))

    print(f"Train T1w count: {len(t1w_train)}")
    print(f"Train mask count: {len(mask_train)}")
    print(f"Test T1w count: {len(t1w_test)}")

    if not t1w_train or not mask_train or not t1w_test:
        print("Missing expected data files; aborting smoke test")
        return 1

    # Load one train example
    t1w_path = t1w_train[0]
    mask_path = mask_train[0]
    t1w, affine, _hdr = load_nifti(t1w_path)
    mask, _aff2, _hdr2 = load_nifti(mask_path)
    print(f"Sample T1w: {t1w_path}")
    print(f"Shape: {t1w.shape}, dtype: {t1w.dtype}")
    print(f"Affine[0:2]: {affine[:2].tolist()}")

    # Build model and run forward pass on one patch
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    model = build_model().to(device)
    model.eval()

    patch_size = (64, 64, 64)
    t1w_norm = normalize_volume(t1w)
    center = tuple(s // 2 for s in t1w.shape)
    patch = extract_patch(t1w_norm, center, patch_size)
    inp = torch.from_numpy(patch[None, None, ...]).float().to(device)
    with torch.no_grad():
        out = model(inp)
    print(f"Forward pass output shape: {tuple(out.shape)}")

    # One training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(patch[None, None, ...]).float().to(device)
    y_patch = extract_patch(mask, center, patch_size)
    y = torch.from_numpy(y_patch[None, None, ...]).float().to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    # Dice loss (manual)
    inter = (probs * y).sum()
    denom = probs.sum() + y.sum() + 1e-6
    loss = 1.0 - (2.0 * inter + 1e-6) / denom
    loss.backward()
    optimizer.step()
    loss_val = loss.item()
    print(f"One-step loss: {loss_val}")
    if not np.isfinite(loss_val):
        print("Loss is not finite; smoke test failed")
        return 1

    # Inference on one test subject
    test_path = t1w_test[0]
    data, affine, header = load_nifti(test_path)
    data = normalize_volume(data)
    inp = torch.from_numpy(data[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = sliding_window_inference(inp, roi_size=patch_size, sw_batch_size=1, predictor=model)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

    header = header.copy()
    header.set_data_dtype(np.uint8)
    pred_img = nb.Nifti1Image(pred.astype(np.uint8), affine, header)

    out_dir = Path("eval/smoke_pred")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = mask_name_from_t1w(test_path.name)
    out_path = out_dir / out_name
    nb.save(pred_img, str(out_path))
    print(f"Wrote smoke prediction: {out_path}")
    if not out_path.exists():
        print("Smoke prediction not written")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
