#!/usr/bin/env python3
"""Minimal 3D UNet training for ATLAS v2.0 (MNI space)."""
from __future__ import annotations

import argparse
import json
import subprocess
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nb
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference


@dataclass
class Sample:
    subject: str
    t1w_path: Path
    mask_path: Path


def list_labeled_samples(deriv_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for sub_dir in sorted(deriv_root.glob("sub-*/ses-1/anat")):
        subject = sub_dir.parent.parent.name
        t1w = sub_dir / f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
        mask = sub_dir / f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        if t1w.exists() and mask.exists():
            samples.append(Sample(subject=subject, t1w_path=t1w, mask_path=mask))
    return samples


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nb.Nifti1Header]:
    img = nb.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if np.any(mask):
        mean = vol[mask].mean()
        std = vol[mask].std()
    else:
        mean = vol.mean()
        std = vol.std()
    if std == 0:
        std = 1.0
    return (vol - mean) / std


def extract_patch(vol: np.ndarray, center: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    slices = []
    pads = []
    for i, (c, p, dim) in enumerate(zip(center, patch_size, vol.shape)):
        start = c - p // 2
        end = start + p
        if start < 0:
            end -= start
            start = 0
        if end > dim:
            start -= (end - dim)
            end = dim
        start = max(start, 0)
        end = min(end, dim)
        slices.append(slice(start, end))
        size = end - start
        pad_before = 0
        pad_after = p - size
        pads.append((pad_before, max(pad_after, 0)))
    patch = vol[tuple(slices)]
    if any(p[1] > 0 for p in pads):
        patch = np.pad(patch, pads, mode="constant", constant_values=0)
    return patch


def sample_center(mask: np.ndarray, patch_size: Tuple[int, int, int], lesion_prob: float) -> Tuple[int, int, int]:
    if mask is not None and mask.max() > 0 and random.random() < lesion_prob:
        coords = np.argwhere(mask > 0)
        idx = random.randrange(coords.shape[0])
        center = coords[idx]
        return int(center[0]), int(center[1]), int(center[2])
    return (
        random.randrange(mask.shape[0]),
        random.randrange(mask.shape[1]),
        random.randrange(mask.shape[2]),
    )


class PatchDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        patch_size: Tuple[int, int, int],
        patches_per_volume: int,
        lesion_prob: float,
    ) -> None:
        self.samples = samples
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.lesion_prob = lesion_prob

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_volume

    def __getitem__(self, idx: int):
        sample = self.samples[idx // self.patches_per_volume]
        vol, _, _ = load_nifti(sample.t1w_path)
        mask, _, _ = load_nifti(sample.mask_path)
        vol = normalize_volume(vol)
        center = sample_center(mask, self.patch_size, self.lesion_prob)
        vol_patch = extract_patch(vol, center, self.patch_size)
        mask_patch = extract_patch(mask, center, self.patch_size)
        vol_patch = torch.from_numpy(vol_patch[None, ...]).float()
        mask_patch = torch.from_numpy(mask_patch[None, ...]).float()
        return vol_patch, mask_patch


class VolumeDataset(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        vol, _, _ = load_nifti(sample.t1w_path)
        mask, _, _ = load_nifti(sample.mask_path)
        vol = normalize_volume(vol)
        vol = torch.from_numpy(vol[None, ...]).float()
        mask = torch.from_numpy(mask[None, ...]).float()
        return vol, mask, sample.subject


def build_model() -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    )


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> Tuple[bool, int]:
    if not is_distributed():
        return False, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_splits(subject_ids: List[str], seed: int) -> dict:
    ids = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_heldout = int(n_total * 0.15)
    n_dev = int((n_total - n_heldout) * 0.20)
    heldout = ids[:n_heldout]
    dev = ids[n_heldout : n_heldout + n_dev]
    train = ids[n_heldout + n_dev :]
    if set(train) & set(dev) or set(train) & set(heldout) or set(dev) & set(heldout):
        raise RuntimeError("Split overlap detected")
    return {
        "train_ids": train,
        "dev_ids": dev,
        "heldout_ids": heldout,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Deterministic split: 15% heldout, 20% of remainder dev.",
    }


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text())


def select_samples(samples: List[Sample], ids: List[str]) -> List[Sample]:
    id_set = {sid if sid.startswith("sub-") else f"sub-{sid}" for sid in ids}
    return [s for s in samples if s.subject in id_set]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline 3D UNet on ATLAS v2.0.")
    parser.add_argument("--data_root", default=None, help="Dataset root (contains train/ and test/). Overrides --gt_deriv if provided.")
    parser.add_argument("--gt_deriv", default=None, help="GT derivative root (defaults to <data_root>/train/derivatives/ATLAS)")
    parser.add_argument("--run_dir", required=True, help="Run directory (e.g., runs/base_cnn/run_0002)")
    parser.add_argument("--splits_json", default="splits/atlas_train_dev_test.json", help="Split JSON path")
    parser.add_argument("--make_splits", action="store_true", help="Create splits and exit")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--patches_per_volume", type=int, default=8)
    parser.add_argument("--lesion_prob", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true", help="Force pin_memory=True")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--loss", default="dice", choices=["dice", "dice_bce"])
    args = parser.parse_args()

    set_seed(args.seed)
    distributed, local_rank = setup_distributed()

    data_root = Path(args.data_root or os.environ.get("ATLAS_DATA_ROOT", "data"))
    if args.gt_deriv is None:
        args.gt_deriv = str(data_root / "train" / "derivatives" / "ATLAS")
    gt_root = Path(args.gt_deriv)
    samples = list_labeled_samples(gt_root)
    if not samples:
        raise SystemExit(f"No labeled samples found under {gt_root}")

    splits_path = Path(args.splits_json)
    if args.make_splits or not splits_path.exists():
        splits = make_splits([s.subject for s in samples], args.seed)
        save_splits(splits, splits_path)
        if args.make_splits:
            print(f"Wrote splits to {splits_path}")
            cleanup_distributed()
            return 0
    else:
        splits = load_splits(splits_path)

    train_samples = select_samples(samples, splits["train_ids"])
    dev_samples = select_samples(samples, splits["dev_ids"])

    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "preds_bids").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_ds = PatchDataset(train_samples, tuple(args.patch_size), args.patches_per_volume, args.lesion_prob)
    dev_ds = VolumeDataset(dev_samples)

    if not distributed or dist.get_rank() == 0:
        print(f"Device: {device}")
        print(f"Train subjects: {len(train_samples)} | Dev subjects: {len(dev_samples)}")

    train_sampler = DistributedSampler(train_ds) if distributed else None

    if args.num_workers is None:
        args.num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = args.pin_memory or torch.cuda.is_available()

    repo_root = Path(__file__).resolve().parents[1]
    git_commit = get_git_commit(repo_root)
    config_path = run_dir / "config" / "train_config.json"
    config_path.write_text(json.dumps({"args": vars(args), "git_commit": git_commit}, indent=2))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model().to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_dice = -1.0
    log_path = run_dir / "logs" / "train_log.csv"
    if not log_path.exists() and (not distributed or dist.get_rank() == 0):
        log_path.write_text("epoch,train_loss,dev_dice\n")

    for epoch in range(1, args.max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(distributed and dist.get_rank() != 0))
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y.to(device)

            use_amp = args.amp and torch.cuda.is_available()
            context = torch.cuda.amp.autocast(enabled=use_amp)
            with context:
                logits = model(x)
                if args.loss == "dice_bce":
                    loss = 0.5 * dice_loss(logits, y) + 0.5 * bce_loss(logits, y)
                else:
                    loss = dice_loss(logits, y)
                loss = loss / args.accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.accum_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * args.accum_steps
            pbar.set_postfix({"loss": running / step})

        dev_dice = 0.0
        if epoch % args.val_interval == 0 and (not distributed or dist.get_rank() == 0):
            model.eval()
            with torch.no_grad():
                scores = []
                for x, y, _sid in tqdm(dev_loader, desc="Valid", leave=False):
                    x = x.to(device)
                    y = y.to(device)
                    logits = sliding_window_inference(x, roi_size=tuple(args.patch_size), sw_batch_size=1, predictor=model)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    scores.append(dice_score(preds, y))
                dev_dice = float(np.mean(scores)) if scores else 0.0

            ckpt_path = run_dir / "checkpoints" / "best.pt"
            if dev_dice > best_dice:
                best_dice = dev_dice
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, ckpt_path)

        if not distributed or dist.get_rank() == 0:
            with log_path.open("a") as f:
                f.write(f"{epoch},{running / max(1, len(train_loader))},{dev_dice}\n")

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
