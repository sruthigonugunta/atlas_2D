#!/usr/bin/env python3
import os, re, json, random
from glob import glob

import numpy as np
import nibabel as nib
from skimage import io

DATA_ROOT = "/home/GTL/gtchoupe/Desktop/atlas_stability/data"
OUT_ROOT  = "/home/GTL/gtchoupe/Desktop/atlas_stability/data/dataset_2D"

PLANE = "axial"
LESION_RATIO = 0.80
SEED = 42
MAX_NONLESION_PER_PATIENT = 60  # None si tu veux pas limiter


def _pid(path):
    m = re.search(r"(sub-[^/]+)", path)
    return m.group(1) if m else None

def _uint8_img(x):
    x = x.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = x - x.min()
    mx = x.max()
    if mx > 0: x = x / mx
    return (255 * x).astype(np.uint8)

def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.imsave(path, arr, check_contrast=False)

def _get2d(vol, axis, z):
    if axis == 2: return vol[:, :, z]
    if axis == 1: return vol[:, z, :]
    if axis == 0: return vol[z, :, :]
    raise ValueError

def find_pairs(split_root):
    # EXACTEMENT ton arborescence ATLAS
    t1s = glob(os.path.join(split_root, "derivatives", "ATLAS", "**", "anat", "*_T1w.nii*"), recursive=True)
    t1s = [p for p in t1s if not os.path.basename(p).startswith("._")]

    pairs = []
    for t1 in t1s:
        pid = _pid(t1)
        if pid is None:
            continue
        anat = os.path.dirname(t1)
        masks = glob(os.path.join(anat, "*_label-L_desc-T1lesion_mask.nii*"))
        masks = [m for m in masks if (not os.path.basename(m).startswith("._")) and (pid in m)]
        if masks:
            pairs.append((pid, t1, masks[0]))
    return pairs

def build(split_name):
    rnd = random.Random(SEED)
    split_root = os.path.join(DATA_ROOT, split_name)
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[PLANE]

    out_img = os.path.join(OUT_ROOT, split_name, "images")
    out_msk = os.path.join(OUT_ROOT, split_name, "masks")

    pairs = find_pairs(split_root)
    if not pairs:
        raise RuntimeError(f"Aucune paire trouvée dans {split_name}.")

    # -------- Pass 1: compter slices lésées et collecter indices non-lésion (sans stocker les arrays)
    lesion_jobs = []          # (pid, t1_path, mk_path, z)
    non_jobs = []             # idem, mais on va en sampler une partie
    non_by_pid_count = {}     # pour limiter max non-lésion / patient

    for pid, t1_path, mk_path in pairs:
        non_by_pid_count[pid] = 0

        vol = nib.load(t1_path).dataobj  # memory-mapped
        msk = nib.load(mk_path).dataobj

        n = vol.shape[axis]
        for z in range(n):
            m2d = _get2d(msk, axis, z)
            # m2d est un proxy memmap -> any() reste léger
            if np.any(np.asarray(m2d) > 0):
                lesion_jobs.append((pid, t1_path, mk_path, z))
            else:
                if MAX_NONLESION_PER_PATIENT is None or non_by_pid_count[pid] < MAX_NONLESION_PER_PATIENT:
                    non_jobs.append((pid, t1_path, mk_path, z))
                    non_by_pid_count[pid] += 1

    L = len(lesion_jobs)
    if L == 0:
        raise RuntimeError(f"Aucune slice lésée trouvée dans {split_name}.")

    target_non = int(round(L * (1 - LESION_RATIO) / LESION_RATIO))
    rnd.shuffle(non_jobs)
    non_keep = non_jobs[:min(target_non, len(non_jobs))]

    jobs = lesion_jobs + non_keep
    rnd.shuffle(jobs)

    # -------- Pass 2: écrire les PNG (on recharge le volume par patient au fur et à mesure)
    # Optim: cache le dernier patient pour éviter de recharger à chaque slice
    last_pid = None
    last_vol = None
    last_msk = None

    for pid, t1_path, mk_path, z in jobs:
        if pid != last_pid:
            last_vol = nib.load(t1_path).get_fdata()
            last_msk = (nib.load(mk_path).get_fdata() > 0).astype(np.uint8)
            last_pid = pid

        img2d = _get2d(last_vol, axis, z)
        msk2d = _get2d(last_msk, axis, z)

        fname = f"{pid}_{PLANE}_z{z:03d}.png"
        _save(os.path.join(out_img, fname), _uint8_img(img2d))
        _save(os.path.join(out_msk, fname), (msk2d.astype(np.uint8) * 255))

    stats = {
        "split": split_name,
        "patients": len(set([p for p,_,_ in pairs])),
        "lesion_slices": L,
        "nonlesion_slices_kept": len(non_keep),
        "total_png": len(jobs),
        "achieved_lesion_ratio": L / len(jobs),
        "plane": PLANE,
        "max_nonlesion_per_patient": MAX_NONLESION_PER_PATIENT,
    }
    return stats

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    train_stats = build("train")
    test_stats  = build("test")

    with open(os.path.join(OUT_ROOT, "stats.json"), "w") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, indent=2)

    print("✅ Done")
    print(train_stats)
    print(test_stats)
    print("Output:", OUT_ROOT)

if __name__ == "__main__":
    main()