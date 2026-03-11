import os
import re
import random
from glob import glob

import numpy as np
import nibabel as nib
from skimage import io

DATA_ROOT = "./data/train"
OUT_ROOT = "./data/dataset_2D"

SPLIT = 0.8
PLANE = 2   # axial
SEED = 42


def get_patient(path):
    m = re.search(r"(sub-[^/]+)", path)
    return m.group(1)


def get_slice(vol, axis, z):
    if axis == 2:
        return vol[:, :, z]
    if axis == 1:
        return vol[:, z, :]
    if axis == 0:
        return vol[z, :, :]


def normalize(x):
    x = x.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (255 * x).astype(np.uint8)


def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.imsave(path, arr, check_contrast=False)


def find_pairs():

    t1_files = glob(DATA_ROOT + "/**/*_T1w.nii.gz", recursive=True)
    t1_files = [f for f in t1_files if not os.path.basename(f).startswith("._")]

    pairs = []

    for t1 in t1_files:

        pid = get_patient(t1)

        mask = t1.replace("_T1w.nii.gz",
                          "_label-L_desc-T1lesion_mask.nii.gz")

        if os.path.exists(mask):
            pairs.append((pid, t1, mask))

    return pairs


def split_patients(pairs):

    patients = list(set([p[0] for p in pairs]))

    random.seed(SEED)
    random.shuffle(patients)

    cut = int(len(patients) * SPLIT)

    train_pat = patients[:cut]
    val_pat = patients[cut:]

    return train_pat, val_pat


def build_dataset(pairs, patients, split_name):

    out_img = os.path.join(OUT_ROOT, split_name, "images")
    out_msk = os.path.join(OUT_ROOT, split_name, "masks")

    for pid, t1_path, mask_path in pairs:

        if pid not in patients:
            continue

        vol = nib.load(t1_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        mask = (mask > 0).astype(np.uint8)

        nslices = vol.shape[PLANE]

        for z in range(nslices):

            img = get_slice(vol, PLANE, z)
            msk = get_slice(mask, PLANE, z)

            name = f"{pid}_z{z:03d}.png"

            save(os.path.join(out_img, name), normalize(img))
            save(os.path.join(out_msk, name), msk * 255)


def main():

    print("Recherche des patients...")

    pairs = find_pairs()

    print("Volumes trouvés:", len(pairs))

    train_pat, val_pat = split_patients(pairs)

    print("Patients train:", len(train_pat))
    print("Patients val:", len(val_pat))

    build_dataset(pairs, train_pat, "train")
    build_dataset(pairs, val_pat, "val")

    print("✅ Dataset 2D créé")


if __name__ == "__main__":
    main()
