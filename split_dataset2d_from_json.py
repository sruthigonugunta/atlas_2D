import os
import json
import shutil

# =========================================================
# Paths
# =========================================================
JSON_PATH = "splits/atlas_train_dev_test.json"
SRC_ROOT = "data/dataset_2D"
OUT_ROOT = "data/dataset_2D_jsonsplit"

# =========================================================
# Helpers
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def patient_id_from_name(fname: str) -> str:
    """
    Exemples de noms possibles :
      sub-r009s097_z111.png
      sub-r009s097_axial_z111.png

    Dans les deux cas, on veut récupérer :
      sub-r009s097
    """
    return fname.split("_")[0]

def collect_pairs(split_root):
    """
    Récupère toutes les paires (image, mask, fname)
    dans:
      split_root/images
      split_root/masks
    """
    img_dir = os.path.join(split_root, "images")
    msk_dir = os.path.join(split_root, "masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        return []

    files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(".png") and os.path.exists(os.path.join(msk_dir, f))
    ])

    pairs = []
    for f in files:
        img_path = os.path.join(img_dir, f)
        msk_path = os.path.join(msk_dir, f)
        pairs.append((img_path, msk_path, f))
    return pairs

# =========================================================
# Main
# =========================================================
def main():
    print("Loading JSON split file...")
    with open(JSON_PATH, "r") as f:
        split_info = json.load(f)

    train_ids = set(split_info["train_ids"])
    val_ids = set(split_info["dev_ids"])         # dev -> val
    test_ids = set(split_info["heldout_ids"])    # heldout -> test

    print(f"Train IDs   : {len(train_ids)}")
    print(f"Val IDs     : {len(val_ids)}")
    print(f"Test IDs    : {len(test_ids)}")

    print("Collecting existing dataset_2D PNG pairs...")
    all_pairs = []
    all_pairs += collect_pairs(os.path.join(SRC_ROOT, "train"))
    all_pairs += collect_pairs(os.path.join(SRC_ROOT, "val"))
    all_pairs += collect_pairs(os.path.join(SRC_ROOT, "test"))  # au cas où il existe déjà

    print(f"Total PNG pairs found: {len(all_pairs)}")

    # Créer l'arborescence de sortie
    for split_name in ["train", "val", "test"]:
        ensure_dir(os.path.join(OUT_ROOT, split_name, "images"))
        ensure_dir(os.path.join(OUT_ROOT, split_name, "masks"))

    # Statistiques
    n_train = 0
    n_val = 0
    n_test = 0
    n_skipped = 0

    patients_seen = {
        "train": set(),
        "val": set(),
        "test": set()
    }

    # Répartition selon le JSON
    print("Copying files according to JSON split...")
    for img_path, msk_path, fname in all_pairs:
        pid = patient_id_from_name(fname)

        if pid in train_ids:
            dst_split = "train"
            n_train += 1
            patients_seen["train"].add(pid)

        elif pid in val_ids:
            dst_split = "val"
            n_val += 1
            patients_seen["val"].add(pid)

        elif pid in test_ids:
            dst_split = "test"
            n_test += 1
            patients_seen["test"].add(pid)

        else:
            n_skipped += 1
            continue

        shutil.copy2(img_path, os.path.join(OUT_ROOT, dst_split, "images", fname))
        shutil.copy2(msk_path, os.path.join(OUT_ROOT, dst_split, "masks", fname))

    print("\n✅ Done.")
    print("--------------------------------------------------")
    print(f"Train PNGs copied : {n_train}")
    print(f"Val PNGs copied   : {n_val}")
    print(f"Test PNGs copied  : {n_test}")
    print(f"Skipped PNGs      : {n_skipped}")
    print("--------------------------------------------------")
    print(f"Train patients found : {len(patients_seen['train'])}")
    print(f"Val patients found   : {len(patients_seen['val'])}")
    print(f"Test patients found  : {len(patients_seen['test'])}")
    print("--------------------------------------------------")
    print(f"Output folder: {OUT_ROOT}")

if __name__ == "__main__":
    main()
