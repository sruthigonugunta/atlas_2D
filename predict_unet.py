import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


# =========================
# Dataset
# =========================
class MRI2D(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".png") and os.path.exists(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = io.imread(os.path.join(self.img_dir, name))
        mask = io.imread(os.path.join(self.mask_dir, name))

        if img.ndim == 3:
            img = img[..., 0]
        if mask.ndim == 3:
            mask = mask[..., 0]

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy((mask > 0).astype(np.uint8)).long()

        img = (img - img.mean()) / (img.std() + 1e-6)

        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(
            mask.unsqueeze(0).float(),
            self.size,
            interpolation=InterpolationMode.NEAREST
        ).squeeze(0).long()

        return img, mask, name


# =========================
# U-Net
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feats=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_ch
        for i, f in enumerate(feats):
            drop = 0.2 if i >= 2 else 0.0
            self.downs.append(DoubleConv(ch, f, dropout=drop))
            ch = f

        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2, dropout=0.3)

        up_in = feats[-1] * 2
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
            up_in = f

        self.final = nn.Conv2d(feats[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# =========================
# Prediction viz
# =========================
def main():
    val_img_dir = "data/dataset_2D/val/images"
    val_mask_dir = "data/dataset_2D/val/masks"
    ckpt_path = "checkpoints/best_unet.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = MRI2D(val_img_dir, val_mask_dir, size=(256, 256))

    model = UNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    os.makedirs("predictions", exist_ok=True)

    # 5 exemples aléatoires
    indices = random.sample(range(len(ds)), 5)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, name = ds[idx]

            x = img.unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)

            img_np = img[0].cpu().numpy()
            mask_np = mask.cpu().numpy()

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(img_np, cmap="gray")
            plt.title("MRI")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("True mask")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(pred, cmap="gray")
            plt.title("Pred mask")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(img_np, cmap="gray")
            plt.imshow(pred, cmap="jet", alpha=0.4)
            plt.title("Overlay")
            plt.axis("off")

            out_path = os.path.join("predictions", f"pred_{i}_{name}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            print("Saved:", out_path)


if __name__ == "__main__":
    main()
