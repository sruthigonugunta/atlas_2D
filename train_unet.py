import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


# =========================
# Dataset
# =========================
class MRI2D(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

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

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(
                mask.unsqueeze(0).float(),
                angle,
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0).long()

            if random.random() < 0.3:
                contrast = random.uniform(0.9, 1.1)
                brightness = random.uniform(-0.1, 0.1)
                img = img * contrast + brightness

            if random.random() < 0.3:
                noise = torch.randn_like(img) * 0.03
                img = img + noise

        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(
            mask.unsqueeze(0).float(),
            self.size,
            interpolation=InterpolationMode.NEAREST
        ).squeeze(0).long()

        return img, mask


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
# Metrics
# =========================
@torch.no_grad()
def iou_from_logits(logits, masks, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    inter = (preds * masks).sum((1, 2, 3))
    union = (preds + masks - preds * masks).sum((1, 2, 3))

    return ((inter + eps) / (union + eps)).mean().item()


# =========================
# Train / Val
# =========================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)

    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device).float().unsqueeze(1)

        logits = model(imgs)
        loss = criterion(logits, masks)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        b = imgs.size(0)
        total_loss += loss.item() * b
        total_iou += iou_from_logits(logits, masks) * b
        n += b

    return total_loss / n, total_iou / n


# =========================
# Main
# =========================
def main():
    print("Start training")

    train_img_dir = "data/dataset_2D_jsonsplit/train/images"
    train_mask_dir = "data/dataset_2D_jsonsplit/train/masks"
    val_img_dir = "data/dataset_2D_jsonsplit/val/images"
    val_mask_dir = "data/dataset_2D_jsonsplit/val/masks"

    print("Train images:", len(os.listdir(train_img_dir)))
    print("Train masks :", len(os.listdir(train_mask_dir)))
    print("Val images  :", len(os.listdir(val_img_dir)))
    print("Val masks   :", len(os.listdir(val_mask_dir)))

    train_ds = MRI2D(train_img_dir, train_mask_dir, augment=False)
    val_ds = MRI2D(val_img_dir, val_mask_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    best_iou = -1.0
    EPOCHS = 10

    for epoch in range(EPOCHS):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_iou = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train loss={train_loss:.4f} IoU={train_iou:.4f} | "
              f"val loss={val_loss:.4f} IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "checkpoints/best_unet.pt")
            print("✅ Saved best model")

    print("Done. Best val IoU:", best_iou)


if __name__ == "__main__":
    main()
