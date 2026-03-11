import matplotlib.pyplot as plt
from skimage import io
import os

IMG_DIR = "data/dataset_2D/train/images"
MASK_DIR = "data/dataset_2D/train/masks"

img_name = os.listdir(IMG_DIR)[0]

img = io.imread(os.path.join(IMG_DIR, img_name))
mask = io.imread(os.path.join(MASK_DIR, img_name))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("MRI")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(img, cmap="gray")
plt.imshow(mask, cmap="jet", alpha=0.4)
plt.title("Overlay")

plt.savefig("check_slice.png")
print("Saved check_slice.png")
