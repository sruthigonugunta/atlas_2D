#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchio as tio


# 2D-safe transforms only
AVAILABLE_ARTIFACTS = {
    "bias_field": (tio.RandomBiasField, dict(coefficients=0.3)),
    "blur": (tio.RandomBlur, dict(std=(0.5, 1.5))),
    "noise": (tio.RandomNoise, dict(mean=0, std=(0, 0.03))),
    "gamma": (tio.RandomGamma, dict(log_gamma=(-0.2, 0.2))),
    "affine": (
        tio.RandomAffine,
        dict(
            scales=(0.98, 1.02),
            degrees=(-8, 8, 0, 0, 0, 0),
            translation=(4, 4, 0),
            image_interpolation="linear",
        ),
    ),
    "flip_lr": (tio.RandomFlip, dict(axes=(0,))),
    "flip_ud": (tio.RandomFlip, dict(axes=(1,))),
}


PRESETS = {
    # safest preset for fair 2D vs 3D comparison
    "compare": {
        "bias_field": 0.10,
        "blur": 0.15,
        "noise": 0.15,
        "gamma": 0.10,
        "affine": 0.15,
        "flip_lr": 0.30,
        "flip_ud": 0.30,
    },
    "light": {
        "bias_field": 0.10,
        "blur": 0.15,
        "noise": 0.15,
        "gamma": 0.10,
        "affine": 0.15,
        "flip_lr": 0.30,
        "flip_ud": 0.30,
    },
    "heavy": {
        "bias_field": 0.25,
        "blur": 0.30,
        "noise": 0.30,
        "gamma": 0.20,
        "affine": 0.30,
        "flip_lr": 0.50,
        "flip_ud": 0.50,
    },
}


class TorchIOAugmentation:
    def __init__(self, transform: tio.Compose) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_2d = image.ndim == 2

        if is_2d:
            image = image[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        tensor = torch.from_numpy(image[None, ...]).float()  # (1, H, W, 1)
        augmented = self.transform(tensor)
        result = augmented.squeeze(0).numpy()

        if is_2d:
            result = result[:, :, 0]

        return result, mask


def _build_transforms(artifact_probs: dict) -> List[tio.Transform]:
    transforms = []
    for name, prob in artifact_probs.items():
        if name not in AVAILABLE_ARTIFACTS:
            raise ValueError(f"Unknown artifact '{name}'. Available: {sorted(AVAILABLE_ARTIFACTS)}")
        cls, kwargs = AVAILABLE_ARTIFACTS[name]
        transforms.append(cls(p=prob, **kwargs))
    return transforms


def get_torchio_augmentation(
    artifacts: Optional[List[str]] = None,
    preset: Optional[str] = None,
) -> TorchIOAugmentation:
    if artifacts is not None:
        base_probs = PRESETS["compare"]
        artifact_probs = {name: base_probs.get(name, 0.15) for name in artifacts}
    else:
        preset = preset or "compare"
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}")
        artifact_probs = PRESETS[preset]

    transforms = _build_transforms(artifact_probs)
    return TorchIOAugmentation(tio.Compose(transforms))
