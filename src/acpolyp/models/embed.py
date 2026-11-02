"""Frozen-backbone embedding extraction and caching (Phase XV step 64, Phase XII step 51)."""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn.functional as F

from ..paths import INTERIM, ensure

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(batch_u8: np.ndarray, size: int, device: str) -> torch.Tensor:
    """(B,H,W,3) uint8 -> normalised (B,3,size,size) float tensor on device."""
    x = (
        torch.from_numpy(np.ascontiguousarray(batch_u8)).to(device).permute(0, 3, 1, 2).float()
        / 255.0
    )
    if x.shape[-1] != size:
        x = F.interpolate(
            x, size=(size, size), mode="bilinear", antialias=True, align_corners=False
        )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def create_backbone(model_name: str, size: int, **kw) -> torch.nn.Module:
    """timm backbone with pooled features; ViT-style models get their pos-embeddings resized to `size`."""
    try:
        return timm.create_model(model_name, pretrained=True, num_classes=0, img_size=size, **kw)
    except TypeError:  # CNNs do not take img_size
        return timm.create_model(model_name, pretrained=True, num_classes=0, **kw)


@torch.inference_mode()
def extract(
    model_name: str, images: np.ndarray, size: int = 224, bs: int = 64, device: str = "cuda"
) -> np.ndarray:
    """Return L2-unnormalised pooled embeddings (N, D) from a frozen pretrained timm backbone."""
    model = create_backbone(model_name, size).to(device).eval()
    out = []
    for i in range(0, len(images), bs):
        x = to_tensor(images[i : i + bs], size, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def cached(model_name: str, key: str = "canon", size: int = 224) -> np.ndarray:
    """Load or compute embeddings for the cached image array `images_{key}.npy`."""
    ensure(INTERIM)
    f = INTERIM / f"emb_{model_name.split('.')[0]}_{key}.npy"
    if f.exists():
        return np.load(f)
    imgs = np.load(INTERIM / f"images_{key}.npy", mmap_mode="r")
    emb = extract(model_name, imgs, size=size)
    np.save(f, emb)
    return emb
