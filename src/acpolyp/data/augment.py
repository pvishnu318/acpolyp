"""Batch-wise GPU augmentation and the shared evaluation transform (Phase X step 40).

One affine warp (random-resized crop, flips, rotation) is applied to image, contour mask and valid mask
with the same sampling grid, so the contour stays aligned with the pixels. The identical `eval_transform`
is used for validation/test, the CLI and the web server, which removes train/serve preprocessing skew.

Inputs  img  (B,3,S,S) uint8, mask (B,1,S,S) uint8, valid (B,1,S,S) uint8
Outputs x    (B,3,out,out) float32 ImageNet-normalised, zero outside the valid FOV
        mask (B,1,out,out) float32 in {0,1},  valid (B,1,out,out) float32 in {0,1}
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _norm(img01: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, device=img01.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=img01.device).view(1, 3, 1, 1)
    return (img01 - m) / s


def eval_transform(img: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, out: int = 224):
    x = img.float() / 255.0
    if x.shape[-1] != out:
        x = F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False)
    m = F.interpolate(mask.float(), size=(out, out), mode="nearest")
    v = F.interpolate(valid.float(), size=(out, out), mode="nearest")
    return _norm(x) * v, m, v


def color_jitter(x: torch.Tensor, gen: torch.Generator, b: float = 0.25, c: float = 0.25, s: float = 0.2) -> torch.Tensor:
    """Per-sample brightness / contrast / saturation jitter on a [0,1] image batch."""
    B = x.shape[0]
    u = lambda w: 1.0 + (torch.rand(B, 1, 1, 1, device=x.device, generator=gen) * 2 - 1) * w
    x = x * u(b)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * u(c) + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * u(s) + gray
    return x.clamp(0, 1)


def train_transform(img: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, gen: torch.Generator, out: int = 224,
                    scale: tuple[float, float] = (0.55, 1.0), ratio: tuple[float, float] = (0.9, 1.1)):
    B, dev = img.shape[0], img.device
    r = lambda *shape: torch.rand(*shape, device=dev, generator=gen)
    area = scale[0] + (scale[1] - scale[0]) * r(B)
    logr = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * r(B)
    asp = torch.exp(logr)
    w, h = torch.sqrt(area * asp).clamp(max=1.0), torch.sqrt(area / asp).clamp(max=1.0)  # crop size as image fraction
    tx, ty = (r(B) * 2 - 1) * (1 - w), (r(B) * 2 - 1) * (1 - h)                             # centre offsets keep the crop inside
    ang = (r(B) * 2 - 1) * math.pi
    fx = torch.where(r(B) < 0.5, -1.0, 1.0).to(dev)
    fy = torch.where(r(B) < 0.5, -1.0, 1.0).to(dev)
    cos, sin = torch.cos(ang), torch.sin(ang)
    # output coords -> source coords: crop scale, then rotate, then flip, then translate (normalised [-1,1] space)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * cos * fx
    theta[:, 0, 1] = -h * sin * fy
    theta[:, 1, 0] = w * sin * fx
    theta[:, 1, 1] = h * cos * fy
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(img.float() / 255.0, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    v = F.grid_sample(valid.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    x = color_jitter(x, gen)
    return _norm(x) * v, m, v
