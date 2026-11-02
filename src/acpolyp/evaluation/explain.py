"""Grad-CAM, permutation importance and TreeSHAP helpers (Phase XVIII step 80)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _last_stage(model: torch.nn.Module) -> torch.nn.Module:
    bb = model.backbone
    if hasattr(bb, "stages"):  # ConvNeXt
        return bb.stages[-1]
    if hasattr(bb, "layer4"):  # ResNet
        return bb.layer4
    if hasattr(bb, "blocks"):  # EfficientNet
        return bb.blocks[-1]
    raise ValueError("unsupported backbone for Grad-CAM")


def gradcam(model: torch.nn.Module, b: dict, device: str, target: int | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) on the last backbone stage for one preprocessed sample."""
    model.eval()
    store: dict[str, torch.Tensor] = {}
    layer = _last_stage(model)
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        kw = {"mask": b["mask"].to(device), "valid": b["valid"].to(device)}
        if b.get("feats") is not None:
            kw["feats"] = b["feats"].to(device)
        x = b["x"].to(device).requires_grad_(False)
        with torch.enable_grad():
            out = model(x, **kw)
            t = int(out.argmax(1)) if target is None else target
            act = store["a"]
            grad = torch.autograd.grad(out[0, t], act)[0]
    finally:
        h.remove()
    w = grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def pointing_game(cam: np.ndarray, gt: np.ndarray) -> bool:
    """True if the CAM maximum falls inside the expert mask."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(gt[y, x] > 0)


def cam_iou(cam: np.ndarray, gt: np.ndarray, top_frac: float = 0.2) -> float:
    """IoU between the top-`top_frac` CAM pixels and the expert mask."""
    thr = np.quantile(cam, 1 - top_frac)
    pred = cam >= thr
    g = gt > 0
    return float((pred & g).sum() / max((pred | g).sum(), 1))
