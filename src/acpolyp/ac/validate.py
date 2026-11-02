"""Quantify active-contour delineation quality against Kvasir-SEG expert masks (Phase XVII)."""

from __future__ import annotations

import cv2
import numpy as np

from ..data.preprocess import canonicalise, crop_resize_mask
from ..paths import KVASIR_SEG_RAW
from .segment import ACResult, central_ellipse, segment

SEG_DIR = KVASIR_SEG_RAW / "Kvasir-SEG"


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    return (2 * inter / max(p.sum() + g.sum(), 1), inter / max(union, 1))


def load_pair(name: str, work: int, size: int = 256):
    """Return (rgb_work, fov_work, gt_work) for a Kvasir-SEG image, after the project's canonicalisation."""
    rgb = cv2.cvtColor(cv2.imread(str(SEG_DIR / "images" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(SEG_DIR / "masks" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
    c = canonicalise(rgb, size)
    gt256 = (crop_resize_mask(gt, c.bbox, size) > 127).astype(np.uint8)
    rs = lambda a, interp: cv2.resize(a, (work, work), interpolation=interp)
    return (
        rs(c.image, cv2.INTER_AREA),
        rs(c.fov, cv2.INTER_NEAREST),
        rs(gt256, cv2.INTER_NEAREST),
        c,
    )


def evaluate(name: str, params: dict) -> dict:
    work = params["work_size"]
    rgb, fov, gt, _ = load_pair(name, work)
    res: ACResult = segment(rgb, fov, params)
    d_final = dice_iou(res.mask, gt)
    d_cv = dice_iou(res.cv_mask, gt)
    d_ell = dice_iou(central_ellipse(fov), gt)
    d_fov = dice_iou(fov, gt)
    return {
        "name": name,
        "dice": d_final[0],
        "iou": d_final[1],
        "dice_cv": d_cv[0],
        "iou_cv": d_cv[1],
        "dice_ellipse": d_ell[0],
        "dice_fov": d_fov[0],
        "fallback": res.fallback,
        "area_frac": res.area_frac,
        "gt_area_frac": float(gt.sum() / max(fov.sum(), 1)),
    }
