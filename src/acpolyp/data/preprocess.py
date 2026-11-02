"""Canonicalisation of raw endoscopy frames (Phase VII, steps 24 and 40).

Kvasir frames carry acquisition artefacts that correlate with the class label: image
resolution (upper-GI vs colon equipment), black margins, burned-in text and the green/teal
ScopeGuide picture-in-picture box. We (1) detect the endoscopic field of view (FOV), (2) crop to
it, (3) fill the ScopeGuide box, and (4) resize to a fixed square. The artefact statistics are
returned so that the shortcut audit can quantify how much they leak the label.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Canon:
    image: np.ndarray        # (S, S, 3) uint8 RGB, FOV-cropped
    fov: np.ndarray          # (S, S) uint8 {0,1}: valid pixels = inside the endoscopic FOV and not masked
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in original pixels
    teal_frac: float         # fraction of the original frame covered by the ScopeGuide box
    fov_frac: float          # FOV area / original frame area
    orig_hw: tuple[int, int]


def fov_mask(rgb: np.ndarray, thr: int = 18) -> np.ndarray:
    """Binary mask of the endoscopic field of view: largest bright blob, holes filled, convex."""
    h, w = rgb.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    m = (gray > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)        # strips text strokes and thin lines
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cnts))
    out = np.zeros_like(comp)
    cv2.fillConvexPoly(out, hull, 1)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def teal_mask(rgb: np.ndarray) -> np.ndarray:
    """ScopeGuide picture-in-picture: flat teal/green rectangle. Returns a dilated binary mask."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    # Teal: G is close to B (G-B in [-15, 45]) and both well above R. Blue indigo-carmine dye has
    # B >> G and is textured, so it is excluded by the G-B test and the flatness test below.
    colour = (g - r > 70) & (b - r > 50) & ((g - b) > -15) & ((g - b) < 45)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (5, 5))
    var = cv2.blur(gray * gray, (5, 5)) - mu * mu
    m = (colour & (var < 25.0)).astype(np.uint8)
    h, w = m.shape
    if m.sum() < 0.004 * h * w:
        return np.zeros_like(m)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = cent[i]
        # the box is a large, filled rectangle in the lower-left of the frame
        if (area > 0.01 * h * w and area / float(bw * bh) > 0.8 and cx < 0.45 * w and cy > 0.5 * h):
            out[lab == i] = 1
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


# ScopeGuide box location measured on 419 flagged 720x576 frames (p5..p95 of relative bbox:
# x 0.047-0.058..0.351, y 0.660..0.965). The mask is applied to EVERY frame so that the mere
# presence of a masked patch carries no label information.
BOX_REL = (0.04, 0.63, 0.36, 0.97)  # x0, y0, x1, y1 as fractions of the original frame


def naive_resize(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """What a pipeline without artefact handling does: squash the whole frame to size x size."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def canonicalise(rgb: np.ndarray, size: int = 256) -> Canon:
    h, w = rgb.shape[:2]
    fov = fov_mask(rgb)
    teal = teal_mask(rgb)
    ys, xs = np.where(fov > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    work = rgb.copy()
    bx0, by0, bx1, by1 = BOX_REL
    work[int(by0 * h) : int(by1 * h), int(bx0 * w) : int(bx1 * w)] = 0
    if teal.any():  # any box that escaped the canonical rectangle
        work[teal > 0] = 0
    work[fov == 0] = 0
    valid = fov.copy()  # pixels that are inside the FOV AND not occluded by the artefact mask
    valid[int(by0 * h) : int(by1 * h), int(bx0 * w) : int(bx1 * w)] = 0
    valid[teal > 0] = 0
    crop = work[y0:y1, x0:x1]
    fcrop = valid[y0:y1, x0:x1].copy()
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    fm = cv2.resize(fcrop, (size, size), interpolation=cv2.INTER_NEAREST)
    return Canon(
        image=img,
        fov=fm,
        bbox=(x0, y0, x1, y1),
        teal_frac=float(teal.mean()),
        fov_frac=float(fov.mean()),
        orig_hw=(h, w),
    )


def crop_resize_mask(mask: np.ndarray, bbox: tuple[int, int, int, int], size: int = 256) -> np.ndarray:
    """Apply the FOV crop used by `canonicalise` to a ground-truth mask (nearest-neighbour)."""
    x0, y0, x1, y1 = bbox
    return cv2.resize(mask[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
