"""Two-stage morphological active contour for lesion-candidate delineation.

Stage 1  morphological Chan-Vese (ACWE; Chan & Vese 2001, morphological form of Marquez-Neila
         et al. 2014) partitions the FOV into two piecewise-constant regions on a salience channel.
Stage 2  morphological geodesic active contour (MorphGAC) snaps the region boundary to image edges.

No labels are used anywhere: the contour is a purely unsupervised spatial prior. All routines work
on a (work_size x work_size) image and return masks at that resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_chan_vese,
    morphological_geodesic_active_contour,
)


@dataclass
class ACResult:
    mask: np.ndarray  # (S, S) uint8 {0,1}  final lesion-candidate region
    cv_mask: np.ndarray  # (S, S) uint8        stage-1 region (before geodesic refinement)
    salience: np.ndarray  # (S, S) float32 in [0,1]
    fallback: bool  # True if the contour degenerated and a central ellipse was used
    area_frac: float  # mask area / FOV area


def _masked_blur(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: Gaussian blur that ignores invalid pixels."""
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(x * v, (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def salience_channel(
    rgb: np.ndarray, fov: np.ndarray, kind: str = "lab_pc1", dog_frac: float = 0.0
) -> np.ndarray:
    """Scalar image in [0,1] on which the contour evolves; invalid pixels are set to the valid mean.

    `dog_frac` > 0 subtracts a wide Gaussian (sigma = dog_frac * size) -> centre-surround contrast, which
    emphasises locally distinct blobs (protruding lesions) over global illumination gradients.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.2)
    inside = fov > 0
    if kind == "lab_pc1":
        px = lab[inside]
        mu = px.mean(0)
        cov = np.cov((px - mu).T)
        w, v = np.linalg.eigh(cov)
        s = (lab - mu) @ v[:, -1]
    elif kind == "a":
        s = lab[..., 1]
    elif kind == "L":
        s = lab[..., 0]
    elif kind == "sat":
        s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    else:
        raise ValueError(kind)
    s = np.where(inside, s, 0.0).astype(np.float32)
    if dog_frac > 0:
        s = s - _masked_blur(s, inside, dog_frac * fov.shape[0])
    lo, hi = np.percentile(s[inside], [1, 99])
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    s[~inside] = s[inside].mean()
    return s


def _disk(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2).astype(np.uint8)


def _largest_component_filled(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    out = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    return ndi.binary_fill_holes(out).astype(np.uint8)


def central_ellipse(fov: np.ndarray, frac: float = 0.35) -> np.ndarray:
    ys, xs = np.where(fov > 0)
    c = (ys.mean(), xs.mean())
    r = frac * 0.5 * (np.ptp(ys) + np.ptp(xs)) / 2 * 2
    return (_disk(fov.shape, c, r) & (fov > 0)).astype(np.uint8)


def segment(rgb: np.ndarray, fov: np.ndarray, p: dict) -> ACResult:
    """rgb, fov are already at work resolution."""
    s = salience_channel(rgb, fov, p.get("channel", "lab_pc1"), float(p.get("dog_frac", 0.0)))
    inside = fov > 0
    ys, xs = np.where(inside)
    center = (ys.mean(), xs.mean())
    span = 0.5 * (np.ptp(ys) + np.ptp(xs))
    init = _disk(fov.shape, center, float(p.get("init_frac", 0.30)) * span) & inside

    u = morphological_chan_vese(
        s,
        num_iter=int(p["cv_iters"]),
        init_level_set=init,
        smoothing=int(p["cv_smoothing"]),
        lambda1=float(p["cv_lambda_in"]),
        lambda2=float(p["cv_lambda_out"]),
    ).astype(np.uint8)

    # Polarity is arbitrary in a two-phase model: the lesion candidate is the region that touches
    # the FOV boundary least (mucosal background wraps around the image border; a lesion does not).
    ring = inside & ~ndi.binary_erosion(inside, iterations=max(2, int(0.06 * fov.shape[0])))
    touch = lambda m: (m & ring).sum() / max(m.sum(), 1)
    cand_a, cand_b = (u & inside).astype(np.uint8), ((1 - u) & inside).astype(np.uint8)
    fg = cand_a if touch(cand_a) <= touch(cand_b) else cand_b
    cv_mask = _largest_component_filled(fg)

    g = inverse_gaussian_gradient(s, alpha=float(p["gac_alpha"]), sigma=float(p["gac_sigma"]))
    refined = morphological_geodesic_active_contour(
        g,
        num_iter=int(p["gac_iters"]),
        init_level_set=cv_mask,
        smoothing=int(p["gac_smoothing"]),
        threshold="auto",
        balloon=float(p["gac_balloon"]),
    ).astype(np.uint8)
    mask = _largest_component_filled(
        refined & ndi.binary_erosion(inside, iterations=1).astype(np.uint8)
    )

    area = mask.sum() / max(inside.sum(), 1)
    fallback = not (0.02 <= area <= 0.85)
    if fallback:
        mask = central_ellipse(fov)
        area = mask.sum() / max(inside.sum(), 1)
    return ACResult(
        mask=mask, cv_mask=cv_mask, salience=s, fallback=fallback, area_frac=float(area)
    )
