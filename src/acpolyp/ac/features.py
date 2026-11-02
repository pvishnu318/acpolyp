"""Hand-crafted feature families computed from the active contour (Phase X steps 36-38).

Families (names are used verbatim by the ablation study):
  global   colour + texture over the whole valid FOV                       (no active contour)
  shape    contour geometry: area, compactness, solidity, Hu moments,
           curvature statistics, border contact, centroid offset
  boundary edge strength along the contour and inside/outside-ring contrast
  region   colour + texture of the contour interior vs exterior and their differences
Every function is deterministic and takes a work-resolution image (S,S,3) uint8, valid-mask (S,S)
and region mask (S,S).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern
from skimage.measure import moments_hu, regionprops

LBP_P, LBP_R = 8, 1
GLCM_LEVELS = 16


def _stats(x: np.ndarray) -> list[float]:
    return [float(x.mean()), float(x.std())] if x.size else [0.0, 0.0]


def colour_features(rgb: np.ndarray, m: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean/std of Lab and HSV channels under mask m, plus a coarse hue histogram."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    mm = m > 0
    out: dict[str, float] = {}
    for name, img in (
        ("L", lab[..., 0]),
        ("a", lab[..., 1]),
        ("b", lab[..., 2]),
        ("S", hsv[..., 1]),
        ("V", hsv[..., 2]),
    ):
        mu, sd = _stats(img[mm])
        out[f"{prefix}_{name}_mean"], out[f"{prefix}_{name}_std"] = mu, sd
    hue = hsv[..., 0][mm]
    h, _ = np.histogram(hue, bins=6, range=(0, 180))
    for i, v in enumerate(h / max(h.sum(), 1)):
        out[f"{prefix}_hue{i}"] = float(v)
    return out


def lbp_features(gray: np.ndarray, m: np.ndarray, prefix: str) -> dict[str, float]:
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    mm = ndi.binary_erosion(m > 0, iterations=1)
    h, _ = np.histogram(lbp[mm], bins=LBP_P + 2, range=(0, LBP_P + 2))
    h = h / max(h.sum(), 1)
    return {f"{prefix}_lbp{i}": float(v) for i, v in enumerate(h)}


def glcm_features(gray: np.ndarray, m: np.ndarray, prefix: str) -> dict[str, float]:
    """Masked grey-level co-occurrence statistics (4 directions, distance 1), computed by hand because
    skimage's graycomatrix cannot restrict pairs to a mask."""
    q = np.clip((gray.astype(np.int32) * GLCM_LEVELS) // 256, 0, GLCM_LEVELS - 1)
    mm = m > 0
    P = np.zeros((GLCM_LEVELS, GLCM_LEVELS), np.float64)
    H, W = q.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        a, b = q[y0:y1, x0:x1], q[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        both = mm[y0:y1, x0:x1] & mm[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        np.add.at(P, (a[both], b[both]), 1.0)
    P = P + P.T
    s = P.sum()
    if s == 0:
        return {
            f"{prefix}_glcm_{k}": 0.0 for k in ("contrast", "homog", "energy", "corr", "entropy")
        }
    P /= s
    i, j = np.indices(P.shape)
    mu_i, mu_j = (i * P).sum(), (j * P).sum()
    sd_i, sd_j = np.sqrt(((i - mu_i) ** 2 * P).sum()), np.sqrt(((j - mu_j) ** 2 * P).sum())
    corr = ((i - mu_i) * (j - mu_j) * P).sum() / max(sd_i * sd_j, 1e-8)
    return {
        f"{prefix}_glcm_contrast": float(((i - j) ** 2 * P).sum()),
        f"{prefix}_glcm_homog": float((P / (1.0 + np.abs(i - j))).sum()),
        f"{prefix}_glcm_energy": float((P**2).sum()),
        f"{prefix}_glcm_corr": float(corr),
        f"{prefix}_glcm_entropy": float(-(P[P > 0] * np.log2(P[P > 0])).sum()),
    }


def global_features(rgb: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = colour_features(rgb, valid, "g")
    f.update(lbp_features(gray, valid, "g"))
    f.update(glcm_features(gray, valid, "g"))
    return f


def _curvature_stats(contour: np.ndarray) -> tuple[float, float, float, float]:
    """Mean |k|, std k, max |k| and fraction of concave points of a smoothed closed contour."""
    pts = contour[:, 0, :].astype(np.float64)
    if len(pts) < 12:
        return 0.0, 0.0, 0.0, 0.0
    k = np.ones(5) / 5.0
    x = np.convolve(np.r_[pts[-4:, 0], pts[:, 0], pts[:4, 0]], k, "valid")[2:-2]
    y = np.convolve(np.r_[pts[-4:, 1], pts[:, 1], pts[:4, 1]], k, "valid")[2:-2]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curv = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-8)
    sign = np.sign(np.mean(curv)) or 1.0
    curv = curv * sign
    return (
        float(np.abs(curv).mean()),
        float(curv.std()),
        float(np.abs(curv).max()),
        float((curv < 0).mean()),
    )


def shape_features(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    S = mask.shape[0]
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    f: dict[str, float] = {}
    fov_area = max(float((valid > 0).sum()), 1.0)
    f["s_area_frac"] = area / fov_area
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or area == 0:
        keys = [
            "s_perim",
            "s_circularity",
            "s_solidity",
            "s_ecc",
            "s_axis_ratio",
            "s_extent",
            "s_cx",
            "s_cy",
            "s_dist_center",
            "s_border_contact",
            "s_curv_mean",
            "s_curv_std",
            "s_curv_max",
            "s_concave_frac",
            "s_orient_sin",
            "s_orient_cos",
            "s_eq_diam",
        ] + [f"s_hu{i}" for i in range(7)]
        return {**f, **{k: 0.0 for k in keys}}
    c = max(cnts, key=cv2.contourArea)
    perim = float(cv2.arcLength(c, True))
    f["s_perim"] = perim / S
    f["s_circularity"] = float(4 * np.pi * area / max(perim**2, 1e-6))
    hull = cv2.convexHull(c)
    f["s_solidity"] = area / max(float(cv2.contourArea(hull)), 1.0)
    rp = regionprops(m)[0]
    f["s_ecc"] = float(rp.eccentricity)
    f["s_axis_ratio"] = float(rp.axis_minor_length / max(rp.axis_major_length, 1e-6))
    f["s_extent"] = float(rp.extent)
    f["s_eq_diam"] = float(rp.equivalent_diameter_area) / S
    cy, cx = rp.centroid
    vy, vx = np.where(valid > 0)
    f["s_cx"], f["s_cy"] = (cx - vx.mean()) / S, (cy - vy.mean()) / S
    f["s_dist_center"] = float(np.hypot(f["s_cx"], f["s_cy"]))
    ring = (valid > 0) & ~ndi.binary_erosion(valid > 0, iterations=max(2, int(0.06 * S)))
    f["s_border_contact"] = float((m.astype(bool) & ring).sum() / max(area, 1.0))
    f["s_curv_mean"], f["s_curv_std"], f["s_curv_max"], f["s_concave_frac"] = _curvature_stats(c)
    f["s_orient_sin"], f["s_orient_cos"] = (
        float(np.sin(2 * rp.orientation)),
        float(np.cos(2 * rp.orientation)),
    )
    hu = moments_hu(m.astype(np.float64))
    for i, v in enumerate(hu):
        f[f"s_hu{i}"] = float(-np.sign(v) * np.log10(abs(v) + 1e-30))
    return f


def boundary_features(rgb: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    m = mask > 0
    edge = m ^ ndi.binary_erosion(m, iterations=1)
    ring_in = m & ~ndi.binary_erosion(m, iterations=4)
    ring_out = ndi.binary_dilation(m, iterations=4) & ~m & (valid > 0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    f: dict[str, float] = {}
    g_on = gm[edge & (valid > 0)]
    f["b_grad_mean"], f["b_grad_std"] = _stats(g_on)
    f["b_grad_p90"] = float(np.percentile(g_on, 90)) if g_on.size else 0.0
    glob_thr = np.percentile(gm[valid > 0], 75)
    f["b_strong_edge_frac"] = float((g_on > glob_thr).mean()) if g_on.size else 0.0
    if ring_in.any() and ring_out.any():
        d = lab[ring_in].mean(0) - lab[ring_out].mean(0)
        f.update(
            {
                "b_dL": float(d[0]),
                "b_da": float(d[1]),
                "b_db": float(d[2]),
                "b_dE": float(np.linalg.norm(d)),
            }
        )
    else:
        f.update({"b_dL": 0.0, "b_da": 0.0, "b_db": 0.0, "b_dE": 0.0})
    return f


def region_features(rgb: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = (mask > 0) & (valid > 0)
    outside = (mask == 0) & (valid > 0)
    f = {}
    fi, fo = colour_features(rgb, inside, "in"), colour_features(rgb, outside, "out")
    f.update(fi)
    f.update(fo)
    for ch in ("L", "a", "b", "S", "V"):
        f[f"d_{ch}"] = fi[f"in_{ch}_mean"] - fo[f"out_{ch}_mean"]
        f[f"r_{ch}_std"] = fi[f"in_{ch}_std"] / max(fo[f"out_{ch}_std"], 1e-3)
    gi, go = glcm_features(gray, inside, "in"), glcm_features(gray, outside, "out")
    f.update(gi)
    f.update(go)
    f.update(lbp_features(gray, inside, "in"))
    f.update(lbp_features(gray, outside, "out"))
    return f


def all_features(rgb: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    f = global_features(rgb, valid)
    f.update(shape_features(mask, valid))
    f.update(boundary_features(rgb, mask, valid))
    f.update(region_features(rgb, mask, valid))
    return f


def family_of(name: str) -> str:
    if name.startswith("g_"):
        return "global"
    if name.startswith("s_"):
        return "shape"
    if name.startswith("b_"):
        return "boundary"
    return "region"
