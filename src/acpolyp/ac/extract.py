"""Run the active-contour pipeline over the cleaned dataset (Phase X step 38).

Outputs (data/processed/):
  ac_masks.npy              (N,W,W) uint8  AC region masks (W = ac.work_size)
  ac_masks_ellipse.npy      control: central ellipse mask (image-independent)
  ac_masks_shuffle.npy      control: AC mask of a *different* image (same mask statistics, wrong image)
  ac_features.parquet       136 hand-crafted features for AC masks
  ac_features_ellipse.parquet / ac_features_shuffle.parquet  same features under control masks
  ac_diagnostics.parquet    per-image area fraction, fallback flag
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import load_config
from ..paths import INTERIM, PROCESSED
from .features import all_features, family_of  # noqa: F401
from .segment import central_ellipse, segment


def _one(rgb256: np.ndarray, valid256: np.ndarray, params: dict):
    w = params["work_size"]
    rgb = cv2.resize(rgb256, (w, w), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid256, (w, w), interpolation=cv2.INTER_NEAREST)
    r = segment(rgb, valid, params)
    ell = central_ellipse(valid)
    return rgb, valid, r.mask, ell, r.fallback, r.area_frac


def _features(rgb, valid, mask):
    return all_features(rgb, valid, mask)


def main(cfg_path: str | None = "ac_tuned.yaml", n_jobs: int = 10) -> None:
    cfg = load_config(cfg_path)
    p = cfg["ac"]
    kept = np.load(PROCESSED / "kept_idx.npy")
    img = np.load(INTERIM / "images_canon.npy", mmap_mode="r")
    fov = np.load(INTERIM / "fov.npy", mmap_mode="r")
    seg = Parallel(n_jobs=n_jobs, batch_size=32, verbose=2)(
        delayed(_one)(np.asarray(img[k]), np.asarray(fov[k]), p) for k in kept
    )
    rgbs = np.stack([s[0] for s in seg])
    valids = np.stack([s[1] for s in seg])
    masks = np.stack([s[2] for s in seg]).astype(np.uint8)
    ells = np.stack([s[3] for s in seg]).astype(np.uint8)
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(masks))
    # a derangement-ish shuffle: reshuffle any fixed points
    while (perm == np.arange(len(perm))).any():
        fix = np.where(perm == np.arange(len(perm)))[0]
        perm[fix] = np.roll(perm[fix], 1) if len(fix) > 1 else rng.integers(len(perm))
        if len(fix) == 1:
            break
    shuf = masks[perm]
    np.save(PROCESSED / "ac_masks.npy", masks)
    np.save(PROCESSED / "ac_masks_ellipse.npy", ells)
    np.save(PROCESSED / "ac_masks_shuffle.npy", shuf)
    np.save(PROCESSED / "ac_shuffle_perm.npy", perm)

    for name, ms in (("", masks), ("_ellipse", ells), ("_shuffle", shuf)):
        feats = Parallel(n_jobs=n_jobs, batch_size=32)(
            delayed(_features)(rgbs[i], valids[i], ms[i]) for i in range(len(ms))
        )
        df = pd.DataFrame(feats).astype("float32")
        assert not df.isna().any().any(), f"NaN features ({name})"
        df.to_parquet(PROCESSED / f"ac_features{name}.parquet", index=False)
        print(name or "ac", df.shape)
    pd.DataFrame({"fallback": [s[4] for s in seg], "area_frac": [s[5] for s in seg]}).to_parquet(
        PROCESSED / "ac_diagnostics.parquet", index=False
    )
    print(
        "fallback rate",
        np.mean([s[4] for s in seg]),
        "mean area frac",
        np.mean([s[5] for s in seg]),
    )


if __name__ == "__main__":
    main()
