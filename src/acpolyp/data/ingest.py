"""Ingestion into a canonical schema + image caches (Phase V step 20, Phase VII steps 24-26).

Produces under data/interim/:
  images_canon.npy  (N,S,S,3) uint8  FOV-cropped, artefact-masked frames (used for all models)
  images_naive.npy  (N,S,S,3) uint8  whole frame squashed to SxS (shortcut-audit baseline only)
  fov.npy           (N,S,S)   uint8  field-of-view masks
  metadata.parquet               canonical per-image table
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from ..paths import INTERIM, KVASIR_RAW, ensure
from .contracts import CLASS_TO_IDX, CLASSES, validate_metadata
from .preprocess import canonicalise, naive_resize

# EXIF tags that could carry personal data: ImageDescription, Make, Model, Software, DateTime,
# Artist, Copyright, plus the GPS IFD pointer.
PII_TAGS = {270, 271, 272, 305, 306, 315, 33432, 34853}


def _process(path: str, size: int) -> dict | None:
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(path) as im:
            im.load()
            exif = im.getexif()
            tags = set(exif.keys())
            comment = im.info.get("comment") or b""
            pil = im.convert("RGB")
    except Exception:
        return {"image_id": f"{Path(path).parent.name}/{Path(path).stem}", "sha256": sha, "decode_ok": False}
    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]
    c = canonicalise(rgb, size)
    gray = cv2.cvtColor(c.image, cv2.COLOR_RGB2GRAY)
    inside = c.fov > 0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "image_id": f"{Path(path).parent.name}/{Path(path).stem}",  # one filename occurs in two classes
        "label_name": Path(path).parent.name,
        "path": str(Path(path).relative_to(KVASIR_RAW)),
        "bytes": len(raw),
        "sha256": sha,
        "decode_ok": True,
        "width": w,
        "height": h,
        "n_exif_tags": len(tags),
        "has_pii_exif": bool((tags & PII_TAGS) or comment),
        "teal_frac": c.teal_frac,
        "fov_frac": c.fov_frac,
        "bbox_x0": c.bbox[0],
        "bbox_y0": c.bbox[1],
        "bbox_x1": c.bbox[2],
        "bbox_y1": c.bbox[3],
        "mean_luma": float(gray[inside].mean()) if inside.any() else 0.0,
        "lap_var": float(lap[inside].var()) if inside.any() else 0.0,
        "phash": str(imagehash.phash(Image.fromarray(c.image), hash_size=8)),
        "_canon": c.image,
        "_naive": naive_resize(rgb, size),
        "_fov": c.fov,
    }


def main(size: int = 256, n_jobs: int = 10) -> None:
    ensure(INTERIM)
    root = KVASIR_RAW / "kvasir-dataset-v2"
    files = sorted(str(p) for c in CLASSES for p in (root / c).glob("*.jpg"))
    assert len(files) == 8000, f"expected 8000 files, found {len(files)}"
    rows = Parallel(n_jobs=n_jobs, batch_size=16, verbose=5)(delayed(_process)(f, size) for f in files)

    bad = [r for r in rows if not r["decode_ok"]]
    good = [r for r in rows if r["decode_ok"]]
    print(f"decoded {len(good)} / {len(rows)}; corrupted: {len(bad)}")

    canon = np.stack([r.pop("_canon") for r in good])
    naive = np.stack([r.pop("_naive") for r in good])
    fov = np.stack([r.pop("_fov") for r in good])
    df = pd.DataFrame(good)
    df["label"] = df["label_name"].map(CLASS_TO_IDX).astype(int)
    df.insert(0, "idx", np.arange(len(df)))
    validate_metadata(df)

    np.save(INTERIM / "images_canon.npy", canon)
    np.save(INTERIM / "images_naive.npy", naive)
    np.save(INTERIM / "fov.npy", fov)
    df.to_parquet(INTERIM / "metadata.parquet", index=False)
    (INTERIM / "ingest_report.json").write_text(
        json.dumps({"n_total": len(rows), "n_corrupted": len(bad), "corrupted": [b["image_id"] for b in bad]}, indent=2),
        encoding="utf-8",
    )
    print(df.groupby("label_name")[["width", "height", "teal_frac", "fov_frac", "has_pii_exif"]].mean().round(3))


if __name__ == "__main__":
    main()
