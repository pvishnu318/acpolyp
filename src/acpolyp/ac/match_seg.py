"""Match Kvasir-SEG frames to Kvasir-v2 'polyps' frames so AC tuning never touches test images."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.preprocess import canonicalise
from ..models.embed import cached, extract
from ..paths import PROCESSED
from .validate import SEG_DIR

EMB = "vit_small_patch14_dinov2.lvd142m"


def _canon(name: str) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.imread(str(SEG_DIR / "images" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    return canonicalise(rgb, 256).image


def main() -> None:
    names = sorted(p.stem for p in (SEG_DIR / "images").glob("*.jpg"))
    imgs = np.stack(Parallel(n_jobs=10)(delayed(_canon)(n) for n in names))
    e_seg = extract(EMB, imgs, 224)
    e_seg /= np.linalg.norm(e_seg, axis=1, keepdims=True)
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    kept = np.load(PROCESSED / "kept_idx.npy")
    e_all = cached(EMB, "canon", 224)[kept]
    e_all = e_all / np.linalg.norm(e_all, axis=1, keepdims=True)
    pol = np.where(meta["label_name"].to_numpy() == "polyps")[0]
    sim = e_seg @ e_all[pol].T
    best, arg = sim.max(1), sim.argmax(1)
    # mutual-nearest-neighbour check
    back = (e_all[pol] @ e_seg.T).argmax(1)
    mutual = back[arg] == np.arange(len(names))
    out = pd.DataFrame(
        {
            "seg_name": names,
            "image_id": meta["image_id"].to_numpy()[pol[arg]],
            "clean_row": pol[arg],
            "cosine": best,
            "mutual": mutual,
            "split": meta["split"].to_numpy()[pol[arg]],
        }
    )
    out["matched"] = (out["cosine"] >= 0.93) & out["mutual"]
    out.to_parquet(PROCESSED / "seg_match.parquet", index=False)
    print("cosine quantiles", np.percentile(best, [0, 5, 25, 50, 75, 100]).round(3))
    print("matched:", int(out.matched.sum()), "/", len(out))
    print(out[out.matched].split.value_counts())


if __name__ == "__main__":
    main()
