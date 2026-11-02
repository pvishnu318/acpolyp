"""Perturbation robustness of the plain vs contour-guided model (Phase XVII step 76).

Perturbations act on the canonical test frame; contour and features are recomputed from the perturbed pixels so
the guided model is evaluated end-to-end. 320 stratified test frames (40 per class) keep the cost manageable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import build_bundle  # noqa: E402

from acpolyp.evaluation import figstyle as fs  # noqa: E402
from acpolyp.inference import Predictor  # noqa: E402
from acpolyp.paths import ARTIFACTS, INTERIM, PROCESSED, RESULTS  # noqa: E402
from acpolyp.training.calibration import temperature_probs  # noqa: E402,F401
from acpolyp.training.metrics import softmax  # noqa: E402

LEVELS = {
    "Gaussian noise (σ)": ("noise", [0.0, 0.02, 0.05, 0.10]),
    "Gaussian blur (σ px)": ("blur", [0.0, 1.0, 2.0, 3.0]),
    "Brightness (×)": ("bright", [1.0, 0.7, 0.5, 1.5]),
    "JPEG quality": ("jpeg", [100, 50, 20, 10]),
}


def perturb(
    img: np.ndarray, valid: np.ndarray, kind: str, lvl: float, rng: np.random.Generator
) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    if kind == "noise" and lvl > 0:
        x = x + rng.normal(0, lvl, x.shape).astype(np.float32)
    elif kind == "blur" and lvl > 0:
        x = cv2.GaussianBlur(x, (0, 0), lvl)
    elif kind == "bright":
        x = x * lvl
    elif kind == "jpeg" and lvl < 100:
        ok, enc = cv2.imencode(
            ".jpg",
            cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, int(lvl)],
        )
        x = (
            cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
            / 255.0
        )
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    out[valid == 0] = 0
    return out


@torch.inference_mode()
def logits_for(p: Predictor, batch: list[dict]) -> np.ndarray:
    out = []
    for b in batch:
        out.append(p._logits(b).numpy()[0])
    return np.stack(out)


def main() -> None:
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    kept = np.load(PROCESSED / "kept_idx.npy")
    test = meta[meta.split == "test"]
    rng = np.random.default_rng(0)
    pick = np.concatenate(
        [rng.choice(test.index[test.label == k], 40, replace=False) for k in range(8)]
    )
    imgs = np.load(INTERIM / "images_canon.npy", mmap_mode="r")
    fovs = np.load(INTERIM / "fov.npy", mmap_mode="r")
    y = meta.label.to_numpy()[pick]

    preds = {}
    for tag in ("ab_none", "ab_acg"):
        build_bundle.main(version=f"eval-{tag}", tag=tag)
        preds[tag] = Predictor(ARTIFACTS / f"eval-{tag}")
    guided = preds["ab_acg"]
    res: dict = {}
    for title, (kind, levels) in LEVELS.items():
        res[title] = {"levels": levels, "ab_none": [], "ab_acg": []}
        for lvl in levels:
            prng = np.random.default_rng(1)
            bat_g, bat_p = [], []
            for r in pick:
                canon, fov = np.asarray(imgs[kept[r]]), np.asarray(fovs[kept[r]])
                x = perturb(canon, fov, kind, lvl, prng)
                bg = guided.prepare_canonical(x, fov)
                bat_g.append(bg)
                bat_p.append({**bg, "feats": None})
            for tag, p, bat in (("ab_acg", guided, bat_g), ("ab_none", preds["ab_none"], bat_p)):
                T = p.cal["temperature"]
                pr = softmax(logits_for(p, bat) / T)
                res[title][tag].append(float((pr.argmax(1) == y).mean()))
            print(
                kind,
                lvl,
                {t: round(res[title][t][-1], 3) for t in ("ab_none", "ab_acg")},
                flush=True,
            )
    (RESULTS / "robustness.json").write_text(
        json.dumps({"n": int(len(pick)), "curves": res}, indent=2)
    )
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    for a, (title, v) in zip(ax, res.items()):
        xs = range(len(v["levels"]))
        for i, (t, nm) in enumerate((("ab_none", "plain ConvNeXt-T"), ("ab_acg", "ACG-ConvNeXt"))):
            a.plot(
                xs, v[t], marker=fs.MARKERS[i], color=fs.SERIES[i], ls=fs.LINESTYLES[i], label=nm
            )
        a.set_xticks(list(xs), [str(lv) for lv in v["levels"]], fontsize=7)
        a.set_title(title, fontsize=8)
    ax[0].set_ylabel("accuracy (n=320)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fs.save(fig, "robustness")


if __name__ == "__main__":
    main()
