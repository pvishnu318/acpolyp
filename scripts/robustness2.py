"""Multi-seed, controlled robustness study (Phase XVII step 76).

Which component, if any, makes a model robust to image degradation? Plain ConvNeXt-T, global-feature, full guided,
shuffled-contour and ellipse variants (3 seeds each) are evaluated on 480 stratified test frames under JPEG compression and
Gaussian blur. Contour + features are recomputed from the degraded pixels; shuffled controls use another image's contour.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import build_bundle  # noqa: E402
from robustness import perturb  # noqa: E402

from acpolyp.ac.features import all_features  # noqa: E402
from acpolyp.ac.segment import central_ellipse, segment  # noqa: E402
from acpolyp.data.augment import eval_transform  # noqa: E402
from acpolyp.evaluation import figstyle as fs  # noqa: E402
from acpolyp.inference import Predictor  # noqa: E402
from acpolyp.paths import ARTIFACTS, INTERIM, PROCESSED, RESULTS  # noqa: E402

VARIANTS = {
    "ab_none": "plain",
    "ab_gfeat": "+ global features",
    "ab_acg": "ACG (full)",
    "ab_acg_shuffle": "shuffled contours",
    "ab_acg_ellipse": "central ellipse",
}
LEVELS = [
    ("clean", "jpeg", 100),
    ("jpeg", "jpeg", 50),
    ("jpeg", "jpeg", 30),
    ("jpeg", "jpeg", 10),
    ("blur", "blur", 1.5),
    ("blur", "blur", 3.0),
]
N_PER_CLASS = 60


def tensors(p: Predictor, small, valid_w, mask128, allf, canon, fov, mask_src_feats) -> dict:
    feats = None
    if p.scaler:
        raw = np.array([mask_src_feats[n] for n in p.scaler["names"]], np.float32)
        lo, hi, mu, sd = (np.array(p.scaler[k], np.float32) for k in ("lo", "hi", "mu", "sd"))
        feats = torch.from_numpy(((np.clip(raw, lo, hi) - mu) / sd).astype(np.float32))[None]
    S = canon.shape[0]
    m = torch.from_numpy(cv2.resize(mask128, (S, S), interpolation=cv2.INTER_NEAREST))[
        None, None
    ].to(torch.uint8)
    x, mm, vv = eval_transform(
        torch.from_numpy(canon).permute(2, 0, 1)[None],
        m,
        torch.from_numpy(fov)[None, None].to(torch.uint8),
        p.size,
    )
    return {"x": x, "mask": mm, "valid": vv, "feats": feats}


def main() -> None:
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    kept = np.load(PROCESSED / "kept_idx.npy")
    test = meta[meta.split == "test"]
    rng = np.random.default_rng(0)
    pick = np.concatenate(
        [rng.choice(test.index[test.label == k], N_PER_CLASS, replace=False) for k in range(8)]
    )
    imgs, fovs = (
        np.load(INTERIM / "images_canon.npy", mmap_mode="r"),
        np.load(INTERIM / "fov.npy", mmap_mode="r"),
    )
    y = meta.label.to_numpy()[pick]

    preds = {}
    for tag in VARIANTS:
        for s in (0, 1, 2):
            v = f"rob-{tag}-s{s}"
            build_bundle.main(version=v, tag=tag, seed=s)
            preds[(tag, s)] = Predictor(ARTIFACTS / v)
    ref = preds[("ab_acg", 0)]
    ac_cfg = ref.cfg["ac"]
    w = ac_cfg["work_size"]
    res = {tag: {f"{k}{lvl}": [] for k, _, lvl in LEVELS} for tag in VARIANTS}
    for name, kind, lvl in LEVELS:
        key = f"{name}{lvl}"
        batches = {t: [] for t in VARIANTS}
        pending = []
        for r in pick:
            canon0, fov = np.asarray(imgs[kept[r]]), np.asarray(fovs[kept[r]])
            canon = perturb(canon0, fov, kind, lvl, np.random.default_rng(1))
            small = cv2.resize(canon, (w, w), interpolation=cv2.INTER_AREA)
            valid_w = cv2.resize(fov, (w, w), interpolation=cv2.INTER_NEAREST)
            seg = segment(small, valid_w, ac_cfg)
            ell = central_ellipse(valid_w)
            pending.append(
                (
                    canon,
                    fov,
                    small,
                    valid_w,
                    seg.mask,
                    ell,
                    all_features(small, valid_w, seg.mask),
                    all_features(small, valid_w, ell),
                )
            )
        perm = np.random.default_rng(2).permutation(len(pending))
        for tag in VARIANTS:
            for i, (canon, fov, _small, _valid_w, mask, ell, f_ac, f_ell) in enumerate(pending):
                if tag == "ab_acg_shuffle":
                    j = perm[i]
                    mask, f = pending[j][4], pending[j][6]
                elif tag == "ab_acg_ellipse":
                    mask, f = ell, f_ell
                else:
                    f = f_ac
                batches[tag].append((canon, fov, mask, f))
        for tag in VARIANTS:
            accs = []
            for s in (0, 1, 2):
                p = preds[(tag, s)]
                lg = np.stack(
                    [
                        p._logits(tensors(p, None, None, m, None, c, fv, f)).numpy()[0]
                        for c, fv, m, f in batches[tag]
                    ]
                )
                accs.append(float((lg.argmax(1) == y).mean()))
            res[tag][key] = accs
        print(key, {t: round(float(np.mean(res[t][key])), 3) for t in VARIANTS}, flush=True)
    out = {
        "n": int(len(pick)),
        "variants": VARIANTS,
        "levels": [f"{n}{lv}" for n, _, lv in LEVELS],
        "acc": res,
    }
    (RESULTS / "robustness_multiseed.json").write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    groups = [
        ("JPEG quality", ["clean100", "jpeg50", "jpeg30", "jpeg10"], ["100", "50", "30", "10"]),
        ("Gaussian blur (sigma, px)", ["clean100", "blur1.5", "blur3.0"], ["0", "1.5", "3"]),
    ]
    for a, (ttl, keys, labels) in zip(ax, groups):
        for i, tag in enumerate(VARIANTS):
            m = [np.mean(res[tag][k]) for k in keys]
            sd = [np.std(res[tag][k]) for k in keys]
            a.errorbar(
                range(len(keys)),
                m,
                yerr=sd,
                marker=fs.MARKERS[i % 4],
                color=(fs.SERIES + ["#898781"])[i % 5],
                ls=fs.LINESTYLES[i % 4],
                capsize=2,
                label=VARIANTS[tag],
            )
        a.set_xticks(range(len(keys)), labels)
        a.set_title(ttl)
    ax[0].set_ylabel(f"accuracy (n={len(pick)}, mean ± SD over 3 seeds)")
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fs.save(fig, "robustness_multiseed")
    for tag in VARIANTS:
        for s in (0, 1, 2):
            shutil.rmtree(ARTIFACTS / f"rob-{tag}-s{s}", ignore_errors=True)


if __name__ == "__main__":
    main()
