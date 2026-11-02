"""Active-contour quality analysis on Kvasir-SEG test-split matches + pipeline illustration (Phase XVII/XVIII)."""

from __future__ import annotations

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from acpolyp.ac.segment import central_ellipse, salience_channel, segment
from acpolyp.ac.validate import dice_iou, load_pair
from acpolyp.config import load_config
from acpolyp.data.preprocess import canonicalise
from acpolyp.evaluation import figstyle as fs
from acpolyp.paths import KVASIR_RAW, PROCESSED, RESULTS, ensure


def boot(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ensure(RESULTS)
    cfg_t, cfg_d = load_config("ac_tuned.yaml")["ac"], load_config()["ac"]
    m = pd.read_parquet(PROCESSED / "seg_match.parquet")
    names = m[m.matched & (m.split == "test")].seg_name.tolist()
    rows = []
    for n in names:
        rgb, fov, gt, _ = load_pair(n, cfg_t["work_size"])
        r = segment(rgb, fov, cfg_t)
        rd = segment(rgb, fov, cfg_d)
        rows.append(
            {
                "name": n,
                "tuned": dice_iou(r.mask, gt)[0],
                "tuned_iou": dice_iou(r.mask, gt)[1],
                "ellipse": dice_iou(central_ellipse(fov), gt)[0],
                "whole_fov": dice_iou(fov, gt)[0],
                "default": dice_iou(rd.mask, gt)[0],
                "gt_area": gt.sum() / max(fov.sum(), 1),
                "pred_area": r.area_frac,
            }
        )
    d = pd.DataFrame(rows)
    out = {"n_test_images": len(d)}
    for k in ("tuned", "tuned_iou", "ellipse", "whole_fov", "default"):
        out[k] = dict(zip(("mean", "ci_low", "ci_high"), boot(d[k].to_numpy())))
    diff = (d.tuned - d.ellipse).to_numpy()
    out["tuned_minus_ellipse"] = dict(zip(("mean", "ci_low", "ci_high"), boot(diff)))
    out["wilcoxon_p_tuned_vs_ellipse"] = float(sps.wilcoxon(d.tuned, d.ellipse).pvalue)
    out["frac_images_tuned_better_than_ellipse"] = float((diff > 0).mean())
    out["dice_by_gt_size"] = {
        "small (<10% of FOV)": float(d[d.gt_area < 0.10].tuned.mean()),
        "medium (10-25%)": float(d[(d.gt_area >= 0.10) & (d.gt_area < 0.25)].tuned.mean()),
        "large (>=25%)": float(d[d.gt_area >= 0.25].tuned.mean()),
    }
    out["median_pred_over_gt_area"] = float((d.pred_area / d.gt_area.clip(lower=1e-3)).median())
    (RESULTS / "ac_quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    # ---- qualitative gallery: best / median / worst by Dice ---------------------------------
    order = d.sort_values("tuned").reset_index(drop=True)
    picks = (
        list(order.index[-4:][::-1])
        + list(order.index[len(order) // 2 - 2 : len(order) // 2 + 2])
        + list(order.index[:4])
    )
    titles = ["best"] * 4 + ["median"] * 4 + ["worst"] * 4
    fig, ax = plt.subplots(3, 4, figsize=(9, 6.8))
    for a, i, t in zip(ax.ravel(), picks, titles):
        rgb, fov, gt, _ = load_pair(order.name[i], cfg_t["work_size"])
        r = segment(rgb, fov, cfg_t)
        a.imshow(rgb)
        a.contour(gt, [0.5], colors=[fs.AQUA], linewidths=1.8)
        a.contour(r.mask, [0.5], colors=[fs.ORANGE], linewidths=1.8)
        a.set_title(f"{t}: Dice {order.tuned[i]:.2f}", fontsize=8)
        a.axis("off")
    fig.suptitle(
        "Active contour (orange) vs expert mask (green), Kvasir-SEG test matches",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fs.save(fig, "ac_gallery")

    # ---- distribution of Dice vs trivial baselines -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, (k, _lab) in enumerate(
        [
            ("tuned", "Active contour (tuned)"),
            ("default", "Active contour (defaults)"),
            ("ellipse", "Central ellipse"),
            ("whole_fov", "Whole FOV"),
        ]
    ):
        v = d[k].to_numpy()
        ax.boxplot(
            v,
            positions=[i],
            vert=False,
            widths=0.55,
            showfliers=False,
            medianprops={"color": fs.INK},
            boxprops={"color": fs.SERIES[min(i, 3)]},
            whiskerprops={"color": fs.MUTED},
        )
        ax.text(1.0, i, f" mean {v.mean():.3f}", va="center", fontsize=8, color=fs.MUTED)
    ax.set_yticks(
        range(4),
        ["Active contour (tuned)", "Active contour (defaults)", "Central ellipse", "Whole FOV"],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel(f"Dice vs expert polyp mask (n={len(d)})")
    ax.set_title("Delineation quality vs trivial priors")
    fs.save(fig, "ac_dice_boxplot")

    # ---- pipeline illustration on three classification images ---------------------------------
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    pick = [
        meta[(meta.label_name == c) & (meta.split == "test")].iloc[3]
        for c in ("polyps", "ulcerative-colitis", "normal-cecum")
    ]
    fig, ax = plt.subplots(3, 5, figsize=(11, 6.8))
    for r_i, row in enumerate(pick):
        raw = cv2.cvtColor(cv2.imread(str(KVASIR_RAW / row["path"])), cv2.COLOR_BGR2RGB)
        c = canonicalise(raw, 256)
        w = cfg_t["work_size"]
        small = cv2.resize(c.image, (w, w), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(c.fov, (w, w), interpolation=cv2.INTER_NEAREST)
        res = segment(small, valid, cfg_t)
        sal = salience_channel(small, valid, cfg_t["channel"], cfg_t["dog_frac"])
        panels = [
            (raw, "raw frame"),
            (c.image, "FOV crop + artefact mask"),
            (sal, "salience (centre-surround)"),
            (small, "Chan-Vese region"),
            (small, "final contour"),
        ]
        for k, (im, t) in enumerate(panels):
            ax[r_i, k].imshow(im, cmap="gray" if k == 2 else None)
            if k == 3:
                ax[r_i, k].contour(res.cv_mask, [0.5], colors=[fs.ORANGE], linewidths=1.6)
            if k == 4:
                ax[r_i, k].contour(res.mask, [0.5], colors=[fs.ORANGE], linewidths=1.6)
            ax[r_i, k].axis("off")
            if r_i == 0:
                ax[r_i, k].set_title(t, fontsize=8)
        ax[r_i, 0].text(
            0, 1.02, row["label_name"], transform=ax[r_i, 0].transAxes, fontsize=8, color=fs.MUTED
        )
    fig.tight_layout()
    fs.save(fig, "ac_pipeline")


if __name__ == "__main__":
    main()
