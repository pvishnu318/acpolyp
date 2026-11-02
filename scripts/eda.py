"""Exploratory data analysis + shortcut audit (Phase VIII steps 28-31, Phase XII step 49).

Writes outputs/results/eda.json and outputs/figures/eda_*.{pdf,png}.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sps
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from acpolyp.ac.features import family_of
from acpolyp.data.contracts import CLASSES
from acpolyp.evaluation import figstyle as fs
from acpolyp.models.embed import cached
from acpolyp.paths import PROCESSED, RESULTS, ensure

NICE = {c: c.replace("-", " ") for c in CLASSES}


def main() -> None:
    ensure(RESULTS)
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    ac = pd.read_parquet(PROCESSED / "ac_features.parquet")
    kept = np.load(PROCESSED / "kept_idx.npy")
    out: dict = {}
    y = meta["label"].to_numpy()
    tr, te = (meta["split"] == s for s in ("train", "test"))

    # ---- class balance (step 29) ------------------------------------------------------------
    counts = meta["label_name"].value_counts().reindex(CLASSES)
    out["class_counts"] = counts.to_dict()
    out["imbalance_ratio_multiclass"] = float(counts.max() / counts.min())
    out["polyp_vs_rest"] = {
        "polyp": int(counts["polyps"]),
        "rest": int(counts.sum() - counts["polyps"]),
        "ratio_rest_to_polyp": float((counts.sum() - counts["polyps"]) / counts["polyps"]),
    }

    # ---- resolution / artefact profile (steps 28, 30) -----------------------------------------
    meta["format"] = meta["width"].astype(str) + "x" + meta["height"].astype(str)
    top = meta["format"].value_counts().index[:3]
    meta["format3"] = np.where(meta["format"].isin(top), meta["format"], "other")
    ct = pd.crosstab(meta["label_name"], meta["format3"], normalize="index").reindex(CLASSES)
    out["format_share_by_class"] = ct.round(4).to_dict("index")
    out["teal_box_share_by_class"] = (
        (meta.assign(has=meta.teal_frac > 0).groupby("label_name")["has"].mean().reindex(CLASSES))
        .round(4)
        .to_dict()
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    bottom = np.zeros(len(CLASSES))
    for i, col in enumerate(ct.columns):
        ax[0].barh(
            range(len(CLASSES)),
            ct[col].to_numpy(),
            left=bottom,
            color=(fs.SERIES + ["#898781"])[i],
            label=col,
            height=0.7,
            edgecolor="white",
            linewidth=1.5,
        )
        bottom += ct[col].to_numpy()
    ax[0].set_yticks(range(len(CLASSES)), [NICE[c] for c in CLASSES])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("share of images")
    ax[0].set_title("Native resolution leaks the label")
    ax[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.45))
    teal = meta.assign(has=meta.teal_frac > 0).groupby("label_name")["has"].mean().reindex(CLASSES)
    ax[1].barh(range(len(CLASSES)), teal.to_numpy(), color=fs.BLUE, height=0.7)
    ax[1].set_yticks(range(len(CLASSES)), [""] * len(CLASSES))
    ax[1].invert_yaxis()
    ax[1].set_xlabel("share with ScopeGuide box")
    ax[1].set_title("ScopeGuide box presence")
    for i, v in enumerate(teal.to_numpy()):
        ax[1].text(v + 0.01, i, f"{v:.0%}", va="center", color=fs.MUTED, fontsize=8)
    fs.save(fig, "eda_artefacts")

    # ---- image statistics by class + normality (step 28) -------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, col, t in zip(
        ax,
        ["mean_luma", "lap_var", "fov_frac"],
        ["Mean luma (FOV)", "Sharpness (Laplacian var.)", "FOV area fraction"],
    ):
        data = [
            np.log10(meta.loc[meta.label_name == c, col] + 1)
            if col == "lap_var"
            else meta.loc[meta.label_name == c, col]
            for c in CLASSES
        ]
        parts = a.violinplot(data, vert=False, showmedians=True, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(fs.BLUE)
            b.set_alpha(0.35)
        a.set_yticks(range(1, 9), [NICE[c] if a is ax[0] else "" for c in CLASSES])
        a.invert_yaxis()
        a.set_title(t)
    fs.save(fig, "eda_image_stats")
    rng = np.random.default_rng(0)
    norm = []
    for col in ["mean_luma", "lap_var", "fov_frac"]:
        x = meta[col].to_numpy()
        xs = rng.choice(x, 1000, replace=False)
        norm.append(
            {
                "feature": col,
                "skew": float(sps.skew(x)),
                "shapiro_p": float(sps.shapiro(xs).pvalue),
                "ks_p": float(sps.kstest((x - x.mean()) / x.std(), "norm").pvalue),
            }
        )
    out["normality_image_stats"] = norm

    # ---- AC feature association: correlation, VIF, mutual information (step 30) --------------
    A = ac[[c for c in ac.columns if family_of(c) != "global"]].copy()
    A = A.loc[:, A.std() > 1e-9]
    corr = A.corr(method="spearman").abs()
    out["n_ac_features"] = int(A.shape[1])
    out["n_pairs_abs_spearman_gt_0.9"] = int(((corr.to_numpy() > 0.9).sum() - A.shape[1]) // 2)
    Xtr = (A[tr.to_numpy()] - A[tr.to_numpy()].mean()) / A[tr.to_numpy()].std()
    sub = Xtr.sample(2000, random_state=0).to_numpy()
    vif = np.array([variance_inflation_factor(sub, i) for i in range(sub.shape[1])])
    out["vif_median"], out["n_vif_gt_10"] = float(np.median(vif)), int((vif > 10).sum())
    mi = mutual_info_classif(A[tr.to_numpy()].to_numpy(), y[tr.to_numpy()], random_state=0)
    order = np.argsort(-mi)[:15]
    out["top_mutual_information"] = [{"feature": A.columns[i], "mi": float(mi[i])} for i in order]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cl = corr.iloc[::2, ::2]
    sns.heatmap(
        cl,
        ax=ax[0],
        cmap=sns.light_palette(fs.BLUE, as_cmap=True),
        cbar_kws={"label": "|Spearman ρ|"},
        xticklabels=False,
        yticklabels=False,
        square=True,
    )
    ax[0].set_title("AC feature collinearity (every 2nd feature)")
    ax[0].grid(False)
    ax[1].barh(range(len(order))[::-1], mi[order], color=fs.BLUE, height=0.7)
    ax[1].set_yticks(range(len(order))[::-1], [A.columns[i] for i in order], fontsize=7)
    ax[1].set_xlabel("mutual information with class (nats)")
    ax[1].set_title("Most informative contour features")
    fs.save(fig, "eda_features")

    # ---- manifold (step 31) ------------------------------------------------------------------
    import umap

    emb = cached("vit_small_patch14_dinov2.lvd142m", "canon", 224)[kept]
    z = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=0).fit_transform(
        emb / np.linalg.norm(emb, axis=1, keepdims=True)
    )
    fig, ax = plt.subplots(figsize=(6.2, 5))
    pal = sns.color_palette("colorblind", 8)
    for i, c in enumerate(CLASSES):
        m = y == i
        ax.scatter(
            z[m, 0],
            z[m, 1],
            s=4,
            color=pal[i],
            alpha=0.7,
            label=NICE[c],
            marker=["o", "s", "^", "D", "v", "P", "X", "<"][i],
            linewidths=0,
        )
    ax.legend(markerscale=2.5, fontsize=7, ncol=2, loc="best")
    ax.set_title("UMAP of frozen DINOv2-S/14 embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fs.save(fig, "eda_umap")

    # ---- shortcut probes: how well do artefacts alone predict the label? (step 30) -----------
    upper = (
        meta["label_name"]
        .isin(["esophagitis", "normal-pylorus", "normal-z-line"])
        .to_numpy()
        .astype(int)
    )

    def probe(cols: list[str], name: str) -> dict:
        X = meta[cols].to_numpy(float)
        res, reg = [], []
        for seed in range(5):
            m = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
                n_jobs=4,
            )
            m.fit(X[tr.to_numpy()], y[tr.to_numpy()])
            res.append(f1_score(y[te.to_numpy()], m.predict(X[te.to_numpy()]), average="macro"))
            r = lgb.LGBMClassifier(
                n_estimators=100, num_leaves=7, random_state=seed, verbose=-1, n_jobs=4
            )
            r.fit(X[tr.to_numpy()], upper[tr.to_numpy()])
            reg.append(float((r.predict(X[te.to_numpy()]) == upper[te.to_numpy()]).mean()))
        return {
            "name": name,
            "f1": float(np.mean(res)),
            "sd": float(np.std(res)),
            "features": cols,
            "region_acc": float(np.mean(reg)),
            "region_sd": float(np.std(reg)),
        }

    meta["aspect"] = meta["width"] / meta["height"]
    meta["bbox_w"] = meta["bbox_x1"] - meta["bbox_x0"]
    meta["bbox_h"] = meta["bbox_y1"] - meta["bbox_y0"]
    out["shortcut_probes"] = [
        {
            "name": "chance",
            "f1": 1 / 8,
            "sd": 0.0,
            "features": [],
            "region_acc": float(max(upper.mean(), 1 - upper.mean())),
            "region_sd": 0.0,
        },
        probe(["width", "height"], "native resolution only"),
        probe(
            ["width", "height", "aspect", "bbox_w", "bbox_h", "fov_frac"],
            "resolution + FOV geometry",
        ),
        probe(
            ["width", "height", "aspect", "bbox_w", "bbox_h", "fov_frac", "teal_frac"],
            "+ ScopeGuide box",
        ),
    ]
    sp = out["shortcut_probes"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 2.8), sharey=True)
    for ax, key, sdk, lab, ttl in (
        (axs[0], "f1", "sd", "macro-F1, 8 classes", "Fine-grained class"),
        (axs[1], "region_acc", "region_sd", "accuracy", "Upper-GI vs colon"),
    ):
        ax.barh(
            range(len(sp))[::-1],
            [s_[key] for s_ in sp],
            xerr=[s_[sdk] for s_ in sp],
            color=fs.BLUE,
            height=0.6,
            capsize=3,
        )
        ax.set_xlim(0, 1.08)
        ax.set_xlabel(lab)
        ax.set_title(ttl)
        for i, s_ in enumerate(sp):
            ax.text(
                s_[key] + s_[sdk] + 0.015,
                len(sp) - 1 - i,
                f"{s_[key]:.2f}",
                va="center",
                color=fs.MUTED,
                fontsize=8,
            )
    axs[0].set_yticks(range(len(sp))[::-1], [s_["name"] for s_ in sp])
    fig.suptitle(
        "Label information carried by acquisition artefacts alone (no pixels used)",
        fontsize=10,
        fontweight="bold",
    )
    fs.save(fig, "eda_shortcut_probe")

    (RESULTS / "eda.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "imbalance_ratio_multiclass",
                    "polyp_vs_rest",
                    "teal_box_share_by_class",
                    "n_ac_features",
                    "vif_median",
                    "n_vif_gt_10",
                )
            },
            indent=2,
        )
    )
    print(
        [(s["name"], round(s["f1"], 3), round(s["region_acc"], 3)) for s in out["shortcut_probes"]]
    )


if __name__ == "__main__":
    main()
