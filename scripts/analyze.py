"""Aggregate every run into tables, statistics and figures (Phases XVI-XVIII, step 85).

python scripts/analyze.py            # reads outputs/runs/**, writes outputs/results + outputs/figures
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from acpolyp.data.contracts import CLASSES, POLYP_IDX
from acpolyp.evaluation import figstyle as fs
from acpolyp.evaluation.aggregate import ALIASES, Group, compare, load_classical, load_deep
from acpolyp.evaluation.stats import holm
from acpolyp.paths import PROCESSED, RESULTS, RUNS, ensure
from acpolyp.training.calibration import (
    conformal_report,
    conformal_sets,
    conformal_threshold,
    risk_coverage,
)
from acpolyp.training.metrics import (
    confusion,
    expected_calibration_error,
    full_metrics,
    per_class_table,
    reliability_curve,
)

NICE = {c: c.replace("-", " ").capitalize() for c in CLASSES}
LABELS = {
    "bb_resnet50": "ResNet-50",
    "bb_efficientnetv2_s": "EfficientNetV2-S",
    "bb_convnext_tiny": "ConvNeXt-T",
    "bb_swin_tiny": "Swin-T",
    "bb_dinov2_s": "DINOv2 ViT-S/14",
    "ab_none": "ConvNeXt-T (plain)",
    "ab_gfeat": "+ global hand-crafted features",
    "ab_ac_pool": "+ contour pooling",
    "ab_ac_feat": "+ contour features",
    "ab_acg": "ACG-ConvNeXt (full)",
    "ab_acg_shuffle": "full, shuffled contours (control)",
    "ab_acg_ellipse": "full, central ellipse (control)",
    "gen_resnet50_acg": "ACG-ResNet-50",
    "sc_naive_frames": "ConvNeXt-T, naive frames",
    "sc_random_split": "ConvNeXt-T, random split",
}
CLASSICAL = {
    "lgbm__global": "LightGBM · global colour/texture",
    "lgbm__ac_all": "LightGBM · contour features only",
    "lgbm__global_ac": "LightGBM · global + contour",
    "logreg__global_ac": "Logistic reg. · global + contour",
    "lgbm__global_ellipse": "LightGBM · global + central-ellipse features (control)",
    "lgbm__global_shuffle": "LightGBM · global + shuffled-contour features (control)",
    "svm_rbf__global_ac": "RBF-SVM · global + contour",
    "rf__global_ac": "Random forest · global + contour",
    "xgb__global_ac": "XGBoost · global + contour",
    "logreg__emb_vit_small_patch14_dinov2.lvd142m": "Linear probe · frozen DINOv2",
}
ABL = [
    "ab_none",
    "ab_gfeat",
    "ab_ac_pool",
    "ab_ac_feat",
    "ab_acg",
    "ab_acg_shuffle",
    "ab_acg_ellipse",
]


def ms(g: Group, k: str) -> list[float]:
    m, s = g.mean_sd(k)
    return [m, s]


def row(g: Group, label: str) -> dict:
    r = {"name": label, "key": g.tag, "n_seeds": len(g.seeds), "kind": g.kind}
    for k in ("macro_f1", "acc", "mcc", "polyp_auroc", "polyp_auprc", "ece", "nll", "brier"):
        r[k] = ms(g, k)
    if g.kind == "deep":
        s = g.summaries
        r["params_m"] = s[0]["params_total"] / 1e6
        r["latency_ms"] = float(np.mean([x["latency_ms_bs1"] for x in s]))
        r["vram_mb"] = float(np.mean([x["peak_vram_mb"] for x in s]))
        r["train_min"] = float(np.mean([x["train_time_s"] for x in s]) / 60)
        r["ckpt_mb"] = float(np.mean([x["ckpt_mb"] for x in s]))
    return r


def latex_table(
    rows: list[dict], cols: list[tuple[str, str, int, int]], caption: str, label: str
) -> str:
    """cols: (key, header, decimals, direction); direction +1 higher-is-better, -1 lower, 0 none."""
    best = {}
    for k, _, _, d in cols:
        if d:
            vals = [
                (r[k][0] if isinstance(r.get(k), list) else r.get(k))
                for r in rows
                if r.get(k) is not None and r.get("kind") != "trivial"
            ]
            best[k] = (max if d > 0 else min)(vals)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "Model & " + " & ".join(h for _, h, _, _ in cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = []
        for k, _, dec, d in cols:
            v = r.get(k)
            if v is None:
                cells.append("--")
                continue
            val = v[0] if isinstance(v, list) else v
            s = f"{val:.{dec}f}" + (
                rf"{{\scriptsize$\pm${v[1]:.{dec}f}}}" if isinstance(v, list) and v[1] > 0 else ""
            )
            if d and r.get("kind") != "trivial" and abs(val - best[k]) < 1e-12:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        lines.append(
            r["name"].replace("&", r"\&").replace("_", r"\_").replace("·", r"$\cdot$")
            + " & "
            + " & ".join(cells)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}\end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines)


def main() -> None:
    ensure(RESULTS)
    tdir = RESULTS / "tables"
    ensure(tdir)
    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    test_rows = np.where(meta.split == "test")[0]
    out: dict = {}

    deep = {
        t: load_deep(t)
        for t in [
            "bb_resnet50",
            "bb_efficientnetv2_s",
            "bb_convnext_tiny",
            "bb_swin_tiny",
            "bb_dinov2_s",
            *ABL[1:],
            "ab_none",
            "gen_resnet50_acg",
            "sc_naive_frames",
            "sc_random_split",
        ]
    }
    deep = {k: v for k, v in deep.items() if v is not None}
    clas = {k: load_classical(k) for k in CLASSICAL}
    clas = {k: v for k, v in clas.items() if v is not None}

    # ---------------------------------------------------------------- leaderboard
    lb = [
        row(deep[t], LABELS[t])
        for t in [
            "bb_resnet50",
            "bb_efficientnetv2_s",
            "bb_convnext_tiny",
            "bb_swin_tiny",
            "bb_dinov2_s",
            "ab_acg",
            "gen_resnet50_acg",
        ]
        if t in deep
    ]
    lb += [row(clas[k], CLASSICAL[k]) for k in CLASSICAL if k in clas]
    triv = (
        json.loads((RUNS / "classical" / "trivial.json").read_text())
        if (RUNS / "classical" / "trivial.json").exists()
        else {}
    )
    for k, nm in (
        ("random", "Random guess"),
        ("majority", "Class prior (majority)"),
        ("colour_centroid_heuristic", "Heuristic · nearest Lab-colour centroid"),
    ):
        if k in triv:
            lb.append(
                {
                    "name": nm,
                    "kind": "trivial",
                    "n_seeds": 1,
                    **{
                        m: [triv[k][m], 0.0]
                        for m in (
                            "macro_f1",
                            "acc",
                            "mcc",
                            "polyp_auroc",
                            "polyp_auprc",
                            "ece",
                            "nll",
                            "brier",
                        )
                    },
                }
            )
    out["leaderboard"] = lb
    cols = [
        ("macro_f1", "Macro-F1", 4, 1),
        ("acc", "Acc.", 4, 1),
        ("mcc", "MCC", 4, 1),
        ("polyp_auroc", "Polyp AUROC", 4, 1),
        ("ece", "ECE", 4, -1),
    ]
    (tdir / "leaderboard.tex").write_text(
        latex_table(
            lb,
            cols,
            "Held-out test performance (mean$\\pm$SD over seeds; deep models $N{=}3$, classical $N{=}5$). Best per column in bold. ECE after temperature scaling.",
            "tab:leaderboard",
        ),
        encoding="utf-8",
    )

    prof = [
        row(deep[t], LABELS[t])
        for t in [
            "bb_resnet50",
            "bb_efficientnetv2_s",
            "bb_convnext_tiny",
            "bb_swin_tiny",
            "bb_dinov2_s",
            "ab_acg",
        ]
        if t in deep
    ]
    out["profiling"] = prof
    (tdir / "profiling.tex").write_text(
        latex_table(
            prof,
            [
                ("params_m", "Params (M)", 1, 0),
                ("latency_ms", "ms/img (bs1)", 1, -1),
                ("vram_mb", "Peak VRAM (MB)", 0, -1),
                ("train_min", "Train (min)", 1, -1),
                ("ckpt_mb", "Ckpt (MB)", 0, -1),
            ],
            "Efficiency on an RTX 4060 Laptop GPU (8 GB), bf16 autocast, batch size 1 latency.",
            "tab:profiling",
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------- ablation with statistics
    abl = {t: deep[t] for t in ABL if t in deep}
    if "ab_none" in abl and len(abl) > 1:
        base = abl["ab_none"]
        comps = {t: compare(g, base) for t, g in abl.items() if t != "ab_none" and len(g.seeds) > 0}
        adj = holm([c["p_boot"] for c in comps.values()])
        rows = [
            {
                "tag": "ab_none",
                "label": LABELS["ab_none"],
                "macro_f1": ms(base, "macro_f1"),
                "acc": ms(base, "acc"),
                "polyp_auroc": ms(base, "polyp_auroc"),
                "delta": 0.0,
                "ci": None,
                "p_adj": None,
            }
        ]
        for (t, c), pa in zip(comps.items(), adj):
            rows.append(
                {
                    "tag": t,
                    "label": LABELS[t],
                    "macro_f1": ms(abl[t], "macro_f1"),
                    "acc": ms(abl[t], "acc"),
                    "polyp_auroc": ms(abl[t], "polyp_auroc"),
                    "delta": c["diff"],
                    "ci": c["ci"],
                    "p_boot": c["p_boot"],
                    "p_adj": pa,
                    "p_mcnemar": c["mcnemar_ensemble"]["p_mcnemar"],
                    "p_delong_polyp": c["delong_polyp_ensemble"]["p_delong"],
                    "seed_level": c["seed_level_f1"],
                }
            )
        out["ablation"], out["ablation_comparisons"] = rows, comps
        tex = [
            {
                "name": r["label"],
                "macro_f1": r["macro_f1"],
                "acc": r["acc"],
                "polyp_auroc": r["polyp_auroc"],
                "delta": r["delta"],
                "ci": f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" if r["ci"] else "--",
                "p": f"{r['p_adj']:.3f}" if r["p_adj"] is not None else "--",
            }
            for r in rows
        ]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Ablation on ConvNeXt-T ($N{=}3$ seeds). $\Delta$: seed-averaged macro-F1 difference to the plain baseline with 95\,\% paired-bootstrap interval over test images; $p$: Holm-adjusted bootstrap p-value.}",
            r"\label{tab:ablation}",
            r"\small",
            r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrl r}",
            r"\toprule",
            r"Variant & Macro-F1 & Acc. & Polyp AUROC & $\Delta$ [95\,\% CI] & $p_\mathrm{Holm}$ \\",
            r"\midrule",
        ]
        for r in tex:
            lines.append(
                f"{r['name']} & {r['macro_f1'][0]:.4f}{{\\scriptsize$\\pm${r['macro_f1'][1]:.4f}}} & {r['acc'][0]:.4f} & {r['polyp_auroc'][0]:.4f} & {r['delta']:+.4f} {r['ci']} & {r['p']} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}\end{adjustbox}", r"\end{table*}"]
        (tdir / "ablation.tex").write_text("\n".join(lines), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        cl = [r for r in rows if r["tag"] != "ab_none"]
        for i, r in enumerate(cl):
            col = (
                fs.ORANGE
                if r["tag"] == "ab_acg"
                else (
                    fs.AQUA
                    if "shuffle" in r["tag"] or "ellipse" in r["tag"] or r["tag"] == "ab_gfeat"
                    else fs.BLUE
                )
            )
            ax.errorbar(
                r["delta"],
                i,
                xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]],
                fmt=fs.MARKERS[0],
                color=col,
                capsize=3,
            )
        ax.axvline(0, color=fs.MUTED, lw=1, ls="--")
        ax.set_yticks(range(len(cl)), [r["label"] for r in cl])
        ax.invert_yaxis()
        ax.set_xlabel("Δ macro-F1 vs plain ConvNeXt-T (95% paired bootstrap)")
        ax.set_title("What does contour guidance actually add?")
        fs.save(fig, "ablation_forest")

    # ---------------------------------------------------------------- headline model + calibration/conformal
    cand = {
        t: g
        for t, g in deep.items()
        if t
        in (
            "bb_resnet50",
            "bb_efficientnetv2_s",
            "bb_convnext_tiny",
            "bb_swin_tiny",
            "bb_dinov2_s",
            "ab_acg",
            "ab_none",
        )
    }
    if cand:
        # The analysed model is pre-declared as the proposed method (never chosen by test score).
        best_tag = "ab_acg" if "ab_acg" in cand else next(iter(cand))
        out["analysed_model"] = best_tag
        g = cand[best_tag]
        # calibration + conformal, per seed then averaged
        cal_rows = []
        for s_i in range(len(g.seeds)):
            sets_q = conformal_threshold(g.val_probs[s_i], g.val_y, 0.05)
            rep = conformal_report(conformal_sets(g.probs[s_i], sets_q), g.y)
            cal_rows.append(
                {
                    "ece_raw": expected_calibration_error(g.raw_probs[s_i], g.y),
                    "ece_cal": expected_calibration_error(g.probs[s_i], g.y),
                    **rep,
                }
            )
        out["calibration_conformal"] = {
            k: [float(np.mean([r[k] for r in cal_rows])), float(np.std([r[k] for r in cal_rows]))]
            for k in cal_rows[0]
        }

        # Validation data were used for early stopping, temperature and threshold, so they are not exchangeable with
        # fresh data and the conformal guarantee need not hold. Remedy: calibrate on a random half of the test *scene
        # groups* and evaluate on the other half; repeat 200 times.
        gid = meta["group_id"].to_numpy()[test_rows]
        ug = np.unique(gid)
        rngc = np.random.default_rng(0)
        cv_rows = []
        for _ in range(200):
            cal_g = set(rngc.choice(ug, len(ug) // 2, replace=False))
            in_cal = np.array([x in cal_g for x in gid])
            s_i = int(rngc.integers(len(g.seeds)))
            q = conformal_threshold(g.probs[s_i][in_cal], g.y[in_cal], 0.05)
            cv_rows.append(conformal_report(conformal_sets(g.probs[s_i][~in_cal], q), g.y[~in_cal]))
        out["conformal_test_split_calibrated"] = {
            k: [float(np.mean([r[k] for r in cv_rows])), float(np.std([r[k] for r in cv_rows]))]
            for k in cv_rows[0]
        }
        out["conformal_note"] = (
            "val-calibrated sets under-cover because validation was reused for model selection; group-aware test-split calibration restores coverage"
        )
        p0, r0, y = g.probs[0], g.raw_probs[0], g.y

        fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.3))
        for a, p, nm, c, mk in (
            (ax[0], r0, "raw", fs.ORANGE, "s"),
            (ax[0], p0, "temperature-scaled", fs.BLUE, "o"),
        ):
            rc = reliability_curve(p, y)
            a.plot(
                rc["conf"],
                rc["acc"],
                marker=mk,
                color=c,
                label=f"{nm} (ECE {expected_calibration_error(p, y):.3f})",
            )
        ax[0].plot([0, 1], [0, 1], ls="--", color=fs.MUTED, lw=1)
        ax[0].set_xlabel("confidence")
        ax[0].set_ylabel("accuracy")
        ax[0].legend(loc="upper left")
        ax[0].set_title("Reliability")
        yb = (y == POLYP_IDX).astype(int)
        for i, t in enumerate(
            [best_tag]
            + [
                x
                for x in ("ab_none",)
                if x in deep and ALIASES.get(x, x) != ALIASES.get(best_tag, best_tag)
            ]
        ):
            gg = deep[ALIASES.get(t, t)]
            fpr, tpr, _ = roc_curve(yb, gg.probs[0][:, POLYP_IDX])
            pr, rc_, _ = precision_recall_curve(yb, gg.probs[0][:, POLYP_IDX])
            ax[1].plot(
                fpr, tpr, color=fs.SERIES[i], ls=fs.LINESTYLES[i], label=f"{LABELS.get(t, t)}"
            )
            ax[2].plot(rc_, pr, color=fs.SERIES[i], ls=fs.LINESTYLES[i], label=LABELS.get(t, t))
        ax[1].set_xlabel("1 − specificity")
        ax[1].set_ylabel("sensitivity")
        ax[1].set_title("Polyp vs rest: ROC")
        ax[1].set_xlim(0, 0.3)
        ax[1].set_ylim(0.7, 1.0)
        ax[1].legend(loc="lower right")
        ax[2].set_xlabel("recall")
        ax[2].set_ylabel("precision")
        ax[2].set_title("Polyp vs rest: PR")
        ax[2].set_ylim(0.8, 1.0)
        fig.tight_layout()
        fs.save(fig, "calibration_roc_pr")

        cov, risk = risk_coverage(p0, y)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ax.plot(cov, risk * 100, color=fs.BLUE)
        for a_ in (0.9, 0.8):
            i = int(a_ * len(cov)) - 1
            ax.annotate(
                f"{a_:.0%} coverage → {risk[i] * 100:.2f}% error",
                (cov[i], risk[i] * 100),
                xytext=(0.2, 3.0 + 1.2 * (a_ == 0.9)),
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": fs.MUTED},
            )
        ax.set_xlabel("coverage (most confident first)")
        ax.set_ylabel("error rate (%)")
        ax.set_title("Selective classification")
        fs.save(fig, "risk_coverage")
        out["risk_coverage"] = {
            "coverage": cov[::40].tolist(),
            "risk": risk[::40].tolist(),
            "model": best_tag,
        }

        # ---------------------------------------------------------------- asymmetric-cost operating point (step 4)
        def exp_cost(
            p_: np.ndarray, y_: np.ndarray, thr: float, c_fn: float = 5.0, c_fp: float = 1.0
        ) -> dict:
            yb_, pr_ = (y_ == POLYP_IDX), p_[:, POLYP_IDX] >= thr
            fn, fp = int((yb_ & ~pr_).sum()), int((~yb_ & pr_).sum())
            return {
                "threshold": float(thr),
                "cost_per_image": (c_fn * fn + c_fp * fp) / len(y_),
                "sens": float(1 - fn / max(yb_.sum(), 1)),
                "spec": float(1 - fp / max((~yb_).sum(), 1)),
                "fn": fn,
                "fp": fp,
            }

        grid = np.linspace(0.005, 0.95, 190)
        cost_rows = []
        for s_i in range(len(g.seeds)):
            vv = [exp_cost(g.val_probs[s_i], g.val_y, t_)["cost_per_image"] for t_ in grid]
            t_cost = float(grid[int(np.argmin(vv))])
            t_youden = g.summaries[s_i]["polyp_threshold"]
            cost_rows.append(
                {
                    "youden": exp_cost(g.probs[s_i], g.y, t_youden),
                    "cost_opt": exp_cost(g.probs[s_i], g.y, t_cost),
                    "argmax_0.5": exp_cost(g.probs[s_i], g.y, 0.5),
                }
            )
        out["cost_sensitive"] = {
            k: {
                m: float(np.mean([r[k][m] for r in cost_rows]))
                for m in ("threshold", "cost_per_image", "sens", "spec", "fn", "fp")
            }
            for k in cost_rows[0]
        }
        out["cost_sensitive"]["c_fn_over_c_fp"] = 5.0

        # ---------------------------------------------------------------- confusion, per-class, errors
        ens = g.ensemble()
        cm = confusion(ens, y)
        out["confusion"] = {
            "model": LABELS.get(best_tag, best_tag) + " (seed ensemble)",
            "labels": CLASSES,
            "matrix": cm.tolist(),
        }
        out["per_class"] = per_class_table(ens, y)
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap=plt.cm.Blues, vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else fs.INK,
                )
        ax.set_xticks(range(8), [NICE[c] for c in CLASSES], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(8), [NICE[c] for c in CLASSES], fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"Confusion matrix, {LABELS.get(best_tag, best_tag)}")
        fs.save(fig, "confusion_matrix")

        pred = ens.argmax(1)
        wrong = np.where(pred != y)[0]
        conf = ens.max(1)
        pairs = (
            pd.Series([f"{CLASSES[y[i]]} → {CLASSES[pred[i]]}" for i in wrong])
            .value_counts()
            .head(8)
        )
        flagged = meta["label_issue"].to_numpy()[test_rows]
        out["errors"] = {
            "n_errors": int(len(wrong)),
            "error_rate": float(len(wrong) / len(y)),
            "top_confusions": pairs.to_dict(),
            "frac_errors_label_issue_flagged": float(flagged[wrong].mean()) if len(wrong) else 0.0,
            "frac_correct_label_issue_flagged": float(flagged[pred == y].mean()),
            "mean_conf_errors": float(conf[wrong].mean()),
            "mean_conf_correct": float(conf[pred == y].mean()),
            "frac_errors_in_two_class_pairs": float(
                np.mean(
                    [
                        {CLASSES[y[i]], CLASSES[pred[i]]}
                        in (
                            {"esophagitis", "normal-z-line"},
                            {"dyed-lifted-polyps", "dyed-resection-margins"},
                        )
                        for i in wrong
                    ]
                )
            ),
        }
        imgs = (
            np.load(PROCESSED / "images_canon_clean.npy", mmap_mode="r")
            if (PROCESSED / "images_canon_clean.npy").exists()
            else None
        )
        if imgs is not None and len(wrong):
            top = wrong[np.argsort(-conf[wrong])][:12]
            fig, ax = plt.subplots(3, 4, figsize=(9.4, 7.4))
            for a, i in zip(ax.ravel(), top):
                a.imshow(imgs[test_rows[i]])
                a.axis("off")
                a.set_title(
                    f"true: {NICE[CLASSES[y[i]]]}\npred: {NICE[CLASSES[pred[i]]]} ({conf[i]:.2f})"
                    + ("\n[label flagged by confident learning]" if flagged[i] else "\n"),
                    fontsize=6.8,
                )
            fig.suptitle("Most confident errors on the test split", fontsize=10, fontweight="bold")
            fig.tight_layout()
            fs.save(fig, "error_gallery")
            out["errors"]["top12_test_positions"] = [int(i) for i in top]
            out["errors"]["top12_pairs"] = [f"{CLASSES[y[i]]} → {CLASSES[pred[i]]}" for i in top]
            out["errors"]["top12_flagged"] = [bool(flagged[i]) for i in top]

        # ---------------------------------------------------------------- subgroup audit
        t = meta.iloc[test_rows].reset_index(drop=True)
        t["correct"] = pred == y
        t["fmt"] = np.where(
            t.width == 720, "720x576 (colon scope)", "larger formats (upper-GI scope)"
        )
        t["box"] = np.where(t.teal_frac > 0, "ScopeGuide box present", "no box")
        sub = {}
        for col in ("fmt", "box", "label_name"):
            sub[col] = (
                t.groupby(col)["correct"]
                .agg(["mean", "size"])
                .rename(columns={"mean": "acc", "size": "n"})
                .round(4)
                .to_dict("index")
            )
        out["subgroups"] = sub
        out["worst_group_acc"] = float(min(v["acc"] for v in sub["label_name"].values()))

    # ---------------------------------------------------------------- shortcut / leakage study
    sc = {}
    for t_ in ("ab_none", "sc_naive_frames", "sc_random_split"):
        if t_ in deep:
            sc[t_] = {
                "label": LABELS[t_],
                "macro_f1": ms(deep[t_], "macro_f1"),
                "acc": ms(deep[t_], "acc"),
                "polyp_auroc": ms(deep[t_], "polyp_auroc"),
                "n_seeds": len(deep[t_].seeds),
            }
    out["shortcut_study"] = sc
    if "sc_naive_frames" in deep and "ab_none" in deep:
        out["naive_vs_canonical"] = compare(deep["sc_naive_frames"], deep["ab_none"])

    # ---------------------------------------------------------------- learning curves
    lc = []
    for m_ in ("none", "acg"):
        for f in (10, 25, 50):
            g = load_deep(f"ld_{m_}_{f}")
            if g:
                lc.append(
                    {
                        "mode": m_,
                        "frac": f / 100,
                        "macro_f1": ms(g, "macro_f1"),
                        "n_seeds": len(g.seeds),
                    }
                )
        if "ab_none" in deep and m_ == "none":
            lc.append(
                {
                    "mode": "none",
                    "frac": 1.0,
                    "macro_f1": ms(deep["ab_none"], "macro_f1"),
                    "n_seeds": len(deep["ab_none"].seeds),
                }
            )
        if "ab_acg" in deep and m_ == "acg":
            lc.append(
                {
                    "mode": "acg",
                    "frac": 1.0,
                    "macro_f1": ms(deep["ab_acg"], "macro_f1"),
                    "n_seeds": len(deep["ab_acg"].seeds),
                }
            )
    out["learning_curve"] = lc
    if lc:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for i, (m_, nm) in enumerate((("none", "plain ConvNeXt-T"), ("acg", "ACG-ConvNeXt"))):
            pts = sorted([r for r in lc if r["mode"] == m_], key=lambda r: r["frac"])
            ax.errorbar(
                [p["frac"] * 100 for p in pts],
                [p["macro_f1"][0] for p in pts],
                yerr=[p["macro_f1"][1] for p in pts],
                marker=fs.MARKERS[i],
                color=fs.SERIES[i],
                ls=fs.LINESTYLES[i],
                capsize=3,
                label=nm,
            )
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.set_xlabel("training data used")
        ax.set_ylabel("test macro-F1")
        ax.legend()
        ax.set_title("Learning curves")
        fs.save(fig, "learning_curves")

    # ---------------------------------------------------------------- learning-rate sensitivity (step 75), seed 0 only
    lr_rows = []
    for m_, base_tag in (("none", "ab_none"), ("acg", "ab_acg")):
        for lr_tag, lr in (("half", 1e-4), ("default", 2e-4), ("double", 4e-4)):
            g_ = deep.get(base_tag) if lr_tag == "default" else load_deep(f"lr_{m_}_{lr_tag}")
            if g_ is not None and 0 in g_.seeds:
                i0 = g_.seeds.index(0)
                lr_rows.append(
                    {
                        "mode": m_,
                        "lr": lr,
                        "macro_f1": float(full_metrics(g_.probs[i0], g_.y)["macro_f1"]),
                    }
                )
    out["lr_sensitivity"] = lr_rows
    # Exploratory replication at 2x learning rate (chosen AFTER seeing seed-0 test results: post hoc, not confirmatory)
    g_n, g_a = load_deep("lr_none_double"), load_deep("lr_acg_double")
    if g_n is not None and g_a is not None and len(g_n.seeds) >= 3 and len(g_a.seeds) >= 3:
        c2 = compare(g_a, g_n)
        out["lr2x_replication"] = {
            "plain": ms(g_n, "macro_f1"),
            "acg": ms(g_a, "macro_f1"),
            "n_seeds": [len(g_n.seeds), len(g_a.seeds)],
            "delta": c2["diff"],
            "ci": c2["ci"],
            "p_boot": c2["p_boot"],
            "p_mcnemar": c2["mcnemar_ensemble"]["p_mcnemar"],
            "note": "post hoc; learning rate selected after observing test results for seed 0",
        }
    if lr_rows:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for i, (m_, nm) in enumerate((("none", "plain ConvNeXt-T"), ("acg", "ACG-ConvNeXt"))):
            pts_ = sorted([r for r in lr_rows if r["mode"] == m_], key=lambda r: r["lr"])
            ax.plot(
                [p["lr"] for p in pts_],
                [p["macro_f1"] for p in pts_],
                marker=fs.MARKERS[i],
                color=fs.SERIES[i],
                ls=fs.LINESTYLES[i],
                label=nm,
            )
        ax.set_xscale("log")
        ax.set_xlabel("head learning rate (backbone = 0.25x)")
        ax.set_ylabel("test macro-F1 (seed 0)")
        ax.legend()
        ax.set_title("Learning-rate sensitivity")
        fs.save(fig, "lr_sensitivity")

    # ---------------------------------------------------------------- backbone ensemble (step 67)
    bb = [
        deep[t]
        for t in (
            "bb_resnet50",
            "bb_efficientnetv2_s",
            "bb_convnext_tiny",
            "bb_swin_tiny",
            "bb_dinov2_s",
        )
        if t in deep
    ]
    if len(bb) >= 3:
        pe = np.mean([g.ensemble() for g in bb], axis=0)
        out["backbone_ensemble"] = {"n_models": len(bb), **full_metrics(pe, bb[0].y)}

    # ---------------------------------------------------------------- Pareto: accuracy vs latency (step 75)
    pts = [r for r in prof]
    if pts:
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        for r in pts:
            ax.scatter(
                r["latency_ms"],
                r["macro_f1"][0],
                s=40 + 4 * r["params_m"],
                color=fs.ORANGE if "ACG" in r["name"] else fs.BLUE,
                marker=fs.MARKERS[0],
                alpha=0.85,
            )
            ax.annotate(
                r["name"],
                (r["latency_ms"], r["macro_f1"][0]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("latency per image, batch 1 (ms)")
        ax.set_ylabel("test macro-F1")
        ax.set_title("Accuracy vs latency (marker area ∝ params)")
        fs.save(fig, "pareto")

    (RESULTS / "analysis.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "analysed_model",
                    "calibration_conformal",
                    "errors",
                    "shortcut_study",
                    "backbone_ensemble",
                    "worst_group_acc",
                )
                if k in out
            },
            indent=2,
            default=float,
        )
    )
    if "ablation" in out:
        print(
            pd.DataFrame(
                [
                    {
                        k: (r[k][0] if isinstance(r[k], list) else r[k])
                        for k in ("label", "macro_f1", "delta", "p_adj")
                    }
                    for r in out["ablation"]
                ]
            )
            .round(4)
            .to_string()
        )


if __name__ == "__main__":
    main()
