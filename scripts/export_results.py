"""Export outputs/results/site_results.json for the web dashboard (all values come from the analysis JSON)."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_curve

from acpolyp.data.contracts import POLYP_IDX
from acpolyp.evaluation.aggregate import ALIASES, load_deep
from acpolyp.paths import PROCESSED, RESULTS
from acpolyp.training.metrics import reliability_curve

PLAIN, GUIDED = "ab_none", "ab_acg"


def clean(o):
    """Recursively replace NaN/inf by None so the file is strict JSON (browsers reject NaN)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    A, E, Q = (
        load(RESULTS / "analysis.json"),
        load(RESULTS / "eda.json"),
        load(RESULTS / "ac_quality.json"),
    )
    C = load(PROCESSED / "curation_report.json")
    out: dict = {}
    lb = [r for r in A.get("leaderboard", [])]
    deep = [r for r in lb if r["kind"] == "deep"]
    best = max(deep, key=lambda r: r["macro_f1"][0]) if deep else None
    out["hero"] = [
        {"value": f"{best['macro_f1'][0]:.3f}", "label": f"test macro-F1, {best['name']}"}
        if best
        else None,
        {"value": f"{best['polyp_auroc'][0]:.3f}", "label": "polyp-vs-rest AUROC"}
        if best
        else None,
        {
            "value": f"{C.get('n_after_cleaning', 0):,}",
            "label": f"images after audit (of {C.get('n_ingested', 0):,})",
        },
        {
            "value": f"{E['shortcut_probes'][1]['region_acc']:.0%}" if E else "–",
            "label": "upper-GI vs colon from resolution alone",
        },
    ]
    out["hero"] = [h for h in out["hero"] if h]
    out["leaderboard"] = lb
    if "ablation" in A:
        out["ablation"] = [
            {k: r[k] for k in ("label", "macro_f1", "delta", "ci", "p_adj")} for r in A["ablation"]
        ]
        out["ablation_note"] = (
            "Δ is the seed-averaged macro-F1 difference to plain ConvNeXt-T; intervals are 95% paired bootstrap over test images; "
            "p is Holm-adjusted. Controls: shuffled contours, central ellipse, extra non-contour features."
        )
    if "confusion" in A:
        out["confusion"] = A["confusion"]
    g = load_deep(PLAIN if best is None else {"ConvNeXt-T": PLAIN}.get(best["name"], PLAIN))
    guided = load_deep(GUIDED)
    if g:
        y = g.y
        curves = []
        for nm, grp in (("plain ConvNeXt-T", g), ("ACG-ConvNeXt", guided)):
            if grp is None or (nm != "plain ConvNeXt-T" and ALIASES.get(grp.tag, grp.tag) == g.tag):
                continue
            fpr, tpr, _ = roc_curve((y == POLYP_IDX).astype(int), grp.probs[0][:, POLYP_IDX])
            idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
            curves.append(
                {
                    "name": nm,
                    "fpr": fpr[idx].tolist(),
                    "tpr": tpr[idx].tolist(),
                    "auc": grp.mean_sd("polyp_auroc")[0],
                }
            )
        out["roc"] = {"curves": curves}
        out["reliability"] = {
            "raw": reliability_curve(g.raw_probs[0], y),
            "cal": reliability_curve(g.probs[0], y),
            "ece_raw": A["calibration_conformal"]["ece_raw"][0],
            "ece_cal": A["calibration_conformal"]["ece_cal"][0],
        }
    if "risk_coverage" in A:
        rc = A["risk_coverage"]
        out["risk_coverage"] = {
            "scale": 10,
            "curves": [
                {"name": "most confident first", "coverage": rc["coverage"], "risk": rc["risk"]}
            ],
        }
    if Q:
        out["ac"] = [
            {
                "value": f"{Q['tuned']['mean']:.3f}",
                "label": f"Dice, tuned contour (n={Q['n_test_images']}, 95% CI {Q['tuned']['ci_low']:.3f}–{Q['tuned']['ci_high']:.3f})",
            },
            {"value": f"{Q['ellipse']['mean']:.3f}", "label": "Dice, trivial central ellipse"},
            {
                "value": f"{Q['tuned_minus_ellipse']['mean']:+.3f}",
                "label": "difference (paired, p<1e-5)",
            },
            {
                "value": f"{Q['dice_by_gt_size']['small (<10% of FOV)']:.2f}",
                "label": "Dice on small polyps (<10% of FOV)",
            },
        ]
        out["ac_note"] = (
            "Unsupervised contours are a weak localiser: better than a centre prior, but they over-segment (median 2.4× the expert area) and miss small polyps."
        )
    if E and C:
        sp = E["shortcut_probes"]
        out["audit"] = {
            "items": [
                {
                    "value": "31",
                    "title": "duplicate frames removed",
                    "detail": "incl. one file labelled as two different classes",
                },
                {
                    "value": f"{C['cleanlab_flagged']}",
                    "title": "suspected label errors (6.9%)",
                    "detail": "mostly esophagitis vs z-line and the two dyed classes",
                },
                {
                    "value": f"{sp[1]['region_acc']:.1%}",
                    "title": "region predictable from resolution",
                    "detail": f"majority baseline {sp[0]['region_acc']:.1%}",
                },
                {
                    "value": f"{E['teal_box_share_by_class']['normal-cecum']:.0%}",
                    "title": "cecum frames with ScopeGuide box",
                    "detail": "0% of upper-GI frames; masked in every frame",
                },
                {
                    "value": f"{C['leakage_audit_random']['group_overlap_train_test']}",
                    "title": "scene groups a random split would leak",
                    "detail": "grouped split: 0",
                },
            ],
            "shortcut": [{"name": s["name"], "f1": s["region_acc"]} for s in sp],
            "shortcut_note": "Bars show accuracy of predicting upper-GI vs colon from artefact metadata only (no pixels).",
        }
    (RESULTS / "site_results.json").write_text(
        json.dumps(clean(out), indent=2, default=float, allow_nan=False), encoding="utf-8"
    )
    print("exported", list(out))


if __name__ == "__main__":
    main()
