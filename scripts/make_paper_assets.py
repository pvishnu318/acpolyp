"""Turn result JSONs into LaTeX macros and copy figures/tables into paper/ (Phase XX step 85).

Every number quoted in the manuscript is a macro defined here from a run log, so text and results cannot drift.
"""

from __future__ import annotations

import json
import re
import shutil

from acpolyp.paths import FIGURES, PROCESSED, RESULTS, ROOT

PAPER = ROOT / "paper"
TAGS = {  # run tag -> macro stem (letters only)
    "ab_none": "plain",
    "ab_gfeat": "gfeat",
    "ab_ac_pool": "acpool",
    "ab_ac_feat": "acfeat",
    "ab_acg": "acg",
    "ab_acg_shuffle": "shuf",
    "ab_acg_ellipse": "ell",
    "gen_resnet50_acg": "resacg",
    "sc_naive_frames": "naive",
    "sc_random_split": "randsplit",
}
NAME = {
    "ResNet-50": "resnet",
    "EfficientNetV2-S": "effnet",
    "ConvNeXt-T": "convnext",
    "Swin-T": "swin",
    "DINOv2 ViT-S/14": "dino",
    "ACG-ResNet-50": "resacg",
}


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    A, E = load(RESULTS / "analysis.json"), load(RESULTS / "eda.json")
    Q, C = load(RESULTS / "ac_quality.json"), load(PROCESSED / "curation_report.json")
    INTERP = load(RESULTS / "interpretability.json")
    out: list[str] = []

    def d(name: str, val) -> None:
        val = str(val)
        if "$" not in val and "{" not in val:  # typographic minus for plain numbers
            val = re.sub(r"(?<![0-9eE])-(?=\d)", "$-$", val)
        out.append(rf"\newcommand{{\{name}}}{{{val}}}")

    pct = lambda v, k=1: f"{100 * v:.{k}f}\\,\\%"

    def sci(v: float) -> str:
        m, e = f"{v:.1e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"

    # ---- data ------------------------------------------------------------------------------
    if C:
        d("nRaw", f"{C['n_ingested']:,}".replace(",", "{,}"))
        d("nClean", f"{C['n_after_cleaning']:,}".replace(",", "{,}"))
        d("nDropped", C["dedup"]["rows_dropped"])
        d("nFlagged", C["cleanlab_flagged"])
        d("flaggedPct", pct(C["cleanlab_flagged"] / C["n_after_cleaning"]))
        d("nGroups", f"{C['scene_groups']['n_groups']:,}".replace(",", "{,}"))
        d("randLeakGroups", C["leakage_audit_random"]["group_overlap_train_test"])
        d("randLeakPct", pct(C["leakage_audit_random"]["test_frac_nn_above_0.92"]))
    if E:
        sp = {s["name"]: s for s in E["shortcut_probes"]}
        d("resOnlyAcc", pct(sp["native resolution only"]["region_acc"]))
        d("resGeomAcc", pct(sp["resolution + FOV geometry"]["region_acc"]))
        d("regionMajority", pct(sp["chance"]["region_acc"]))
        d("resGeomF", f"{sp['resolution + FOV geometry']['f1']:.2f}")
        d("tealCecum", pct(E["teal_box_share_by_class"]["normal-cecum"], 0))
        d("vifMedian", f"{E['vif_median']:.0f}")
        d("nVifTen", E["n_vif_gt_10"])
    # ---- active contour --------------------------------------------------------------------
    if Q:
        for k, m in (
            ("tuned", "acDice"),
            ("ellipse", "ellDice"),
            ("default", "defDice"),
            ("whole_fov", "fovDice"),
        ):
            d(m, f"{Q[k]['mean']:.3f}")
        d("acDiceLo", f"{Q['tuned']['ci_low']:.3f}")
        d("acDiceHi", f"{Q['tuned']['ci_high']:.3f}")
        d("acIoU", f"{Q['tuned_iou']['mean']:.3f}")
        t = Q["tuned_minus_ellipse"]
        d("acDelta", f"{t['mean']:+.3f}")
        d("acDeltaLo", f"{t['ci_low']:+.3f}")
        d("acDeltaHi", f"{t['ci_high']:+.3f}")
        d("acWilcoxP", sci(Q["wilcoxon_p_tuned_vs_ellipse"]))
        d("acBetterFrac", pct(Q["frac_images_tuned_better_than_ellipse"], 0))
        d("acNtest", Q["n_test_images"])
        d("acOverSeg", f"{Q['median_pred_over_gt_area']:.1f}")
        for k, m in zip(Q["dice_by_gt_size"], ("acDiceSmall", "acDiceMed", "acDiceLarge")):
            d(m, f"{Q['dice_by_gt_size'][k]:.2f}")
    # ---- models ----------------------------------------------------------------------------
    for r in A.get("leaderboard", []):
        stem = NAME.get(r["name"])
        if stem:
            for k, m in (("macro_f1", "F"), ("acc", "Acc"), ("polyp_auroc", "AU"), ("ece", "ECE")):
                d(f"{stem}{m}", f"{r[k][0]:.4f}")
                d(f"{stem}{m}SD", f"{r[k][1]:.4f}")
    CL = {
        "lgbm__global": "lgbG",
        "lgbm__global_ac": "lgbGAC",
        "lgbm__global_ellipse": "lgbGEll",
        "lgbm__global_shuffle": "lgbGShuf",
        "lgbm__ac_all": "lgbAC",
        "logreg__emb_vit_small_patch14_dinov2.lvd142m": "dinoProbe",
        "svm_rbf__global_ac": "svmGAC",
        "xgb__global_ac": "xgbGAC",
        "rf__global_ac": "rfGAC",
        "logreg__global_ac": "lrGAC",
    }
    for r in A.get("leaderboard", []):
        stem = CL.get(r.get("key", ""))
        if stem:
            d(f"{stem}F", f"{r['macro_f1'][0]:.3f}")
            d(f"{stem}Acc", f"{r['acc'][0]:.3f}")
            d(f"{stem}AU", f"{r['polyp_auroc'][0]:.3f}")
    for r in A.get("leaderboard", []):
        if r["kind"] == "trivial":
            k = {
                "Random guess": "rand",
                "Class prior (majority)": "prior",
                "Heuristic · nearest Lab-colour centroid": "heur",
            }[r["name"]]
            d(f"{k}F", f"{r['macro_f1'][0]:.3f}")
            d(f"{k}AU", f"{r['polyp_auroc'][0]:.3f}")
    for r in A.get("ablation", []):
        stem = TAGS.get(r["tag"])
        for k, m in (("macro_f1", "F"), ("acc", "Acc"), ("polyp_auroc", "AU")):
            d(f"{stem}{m}", f"{r[k][0]:.4f}")
            d(f"{stem}{m}SD", f"{r[k][1]:.4f}")
        if r.get("ci"):
            d(f"{stem}Delta", f"{r['delta']:+.4f}")
            d(f"{stem}DeltaLo", f"{r['ci'][0]:+.4f}")
            d(f"{stem}DeltaHi", f"{r['ci'][1]:+.4f}")
            d(f"{stem}PHolm", f"{r['p_adj']:.3f}")
            d(f"{stem}PMcnemar", f"{r['p_mcnemar']:.3f}")
    for t, v in A.get("shortcut_study", {}).items():
        stem = TAGS.get(t)
        if stem and t != "ab_none":
            d(f"{stem}F", f"{v['macro_f1'][0]:.4f}")
            d(f"{stem}FSD", f"{v['macro_f1'][1]:.4f}")
            d(f"{stem}AU", f"{v['polyp_auroc'][0]:.4f}")
    RB = load(RESULTS / "robustness_multiseed.json")
    if RB:
        vt = {
            "ab_none": "Plain",
            "ab_gfeat": "Gfeat",
            "ab_acg": "Acg",
            "ab_acg_shuffle": "Shuf",
            "ab_acg_ellipse": "Ell",
        }
        lv = {
            "clean100": "Clean",
            "jpeg50": "JpegFifty",
            "jpeg10": "JpegTen",
            "blur3.0": "BlurThree",
        }
        for t, tn in vt.items():
            for k, kn in lv.items():
                d(f"rob{tn}{kn}", f"{sum(RB['acc'][t][k]) / len(RB['acc'][t][k]):.3f}")
        d("robN", RB["n"])
    for r in A.get("lr_sensitivity", []):
        tag = {1e-4: "Half", 2e-4: "Def", 4e-4: "Double"}[r["lr"]]
        d(f"lr{'None' if r['mode'] == 'none' else 'Acg'}{tag}", f"{r['macro_f1']:.4f}")
    for r in A.get("learning_curve", []):
        word = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Full"}[int(round(r["frac"] * 100))]
        d(f"lc{'None' if r['mode'] == 'none' else 'Acg'}{word}", f"{r['macro_f1'][0]:.4f}")
    if "calibration_conformal" in A:
        c = A["calibration_conformal"]
        d("eceRaw", f"{c['ece_raw'][0]:.3f}")
        d("eceCal", f"{c['ece_cal'][0]:.3f}")
        d("confCovVal", pct(c["coverage"][0]))
        t = A["conformal_test_split_calibrated"]
        d("confCov", pct(t["coverage"][0]))
        d("confCovSD", pct(t["coverage"][1]))
        d("confSize", f"{t['mean_set_size'][0]:.2f}")
        d("confSingle", pct(t["frac_singleton"][0], 0))
    if "cost_sensitive" in A:
        cs = A["cost_sensitive"]
        d("costHalf", f"{cs['argmax_0.5']['cost_per_image']:.3f}")
        d("costYouden", f"{cs['youden']['cost_per_image']:.3f}")
        d("costOpt", f"{cs['cost_opt']['cost_per_image']:.3f}")
        d("sensHalf", pct(cs["argmax_0.5"]["sens"]))
        d("specHalf", pct(cs["argmax_0.5"]["spec"]))
        d("sensYouden", pct(cs["youden"]["sens"]))
        d("specYouden", pct(cs["youden"]["spec"]))
    if "subgroups" in A:
        sg = A["subgroups"]
        d("accColonScope", pct(sg["fmt"]["720x576 (colon scope)"]["acc"]))
        d("accUpperScope", pct(sg["fmt"]["larger formats (upper-GI scope)"]["acc"]))
        d("accBox", pct(sg["box"]["ScopeGuide box present"]["acc"]))
        d("accNoBox", pct(sg["box"]["no box"]["acc"]))
        d("worstClassAcc", pct(A["worst_group_acc"]))
    if "errors" in A:
        e = A["errors"]
        d("nErrors", e["n_errors"])
        d("errRate", pct(e["error_rate"]))
        d("errFlagged", pct(e["frac_errors_label_issue_flagged"], 0))
        d("corrFlagged", pct(e["frac_correct_label_issue_flagged"], 0))
        d("errPairs", pct(e["frac_errors_in_two_class_pairs"], 0))
    if "backbone_ensemble" in A:
        d("ensF", f"{A['backbone_ensemble']['macro_f1']:.4f}")
        d("ensAcc", f"{A['backbone_ensemble']['acc']:.4f}")
    if INTERP.get("gradcam_pointing"):
        for t, v in INTERP["gradcam_pointing"].items():
            s = TAGS.get(t, t)
            d(f"{s}Point", pct(v["pointing_game"], 0))
            d(f"{s}CamIoU", f"{v['cam_top20_iou']:.3f}")
            d(f"{s}CamInAC", pct(v["cam_peak_inside_contour"], 0))
    if "lr2x_replication" in A:
        r2 = A["lr2x_replication"]
        d("lrTwoPlain", f"{r2['plain'][0]:.4f}")
        d("lrTwoPlainSD", f"{r2['plain'][1]:.4f}")
        d("lrTwoAcg", f"{r2['acg'][0]:.4f}")
        d("lrTwoAcgSD", f"{r2['acg'][1]:.4f}")
        d("lrTwoDelta", f"{r2['delta']:+.4f}")
        d("lrTwoLo", f"{r2['ci'][0]:+.4f}")
        d("lrTwoHi", f"{r2['ci'][1]:+.4f}")
        d("lrTwoP", f"{r2['p_boot']:.2f}")
    if INTERP.get("shap"):
        for fam, v in INTERP["shap"]["family_share"].items():
            d(f"shap{fam.capitalize()}", pct(v, 0))
    if INTERP.get("layerwise_probe"):
        lp = INTERP["layerwise_probe"]
        d("lwPreFour", f"{lp['pretrained'][3]:.3f}")
        d("lwFtFour", f"{lp['finetuned'][3]:.3f}")
        d("lwPreThree", f"{lp['pretrained'][2]:.3f}")
        d("lwFtThree", f"{lp['finetuned'][2]:.3f}")
        d("lwPreOne", f"{lp['pretrained'][0]:.3f}")
        d("lwFtOne", f"{lp['finetuned'][0]:.3f}")
    (PAPER / "numbers.tex").write_text(
        "% generated by scripts/make_paper_assets.py -- do not edit\n" + "\n".join(out) + "\n",
        encoding="utf-8",
    )
    (PAPER / "figures").mkdir(exist_ok=True)
    (PAPER / "tables").mkdir(exist_ok=True)
    for f in FIGURES.glob("*.pdf"):
        shutil.copy(f, PAPER / "figures" / f.name)
    for f in (RESULTS / "tables").glob("*.tex"):
        shutil.copy(f, PAPER / "tables" / f.name)
    print(f"wrote {len(out)} macros; copied {len(list(FIGURES.glob('*.pdf')))} figures")


if __name__ == "__main__":
    main()
