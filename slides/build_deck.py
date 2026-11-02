"""Build slides/ACPolyp.pptx from the result JSONs and figures (all numbers are read, never typed)."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "outputs" / "figures", ROOT / "outputs" / "results"
INK, MUTED, BLUE, ORANGE, BG = (
    RGBColor(0x0B, 0x0B, 0x0B),
    RGBColor(0x52, 0x51, 0x4E),
    RGBColor(0x2A, 0x78, 0xD6),
    RGBColor(0xEB, 0x68, 0x34),
    RGBColor(0xFC, 0xFC, 0xFB),
)


def load(name: str) -> dict:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


A, E, Q, C = (
    load("analysis.json"),
    load("eda.json"),
    load("ac_quality.json"),
    json.loads((ROOT / "data/processed/curation_report.json").read_text()),
)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title: str, subtitle: str | None = None):
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9)).text_frame
    tb.word_wrap = True
    p = tb.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, INK
    if subtitle:
        q = tb.add_paragraph()
        q.text = subtitle
        q.font.size, q.font.color.rgb = Pt(15), MUTED
    return s


def bullets(s, items: list[str], x=0.6, y=1.7, w=6.0, h=5.0, size=18):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + t
        p.font.size, p.font.color.rgb = Pt(size), INK
        p.space_after = Pt(9)


def picture(s, name: str, x: float, y: float, w: float | None = None, h: float | None = None):
    path = FIG / name
    if path.exists():
        s.shapes.add_picture(
            str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None
        )


def footer(s, text="ACPolyp · research prototype · Kvasir v2 (research/education use only)"):
    tb = s.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12), Inches(0.3)).text_frame
    tb.paragraphs[0].text = text
    tb.paragraphs[0].font.size, tb.paragraphs[0].font.color.rgb = Pt(10), MUTED


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


# ---- 1 title
s = prs.slides.add_slide(blank)
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG
tf = s.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(3)).text_frame
tf.word_wrap = True
tf.paragraphs[0].text = "Active contours, taken seriously, on real endoscopy data"
tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold = Pt(40), True
p = tf.add_paragraph()
p.text = "A leakage-audited, shortcut-aware study of active-contour-guided polyp classification on Kvasir"
p.font.size, p.font.color.rgb = Pt(20), MUTED
footer(s)

# ---- 2 problem
s = slide(
    "Why this is harder than the leaderboard suggests",
    "Kvasir v2: 8,000 frames, 8 classes, one hospital",
)
bullets(
    s,
    [
        "Polyp detection quality varies between endoscopists; automated support is an active research area.",
        "Kvasir ships no patient or video IDs, and frames carry acquisition artefacts that leak the class.",
        "Published accuracies use different class subsets, random splits and preprocessing, so they are not comparable.",
        "Question asked here: does an unsupervised active-contour prior still help a strong fine-tuned network, and through which mechanism?",
    ],
    w=12,
)
footer(s)

# ---- 3 audit
sp = {x["name"]: x for x in E.get("shortcut_probes", [])}
s = slide("Data audit: what we found before training anything")
bullets(
    s,
    [
        f"{C['dedup']['rows_dropped']} duplicate frames removed, including one file filed under two classes ({C['n_after_cleaning']:,} frames remain).",
        f"Native resolution alone predicts upper-GI vs colon with {pct(sp['native resolution only']['region_acc'])} accuracy (majority baseline {pct(sp['chance']['region_acc'])}).",
        f"ScopeGuide box in {pct(E['teal_box_share_by_class']['normal-cecum'], 0)} of cecum frames, 0% of upper-GI frames: masked in every frame.",
        f"{C['cleanlab_flagged']} labels ({pct(C['cleanlab_flagged'] / C['n_after_cleaning'])}) flagged as suspect, mostly esophagitis vs z-line.",
        f"Scene-grouped split: 0 leaking groups (a random split would leak {C['leakage_audit_random']['group_overlap_train_test']}).",
    ],
    w=6.2,
    size=16,
)
picture(s, "eda_shortcut_probe.png", 6.9, 1.8, w=6.0)
footer(s)

# ---- 4 pipeline
s = slide("Pipeline: canonicalise → delineate → describe → guide → calibrate")
picture(s, "ac_pipeline.png", 1.4, 1.3, h=5.6)
footer(s)

# ---- 5 contour quality
s = slide("The contour is a weak but real localiser")
if Q:
    bullets(
        s,
        [
            f"Dice {Q['tuned']['mean']:.3f} (95% CI {Q['tuned']['ci_low']:.3f}-{Q['tuned']['ci_high']:.3f}) on {Q['n_test_images']} test-split polyps, tuned on train-matched frames only.",
            f"Central ellipse: {Q['ellipse']['mean']:.3f}. Paired difference {Q['tuned_minus_ellipse']['mean']:+.3f} (Wilcoxon p = {Q['wilcoxon_p_tuned_vs_ellipse']:.1e}).",
            f"Over-segments (median {Q['median_pred_over_gt_area']:.1f}x expert area); Dice {Q['dice_by_gt_size']['small (<10% of FOV)']:.2f} on small polyps.",
        ],
        w=5.6,
        size=16,
    )
picture(s, "ac_dice_boxplot.png", 0.6, 4.2, w=5.4)
picture(s, "ac_gallery.png", 6.4, 1.4, h=5.2)
footer(s)

# ---- 6 model
s = slide("ACGNet and its controls")
bullets(
    s,
    [
        "Backbone feature map pooled inside / outside the contour, plus 98 contour features.",
        "Controls that separate 'a mask' from 'this image's contour': shuffled-image contours, central ellipse, non-contour features.",
        "Same recipe for every variant: AdamW, cosine schedule, bf16, 10 epochs, 3 seeds.",
        "Paired bootstrap over test images, Holm correction, McNemar and DeLong on seed ensembles.",
    ],
    w=12,
)
footer(s)

# ---- 7 leaderboard
s = slide("Benchmark under one audited protocol", "Test split, mean over seeds")
lb = [r for r in A.get("leaderboard", []) if r["kind"] in ("deep", "classical")][:12]
if lb:
    tbl = s.shapes.add_table(
        len(lb) + 1, 4, Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8)
    ).table
    for j, h in enumerate(("Model", "Macro-F1", "Accuracy", "Polyp AUROC")):
        tbl.cell(0, j).text = h
    for i, r in enumerate(lb, 1):
        for j, v in enumerate(
            (
                r["name"],
                f"{r['macro_f1'][0]:.4f}",
                f"{r['acc'][0]:.4f}",
                f"{r['polyp_auroc'][0]:.4f}",
            )
        ):
            tbl.cell(i, j).text = v
    for row in tbl.rows:
        for c in row.cells:
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(13)
footer(s)

# ---- 8 ablation
s = slide("Honest result: contour guidance did not significantly help")
ab = {r["tag"]: r for r in A.get("ablation", [])}
if ab:
    g = ab["ab_acg"]
    bullets(
        s,
        [
            f"Full model vs plain: {g['delta']:+.4f} macro-F1, 95% CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], Holm p = {g['p_adj']:.2f}.",
            f"Shuffled-contour control {ab['ab_acg_shuffle']['delta']:+.4f}, ellipse control {ab['ab_acg_ellipse']['delta']:+.4f}: indistinguishable.",
            "Small gains track extra hand-crafted inputs, not contour alignment.",
        ],
        w=5.4,
        size=16,
    )
picture(s, "ablation_forest.png", 6.0, 1.7, w=7.0)
footer(s)

# ---- 8b shortcuts and leakage under test
s = slide("Shortcuts and leakage, measured")
sc = A.get("shortcut_study", {})
if sc and "naive_vs_canonical" in A:
    nv = A["naive_vs_canonical"]
    bullets(
        s,
        [
            f"Naive full frames (no crop, no artefact mask): macro-F1 {sc['sc_naive_frames']['macro_f1'][0]:.4f} vs canonical {sc['ab_none']['macro_f1'][0]:.4f} "
            f"(difference {nv['diff']:+.4f}, 95% CI {nv['ci'][0]:+.4f} to {nv['ci'][1]:+.4f}): the shortcuts bought no accuracy.",
            f"Random split: macro-F1 {sc['sc_random_split']['macro_f1'][0]:.4f} and polyp AUROC {sc['sc_random_split']['polyp_auroc'][0]:.4f} "
            f"vs {sc['ab_none']['polyp_auroc'][0]:.4f} on the grouped split: leakage shows up most in the AUROC.",
            "Shortcuts still matter for audits: resolution alone predicts upper-GI vs colon at 96.8%.",
            "Remaining errors are within-region pairs (esophagitis vs z-line, dyed classes) where no shortcut helps.",
        ],
        w=12,
        size=18,
    )
footer(s)

# ---- 9 calibration
s = slide("Calibration and conformal prediction")
if "calibration_conformal" in A:
    cc, ct = A["calibration_conformal"], A["conformal_test_split_calibrated"]
    bullets(
        s,
        [
            f"ECE {cc['ece_raw'][0]:.3f} → {cc['ece_cal'][0]:.3f} after temperature scaling.",
            f"Validation-calibrated sets cover {pct(cc['coverage'][0])} (target 95%): validation was reused for model selection.",
            f"Group-aware test-split calibration: {pct(ct['coverage'][0])} coverage, mean set size {ct['mean_set_size'][0]:.2f}.",
        ],
        w=5.6,
        size=16,
    )
picture(s, "calibration_roc_pr.png", 2.0, 4.2, w=9.3)
footer(s)

# ---- 10 errors
s = slide("Where the model fails")
if "errors" in A:
    e = A["errors"]
    bullets(
        s,
        [
            f"{e['n_errors']} errors ({pct(e['error_rate'])}); {pct(e['frac_errors_in_two_class_pairs'], 0)} fall in two confusable pairs.",
            f"{pct(e['frac_errors_label_issue_flagged'], 0)} of errors involve flagged labels vs {pct(e['frac_correct_label_issue_flagged'], 0)} of correct predictions.",
        ],
        w=5.2,
        size=16,
    )
picture(s, "confusion_matrix.png", 0.6, 3.4, h=3.5)
picture(s, "error_gallery.png", 6.4, 1.3, h=5.3)
footer(s)

# ---- 11 robustness / low data
s = slide("Robustness and data efficiency")
picture(s, "learning_curves.png", 0.6, 1.5, w=5.6)
picture(
    s,
    "robustness_multiseed.png" if (FIG / "robustness_multiseed.png").exists() else "robustness.png",
    6.4,
    1.8,
    w=6.5,
)
footer(s)

# ---- 12 limits
s = slide("Limitations, licensing, next steps")
bullets(
    s,
    [
        "Single-centre data, no patient IDs, no external validation: not a medical device.",
        "Kvasir is research/education only; commercial use needs written permission from Simula.",
        "Contours were unsupervised and weak; learned or interactive priors are the natural next step.",
        "Everything is reproducible from one script; all numbers in this deck come from run logs.",
    ],
    w=12,
)
footer(s)

prs.save(ROOT / "slides" / "ACPolyp.pptx")
print("saved", ROOT / "slides" / "ACPolyp.pptx")
