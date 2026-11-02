"""Fill the results block of README.md from the analysis JSON (between the RESULTS markers)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"


def load(n: str) -> dict:
    return json.loads((RES / n).read_text(encoding="utf-8"))


def pm(v: list[float], d: int = 4) -> str:
    return f"{v[0]:.{d}f} ± {v[1]:.{d}f}" if v[1] > 0 else f"{v[0]:.{d}f}"


def main() -> None:
    A, Q, E = load("analysis.json"), load("ac_quality.json"), load("eda.json")
    ab = {r["tag"]: r for r in A["ablation"]}
    g, sh, el = ab["ab_acg"], ab["ab_acg_shuffle"], ab["ab_acg_ellipse"]
    lb = {r["name"]: r for r in A["leaderboard"]}
    cc, ct = A["calibration_conformal"], A["conformal_test_split_calibrated"]
    sc, er = A["shortcut_study"], A["errors"]
    sp = {s["name"]: s for s in E["shortcut_probes"]}
    rows = [
        "ResNet-50",
        "EfficientNetV2-S",
        "ConvNeXt-T",
        "Swin-T",
        "DINOv2 ViT-S/14",
        "ACG-ConvNeXt (full)",
        "Linear probe · frozen DINOv2",
        "LightGBM · global + contour",
        "Heuristic · nearest Lab-colour centroid",
    ]
    table = ["| Model | Macro-F1 | Accuracy | Polyp-vs-rest AUROC | ECE |", "|---|---|---|---|---|"]
    for n in rows:
        r = lb[n]
        table.append(
            f"| {n} | {pm(r['macro_f1'])} | {pm(r['acc'])} | {pm(r['polyp_auroc'])} | {pm(r['ece'])} |"
        )
    md = f"""**Headline: an unsupervised active-contour prior did *not* measurably improve a fine-tuned classifier.**

* ACG-ConvNeXt {pm(g["macro_f1"])} vs plain ConvNeXt-T {pm(ab["ab_none"]["macro_f1"])}: Δ = {g["delta"]:+.4f}
  (95 % paired-bootstrap CI {g["ci"][0]:+.4f} to {g["ci"][1]:+.4f}, Holm p = {g["p_adj"]:.2f}). Controls with the *wrong image's* contour
  ({sh["delta"]:+.4f}) or a central ellipse ({el["delta"]:+.4f}) are indistinguishable. Replicating at a better learning rate does not change this.
* The contour is a weak localiser: Dice {Q["tuned"]["mean"]:.3f} (95 % CI {Q["tuned"]["ci_low"]:.3f}–{Q["tuned"]["ci_high"]:.3f}) vs {Q["ellipse"]["mean"]:.3f} for a trivial central ellipse.
* Kvasir v2 audit: 31 duplicates (one file filed under two classes), {A["errors"]["n_errors"]} test errors of which {er["frac_errors_label_issue_flagged"]:.0%} involve suspect labels,
  resolution alone predicts upper-GI vs colon at {sp["native resolution only"]["region_acc"]:.1%} (majority {sp["chance"]["region_acc"]:.1%}) — yet a naive-frame model scores the same as the canonical one
  ({sc["sc_naive_frames"]["macro_f1"][0]:.4f} vs {sc["ab_none"]["macro_f1"][0]:.4f}), while a random split inflates polyp AUROC to {sc["sc_random_split"]["polyp_auroc"][0]:.4f}.
* Calibration: ECE {cc["ece_raw"][0]:.3f} → {cc["ece_cal"][0]:.3f}. Validation-calibrated conformal sets cover only {cc["coverage"][0]:.1%} at a 95 % target (validation was reused for model selection);
  group-aware calibration gives {ct["coverage"][0]:.1%}.

{chr(10).join(table)}

Test split (1,594 scene-grouped frames), mean ± SD over 3 seeds (classical: 5). Full tables, statistics and figures: `outputs/results`, `outputs/figures`, `paper/main.pdf`, `report/report.pdf`.
"""
    p = ROOT / "README.md"
    s = p.read_text(encoding="utf-8")
    s = re.sub(
        r"<!--RESULTS-START-->.*<!--RESULTS-END-->",
        lambda _: "<!--RESULTS-START-->\n" + md + "<!--RESULTS-END-->",
        s,
        flags=re.S,
    )
    p.write_text(s, encoding="utf-8")
    print("README updated")


if __name__ == "__main__":
    main()
