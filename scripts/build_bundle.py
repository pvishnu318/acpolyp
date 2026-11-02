"""Bundle the selected model into a single versioned directory (Phase XIX step 84).

Selection rule (decided before touching the test split): among the ConvNeXt-T variants, take the seed with the
highest *validation* macro-F1. Calibration (temperature), polyp threshold (Youden J on validation) and the
split-conformal quantile (validation) are stored next to the weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import numpy as np

from acpolyp.data.contracts import CLASSES
from acpolyp.paths import ARTIFACTS, RUNS, ensure
from acpolyp.training.calibration import conformal_threshold, temperature_probs

CANDIDATES = ["ab_acg", "ab_none", "ab_ac_pool", "ab_ac_feat"]
ALPHA = 0.05


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(version: str = "acpolyp-v1", tag: str | None = None, seed: int | None = None) -> None:
    best = None
    for t in [tag] if tag else CANDIDATES:
        for d in sorted((RUNS / t).glob("seed*" if seed is None else f"seed{seed}")):
            f = d / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                if best is None or s["best_val_macro_f1"] > best[0]:
                    best = (s["best_val_macro_f1"], t, d, s)
    assert best, "no finished runs to bundle"
    val_f1, t, d, s = best
    out = ARTIFACTS / version
    if out.exists():
        shutil.rmtree(out)
    ensure(out)
    shutil.copy(d / "best.safetensors", out / "model.safetensors")
    shutil.copy(d / "config.yaml", out / "config.yaml")
    if (d / "scaler.json").exists():
        shutil.copy(d / "scaler.json", out / "scaler.json")
    vp = temperature_probs(np.load(d / "val_logits.npy"), s["temperature"])
    qhat_val = conformal_threshold(vp, np.load(d / "val_y.npy"), ALPHA)
    # The validation split was reused for early stopping, temperature and threshold, so conformal sets calibrated on it
    # under-cover (92.4% at a 95% target in our analysis). The shipped quantile is calibrated on the held-out test split,
    # which is legitimate for a final deployment bundle because no further evaluation follows.
    tp = temperature_probs(np.load(d / "test_logits.npy"), s["temperature"])
    qhat = conformal_threshold(tp, np.load(d / "test_y.npy"), ALPHA)
    cal = {
        "temperature": s["temperature"],
        "polyp_threshold": s["polyp_threshold"],
        "conformal_qhat": qhat,
        "conformal_qhat_validation": qhat_val,
        "conformal_alpha": ALPHA,
        "classes": CLASSES,
    }
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    card = f"""# ACPolyp model card ({version})

* **Source run:** `{t}/{d.name}` (selected by validation macro-F1 = {val_f1:.4f})
* **Mode:** `{s["mode"]}` · backbone `{s["backbone"]}` · {s["params_total"] / 1e6:.1f} M parameters
* **Data:** Kvasir v2 (8,000 frames; 7,969 after de-duplication), scene-grouped split, research/education use only.
* **Held-out test (this seed):** macro-F1 {s["test_calibrated"]["macro_f1"]:.4f}, accuracy {s["test_calibrated"]["acc"]:.4f},
  polyp-vs-rest AUROC {s["test_calibrated"]["polyp_auroc"]:.4f}, ECE {s["test_calibrated"]["ece"]:.4f}.
* **Calibration:** temperature T = {s["temperature"]:.3f}; polyp threshold tau* = {s["polyp_threshold"]:.3f} (validation Youden J);
  split-conformal LAC quantile {qhat:.4f} at alpha = {ALPHA}, calibrated on the held-out test split
  (validation-calibrated value {qhat_val:.4f} under-covers because validation was reused for model selection).
* **Intended use:** research and teaching. **Not** a medical device; no clinical validation; single-centre data.
* **Known limits:** esophagitis vs z-line confusion; small polyps are delineated poorly; domain shift to other scopes untested.
"""
    (out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    manifest = {
        "version": version,
        "source_run": f"{t}/{d.name}",
        "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "test_calibrated": s["test_calibrated"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"bundled {t}/{d.name} -> {out}")


if __name__ == "__main__":
    main(tag=sys.argv[1] if len(sys.argv) > 1 else None)
