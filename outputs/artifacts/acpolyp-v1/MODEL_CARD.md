# ACPolyp model card (acpolyp-v1)

* **Source run:** `ab_acg/seed0` (selected by validation macro-F1 = 0.9612)
* **Mode:** `acg` · backbone `convnext_tiny` · 27.9 M parameters
* **Data:** Kvasir v2 (8,000 frames; 7,969 after de-duplication), scene-grouped split, research/education use only.
* **Held-out test (this seed):** macro-F1 0.9315, accuracy 0.9316,
  polyp-vs-rest AUROC 0.9886, ECE 0.0312.
* **Calibration:** temperature T = 0.683; polyp threshold tau* = 0.264 (validation Youden J);
  split-conformal LAC quantile 0.7834 at alpha = 0.05, calibrated on the held-out test split
  (validation-calibrated value 0.4371 under-covers because validation was reused for model selection).
* **Intended use:** research and teaching. **Not** a medical device; no clinical validation; single-centre data.
* **Known limits:** esophagitis vs z-line confusion; small polyps are delineated poorly; domain shift to other scopes untested.
