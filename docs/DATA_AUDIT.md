# Data audit: Kvasir v2

Everything below is reproduced by `python -m acpolyp.data.ingest`, `python -m acpolyp.data.curate` and
`python scripts/eda.py`; raw numbers live in `data/processed/curation_report.json` and
`outputs/results/eda.json`.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Source | <https://datasets.simula.no/kvasir/> (Kvasir v2), downloaded 2026-11-02 |
| Archive | `kvasir-dataset-v2.zip`, 2,489,312,085 bytes (matches the server's `Content-Length`), SHA-256 `1c7129a3…d297fc4` |
| Archive test | `zipfile.testzip()` clean; extraction refuses absolute paths and `..` (zip-slip) |
| Frames | 8,000 JPEG, exactly 8 classes × 1,000 |
| Companion | Kvasir-SEG, 1,000 polyp frames with expert masks (46.2 MB, SHA-256 `03b30e21…153478f7`) |
| Download incident | The first transfer was silently truncated at 771 MB (server closed the connection; `curl` exit status was swallowed by a pipe). It was caught by the size check and resumed with HTTP Range requests. The downloader in `data/download.py` now loops until the byte count matches. |

## 2. Defects found, and what we did

| # | Finding | Evidence | Action |
|---|---|---|---|
| 1 | **Contradictory duplicate**: one file is byte-identical in `dyed-lifted-polyps` and `dyed-resection-margins` | SHA-256 + unique-ID contract failed on first ingest | both copies dropped |
| 2 | 11 further byte-identical within-class pairs; embedding duplicates (DINOv2 cosine ≥ 0.98) bring the total to 30 pairs | `curate.find_duplicates` | keep first, drop the rest. **31 rows removed, 7,969 remain** |
| 3 | **Resolution leaks the label.** Colon classes are ≈ 97–100 % 720×576; esophagitis, pylorus and z-line are ≈ 90–97 % 1280×1024 or 1920×1072 | `eda_artefacts` | FOV crop + square resize removes the native size from the pixels. A gradient-boosted classifier on resolution alone separates upper-GI from colon with **96.8 %** accuracy (majority baseline 62.4 %); with FOV geometry 99.3 % |
| 4 | **ScopeGuide picture-in-picture box** present in 95 % of cecum, ≈ 52 % of polyp, ≈ 53–54 % of dyed, 33 % of colitis and 0 % of upper-GI frames | `teal_box_share_by_class` | box detected (flat teal, G≈B, filled rectangle, lower-left) **and** its canonical rectangle masked in *every* frame so the mask itself carries no information |
| 5 | A first version of the teal detector also painted over blue indigo-carmine dye (a real lesion cue) | visual check of 8 dyed frames | G−B window + flatness test + placement; re-verified |
| 6 | **Burned-in text overlays** in the left margin of many frames | visual | no EXIF PII in any file (0/8000 carry the PII tags we scan for). Overlay text is not removed from inside the FOV; we report this as a limitation and test it indirectly through the naive-frame experiment |
| 7 | **No patient or video IDs.** Consecutive frames of one procedure cannot be linked directly | dataset documentation | **scene groups**: single-linkage on DINOv2 cosine ≥ 0.92 (0.90 chains into a 546-image blob; 0.92 gives 7,812 groups, largest 7). Groups are never split across train/val/test |
| 8 | pHash is useless here: the shared octagonal black border dominates low-frequency bits (at Hamming ≤ 2, 63 of 134 pairs were cross-class) | `near.py` scratch analysis | pHash only used for exact matching; semantic embeddings used for near-duplicates |
| 9 | **Label noise**: `cleanlab` confident learning flags **550 / 7,969 (6.9 %)** images, concentrated in esophagitis (159), z-line (121), dyed-lifted (92) and dyed-resection (78) | grouped out-of-fold logistic regression on DINOv2 | not removed (that would tune the data to the test); flagged in metadata and used in error analysis. Clinically plausible: esophagitis is diagnosed *at* the z-line |

## 3. Leakage audit

| Split | train∩test groups | train∩val | val∩test | SHA overlap | share of test images with a train twin at cos ≥ 0.92 |
|---|---|---|---|---|---|
| **grouped (used)** | **0** | **0** | **0** | **0** | 0.0 % |
| naive stratified random (control) | 41 | 15 | 8 | 0 | 2.7 % |

Near-duplicate leakage in Kvasir v2 is therefore *modest* (about 3 % of a randomly drawn test set), which is
the honest size of that effect. The larger shortcut is acquisition artefacts (items 3 and 4), measured in the
`sc_naive_frames` experiment.

Split sizes: train 5,578 · val 797 · test 1,594; per-class counts within ±1 of each other. Split hashes (SHA-256 of
the sorted ID lists) are stored in `curation_report.json`.

## 4. Distributions and collinearity

* Mean luma is approximately normal (Shapiro–Wilk p = 0.16, skew 0.01); Laplacian variance is right-skewed
  (skew 1.36, p ≈ 5·10⁻¹⁹) and is analysed on a log scale; FOV fraction is not normal (KS p ≈ 10⁻²⁴⁹).
* Class balance is perfect (max/min = 1.007) for the 8-class task; for polyp-vs-rest it is 1 : 7.0 (993 vs 6,976).
* The 98 contour-derived features are heavily collinear: median VIF 220, 79 features with VIF > 10, 25 pairs with
  |Spearman ρ| > 0.9. Consequence: tree models and L2-regularised linear models are used; coefficients are never
  interpreted individually, SHAP is used instead.
* Most informative contour features by mutual information with the class: `in_b_mean`, `out_b_mean`, `out_S_mean`
  (colour of the region inside / outside the contour), i.e. the model benefits from *where* colour statistics are measured.

## 5. What this audit does not establish

* That the grouped split is free of patient overlap. Without IDs this is unknowable; scene groups are a best-effort guard.
* That overlay text inside the FOV is harmless.
* That results transfer beyond this hospital, scope vendor or era (2008–2016, one centre).
