# ACPolyp · Active-contour-guided polyp classification on Kvasir

> A leakage-audited, shortcut-aware, statistically controlled study of whether an **unsupervised active-contour prior**
> improves deep classification of endoscopic findings (Kvasir v2), plus a calibrated, conformal, explainable inference stack
> and a local web app.

**Research prototype. Not a medical device. Kvasir is licensed for research and education only (see [docs/LICENSING.md](docs/LICENSING.md)).**

## What this project actually found

<!--RESULTS-START-->
**Headline: an unsupervised active-contour prior did *not* measurably improve a fine-tuned classifier.**

* ACG-ConvNeXt 0.9277 ± 0.0035 vs plain ConvNeXt-T 0.9233 ± 0.0006: Δ = +0.0045
  (95 % paired-bootstrap CI -0.0004 to +0.0091, Holm p = 0.30). Controls with the *wrong image's* contour
  (+0.0043) or a central ellipse (+0.0035) are indistinguishable. Replicating at a better learning rate does not change this.
* The contour is a weak localiser: Dice 0.406 (95 % CI 0.373–0.440) vs 0.375 for a trivial central ellipse.
* Kvasir v2 audit: 31 duplicates (one file filed under two classes), 117 test errors of which 53% involve suspect labels,
  resolution alone predicts upper-GI vs colon at 96.8% (majority 62.4%) — yet a naive-frame model scores the same as the canonical one
  (0.9248 vs 0.9233), while a random split inflates polyp AUROC to 0.9994.
* Calibration: ECE 0.046 → 0.033. Validation-calibrated conformal sets cover only 92.9% at a 95 % target (validation was reused for model selection);
  group-aware calibration gives 95.2%.

| Model | Macro-F1 | Accuracy | Polyp-vs-rest AUROC | ECE |
|---|---|---|---|---|
| ResNet-50 | 0.8708 ± 0.0043 | 0.8712 ± 0.0045 | 0.9916 ± 0.0017 | 0.0300 ± 0.0026 |
| EfficientNetV2-S | 0.9257 ± 0.0025 | 0.9258 ± 0.0025 | 0.9957 ± 0.0004 | 0.0256 ± 0.0035 |
| ConvNeXt-T | 0.9233 ± 0.0006 | 0.9235 ± 0.0006 | 0.9931 ± 0.0029 | 0.0353 ± 0.0027 |
| Swin-T | 0.9327 ± 0.0045 | 0.9327 ± 0.0045 | 0.9947 ± 0.0014 | 0.0306 ± 0.0050 |
| DINOv2 ViT-S/14 | 0.9344 ± 0.0058 | 0.9343 ± 0.0058 | 0.9916 ± 0.0015 | 0.0242 ± 0.0035 |
| ACG-ConvNeXt (full) | 0.9277 ± 0.0035 | 0.9279 ± 0.0035 | 0.9900 ± 0.0013 | 0.0325 ± 0.0016 |
| Linear probe · frozen DINOv2 | 0.8889 ± 0.0000 | 0.8890 | 0.9951 ± 0.0000 | 0.0354 |
| LightGBM · global + contour | 0.7954 ± 0.0045 | 0.7964 ± 0.0045 | 0.9716 ± 0.0006 | 0.1310 ± 0.0056 |
| Heuristic · nearest Lab-colour centroid | 0.4762 | 0.4793 | 0.8372 | 0.2669 |

Test split (1,594 scene-grouped frames), mean ± SD over 3 seeds (classical: 5). Full tables, statistics and figures: `outputs/results`, `outputs/figures`, `paper/main.pdf`, `report/report.pdf`.
<!--RESULTS-END-->

## Quick start

```powershell
# 1. environment (Python 3.12, NVIDIA GPU with CUDA 12.8 drivers)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# 2. everything: download -> audit -> contours -> baselines -> deep study -> analysis -> paper (~7-8 h on an RTX 4060 laptop)
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
#    smoke test (~15 min):  ... run_experiments.ps1 -Quick

# 3. web app (needs outputs/artifacts/acpolyp-v1, created by the study)
.\.venv\Scripts\python.exe -m acpolyp.cli serve         # http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m acpolyp.cli predict frame.jpg --cam cam.png
.\.venv\Scripts\python.exe -m acpolyp.cli evaluate --limit 150   # train/serve parity check
```

Use `python -I` and run from the repository root when touching downloaded data (untrusted input).

## Repository map

| Path | Content |
|---|---|
| `src/acpolyp/data` | verified download, ingestion, artefact-aware canonicalisation, de-duplication, scene-grouped splits, GPU augmentation |
| `src/acpolyp/ac` | morphological Chan–Vese + geodesic active contour, 136 features, tuning against Kvasir-SEG |
| `src/acpolyp/models` | ACGNet (+ ablation modes), backbone factory, frozen embeddings |
| `src/acpolyp/training` | engine, metrics, calibration, conformal, classical baselines |
| `src/acpolyp/evaluation` | paired statistics, aggregation, Grad-CAM, figure style |
| `src/acpolyp/app` | FastAPI service + single-page web UI |
| `scripts/` | experiment table, analysis, exporters, bundling, robustness, interpretability, master script |
| `tests/` | 48 automated tests |
| `docs/` | data audit, formulation and risk register, literature, licensing, 85-step compliance matrix |
| `paper/` | IEEE-format manuscript (LaTeX), auto-generated number macros, figures, tables |
| `report/` | full technical report (LaTeX) |
| `slides/` | presentation deck |

## Method in one paragraph

Frames are cropped to the endoscopic field of view and a canonical rectangle covering the ScopeGuide box is masked in *every* frame
(resolution and box otherwise leak the class). A label-free two-stage morphological active contour (Chan–Vese, optional geodesic refinement)
delineates a lesion candidate on a centre–surround Lab-PCA salience map; its hyper-parameters are tuned against Kvasir-SEG masks of
training-matched frames only. ACGNet pools backbone features inside/outside the contour and fuses 98 contour features. Controls (shuffled-image contours,
a central ellipse, non-contour features) separate "a mask" from "this image's contour". Temperature scaling, a validation-tuned polyp threshold and
split-conformal sets provide calibrated uncertainty.

## Reproducibility

Every run directory (`outputs/runs/<tag>/seed<k>`) stores its configuration, git commit, command, environment, per-epoch history, logits and summary.
Splits are hashed (`data/processed/curation_report.json`). Bit-exact replay is not guaranteed (a few CUDA kernels are non-deterministic); seed-to-seed spread is reported.

## Authors

P. Vishnu Vardhan Reddy, V. Guru Puneeth Reddy, D. Greeshmanth Reddy: Dept. of Computer Science and Engineering,
Amrita Vishwa Vidyapeetham, Bangalore Campus. Repository: https://github.com/pvishnu318/acpolyp. Cite via [CITATION.cff](CITATION.cff).

## Citation and data credit

Pogorelov et al., *KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection*, MMSys 2017, doi:10.1145/3083187.3083212 ·
Jha et al., *Kvasir-SEG*, MMM 2020, doi:10.1007/978-3-030-37734-2_37.

## Licence

Code: MIT. Datasets: research and educational use only (Simula); the datasets are not redistributed (a few frames appear as attributed illustrative figure panels). Pretrained weights: Apache-2.0 / MIT (checked).
