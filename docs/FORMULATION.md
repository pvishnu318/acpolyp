# Problem formulation, hypotheses and design (Phases I–III, XI)

## 1. Domain and why a learned model

Colorectal polyps are mucosal outgrowths, some of which become cancer; detecting and characterising them during
colonoscopy is a quality indicator, and endoscopists differ in detection rate. A rule such as "red blob with a convex
boundary" fails because polyps are flat, pedunculated or dyed, mucosa varies in colour, and specular highlights, folds
and fluid create convex red blobs everywhere. A hand-written colour rule (nearest Lab-colour centroid) reaches macro-F1 0.476 and polyp AUROC 0.837 on the
test split (chance: 0.125 / 0.5), while fine-tuned networks exceed 0.9 and 0.99, so a learnable statistical pattern
exists and a deterministic rule is inadequate.

## 2. Learning problem

* Paradigm: supervised, single-label, 8-way classification (Kvasir v2 classes).
  Inputs `x ∈ R^{3×H×W}` (endoscopic frame) → `y ∈ {1..8}`; model `f_θ : X → Δ^7` (probability simplex).
* **Polyp view.** The clinical question "is there a polyp?" is read off the same posterior:
  `s(x) = P(y = polyps | x)`, thresholded at `τ*` chosen on validation data. This is a one-vs-rest task with
  a 1 : 7 imbalance (993 polyp vs 6,976 other frames), and the *hard negatives* are the other colon classes
  (dyed lifted polyps, dyed resection margins, ulcerative colitis, normal cecum).
* Note on naming: the dataset's "polyps" class is one of 8 classes. This project therefore reports both the 8-class
  and the polyp-vs-rest results, and never calls the 8-class score "polyp accuracy".

## 3. Objectives, research questions, hypotheses

**Primary scientific objective.** Quantify whether an *unsupervised* active-contour region prior improves
classification of endoscopic findings beyond a strong fine-tuned backbone, and attribute any effect to a mechanism.

| RQ | Question | H0 / H1 | Evidence |
|---|---|---|---|
| RQ1 | Does contour guidance improve macro-F1 over plain ConvNeXt-T? | H0: Δ = 0 / H1: Δ ≠ 0 | paired bootstrap CI, seed-level paired t-test, McNemar (seed-ensemble), DeLong (polyp AUROC); Holm over the family of ablations |
| RQ2 | Is any gain due to the *image-specific* contour, to extra hand-crafted features, or to any mask? | H0: shuffled-contour and ellipse controls equal the full model | `ab_acg_shuffle`, `ab_acg_ellipse`, `ab_gfeat` |
| RQ3 | How much apparent accuracy is attributable to acquisition shortcuts and split leakage? | – | resolution/overlay probes, naive-frame run, random-split run |
| RQ4 | Does guidance help more with little training data? | – | learning curves at 10/25/50/100 % |
| RQ5 | Are the probabilities calibrated, and do conformal sets attain nominal coverage? | H0: coverage ≥ 1−α | ECE, reliability, split-conformal coverage |

**Secondary engineering constraints.** Training on one 8 GB GPU; ≤ 25 ms/image batch-1 inference; reproducible from
one command; the contour stage unit-tested and label-free.

We pre-specify that a **null or negative result for RQ1 is an acceptable outcome**. The fallback plan (risk R3) is to
report it, keep the audit, calibration and shortcut findings as the contribution, and say so in the abstract.

## 4. Metric taxonomy

| Role | Metric |
|---|---|
| Primary (8-class) | **macro-F1** (balanced classes, so macro-F1 ≈ accuracy; macro-F1 is used because it is the standard for imbalanced follow-ups) |
| Primary (polyp view) | **AUROC** of `P(polyps)`; AUPRC because of the 1 : 7 imbalance |
| Diagnostic | accuracy, balanced accuracy, MCC, per-class P/R/F1, sensitivity/specificity/PPV at `τ*`, NLL, Brier, **ECE** (15 bins) |
| Uncertainty | conformal coverage and mean set size at α = 0.05, risk–coverage curve |
| Efficiency | parameter count, peak VRAM, train time, batch-1 and batch-64 latency (after warm-up, with `cuda.synchronize`), checkpoint size |
| Contour quality | Dice/IoU vs Kvasir-SEG masks (test-split matches only) |

*Optional asymmetric cost.* For screening, a missed polyp (FN) is costlier than a false alarm. We report
expected cost per image under `C_FN : C_FP = 5 : 1` for the val-tuned threshold and the cost-optimal threshold.

## 5. Feasibility audit

| Resource | Finding |
|---|---|
| Data | 8,000 labelled frames + 1,000 masked polyps, openly downloadable; ≈ 1,000 per class is adequate for fine-tuning pretrained backbones (verified empirically by the learning curves) |
| Hardware | RTX 4060 Laptop, 8 GB VRAM, i7-13650HX (14 cores), 23.7 GB RAM, 195 GB free disk |
| Cost | ConvNeXt-T at 224 px, batch 32, bf16: ≈ 25 s / epoch, ≈ 2.5 GB peak VRAM, ≈ 6 min / run incl. start-up |
| Budget | ≈ 70 deep runs × ≈ 5 min ≈ 6 h when the GPU is idle (it was shared with another job for part of the study, which multiplied wall time) + ≈ 1 h classical + ≈ 1 h analysis |
| Caveat | Windows needs spawned DataLoader workers: datasets must pickle as file paths (memory-mapped caches), otherwise spawn fails on 1.5 GB arrays (observed and fixed) |

## 6. Scope

* **In scope:** Kvasir v2 (8 classes), Kvasir-SEG masks as an auxiliary benchmark for the contour; CNN and ViT
  backbones from timm; classical ML baselines; calibration; conformal prediction; explainability; a local demo.
* **Out of scope / non-goals:** detection or segmentation as a product; video; real-time deployment; external
  clinical validation; regulatory claims; HyperKvasir (overlaps with Kvasir frames, so it is not an independent test).
* **Distribution assumption at inference:** white-light or chromo-endoscopy frames from Olympus-style systems similar to
  those of Bærum Hospital 2008–2016.
* **MVM vs stretch.** MVM: data audit + canonicalisation + baselines + guided model + ablation. Stretch: low-data
  curves, robustness curves, layer-wise probes, ensemble, web app.

## 7. Risk register (Phase III, step 12)

| ID | Risk | P | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R1 | Download truncation / corruption | M | H | byte-count + SHA-256 + zip test, resumable Range downloader | **occurred** (771 MB of 2.49 GB), caught and resumed |
| R2 | Label contradiction or duplicates | M | H | SHA + embedding dedup, unique-ID data contract | **occurred** (1 cross-class + 30 pairs) |
| R3 | Contour guidance gives no gain (null result) | **H** | M | controls (shuffle, ellipse, gfeat), pre-specified acceptance of null, pivot to audit/calibration contribution | see results |
| R4 | Shortcut learning (resolution, overlay box) | H | H | canonicalisation, naive-vs-canonical experiment | **occurred**, mitigated |
| R5 | Split leakage via near-identical frames | M | H | scene-group split, random-split control | mitigated, effect small (≈ 3 %) |
| R6 | OOM / VRAM | L | M | bf16, batch 32, memory-mapped data | no OOM; **commit-memory exhaustion** on Windows seen once with parallel jobs → serialised heavy jobs |
| R7 | Silent config coercion (`1e-3` parsed as string by YAML 1.1) | M | M | override parser + unit test | **occurred in unit test**, fixed |
| R8 | DeLong/bootstrap implementation bug | M | H | unit tests vs scikit-learn and hand cases | **occurred** (AUC mismatch), fixed |
| R9 | Over-claiming | H | H | every number generated from run logs; limitations section; CIs | ongoing |
| R10 | Commercial misuse of a research-only dataset | M | H | licence audit, no redistribution | documented in `LICENSING.md` |

## 8. Mathematical specification

### 8.1 Active-contour stage (label-free)

Let `I` be a scalar salience image on the valid-FOV domain `Ω`. The Chan–Vese energy
(Chan & Vese, 2001) of a closed curve `C` is

`F(C, c₁, c₂) = μ·Length(C) + λ₁ ∫_{in(C)} (I − c₁)² dx + λ₂ ∫_{out(C)} (I − c₂)² dx`,

with `c₁, c₂` the mean intensities inside/outside. We minimise it with the **morphological** scheme of
Márquez-Neila et al. (2014): the level set is a binary function `u`; each iteration (i) sets `u(x) ← 1` where
`λ₁(I − c₁)² < λ₂(I − c₂)²` (data term) and (ii) applies the curvature-smoothing operator `SI∘IS` repeated
`smoothing` times (length term), with no re-initialisation. An optional second stage minimises the geodesic energy
`∫ g(|∇(G_σ*I)|) ds`, `g = 1/√(1 + α|∇(G_σ*I)|)`, via morphological geodesic active contours.

*Salience.* `S = PC₁(Lab) − G_{σ_w} * PC₁(Lab)` (centre–surround on the first Lab principal component of the valid
pixels, `σ_w = dog_frac · size`), percentile-normalised to [0,1]; invalid pixels get the valid mean. The polarity of
the two-phase result is resolved by choosing the region that touches the FOV border least. A central-ellipse fallback
fires when the area is outside [2 %, 85 %] of the FOV.

### 8.2 ACGNet

`F = backbone(x) ∈ R^{C×h×w}`, `V` the valid mask and `M` the contour mask, both area-downsampled to `h×w`:

`g = LN( Σ V·F / Σ V )`, `f_in = LN( Σ M·F / (Σ M + ε) )`, `f_out = LN( Σ (V − M)₊·F / (Σ (V − M)₊ + ε) )`,
`e = GELU(W₂ GELU(W₁ s))`, `z = W [Dropout([g; f_in; f_out; e])]`, `p = softmax(z)`,

where `s ∈ R^{98}` are the standardised contour features (shape, boundary, region). Regions with mass < 0.05
fall back to `g`. Ablation modes drop `[f_in; f_out]`, `e` or both. **Inductive bias:** classification evidence
inside a delineated candidate and its surrounding mucosa should be evaluated separately, the way an endoscopist
compares a lesion with its background, and shape/boundary statistics are cheap, interpretable cues.

### 8.3 Loss and regularisation

`L = CE_{ε=0.1}(softmax(z), y)` (label smoothing 0.1) with AdamW (weight decay 0.05, none on biases/norm
parameters), linear warm-up + cosine decay, gradient clipping at 1.0, stochastic depth 0.1, head dropout 0.2.
Backbone learning rate = 0.25 × head learning rate. Post-hoc: temperature scaling `softmax(z/T)` with `T` fit by
minimum validation NLL; split-conformal LAC sets `{k : 1 − p_k ≤ q̂}` with `q̂` the `⌈(n+1)(1−α)⌉/n` quantile
of validation non-conformity scores `1 − p_y`.

### 8.4 Parameters and memory

ConvNeXt-T 27.83 M parameters in total (measured); the guided variant has 27.87 M (+0.04 M); all parameters trainable (full fine-tuning).
Peak VRAM ≈ 2.5 GB at batch 32 with bf16 autocast (measured).
