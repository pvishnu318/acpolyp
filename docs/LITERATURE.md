# Literature, benchmark and codebase survey (Phase II, steps 7–10)

## Search log (what was and was not searched)

Searches were run on 2026-11-02 through a general web-search tool, reading publisher, arXiv, PMC, PMLR, CVF and
repository pages. **Not directly accessible:** Scopus, IEEE Xplore and ScienceDirect search (paywalled/authenticated), so
those venues were reached only through public landing pages of individual papers. The survey is therefore targeted, not
a systematic review. Bibliographic details below were checked against at least one primary landing page; claims about
*content* are limited to what abstracts and landing pages state.

## 1. Datasets and benchmark flaws

| Dataset | Content | Notes verified from the source pages |
|---|---|---|
| Kvasir v1/v2 (Pogorelov et al., MMSys'17) | 3 anatomical landmarks, 3 findings, 2 polyp-removal classes; v1: 500/class, v2: 8,000 images | research/education use only; frames 720×576 to 1920×1072; some classes carry a green ScopeGuide picture-in-picture |
| Kvasir-SEG (Jha et al., MMM 2020) | 1,000 polyp frames with masks, derived from Kvasir v2; 332×487 to 1920×1072 | masks by a medical doctor, verified by a gastroenterologist |
| HyperKvasir (Borgli et al., Sci. Data 2020) | 110,079 images, 374 videos from Bærum Hospital, 2008–2016 | CC BY 4.0 article; images not class balanced |

**Known flaws (this study measured the first three; the rest come from the dataset documentation):** acquisition shortcuts
(resolution, ScopeGuide box), duplicates including a cross-class contradiction, suspected label noise (6.9 %), no patient/video
identifiers, single centre, overlapping Kvasir/HyperKvasir/Kvasir-SEG frames (so HyperKvasir is *not* an independent external test).
For video-derived endoscopy collections, grouping by video is already the recommended protocol (e.g. in Kvasir-Capsule studies);
Kvasir v2 offers no such handle.

## 2. Active-contour lineage

| Year | Work | Idea |
|---|---|---|
| 1988 | Kass, Witkin, Terzopoulos, *Snakes*, IJCV 1(4):321–331 | energy-minimising spline; edge attraction + smoothness |
| 1999/2001 | Chan & Vese, *Active contours without edges*, IEEE TIP 10(2):266–277 | region-based Mumford–Shah fitting; works on weak boundaries |
| 2014 | Márquez-Neila, Baumela, Álvarez, IEEE TPAMI 36(1):2–17 | morphological operators replace PDEs: simple, fast, no re-initialisation |
| 2022 | Ismail et al., *Image Anal. Stereol.* 41(1):7–23 | compares geodesic and Chan–Vese on colonoscopy segmentation; reports Chan–Vese fits polyps better when initialised inside, and that artefact removal/filtering matters |

These works evaluate **segmentation**. In our (non-exhaustive) search we found no study that tests an *unsupervised contour as a spatial
prior for a fine-tuned deep classifier on Kvasir with controls* (shuffled contours, image-independent masks, non-contour feature controls).

## 3. Backbones and tooling audited

| Model (timm id) | Paper | Licence of weights | bf16/fp16 |
|---|---|---|---|
| `resnet50.a1_in1k` | He et al. CVPR'16; Wightman et al. 2021 (training recipe) | Apache-2.0 | ✓ |
| `tf_efficientnetv2_s.in21k_ft_in1k` | Tan & Le, ICML'21 | Apache-2.0 | ✓ |
| `convnext_tiny.fb_in22k_ft_in1k` | Liu et al. CVPR'22 | Apache-2.0 | ✓ |
| `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | Liu et al. 2021 (arXiv:2103.14030) | MIT | ✓ |
| `vit_small_patch14_dinov2.lvd142m` | Oquab et al., TMLR 2024 | Apache-2.0 | ✓ |

All five load and fine-tune on an 8 GB GPU at 224 px in bf16 (measured peak VRAM in `outputs/results/tables/profiling.tex`).

## 4. Reported Kvasir accuracies are not comparable

Search results surface Kvasir classification papers reporting figures from the mid-80s to the high-90s percent (for example Inception-v3
transfer learning around 85 % on 8 classes; a 2026 study with a ViT on a *five-class* subset at 97.2 % on a single split). These figures use different
class subsets, random frame-level splits, and preprocessing, and we could not verify their protocols. We therefore make **no claim of beating published
numbers**; every method here is compared under one audited protocol.

## 5. Gap and contributions

| Gap in the literature we could find | Our response |
|---|---|
| Benchmark scores confounded by acquisition artefacts and frame leakage | measured and mitigated; effect reported (naive vs canonical, random vs grouped) |
| Active contours studied for segmentation, not as a classification prior | ACGNet + controls separating "a mask" from "this image's contour" |
| Accuracy-only reporting | calibration, conformal sets, selective classification, subgroup and error analysis |
| Seldom: statistical testing of component gains | paired bootstrap, McNemar, DeLong, Holm, seed-level tests |

## 6. Codebase audit

No official reference implementation exists for Kvasir classification, so there are no reference checkpoints to re-evaluate (step 8, optional
item not applicable). The contour stage uses scikit-image's `morphological_chan_vese` and `morphological_geodesic_active_contour`; the rest is
original and covered by 48 unit tests.
