---
# Copy this file to README.md inside the Hugging Face model repo (see docs/PUBLISHING.md).
license: other
license_name: research-and-education-only
license_link: https://datasets.simula.no/kvasir/
library_name: timm
pipeline_tag: image-classification
tags:
  - medical-imaging
  - endoscopy
  - kvasir
  - polyp
  - research-only
  - not-a-medical-device
---

# ACPolyp-v1 (ACG-ConvNeXt-T, Kvasir v2)

**Research prototype. Not a medical device. No clinical validation.**

The weights were trained on Kvasir v2 (Simula), which is licensed for research and education only.
Commercial use of this model is not permitted without Simula's prior written permission.
The code is MIT-licensed; these weights are **not**, because they inherit the dataset restriction.
The Kvasir images are not included.

## What it is

ConvNeXt-T (27.9 M parameters, ImageNet-22k to 1k pretrained, Apache-2.0) with the ACGNet head: backbone
features pooled inside and outside an unsupervised active-contour mask, fused with 98 contour features,
8 Kvasir classes. Temperature scaling and a split-conformal quantile are stored in `calibration.json`.

## Honest results

* Bundled model: run `ab_acg/seed0`, chosen by validation macro-F1. Test macro-F1 0.9315, accuracy 0.9316,
  polyp-vs-rest AUROC 0.9886, ECE 0.0312 (after temperature scaling).
* That is one seed. The 3-seed mean of the contour-guided model is 0.9277 against 0.9233 for plain ConvNeXt-T:
  the difference is **not statistically supported** (Holm p = 0.30). Contour guidance did not help.
* The conformal quantile in the bundle was calibrated on the test split (validation was reused for model
  selection and under-covers), so the test numbers above are not an unbiased estimate of conformal coverage.

## Limits

Single-centre data, no patient identifiers (scene-grouped split is the best available leakage guard), no
external validation, esophagitis vs z-line is the main confusion, and a fraction of Kvasir labels are suspect
(see the audit in the repository).

## Use

Load with the `acpolyp` package from the code repository: `python -m acpolyp.cli predict --help`.
The folder is verified by SHA-256 in `manifest.json`.

## Citation

Cite the code repository (see `CITATION.cff` there) and the datasets:
Pogorelov et al., MMSys 2017, doi:10.1145/3083187.3083212; Jha et al., MMM 2020, doi:10.1007/978-3-030-37734-2_37.
