# Licensing and permitted use (Phase III, step 14)

## Data

| Asset | Terms (verbatim summary from the dataset pages, accessed 2026-11-02) |
|---|---|
| **Kvasir v2** (Pogorelov et al., MMSys 2017) | "restricted for research and educational purposes only … for other purposes including commercial purposes is forbidden without prior written permission." Any paper using it must cite the dataset paper. |
| **Kvasir-SEG** (Jha et al., MMM 2020) | Same research/education restriction; commercial use needs prior written permission. |

**Consequences for this repository**

1. **The datasets are not redistributed** (`data/` is git-ignored; the downloader fetches them from the owner). A small number of frames appear as *illustrative panels* in figures
   (contour gallery, pipeline illustration, error gallery), as is usual in research papers, with attribution to Pogorelov et al. (2017) and Jha et al. (2020) and for research/education purposes only.
   If you republish this repository commercially or want to remove those panels, regenerate the figures without them or obtain Simula's permission.
2. The web demo only processes images the user uploads or, if the dataset happens to exist locally, a handful of local
   samples that are never shipped with the code.
3. **Monetisation.** The *code* is MIT-licensed, but models trained on Kvasir and any product built on them fall under
   the dataset's non-commercial terms. Selling, licensing or deploying this model commercially requires written
   permission from Simula (contact on the dataset page), or retraining on data whose licence allows it. A portfolio
   piece, a paper, a teaching resource and open-source code are all within the terms.

## Code and pretrained weights

Checked against the Hugging Face model metadata on 2026-11-02:

| Weights (timm id) | Licence | Pretraining data |
|---|---|---|
| `convnext_tiny.fb_in22k_ft_in1k` | Apache-2.0 | ImageNet-1k / 22k |
| `resnet50.a1_in1k` | Apache-2.0 | ImageNet-1k |
| `tf_efficientnetv2_s.in21k_ft_in1k` | Apache-2.0 | ImageNet-1k / 21k |
| `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | MIT | ImageNet-1k / 22k |
| `vit_small_patch14_dinov2.lvd142m` | Apache-2.0 | LVD-142M |

The weights are permissive; ImageNet itself carries its own research-use terms, which downstream commercial users
should review. Libraries (PyTorch, timm, scikit-image, scikit-learn, LightGBM, XGBoost, Optuna, cleanlab, SHAP,
FastAPI) are BSD/MIT/Apache-2.0 licensed.

## Privacy

* 0 of 8,000 files carry EXIF/JPEG-comment fields from the personal-data list we scan (description, make, model,
  software, date, artist, copyright, GPS).
* Frames contain burned-in overlay text (field labels and timestamps). Names appear blank/anonymised in the frames we inspected,
  but we cannot guarantee it for all 8,000. The web demo never stores uploads.

## Not a medical device

Nothing here has regulatory clearance. The web UI, model card and paper state that it is a research prototype.
