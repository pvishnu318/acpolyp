# Presentation briefing (10 topics)

Every number below comes from this repository's own results (`docs/`, `paper/tables/`, `outputs/results/`).
Say "our result" for numbers and "the dataset documentation" for facts about Kvasir.

---

## 1. Motivation

* Colorectal polyps are growths on the bowel lining. Some turn into cancer, so finding and characterising them during
  colonoscopy is a quality measure, and endoscopists differ in how many they detect (see `docs/FORMULATION.md`).
* Computer-aided classification of endoscopic frames is a standard research topic, and Kvasir is a popular benchmark for it.
* **Two observations motivated this work.**
  1. Reported Kvasir scores are hard to compare. Papers use different class subsets, random frame-level splits
     and different preprocessing, and the frames carry acquisition artefacts (image resolution, an overlay box) that
     correlate with the class. High scores can come from shortcuts rather than medicine.
  2. *Active contours* (snakes, Chan-Vese) are a classical, label-free way to outline a region of interest. They are
     studied for segmentation. We found no study using an unsupervised contour as a spatial prior for a fine-tuned deep
     classifier with proper controls (non-exhaustive search, see `docs/LITERATURE.md`).
* **So the project asks, sceptically:** does a classical active-contour prior still help a strong modern network, and if
  it appears to, is it the contour or just extra features?

## 2. Problem statement

* **Task:** supervised single-label classification of an endoscopic frame `x` into one of 8 Kvasir classes, `f_theta: X -> 8-way probability`.
* **Polyp view:** the same model gives `P(polyps | x)`; thresholding it answers "is there a polyp?" (1 : 7 imbalance, 993 polyp vs 6,976 other frames).
* **Research questions**
  * RQ1: does contour guidance improve macro-F1 over plain ConvNeXt-T? (H0: no difference)
  * RQ2: is any gain due to this image's contour, to extra hand-crafted features, or to any mask at all?
  * RQ3: how much accuracy comes from shortcuts and split leakage?
  * RQ4: does guidance help more with little training data?
  * RQ5: are the probabilities calibrated and do conformal sets reach their nominal coverage?
* **Pre-declared:** a null result for RQ1 is acceptable, and the audit, calibration and shortcut findings are then the contribution.
* **Not claimed:** clinical use, detection or segmentation as a product, external validation.

## 3. Models chosen

| Group | Models |
|---|---|
| Deep, fine-tuned (timm, ImageNet or LVD pretrained) | ResNet-50, EfficientNetV2-S, ConvNeXt-T, Swin-T, DINOv2 ViT-S/14 |
| Proposed | **ACG-ConvNeXt** = ConvNeXt-T plus the contour-guided head (ACGNet), with ablation and control variants |
| Classical ML on 136 hand-crafted features | Logistic regression, RBF-SVM, Random forest, XGBoost, LightGBM |
| Other baselines | Linear probe on frozen DINOv2; random guess; majority class; nearest Lab-colour-centroid heuristic |
| Contour stage (no learning) | Morphological Chan-Vese (+ optional morphological geodesic active contour) |
| Post-hoc | Temperature scaling, Youden threshold for polyp-vs-rest, split-conformal prediction sets, 5-model soft-voting ensemble |

## 4. How it helps (honest version)

* **The contour did not help classification.** ACG-ConvNeXt macro-F1 0.9277 vs plain ConvNeXt-T 0.9233, difference +0.0045,
  95% CI -0.0004 to +0.0091, Holm p = 0.30: not significant. Controls with the wrong image's contour (+0.0043) or a central
  ellipse (+0.0035) do just as well, so the gain, if any, is not from this image's contour.
* **What does help or is useful:**
  * A strong fine-tuned backbone gives about 0.92-0.93 macro-F1 and polyp-vs-rest AUROC above 0.99. A five-model ensemble reaches 0.9385.
  * The **data audit** (duplicates, a cross-class duplicate, 6.9% suspect labels, resolution shortcut, leakage) is reusable by anyone using Kvasir.
  * **Calibration:** temperature scaling lowers ECE from 0.046 to 0.033.
  * **Conformal sets:** calibrated on validation they covered only 92.9% at a 95% target (validation was reused for model selection);
    group-aware calibration gave 95.2%.
  * A statistically controlled protocol (3 seeds, paired bootstrap, McNemar, DeLong, Holm) that others can copy.
* **The contour itself:** Dice 0.406 against expert masks vs 0.375 for a trivial central ellipse (better on 62% of images, Wilcoxon p = 2.5e-6),
  but poor on small polyps (Dice 0.21). It is a weak localiser, and that explains why it cannot add much.

## 5. Why these models

* **Coverage of design families**, so the conclusion does not depend on one architecture: classic residual CNN (ResNet-50),
  efficiency-oriented CNN (EfficientNetV2-S), modernised CNN (ConvNeXt-T), hierarchical transformer (Swin-T), self-supervised
  transformer (DINOv2 ViT-S/14).
* **Practical limits:** all fit an 8 GB laptop GPU in bf16 at 224 px (peak 1.9-2.8 GB), all have free pretrained weights with
  permissive licences (Apache-2.0 or MIT), all are in `timm`, all are 20-28 M parameters so comparisons are fair.
* **Why ConvNeXt-T hosts the proposed head:** it is strong, fast (15 ms per image, batch 1) and CNN-style, so its last feature map
  keeps spatial layout, which the mask pooling needs.
* **Why classical models:** the contour features are hand-crafted and tabular, so tree and kernel models are the natural baseline,
  and they show how far features alone go (macro-F1 about 0.75-0.81).
* **Honest caveats:** ResNet-50 trails (0.871), probably because of the training recipe, not because the architecture is
  inherently worse. Deep models share one recipe and were not tuned individually. This is stated in the paper.

## 6. How each model works

* **ResNet-50.** 50 layers of bottleneck blocks (1x1, 3x3, 1x1 convolutions). Each block learns a residual `y = x + F(x)`; the skip connection
  makes deep networks trainable. Global average pooling then a linear classifier. 23.5 M parameters.
* **EfficientNetV2-S.** Uses MBConv blocks (inverted bottleneck with depthwise convolution and squeeze-and-excitation) and
  Fused-MBConv in early stages for speed. Depth, width and resolution are scaled together, and the architecture was found by search. 20.2 M parameters.
* **ConvNeXt-T.** A ResNet rebuilt with transformer-era design: a 4x4 patch-stem, stages of depth (3, 3, 9, 3),
  7x7 depthwise convolutions, inverted-bottleneck MLP, LayerNorm and GELU. A pure convolutional net with transformer-level accuracy. 27.8 M parameters.
* **Swin-T.** A vision transformer that splits the image into patches and runs self-attention inside local windows (7x7), shifting the windows
  between layers so information crosses window borders; patch merging builds a hierarchy of resolutions. 27.5 M parameters.
* **DINOv2 ViT-S/14.** A small vision transformer (14x14 patches) pretrained **without labels** by self-distillation on a large curated image set.
  Its features transfer well; we fine-tune it, and also use it frozen for the linear probe and for duplicate detection. 21.6 M parameters.
* **Chan-Vese active contour (morphological).** A curve is evolved to split the image into two regions, each close to its own mean
  intensity (energy = length + inside error + outside error). The morphological version uses fast binary dilation, erosion and smoothing operators
  instead of differential equations. No labels are involved.
* **Classical models.** Logistic regression: a linear decision boundary with regularisation. RBF-SVM: maximum-margin classifier with a kernel.
  Random forest: many decorrelated decision trees voting. XGBoost and LightGBM: trees built one after another, each correcting the previous errors (gradient boosting).
* **Temperature scaling and conformal sets.** Temperature scaling divides logits by one learned number to fix over-confidence. Split-conformal
  prediction picks a probability cut-off on held-out data so that the returned set of classes contains the truth with the chosen frequency (95% target).

## 7. The main architecture (ACGNet)

```
frame -> canonicalise (FOV crop, mask the overlay box) -> 256x256 cache
      |-> salience channel -> morphological Chan-Vese -> contour mask M ; 136 features (38 global + 98 contour)
      '-> ConvNeXt-T backbone -> feature map F (C x h x w)
Pool F three ways using the valid mask V and contour mask M:
   g     = LayerNorm(mean of F over V)            whole field of view
   f_in  = LayerNorm(mean of F over M)            inside the contour
   f_out = LayerNorm(mean of F over V minus M)    surrounding mucosa
e = MLP(standardised contour features)            interpretable cues
logits z = Linear(Dropout([g ; f_in ; f_out ; e])) -> softmax -> 8 classes
```

* **Idea:** judge the candidate lesion and its background separately, the way an endoscopist compares a lesion with the mucosa around it.
* **Cost:** +0.04 M parameters (27.83 M to 27.87 M), same speed (13.8 vs 15.1 ms per image).
* **Ablations:** only global features; only contour pooling; only contour features; full model.
  **Controls:** contours shuffled between images; a central ellipse instead of the contour.
* **Training recipe:** AdamW, head LR 2e-4 and backbone LR 0.25x, weight decay 0.05 (not on biases or norms), 1 warm-up epoch then cosine decay,
  gradient clipping 1.0, label smoothing 0.1, stochastic depth 0.1, head dropout 0.2, bf16, batch 32, 10 epochs, early stopping (patience 4), seeds 0, 1, 2.
* **Tuned contour settings (used for every result):** HSV **saturation** channel with centre-surround filtering, 20 Chan-Vese iterations, geodesic stage
  switched off by the search (it did not help). The documented default (Lab principal component) is the untuned starting point.

## 8. How the data is processed (what the model sees)

1. **Canonicalise each frame:** find the circular field of view (threshold, morphological opening, largest blob, convex hull), crop to it, resize to 256x256.
2. **Hide acquisition artefacts:** zero out the ScopeGuide picture-in-picture box at its canonical rectangle in **every** frame (so its presence tells nothing), and zero everything outside the FOV.
3. **Cache** as uint8 arrays (memory-mapped) together with a valid mask.
4. **Training augmentation, on the GPU:** random-resized crop (area 0.55-1.0), rotation, flips, brightness, contrast and saturation jitter; image, contour mask and valid mask are warped with the **same** grid.
5. **Evaluation transform:** resize 256 to 224, ImageNet normalisation, multiply by the valid mask. Identical in training, evaluation, CLI and the website, so there is no train/serve mismatch.
6. **Hand-crafted features:** 136 per image (38 global colour/texture; 98 from the contour: shape, boundary, region). Clipped to the training 1st-99th percentile, then z-scored using **training data only**.
7. **Preprocessors saved** (`scaler.json`, config, calibration) in the model bundle.

## 9. Understanding of the dataset

* **Kvasir v2** (Simula, Pogorelov et al., MMSys 2017): 8,000 endoscopic images from gastrointestinal examinations at Baerum Hospital (Norway), annotated by medical doctors; 8 classes x 1,000 images.
* **Classes:** 3 anatomical landmarks (normal z-line, normal pylorus, normal cecum), 3 pathological findings (esophagitis, polyps, ulcerative colitis), 2 polyp-removal categories (dyed lifted polyps, dyed resection margins).
* Frames are JPEG, between 720x576 and 1920x1072 pixels, with a black margin and often burned-in overlay text.
* **Known problems that we found and measured**
  * One file appears under two classes (contradictory label); 30 more duplicate pairs.
  * **Resolution leaks the label:** a classifier using only image size separates upper-GI from colon at 96.8% (majority class 62.4%).
  * The green **ScopeGuide box** appears in 95% of cecum frames, about 52% of polyp frames and 0% of upper-GI frames.
  * **No patient or video IDs**, so frames of one procedure cannot be linked directly, and near-identical consecutive frames could leak across splits.
  * **Suspected label noise:** confident-learning flags 550 of 7,969 images (6.9%), mostly esophagitis vs z-line, which is clinically plausible because esophagitis is diagnosed at the z-line.
  * Single hospital, 2008-2016, one scope family.
* **Companion data:** Kvasir-SEG, 1,000 polyp images with expert masks, used only to tune and score the contour stage, not for classification labels.
* **Licence:** research and education only; not redistributed here; commercial use needs Simula's written permission.
* **Versions:** we use v2. v1 is the smaller 500-per-class release; v3 is listed on the site without a description and we did not use it.

## 10. How we processed the dataset (pipeline order)

1. **Download and verify:** exact byte count, SHA-256, zip integrity test, safe extraction. (The first download was silently truncated at 771 MB; a size check caught it and a resumable downloader fixed it.)
2. **Ingest:** decode each image, record size, hashes, FOV statistics, EXIF check (0 of 8,000 files carry personal-data tags), into one table with a strict schema (pydantic contract; unique IDs, valid labels, no NaNs).
3. **Duplicates:** exact SHA-256 plus near-duplicates using frozen DINOv2 embeddings (cosine at least 0.98). Same-class duplicates keep one copy; cross-class duplicates drop both.
   Result: **31 frames removed, 7,969 remain**. (pHash was tried and rejected: the shared black border made it produce false matches.)
4. **Scene groups:** frames with cosine at least 0.92 are linked into 7,812 groups (largest 7), and a group is never split across train/val/test.
5. **Label-noise audit:** cleanlab on out-of-fold predictions flags 550 images. They are flagged, **not removed**, because removing them would tune the data toward the test set. They are used in the error analysis.
6. **Split:** stratified, group-aware, seed 42: train 5,578, validation 797, test 1,594, classes balanced within one image. Zero overlap of groups or hashes across splits (checked). A random split is kept only as a leakage control.
7. **Fingerprint:** split ID lists are stored with SHA-256 hashes.
8. **Features and scaling:** fit on the training split only.
9. **Evaluation discipline:** the test split is touched once per frozen run; thresholds, temperature and model selection use validation only. Three seeds per deep model.

---

## Numbers worth memorising

| Item | Value |
|---|---|
| Images / classes | 8,000 (7,969 after cleaning) / 8 |
| Split | 5,578 / 797 / 1,594 |
| Best single model (macro-F1) | DINOv2 ViT-S/14 0.9344, Swin-T 0.9327, EfficientNetV2-S 0.9257, ConvNeXt-T 0.9233, ResNet-50 0.8708 |
| Proposed | ACG-ConvNeXt 0.9277 (difference +0.0045, Holm p = 0.30, not significant) |
| Ensemble | 0.9385 macro-F1 |
| Classical on features | 0.75-0.81 macro-F1 |
| Floor | random 0.131, colour heuristic 0.476 |
| Contour Dice | 0.406 vs ellipse 0.375 |
| Calibration | ECE 0.046 to 0.033 |
| Conformal | 92.9% (validation-calibrated) vs 95.2% (group-aware) at 95% target |
| Speed (ConvNeXt-T) | 15 ms per image, 2.5 GB VRAM, about 4.7 min training |

## Likely questions and honest answers

* **"Did your method work?"** Not in accuracy. The contour prior gave no statistically supported gain, and the controls show why. The contribution is the audit, the controlled test and calibrated outputs.
* **"Why is that still worth presenting?"** A negative result with proper controls tells others not to spend time on this, and the data audit applies to every Kvasir paper.
* **"Is 93% good?"** It is competitive with published figures but those use different protocols, so we make no ranking claim. Our protocol is stricter (grouped split, duplicates removed).
* **"Can it be used clinically?"** No. Single centre, no external validation, research prototype.
* **"Why not remove the noisy labels?"** That would tune the dataset to the model and inflate results.
* **"Why does the contour not help?"** It is a weak localiser (Dice 0.41, 0.21 on small polyps), and a strong pretrained network already finds the lesion on its own. This is our interpretation, supported by the controls but not proved.
* **"Can you sell it?"** The code is MIT, but Kvasir forbids commercial use without Simula's written permission, so no.
