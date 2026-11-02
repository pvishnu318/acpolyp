# Failure taxonomy of the most confident errors (Phase XVIII, step 79)

Model: seed-ensembled ACG-ConvNeXt on the 1,594-frame grouped test split (117 errors). The **36 most confident errors** were inspected
(`outputs/figures/error_gallery.*` shows the top 12; ranks 13–36 were inspected the same way) and sorted into four categories.

> **Caveat.** The categories are visual judgements by the project author, who is **not a clinician**. They describe what the frames
> look like, not a clinical adjudication. A gastroenterologist should review them before any conclusion about individual cases.

| Category | n / 36 | flagged by confident learning | What the frames show |
|---|---|---|---|
| **A. Esophagitis ↔ normal z-line** | 17 | 15 | Irregular, inflamed or whitish mucosa at or near the gastro-oesophageal junction. Both labels are plausible; esophagitis is *defined* relative to the z-line. |
| **B. Dyed-lifted-polyp ↔ dyed-resection-margin** | 7 | 5 | Blue-dye craters or lifted lesions at different stages of endoscopic mucosal resection; the visible state frequently supports either label. |
| **C. "Polyp" frames predicted as another colon class (or region)** | 7 | 2 | Frames labelled *polyps* in which no distinct lesion is visible (homogeneous mucosa) or the lesion is small or at the edge. Predicted cecum, colitis or pylorus. Fewest flagged, so these may include genuinely missed lesions: the clinically important group. |
| **D. Other** | 5 | 4 | Diffuse redness labelled colitis predicted as cecum or esophagitis; a snare loop with bleeding labelled colitis. |

**26 of the 36 (72 %) involve labels flagged by confident learning**, versus about 5 % of correctly classified test frames; 73 % of *all* 117 errors lie in
the two confusable pairs A and B. Implication: much of the residual error is label ambiguity, and further model capacity is unlikely to help
without adjudicated labels.

## Qualitative comparison (representative)

| Input (rank) | Ground truth | Prediction | Category |
|---|---|---|---|
| rank 1 (top-12 gallery) | dyed lifted polyps | dyed resection margins (1.00) | B |
| rank 5 | polyps | normal pylorus (0.99) | C, homogeneous mucosa, no visible lesion |
| rank 14, 15, 17, 18 | esophagitis | normal z-line (0.97–0.98) | A |
| rank 19 | ulcerative colitis | polyps (0.97) | D, bleeding with snare-like loop |
| rank 27 | dyed resection margins | dyed lifted polyps (0.96) | B |
