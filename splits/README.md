# splits

Immutable partitions used in every experiment (image IDs only, no images; Kvasir is research/education-licensed and is not redistributed).

* `train_ids.txt`, `val_ids.txt`, `test_ids.txt` — class-prefixed IDs (`<class>/<uuid>`), scene-grouped and stratified, seed 42.
* `curation_report.json` — duplicates removed, scene-group statistics, leakage audit, SHA-256 of each ID list.
* `dataset_manifest.json` — archive sizes and SHA-256 of the downloaded Kvasir v2 and Kvasir-SEG zips.

Rebuild them with `python -m acpolyp.data.curate` (deterministic given the downloaded data).
