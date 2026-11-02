# 20-phase, 85-step plan and compliance matrix

Legend: ✅ done · ➖ not applicable (reason given) · ⚠️ done with a stated limitation. "Core" items are all ✅ or ➖.
Paths are relative to the repository root.

## Phase I: Problem formulation, ML framing, feasibility
| # | Step | Status | Where |
|---|---|---|---|
| 1 | Domain discovery; why rules fail | ✅ | `docs/FORMULATION.md` §1 (heuristic baseline measured) |
| 2 | Task framing, `f_θ: X→Y` | ✅ | §2 |
| 3 | Objectives, RQs, H0/H1 | ✅ | §3 (RQ1–RQ5) |
| 4 | Metric taxonomy (+ cost matrix) | ✅ | §4; `scripts/analyze.py` (5:1 cost operating point) |
| 5 | Data/compute/latency feasibility | ✅ | §5 |
| 6 | Scope, non-goals, MVM vs stretch | ✅ | §6 |

## Phase II: Literature, SOTA, baselines
| 7 | Literature survey | ✅ | `docs/LITERATURE.md`, `paper/references.bib` (verified entries) |
| 8 | Codebase / model-zoo audit | ✅ ⚠️ | §3 of LITERATURE; no official Kvasir baselines exist to re-run (optional item ➖) |
| 9 | Benchmark & leaderboard audit | ✅ ⚠️ | §1, §4 of LITERATURE; no verifiable public leaderboard for this protocol |
| 10 | Gap synthesis & contributions | ✅ | §5 of LITERATURE; paper §I |

## Phase III: Governance, licensing, planning
| 11 | Work breakdown & roadmap | ✅ | this file, `scripts/run_deep.py` (declarative experiment table), `scripts/run_experiments.ps1` |
| 12 | Risk register & fallback | ✅ | `docs/FORMULATION.md` §7 |
| 13 | PII sanitisation | ➖ conditional (public images, no user text) | EXIF/comment scan: 0/8000 hits (`data/ingest.py`); burned-in text discussed in `docs/DATA_AUDIT.md` |
| 14 | Licence audit | ✅ | `docs/LICENSING.md` |

## Phase IV: Environment
| 15 | Repo structure, git, ignores | ✅ | standard layout; `.gitignore`; `ruff` configured in `pyproject.toml`; `.pre-commit-config.yaml` |
| 16 | Locked environment | ✅ | `.venv`, `requirements.txt` (frozen), `pyproject.toml` |
| 17 | GPU/OS configuration | ✅ | CUDA torch 2.11+cu128 verified; Windows-specific fixes documented (spawn, commit limit) |
| 18 | Run logging | ✅ | `runlog.py`: config, git hash, command, environment, CSV/JSON per run. TensorBoard/MLflow optional ➖ |

## Phase V: Acquisition
| 19 | Verified idempotent download | ✅ | `data/download.py` (byte count, SHA-256, zip test, zip-slip guard, resume) |
| 20 | Ingestion into canonical schema | ✅ | `data/ingest.py` (images → uint8 caches + parquet; no CSV/JSONL inputs exist) |

## Phase VI: Ground truth
| 21 | Labelling schema | ➖ conditional (dataset labels used as given) | class taxonomy in `data/contracts.py`; boundary cases (esophagitis↔z-line) in the audit |
| 22 | Self-annotated gold set | ➖ optional (no clinical expertise to annotate honestly) | Kvasir-SEG expert masks serve as the gold set for the contour stage |
| 23 | Programmatic labelling | ➖ optional | cleanlab used for *flagging*, not relabelling |

## Phase VII: Cleaning and QA
| 24 | Sanitisation | ✅ | decode check, valid-FOV, artefact masking (`data/preprocess.py`) |
| 25 | Duplicates | ✅ | SHA-256 + embedding cosine (pHash evaluated and rejected, with evidence) |
| 26 | Outliers / label noise | ✅ | range checks, IQR blur outliers, `cleanlab` (550 flagged) |
| 27 | Schema validation | ✅ | `data/contracts.py` (pydantic + assertions); it caught the duplicate-ID defect |

## Phase VIII: EDA
| 28 | Distributions, normality | ✅ | `scripts/eda.py`; `docs/DATA_AUDIT.md` §4 |
| 29 | Imbalance, subgroups | ✅ | 1.007 (8-class), 1:7 (polyp-vs-rest), per-format and per-box subgroups |
| 30 | Collinearity, shortcuts | ✅ | VIF, Spearman, MI; resolution/box probes |
| 31 | Manifold | ✅ | UMAP (`outputs/figures/eda_umap.*`) |

## Phase IX: Splits and leakage
| 32 | Deterministic split | ✅ | stratified **group** split, seed 42; random split as control; OOD hold-out optional ➖ |
| 33 | Leakage audit | ✅ | `curate.leakage_audit`; zero group/hash overlap asserted |
| 34 | CV folds | ✅ | 5 grouped folds serialised (`cv_fold`), used for out-of-fold label-noise estimation |
| 35 | Versioned artefacts | ✅ | `data/processed/{train,val,test}_ids.txt` + SHA-256 in `curation_report.json`; Arrow/HDF5 optional ➖ (memory-mapped `.npy` used) |

## Phase X: Features and preprocessing
| 36 | Scaling | ✅ | train-only 1/99 winsorise + z-score (`dataset.fit_scaler`); power transforms optional ➖ |
| 37 | Categorical/temporal encoding | ➖ | no such features |
| 38 | Domain features, selection | ✅ ⚠️ | 136 features (colour, LBP, GLCM, shape, boundary); importance by MI and SHAP rather than pruning |
| 39 | Tokenisation | ➖ | vision task |
| 40 | Augmentation | ✅ | GPU affine + colour jitter, contour mask follows geometry (`data/augment.py`, tested) |
| 41 | Imbalance mitigation | ➖ | classes balanced; polyp-vs-rest handled by threshold tuning, not resampling |
| 42 | Pipeline serialisation | ✅ | `scaler.json`, `calibration.json`, `config.yaml` inside each bundle |

## Phase XI: Modelling
| 43–46 | Formulation, loss, architecture, parameters/VRAM | ✅ | `docs/FORMULATION.md` §8; `models/acgnet.py`; measured params and VRAM in profiling table |

## Phase XII: Skeleton and baselines
| 47 | Dataset/DataLoader | ✅ | `data/dataset.py` (uint8 contract, in-process loader; spawn workers rejected on Windows) |
| 48 | Walking skeleton | ✅ | `tests/test_walking_skeleton.py` (constant model, real loader, runs in seconds) |
| 49 | Trivial & heuristic baselines | ✅ | random, prior, Lab-centroid heuristic (`classical.run_trivial`) |
| 50 | Classical baselines | ✅ | LR, RBF-SVM, RF, LightGBM, XGBoost |
| 51 | Pretrained / confidence baselines | ✅ | frozen-DINOv2 linear probe; max-softmax risk–coverage |

## Phase XIII: Implementation
| 52 | Config schema | ✅ | `configs/*.yaml` + dotted CLI overrides (unit-tested; caught the YAML `1e-3` string bug) |
| 53 | Modular architecture, hooks | ✅ | `ACGNet`; forward hooks for Grad-CAM and layer-wise probes |
| 54 | Metrics | ✅ | `training/metrics.py` (`ignore_index` ➖ no padding) |
| 55 | Optimiser, scheduler, clipping | ✅ | AdamW, no decay on bias/norm, warm-up + cosine, clip 1.0 |
| 56 | Train/eval loops | ✅ | `training/engine.py` (accumulation optional ➖ not needed) |
| 57 | Seeding | ✅ ⚠️ | all RNGs seeded; cuDNN deterministic; `use_deterministic_algorithms(warn_only=True)`, so a few CUDA kernels remain non-deterministic (bit-exact replay is not guaranteed; seed-to-seed spread is reported) |
| 58 | Checkpoint, early stopping | ✅ | best `.safetensors` by val macro-F1, patience 4 |

## Phase XIV: Debugging
| 59 | Shape/device asserts | ✅ | in `ACGNet.forward`, `predict` (finite logits) and tests |
| 60 | Single-batch overfit | ✅ | `tests/test_model_sanity.py` |
| 61 | Gradient flow | ✅ | test (after one step, because timm zero-inits last BN gains); grad norm logged each epoch |
| 62 | Numerical stability | ✅ | fp32 loss on bf16 logits, finite asserts, stable softmax |
| 63 | Unit tests | ✅ | 48 tests; batch-order invariance included |

## Phase XV: Optimisation
| 64 | AMP, SDPA, cached features | ✅ | bf16 autocast; timm fused attention for ViT/Swin; cached frozen embeddings |
| 65 | PEFT | ⚠️ optional | frozen-backbone linear probe only; LoRA not used |
| 66 | HPO | ✅ ⚠️ | Optuna/TPE on validation for the contour (80 trials) and classical models (25 each); deep nets use one shared recipe. A learning-rate sweep (seed 0) and a post hoc 3-seed replication at 2x LR quantify what that costs: the plain model improves by 0.8 points at 2x LR and the guided-vs-plain gap stays null |
| 67 | Ensembling | ✅ | seed and backbone soft-voting |

## Phase XVI: Post-processing
| 68 | Threshold tuning | ✅ | Youden J on validation (+ cost-optimal threshold) |
| 69 | Calibration | ✅ | temperature scaling, reliability diagrams, ECE, Brier |
| 70 | Uncertainty | ✅ | split-conformal (LAC), risk–coverage |

## Phase XVII: Evaluation
| 71 | Locked test evaluation | ✅ | test touched once per frozen run; secondary dataset optional ➖ (HyperKvasir overlaps) |
| 72 | Multi-seed | ✅ | deep N=3, classical N=5, mean ± SD |
| 73 | Significance | ✅ | paired bootstrap (2,000), McNemar, DeLong, Holm, Cohen's d, seed-level tests |
| 74 | Ablations | ✅ | 7 variants + 3 controls |
| 75 | Sensitivity, scaling, Pareto | ✅ | learning curves, LR sweep, accuracy–latency |
| 76 | OOD / perturbation | ✅ | JPEG compression and Gaussian blur on 480 stratified frames, 3 seeds x 5 variants incl. controls (`scripts/robustness2.py`); single-seed noise/blur/brightness/JPEG screen (`scripts/robustness.py`, helper) |
| 77 | Profiling | ✅ | latency (warm-up + sync), VRAM, train time, checkpoint size |

## Phase XVIII: Interpretability and errors
| 78 | Confusion, per-class, ROC/PR | ✅ | `analyze.py` |
| 79 | Failure taxonomy | ✅ | error gallery + automated categories (label-flagged, confusable pairs) + manual inspection notes in the report |
| 80 | Attribution | ✅ | TreeSHAP, Grad-CAM (pointing game vs expert masks) |
| 81 | Layer-wise probes | ✅ | per-stage linear probes, pretrained vs fine-tuned |
| 82 | Subgroup audit | ✅ | by scope format, ScopeGuide box, class; worst-group accuracy |

## Phase XIX: Packaging
| 83 | One-command CLI | ✅ | `scripts/run_experiments.ps1`; entry points `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py` (wrappers over `acpolyp.training.train` / `acpolyp.cli`), `python -m acpolyp.cli serve` |
| 84 | Bundle + local UI | ✅ | `outputs/artifacts/acpolyp-v1/` (weights, scaler, calibration, model card, hashes); FastAPI + custom single-page UI (instead of Gradio) |

## Phase XX: Documentation and archival
| 85 | README, figures/tables script, manuscript, slides, clean-room check | ✅ ⚠️ | `README.md`; `scripts/make_figures.py` (runs EDA, contour, analysis, export, paper-asset scripts: 300-dpi vector PDFs, `booktabs` tables with bold-best and ± SD); `paper/main.tex` (IEEEtran, 7 pp.); `report/report.pdf` (34 pp.); `slides/ACPolyp.pptx` (13 slides); clean-room clone + install + 48 tests run locally (the 8 h pipeline was not re-run from scratch). **Publishing to GitHub/Zenodo/Hugging Face is left to the owner** (needs their accounts) |
