# Publishing checklist (Phase XX, step 85)

Nothing here has been run for you: GitHub, Zenodo and Hugging Face need your accounts. Do the steps in order.

## 0. What may and may not be published

| Item | Publish? | Why |
|---|---|---|
| Code, tests, configs, split id lists, paper and report sources, figures | Yes (MIT) | Figures contain a few attributed Kvasir frames as illustrative panels; see `docs/LICENSING.md` |
| Kvasir / Kvasir-SEG images | **No** | Not redistributable; the downloader fetches them from Simula |
| `model.safetensors` (107 MB) | Hugging Face only, with a research-only licence | Weights trained on Kvasir inherit the research and education restriction |
| `data/`, `outputs/runs/`, caches | **No** | Large and derived from the dataset (already git-ignored) |

If you want a commercial product, ask Simula for written permission first (contact on https://datasets.simula.no/kvasir/).

## 1. Fill in the placeholders

Replace every `REPLACE_*` in `CITATION.cff` and `.zenodo.json` with your real name, GitHub user and repository name.
Optionally set your name in the `\author` block of `paper/main.tex` and `report/main.tex`, then rebuild the PDFs.
Set a real git identity (the history currently uses "ACPolyp author"; rewriting history is your call, a new
commit identity from now on is enough):

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

## 2. Run the gate

```bash
python scripts/prepublish_check.py
```

It must print `OK`. It fails on placeholders, tracked data or weights, files over 20 MB, secrets and uncommitted changes.
Commit the placeholder edits first.

## 3. GitHub

Create an empty public repository named as in `CITATION.cff` (no README, no licence, so the history pushes cleanly), then:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USER/acpolyp.git
git push -u origin main
git tag v1.0.0
git push origin v1.0.0
```

The local branch is currently called `master`; renaming to `main` matches the default.

## 4. Zenodo (DOI for the code)

1. Sign in to https://zenodo.org with GitHub and switch the repository on under Account, GitHub.
2. On GitHub, publish a release from tag `v1.0.0`. Zenodo archives it and reads `.zenodo.json` / `CITATION.cff`.
3. Copy the DOI badge into the top of `README.md` and the DOI into `CITATION.cff`, commit, push.

## 5. Hugging Face (weights and model card)

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create acpolyp-v1 --type model
cp docs/HF_MODEL_CARD.md outputs/artifacts/acpolyp-v1/README.md
huggingface-cli upload YOUR_USER/acpolyp-v1 outputs/artifacts/acpolyp-v1 .
```

Do not paste your token into chat, scripts or the repository; `login` stores it locally.
Consider making the repository **gated** so users must accept the research-only terms.

## 6. After publishing

* Put the three links (GitHub, Zenodo DOI, Hugging Face) into the README header.
* Verify from a clean machine: `git clone`, create a venv, `pip install -r requirements.txt`, `pytest`.
  The 8-hour training pipeline has not been re-run from scratch; say so if you claim full reproduction.
* Do not describe the project as a clinical tool, a state-of-the-art result or a working improvement from contour
  guidance. The supported claims are the audit, the null result, calibration and the conformal analysis.
