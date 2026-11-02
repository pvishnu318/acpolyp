# Master script: rebuild everything from an empty checkout.  ~7-8 h on an RTX 4060 laptop.
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1            # full study
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Quick     # smoke test (2 epochs, 1 seed, ~15 min)
param([switch]$Quick, [switch]$SkipSetup)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$env:HF_HOME = "$PWD\data\cache\hf"; $env:TORCH_HOME = "$PWD\data\cache\torch"; $env:PYTHONUNBUFFERED = "1"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $name" }
}

if (-not $SkipSetup) {
    Step "1/14 environment" {
        if (-not (Test-Path .venv)) { python -m venv .venv }
        & $py -m pip install --upgrade pip
        & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        & $py -m pip install -r requirements.txt
        & $py -m pip install -e . --no-deps
    }
}
Step "2/14 unit tests"                    { & $py -m pytest tests -q }
Step "3/14 download + verify datasets"    { & $py -I -m acpolyp.data.download }
Step "4/14 ingest + canonicalise"         { & $py -I -m acpolyp.data.ingest }
Step "5/14 dedup, groups, splits, audit"  { & $py -I -m acpolyp.data.curate }
Step "6/14 match Kvasir-SEG frames"       { & $py -I -m acpolyp.ac.match_seg }
Step "7/14 tune active contour"           { & $py -I -m acpolyp.ac.tune }
Step "8/14 contours + features"           { & $py -I -m acpolyp.ac.extract }
Step "9/14 EDA + shortcut audit"          { & $py -I scripts\eda.py }

$seeds = if ($Quick) { @("0") } else { @("0", "1", "2") }
$extra = if ($Quick) { @("--set", "train.epochs=2") } else { @() }
Step "10/14 classical baselines"          { & $py -I -m acpolyp.training.classical --trials $(if ($Quick) { 3 } else { 25 }) --seeds $(if ($Quick) { "0" } else { "0","1","2","3","4" }) }
Step "11/14 deep experiments"             {
    foreach ($g in "ablation", "backbones", "generality", "shortcut", "lowdata") {
        & $py -u -I scripts\run_deep.py --group $g --seeds @seeds @extra
    }
}
Step "12/14 analysis + figures"           { & $py -I scripts\analyze.py; & $py -I scripts\ac_figures.py }
Step "13/14 bundle + interpretability"    { & $py -I scripts\build_bundle.py; & $py -I scripts\interpretability.py }
Step "14/14 paper assets + PDF"           {
    & $py -I scripts\make_paper_assets.py
    Push-Location paper; pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex; Pop-Location
}
Write-Host "`nDone. Results: outputs\results, figures: outputs\figures, paper: paper\main.pdf" -ForegroundColor Green
