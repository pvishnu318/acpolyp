"""Canonical filesystem locations. Every other module imports paths from here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
DOWNLOAD = RAW / "download"
DOWNLOAD_SEG = RAW / "download_seg"
KVASIR_RAW = RAW / "kvasir-v2"
KVASIR_SEG_RAW = RAW / "kvasir-seg"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"

CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
RUNS = OUTPUTS / "runs"
RESULTS = OUTPUTS / "results"
FIGURES = OUTPUTS / "figures"
ARTIFACTS = OUTPUTS / "artifacts"  # bundled, versioned deployable model


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
