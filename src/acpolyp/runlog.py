"""Structured per-run directories: config, git hash, command, environment, metrics (step 18)."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .paths import ROOT, RUNS, ensure


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "no-commit"
    except Exception:
        return "no-git"


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return info


class RunDir:
    """A timestamped run directory with config snapshot and JSON/CSV metric sinks."""

    def __init__(self, name: str, cfg: dict | None = None, root: Path = RUNS):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = ensure_dir(root / f"{name}")
        self.name = name
        self.stamp = stamp
        meta = {
            "name": name,
            "started": stamp,
            "git_commit": git_hash(),
            "command": " ".join(sys.argv),
            "environment": environment_info(),
        }
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if cfg is not None:
            (self.path / "config.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
        self._csv_header: list[str] | None = None

    def log_row(self, row: dict, filename: str = "metrics.csv") -> None:
        f = self.path / filename
        new = not f.exists()
        with f.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def save_json(self, obj: dict, filename: str) -> Path:
        p = self.path / filename
        p.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        return p


def ensure_dir(p: Path) -> Path:
    ensure(p)
    return p


def _json_default(o):  # numpy scalars / arrays
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
