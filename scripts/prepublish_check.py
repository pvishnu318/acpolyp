"""Pre-publication gate: refuses to pass while placeholders, data, weights or secrets are tracked.

Run from the repository root:  python scripts/prepublish_check.py
Exit code 0 means the repository is safe to push; 1 lists every problem found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"REPLACE_[A-Z_]+")
BLOCKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".joblib",
    ".zip",
}
BLOCKED_DIRS = ("data/", "outputs/runs/", "outputs/cache/")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})")
MAX_BYTES = 20 * 1024 * 1024
PLACEHOLDER_FILES = ["CITATION.cff", ".zenodo.json", "LICENSE", "pyproject.toml", "paper/main.tex"]


def tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8")
    return [ROOT / p for p in out.split("\0") if p]


def main() -> int:
    problems: list[str] = []
    files = tracked()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in BLOCKED_SUFFIXES or rel.startswith(BLOCKED_DIRS):
            problems.append(f"blocked file is tracked: {rel}")
        if p.exists() and p.stat().st_size > MAX_BYTES:
            problems.append(f"file over 20 MB is tracked: {rel}")
        if p.suffix.lower() in {
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".tex",
            ".cff",
            ".txt",
        }:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if SECRET.search(text):
                problems.append(f"possible secret in: {rel}")
    for name in PLACEHOLDER_FILES:
        f = ROOT / name
        if f.exists() and PLACEHOLDER.search(f.read_text(encoding="utf-8", errors="ignore")):
            problems.append(f"unreplaced REPLACE_* placeholder in: {name}")
    if (ROOT / ".env").exists() and ".env" in {p.name for p in files}:
        problems.append(".env is tracked")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    if dirty.strip():
        problems.append("working tree has uncommitted changes")
    if problems:
        print("NOT READY TO PUBLISH:")
        for q in problems:
            print("  -", q)
        return 1
    print(f"OK: {len(files)} tracked files, no placeholders, data, weights or secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
