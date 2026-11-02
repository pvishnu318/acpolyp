"""Idempotent, verified dataset acquisition (Phase V, step 19).

Downloads are resumable (HTTP Range). Archives are untrusted: extraction rejects absolute
paths and '..' components (zip-slip) and always lands in a fresh, empty directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from ..paths import DOWNLOAD, DOWNLOAD_SEG, KVASIR_RAW, KVASIR_SEG_RAW, RAW, ensure

SOURCES = {
    "kvasir-v2": {
        "url": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
        "dest": DOWNLOAD / "kvasir-dataset-v2.zip",
        "bytes": 2_489_312_085,  # Content-Length reported by the server, 2026-11-02
        "extract_to": KVASIR_RAW,
        "expected_images": 8000,
    },
    "kvasir-seg": {
        "url": "https://datasets.simula.no/downloads/kvasir-seg.zip",
        "dest": DOWNLOAD_SEG / "kvasir-seg.zip",
        "bytes": 46_227_172,
        "extract_to": KVASIR_SEG_RAW,
        "expected_images": 2000,  # 1000 images + 1000 masks
    },
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> Path:
    spec = SOURCES[name]
    dest: Path = spec["dest"]
    ensure(dest.parent)
    total = spec["bytes"]
    for _ in range(50):
        have = dest.stat().st_size if dest.exists() else 0
        if have == total:
            return dest
        if have > total:
            dest.unlink()
            have = 0
        req = urllib.request.Request(spec["url"], headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, dest.open("ab") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except Exception as e:  # server drops long connections; resume
            print(f"[download] {name}: {e!r}; resuming from {dest.stat().st_size}")
    raise RuntimeError(f"could not complete download of {name}")


def safe_extract(zip_path: Path, out_dir: Path) -> int:
    """Extract into an empty directory, refusing path traversal. Returns number of members."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return sum(1 for _ in out_dir.rglob("*") if _.is_file())
    ensure(out_dir)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in archive: {bad}")
        for m in z.infolist():
            target = (out_dir / m.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
        z.extractall(out_dir)
        return len(z.infolist())


def main() -> None:
    manifest = {}
    for name, spec in SOURCES.items():
        path = download(name)
        digest = sha256_file(path)
        n = safe_extract(path, spec["extract_to"])
        imgs = sum(
            1
            for p in spec["extract_to"].rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        manifest[name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "members": n,
            "images_found": imgs,
            "expected_images": spec["expected_images"],
            "url": spec["url"],
        }
        print(name, manifest[name])
        if name == "kvasir-v2":
            assert imgs == spec["expected_images"], (
                f"expected {spec['expected_images']} images, found {imgs}"
            )
    ensure(RAW)
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
