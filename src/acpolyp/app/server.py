"""FastAPI backend + static single-page frontend (Phase XIX step 84).

    python -m acpolyp.app.server            # http://127.0.0.1:8000

Endpoints
  GET  /api/health          model/bundle status and hardware
  POST /api/predict         multipart image -> calibrated prediction, conformal set, contour + Grad-CAM overlays
  GET  /api/results         aggregated study results (written by scripts/export_results.py)
  GET  /api/samples         list of local sample images (only if the dataset exists on this machine)
  GET  /api/sample/{name}   one local sample image (never shipped with the repository)

Research prototype. Not a medical device; not for clinical use.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..data.contracts import CLASSES
from ..inference import Predictor
from ..paths import ARTIFACTS, KVASIR_RAW, RESULTS

STATIC = Path(__file__).parent / "static"
BUNDLE = ARTIFACTS / "acpolyp-v1"
MAX_BYTES = 12 * 1024 * 1024

app = FastAPI(
    title="ACPolyp", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json"
)
_predictor: Predictor | None = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not (BUNDLE / "model.safetensors").exists():
            raise HTTPException(503, "model bundle missing; run scripts/build_bundle.py")
        _predictor = Predictor(BUNDLE)
    return _predictor


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay_contour(img: np.ndarray, mask: np.ndarray, colour=(0, 255, 170)) -> np.ndarray:
    out = img.copy()
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, cnts, -1, colour, 2, cv2.LINE_AA)
    return out


def _overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam256 = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam256 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


@app.get("/api/health")
def health() -> dict:
    import torch

    ok = (BUNDLE / "model.safetensors").exists()
    return {
        "status": "ok" if ok else "no-model",
        "bundle": BUNDLE.name,
        "cuda": torch.cuda.is_available(),
        "classes": CLASSES,
        "research_use_only": True,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image too large (max 12 MB)")
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or min(arr.shape[:2]) < 64:
        raise HTTPException(400, "could not decode an image of at least 64x64 pixels")
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    p = predictor()
    t0 = time.perf_counter()
    pred = p.predict(rgb, with_cam=True)
    ms = (time.perf_counter() - t0) * 1000
    canon = pred.canon
    return JSONResponse(
        {
            "top": pred.top,
            "confidence": pred.confidence,
            "probs": {c: float(v) for c, v in zip(CLASSES, pred.probs)},
            "polyp_prob": pred.polyp_prob,
            "polyp_flag": pred.polyp_flag,
            "polyp_threshold": p.cal["polyp_threshold"],
            "conformal_set": pred.conformal_set,
            "conformal_alpha": p.cal["conformal_alpha"],
            "latency_ms": round(ms, 1),
            "mode": p.mode,
            "images": {
                "canonical": _png_b64(canon),
                "contour": _png_b64(_overlay_contour(canon, pred.ac_mask)),
                "cam": _png_b64(_overlay_cam(canon, pred.cam)) if pred.cam is not None else None,
                "mask": _png_b64(np.repeat((pred.ac_mask[..., None] * 255).astype(np.uint8), 3, 2)),
            },
            "ac_area_frac": float(pred.ac_mask.sum() / max(pred.valid.sum(), 1)),
            "disclaimer": "Research prototype trained on Kvasir v2. Not a medical device.",
        }
    )


@app.get("/api/results")
def results() -> JSONResponse:
    f = RESULTS / "site_results.json"
    if not f.exists():
        raise HTTPException(404, "results not exported yet; run scripts/export_results.py")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/samples")
def samples() -> dict:
    root = KVASIR_RAW / "kvasir-dataset-v2"
    if not root.exists():
        return {"available": False, "samples": []}
    rng = np.random.default_rng(7)
    out = []
    for c in CLASSES:
        files = sorted((root / c).glob("*.jpg"))
        for i in rng.choice(len(files), 2, replace=False):
            out.append({"name": f"{c}/{files[i].name}", "class": c})
    return {"available": True, "samples": out}


@app.get("/api/sample/{cls}/{name}")
def sample(cls: str, name: str) -> FileResponse:
    if cls not in CLASSES or "/" in name or "\\" in name or not name.endswith(".jpg"):
        raise HTTPException(400, "bad sample id")
    f = KVASIR_RAW / "kvasir-dataset-v2" / cls / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    uvicorn.run("acpolyp.app.server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
