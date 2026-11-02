"""Single-image inference with the bundled model (Phase XIX step 84).

A bundle directory contains: model.safetensors, config.yaml, scaler.json, calibration.json.
The preprocessing path (canonicalise -> active contour -> features -> resize/normalise) reuses exactly the
training-time functions, so train/serve skew is limited to JPEG decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from .ac.features import all_features
from .ac.segment import segment
from .data.augment import eval_transform
from .data.contracts import CLASSES, POLYP_IDX
from .data.preprocess import canonicalise
from .models.factory import build_model
from .training.calibration import conformal_sets
from .training.metrics import softmax


@dataclass
class Prediction:
    probs: np.ndarray  # (8,) temperature-calibrated
    top: str
    confidence: float
    polyp_prob: float
    polyp_flag: bool  # P(polyp) >= validation-tuned threshold
    conformal_set: list[str]  # classes in the (1-alpha)-coverage prediction set
    canon: np.ndarray  # (256,256,3) canonicalised image
    valid: np.ndarray  # (256,256) valid FOV
    ac_mask: np.ndarray  # (256,256) active-contour region
    cam: np.ndarray | None  # (224,224) Grad-CAM in [0,1]


class Predictor:
    def __init__(self, bundle: str | Path, device: str | None = None):
        b = Path(bundle)
        self.cfg = yaml.safe_load((b / "config.yaml").read_text(encoding="utf-8"))
        self.scaler = (
            json.loads((b / "scaler.json").read_text()) if (b / "scaler.json").exists() else None
        )
        self.cal = json.loads((b / "calibration.json").read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {**self.cfg, "model": {**self.cfg["model"], "pretrained": False}}
        n_feats = len(self.scaler["names"]) if self.scaler else 0
        self.model = build_model(cfg, n_feats=n_feats)
        self.model.load_state_dict(load_file(str(b / "model.safetensors")))
        self.model.to(self.device).eval()
        self.size = self.cfg["data"]["input_size"]
        self.mode = self.cfg["model"].get("ac_mode", "none")

    # ------------------------------------------------------------------ preprocessing
    def prepare(self, rgb: np.ndarray) -> dict:
        c = canonicalise(rgb, self.cfg["data"]["image_size"])
        return self.prepare_canonical(c.image, c.fov)

    def prepare_canonical(self, canon: np.ndarray, fov: np.ndarray) -> dict:
        """Contour -> features -> model tensors from an already canonicalised (S,S,3) uint8 frame."""
        p = self.cfg["ac"]
        w = p["work_size"]
        small = cv2.resize(canon, (w, w), interpolation=cv2.INTER_AREA)
        valid_w = cv2.resize(fov, (w, w), interpolation=cv2.INTER_NEAREST)
        res = segment(small, valid_w, p)
        feats = None
        if self.scaler:
            allf = all_features(small, valid_w, res.mask)
            raw = np.array([allf[n] for n in self.scaler["names"]], np.float32)
            lo, hi, mu, sd = (
                np.array(self.scaler[k], np.float32) for k in ("lo", "hi", "mu", "sd")
            )
            feats = torch.from_numpy(((np.clip(raw, lo, hi) - mu) / sd).astype(np.float32))[None]
        S = canon.shape[0]
        mask256 = cv2.resize(res.mask, (S, S), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(canon).permute(2, 0, 1)[None]  # uint8 (1,3,S,S)
        m = torch.from_numpy(mask256)[None, None].to(torch.uint8)  # uint8 (1,1,S,S)
        v = torch.from_numpy(fov)[None, None].to(torch.uint8)
        x, mm, vv = eval_transform(img, m, v, self.size)  # identical to validation/test
        return {
            "x": x,
            "mask": mm,
            "valid": vv,
            "feats": feats,
            "canon": canon,
            "valid256": fov,
            "mask256": mask256,
        }

    # ------------------------------------------------------------------ prediction
    @torch.inference_mode()
    def _logits(self, b: dict) -> torch.Tensor:
        kw = {"mask": b["mask"].to(self.device), "valid": b["valid"].to(self.device)}
        if b["feats"] is not None:
            kw["feats"] = b["feats"].to(self.device)
        return self.model(b["x"].to(self.device), **kw).float().cpu()

    def predict(self, rgb: np.ndarray, with_cam: bool = False) -> Prediction:
        b = self.prepare(rgb)
        logits = self._logits(b).numpy()
        probs = softmax(logits / self.cal["temperature"])[0]
        sets = conformal_sets(probs[None], self.cal["conformal_qhat"])[0]
        cam = None
        if with_cam:
            from .evaluation.explain import gradcam

            cam = gradcam(self.model, b, self.device, int(probs.argmax()))
        return Prediction(
            probs=probs,
            top=CLASSES[int(probs.argmax())],
            confidence=float(probs.max()),
            polyp_prob=float(probs[POLYP_IDX]),
            polyp_flag=bool(probs[POLYP_IDX] >= self.cal["polyp_threshold"]),
            conformal_set=[CLASSES[i] for i in np.where(sets)[0]],
            canon=b["canon"],
            valid=b["valid256"],
            ac_mask=b["mask256"],
            cam=cam,
        )
