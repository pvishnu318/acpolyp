"""Command line entry points: predict, evaluate (bundle parity check), serve.

python -m acpolyp.cli predict path/to/frame.jpg --cam out.png
python -m acpolyp.cli evaluate            # re-runs the bundle on raw test images, compares with training metrics
python -m acpolyp.cli serve               # web app on http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from .data.contracts import CLASSES
from .inference import Predictor
from .paths import ARTIFACTS, KVASIR_RAW, PROCESSED
from .training.metrics import full_metrics


def cmd_predict(a: argparse.Namespace) -> None:
    arr = cv2.imread(a.image)
    if arr is None:
        raise SystemExit(f"cannot read image: {a.image}")
    p = Predictor(ARTIFACTS / a.bundle)
    r = p.predict(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), with_cam=bool(a.cam))
    print(
        json.dumps(
            {
                "top": r.top,
                "confidence": round(r.confidence, 4),
                "polyp_probability": round(r.polyp_prob, 4),
                "polyp_flag": r.polyp_flag,
                "conformal_set": r.conformal_set,
                "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, r.probs)},
            },
            indent=2,
        )
    )
    if a.cam:
        cam = cv2.resize(r.cam, r.canon.shape[:2][::-1])
        heat = cv2.cvtColor(
            cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
        )
        cv2.imwrite(
            a.cam, cv2.cvtColor((r.canon * 0.55 + heat * 0.45).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        print("saved Grad-CAM overlay to", a.cam)


def cmd_evaluate(a: argparse.Namespace) -> None:
    """Run the bundle on raw test JPEGs (full preprocessing path) and compare, image by image, with the logits
    stored at training time. This is the train/serve parity check."""
    from .paths import RUNS
    from .training.metrics import softmax

    meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
    all_test = meta[meta.split == "test"]
    test = all_test.sample(a.limit, random_state=0) if a.limit else all_test
    manifest = json.loads((ARTIFACTS / a.bundle / "manifest.json").read_text())
    p = Predictor(ARTIFACTS / a.bundle)
    probs = []
    for path in test["path"]:
        rgb = cv2.cvtColor(cv2.imread(str(KVASIR_RAW / path)), cv2.COLOR_BGR2RGB)
        probs.append(p.predict(rgb).probs)
    probs = np.stack(probs)
    m = full_metrics(probs, test["label"].to_numpy())
    ref_logits = np.load(RUNS / manifest["source_run"] / "test_logits.npy")
    pos = np.array([all_test.index.get_loc(i) for i in test.index])
    ref = softmax(ref_logits[pos] / p.cal["temperature"])
    agree = float((probs.argmax(1) == ref.argmax(1)).mean())
    print(
        f"n={len(test)}  bundle: macro_f1={m['macro_f1']:.4f} acc={m['acc']:.4f} polyp_auroc={m['polyp_auroc']:.4f} ece={m['ece']:.4f}"
    )
    print(
        f"parity vs training-time logits: argmax agreement {agree:.4f}, max |dp| {np.abs(probs - ref).max():.4f}, mean |dp| {np.abs(probs - ref).mean():.5f}"
    )


def cmd_serve(_: argparse.Namespace) -> None:
    from .app.server import main

    main()


def main() -> None:
    ap = argparse.ArgumentParser(prog="acpolyp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("predict")
    s.add_argument("image")
    s.add_argument("--bundle", default="acpolyp-v1")
    s.add_argument("--cam", default=None)
    s.set_defaults(f=cmd_predict)
    s = sub.add_parser("evaluate")
    s.add_argument("--bundle", default="acpolyp-v1")
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(f=cmd_evaluate)
    s = sub.add_parser("serve")
    s.set_defaults(f=cmd_serve)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
