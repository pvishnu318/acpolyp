"""Declarative table of every deep-learning experiment in the study; resumable (finished runs are skipped).

python scripts/run_deep.py --group backbones
python scripts/run_deep.py --group all --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time

from acpolyp.config import load_config
from acpolyp.training.train import run

CN = "convnext_tiny"

# group -> {tag: dotted-override list}
GROUPS: dict[str, dict[str, list[str]]] = {
    # backbone benchmark, plain fine-tuning (no active-contour guidance)
    "backbones": {
        "bb_resnet50": ["model.name=resnet50", "model.ac_mode=none"],
        "bb_efficientnetv2_s": ["model.name=efficientnetv2_s", "model.ac_mode=none"],
        "bb_swin_tiny": ["model.name=swin_tiny", "model.ac_mode=none"],
        "bb_dinov2_s": ["model.name=dinov2_s", "model.ac_mode=none", "train.backbone_lr_mult=0.1"],
    },
    # component ablation + controls on the strongest NCHW backbone
    "ablation": {
        "ab_none": [f"model.name={CN}", "model.ac_mode=none"],
        "ab_gfeat": [
            f"model.name={CN}",
            "model.ac_mode=gfeat",
        ],  # extra hand-crafted features, no contour
        "ab_ac_pool": [f"model.name={CN}", "model.ac_mode=ac_pool"],  # contour-masked pooling only
        "ab_ac_feat": [f"model.name={CN}", "model.ac_mode=ac_feat"],  # contour features only
        "ab_acg": [f"model.name={CN}", "model.ac_mode=acg"],  # full model
        "ab_acg_shuffle": [
            f"model.name={CN}",
            "model.ac_mode=acg",
            "model.mask_source=shuffle",
        ],  # wrong-image contour
        "ab_acg_ellipse": [
            f"model.name={CN}",
            "model.ac_mode=acg",
            "model.mask_source=ellipse",
        ],  # image-independent mask
    },
    # does guidance transfer to another backbone?
    "generality": {
        "gen_resnet50_acg": ["model.name=resnet50", "model.ac_mode=acg"],
    },
    # learning curves: does the contour prior help more when training data is scarce?
    "lowdata": {
        f"ld_{m}_{int(f * 100)}": [f"model.name={CN}", f"model.ac_mode={m}", f"data.train_frac={f}"]
        for f in (0.1, 0.25, 0.5)
        for m in ("none", "acg")
    },
    # learning-rate sensitivity (deep hyper-parameters are NOT tuned per variant; this shows how much that matters)
    "sensitivity": {
        f"lr_{m}_{tag}": [f"model.name={CN}", f"model.ac_mode={m}", f"train.lr={v}"]
        for tag, v in (("half", 1.0e-4), ("double", 4.0e-4))
        for m in ("none", "acg")
    },
    # shortcut and leakage studies
    "shortcut": {
        "sc_naive_frames": [f"model.name={CN}", "model.ac_mode=none", "data.image_key=naive"],
        "sc_random_split": [
            f"model.name={CN}",
            "model.ac_mode=none",
            "data.split_col=split_random",
        ],
    },
}
GROUPS["all"] = {
    k: v
    for g in ("backbones", "ablation", "generality", "shortcut", "lowdata", "sensitivity")
    for k, v in GROUPS[g].items()
}
# bb_convnext_tiny == ab_none and gen_resnet50_none == bb_resnet50 (identical configs); aliased in analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=list(GROUPS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these tags")
    ap.add_argument(
        "--set", nargs="*", default=[], help="extra overrides for every run, e.g. train.epochs=2"
    )
    ap.add_argument("--prefix", default="", help="tag prefix (use for smoke tests)")
    a = ap.parse_args()
    t0 = time.time()
    for tag, ov in GROUPS[a.group].items():
        if a.only and tag not in a.only:
            continue
        cfg = load_config("ac_tuned.yaml", ov + a.set)
        for s in a.seeds:
            run(cfg, f"{a.prefix}{tag}", s)
    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
