"""Layer-wise probes, TreeSHAP and Grad-CAM pointing game (Phase XVIII steps 80-81).

Requires finished runs for ab_none / ab_acg and the LightGBM global+ac classical run.
"""

from __future__ import annotations

import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import timm
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import build_bundle  # noqa: E402

from acpolyp.ac.features import family_of  # noqa: E402
from acpolyp.ac.validate import SEG_DIR  # noqa: E402
from acpolyp.data.preprocess import canonicalise, crop_resize_mask  # noqa: E402
from acpolyp.evaluation import figstyle as fs  # noqa: E402
from acpolyp.evaluation.explain import cam_iou, gradcam, pointing_game  # noqa: E402
from acpolyp.inference import Predictor  # noqa: E402
from acpolyp.models.embed import to_tensor  # noqa: E402
from acpolyp.models.factory import build_model  # noqa: E402
from acpolyp.paths import ARTIFACTS, PROCESSED, RESULTS, RUNS  # noqa: E402
from acpolyp.training.classical import feature_set, load_meta, make_model  # noqa: E402


@torch.inference_mode()
def stage_features(model: torch.nn.Module, imgs: np.ndarray, dev: str = "cuda") -> list[np.ndarray]:
    """Global-average-pooled output of each ConvNeXt stage for all images."""
    stages = model.stages
    store: list[torch.Tensor] = []
    hooks = [
        s.register_forward_hook(lambda m, i, o: store.append(o.float().mean((2, 3)).cpu()))
        for s in stages
    ]
    outs: list[list[torch.Tensor]] = [[] for _ in stages]
    model.eval().to(dev)
    for i in range(0, len(imgs), 64):
        store.clear()
        x = to_tensor(imgs[i : i + 64], 224, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            model.forward_features(x)
        for k, t in enumerate(store):
            outs[k].append(t)
    for h in hooks:
        h.remove()
    return [torch.cat(o).numpy() for o in outs]


def layerwise(meta: pd.DataFrame) -> dict:
    imgs = np.load(PROCESSED / "images_canon_clean.npy", mmap_mode="r")
    y = meta["label"].to_numpy()
    tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()
    res = {}
    for name in ("pretrained", "finetuned"):
        if name == "pretrained":
            m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
        else:
            d = sorted((RUNS / "ab_none").glob("seed0"))[0]
            cfg = json.loads(
                json.dumps(__import__("yaml").safe_load((d / "config.yaml").read_text()))
            )
            cfg["model"]["pretrained"] = False
            net = build_model(cfg)
            net.load_state_dict(load_file(str(d / "best.safetensors")))
            m = net.backbone
        feats = stage_features(m, imgs)
        f1s = []
        for f in feats:
            sc = StandardScaler().fit(f[tr])
            clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(f[tr]), y[tr])
            f1s.append(float(f1_score(y[te], clf.predict(sc.transform(f[te])), average="macro")))
        res[name] = f1s
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for i, (k, nm) in enumerate(
        (("pretrained", "ImageNet-22k pretrained"), ("finetuned", "fine-tuned on Kvasir"))
    ):
        ax.plot(
            range(1, 5),
            res[k],
            marker=fs.MARKERS[i],
            color=fs.SERIES[i],
            ls=fs.LINESTYLES[i],
            label=nm,
        )
    ax.set_xticks(range(1, 5), [f"stage {i}" for i in range(1, 5)])
    ax.set_ylabel("linear-probe test macro-F1")
    ax.legend()
    ax.set_title("Layer-wise linear probes (ConvNeXt-T)")
    fs.save(fig, "layerwise_probe")
    return res


def shap_analysis(meta: pd.DataFrame) -> dict:
    X, names = feature_set("global+ac")
    y = meta["label"].to_numpy()
    tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()
    params = json.loads(
        (RUNS / "classical" / "lgbm__global_ac" / "seed0" / "summary.json").read_text()
    )["params"]
    model = make_model("lgbm", params, 0).fit(X[tr], y[tr])
    sub = np.random.default_rng(0).choice(np.where(te)[0], 600, replace=False)
    sv = shap.TreeExplainer(model).shap_values(X[sub])
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv  # (n, f, classes)
    imp = np.abs(sv).mean((0, 2))
    order = np.argsort(-imp)[:20]
    fam_col = {"global": fs.BLUE, "shape": fs.ORANGE, "boundary": fs.AQUA, "region": fs.VIOLET}
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.barh(
        range(20)[::-1], imp[order], color=[fam_col[family_of(names[i])] for i in order], height=0.7
    )
    ax.set_yticks(range(20)[::-1], [names[i] for i in order], fontsize=7)
    ax.set_xlabel("mean |SHAP| (all classes)")
    for fam, c in fam_col.items():
        ax.barh([], [], color=c, label=fam)
    ax.legend(loc="lower right")
    ax.set_title("TreeSHAP: LightGBM on global + contour features")
    fs.save(fig, "shap_top20")
    fam_imp = {f: float(imp[[family_of(n) == f for n in names]].sum() / imp.sum()) for f in fam_col}
    return {
        "top20": [
            {"feature": names[i], "family": family_of(names[i]), "mean_abs_shap": float(imp[i])}
            for i in order
        ],
        "family_share": fam_imp,
    }


def pointing(tags: list[str]) -> dict:
    m = pd.read_parquet(PROCESSED / "seg_match.parquet")
    names = m[m.matched & (m.split == "test")].seg_name.tolist()
    out = {}
    for tag in tags:
        build_bundle.main(version=f"eval-{tag}", tag=tag)
        p = Predictor(ARTIFACTS / f"eval-{tag}")
        pg, iou, in_ac = [], [], []
        for n in names:
            rgb = cv2.cvtColor(cv2.imread(str(SEG_DIR / "images" / f"{n}.jpg")), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(SEG_DIR / "masks" / f"{n}.jpg"), cv2.IMREAD_GRAYSCALE)
            c = canonicalise(rgb, 256)
            gt224 = cv2.resize(
                (crop_resize_mask(gt, c.bbox, 256) > 127).astype(np.uint8),
                (224, 224),
                interpolation=cv2.INTER_NEAREST,
            )
            b = p.prepare(rgb)
            cam = gradcam(p.model, b, p.device, 6)
            pg.append(pointing_game(cam, gt224))
            iou.append(cam_iou(cam, gt224))
            ac224 = cv2.resize(b["mask256"], (224, 224), interpolation=cv2.INTER_NEAREST)
            in_ac.append(pointing_game(cam, ac224))
        out[tag] = {
            "n": len(names),
            "pointing_game": float(np.mean(pg)),
            "cam_top20_iou": float(np.mean(iou)),
            "cam_peak_inside_contour": float(np.mean(in_ac)),
        }
    return out


def main() -> None:
    meta = load_meta()
    res = {"layerwise_probe": layerwise(meta)}
    try:
        res["shap"] = shap_analysis(meta)
    except FileNotFoundError:
        print("skipping SHAP: classical run missing")
    res["gradcam_pointing"] = pointing(["ab_none", "ab_acg"])
    (RESULTS / "interpretability.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "shap"}, indent=2))


if __name__ == "__main__":
    main()
