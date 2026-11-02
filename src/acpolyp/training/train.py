"""Train one model for one seed and write a complete, self-describing run directory.

python -m acpolyp.training.train --tag convnext_acg --set model.name=convnext_tiny model.ac_mode=acg --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import KvasirData, KvasirSplit, make_loader
from ..models.acgnet import count_params
from ..models.factory import build_model
from ..paths import RUNS, ensure
from ..runlog import RunDir
from ..seed import seed_everything
from .calibration import fit_temperature, temperature_probs
from .engine import fit, latency, predict
from .metrics import full_metrics, per_class_table, polyp_binary, softmax

# which inputs each mode needs
MODE_FEATS = {
    "none": None,
    "ac_pool": None,
    "ac_feat": ["shape", "boundary", "region"],
    "acg": ["shape", "boundary", "region"],
    "gfeat": ["global"],
}


def youden_threshold(p: np.ndarray, y_bin: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_bin, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def run(cfg: dict, tag: str, seed: int, force: bool = False, quiet: bool = False) -> dict:
    run_path = RUNS / tag / f"seed{seed}"
    if (run_path / "summary.json").exists() and not force:
        return json.loads((run_path / "summary.json").read_text())
    ensure(run_path)
    log = (lambda *_: None) if quiet else print
    rd = RunDir(f"{tag}/seed{seed}", cfg)
    seed_everything(seed)
    m = cfg["model"]
    mode = m.get("ac_mode", "none")
    feats = m.get("feat_families", MODE_FEATS[mode])
    data = KvasirData(
        cfg["data"].get("image_key", "canon"),
        m.get("mask_source", "ac"),
        feats,
        cfg["data"].get("split_col", "split"),
    )
    scaler = data.fit_scaler()
    cfg["seed_run"] = seed
    size, bs = cfg["data"]["input_size"], cfg["train"]["batch_size"]
    tr = KvasirSplit(
        data, "train", scaler, indices=data.idx("train", cfg["data"].get("train_frac", 1.0))
    )
    va = KvasirSplit(data, "val", scaler)
    te = KvasirSplit(data, "test", scaler)
    tl, vl, tel = (
        make_loader(tr, bs, True, seed),
        make_loader(va, 64, False, seed),
        make_loader(te, 64, False, seed),
    )

    if (
        scaler is not None
    ):  # persisted so that inference applies the identical standardisation (step 42)
        mu, sd, lo, hi = scaler
        rd.save_json(
            {"names": data.feat_names, "mu": mu, "sd": sd, "lo": lo, "hi": hi}, "scaler.json"
        )
    model = build_model(cfg, n_feats=0 if data.feats is None else data.feats.shape[1])
    total, trainable = count_params(model)
    log(
        f"[{tag} seed{seed}] mode={mode} params={total / 1e6:.2f}M trainable={trainable / 1e6:.2f}M train={len(tr)} val={len(va)} test={len(te)}"
    )
    fit_info = fit(model, tl, vl, cfg, rd.path, log=log)

    val_logits, val_y = predict(model, vl, "cuda", cfg["train"]["amp"])
    test_logits, test_y = predict(model, tel, "cuda", cfg["train"]["amp"])
    np.save(rd.path / "val_logits.npy", val_logits)
    np.save(rd.path / "test_logits.npy", test_logits)
    np.save(rd.path / "val_y.npy", val_y)
    np.save(rd.path / "test_y.npy", test_y)

    # post-processing fitted on validation only (steps 68-69)
    T = fit_temperature(val_logits, val_y)
    pv_cal, pt_cal = temperature_probs(val_logits, T), temperature_probs(test_logits, T)
    thr = youden_threshold(pv_cal[:, 6], (val_y == 6).astype(int))
    test_raw, test_cal = full_metrics(softmax(test_logits), test_y), full_metrics(pt_cal, test_y)
    test_cal.update(
        {
            f"{k}_tuned": v
            for k, v in polyp_binary(pt_cal, test_y, thr).items()
            if k != "polyp_auroc" and k != "polyp_auprc"
        }
    )

    ex = {k: v for k, v in next(iter(tel)).items()}
    ex1 = {k: v[:1] for k, v in ex.items()}
    summary = {
        "tag": tag,
        "seed": seed,
        "mode": mode,
        "backbone": m["name"],
        "params_total": total,
        "params_trainable": trainable,
        "temperature": T,
        "polyp_threshold": thr,
        "test_raw": test_raw,
        "test_calibrated": test_cal,
        "val": full_metrics(pv_cal, val_y),
        "per_class_test": per_class_table(pt_cal, test_y),
        "latency_ms_bs1": latency(model, ex1, size=size),
        "latency_ms_bs64": latency(model, ex, size=size),
        "ckpt_mb": (rd.path / "best.safetensors").stat().st_size / 2**20,
        **{k: v for k, v in fit_info.items() if k != "history"},
    }
    rd.save_json(summary, "summary.json")
    rd.save_json({"history": fit_info["history"]}, "history.json")
    log(
        f"[{tag} seed{seed}] TEST macro_f1={test_raw['macro_f1']:.4f} acc={test_raw['acc']:.4f} polyp_auroc={test_raw['polyp_auroc']:.4f} "
        f"ece raw/cal={test_raw['ece']:.4f}/{test_cal['ece']:.4f} time={fit_info['train_time_s']:.0f}s"
    )
    del model
    torch.cuda.empty_cache()
    return summary


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="run several seeds sequentially"
    )
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=5")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse(argv)
    cfg = load_config("ac_tuned.yaml" if a.config is None else a.config, a.set)
    for s in a.seeds if a.seeds else [a.seed]:
        t0 = time.time()
        run(cfg, a.tag, s, a.force)
        print(f"done {a.tag} seed{s} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
