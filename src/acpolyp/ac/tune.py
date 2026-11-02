"""Tune the active-contour hyperparameters against Kvasir-SEG masks (Phase XV step 66).

Protocol: optimise mean Dice on `train`-matched images, pick the trial by `val`-matched Dice, and
report once on `test`-matched images. Classification labels are never used.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import pandas as pd
import yaml

from ..config import load_config
from ..paths import CONFIGS, PROCESSED, RESULTS, ensure
from .segment import central_ellipse, segment
from .validate import dice_iou, load_pair

_PAIRS: dict[str, tuple] = {}


def _init(names: list[str], work: int) -> None:
    for n in names:
        rgb, fov, gt, _ = load_pair(n, work)
        _PAIRS[n] = (rgb, fov, gt)


def _score(args: tuple[list[str], dict]) -> list[tuple[float, float, float, float, bool]]:
    names, p = args
    out = []
    for n in names:
        rgb, fov, gt = _PAIRS[n]
        r = segment(rgb, fov, p)
        d, i = dice_iou(r.mask, gt)
        out.append(
            (d, i, dice_iou(r.cv_mask, gt)[0], dice_iou(central_ellipse(fov), gt)[0], r.fallback)
        )
    return out


def sample_params(trial: optuna.Trial, base: dict) -> dict:
    p = dict(base)
    p["channel"] = trial.suggest_categorical("channel", ["lab_pc1", "a", "L", "sat"])
    p["dog_frac"] = trial.suggest_categorical("dog_frac", [0.0, 0.08, 0.12, 0.2, 0.3])
    p["init_frac"] = trial.suggest_float("init_frac", 0.1, 0.4)
    p["cv_iters"] = trial.suggest_int("cv_iters", 20, 120, step=20)
    p["cv_smoothing"] = trial.suggest_int("cv_smoothing", 1, 5)
    p["cv_lambda_in"] = trial.suggest_float("cv_lambda_in", 0.5, 3.0, log=True)
    p["cv_lambda_out"] = 1.0
    p["gac_iters"] = trial.suggest_categorical("gac_iters", [0, 10, 25, 50])
    p["gac_alpha"] = trial.suggest_float("gac_alpha", 20, 300, log=True)
    p["gac_sigma"] = trial.suggest_float("gac_sigma", 1.0, 4.0)
    p["gac_smoothing"] = trial.suggest_int("gac_smoothing", 1, 3)
    return p


def evaluate_split(pool, names: list[str], p: dict, workers: int) -> np.ndarray:
    chunks = [names[i::workers] for i in range(workers)]
    res = pool.map(_score, [(c, p) for c in chunks if c])
    return np.array([r for part in res for r in part], dtype=float)


def main(n_trials: int = 80, n_tune: int = 300, workers: int = 10) -> None:
    cfg = load_config()
    base = cfg["ac"]
    m = pd.read_parquet(PROCESSED / "seg_match.parquet")
    m = m[m.matched]
    tune = (
        m[m.split == "train"]
        .sample(n=min(n_tune, (m.split == "train").sum()), random_state=0)
        .seg_name.tolist()
    )
    val = m[m.split == "val"].seg_name.tolist()
    test = m[m.split == "test"].seg_name.tolist()
    allnames = tune + val + test
    with ProcessPoolExecutor(
        workers, initializer=_init, initargs=(allnames, base["work_size"])
    ) as pool:
        # every worker loads every pair (small: 128x128) so any chunking works
        def objective(trial: optuna.Trial) -> float:
            p = sample_params(trial, base)
            r = evaluate_split(pool, tune, p, workers)
            trial.set_user_attr("fallback", float(r[:, 4].mean()))
            return float(r[:, 0].mean())

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
        )
        study.optimize(objective, n_trials=n_trials)
        top = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
        val_scores = []
        for t in top:
            p = {**base, **t.params, "cv_lambda_out": 1.0}
            val_scores.append(float(evaluate_split(pool, val, p, workers)[:, 0].mean()))
        best_t = top[int(np.argmax(val_scores))]
        best = {**base, **best_t.params, "cv_lambda_out": 1.0}
        rt = evaluate_split(pool, test, best, workers)
        rd = evaluate_split(pool, test, base, workers)
    report = {
        "n_trials": n_trials,
        "n_tune": len(tune),
        "n_val": len(val),
        "n_test": len(test),
        "best_params": best_t.params,
        "train_dice": best_t.value,
        "val_dice": max(val_scores),
        "test": {
            "dice_tuned": float(rt[:, 0].mean()),
            "iou_tuned": float(rt[:, 1].mean()),
            "dice_cv_only": float(rt[:, 2].mean()),
            "dice_central_ellipse": float(rt[:, 3].mean()),
            "dice_default_params": float(rd[:, 0].mean()),
            "fallback_rate": float(rt[:, 4].mean()),
        },
        "test_per_image_dice_tuned": rt[:, 0].tolist(),
        "test_per_image_dice_ellipse": rt[:, 3].tolist(),
    }
    ensure(RESULTS)
    (RESULTS / "ac_tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (CONFIGS / "ac_tuned.yaml").write_text(
        yaml.safe_dump({"ac": best}, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("test_per")}, indent=2))


if __name__ == "__main__":
    main()
