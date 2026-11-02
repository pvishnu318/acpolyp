"""Trivial, heuristic and classical-ML baselines on tabular feature sets (Phase XII steps 49-51, XV 66).

All preprocessing is fit on the training split only; hyper-parameters are tuned on the validation split
(Optuna/TPE) and the test split is evaluated exactly once per frozen configuration.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..ac.features import family_of
from ..models.embed import cached
from ..paths import PROCESSED, RUNS
from ..runlog import ensure_dir as ensure
from .metrics import full_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
K = 8
NJOBS = int(os.environ.get("ACP_NJOBS", 10))


class Winsorise(BaseEstimator, TransformerMixin):
    """Clip each feature to the [1, 99] percentile range seen in training (heavy-tail guard)."""

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.percentile(X, [1, 99], axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def load_meta() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "metadata_clean.parquet")


def feature_set(name: str) -> tuple[np.ndarray, list[str]]:
    """Named feature sets used in the study. Rows are aligned with metadata_clean.parquet."""
    ac = pd.read_parquet(PROCESSED / "ac_features.parquet")
    ell = pd.read_parquet(PROCESSED / "ac_features_ellipse.parquet")
    shf = pd.read_parquet(PROCESSED / "ac_features_shuffle.parquet")
    cols = lambda df, fams: [c for c in df.columns if family_of(c) in fams]
    parts: dict[str, pd.DataFrame] = {
        "global": ac[cols(ac, ["global"])],
        "ac_shape": ac[cols(ac, ["shape"])],
        "ac_boundary": ac[cols(ac, ["boundary"])],
        "ac_region": ac[cols(ac, ["region"])],
        "ac_all": ac[cols(ac, ["shape", "boundary", "region"])],
        "global+ac": ac,
        "global+ellipse": pd.concat(
            [
                ac[cols(ac, ["global"])],
                ell[cols(ell, ["shape", "boundary", "region"])].add_prefix("ell_"),
            ],
            axis=1,
        ),
        "global+shuffle": pd.concat(
            [
                ac[cols(ac, ["global"])],
                shf[cols(shf, ["shape", "boundary", "region"])].add_prefix("shf_"),
            ],
            axis=1,
        ),
    }
    if name.startswith("emb:"):
        kept = np.load(PROCESSED / "kept_idx.npy")
        e = cached(name[4:], "canon", 224)[kept]
        return e.astype(np.float32), [f"e{i}" for i in range(e.shape[1])]
    df = parts[name]
    return df.to_numpy(np.float32), list(df.columns)


def _score(model, X, y):
    return float(f1_score(y, model.predict(X), average="macro"))


def make_model(kind: str, params: dict, seed: int):
    if kind == "logreg":
        return make_pipeline(
            Winsorise(),
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed, **params),
        )
    if kind == "svm_rbf":
        return make_pipeline(
            Winsorise(),
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=seed, **params),
        )
    if kind == "rf":
        return RandomForestClassifier(n_jobs=NJOBS, random_state=seed, **params)
    if kind == "lgbm":
        return lgb.LGBMClassifier(
            objective="multiclass", n_jobs=NJOBS, random_state=seed, verbose=-1, **params
        )
    if kind == "xgb":
        return xgb.XGBClassifier(
            objective="multi:softprob",
            n_jobs=NJOBS,
            random_state=seed,
            tree_method="hist",
            **params,
        )
    raise ValueError(kind)


def suggest(kind: str, t: optuna.Trial) -> dict:
    if kind == "logreg":
        return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if kind == "svm_rbf":
        return {
            "C": t.suggest_float("C", 1e-1, 1e2, log=True),
            "gamma": t.suggest_float("gamma", 1e-4, 1e-1, log=True),
        }
    if kind == "rf":
        return {
            "n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": t.suggest_int("max_depth", 8, 40, step=4),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 5),
            "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3]),
        }
    if kind == "lgbm":
        return {
            "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 63),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 40),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if kind == "xgb":
        return {
            "n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 9),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(kind)


def tune(kind: str, X, y, tr, va, n_trials: int, seed: int = 0) -> dict:
    def obj(t):
        m = make_model(kind, suggest(kind, t), seed).fit(X[tr], y[tr])
        return _score(m, X[va], y[va])

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params


def run_trivial(meta: pd.DataFrame) -> dict[str, dict]:
    """Metric floors: uniform-random and class-prior (majority) predictions, plus a centroid heuristic."""
    y = meta["label"].to_numpy()
    te = (meta["split"] == "test").to_numpy()
    tr = (meta["split"] == "train").to_numpy()
    out = {}
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(K), size=te.sum())  # random guesser with random scores
    out["random"] = full_metrics(p, y[te])
    prior = np.bincount(y[tr], minlength=K) / tr.sum()
    out["majority"] = full_metrics(
        np.tile(prior, (te.sum(), 1)) + 1e-9 * rng.random((te.sum(), K)), y[te]
    )
    # Heuristic: nearest class centroid on mean Lab colour inside the FOV (no learning beyond 8 means).
    X, names = feature_set("global")
    cols = [names.index(f"g_{c}_mean") for c in ("L", "a", "b")]
    cen = np.stack([X[tr][y[tr] == k][:, cols].mean(0) for k in range(K)])
    d = ((X[te][:, cols][:, None, :] - cen[None]) ** 2).sum(-1)
    pr = np.exp(-(d - d.min(1, keepdims=True)) / 50.0)
    out["colour_centroid_heuristic"] = full_metrics(pr / pr.sum(1, keepdims=True), y[te])
    return out


def run_all(
    configs: list[tuple[str, str]], seeds: list[int], n_trials: int, tag: str = "classical"
) -> None:
    meta = load_meta()
    y = meta["label"].to_numpy()
    tr, va, te = (np.where(meta["split"] == s)[0] for s in ("train", "val", "test"))
    root = ensure(RUNS / tag)
    (root / "trivial.json").write_text(json.dumps(run_trivial(meta), indent=2))
    for fs, kind in configs:
        name = f"{kind}__{fs.replace(':', '_').replace('+', '_')}"
        if all((root / name / f"seed{s}" / "summary.json").exists() for s in seeds):
            continue
        X, _ = feature_set(fs)
        t0 = time.time()
        best = tune(kind, X, y, tr, va, n_trials)
        deterministic = kind in ("logreg", "svm_rbf")
        for s in seeds:
            d = ensure(root / name / f"seed{s}")
            m = make_model(kind, best, s).fit(X[tr], y[tr])
            pt = m.predict_proba(X[te])
            np.save(d / "test_probs.npy", pt)
            np.save(d / "test_y.npy", y[te])
            res = {
                "set": fs,
                "model": kind,
                "seed": s,
                "params": best,
                "test": full_metrics(pt, y[te]),
                "val_macro_f1": _score(m, X[va], y[va]),
                "n_features": X.shape[1],
                "deterministic": deterministic,
            }
            (d / "summary.json").write_text(json.dumps(res, indent=2, default=str))
            if deterministic:
                for s2 in seeds[1:]:  # identical result; copy for uniform downstream handling
                    d2 = ensure(root / name / f"seed{s2}")
                    np.save(d2 / "test_probs.npy", pt)
                    np.save(d2 / "test_y.npy", y[te])
                    (d2 / "summary.json").write_text(
                        json.dumps({**res, "seed": s2}, indent=2, default=str)
                    )
                break
        r = json.loads((root / name / f"seed{seeds[0]}" / "summary.json").read_text())["test"]
        print(
            f"{name:34s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f} polyp_auroc={r['polyp_auroc']:.4f} ({time.time() - t0:.0f}s)"
        )


SETS_FOR_ABLATION = [
    "global",
    "ac_shape",
    "ac_boundary",
    "ac_region",
    "ac_all",
    "global+ac",
    "global+ellipse",
    "global+shuffle",
]

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfgs = [(fs, "lgbm") for fs in SETS_FOR_ABLATION]
    cfgs += [("global+ac", k) for k in ("logreg", "svm_rbf", "rf", "xgb")]
    cfgs += [(f"emb:{e}", "logreg") for e in ("vit_small_patch14_dinov2.lvd142m",)]
    run_all(cfgs, a.seeds, a.trials)
