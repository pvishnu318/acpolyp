"""Load completed runs, aggregate across seeds and compare models with paired statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..paths import RUNS
from ..training.calibration import temperature_probs
from ..training.metrics import full_metrics, softmax
from . import stats

K = 8
METRICS = [
    "macro_f1",
    "acc",
    "mcc",
    "polyp_auroc",
    "polyp_auprc",
    "ece",
    "nll",
    "brier",
    "auroc_ovr",
    "bal_acc",
]
ALIASES = {
    "bb_convnext_tiny": "ab_none",
    "gen_resnet50_none": "bb_resnet50",
}  # identical configs, trained once


@dataclass
class Group:
    tag: str
    seeds: list[int] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)
    probs: list[np.ndarray] = field(default_factory=list)  # temperature-calibrated, per seed
    raw_probs: list[np.ndarray] = field(default_factory=list)
    val_probs: list[np.ndarray] = field(default_factory=list)
    y: np.ndarray | None = None
    val_y: np.ndarray | None = None
    kind: str = "deep"

    def metric_matrix(self, name: str) -> np.ndarray:
        return np.array([full_metrics(p, self.y)[name] for p in self.probs])

    def mean_sd(self, name: str) -> tuple[float, float]:
        v = self.metric_matrix(name)
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    def ensemble(self) -> np.ndarray:
        return np.mean(self.probs, axis=0)


def load_deep(tag: str) -> Group | None:
    tag = ALIASES.get(tag, tag)
    g = Group(tag)
    for d in sorted((RUNS / tag).glob("seed*")):
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        lt, lv = np.load(d / "test_logits.npy"), np.load(d / "val_logits.npy")
        T = s["temperature"]
        g.seeds.append(int(d.name[4:]))
        g.summaries.append(s)
        g.probs.append(temperature_probs(lt, T))
        g.raw_probs.append(softmax(lt))
        g.val_probs.append(temperature_probs(lv, T))
        g.y, g.val_y = np.load(d / "test_y.npy"), np.load(d / "val_y.npy")
    return g if g.seeds else None


def load_classical(name: str) -> Group | None:
    g = Group(name, kind="classical")
    for d in sorted((RUNS / "classical" / name).glob("seed*")):
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        g.seeds.append(int(d.name[4:]))
        g.summaries.append(s)
        g.probs.append(np.load(d / "test_probs.npy"))
        g.y = np.load(d / "test_y.npy")
    return g if g.seeds else None


def fast_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def seedavg_bootstrap_diff(a: Group, b: Group, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the test set; for each resample average macro-F1 over seeds for each model and diff."""
    rng = np.random.default_rng(seed)
    y = a.y
    n = len(y)
    pa = [p.argmax(1) for p in a.probs]
    pb = [p.argmax(1) for p in b.probs]
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = np.mean([fast_macro_f1(y[idx], p[idx]) for p in pa]) - np.mean(
            [fast_macro_f1(y[idx], p[idx]) for p in pb]
        )
    obs = np.mean([fast_macro_f1(y, p) for p in pa]) - np.mean([fast_macro_f1(y, p) for p in pb])
    return {
        "diff": float(obs),
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
    }


def compare(a: Group, b: Group, n_boot: int = 2000) -> dict:
    """a (candidate) vs b (baseline): bootstrap CI, McNemar + DeLong on seed-ensembles, seed-level tests."""
    out = {"a": a.tag, "b": b.tag, **seedavg_bootstrap_diff(a, b, n_boot)}
    ea, eb = a.ensemble(), b.ensemble()
    y = a.y
    out["mcnemar_ensemble"] = stats.mcnemar(
        (ea.argmax(1) == y).astype(int), (eb.argmax(1) == y).astype(int)
    )
    yb = (y == 6).astype(int)
    out["delong_polyp_ensemble"] = stats.delong_test(ea[:, 6], eb[:, 6], yb)
    n = min(len(a.seeds), len(b.seeds))
    out["seed_level_f1"] = stats.seed_level_test(
        a.metric_matrix("macro_f1")[:n], b.metric_matrix("macro_f1")[:n]
    )
    return out
