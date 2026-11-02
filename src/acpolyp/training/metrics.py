"""Numerically stable metrics computed from accumulated epoch predictions (Phase XIII step 54)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..data.contracts import CLASSES, POLYP_IDX

EPS = 1e-8


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, list[float]]:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out: dict[str, list[float]] = {"conf": [], "acc": [], "count": []}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        out["count"].append(int(m.sum()))
        out["conf"].append(float(conf[m].mean()) if m.any() else float((lo + hi) / 2))
        out["acc"].append(float(acc[m].mean()) if m.any() else float("nan"))
    return out


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1)).mean())


def polyp_binary(
    probs: np.ndarray, y: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Polyp-vs-rest view using P(polyp) from the 8-class posterior."""
    yb = (y == POLYP_IDX).astype(int)
    s = probs[:, POLYP_IDX]
    out = {
        "polyp_auroc": float(roc_auc_score(yb, s)),
        "polyp_auprc": float(average_precision_score(yb, s)),
    }
    thr = 0.5 if threshold is None else threshold
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
    out.update(
        {
            "polyp_sens": float(tp / max(tp + fn, 1)),
            "polyp_spec": float(tn / max(tn + fp, 1)),
            "polyp_ppv": float(tp / max(tp + fp, 1)),
            "polyp_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "polyp_threshold": float(thr),
        }
    )
    return out


def full_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    m = {
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc_ovr": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        "nll": nll(probs, y),
        "brier": brier_multiclass(probs, y),
        "ece": expected_calibration_error(probs, y),
    }
    m.update(polyp_binary(probs, y))
    return m


def per_class_table(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(
        y, probs.argmax(1), labels=range(len(CLASSES)), zero_division=0
    )
    return [
        {
            "class": c,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, c in enumerate(CLASSES)
    ]


def confusion(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, probs.argmax(1), labels=range(len(CLASSES)))
