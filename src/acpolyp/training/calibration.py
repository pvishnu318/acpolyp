"""Post-hoc temperature scaling and split-conformal prediction, fit on validation only (steps 69-70)."""

from __future__ import annotations

import numpy as np
import torch

from .metrics import softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimise validation NLL over a single scalar T > 0 (LBFGS on log T)."""
    z = torch.from_numpy(logits).double()
    t = torch.from_numpy(y).long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def conformal_threshold(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """Least-ambiguous-set (LAC) score s = 1 - p_true; finite-sample corrected quantile."""
    s = 1.0 - cal_probs[np.arange(len(cal_y)), cal_y]
    n = len(s)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q, method="higher"))


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (N,K) prediction sets {k : 1 - p_k <= qhat}; never empty (falls back to argmax)."""
    sets = (1.0 - probs) <= qhat
    empty = ~sets.any(1)
    sets[empty, probs[empty].argmax(1)] = True
    return sets


def conformal_report(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    cover = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    return {
        "coverage": float(cover.mean()),
        "mean_set_size": float(size.mean()),
        "frac_singleton": float((size == 1).mean()),
    }


def risk_coverage(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective classification: error rate on the `coverage` fraction of most confident samples."""
    conf = probs.max(1)
    order = np.argsort(-conf)
    err = (probs.argmax(1)[order] != y[order]).astype(float)
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(err) / n


def temperature_probs(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits / T)
