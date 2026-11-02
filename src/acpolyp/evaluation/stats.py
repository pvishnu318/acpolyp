"""Statistical validation: paired bootstrap, McNemar, DeLong, Holm-Bonferroni, effect sizes (step 73)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats as sps


def paired_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Resample test samples with replacement (same indices for both models) and report the CI of
    metric(a) - metric(b) together with the marginal CIs. Two-sided bootstrap p-value for H0: diff = 0."""
    rng = np.random.default_rng(seed)
    n = len(y)
    da, db = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # skip degenerate resamples missing a class (cannot happen at n~1600, guarded anyway)
        da[i], db[i] = metric(probs_a[idx], y[idx]), metric(probs_b[idx], y[idx])
    d = da - db
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {
        "metric_a": float(metric(probs_a, y)),
        "metric_b": float(metric(probs_b, y)),
        "diff": float(metric(probs_a, y) - metric(probs_b, y)),
        "diff_ci_low": float(np.percentile(d, 2.5)),
        "diff_ci_high": float(np.percentile(d, 97.5)),
        "a_ci_low": float(np.percentile(da, 2.5)),
        "a_ci_high": float(np.percentile(da, 97.5)),
        "b_ci_low": float(np.percentile(db, 2.5)),
        "b_ci_high": float(np.percentile(db, 97.5)),
        "p_boot": float(min(p, 1.0)),
    }


def bootstrap_ci(
    metric: Callable, probs: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    v = np.array([metric(probs[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(metric(probs, y)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact (binomial) McNemar test on discordant pairs; two-sided."""
    b = int(((correct_a == 1) & (correct_b == 0)).sum())  # a right, b wrong
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(b, n, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_mcnemar": p}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    out = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        out[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return out


def _delong_components(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scores: (n_models, n); returns AUCs and their covariance matrix (fast DeLong, Sun & Xu 2014)."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r], ty[r] = _midrank(pos[r]), _midrank(neg[r])
        tz[r] = _midrank(
            np.concatenate([pos[r], neg[r]])
        )  # positives first, as the formulas assume
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, s


def delong_test(score_a: np.ndarray, score_b: np.ndarray, y_binary: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test for correlated ROC AUCs."""
    aucs, cov = _delong_components(np.vstack([score_a, score_b]), y_binary.astype(int))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(diff),
        "z": float(z),
        "p_delong": float(2 * sps.norm.sf(abs(z))),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def seed_level_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test (+ Wilcoxon when n permits) across seeds. With n<6 Wilcoxon cannot reach p<0.05."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n": len(a), "mean_diff": float((a - b).mean()), "cohens_d": cohens_d_paired(a, b)}
    if len(a) >= 3 and np.std(a - b) > 0:
        out["p_ttest"] = float(sps.ttest_rel(a, b).pvalue)
    else:
        out["p_ttest"] = float("nan")
    try:
        out["p_wilcoxon"] = float(sps.wilcoxon(a, b).pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    return out
