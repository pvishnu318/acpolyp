import numpy as np
from sklearn.metrics import roc_auc_score

from acpolyp.evaluation import stats


def test_delong_auc_matches_sklearn_and_is_symmetric():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a = y + rng.normal(0, 0.9, 400)
    b = y + rng.normal(0, 1.4, 400)
    r = stats.delong_test(a, b, y)
    assert abs(r["auc_a"] - roc_auc_score(y, a)) < 1e-12
    assert abs(r["auc_b"] - roc_auc_score(y, b)) < 1e-12
    r2 = stats.delong_test(b, a, y)
    assert abs(r["p_delong"] - r2["p_delong"]) < 1e-12 and abs(r["z"] + r2["z"]) < 1e-12


def test_delong_identical_scores_give_p_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert stats.delong_test(s, s, y)["p_delong"] == 1.0


def test_delong_detects_a_clearly_better_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 600)
    good, bad = y + rng.normal(0, 0.4, 600), rng.normal(0, 1, 600)
    assert stats.delong_test(good, bad, y)["p_delong"] < 1e-6


def test_holm_known_example():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max -> 0.06
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_mcnemar_exact_counts():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 0])
    b = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    r = stats.mcnemar(a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (3, 2)
    assert abs(r["p_mcnemar"] - 1.0) < 1e-12  # two-sided binomial(3 of 5) = 1


def test_paired_bootstrap_ci_contains_diff_and_is_deterministic():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    pa = np.eye(3)[y] * 0.7 + 0.1
    pb = rng.dirichlet(np.ones(3), 300)

    def acc(p, t):
        return float((p.argmax(1) == t).mean())

    r1 = stats.paired_bootstrap(acc, pa, pb, y, 500, seed=0)
    r2 = stats.paired_bootstrap(acc, pa, pb, y, 500, seed=0)
    assert r1 == r2 and r1["diff_ci_low"] < r1["diff"] < r1["diff_ci_high"]


def test_effect_sizes():
    assert stats.cliffs_delta(np.array([3, 4, 5]), np.array([1, 2, 3])) == 8 / 9
    assert stats.cohens_d_paired(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 0.0
