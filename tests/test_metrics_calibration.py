import numpy as np

from acpolyp.training import calibration as cal
from acpolyp.training import metrics as M


def test_softmax_is_stable_for_huge_logits():
    p = M.softmax(np.array([[1000.0, 1001.0, -1000.0]]))
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-12 and p.argmax() == 1


def test_ece_zero_for_perfectly_calibrated_and_large_for_overconfident():
    y = np.array([0] * 8 + [1] * 2)
    p = np.tile([0.8, 0.2], (10, 1))  # says 80%, is right 80% of the time
    assert M.expected_calibration_error(p, y) < 1e-12
    p_over = np.tile([0.99, 0.01], (10, 1))
    assert abs(M.expected_calibration_error(p_over, y) - 0.19) < 1e-9


def test_brier_and_nll_hand_computed():
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 1])
    assert abs(M.brier_multiclass(p, y) - ((0.09 + 0.09) + (0.04 + 0.04)) / 2) < 1e-12
    assert abs(M.nll(p, y) - (-np.log(0.7) - np.log(0.8)) / 2) < 1e-12


def test_full_metrics_perfect_classifier():
    y = np.repeat(np.arange(8), 5)
    m = M.full_metrics(np.eye(8)[y] * 0.99 + 0.01 / 8, y)
    assert m["acc"] == 1 and m["macro_f1"] == 1 and m["mcc"] == 1 and m["polyp_auroc"] == 1


def test_temperature_scaling_reduces_nll_of_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    logits = np.eye(4)[y] * 2.0 + rng.normal(0, 1.5, (2000, 4))
    over = logits * 5.0
    T = cal.fit_temperature(over, y)
    assert T > 1.5
    assert M.nll(cal.temperature_probs(over, T), y) < M.nll(M.softmax(over), y)


def test_conformal_coverage_guarantee_on_exchangeable_data():
    rng = np.random.default_rng(1)
    n, k = 6000, 5
    y = rng.integers(0, k, n)
    probs = M.softmax(np.eye(k)[y] * 1.5 + rng.normal(0, 1, (n, k)))
    q = cal.conformal_threshold(probs[:2000], y[:2000], 0.1)
    rep = cal.conformal_report(cal.conformal_sets(probs[2000:], q), y[2000:])
    assert rep["coverage"] >= 0.88  # >= 1 - alpha up to sampling noise


def test_risk_coverage_at_full_coverage_equals_error_rate():
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 1, 0])
    cov, risk = cal.risk_coverage(p, y)
    assert cov[-1] == 1 and abs(risk[-1] - 1 / 3) < 1e-12 and risk[0] == 0
