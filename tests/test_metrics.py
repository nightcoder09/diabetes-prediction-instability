import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ehai import metrics as M
from ehai.stats import _Sorted, bootstrap, bootstrap_mean_diff, delong_ci, delong_test, holm, w_auprc, w_auroc


def _data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 6, n)
    y = (rng.random(n) < p).astype(int)  # perfectly calibrated by construction
    return y, p


def test_weighted_auc_matches_sklearn_with_ties_and_resampling():
    y, p = _data()
    p = np.round(p, 2)  # many ties
    S = _Sorted(y, p)
    assert np.isclose(w_auroc(S, np.ones(len(y))), roc_auc_score(y, p))
    assert np.isclose(w_auprc(S, np.ones(len(y))), average_precision_score(y, p))
    idx = np.random.default_rng(1).integers(0, len(y), len(y))
    w = np.bincount(idx, minlength=len(y)).astype(float)
    assert np.isclose(w_auroc(S, w), roc_auc_score(y[idx], p[idx]))
    assert np.isclose(w_auprc(S, w), average_precision_score(y[idx], p[idx]))


def test_delong_ci_agrees_with_bootstrap():
    y, p = _data(8000)
    auc, lo, hi = delong_ci(y, p)
    b = bootstrap(y, {"m": p}, ["auroc"], B=300, seed=2)["auroc"]["per_model"]["m"]
    assert abs(b["ci_low"] - lo) < 0.01 and abs(b["ci_high"] - hi) < 0.01


def test_delong_identical_models_zero_difference():
    y, p = _data()
    r = delong_test(y, p, p)
    assert abs(r["diff"]) < 1e-12


def test_calibration_measures_on_calibrated_data():
    y, p = _data(40000)
    assert abs(M.calibration_slope(y, p) - 1) < 0.08
    assert abs(M.calibration_intercept(y, p)) < 0.05
    assert M.ece(y, p) < 0.02
    assert abs(M.observed_expected(y, p) - 1) < 0.05


def test_threshold_for_sensitivity_meets_target():
    y, p = _data()
    t = M.threshold_for_sensitivity(y, p, 0.8)
    sens = M.threshold_metrics(y, p, t)["sensitivity"]
    assert sens >= 0.8 and sens < 0.81


def test_net_benefit_formula():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    t = 0.5
    # TP=1, FP=1 at t=0.5 -> NB = 1/4 - 1/4 * 1 = 0
    assert np.isclose(M.net_benefit(y, p, [t])[0], 0.0)


def test_bootstrap_mean_diff_and_holm():
    a = np.linspace(0, 1, 200)
    r = bootstrap_mean_diff(a + 0.1, a, B=200, seed=0)
    assert np.isclose(r["diff"], 0.1) and r["ci_low"] <= 0.1 <= r["ci_high"]
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and adj["b"] >= adj["c"] >= adj["a"]
