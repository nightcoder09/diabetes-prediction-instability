import numpy as np
import pandas as pd
import pytest

from ehai.data import DOMAINS, FEATURES
from ehai.explain import local_stability, pairwise_global_stability, perturb_small
from ehai.fairness import group_labels
from ehai.predict import InputError, validate_records
from ehai.robustness import MedianImputer, mask_columns, mask_mcar, noise_bmi, noise_days, noise_genhlth
from ehai.selective import aurc, decision_loss, retained_mask, risk_coverage


def test_mcar_rate_and_protection(Xy):
    X, _ = Xy
    Xm = mask_mcar(X, 0.3, seed=0, protected=["Sex", "Age"])
    assert Xm[["Sex", "Age"]].isna().sum().sum() == 0
    rate = Xm.drop(columns=["Sex", "Age"]).isna().to_numpy().mean()
    assert abs(rate - 0.3) < 0.02
    assert X.isna().sum().sum() == 0  # original untouched


def test_imputer_fills_everything(Xy):
    X, _ = Xy
    imp = MedianImputer().fit(X)
    out = imp.transform(mask_columns(X, ["Income", "BMI"]))
    assert out.isna().sum().sum() == 0


@pytest.mark.parametrize("fn", [lambda X: noise_bmi(X, 3.0, 0), lambda X: noise_bmi(X, 0.0, 0, shift=-1),
                                lambda X: noise_days(X, 3, 0), lambda X: noise_genhlth(X, 0.5, 0),
                                lambda X: perturb_small(X, {"BMI": 1, "MentHlth": 1, "PhysHlth": 1}, 0)])
def test_perturbations_stay_in_valid_domain(Xy, fn):
    X, _ = Xy
    Xn = fn(X)
    for f, (lo, hi) in DOMAINS.items():
        assert Xn[f].min() >= lo and Xn[f].max() <= hi
        assert np.all(np.mod(Xn[f], 1) == 0)


def test_risk_coverage_oracle_signal_beats_random():
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(0, 1, n)
    y = (rng.random(n) < p).astype(int)
    t = 0.3
    loss = decision_loss(y, p, t)
    oracle = loss + rng.uniform(0, 1e-6, n)
    assert aurc(y, p, t, oracle) < aurc(y, p, t, rng.random(n))
    cov, risk = risk_coverage(y, p, t, oracle, [1.0, 0.8, 0.5])
    assert risk[0] >= risk[1] >= risk[2]
    assert retained_mask(oracle, 0.8).sum() == int(0.8 * n)


def test_threshold_consistent_cost():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.05, 0.5, 0.9, 0.01])
    loss = decision_loss(y, p, 0.2)
    # row0: FN -> 1000*(1-0.2)=800 ; row1: FP -> 1000*0.2=200
    assert np.allclose(loss, [800, 200, 0, 0])


def test_explanation_stability_metrics_extremes():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(30, 21))
    same = local_stability([phi, phi, phi])
    assert np.isclose(same["spearman_mean"], 1.0) and np.isclose(same["relative_sd"], 0.0)
    g = pairwise_global_stability([np.arange(21.0), np.arange(21.0)])
    assert np.isclose(g["kendall_tau_mean"], 1.0)


def test_group_labels_cover_all_rows(Xy):
    X, _ = Xy
    g = group_labels(X)
    assert g.notna().all().all()


def test_validate_records_rejects_bad_inputs(Xy):
    X, _ = Xy
    ok = validate_records(X.iloc[:3])
    assert list(ok.columns) == FEATURES
    with pytest.raises(InputError):
        validate_records(X.iloc[:3].drop(columns=["BMI"]))
    bad = X.iloc[:3].copy()
    bad.loc[bad.index[0], "Age"] = 14
    with pytest.raises(InputError):
        validate_records(bad)
    bad2 = X.iloc[:3].astype(object).copy()
    bad2.loc[bad2.index[1], "GenHlth"] = "good"
    with pytest.raises(InputError):
        validate_records(bad2)
