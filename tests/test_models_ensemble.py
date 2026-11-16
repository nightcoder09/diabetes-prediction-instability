import numpy as np
import pytest
from functools import partial

from ehai.ensemble import LogitBag, LogitStackingEnsemble, fit_meta, oof_margins_for
from ehai.models import ADDITIVE_LOGIT, make_model

SMALL = {"xgboost": {"n_estimators": 60}, "lightgbm": {"n_estimators": 60}, "catboost": {"iterations": 80},
         "random_forest": {"n_estimators": 50}}


def _fac(name, seed):
    return make_model(name, SMALL.get(name), seed=seed, n_jobs=1)


@pytest.mark.parametrize("name", sorted(ADDITIVE_LOGIT))
def test_shap_local_accuracy(name, Xy):
    X, y = Xy
    m = _fac(name, 0).fit(X.iloc[:3000], y[:3000])
    Xe, bg = X.iloc[3000:3100], X.iloc[:80]
    phi, base = m.shap_logit(Xe, bg)
    assert phi.shape == (100, 21)
    assert np.allclose(base + phi.sum(1), m.margin(Xe), atol=1e-4)
    assert np.isclose(base, m.margin(bg).mean(), atol=1e-4)


def test_margin_is_logit_of_proba(Xy):
    X, y = Xy
    for name in ("logreg", "xgboost", "lightgbm", "catboost"):
        m = _fac(name, 1).fit(X.iloc[:2000], y[:2000])
        p = m.predict_proba(X.iloc[2000:2200])
        assert np.allclose(1 / (1 + np.exp(-m.margin(X.iloc[2000:2200]))), p, atol=1e-5)


def test_fit_meta_nonnegative_and_recovers_signal():
    rng = np.random.default_rng(0)
    n = 5000
    z, v = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.5 * z - 0.8 * v)))).astype(int)
    M = np.column_stack([z, v, rng.normal(size=n)])
    b, w = fit_meta(M, y, nonnegative=True, l2=0)
    assert (w >= 0).all()
    assert w[0] > 1.0 and w[1] < 1e-6          # negatively associated margin is clipped to zero
    b2, w2 = fit_meta(M, y, nonnegative=False, l2=0)
    assert w2[1] < -0.5                         # ...whereas the unconstrained fit uses it


class _Recorder:
    """Fake model that records which rows it was trained on and memorises their labels."""

    trained = []

    def __init__(self, seed):
        pass

    def fit(self, X, y):
        self.idx = set(X.index.tolist())
        _Recorder.trained.append(self.idx)
        self.lookup = dict(zip(X.index.tolist(), y))
        return self

    def margin(self, X):
        # +5 if the row was seen in training with label 1, -5 if seen with 0, 0 otherwise
        return np.array([0.0 if i not in self.lookup else (5.0 if self.lookup[i] else -5.0) for i in X.index])


def test_oof_margins_never_use_rows_seen_in_training(Xy):
    X, y = Xy
    folds = np.arange(len(y)) % 5
    oof = oof_margins_for(_Recorder, X, y, folds, seed=0)
    # a memorising model can only predict 0 for unseen rows -> any non-zero OOF value is leakage
    assert np.all(oof == 0)


def test_stack_cross_fitted_oof_and_exact_shap(Xy):
    X, y = Xy
    facs = {n: partial(_fac, n) for n in ("logreg", "lightgbm")}
    ens = LogitStackingEnsemble(facs, n_splits=3, seed=0).fit(X.iloc[:3000], y[:3000])
    assert ens.oof_proba_.shape == (3000,)
    assert (ens.weights_ >= 0).all()
    Xe, bg = X.iloc[3000:3060], X.iloc[:60]
    phi, base = ens.shap_logit(Xe, bg)
    assert np.allclose(base + phi.sum(1), ens.margin(Xe), atol=1e-4)
    p = ens.predict_proba(Xe)
    assert np.all((p > 0) & (p < 1))


def test_logit_bag_shap_is_member_average(Xy):
    X, y = Xy
    bag = LogitBag(partial(_fac, "lightgbm"), size=2, seed=0).fit(X.iloc[:2000], y[:2000])
    Xe, bg = X.iloc[2000:2040], X.iloc[:40]
    phi, base = bag.shap_logit(Xe, bg)
    assert np.allclose(base + phi.sum(1), bag.margin(Xe), atol=1e-4)


def test_seed_average_is_exact_and_reduces_to_plain_model(Xy):
    from ehai.ensemble import SeedAverage, seed_average_factory

    X, y = Xy
    fac = partial(_fac, "xgboost")
    sa = SeedAverage(fac, n_seeds=3, seed=11).fit(X.iloc[:2000], y[:2000])
    Xe, bg = X.iloc[2000:2050], X.iloc[:60]
    singles = [fac(11 + 7919 * s).fit(X.iloc[:2000], y[:2000]) for s in range(3)]
    assert np.allclose(sa.margin(Xe), np.mean([m.margin(Xe) for m in singles], axis=0), atol=1e-6)
    phi, base = sa.shap_logit(Xe, bg)
    assert np.allclose(base + phi.sum(1), sa.margin(Xe), atol=1e-4)      # exact SHAP survives averaging
    assert seed_average_factory(fac, 1) is fac                             # one seed = the plain model
    assert isinstance(seed_average_factory(fac, 3)(5), SeedAverage)
