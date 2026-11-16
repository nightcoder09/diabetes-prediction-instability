"""Leakage-safe logit-space stacking with non-negative weights and exact SHAP decomposition.

Ensemble log-odds:   g(x) = b + sum_k w_k * m_k(x),   w_k >= 0
where m_k is the native log-odds output of member k. The weights (b, w) are fitted by penalised
maximum likelihood on OUT-OF-FOLD member margins only (Wolpert 1992; Breiman 1996), so no member
prediction used to fit the meta-learner was produced by a model that saw that row.

Because g is linear in the member margins and each member has an exact additive SHAP
decomposition in log-odds space, the ensemble SHAP values are  phi_j(x) = sum_k w_k * phi_kj(x)
(linearity axiom of Shapley values) - exact, with no surrogate model or sampling approximation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from .utils import sigmoid


def fit_meta(M: np.ndarray, y: np.ndarray, nonnegative: bool = True, l2: float = 1e-4,
             sample_weight: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Penalised logistic regression of y on member margins M (n x K). Returns (intercept, weights)."""
    M = np.asarray(M, float)
    y = np.asarray(y, float)
    n, K = M.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    sw = sw / sw.mean()

    def f(theta):
        b, w = theta[0], theta[1:]
        z = b + M @ w
        # stable log-loss: log(1+exp(z)) - y z
        ll = np.logaddexp(0.0, z) - y * z
        loss = np.mean(sw * ll) + l2 * np.sum(w ** 2)
        r = sw * (sigmoid(z) - y) / n
        grad = np.concatenate([[r.sum()], M.T @ r + 2 * l2 * w])
        return loss, grad

    x0 = np.concatenate([[0.0], np.full(K, 1.0 / K)])
    bounds = [(None, None)] + [((0.0, None) if nonnegative else (None, None))] * K
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-14})
    if not res.success:  # pragma: no cover - logged by caller through attributes
        pass
    return float(res.x[0]), np.asarray(res.x[1:])


class LogitStackingEnsemble:
    """Stacked ensemble over members exposing ``margin`` and ``shap_logit``.

    Parameters
    ----------
    factories : dict name -> callable(seed) returning an *unfitted* model wrapper
    n_splits  : folds for out-of-fold margins
    """

    def __init__(self, factories: dict[str, Callable[[int], object]], n_splits: int = 5, seed: int = 0,
                 nonnegative: bool = True, l2: float = 1e-4, equal_weights: bool = False):
        self.factories = factories
        self.names = list(factories)
        self.n_splits = n_splits
        self.seed = seed
        self.nonnegative = nonnegative
        self.l2 = l2
        self.equal_weights = equal_weights

    # ---------------------------------------------------------------- fitting
    def fit(self, X, y, folds: np.ndarray | None = None):
        """Full procedure: OOF margins -> meta-learner -> cross-fitted OOF probs -> refit members."""
        y = np.asarray(y)
        n = len(y)
        if folds is None:
            folds = np.empty(n, dtype=int)
            skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
            for k, (_, va) in enumerate(skf.split(np.zeros(n), y)):
                folds[va] = k
        M = np.zeros((n, len(self.names)))
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            for j, name in enumerate(self.names):
                m = self.factories[name](self.seed + 1000 * (k + 1) + j).fit(X.iloc[tr], y[tr])
                M[va, j] = m.margin(X.iloc[va])
        members = {name: self.factories[name](self.seed + j).fit(X, y) for j, name in enumerate(self.names)}
        return self.fit_from_oof(members, M, y, folds)

    def fit_from_oof(self, members: dict, oof_margins: np.ndarray, y: np.ndarray, folds: np.ndarray):
        """Use precomputed OOF member margins (same fold scheme for all members) and full-data members."""
        y = np.asarray(y)
        self.members_ = members
        self.names = list(members)
        self.oof_margins_ = np.asarray(oof_margins, float)
        self.folds_ = np.asarray(folds)
        self.intercept_, self.weights_ = self._fit_meta(self.oof_margins_, y)
        # Cross-fitted OOF ensemble predictions: meta-learner for fold k never sees fold-k rows.
        oof = np.zeros(len(y))
        for k in np.unique(self.folds_):
            tr, va = self.folds_ != k, self.folds_ == k
            b, w = self._fit_meta(self.oof_margins_[tr], y[tr])
            oof[va] = b + self.oof_margins_[va] @ w
        self.oof_margin_ = oof
        self.oof_proba_ = sigmoid(oof)
        return self

    def _fit_meta(self, M, y):
        if self.equal_weights:
            # equal-weight logit average followed by a fitted intercept+slope (Platt) on the average
            avg = M.mean(axis=1, keepdims=True)
            b, w = fit_meta(avg, y, nonnegative=True, l2=0.0)
            return b, np.full(M.shape[1], w[0] / M.shape[1])
        return fit_meta(M, y, nonnegative=self.nonnegative, l2=self.l2)

    # ---------------------------------------------------------------- prediction
    def member_margins(self, X) -> np.ndarray:
        return np.column_stack([self.members_[n].margin(X) for n in self.names])

    def member_proba(self, X) -> np.ndarray:
        return sigmoid(self.member_margins(X))

    def margin(self, X) -> np.ndarray:
        return self.intercept_ + self.member_margins(X) @ self.weights_

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self.margin(X))

    # ---------------------------------------------------------------- explanation
    def shap_logit(self, X, background):
        phi = None
        base = self.intercept_
        for w, name in zip(self.weights_, self.names):
            if w == 0:
                continue
            p_k, b_k = self.members_[name].shap_logit(X, background)
            phi = w * p_k if phi is None else phi + w * p_k
            base += w * b_k
        if phi is None:  # degenerate: all weights zero
            phi = np.zeros((len(X), X.shape[1]))
        return phi, float(base)

    def weight_table(self) -> dict:
        return {"intercept": float(self.intercept_), **{n: float(w) for n, w in zip(self.names, self.weights_)}}


class LogitBag:
    """Homogeneous bag: K copies of one model family trained on bootstrap resamples, averaged in
    log-odds space. Used as the ablation that separates 'averaging' from 'heterogeneity'."""

    def __init__(self, factory: Callable[[int], object], size: int = 4, seed: int = 0):
        self.factory = factory
        self.size = size
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        self.members_ = []
        for b in range(self.size):
            idx = rng.integers(0, len(y), len(y))
            self.members_.append(self.factory(self.seed + 17 * (b + 1)).fit(X.iloc[idx], y[idx]))
        return self

    def margin(self, X):
        return np.mean([m.margin(X) for m in self.members_], axis=0)

    def predict_proba(self, X):
        return sigmoid(self.margin(X))

    def shap_logit(self, X, background):
        parts = [m.shap_logit(X, background) for m in self.members_]
        return np.mean([p for p, _ in parts], axis=0), float(np.mean([b for _, b in parts]))


class SeedAverage:
    """S copies of one model trained on the SAME rows with different random seeds, averaged in log-odds.

    Removes the part of re-training variability that comes from algorithmic randomness (row/column
    subsampling, tie-breaking) without changing the data a model sees. An average of additive
    log-odds models is additive, so SHAP values stay exact (linearity axiom)."""

    def __init__(self, factory: Callable[[int], object], n_seeds: int = 5, seed: int = 0):
        self.factory = factory
        self.n_seeds = int(n_seeds)
        self.seed = int(seed)

    def fit(self, X, y):
        y = np.asarray(y)
        self.members_ = [self.factory(self.seed + 7919 * s).fit(X, y) for s in range(self.n_seeds)]
        return self

    def margin(self, X):
        return np.mean([m.margin(X) for m in self.members_], axis=0)

    def predict_proba(self, X):
        return sigmoid(self.margin(X))

    def shap_logit(self, X, background):
        parts = [m.shap_logit(X, background) for m in self.members_]
        return np.mean([p for p, _ in parts], axis=0), float(np.mean([b for _, b in parts]))


def seed_average_factory(base_factory: Callable[[int], object], n_seeds: int) -> Callable[[int], object]:
    """Picklable factory seed -> unfitted SeedAverage (plain model when n_seeds == 1)."""
    from functools import partial

    return base_factory if int(n_seeds) <= 1 else partial(_make_seed_average, base_factory, int(n_seeds))


def _make_seed_average(base_factory, n_seeds, seed):
    return SeedAverage(base_factory, n_seeds, seed)


def oof_margins_for(factory: Callable[[int], object], X, y, folds, seed: int) -> np.ndarray:
    """Out-of-fold margins of a single model family for a given fold assignment."""
    y = np.asarray(y)
    out = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        out[va] = factory(seed + 1000 * (k + 1)).fit(X.iloc[tr], y[tr]).margin(X.iloc[va])
    return out
