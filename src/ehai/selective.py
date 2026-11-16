"""Selective prediction (deferral to confirmatory testing) and risk-coverage analysis.

An uncertainty score u(x) ranks individuals; the most uncertain (1 - coverage) fraction is deferred
(e.g. to HbA1c testing) and the loss of the automatic decision is measured on the rest. The scores
are heuristics for ranking - they are NOT validated probabilities of being correct.

Loss. A decision threshold t on a calibrated probability implies a misclassification-cost ratio
FP : FN = t : (1 - t) (the same equivalence underlies decision-curve analysis). We therefore score
each automatic decision with the threshold-consistent cost  (1 - t) * FN + t * FP,  reported per
1,000 decided individuals. Plain 0/1 error is reported as a secondary measure only, because at a
screening threshold far below 0.5 it rewards deferring high-risk people simply for being positive.
"""
from __future__ import annotations

import numpy as np

from .utils import logit

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 / < 2.0


def margin_uncertainty(p, t):
    """Closeness to the decision threshold on the log-odds scale (higher = more uncertain)."""
    return -np.abs(logit(p) - logit(t))


def disagreement_uncertainty(member_p: np.ndarray):
    """Standard deviation of member probabilities (higher = members disagree more)."""
    return np.asarray(member_p).std(axis=1)


def combined_uncertainty(u1, u2, ref1, ref2):
    """Sum of z-scores, standardised with development-set (OOF) reference distributions."""
    z1 = (u1 - np.mean(ref1)) / np.std(ref1)
    z2 = (u2 - np.mean(ref2)) / np.std(ref2)
    return z1 + z2


def decision_loss(y, p, t, kind: str = "cost") -> np.ndarray:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= t).astype(int)
    if kind == "error":
        return (yhat != y).astype(float)
    fn = (y == 1) & (yhat == 0)
    fp = (y == 0) & (yhat == 1)
    return 1000.0 * ((1 - t) * fn + t * fp)


def risk_coverage(y, p, t, u, coverages=None, kind: str = "cost"):
    """Mean loss of thresholded decisions on retained cases as coverage decreases.

    Returns (coverage grid, risk per coverage). Ties in u are broken by a fixed permutation.
    """
    loss = decision_loss(y, p, t, kind)
    order = np.lexsort((np.arange(len(u)), np.asarray(u)))  # most certain first
    cum = np.cumsum(loss[order])
    n = len(loss)
    if coverages is None:
        k = np.arange(1, n + 1)
        return k / n, cum / k
    out = []
    for c in coverages:
        k = max(int(np.floor(c * n)), 1)
        out.append(cum[k - 1] / k)
    return np.asarray(coverages), np.asarray(out)


def aurc(y, p, t, u, min_coverage: float = 0.5, kind: str = "cost") -> float:
    """Area under the risk-coverage curve over coverage in [min_coverage, 1] (normalised)."""
    cov, risk = risk_coverage(y, p, t, u, kind=kind)
    m = cov >= min_coverage
    return float(_trapz(risk[m], cov[m]) / (1 - min_coverage))


def retained_mask(u, coverage: float) -> np.ndarray:
    n = len(u)
    k = int(np.floor(coverage * n))
    order = np.lexsort((np.arange(n), np.asarray(u)))
    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m
