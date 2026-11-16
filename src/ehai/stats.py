"""Uncertainty quantification: fast DeLong test and a fast weighted (multinomial) bootstrap.

The bootstrap draws multinomial resampling *counts* (stratified by outcome) rather than index
arrays; every metric below is computed exactly on the resample through count weights, which is
mathematically identical to recomputing the metric on the resampled rows but O(n) per replicate.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .utils import logit, sigmoid


# ----------------------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """Sun & Xu (2014) fast DeLong. preds_sorted: (k, n) with the m positives first."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(r[:m]) for r in preds_sorted])
    ty = np.array([_midrank(r[m:]) for r in preds_sorted])
    tz = np.array([_midrank(r) for r in preds_sorted])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2) -> dict:
    """Two-sided DeLong test for the difference between two correlated AUROCs."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else np.inf
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "se": se,
            "ci_low": float(diff - 1.959964 * se), "ci_high": float(diff + 1.959964 * se),
            "z": float(z), "p_value": float(p)}


def delong_ci(y, p, level: float = 0.95) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    aucs, cov = _fast_delong(np.asarray(p, float)[order][None, :], int(y.sum()))
    se = float(np.sqrt(cov[0, 0]))
    zc = stats.norm.ppf(0.5 + level / 2)
    return float(aucs[0]), float(aucs[0] - zc * se), float(aucs[0] + zc * se)


# ----------------------------------------------------------------------------- weighted metrics
class _Sorted:
    """Pre-sorted view of (y, p) for O(n) weighted AUROC / AUPRC."""

    def __init__(self, y, p):
        p = np.asarray(p, float)
        self.order = np.argsort(-p, kind="mergesort")  # descending score
        ps = p[self.order]
        # group ties: start index of each distinct value
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
        self.y = np.asarray(y, int)[self.order]


def w_auroc(S: _Sorted, w: np.ndarray) -> float:
    ws = w[S.order]
    pos = np.add.reduceat(ws * S.y, S.starts)
    neg = np.add.reduceat(ws * (1 - S.y), S.starts)
    W1, W0 = pos.sum(), neg.sum()
    neg_below = W0 - np.cumsum(neg)          # strictly lower scores (descending order)
    return float((pos * (neg_below + 0.5 * neg)).sum() / (W1 * W0))


def w_auprc(S: _Sorted, w: np.ndarray) -> float:
    ws = w[S.order]
    tp = np.cumsum(np.add.reduceat(ws * S.y, S.starts))
    fp = np.cumsum(np.add.reduceat(ws * (1 - S.y), S.starts))
    W1 = tp[-1]
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / W1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def w_brier(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def w_logloss(y, p, w, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))) / w.sum())


def w_logistic(x, y, w, offset=None, fit_slope=True):
    off = np.zeros_like(x) if offset is None else offset
    W = w.sum()

    def f(t):
        a = t[0]
        b = t[1] if fit_slope else 0.0
        z = a + b * x + off
        loss = np.sum(w * (np.logaddexp(0, z) - y * z)) / W
        r = w * (sigmoid(z) - y) / W
        g = [r.sum()] + ([float(r @ x)] if fit_slope else [])
        return loss, np.array(g)

    t0 = np.array([0.0, 1.0]) if fit_slope else np.array([0.0])
    res = minimize(f, t0, jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 200})
    return res.x


def w_cal_slope(y, p, w):
    return float(w_logistic(logit(p), y, w)[1])


def w_cal_intercept(y, p, w):
    return float(w_logistic(np.zeros_like(p), y, w, offset=logit(p), fit_slope=False)[0])


def w_oe(y, p, w):
    return float(np.sum(w * y) / np.sum(w * p))


METRICS = {"auroc": "sorted", "auprc": "sorted", "brier": w_brier, "log_loss": w_logloss,
           "cal_slope": w_cal_slope, "cal_intercept": w_cal_intercept, "oe_ratio": w_oe}


class _WeightSampler:
    """Draws multinomial resampling counts one replicate at a time (memory O(n)), stratified by outcome."""

    def __init__(self, y, seed: int, stratify: bool = True):
        y = np.asarray(y)
        self.n = len(y)
        self.rng = np.random.default_rng(seed)
        self.groups = [np.flatnonzero(y == v) for v in np.unique(y)] if stratify else [np.arange(self.n)]

    def draw(self) -> np.ndarray:
        w = np.zeros(self.n)
        for g in self.groups:
            w[g] = self.rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)))
        return w


def bootstrap(y, preds: dict[str, np.ndarray], metrics: list[str], B: int = 2000, seed: int = 0,
              level: float = 0.95, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Percentile bootstrap CIs for each metric x model and paired differences (a - b).

    The same resamples are used for every model and metric (paired design).
    Returns {metric: {"per_model": {name: {estimate, ci_low, ci_high}},
                      "pairs": {"a - b": {estimate, ci_low, ci_high, p_boot}}}}
    """
    y = np.asarray(y, float)
    preds = {k: np.asarray(v, float) for k, v in preds.items()}
    sorted_cache = {n: _Sorted(y, p) for n, p in preds.items()}

    def ev(met, name, w):
        kind = METRICS[met]
        if kind == "sorted":
            return (w_auroc if met == "auroc" else w_auprc)(sorted_cache[name], w)
        return kind(y, preds[name], w)

    sampler = _WeightSampler(y, seed)
    vals = {m: {n: np.empty(B) for n in preds} for m in metrics}
    for b in range(B):
        w = sampler.draw()
        for m in metrics:
            for n in preds:
                vals[m][n][b] = ev(m, n, w)
    ones = np.ones(len(y))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for m in metrics:
        res = {"per_model": {}, "pairs": {}}
        for n in preds:
            res["per_model"][n] = {"estimate": ev(m, n, ones), "ci_low": float(np.percentile(vals[m][n], lo)),
                                   "ci_high": float(np.percentile(vals[m][n], hi))}
        for a, bb in pairs or []:
            d = vals[m][a] - vals[m][bb]
            pval = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
            res["pairs"][f"{a} - {bb}"] = {"estimate": ev(m, a, ones) - ev(m, bb, ones),
                                           "ci_low": float(np.percentile(d, lo)),
                                           "ci_high": float(np.percentile(d, hi)), "p_boot": float(min(pval, 1.0))}
        out[m] = res
    return out


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, B: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Paired bootstrap for the mean of per-individual quantities (e.g. prediction SD)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sampler = _WeightSampler(np.zeros(len(d)), seed, stratify=False)
    boots = np.array([sampler.draw() @ d / len(d) for _ in range(B)])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    wil = stats.wilcoxon(a, b, zero_method="wilcox") if np.any(d != 0) else None
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_low": float(np.percentile(boots, lo)), "ci_high": float(np.percentile(boots, hi)),
            "ratio": float(a.mean() / b.mean()) if b.mean() != 0 else float("nan"),
            "wilcoxon_p": float(wil.pvalue) if wil is not None else 1.0}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
