"""Discrimination, calibration and threshold metrics, plus decision-curve analysis."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import logit, sigmoid


# ----------------------------------------------------------------------------- discrimination
def auroc(y, p) -> float:
    return float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


# ----------------------------------------------------------------------------- calibration
def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def scaled_brier(y, p) -> float:
    """1 - Brier / Brier(prevalence model). 0 = no better than predicting prevalence."""
    y = np.asarray(y, float)
    ref = np.mean((y.mean() - y) ** 2)
    return float(1 - brier(y, p) / ref)


def log_loss(y, p, eps: float = 1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _logistic_fit(x, y, offset=None, fit_slope=True):
    """Unpenalised logistic fit of y ~ a + b*x (+offset). Returns (a, b)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, float)

    def f(t):
        a = t[0]
        b = t[1] if fit_slope else 0.0
        z = a + b * x + off
        loss = np.mean(np.logaddexp(0, z) - y * z)
        r = (sigmoid(z) - y) / len(y)
        g = [r.sum()] + ([float(r @ x)] if fit_slope else [])
        return loss, np.array(g)

    t0 = np.array([0.0, 1.0]) if fit_slope else np.array([0.0])
    res = minimize(f, t0, jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 500})
    return (float(res.x[0]), float(res.x[1])) if fit_slope else (float(res.x[0]), 1.0)


def calibration_slope(y, p) -> float:
    """Slope b of the logistic recalibration model y ~ a + b * logit(p). Ideal = 1."""
    return _logistic_fit(logit(p), y)[1]


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: intercept a in y ~ a + offset(logit(p)). Ideal = 0."""
    return _logistic_fit(np.zeros(len(p)), y, offset=logit(p), fit_slope=False)[0]


def observed_expected(y, p) -> float:
    """O/E ratio: observed events / expected events. Ideal = 1."""
    return float(np.sum(y) / np.sum(p))


def ece(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    """Expected calibration error with equal-mass (default) or equal-width bins.

    Caveat: ECE is a biased, binning-dependent estimator (Nixon et al. 2019; Kumar et al. 2019);
    it is reported as a secondary summary next to the calibration slope/intercept and curves.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tot = 0.0
    for b in range(len(edges) - 1):
        m = ids == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def reliability_curve(y, p, n_bins: int = 15):
    """Equal-mass bins: (mean predicted, observed fraction, count) per bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    return (np.array([p[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks]),
            np.array([len(c) for c in chunks]))


# ----------------------------------------------------------------------------- thresholds
def threshold_for_sensitivity(y, p, target: float = 0.80) -> float:
    """Largest threshold t with sensitivity P(p >= t | y = 1) >= target (computed on dev OOF)."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(k - 1, 0)])


def threshold_metrics(y, p, t: float) -> dict:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= t).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return {"threshold": float(t), "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "f1": f1, "balanced_accuracy": (sens + spec) / 2, "accuracy": (tp + tn) / len(y),
            "flagged_fraction": float(yhat.mean()), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def all_metrics(y, p, t: float | None = None, ece_bins: int = 15) -> dict:
    out = {"auroc": auroc(y, p), "auprc": auprc(y, p), "brier": brier(y, p), "scaled_brier": scaled_brier(y, p),
           "log_loss": log_loss(y, p), "cal_intercept": calibration_intercept(y, p),
           "cal_slope": calibration_slope(y, p), "oe_ratio": observed_expected(y, p),
           "ece": ece(y, p, ece_bins)}
    if t is not None:
        out.update(threshold_metrics(y, p, t))
    return out


# ----------------------------------------------------------------------------- decision curves
def net_benefit(y, p, thresholds) -> np.ndarray:
    """Vickers & Elkin (2006): NB(t) = TP/n - FP/n * t/(1-t)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        yhat = p >= t
        tp = np.sum(yhat & (y == 1))
        fp = np.sum(yhat & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def net_benefit_treat_all(y, thresholds) -> np.ndarray:
    prev = float(np.mean(y))
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])
