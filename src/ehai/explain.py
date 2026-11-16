"""Explanation analysis: global importance, stability across retraining, sensitivity to small input
changes, and agreement between model families. All SHAP values are in log-odds space."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .data import DOMAINS, FEATURES


def global_importance(phi: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature."""
    return np.abs(phi).mean(axis=0)


def topk_set(v: np.ndarray, k: int) -> set:
    return set(np.argsort(-np.asarray(v))[:k].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def pairwise_global_stability(importances: list[np.ndarray], k: int = 5) -> dict:
    """Kendall's tau-b between importance vectors and top-k Jaccard, averaged over replicate pairs."""
    taus, jac, top1 = [], [], []
    for a, b in combinations(importances, 2):
        taus.append(kendalltau(a, b).statistic)
        jac.append(jaccard(topk_set(a, k), topk_set(b, k)))
        top1.append(float(np.argmax(a) == np.argmax(b)))
    return {"kendall_tau_mean": float(np.mean(taus)), "kendall_tau_min": float(np.min(taus)),
            f"top{k}_jaccard_mean": float(np.mean(jac)), "top1_agreement": float(np.mean(top1)),
            "n_pairs": len(taus)}


def local_stability(phis: list[np.ndarray], k: int = 3) -> dict:
    """Per-instance agreement of SHAP vectors across replicates.

    phis: list of (n_instances, n_features) arrays, one per replicate (same instances).
    Returns means over instances of: pairwise Spearman correlation, top-k set agreement (Jaccard),
    top-1 agreement, and the coefficient of variation of attributions.
    """
    P = np.stack(phis)  # (R, n, F)
    R, n, F = P.shape
    rho = np.zeros(n)
    jac = np.zeros(n)
    top1 = np.zeros(n)
    pairs = list(combinations(range(R), 2))
    ranks = np.argsort(np.argsort(-np.abs(P), axis=2), axis=2)  # rank of |phi| per replicate/instance
    for i in range(n):
        r_i, j_i, t_i = [], [], []
        for a, b in pairs:
            r_i.append(spearmanr(P[a, i], P[b, i]).statistic)
            sa = set(np.flatnonzero(ranks[a, i] < k))
            sb = set(np.flatnonzero(ranks[b, i] < k))
            j_i.append(jaccard(sa, sb))
            t_i.append(float(np.argmax(np.abs(P[a, i])) == np.argmax(np.abs(P[b, i]))))
        rho[i], jac[i], top1[i] = np.nanmean(r_i), np.mean(j_i), np.mean(t_i)
    sd = P.std(axis=0)                      # (n, F) across replicates
    scale = np.abs(P).mean(axis=0).sum(axis=1, keepdims=True)  # total attribution mass per instance
    return {"spearman_mean": float(np.nanmean(rho)), f"top{k}_jaccard_mean": float(jac.mean()),
            "top1_agreement": float(top1.mean()),
            "relative_sd": float((sd.sum(axis=1) / np.maximum(scale.ravel(), 1e-12)).mean()),
            "per_instance_spearman": rho, "per_instance_topk": jac}


def perturb_small(X: pd.DataFrame, deltas: dict[str, float], seed: int) -> pd.DataFrame:
    """Apply a random +/- delta to selected integer features (clipped to their valid domain)."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for f, d in deltas.items():
        sign = rng.choice([-1.0, 1.0], size=len(X))
        lo, hi = DOMAINS[f]
        Xp[f] = np.clip(Xp[f].to_numpy() + sign * d, lo, hi)
    return Xp


def perturbation_sensitivity(phi0: np.ndarray, phi1: np.ndarray, m0: np.ndarray, m1: np.ndarray,
                             k: int = 3) -> dict:
    """Change of explanations under a clinically negligible input change."""
    n = phi0.shape[0]
    rho = np.array([spearmanr(phi0[i], phi1[i]).statistic for i in range(n)])
    jac = np.array([jaccard(topk_set(np.abs(phi0[i]), k), topk_set(np.abs(phi1[i]), k)) for i in range(n)])
    rel = np.abs(phi1 - phi0).sum(axis=1) / np.maximum(np.abs(phi0).sum(axis=1), 1e-12)
    return {"spearman_mean": float(np.nanmean(rho)), f"top{k}_jaccard_mean": float(jac.mean()),
            "relative_l1_change_mean": float(rel.mean()), "relative_l1_change_p90": float(np.quantile(rel, 0.9)),
            "logit_change_mean": float(np.abs(m1 - m0).mean())}


def cross_model_agreement(importances: dict[str, np.ndarray], k: int = 5) -> pd.DataFrame:
    names = list(importances)
    rows = []
    for a, b in combinations(names, 2):
        rows.append({"model_a": a, "model_b": b,
                     "kendall_tau": float(kendalltau(importances[a], importances[b]).statistic),
                     f"top{k}_jaccard": jaccard(topk_set(importances[a], k), topk_set(importances[b], k))})
    return pd.DataFrame(rows)


def importance_frame(importances: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(importances, index=FEATURES)
    df.index.name = "feature"
    return df
