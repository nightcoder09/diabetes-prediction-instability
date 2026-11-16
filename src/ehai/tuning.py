"""Randomised hyper-parameter search on the development set only (test labels never touched)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from .models import DEFAULT_PARAMS, SEARCH_SPACES, make_model


def _clean(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def random_search(name: str, X: pd.DataFrame, y: np.ndarray, n_iter: int, n_splits: int, seed: int,
                  n_jobs: int, logger=None) -> tuple[dict, pd.DataFrame]:
    """Evaluate the default configuration plus ``n_iter`` random draws by stratified K-fold AUROC."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    candidates = [dict(DEFAULT_PARAMS[name])]
    candidates += [_clean(p) for p in ParameterSampler(SEARCH_SPACES[name], n_iter=n_iter, random_state=seed)]
    rows = []
    for i, params in enumerate(candidates):
        t0 = time.time()
        scores = []
        for k, (tr, va) in enumerate(splits):
            m = make_model(name, {**DEFAULT_PARAMS[name], **params}, seed=seed + k, n_jobs=n_jobs)
            m.fit(X.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])))
        row = {"candidate": i, **params, "auroc_mean": float(np.mean(scores)), "auroc_sd": float(np.std(scores)),
               "seconds": time.time() - t0}
        rows.append(row)
        if logger:
            logger.info(f"[{name}] cand {i}/{len(candidates) - 1}: AUROC {row['auroc_mean']:.4f} "
                        f"(+/-{row['auroc_sd']:.4f}) {row['seconds']:.0f}s {params}")
    res = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
    best = {k: candidates[int(res.iloc[0]["candidate"])][k] for k in candidates[int(res.iloc[0]["candidate"])]}
    return {**DEFAULT_PARAMS[name], **best}, res
