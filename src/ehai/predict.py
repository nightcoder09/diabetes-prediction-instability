"""Inference on new, appropriately formatted records with input validation and explanations.

IMPORTANT: outputs are model scores from a model trained on retrospective BRFSS 2015 survey data.
They are not a validated clinical risk estimate and must not be used for individual care decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DOMAINS, FEATURES
from .pipeline import exp_dir
from .utils import sigmoid

DISCLAIMER = ("Research use only. Scores come from a model trained on self-reported BRFSS 2015 survey data and "
              "are not validated estimates of an individual's clinical risk.")


class InputError(ValueError):
    pass


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise InputError(f"Missing required columns: {missing}")
    X = df[FEATURES].copy()
    problems = []
    for f in FEATURES:
        v = pd.to_numeric(X[f], errors="coerce")
        if v.isna().any():
            problems.append(f"{f}: non-numeric or empty values in rows {[int(i) for i in np.flatnonzero(v.isna())[:5]]}")
            continue
        lo, hi = DOMAINS[f]
        bad = (v < lo) | (v > hi) | (np.mod(v, 1) != 0)
        if bad.any():
            problems.append(f"{f}: values must be integers in [{lo}, {hi}] (rows {[int(i) for i in np.flatnonzero(bad)[:5]]})")
        X[f] = v.astype(float)
    if problems:
        raise InputError("; ".join(problems))
    return X


def _load_inference(cfg):
    """Only what inference needs: frozen metadata, the stacked ensemble and a SHAP background sample.

    The background sample (development rows) is cached in experiments/models/ so that scoring new
    records does not require the training data to be present."""
    import joblib

    from .utils import load_json

    mdir = exp_dir(cfg, cfg.get("inference", {}).get("models_dir", "models"))
    if not (mdir / "stack.joblib").exists():
        raise FileNotFoundError(f"No trained model in {mdir}. Run: python scripts/run_all.py")
    frozen = load_json(mdir / "frozen.json")
    stack = joblib.load(mdir / "stack.joblib")
    bg_path = mdir / "shap_background.csv"
    if bg_path.exists():
        bg = pd.read_csv(bg_path)[FEATURES].astype(float)
    else:
        from .data import load_split
        from .experiments import _background

        Xd, _, _ = load_split(cfg, "dev")
        bg = _background(cfg, Xd)
        bg.to_csv(bg_path, index=False)
    return frozen, stack, bg


def predict(cfg, records: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    X = validate_records(records)
    frozen, ens, bg = _load_inference(cfg)
    t = frozen["thresholds"]["stack"]
    p = ens.predict_proba(X)
    mp = ens.member_proba(X)
    phi, base = ens.shap_logit(X, bg)
    rows = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(phi[i]))[:top_k]
        rows.append({
            "score": float(p[i]),
            "screen_positive": bool(p[i] >= t),
            "threshold": float(t),
            "member_disagreement_sd": float(mp[i].std()),
            "top_contributions": "; ".join(f"{FEATURES[j]}={X.iloc[i, j]:g} ({phi[i, j]:+.3f} log-odds)" for j in order),
            "base_score": float(sigmoid(base)),
        })
    out = pd.DataFrame(rows, index=records.index)
    return out
