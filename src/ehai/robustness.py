"""Test-time stress tests: missing inputs (MCAR and structured) and measurement noise.

All perturbations respect the valid integer domain of each item, so no medically impossible values
are introduced. Missing values are filled with development-set medians (fitted once, never on test
data) so that every model family is evaluated under the same, deployable handling rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DOMAINS, FEATURES


class MedianImputer:
    def fit(self, X: pd.DataFrame):
        self.values_ = X[FEATURES].median().round().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self.values_)


def mask_mcar(X: pd.DataFrame, rate: float, seed: int, protected: list[str] | None = None) -> pd.DataFrame:
    """Set each eligible cell to NaN independently with probability ``rate``."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    cols = [c for c in FEATURES if c not in (protected or [])]
    M = rng.random((len(X), len(cols))) < rate
    vals = Xm[cols].to_numpy(dtype=float, copy=True)
    vals[M] = np.nan
    Xm[cols] = vals
    return Xm


def mask_columns(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    Xm = X.copy()
    Xm[cols] = np.nan
    return Xm


def noise_bmi(X: pd.DataFrame, sd: float, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    lo, hi = DOMAINS["BMI"]
    Xn["BMI"] = np.clip(np.round(Xn["BMI"].to_numpy() + shift + rng.normal(0, sd, len(X))), lo, hi)
    return Xn


def noise_days(X: pd.DataFrame, max_shift: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for f in ["MentHlth", "PhysHlth"]:
        lo, hi = DOMAINS[f]
        Xn[f] = np.clip(Xn[f].to_numpy() + rng.integers(-max_shift, max_shift + 1, len(X)), lo, hi)
    return Xn


def noise_genhlth(X: pd.DataFrame, prob: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    flip = rng.random(len(X)) < prob
    step = rng.choice([-1, 1], size=len(X))
    lo, hi = DOMAINS["GenHlth"]
    Xn["GenHlth"] = np.clip(Xn["GenHlth"].to_numpy() + flip * step, lo, hi)
    return Xn
