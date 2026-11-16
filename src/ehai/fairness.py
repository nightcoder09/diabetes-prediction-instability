"""Subgroup performance and calibration with bootstrap uncertainty.

Subgroups use variables present in the dataset only (sex, age band, household income band,
education band). The dataset has no race/ethnicity variable, so racial equity cannot be assessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import auroc, brier, calibration_intercept, observed_expected, threshold_metrics

SUBGROUPS = {
    "Sex": {"Female": [0], "Male": [1]},
    "Age band": {"18-44": [1, 2, 3, 4, 5], "45-64": [6, 7, 8, 9], "65+": [10, 11, 12, 13]},
    "Income": {"< $25k": [1, 2, 3, 4], "$25k-$50k": [5, 6], ">= $50k": [7, 8]},
    "Education": {"High school or less": [1, 2, 3, 4], "Some college": [5], "College graduate": [6]},
}
SOURCE_COL = {"Sex": "Sex", "Age band": "Age", "Income": "Income", "Education": "Education"}


def group_labels(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for attr, groups in SUBGROUPS.items():
        col = X[SOURCE_COL[attr]].astype(int)
        lab = pd.Series(index=X.index, dtype=object)
        for g, codes in groups.items():
            lab[col.isin(codes)] = g
        out[attr] = lab
    return pd.DataFrame(out)


def _group_metrics(y, p, t):
    tm = threshold_metrics(y, p, t)
    return {"auroc": auroc(y, p) if 0 < y.sum() < len(y) else np.nan, "tpr": tm["sensitivity"],
            "fpr": 1 - tm["specificity"], "ppv": tm["ppv"], "oe_ratio": observed_expected(y, p),
            "cal_intercept": calibration_intercept(y, p), "brier": brier(y, p)}


def subgroup_table(y, p, t, groups: pd.DataFrame, B: int = 500, seed: int = 0, min_size: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for attr in groups.columns:
        for g in SUBGROUPS[attr]:
            m = (groups[attr] == g).to_numpy()
            n = int(m.sum())
            if n < min_size:
                continue
            yg, pg = y[m], p[m]
            est = _group_metrics(yg, pg, t)
            boots = {k: [] for k in est}
            pos, neg = np.flatnonzero(yg == 1), np.flatnonzero(yg == 0)
            for _ in range(B):
                ib = np.r_[pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]]
                r = _group_metrics(yg[ib], pg[ib], t)
                for k in r:
                    boots[k].append(r[k])
            row = {"attribute": attr, "group": g, "n": n, "prevalence": float(yg.mean())}
            for k, v in est.items():
                row[k] = v
                row[f"{k}_lo"] = float(np.nanpercentile(boots[k], 2.5))
                row[f"{k}_hi"] = float(np.nanpercentile(boots[k], 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


def disparity_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Max - min gap across groups of each attribute (O/E reported as max/min ratio)."""
    rows = []
    for attr, sub in tab.groupby("attribute", sort=False):
        rows.append({"attribute": attr, "auroc_gap": sub.auroc.max() - sub.auroc.min(),
                     "tpr_gap": sub.tpr.max() - sub.tpr.min(), "fpr_gap": sub.fpr.max() - sub.fpr.min(),
                     "oe_ratio_range": sub.oe_ratio.max() / sub.oe_ratio.min(),
                     "cal_intercept_gap": sub.cal_intercept.max() - sub.cal_intercept.min()})
    return pd.DataFrame(rows)
