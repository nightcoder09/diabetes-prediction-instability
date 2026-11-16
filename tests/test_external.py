"""Tests for the raw-BRFSS derivation and the temporal-validation helpers (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from ehai.brfss_raw import recode, resolve_sources
from ehai.data import FEATURES, TARGET
from ehai.external import membership_auc, recalibrate, smd
from ehai.utils import logit, sigmoid


def _raw_2015_like(n=6):
    """A few hand-coded respondents in raw BRFSS 2015 codes."""
    return pd.DataFrame({
        "DIABETE3": [1, 2, 3, 4, 7, 1],          # diabetes, pregnancy-only, no, prediabetes, don't know, diabetes
        "_RFHYPE5": [2, 1, 1, 2, 1, 9],          # risk flags: 2 = yes, 1 = no, 9 = missing
        "TOLDHI2": [1, 2, 2, 1, 2, 1],
        "_CHOLCHK": [1, 2, 3, 1, 1, 1],
        "_BMI5": [2849, 2250, 3151, 4000, 2600, 2700],
        "SMOKE100": [1, 2, 2, 1, 2, 1],
        "CVDSTRK3": [2, 2, 2, 1, 2, 2],
        "_MICHD": [2, 2, 2, 1, 2, 2],
        "_TOTINDA": [1, 1, 2, 1, 1, 1],
        "_FRTLT1": [1, 2, 1, 1, 1, 1],
        "_VEGLT1": [1, 1, 1, 2, 1, 1],
        "_RFDRHV5": [1, 1, 2, 1, 1, 1],
        "HLTHPLN1": [1, 1, 2, 1, 1, 1],
        "MEDCOST": [2, 2, 1, 2, 2, 2],
        "GENHLTH": [3, 2, 5, 4, 1, 2],
        "MENTHLTH": [88, 5, 30, 88, 2, 88],
        "PHYSHLTH": [88, 88, 15, 3, 88, 88],
        "DIFFWALK": [2, 2, 1, 1, 2, 2],
        "SEX": [1, 2, 2, 1, 1, 2],
        "_AGEG5YR": [9, 3, 13, 11, 5, 14],      # 14 = don't know -> dropped
        "EDUCA": [6, 5, 4, 3, 6, 9],
        "INCOME2": [8, 7, 1, 5, 77, 6],
    }).iloc[:n]


def test_recode_matches_published_rules():
    raw = _raw_2015_like()
    out = recode(raw, resolve_sources(raw.columns))
    assert list(out.columns) == [TARGET, "Diabetes_012"] + FEATURES
    # outcome: only diagnosed diabetes is positive; prediabetes and pregnancy-only diabetes are 0
    assert out["Diabetes_012"].tolist()[:4] == [2, 0, 0, 1]
    assert out[TARGET].tolist()[:4] == [1, 0, 0, 0]
    assert np.isnan(out.loc[4, TARGET])                               # don't know -> missing
    assert out["HighBP"].tolist()[:5] == [1, 0, 0, 1, 0] and np.isnan(out.loc[5, "HighBP"])
    assert out["CholCheck"].tolist() == [1, 0, 0, 1, 1, 1]
    assert out["BMI"].tolist() == [28, 22, 32, 40, 26, 27]           # _BMI5 has two implied decimals
    assert out["MentHlth"].tolist() == [0, 5, 30, 0, 2, 0]            # 88 = none -> 0
    assert out["Sex"].tolist() == [1, 0, 0, 1, 1, 0]
    assert np.isnan(out.loc[5, "Age"]) and np.isnan(out.loc[5, "Education"]) and np.isnan(out.loc[4, "Income"])


def test_income3_is_collapsed_to_2015_top_band():
    raw = _raw_2015_like().rename(columns={"INCOME2": "INCOME3"})
    raw["INCOME3"] = [11, 9, 8, 7, 1, 77]
    out = recode(raw, resolve_sources(raw.columns))
    assert out["Income"].tolist()[:5] == [8, 8, 8, 7, 1]
    assert np.isnan(out.loc[5, "Income"])


def test_resolve_sources_handles_renames_and_optional_items():
    raw = _raw_2015_like().rename(columns={"DIABETE3": "DIABETE4", "_RFHYPE5": "_RFHYPE6"})
    src = resolve_sources(raw.drop(columns=["_FRTLT1", "_VEGLT1"]).columns)
    assert src["diabetes"] == "DIABETE4" and src["HighBP"] == "_RFHYPE6"
    assert src["Fruits"] is None and src["Veggies"] is None          # optional items may be absent
    with pytest.raises(KeyError):
        resolve_sources(raw.drop(columns=["GENHLTH"]).columns)        # required items may not


def test_intercept_update_recovers_a_shift():
    rng = np.random.default_rng(0)
    p_true = sigmoid(rng.normal(-1.8, 1.0, 40_000))
    y = (rng.random(len(p_true)) < p_true).astype(int)
    p_model = sigmoid(logit(p_true) - 0.3)                           # model under-predicts by 0.3 log-odds
    fit, ev = np.arange(0, 40_000, 2), np.arange(1, 40_000, 2)
    rc = recalibrate(p_model[fit], y[fit], p_model[ev])
    assert rc["params"]["delta_intercept"] == pytest.approx(0.3, abs=0.06)
    assert y[ev].mean() / rc["intercept_update"].mean() == pytest.approx(1.0, abs=0.03)
    assert y[ev].mean() / p_model[ev].mean() > 1.15                  # the frozen model was miscalibrated


def test_membership_auc_and_smd():
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"x": rng.normal(size=4000), "z": rng.integers(0, 2, 4000)})
    b = pd.DataFrame({"x": rng.normal(size=4000), "z": rng.integers(0, 2, 4000)})
    c = pd.DataFrame({"x": rng.normal(1.0, 1.0, 4000), "z": rng.integers(0, 2, 4000)})
    assert membership_auc(a, b, seed=0, n=4000) < 0.56                # same distribution -> ~0.5
    assert membership_auc(a, c, seed=0, n=4000) > 0.70                # shifted -> separable
    assert smd(a["x"], c["x"]) == pytest.approx(1.0, abs=0.1)
