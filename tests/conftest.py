"""Shared fixtures. All data here are SYNTHETIC test fixtures with the BRFSS schema - never results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehai.data import DOMAINS, FEATURES, TARGET  # noqa: E402


def synthetic_brfss(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Random records inside the documented value domains with a logistic outcome model."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.integers(lo, hi + 1, n) for f, (lo, hi) in DOMAINS.items()})
    df["BMI"] = np.clip(np.round(rng.normal(28, 6, n)), 12, 98).astype(int)
    df["MentHlth"] = np.where(rng.random(n) < 0.7, 0, rng.integers(1, 31, n))
    z = (-3.2 + 0.9 * df.HighBP + 0.6 * df.HighChol + 0.08 * (df.BMI - 28) + 0.35 * (df.GenHlth - 2.5)
         + 0.15 * (df.Age - 8) - 0.1 * (df.Income - 6))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df[FEATURES + [TARGET]]


@pytest.fixture(scope="session")
def synth():
    return synthetic_brfss()


@pytest.fixture(scope="session")
def Xy(synth):
    return synth[FEATURES].astype(float), synth[TARGET].to_numpy()
