"""Post-hoc calibration fitted on development-set out-of-fold predictions only."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .metrics import _logistic_fit
from .utils import logit, sigmoid


class PlattCalibrator:
    """p' = sigmoid(a + b * logit(p)) (Platt 1999), fitted by maximum likelihood."""

    def fit(self, p, y):
        self.a_, self.b_ = _logistic_fit(logit(p), y)
        return self

    def transform(self, p):
        return sigmoid(self.a_ + self.b_ * logit(p))


class IsotonicCalibrator:
    """Monotone non-parametric calibration (Zadrozny & Elkan 2002)."""

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6, increasing=True)
        self.iso_.fit(np.asarray(p, float), np.asarray(y, float))
        return self

    def transform(self, p):
        return self.iso_.predict(np.asarray(p, float))


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, float)


CALIBRATORS = {"none": IdentityCalibrator, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def make_calibrator(kind: str):
    return CALIBRATORS[kind]()
