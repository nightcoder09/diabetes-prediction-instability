"""Model wrappers with a uniform interface: fit / predict_proba / margin (log-odds) / shap_logit.

``shap_logit`` returns interventional SHAP values in log-odds space aggregated to the 21 original
features, plus the base value, such that  base + phi.sum(axis=1) == margin(X)  (local accuracy).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import BINARY, DOMAINS, FEATURES, ORDINAL
from .utils import logit

DISPLAY = {
    "logreg": "Logistic regression", "decision_tree": "Decision tree", "random_forest": "Random forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "stack": "Stacked ensemble", "bag": "Homogeneous bag",
}

# Clinically motivated spline knots (BMI WHO cut-points; day-count bands for health-days items).
SPLINE_KNOTS = {
    "BMI": [12, 18.5, 25, 30, 35, 40, 50, 98],
    "MentHlth": [0, 1, 3, 7, 14, 30],
    "PhysHlth": [0, 1, 3, 7, 14, 30],
}

DEFAULT_PARAMS = {
    "logreg": {"C": 0.1},
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 200, "criterion": "gini"},
    "random_forest": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 20, "max_depth": None},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5.0,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.0},
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"iterations": 600, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3.0},
}

SEARCH_SPACES = {
    "logreg": {"C": loguniform(1e-3, 10)},
    "decision_tree": {"max_depth": randint(4, 15), "min_samples_leaf": [20, 50, 100, 200, 500, 1000],
                      "criterion": ["gini", "entropy"]},
    "random_forest": {"n_estimators": [300], "max_features": ["sqrt", 0.33],
                      "min_samples_leaf": [10, 20, 50, 100], "max_depth": [None, 16]},
    "xgboost": {"n_estimators": randint(200, 1001), "learning_rate": loguniform(0.01, 0.2),
                "max_depth": randint(3, 9), "min_child_weight": loguniform(1, 50),
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "reg_lambda": loguniform(0.1, 20), "gamma": uniform(0, 2)},
    "lightgbm": {"n_estimators": randint(200, 1001), "learning_rate": loguniform(0.01, 0.2),
                 "num_leaves": randint(15, 128), "min_child_samples": randint(20, 501),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                 "reg_lambda": loguniform(0.1, 20)},
    "catboost": {"iterations": randint(300, 1001), "learning_rate": loguniform(0.02, 0.2),
                 "depth": randint(4, 9), "l2_leaf_reg": loguniform(1, 20)},
}

ADDITIVE_LOGIT = {"logreg", "xgboost", "lightgbm", "catboost"}


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X[FEATURES].astype(float)
    return pd.DataFrame(np.asarray(X, dtype=float), columns=FEATURES)


class BaseModel:
    name = "base"

    def __init__(self, params: dict | None = None, seed: int = 0, n_jobs: int = 1):
        self.params = {**DEFAULT_PARAMS.get(self.name, {}), **(params or {})}
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.est = None

    @property
    def additive_logit(self) -> bool:
        return self.name in ADDITIVE_LOGIT

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        return self.est.predict_proba(_as_frame(X))[:, 1]

    def margin(self, X) -> np.ndarray:
        return logit(self.predict_proba(X))

    def shap_logit(self, X, background):
        raise NotImplementedError(f"{self.name} does not provide exact log-odds SHAP values")


class LogRegModel(BaseModel):
    """Flexible logistic regression: one-hot ordinal categories + cubic B-splines (fixed knots)."""

    name = "logreg"

    def _pipeline(self) -> Pipeline:
        transformers = [("bin", "passthrough", BINARY),
                        ("ord", OneHotEncoder(categories=[list(range(DOMAINS[f][0], DOMAINS[f][1] + 1))
                                                          for f in ORDINAL],
                                              handle_unknown="ignore", sparse_output=False), ORDINAL)]
        for f, knots in SPLINE_KNOTS.items():
            transformers.append((f"sp_{f}", SplineTransformer(knots=np.array(knots, dtype=float).reshape(-1, 1),
                                                              degree=3, extrapolation="constant"), [f]))
        pre = ColumnTransformer(transformers, verbose_feature_names_out=True)
        lr = LogisticRegression(C=float(self.params["C"]), max_iter=5000, tol=1e-6)
        return Pipeline([("pre", pre), ("scale", StandardScaler()), ("lr", lr)])

    def fit(self, X, y):
        self.est = self._pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(_as_frame(X), np.asarray(y))
        names = self.est["pre"].get_feature_names_out()
        groups = np.empty(len(names), dtype=int)
        for i, n in enumerate(names):
            prefix, rest = n.split("__", 1)
            if prefix in ("bin",):
                orig = rest
            elif prefix == "ord":
                orig = rest.rsplit("_", 1)[0]
            else:  # sp_<feature>
                orig = prefix[3:]
            groups[i] = FEATURES.index(orig)
        self._groups = groups
        return self

    def margin(self, X):
        return self.est.decision_function(_as_frame(X))

    def shap_logit(self, X, background):
        pre, sc, lr = self.est["pre"], self.est["scale"], self.est["lr"]
        Z = sc.transform(pre.transform(_as_frame(X)))
        Zb = sc.transform(pre.transform(_as_frame(background)))
        beta = lr.coef_.ravel()
        contrib = (Z - Zb.mean(axis=0)) * beta
        phi = np.zeros((Z.shape[0], len(FEATURES)))
        np.add.at(phi.T, self._groups, contrib.T)
        base = float(lr.intercept_[0] + (Zb.mean(axis=0) * beta).sum())
        return phi, base


class _TreeShapMixin:
    def shap_logit(self, X, background):
        import shap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expl = shap.TreeExplainer(self.est, data=_as_frame(background),
                                      feature_perturbation="interventional", model_output="raw")
            sv = expl.shap_values(_as_frame(X), check_additivity=False)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:
            sv = sv[..., -1]
        base = expl.expected_value
        base = float(np.ravel(base)[-1])
        return sv, base


class XGBModel(_TreeShapMixin, BaseModel):
    name = "xgboost"

    def fit(self, X, y):
        from xgboost import XGBClassifier

        p = dict(self.params)
        self.est = XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=self.seed,
                                 n_jobs=self.n_jobs, verbosity=0)
        self.est.fit(_as_frame(X), np.asarray(y))
        return self

    def margin(self, X):
        return self.est.predict(_as_frame(X), output_margin=True).astype(float)


class LGBMModel(_TreeShapMixin, BaseModel):
    name = "lightgbm"

    def fit(self, X, y):
        from lightgbm import LGBMClassifier

        p = dict(self.params)
        self.est = LGBMClassifier(**p, subsample_freq=1, random_state=self.seed, n_jobs=self.n_jobs,
                                  deterministic=True, force_row_wise=True, verbose=-1)
        self.est.fit(_as_frame(X), np.asarray(y))
        return self

    def margin(self, X):
        return self.est.predict(_as_frame(X), raw_score=True).astype(float)


class CatBoostModel(_TreeShapMixin, BaseModel):
    name = "catboost"

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        p = dict(self.params)
        self.est = CatBoostClassifier(**p, random_seed=self.seed, thread_count=self.n_jobs, verbose=0,
                                      allow_writing_files=False)
        self.est.fit(_as_frame(X), np.asarray(y))
        return self

    def margin(self, X):
        return np.asarray(self.est.predict(_as_frame(X), prediction_type="RawFormulaVal"), dtype=float)


class RFModel(BaseModel):
    name = "random_forest"

    def fit(self, X, y):
        self.est = RandomForestClassifier(**self.params, random_state=self.seed, n_jobs=self.n_jobs)
        self.est.fit(_as_frame(X), np.asarray(y))
        return self


class DTModel(BaseModel):
    name = "decision_tree"

    def fit(self, X, y):
        self.est = DecisionTreeClassifier(**self.params, random_state=self.seed)
        self.est.fit(_as_frame(X), np.asarray(y))
        return self


REGISTRY = {"logreg": LogRegModel, "xgboost": XGBModel, "lightgbm": LGBMModel, "catboost": CatBoostModel,
            "random_forest": RFModel, "decision_tree": DTModel}


def make_model(name: str, params: dict | None = None, seed: int = 0, n_jobs: int = 1) -> BaseModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {sorted(REGISTRY)}")
    return REGISTRY[name](params=params, seed=seed, n_jobs=n_jobs)
